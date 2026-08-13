# Neural terrain: the heightfield is a small MLP f(x,z) -> height, trained through
# the same differentiable heat method. So instead of a height per vertex, the
# optimization variable is the MLP weights. The mesh resolution is separate from
# the MLP, so we can train on a cheap coarse mesh and evaluate on a fine one.
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import time
import csv
import sys

import iskra.sparse_linalg as sparse_linalg
from heightfield_optimization import (
    make_adjacent_face_pairs as adjacent_face_pairs,
    make_heightfield,
    normal_smoothness_loss,
)
from iskra.geometry.geodesics import heat_method_distance
from fourier_analysis import analyze_height_spectrum

sparse_linalg._cholmod_available = False


def make_grid(n):
    verts, faces = make_heightfield(
        n,
        n,
        dtype=torch.double,
        device="cpu",
    )
    return verts[:, [0, 2]], faces


def diagonal(n, k):
    return torch.tensor(
        [i * n + j for i in range(n) for j in range(n) if i + j == k],
        dtype=torch.long,
    )


class MLP(nn.Module):
    # maps (x,z) in [0,1]^2 to a height. This is the whole "neural" part.
    def __init__(self, hidden=64, scale=0.2, activation = "tanh"):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(2, hidden),
        #     nn.Tanh(),
        #     nn.Linear(hidden, hidden),
        #     nn.Tanh(),
        #     nn.Linear(hidden, 1),
        # ).double()
        # self.scale = scale
        activation_classes = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
            "silu": nn.SiLU,
        }
        if activation not in activation_classes:
            raise ValueError(f"Unknown activation: {activation}")

        activation_class = activation_classes[activation]
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            activation_class(),
            nn.Linear(hidden, hidden),
            activation_class(),
            nn.Linear(hidden, 1),
        ).double()
        self.scale = scale
        self.activation_name = activation

    def forward(self, xz):
        raw = self.net(2.0 * xz - 1.0).squeeze(-1)
        return torch.tanh(raw) * self.scale


def heights_to_verts(xz, h):
    return torch.stack([xz[:, 0], h, xz[:, 1]], dim=1)


def train(mlp, n, k, iters, lr=1e-2, weight=1e-04, target_tolerance=1e-3, desired_distance=0.5):
    # train the MLP on an n x n mesh so the i+j=k diagonal becomes one distance contour
    xz, F = make_grid(n)
    pairs = adjacent_face_pairs(F)
    src = torch.tensor([0])  # two opposite corners
    diag = diagonal(n, k)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    converged = False   
    start_time = time.perf_counter()

    for iteration in range(iters + 1):
        opt.zero_grad()
        V = heights_to_verts(xz, mlp(xz))
        phi = heat_method_distance(V, F, src, t_factor=10.0)
        d = phi[diag]
        # l2_error = torch.linalg.vector_norm(error)

        # if iteration % 25 == 0:
        #     print(f"  {n}x{n}: {iteration} iters, l2_error={l2_error:.2e}", flush=True)

        # if l2_error <= l2_tolerance:
        #     converged = True
        #     break

        target_error = d - desired_distance
        target_mse = target_error.square().mean()
        target_rmse = target_mse.sqrt()

        if iteration % 25 == 0:
            print(
                f"  {n}x{n}: "
                f"iteration={iteration}, "
                f"target_mse={target_mse.item():.6e}, "
                f"target_rmse={target_rmse.item():.6e}",
                flush=True,
            )

        if target_mse.item() <= target_tolerance:
            converged = True
            break

        if iteration == iters: 
            break
        
        loss_smoothness = normal_smoothness_loss(V, F, pairs)
        loss = target_mse + weight * loss_smoothness
        loss.backward()

        for name, param in mlp.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Missing gradient for {name} " f"at iteration {iteration}")

            if not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Non-finite gradient for {name} " f"at iteration {iteration}")
        opt.step()

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        V = heights_to_verts(xz, mlp(xz))
        phi = heat_method_distance(V, F, src, t_factor=10.0)
        d = phi[diag]

        target_error = d - desired_distance
        target_mse = target_error.square().mean().item()
        target_rmse = target_error.square().mean().sqrt().item()
        target_l2 = torch.linalg.vector_norm(target_error).item()
        target_max_error = target_error.abs().max().item()

        contour_std = d.std(correction=0).item()
        mean_distance = d.mean().item()

        final_smoothness = normal_smoothness_loss(V, F, pairs).item()
        height_range = (V[:, 1].max() - V[:, 1].min()).item()

    print(
        f"  {n}x{n}: "
        f"converged={converged}, "
        f"iterations={iteration}, "
        f"time={elapsed:.4f}s, "
        f"target_mse={target_mse:.6e}, "
        f"target_rmse={target_rmse:.6e}, "
        f"target_l2={target_l2:.6e}, "
        f"target_max_error={target_max_error:.6e}, "
        f"contour_std={contour_std:.6e}, "
        f"mean_distance={mean_distance:.6e}",
        flush=True,
    )

    return {
        "scale": mlp.scale,
        "converged": converged,
        "iterations": iteration,
        "target_mse": target_mse,
        "target_rmse": target_rmse,
        "normal_smoothness": final_smoothness,
        "height_range": height_range,
        "runtime": elapsed,
    }

def sample(mlp, n, k):
    # evaluate the trained MLP at any resolution (no training) -> continuous surface
    xz, F = make_grid(n)
    src = torch.tensor([0])
    with torch.no_grad():
        V = heights_to_verts(xz, mlp(xz))
        print(
            f"height min={V[:, 1].min().item():.6f}, "
            f"max={V[:, 1].max().item():.6f}, "
            f"range={(V[:, 1].max() - V[:, 1].min()).item():.6f}"
        )
        phi = heat_method_distance(V, F, src, 10.0)
    return xz, F, V, phi, src, diagonal(n, k)


def render(V, F, phi, src, diag, out):
    import polyscope as ps

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not ps.is_initialized():
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("shadow_only")
    Vn, Fn = V.numpy(), F.numpy()
    ps.remove_all_structures()
    m = ps.register_surface_mesh("terrain", Vn, Fn)
    q = m.add_scalar_quantity(
        "distance",
        phi.numpy(),
        defined_on="vertices",
        cmap="turbo",
        isolines_enabled=True,
        enabled=True,
    )
    try:
        q.set_isoline_width(float(phi.max() - phi.min()) / 18.0, relative=False)
    except Exception:
        pass
    ps.register_point_cloud("src", Vn[src.numpy()], radius=0.012).set_color(
        (1, 0.6, 0)
    )
    pts = Vn[diag.numpy()]
    e = np.stack(
        [np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1
    )
    ps.register_curve_network("diag", pts, e, radius=0.004).set_color((1, 0, 0))
    ps.look_at((1.9, 1.5, 1.9), (0.5, 0.0, 0.5))
    # ps.screenshot(out)
    # print(f"  wrote {out}", flush=True)
    ps.show()


def run_scale_analysis():
    scales = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
    ]
    results = []

    for scale in scales:
        print(f"\nRunning scale={scale:.2f}", flush=True)
        # Every experiment starts from the same random initialization.
        torch.manual_seed(0)
        mlp = MLP(scale=scale)
        result = train(mlp, n=32, k=8, iters=400, lr=1e-2, weight=1e-4, target_tolerance=1e-3,
                       desired_distance=0.5)
        results.append(result)
        print(
            f"scale={result['scale']:.2f}, "
            f"converged={result['converged']}, "
            f"iterations={result['iterations']}, "
            f"mse={result['target_mse']:.6e}, "
            f"rmse={result['target_rmse']:.6e}, "
            f"smoothness={result['normal_smoothness']:.6e}, "
            f"height_range={result['height_range']:.6f}, "
            f"time={result['runtime']:.2f}s",
            flush=True,
        )

    # Save all results to a CSV file.
    output_dir = Path("results/geodesics/scale_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scale_analysis.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scale",
                "converged",
                "iterations",
                "target_mse",
                "target_rmse",
                "normal_smoothness",
                "height_range",
                "runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nScale analysis complete:")
    print(f"Results saved to: {csv_path.resolve()}")

    return results

def run_target_analysis():
    # target_distances = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    k_values = [4, 6, 8, 10, 12, 14, 15]

    scale = 0.2
    results = []

    for k in k_values:
        print(f"\nRunning k={k}", flush=True)
        torch.manual_seed(0)
        mlp = MLP(scale=scale)
        result = train(
            mlp,
            n=32,
            k=k,
            iters=400,
            lr=1e-2,
            weight=1e-3,
            target_tolerance=1e-3,
            desired_distance=0.45,
        )
        result["k"] = k
        results.append(result)

    output_dir = Path("results/geodesics/target_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "target_analysis.csv"

    fieldnames = [
        "k",
        "scale",
        "converged",
        "iterations",
        "target_mse",
        "target_rmse",
        "normal_smoothness",
        "height_range",
        "runtime",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTarget analysis summary:")
    for result in results:
        print(
            f"k={result['k']}, "
            f"converged={result['converged']}, "
            f"iterations={result['iterations']}, "
            f"mse={result['target_mse']:.6e}, "
            f"rmse={result['target_rmse']:.6e}, "
            f"smoothness={result['normal_smoothness']:.6e}, "
            f"{result['normal_smoothness']:.6e}, "
            f"height_range="
            f"{result['height_range']:.6f}, "
            f"time={result['runtime']:.2f}s"
        )
    print(f"\nResults saved to: {csv_path.resolve()}")

    return results

def run_activation_analysis():
    activations = [
        "tanh",
        "relu",
        "softplus",
        "silu",
    ]
    seeds = [0, 1, 2, 3, 4]
    results = []

    for activation in activations:
        for seed in seeds:
            print(f"\nRunning activation={activation}, " f"seed={seed}", flush=True)
            torch.manual_seed(seed)
            mlp = MLP(scale=0.2, activation=activation)
            result = train(
                mlp,
                n=32,
                k=8,
                iters=400,
                lr=1e-2,
                weight=1e-4,
                target_tolerance=1e-3,
                desired_distance=0.5,
            )

            result["activation"] = activation
            result["seed"] = seed
            results.append(result)

    output_dir = Path("results/geodesics/activation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save every individual run.
    csv_path = output_dir / "activation_analysis.csv"
    fieldnames = [
        "activation",
        "seed",
        "scale",
        "converged",
        "iterations",
        "target_mse",
        "target_rmse",
        "normal_smoothness",
        "height_range",
        "runtime",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    # Calculate median results for each activation.
    summary_results = []
    for activation in activations:
        activation_results = [
            result
            for result in results
            if result["activation"] == activation
        ]

        def median(key):
            values = [
                result[key]
                for result in activation_results
            ]
            return float(np.median(values))

        convergence_rate = sum(
            result["converged"]
            for result in activation_results
        ) / len(activation_results)

        summary_results.append({
            "activation": activation,
            "convergence_rate": convergence_rate,
            "median_iterations": median("iterations"),
            "median_rmse": median("target_rmse"),
            "median_smoothness": median(
                "normal_smoothness"
            ),
            "median_runtime": median("runtime"),
        })

    summary_csv_path = (output_dir / "activation_summary.csv")

    with summary_csv_path.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_results)
    print("\nMulti-seed activation summary:")

    for result in summary_results:
        print(
            f"{result['activation']}: "
            f"converged="
            f"{result['convergence_rate']:.0%}, "
            f"iterations="
            f"{result['median_iterations']:.1f}, "
            f"rmse="
            f"{result['median_rmse']:.6e}, "
            f"smoothness="
            f"{result['median_smoothness']:.6e}, "
            f"time="
            f"{result['median_runtime']:.2f}s"
        )

    print(f"\nRaw results: {csv_path.resolve()}")
    print(f"Summary: {summary_csv_path.resolve()}")

    return results, summary_results

def main():
    run_scale_analysis()
    '''
    if "--preliminary" not in sys.argv:
        print(
            "Usage: python neural_terrain.py --preliminary"
        )
        return

    # Fixed initialization for reproducibility.
    torch.manual_seed(0)

    mlp = MLP(
        hidden=64,
        scale=0.2,
        activation="tanh",
    )

    print("Training the plain MLP:")
    result = train(
        mlp,
        n=32,
        k=8,
        iters=800,
        lr=5e-3,
        weight=0.0,
        target_tolerance=1e-6,
        desired_distance=0.5,
    )

    print("\nTraining result:")
    print(
        f"  RMSE={result['target_rmse']:.6e}"
    )
    print(
        f"  height_range={result['height_range']:.6e}"
    )

    # Sample the same trained MLP at a higher resolution.
    print("\nSampling the trained MLP at 128 x 128:")
    xz, faces, vertices, distance, source, target = sample(
        mlp,
        n=128,
        k=33,
    )

    # Fourier analysis of the same trained surface.
    print("\nRunning Fourier analysis:")
    analyze_height_spectrum(
        mlp,
        n=128,
        cutoff=0.25,
        out=(
            "results/geodesics/"
            "fourier_analysis.png"
        ),
        show=True,
    )

    # Interactive Polyscope visualization.
    print("\nOpening the preliminary MLP visualization:")
    render(
        vertices,
        faces,
        distance,
        source,
        target,
        (
            "results/geodesics/"
            "neural_terrain.png"
        ),
    )
    '''
    
if __name__ == "__main__":
    
    main()

    
'''
cd /Users/huyufan/Documents/GitHub/iskra

/Users/huyufan/iskra-heightfield-publish/.venv/bin/python \
/Users/huyufan/Documents/GitHub/iskra/neural_terrain.py \
--preliminary
'''