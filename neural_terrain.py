# Neural terrain: the heightfield is a small MLP f(x,z) -> height, trained through
# the same differentiable heat method. So instead of a height per vertex, the
# optimization variable is the MLP weights. The mesh resolution is separate from
# the MLP, so we can train on a cheap coarse mesh and evaluate on a fine one.
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import time

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
    def __init__(self, hidden=64, scale=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        ).double()
        self.scale = scale

    def forward(self, xz):
        raw = self.net(2.0 * xz - 1.0).squeeze(-1)
        return torch.tanh(raw) * self.scale


def heights_to_verts(xz, h):
    return torch.stack([xz[:, 0], h, xz[:, 1]], dim=1)


def train(mlp, n, k, iters, lr=1e-2, weight=8e-3, contour_tolerance=1e-3, desired_distance=0.5):
    # train the MLP on an n x n mesh so the i+j=k diagonal becomes one distance contour
    xz, F = make_grid(n)
    pairs = adjacent_face_pairs(F)
    src = torch.tensor([0, n * n - 1])  # two opposite corners
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

        contour_loss = (d - d.mean()).square().mean()
        contour_std = d.std(correction=0)

        if iteration % 25 == 0:
            print(
                f"  {n}x{n}: "
                f"iteration={iteration}, "
                f"contour_loss={contour_loss.item():.6e}, "
                f"contour_std={contour_std.item():.6e}",
                flush=True,
            )

        if contour_std.item() <= contour_tolerance:
            converged = True
            break

        if iteration == iters: 
            break
        
        loss_smoothness = normal_smoothness_loss(V, F, pairs)
        loss = contour_loss + weight * loss_smoothness
        loss.backward()

        for name, param in mlp.named_parameters():
            if param.grad is not None:
                print(f"  {name}: {param.grad.norm().item():.2e}")

            if not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Non-finite gradient detected in {name} at iteration {iteration}")
        opt.step()

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        V = heights_to_verts(xz, mlp(xz))
        phi = heat_method_distance(V, F, src, t_factor=10.0)
        d = phi[diag]

        contour_loss = (d - d.mean()).square().mean().item()
        contour_std = d.std(correction=0).item()

        # Desired-distance accuracy is reported only as an extra metric.
        target_error = d - desired_distance
        target_l2 = torch.linalg.vector_norm(target_error).item()
        target_rmse = target_error.square().mean().sqrt().item()
        target_max_error = target_error.abs().max().item()
        mean_distance = d.mean().item()
    
    print(
        f"  {n}x{n}: "
        f"converged={converged}, "
        f"iterations={iteration}, "
        f"time={elapsed:.4f}s, "
        f"contour_loss={contour_loss:.6e}, "
        f"contour_std={contour_std:.6e}, "
        f"mean_distance={mean_distance:.6e}, "
        f"target_l2={target_l2:.6e}, "
        f"target_rmse={target_rmse:.6e}, "
        f"target_max_error={target_max_error:.6e}",
        flush=True,
    )

    return xz, F, src, diag



def sample(mlp, n, k):
    # evaluate the trained MLP at any resolution (no training) -> continuous surface
    xz, F = make_grid(n)
    src = torch.tensor([0, n * n - 1])
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


def main():
    torch.manual_seed(0)
    mlp = MLP(scale=0.1)

    print("train the MLP on a cheap 32x32 mesh:")
    train(mlp, 32, k=8, iters=400, contour_tolerance=1e-3, desired_distance=0.5)

    print("analyze the trained MLP with Fourier analysis:")
    analyze_height_spectrum(mlp, n=128, cutoff=0.25, out="results/geodesics/fourier_analysis.png", show=True)

    print("evaluate the SAME weights at 128x128 (no retraining):")
    xz, F, V, phi, src, diag = sample(mlp, 128, k=31)
    render(V, F, phi, src, diag, "results/geodesics/neural_terrain.png")


if __name__ == "__main__":
    main()


'''
cd /Users/huyufan/iskra-heightfield-publish
source .venv/bin/activate
python neural_terrain.py
'''