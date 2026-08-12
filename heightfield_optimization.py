"""
Optimize a heightfield so the Heat Method distances on a target
vertex set match a fixed desired distance.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import sys

import torch

import iskra.sparse as sp
import iskra.sparse_linalg as sparse_linalg
from iskra.sparse_linalg import linear_solve, min_quadratic_energy
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry import triangle_areas
from iskra.sparse_linalg import linear_solve
from iskra.topology import face_index

# Use the installed cholespy backend with this Iskra revision.
sparse_linalg._cholmod_available = False


@dataclass
class OptimizationResult:
    """Geometry and diagnostics returned by one optimization run."""

    vertices: torch.Tensor
    distance: torch.Tensor
    distance_mean: float
    distance_min: float
    distance_max: float
    distance_std: float
    distance_range: float
    smoothness: float
    distance_rmse: float
    distance_max_error: float


def make_heightfield(
    nx: int,
    nz: int,
    *,
    width: float = 1.0,
    depth: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",) -> tuple[torch.Tensor, torch.Tensor]:
    """Return planar [x, y, z] vertices and a fixed triangular connectivity."""
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must both be at least 2")

    x = torch.linspace(0.0, width, nx, dtype=dtype, device=device)
    z = torch.linspace(0.0, depth, nz, dtype=dtype, device=device)
    # xx[i, j] == x[i]; zz[i, j] == z[j]
    xx, zz = torch.meshgrid(x, z, indexing="ij") 
    
    vertices = torch.stack(
        (xx.reshape(-1),torch.zeros(nx * nz, dtype=dtype, device=device),zz.reshape(-1),),dim=1,
    )

    faces: list[list[int]] = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            v00 = i * nz + j
            v01 = v00 + 1
            v10 = (i + 1) * nz + j
            v11 = v10 + 1
            faces.append([v00, v01, v10])
            faces.append([v10, v01, v11])

    return vertices, torch.tensor(faces, dtype=torch.long, device=device)

'''
def make_diagonals(nx: int, nz: int, device: str) -> list[torch.Tensor]:
    """Group vertex indices by i + j = k, skipping one-vertex diagonals."""
    diagonals: list[torch.Tensor] = []
    for k in range(nx + nz - 1):
        indices = [i * nz + j for i in range(nx) for j in range(nz) if i + j == k]
        if len(indices) >= 2:
            diagonals.append(torch.tensor(indices, dtype=torch.long, device=device))
    return diagonals


def diagonal_distance_loss(distance: torch.Tensor, diagonals: list[torch.Tensor]) -> torch.Tensor:
    """Sum the distance variance on every grid diagonal."""
    losses = []
    for diagonal in diagonals:
        phi = distance[diagonal]
        losses.append((phi - phi.mean()).square().mean())
    return torch.stack(losses).sum()
'''


def height_smoothness_loss(heights: torch.Tensor, nx: int, nz: int) -> torch.Tensor:
    """Sum squared height differences over all axis-aligned grid edges."""
    y = heights.reshape(nx, nz)

    # Edges (i, j) -- (i + 1, j) in the x direction.
    x_edge_loss = (y[1:, :] - y[:-1, :]).square().sum()

    # Edges (i, j) -- (i, j + 1) in the z direction.
    z_edge_loss = (y[:, 1:] - y[:, :-1]).square().sum()

    return x_edge_loss + z_edge_loss


def make_adjacent_face_pairs(faces: torch.Tensor) -> torch.Tensor:
    """Return pairs of triangle indices whose faces share an edge."""
    edge_to_face: dict[tuple[int, int], int] = {}
    adjacent_pairs: list[list[int]] = []

    for current_face, face in enumerate(faces.detach().cpu().tolist()):
        edges = (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
        for vertex_a, vertex_b in edges:
            edge = (min(vertex_a, vertex_b), max(vertex_a, vertex_b))
            if edge in edge_to_face:
                adjacent_pairs.append([edge_to_face[edge], current_face])
            else:
                edge_to_face[edge] = current_face

    return torch.tensor(adjacent_pairs, dtype=torch.long, device=faces.device)

def normal_smoothness_loss(vertices: torch.Tensor, faces: torch.Tensor, adjacent_face_pairs: torch.Tensor) -> torch.Tensor:
    """Penalize unit-normal differences between adjacent triangles."""
    triangles = vertices[faces]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]

    normals = torch.linalg.cross(edge_1, edge_2, dim=1)
    normals = normals / torch.linalg.vector_norm(
        normals, dim=1, keepdim=True
    ).clamp_min(1e-12)

    normal_1 = normals[adjacent_face_pairs[:, 0]]
    normal_2 = normals[adjacent_face_pairs[:, 1]]
    return (normal_1 - normal_2).square().sum(dim=1).mean()

def heat_method_distance(vertices, faces, source, t_factor=1.0):
    # geodesic distance from the source(s) to every vertex.
    # t_factor scales the diffusion time (t = t_factor * mean_edge_len^2),
    # bunny needs ~10, clean grids ~1
    n = vertices.shape[0]
    source = torch.as_tensor(source, dtype=torch.long, device=vertices.device).reshape(-1)

    lap, mass = laplacian(vertices, faces)
    tri = face_index(vertices, faces)
    h = torch.linalg.vector_norm(tri - tri[:, [1, 2, 0], :], dim=-1).mean()
    time = t_factor * h.square()

    # diffuse heat from the source
    delta = vertices.new_zeros(n)
    delta[source] = 1.0
    u = linear_solve(lap * time + mass, delta)[1]

    # normalized gradient, pointing away from the source
    G = grad(vertices, faces, stack=True)
    g = sp.matmul(G, u).reshape(3, -1)
    X = -g / torch.linalg.vector_norm(g, dim=0, keepdim=True).clamp_min(1e-12)

    # solve L phi = div(X) with phi fixed to 0 at the source
    areas = triangle_areas(tri)
    div = sp.mul(torch.cat(3 * [areas])[None, :], G.mT.coalesce())
    rhs = sp.matmul(div, X.flatten())
    return min_quadratic_energy(lap, rhs, source, rhs.new_zeros(source.numel()))[1]

def optimize_heightfield(
    initial_heights: torch.Tensor,
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    target_diagonal: torch.Tensor,
    adjacent_face_pairs: torch.Tensor,
    *,
    nx: int,
    nz: int,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    desired_distance: float,
    smoothness_weight: float,
    print_every: int | None = 50,
) -> OptimizationResult:
    """Optimize one heightfield for one smoothness weight."""
    # clone() is important: every Pareto experiment starts from exactly the
    # same initial heightfield instead of continuing from the previous run.
    heights = torch.nn.Parameter(initial_heights.clone())
    optimizer = torch.optim.Adam([heights], lr=learning_rate)

    for iteration in range(iterations):
        optimizer.zero_grad()

        # y column depends on the optimizer
        vertices = torch.stack((fixed_xz[:, 0], heights, fixed_xz[:, 1]), dim=1)
        distance = heat_method_distance(vertices, faces, source, t_factor=t_factor)

        # All-diagonals loss:
        # loss = diagonal_distance_loss(distance, diagonals)
        # Equivalent name used by this reorganized function:
        # loss_distance = diagonal_distance_loss(distance, diagonals)

        target_phi = distance[target_diagonal]
        loss_distance = (target_phi - desired_distance).square().mean()
        loss_smoothness = normal_smoothness_loss(vertices, faces, adjacent_face_pairs)
        # loss_height = height_smoothness_loss(heights, nx, nz)
        # loss = loss_distance + smoothness_weight * loss_height
        loss = loss_distance + smoothness_weight * loss_smoothness
        loss.backward()

        if heights.grad is None or not torch.isfinite(heights.grad).all():
            raise RuntimeError("The height gradient is missing or non-finite")

        gradient_norm = heights.grad.norm().item()
        optimizer.step()

        # Remove the irrelevant global y translation by anchoring the source
        # Global y translation does not change intrinsic distances, so source
        # y=0 removes an irrelevant degree of freedom.
        # with torch.no_grad():
        #     heights.sub_(heights[source].mean().item())

        should_print = print_every is not None and (
            iteration % print_every == 0 or iteration == iterations - 1
        )
        if should_print:
            print(
                f"iteration={iteration:04d}, "
                f"total={loss.item():.8e}, "
                f"distance={loss_distance.item():.8e}, "
                f"smooth={loss_smoothness.item():.8e}, "
                f"weighted_smoothness="
                f"{(smoothness_weight * loss_smoothness).item():.8e}, "
                f"grad={gradient_norm:.8e}, "
                f"y_range=[{heights.min().item():.4e}, "
                f"{heights.max().item():.4e}]"
            )

    optimized_vertices = torch.stack(
        (fixed_xz[:, 0], heights.detach(), fixed_xz[:, 1]), dim=1
    )
    final_distance = heat_method_distance(optimized_vertices, faces, source, t_factor=t_factor)
    final_target_phi = final_distance[target_diagonal]

    target_error = final_target_phi - desired_distance
    return OptimizationResult(
        vertices=optimized_vertices,
        distance=final_distance.detach(),
        distance_mean=final_target_phi.mean().item(),
        distance_min=final_target_phi.min().item(),
        distance_max=final_target_phi.max().item(),
        distance_std=final_target_phi.std(correction=0).item(),
        distance_range=(
            final_target_phi.max() - final_target_phi.min()
        ).item(),
        distance_rmse=target_error.square().mean().sqrt().item(),
        distance_max_error=target_error.abs().max().item(),
        smoothness=normal_smoothness_loss(
            optimized_vertices,
            faces,
            adjacent_face_pairs,
        ).item(),
    )


# ---------------------------------------------------------------------------
# Pareto analysis (distance vs smoothness)
# ---------------------------------------------------------------------------

def run_pareto_analysis(
    initial_heights: torch.Tensor,
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    target_diagonal: torch.Tensor,
    *,
    nx: int,
    nz: int,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    desired_distance: float,
    output_dir: Path,
) -> None:
    """Sweep smoothness weights and plot the distance-smoothness trade-off."""
    import matplotlib.pyplot as plt

    weights = [
        0.0,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ]
    
    rows: list[dict[str, float]] = []

    adjacent_pairs = make_adjacent_face_pairs(faces)
    for weight in weights:
        print(f"Pareto run: smoothness_weight={weight:.1e}")
        result = optimize_heightfield(
        initial_heights,    
        fixed_xz,
        faces,
        source,
        target_diagonal,
        adjacent_pairs,
        nx=nx,
        nz=nz,
        iterations=iterations,
        learning_rate=learning_rate,
        t_factor=t_factor,
        desired_distance=desired_distance,
        smoothness_weight=weight,
        )
        row = {
            "weight": weight,
            "distance_rmse": result.distance_rmse,
            "distance_max_error": result.distance_max_error,
            "distance_std": result.distance_std,
            "distance_range": result.distance_range,
            "smoothness": result.smoothness,
        }
        rows.append(row)
        print(
            f"  std={row['distance_std']:.8e}, "
            f"range={row['distance_range']:.8e}, "
            f"smoothness={row['smoothness']:.8e}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pareto_results.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    smoothness_values = [row["smoothness"] for row in rows]
    distance_errors = [row["distance_rmse"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        smoothness_values,
        distance_errors,
        s=60,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    for row in rows:
        axis.annotate(
            f"lambda={row['weight']:.0e}",
            (row["smoothness"], row["distance_rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("Normal smoothness energy")
    axis.set_ylabel("Target distance RMSE")
    axis.set_title("Distance–smoothness Pareto analysis")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    figure_path = output_dir / "pareto.png"
    figure.savefig(figure_path, dpi=200)
    print(f"Pareto data saved to: {csv_path.resolve()}")
    print(f"Pareto plot saved to: {figure_path.resolve()}")
    print(
        f"  rmse={row['distance_rmse']:.8e}, "
        f"max_error={row['distance_max_error']:.8e}, "
        f"std={row['distance_std']:.8e}, "
        f"smoothness={row['smoothness']:.8e}"
    )
    plt.show()

def visualize_result(
    display_vertices: torch.Tensor,
    faces: torch.Tensor,
    distance: torch.Tensor,
    source: torch.Tensor,
    target_diagonal: torch.Tensor,
    target_k: int,
) -> None:
    """Display the optimized surface, distance contours, and target diagonal."""
    import polyscope as ps

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps_mesh = ps.register_surface_mesh(
        # "geodesic contours on flat domain",
        "optimized heightfield",
        display_vertices.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
    )
    ps_mesh.add_scalar_quantity(
        "heat-method distance",
        distance.detach().cpu().numpy(),
        defined_on="vertices",
        isolines_enabled=True,
        enabled=True,
    )
    ps.register_point_cloud(
        "source",
        display_vertices[source].detach().cpu().numpy(),
        enabled=True,
        radius=0.005,
    )

    # Relaxed single-diagonal debugging visual:
    target_points = display_vertices[target_diagonal].detach().cpu().numpy()
    n_target = target_diagonal.numel()
    target_edges = torch.stack(
        (torch.arange(n_target - 1), torch.arange(1, n_target)), dim=1
    ).numpy()
    ps.register_curve_network(
        f"target diagonal k={target_k}",
        target_points,
        target_edges,
        color=(1.0, 0.0, 0.0),
        radius=0.004,
        enabled=True,
    )
    ps.show()


# ---------------------------------------------------------------------------
# Comparison table experiments (fixed source or fixed target)
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    name: str
    source: torch.Tensor
    target: torch.Tensor
    source_label: str
    target_label: str

def make_target_line(nx: int, nz: int, kind: str, value: int, device: str) -> torch.Tensor:
    """Return vertex indices for a target line: diagonal, row, or column."""
    indices: list[int] = []
    if kind == "diag":
        for i in range(nx):
            for j in range(nz):
                if i + j == value:
                    indices.append(i * nz + j)
    elif kind == "row":
        i = value
        for j in range(nz):
            indices.append(i * nz + j)
    elif kind == "col":
        j = value
        for i in range(nx):
            indices.append(i * nz + j)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    if len(indices) < 2:
        raise ValueError(
            f"Target line '{kind}={value}' has fewer than 2 vertices"
        )
    return torch.tensor(indices, dtype=torch.long, device=device)

def run_comparison_table(
    initial_heights: torch.Tensor,
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    adjacent_face_pairs: torch.Tensor,
    *,
    nx: int,
    nz: int,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    smoothness_weight: float,   
    desired_distance: float,
    configs: list[ExpConfig],
    output_dir: Path,
) -> None:
    """Run one weight for all configs and save a CSV table."""
    rows: list[dict[str, float | str]] = []
    for cfg in configs:
        print(f"\n=== {cfg.name}: {cfg.source_label} → {cfg.target_label} ===")
        result = optimize_heightfield(
        initial_heights,
        fixed_xz,
        faces,
        cfg.source,
        cfg.target,
        adjacent_face_pairs,
        nx=nx,
        nz=nz,
        iterations=iterations,
        learning_rate=learning_rate,
        t_factor=t_factor,
        desired_distance=desired_distance,
        smoothness_weight=smoothness_weight,
        print_every=None,
    )
        rows.append(
            {
                "experiment": cfg.name,
                "source": cfg.source_label,
                "target": cfg.target_label,
                "weight": smoothness_weight,
                "distance_mean": result.distance_mean,
                "distance_std": result.distance_std,
                "distance_range": result.distance_range,
                "smoothness": result.smoothness,
            }
        )
        print(
            f"  std={result.distance_std:.4e}, "
            f"range={result.distance_range:.4e}, "
            f"smooth={result.smoothness:.4e}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison table saved to: {csv_path.resolve()}")
def run_compare_experiments(
    initial_heights: torch.Tensor,
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    adjacent_face_pairs: torch.Tensor,
    nx: int,
    nz: int,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    output_dir: Path,
    desired_distance: float,
    device: str,
) -> None:
    """Entry point for --compare: fixed weight, vary source/target geometry."""
    fixed_weight = 1e-4 

    # ------------------------------------------------------------------
    # Experiment A: fixed source, vary target
    # ------------------------------------------------------------------
    source_fixed = torch.tensor([0, nx * nz - 1], dtype=torch.long, device=device)

    configs_a = [
        ExpConfig("A1", source_fixed, make_target_line(nx, nz, "diag", (nx + nz - 2) // 2, device), "diag[0,399]", "diag_center"),
        ExpConfig("A2", source_fixed, make_target_line(nx, nz, "diag", (nx + nz - 2) // 4, device), "diag[0,399]", "diag_near"),
        ExpConfig("A3", source_fixed, make_target_line(nx, nz, "diag", 3 * (nx + nz - 2) // 4, device), "diag[0,399]", "diag_far"),
        ExpConfig("A4", source_fixed, make_target_line(nx, nz, "row", nx // 2, device), "diag[0,399]", "row_center"),
        ExpConfig("A5", source_fixed, make_target_line(nx, nz, "col", nz // 2, device), "diag[0,399]", "col_center"),
    ]

    run_comparison_table(
        initial_heights,
        fixed_xz,
        faces,
        adjacent_face_pairs,
        nx=nx,
        nz=nz,
        iterations=iterations,
        learning_rate=learning_rate,
        t_factor=t_factor,
        smoothness_weight=fixed_weight,
        desired_distance=desired_distance,
        configs=configs_a,
        output_dir=output_dir / "exp_A_fixed_source",
    )

    # ------------------------------------------------------------------
    # Experiment B: fixed target, vary source
    # ------------------------------------------------------------------
    target_fixed = make_target_line(nx, nz, "diag", (nx + nz - 2) // 2 - 1, device)

    configs_b = [
        ExpConfig("B1", torch.tensor([0], dtype=torch.long, device=device), target_fixed, "corner_0", "diag_center"),
        ExpConfig("B2", torch.tensor([nx * nz - 1], dtype=torch.long, device=device), target_fixed, "corner_399", "diag_center"),
        ExpConfig("B3", torch.tensor([0, nx * nz - 1], dtype=torch.long, device=device), target_fixed, "diag_pair", "diag_center"),
        ExpConfig("B4", torch.tensor([nz - 1, (nx - 1) * nz], dtype=torch.long, device=device), target_fixed, "anti_diag", "diag_center"),
        ExpConfig("B5", torch.tensor([(nx // 2) * nz + (nz // 2)], dtype=torch.long, device=device), target_fixed, "center", "diag_center"),
    ]

    run_comparison_table(
        initial_heights,
        fixed_xz,
        faces,
        adjacent_face_pairs,
        nx=nx,
        nz=nz,
        iterations=iterations,
        learning_rate=learning_rate,
        t_factor=t_factor,
        smoothness_weight=fixed_weight,
        desired_distance=desired_distance,
        configs=configs_b,
        output_dir=output_dir / "exp_B_fixed_target",
    )

def main() -> None:
    # Imitating inflate.py
    nx = 20
    nz = 20
    width = 1.0
    depth = 1.0
    iterations = 300
    learning_rate = 1e-3
    t_factor = 10.0
    smoothness_weight = 1e-4
    pareto_iterations = 200
    device = "cpu"
    dtype = torch.float64
    desired_distance = 0.5

    verts, faces = make_heightfield(
        nx, nz, width=width, depth=depth, dtype=dtype, device=device
    )
    fixed_xz = verts[:, [0, 2]].clone()

    target_k = 13
    target_diagonal = torch.tensor(
        [
            i * nz + j
            for i in range(nx)
            for j in range(nz)
            if i + j == target_k
        ],
        dtype=torch.long,
        device=device,
    )

    source = torch.tensor([0], dtype=torch.long, device=device)

    # source = torch.tensor(
    #     [0,(nz - 1), (nx - 1) * nz],
    #     dtype=torch.long,
    #     device=device,
    # )

    # A small, smooth, non-constant bump breaks the symmetry of the perfectly
    # flat mesh without introducing high-frequency random noise.
    u = torch.linspace(0.0, torch.pi, nx, dtype=dtype, device=device)
    v = torch.linspace(0.0, torch.pi, nz, dtype=dtype, device=device)
    initial_height_grid = torch.sin(u)[:, None] * torch.sin(v)[None, :]
    initial_heights = 1e-3 * initial_height_grid.reshape(-1)
    initial_heights[source] = 0.0

    adjacent_pairs = make_adjacent_face_pairs(faces)

    if "--pareto" in sys.argv:
        run_pareto_analysis(
            initial_heights,
            fixed_xz,
            faces,
            source,
            target_diagonal,
            nx=nx,
            nz=nz,
            iterations=pareto_iterations,
            learning_rate=learning_rate,
            t_factor=t_factor,
            desired_distance=desired_distance,
            output_dir=Path("results/heightfield_pareto"),
        )
        return

    if "--compare" in sys.argv:
        run_compare_experiments(
            initial_heights,
            fixed_xz,
            faces,
            adjacent_pairs,
            nx=nx,
            nz=nz,
            iterations=100,
            learning_rate=learning_rate,
            t_factor=t_factor,
            desired_distance=desired_distance,
            output_dir=Path("results/heightfield_compare"),
            device=device,
        )
        return
    
    result = optimize_heightfield(
        initial_heights,
        fixed_xz,
        faces,
        source,
        target_diagonal,
        adjacent_pairs,
        nx=nx,
        nz=nz,
        iterations=iterations,
        learning_rate=learning_rate,
        t_factor=t_factor,
        desired_distance=desired_distance,
        smoothness_weight=smoothness_weight,
    )

    optimized_verts = result.vertices
    final_distance = result.distance
    print("Target diagonal diagnostics:")
    print(f"  k={target_k}")
    print(f"  vertices={target_diagonal.numel()}")
    print(f"  desired_distance={desired_distance:.8e}")
    print(f"  mean={result.distance_mean:.8e}")
    print(f"  min={result.distance_min:.8e}")
    print(f"  max={result.distance_max:.8e}")
    print(f"  rmse={result.distance_rmse:.8e}")
    print(f"  max_error={result.distance_max_error:.8e}")
    print(f"  std={result.distance_std:.8e}")
    print(f"  range={result.distance_range:.8e}")
    print(f"  smoothness={result.smoothness:.8e}")

    output_dir = Path("results/heightfield_contours")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vertices": optimized_verts,
            "faces": faces,
            "distance": final_distance.detach(),
            "source": source,
            "nx": nx,
            "nz": nz,
            "target_k": target_k,
            "target_diagonal": target_diagonal,
        },
        output_dir / "optimized_heightfield.pt",
    )
    print(f"Result saved to: {(output_dir / 'optimized_heightfield.pt').resolve()}")

    if "--visualize" in sys.argv:
        # Original visualization on the optimized 3D heightfield:
        display_verts = optimized_verts

        # Debugging visualization: display the optimized distance field on the
        # original flat parameter domain so contour shape is easier to inspect.
        # display_verts = verts
        visualize_result(
            display_verts,
            faces,
            final_distance,
            source,
            target_diagonal,
            target_k,
        )

if __name__ == "__main__":
    main()


'''
cd /Users/huyufan/iskra-heightfield-publish
source .venv/bin/activate

python -c "import torch; import iskra.sparse; print('environment OK')"
'''


'''
/Users/huyufan/iskra-heightfield-publish/.venv/bin/python \
  /Users/huyufan/Documents/GitHub/iskra/heightfield_optimization.py \
  --visualize
'''

'''
/Users/huyufan/iskra-heightfield-publish/.venv/bin/python \
  /Users/huyufan/iskra-heightfield-publish/heightfield_optimization.py \
  --pareto
'''

'''
/Users/huyufan/iskra-heightfield-publish/.venv/bin/python \
  /Users/huyufan/iskra-heightfield-publish/heightfield_optimization.py \
  --compare
'''
