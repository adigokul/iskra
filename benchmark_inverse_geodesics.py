"""Benchmark Iskra's inverse-geodesics application without visualization.

This keeps the optimization used by ``iskra/apps/inverse_geodesics.py`` but
removes Polyscope, OBJ output, and per-iteration logging.  Convergence is
defined only by the L2 norm of the target-distance error.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import iskra.sparse as sp
import iskra.sparse_linalg as sparse_linalg
from iskra.apps.inverse_geodesics import rdg_solve
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry import triangle_areas
from iskra.mesh import Mesh
from iskra.sparse_linalg import linear_solve, min_quadratic_energy
from iskra.topology import face_index


# Match the backend choice in the supplied Heat Method program.
sparse_linalg._cholmod_available = False


@dataclass
class RunResult:
    method: str
    converged: bool
    elapsed_seconds: float
    iterations: int
    final_l2: float
    final_rmse: float
    final_max_error: float


class HeightfieldFromVertices(torch.nn.Module):
    """Optimize only y while preserving the supplied initial geometry."""

    def __init__(self, vertices: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("xz", vertices[:, (0, 2)].clone())
        # Iskra's inverse-geodesics app represents height as a [V, 1]
        # column vector; keep that convention for its quadratic forms.
        self.y = torch.nn.Parameter(vertices[:, 1:2].clone())

    def forward(self) -> torch.Tensor:
        return torch.cat((self.xz[:, :1], self.y, self.xz[:, 1:]), dim=1)


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading the wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distance_errors(
    distance: torch.Tensor,
    targets: torch.Tensor,
    desired_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = distance[targets] - desired_distance
    return (
        torch.linalg.vector_norm(error),
        torch.sqrt(torch.mean(error.square())),
        error.abs().max(),
    )


def heat_method_distance(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    t_factor: float,
) -> torch.Tensor:
    """The differentiable Heat Method from the supplied implementation."""
    lap, mass = laplacian(vertices, faces)
    triangles = face_index(vertices, faces)
    edges = triangles - triangles[:, [1, 2, 0], :]
    mean_edge_length = torch.linalg.vector_norm(edges, dim=-1).mean()
    heat_time = t_factor * mean_edge_length.square()

    delta = vertices.new_zeros(vertices.shape[0])
    delta[source] = 1.0
    temperature = linear_solve(lap * heat_time + mass, delta)[1]

    gradient = grad(vertices, faces, stack=True)
    grad_temperature = sp.matmul(gradient, temperature).reshape(3, -1)
    direction = -grad_temperature / torch.linalg.vector_norm(
        grad_temperature, dim=0, keepdim=True
    ).clamp_min(1e-12)

    areas = triangle_areas(triangles)
    divergence = sp.mul(torch.cat(3 * [areas])[None, :], gradient.mT.coalesce())
    rhs = sp.matmul(divergence, direction.flatten())
    distance = linear_solve(lap + 1e-8 * mass, rhs)[1]
    return distance - distance[source].mean()


def make_adjacent_face_pairs(faces: torch.Tensor) -> torch.Tensor:
    """Build the fixed face adjacency once, outside the timed loop."""
    edge_to_face: dict[tuple[int, int], int] = {}
    pairs: list[list[int]] = []
    for face_index_value, face in enumerate(faces.detach().cpu().tolist()):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(a, b), max(a, b))
            if edge in edge_to_face:
                pairs.append([edge_to_face[edge], face_index_value])
            else:
                edge_to_face[edge] = face_index_value
    return torch.tensor(pairs, dtype=torch.long, device=faces.device)


def normal_smoothness_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    adjacent: torch.Tensor,
) -> torch.Tensor:
    """Mean unit-normal jump over interior edges."""
    triangles = vertices[faces]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    normals = normals / torch.linalg.vector_norm(
        normals, dim=1, keepdim=True
    ).clamp_min(1e-12)
    return (normals[adjacent[:, 0]] - normals[adjacent[:, 1]]).square().sum(1).mean()


def run_heat_method(
    initial_vertices: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    targets: torch.Tensor,
    desired_distance: float,
    *,
    l2_tolerance: float,
    max_iterations: int,
    learning_rate: float,
    smoothing: float,
    t_factor: float,
) -> RunResult:
    """Run the supplied Heat Method optimizer to the common L2 threshold."""
    device = initial_vertices.device
    terrain = HeightfieldFromVertices(initial_vertices)
    optimizer = torch.optim.Adam(terrain.parameters(), lr=learning_rate)
    source_indices = source.to(device=device, dtype=torch.long).reshape(-1)
    adjacent = make_adjacent_face_pairs(faces)
    converged = False
    iterations = 0

    synchronize(device)
    start_time = time.perf_counter()
    for iteration in range(max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        vertices = terrain()
        distance = heat_method_distance(vertices, faces, source_indices, t_factor)
        l2_error, rmse, max_error = distance_errors(
            distance, targets, desired_distance
        )
        if l2_error.item() <= l2_tolerance:
            converged = True
            iterations = iteration
            break
        if iteration == max_iterations:
            iterations = max_iterations
            break

        loss = l2_error.square() + smoothing * normal_smoothness_loss(
            vertices, faces, adjacent
        )
        loss.backward()
        if terrain.y.grad is None or not torch.isfinite(terrain.y.grad).all():
            raise RuntimeError("Heat Method produced a missing or non-finite gradient")
        optimizer.step()
        iterations = iteration + 1

    synchronize(device)
    elapsed = time.perf_counter() - start_time
    return RunResult(
        method="heat_method",
        converged=converged,
        elapsed_seconds=elapsed,
        iterations=iterations,
        final_l2=l2_error.item(),
        final_rmse=rmse.item(),
        final_max_error=max_error.item(),
    )


def run_iskra(
    initial_vertices: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    targets: torch.Tensor,
    desired_distance: float,
    *,
    l2_tolerance: float,
    max_iterations: int,
    learning_rate: float,
    smoothing: float,
    sobolev_factor: float,
) -> RunResult:
    """Run the optimization from Iskra's inverse_geodesics application."""
    device = initial_vertices.device
    dtype = initial_vertices.dtype
    terrain = HeightfieldFromVertices(initial_vertices).to(device=device, dtype=dtype)
    optimizer = torch.optim.SGD(terrain.parameters(), lr=learning_rate)
    bc_idx = source.to(device=device, dtype=torch.long).reshape(-1)
    solver = None
    converged = False
    iterations = 0

    synchronize(device)
    start_time = time.perf_counter()

    for iteration in range(max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)

        vertices = terrain()
        distance = rdg_solve(vertices, faces, bc_idx)
        l2_error, rmse, max_error = distance_errors(
            distance, targets, desired_distance
        )

        # This is the common, method-independent termination condition.
        if l2_error.item() <= l2_tolerance:
            converged = True
            iterations = iteration
            break

        if iteration == max_iterations:
            iterations = max_iterations
            break

        # Preserve the objective and Sobolev-gradient update from Iskra's app.
        distance_loss = l2_error.square()
        lap, mass = laplacian(vertices, faces)
        smooth_loss = sp.matmul(terrain.y.mT, sp.matmul(lap, terrain.y))
        nonnegative_loss = torch.relu(-torch.log(terrain.y + 1)).sum()
        loss = distance_loss + smoothing * smooth_loss + 0.1 * nonnegative_loss
        loss.backward()

        with torch.no_grad():
            assert terrain.y.grad is not None
            sobolev_matrix = mass + sobolev_factor * lap
            solver, smoothed_gradient = min_quadratic_energy(
                sobolev_matrix,
                sp.matmul(mass, terrain.y.grad),
                bc_idx,
                torch.zeros((1, 1), dtype=dtype, device=device),
                solver=solver,
            )

            step_size = learning_rate
            max_gradient = smoothed_gradient.abs().max().item()
            while step_size * max_gradient > 0.1:
                step_size *= 0.5
            for group in optimizer.param_groups:
                group["lr"] = step_size
            terrain.y.grad = smoothed_gradient

        optimizer.step()
        iterations = iteration + 1

    synchronize(device)
    elapsed = time.perf_counter() - start_time

    return RunResult(
        method="iskra",
        converged=converged,
        elapsed_seconds=elapsed,
        iterations=iterations,
        final_l2=l2_error.item(),
        final_rmse=rmse.item(),
        final_max_error=max_error.item(),
    )


def parse_indices(text: str, device: torch.device) -> torch.Tensor:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one target index is required")
    return torch.tensor(values, dtype=torch.long, device=device)


def make_heightfield(
    nx: int,
    nz: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the generated grid and smooth initial bump from the attachment."""
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype, device=device)
    z = torch.linspace(0.0, 1.0, nz, dtype=dtype, device=device)
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    u = torch.linspace(0.0, torch.pi, nx, dtype=dtype, device=device)
    v = torch.linspace(0.0, torch.pi, nz, dtype=dtype, device=device)
    height = 1e-3 * (torch.sin(u)[:, None] * torch.sin(v)[None, :]).reshape(-1)
    vertices = torch.stack((xx.reshape(-1), height, zz.reshape(-1)), dim=1)

    faces: list[list[int]] = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            v00 = i * nz + j
            v01 = v00 + 1
            v10 = (i + 1) * nz + j
            v11 = v10 + 1
            faces.extend(([v00, v01, v10], [v10, v01, v11]))
    return vertices, torch.tensor(faces, dtype=torch.long, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time Iskra inverse geodesics to a common L2 threshold."
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        help="Optional mesh path; omit it to generate the attachment's grid",
    )
    parser.add_argument(
        "--source", default="0", help="One index or comma-separated source indices"
    )
    parser.add_argument(
        "--targets", help="Comma-separated targets; defaults to a grid diagonal"
    )
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--target-diagonal", type=int, default=10)
    parser.add_argument("--desired-distance", type=float, required=True)
    parser.add_argument("--l2-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=500.0)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--sobolev-factor", type=float, default=20.0)
    parser.add_argument("--heat-learning-rate", type=float, default=1e-2)
    parser.add_argument("--heat-smoothing", type=float, default=1e-4)
    parser.add_argument("--t-factor", type=float, default=10.0)
    parser.add_argument(
        "--method", choices=("both", "iskra", "heat"), default="both"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Complete runs excluded from reported timing",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_iskra.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    dtype = torch.float64
    if args.mesh is None:
        vertices, faces = make_heightfield(
            args.nx, args.nz, dtype=dtype, device=device
        )
        if args.targets is None:
            targets = torch.tensor(
                [
                    i * args.nz + j
                    for i in range(args.nx)
                    for j in range(args.nz)
                    if i + j == args.target_diagonal
                ],
                dtype=torch.long,
                device=device,
            )
        else:
            targets = parse_indices(args.targets, device)
    else:
        mesh, _ = Mesh.from_path(args.mesh, dtype=dtype, device=str(device))
        vertices = mesh.geom.vertices.to(device=device, dtype=dtype)
        faces = mesh.topo.faces.to(device=device)
        if args.targets is None:
            raise SystemExit("--targets is required when a mesh file is supplied")
        targets = parse_indices(args.targets, device)
    source = parse_indices(args.source, device)
    vertices[source, 1] = 0.0

    common = dict(
        initial_vertices=vertices,
        faces=faces,
        source=source,
        targets=targets,
        desired_distance=args.desired_distance,
        l2_tolerance=args.l2_tolerance,
        max_iterations=args.max_iterations,
    )
    runners = []
    if args.method in ("both", "iskra"):
        runners.append(
            lambda: run_iskra(
                **common,
                learning_rate=args.learning_rate,
                smoothing=args.smoothing,
                sobolev_factor=args.sobolev_factor,
            )
        )
    if args.method in ("both", "heat"):
        runners.append(
            lambda: run_heat_method(
                **common,
                learning_rate=args.heat_learning_rate,
                smoothing=args.heat_smoothing,
                t_factor=args.t_factor,
            )
        )

    all_results: list[RunResult] = []
    for runner in runners:
        for _ in range(args.warmup_runs):
            runner()
        all_results.extend(runner() for _ in range(args.repetitions))

    by_method: dict[str, list[RunResult]] = {}
    for result in all_results:
        by_method.setdefault(result.method, []).append(result)
    report = {
        "configuration": {
            "mesh": str(args.mesh.resolve()) if args.mesh else "generated-grid",
            "source": source.detach().cpu().tolist(),
            "targets": targets.detach().cpu().tolist(),
            "desired_distance": args.desired_distance,
            "l2_tolerance": args.l2_tolerance,
            "max_iterations": args.max_iterations,
            "device": str(device),
            "dtype": str(dtype),
        },
        "methods": {
            method: {
                "median_elapsed_seconds": statistics.median(
                    result.elapsed_seconds for result in results
                ),
                "runs": [asdict(result) for result in results],
            }
            for method, results in by_method.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
