"""Compare direct-height and MLP-height Pareto sweeps.

Both parameterizations use the same mesh, differentiable Heat Method, distance
objective, normal-smoothness energy, Adam optimizer, and fixed iteration budget.
Only the representation of the heightfield changes.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from heightfield_optimization import (
    make_adjacent_face_pairs,
    make_heightfield,
    normal_smoothness_loss,
)
from iskra.geometry.geodesics import heat_method_distance
from neural_terrain import MLP, diagonal, heights_to_verts


DEFAULT_WEIGHTS = [
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


@dataclass
class ParetoResult:
    method: str
    weight: float
    contour_std: float
    contour_range: float
    target_l2: float
    normal_smoothness: float
    elapsed_seconds: float
    height_range: float
    parameter_count: int


def distance_loss(
    target_distances: torch.Tensor,
    objective: str,
    desired_distance: float,
) -> torch.Tensor:
    """Return the distance objective selected for this complete sweep."""
    if objective == "contour":
        return (target_distances - target_distances.mean()).square().mean()
    if objective == "target":
        error = target_distances - desired_distance
        return error.square().sum()
    raise ValueError(f"unknown objective: {objective}")


def evaluate(
    method: str,
    weight: float,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    targets: torch.Tensor,
    adjacent_pairs: torch.Tensor,
    *,
    desired_distance: float,
    t_factor: float,
    elapsed_seconds: float,
    parameter_count: int,
) -> ParetoResult:
    """Measure both distance metrics and normal smoothness on a final surface."""
    with torch.no_grad():
        distances = heat_method_distance(
            vertices,
            faces,
            source,
            t_factor=t_factor,
        )
        target_distances = distances[targets]
        target_error = target_distances - desired_distance

        return ParetoResult(
            method=method,
            weight=weight,
            contour_std=target_distances.std(correction=0).item(),
            contour_range=(target_distances.max() - target_distances.min()).item(),
            target_l2=torch.linalg.vector_norm(target_error).item(),
            normal_smoothness=normal_smoothness_loss(
                vertices,
                faces,
                adjacent_pairs,
            ).item(),
            elapsed_seconds=elapsed_seconds,
            height_range=(vertices[:, 1].max() - vertices[:, 1].min()).item(),
            parameter_count=parameter_count,
        )


def check_gradient(name: str, gradient: torch.Tensor | None) -> None:
    if gradient is None:
        raise RuntimeError(f"missing gradient: {name}")
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"non-finite gradient: {name}")


def optimize_direct_heights(
    initial_heights: torch.Tensor,
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    targets: torch.Tensor,
    adjacent_pairs: torch.Tensor,
    *,
    objective: str,
    desired_distance: float,
    weight: float,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    height_scale: float,
) -> ParetoResult:
    """Optimize one bounded height parameter per mesh vertex."""
    normalized_heights = (initial_heights / height_scale).clamp(
        min=-1.0 + 1e-12,
        max=1.0 - 1e-12,
    )
    raw_heights = torch.nn.Parameter(
        torch.atanh(normalized_heights).clone()
    )
    optimizer = torch.optim.Adam([raw_heights], lr=learning_rate)

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        heights = height_scale * torch.tanh(raw_heights)
        vertices = heights_to_verts(fixed_xz, heights)
        distances = heat_method_distance(
            vertices,
            faces,
            source,
            t_factor=t_factor,
        )
        loss = distance_loss(
            distances[targets],
            objective,
            desired_distance,
        ) + weight * normal_smoothness_loss(
            vertices,
            faces,
            adjacent_pairs,
        )
        loss.backward()
        check_gradient("direct heights", raw_heights.grad)
        optimizer.step()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        heights = height_scale * torch.tanh(raw_heights)
        vertices = heights_to_verts(fixed_xz, heights)

    return evaluate(
        "direct_height",
        weight,
        vertices,
        faces,
        source,
        targets,
        adjacent_pairs,
        desired_distance=desired_distance,
        t_factor=t_factor,
        elapsed_seconds=elapsed,
        parameter_count=raw_heights.numel(),
    )


def make_initial_mlp(hidden: int, height_scale: float, seed: int) -> MLP:
    """Recreate the same deterministic MLP for every smoothness weight."""
    torch.manual_seed(seed)
    return MLP(hidden=hidden, scale=height_scale)


def optimize_mlp_heights(
    fixed_xz: torch.Tensor,
    faces: torch.Tensor,
    source: torch.Tensor,
    targets: torch.Tensor,
    adjacent_pairs: torch.Tensor,
    *,
    objective: str,
    desired_distance: float,
    weight: float,
    iterations: int,
    learning_rate: float,
    t_factor: float,
    height_scale: float,
    hidden: int,
    seed: int,
) -> ParetoResult:
    """Optimize MLP weights; recreate the same initial MLP for every weight."""
    mlp = make_initial_mlp(hidden, height_scale, seed)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        vertices = heights_to_verts(fixed_xz, mlp(fixed_xz))
        distances = heat_method_distance(
            vertices,
            faces,
            source,
            t_factor=t_factor,
        )
        loss = distance_loss(
            distances[targets],
            objective,
            desired_distance,
        ) + weight * normal_smoothness_loss(
            vertices,
            faces,
            adjacent_pairs,
        )
        loss.backward()
        for name, parameter in mlp.named_parameters():
            check_gradient(name, parameter.grad)
        optimizer.step()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        vertices = heights_to_verts(fixed_xz, mlp(fixed_xz))

    return evaluate(
        "mlp_height",
        weight,
        vertices,
        faces,
        source,
        targets,
        adjacent_pairs,
        desired_distance=desired_distance,
        t_factor=t_factor,
        elapsed_seconds=elapsed,
        parameter_count=sum(parameter.numel() for parameter in mlp.parameters()),
    )


def write_results(rows: list[ParetoResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pareto_parameterizations.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return csv_path


def plot_results(
    rows: list[ParetoResult],
    objective: str,
    output_dir: Path,
    *,
    show: bool,
) -> Path:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_field = "contour_std" if objective == "contour" else "target_l2"
    y_label = "Target diagonal distance std" if objective == "contour" else "Target-distance L2 error"

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    styles = {
        "direct_height": ("o", "Direct vertex heights"),
        "mlp_height": ("s", "MLP heightfield"),
    }
    for method, (marker, label) in styles.items():
        method_rows = [row for row in rows if row.method == method]
        x_values = [row.normal_smoothness for row in method_rows]
        y_values = [getattr(row, y_field) for row in method_rows]
        axis.plot(x_values, y_values, marker=marker, label=label)
        for row, x_value, y_value in zip(method_rows, x_values, y_values):
            axis.annotate(
                f"{row.weight:.0e}",
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    axis.set_xlabel("Normal smoothness energy")
    axis.set_ylabel(y_label)
    axis.set_title("Height parameterization Pareto comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure_path = output_dir / f"pareto_{objective}.png"
    figure.savefig(figure_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return figure_path


def run_pareto_analysis(args: argparse.Namespace) -> None:
    vertices, faces = make_heightfield(
        args.n,
        args.n,
        dtype=torch.double,
        device="cpu",
    )
    fixed_xz = vertices[:, [0, 2]]
    source = torch.tensor([0, args.n * args.n - 1], dtype=torch.long)
    targets = diagonal(args.n, args.k)
    adjacent_pairs = make_adjacent_face_pairs(faces)

    # A perfectly flat heightfield can have a zero first derivative with respect
    # to height. Use one deterministic, slightly non-flat MLP surface to give
    # both representations exactly the same non-degenerate initialization.
    initial_mlp = make_initial_mlp(args.hidden, args.height_scale, args.seed)
    with torch.no_grad():
        initial_heights = initial_mlp(fixed_xz)

    rows: list[ParetoResult] = []
    for weight in args.weights:
        print(f"smoothness_weight={weight:.1e}", flush=True)
        direct_result = optimize_direct_heights(
            initial_heights,
            fixed_xz,
            faces,
            source,
            targets,
            adjacent_pairs,
            objective=args.objective,
            desired_distance=args.desired_distance,
            weight=weight,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            t_factor=args.t_factor,
            height_scale=args.height_scale,
        )
        mlp_result = optimize_mlp_heights(
            fixed_xz,
            faces,
            source,
            targets,
            adjacent_pairs,
            objective=args.objective,
            desired_distance=args.desired_distance,
            weight=weight,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            t_factor=args.t_factor,
            height_scale=args.height_scale,
            hidden=args.hidden,
            seed=args.seed,
        )
        rows.extend([direct_result, mlp_result])

        for result in (direct_result, mlp_result):
            print(
                f"  {result.method}: "
                f"std={result.contour_std:.6e}, "
                f"target_l2={result.target_l2:.6e}, "
                f"normal_smoothness={result.normal_smoothness:.6e}, "
                f"time={result.elapsed_seconds:.3f}s",
                flush=True,
            )

    csv_path = write_results(rows, args.output_dir)
    figure_path = plot_results(
        rows,
        args.objective,
        args.output_dir,
        show=not args.no_show,
    )
    print(f"Pareto data saved to: {csv_path.resolve()}")
    print(f"Pareto plot saved to: {figure_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare direct-height and MLP-height Pareto sweeps."
    )
    parser.add_argument("--objective", choices=("contour", "target"), default="contour")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--t-factor", type=float, default=10.0)
    parser.add_argument("--height-scale", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--desired-distance", type=float, default=0.5)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/geodesics/pareto_parameterizations"),
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    run_pareto_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
