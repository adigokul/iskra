"""Optimize a heightfield so Heat Method distance is constant on grid diagonals.

The grid lies initially in the x-z plane.  Its x and z coordinates and triangle
connectivity stay fixed; only the y coordinate of each vertex is optimized.
"""

from pathlib import Path
import sys

import torch

from run_heat_method_core import heat_method_distance


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


def main() -> None:
    # Imitating inflate.py
    nx = 20
    nz = 20
    width = 1.0
    depth = 1.0
    iterations = 100
    learning_rate = 1e-2
    t_factor = 10.0
    device = "cpu"
    dtype = torch.float64

    verts, faces = make_heightfield(
        nx, nz, width=width, depth=depth, dtype=dtype, device=device
    )
    fixed_xz = verts[:, [0, 2]].clone()
    diagonals = make_diagonals(nx, nz, device)

    # target contours i+j=k
    source = torch.tensor([0], dtype=torch.long, device=device)

    torch.manual_seed(42)
    grid_spacing = min(width / (nx - 1), depth / (nz - 1))
    # Set a very small initial height to ensure a non-zero gradient, then enabling the optimizer to perform updates.
    initial_heights = 1e-3 * grid_spacing * torch.randn(
        nx * nz, dtype=dtype, device=device
    )
    initial_heights[source] = 0.0
    heights = torch.nn.Parameter(initial_heights)
    optimizer = torch.optim.SGD([heights], lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()

        # y column depends on the optimizer
        vertices = torch.stack((fixed_xz[:, 0], heights, fixed_xz[:, 1]), dim=1)
        distance, _, _ = heat_method_distance(
            vertices, faces, source, t_factor=t_factor
        )

        loss = diagonal_distance_loss(distance, diagonals)
        loss.backward()

        if heights.grad is None or not torch.isfinite(heights.grad).all():
            raise RuntimeError("The height gradient is missing or non-finite")

        optimizer.step()

        # Remove the irrelevant global y translation by anchoring the source
        with torch.no_grad():
            heights.sub_(heights[source].item())

        print(f"iteration={i:03d}, loss={loss.item():.8e}")

    optimized_verts = torch.stack(
        (fixed_xz[:, 0], heights.detach(), fixed_xz[:, 1]), dim=1
    )
    final_distance, _, _ = heat_method_distance(
        optimized_verts, faces, source, t_factor=t_factor
    )

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
        },
        output_dir / "optimized_heightfield.pt",
    )
    print(f"Result saved to: {(output_dir / 'optimized_heightfield.pt').resolve()}")

    if "--visualize" in sys.argv:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps_mesh = ps.register_surface_mesh(
            "optimized heightfield",
            optimized_vertices.cpu().numpy(),
            faces.cpu().numpy(),
        )
        ps_mesh.add_scalar_quantity(
            "heat-method distance",
            final_distance.detach().cpu().numpy(),
            defined_on="vertices",
            isolines_enabled=True,
            enabled=True,
        )
        ps.register_point_cloud(
            "source",
            optimized_vertices[source].cpu().numpy(),
            enabled=True,
            radius=0.005,
        )
        ps.show()

if __name__ == "__main__":
    main()
