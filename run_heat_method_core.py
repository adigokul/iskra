from pathlib import Path
import sys

import torch

import iskra.sparse as sp
import iskra.sparse_linalg as sparse_linalg
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry import triangle_areas
from iskra.mesh import Mesh
from iskra.sparse_linalg import linear_solve
from iskra.topology import face_index

# scikit-sparse 0.5 is incompatible with the CHOLMOD call used by this Iskra
# revision. Use the installed cholespy backend in this standalone runner.
sparse_linalg._cholmod_available = False


def heat_method_distance(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source: int | list[int] | torch.Tensor,
    t_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Approximate distance to source vertices using the three-step heat method."""
    n_vertices = vertices.shape[0]
    source = torch.as_tensor(source, dtype=torch.long, device=vertices.device).reshape(-1)

    # Cotangent stiffness matrix L and lumped mass matrix M.
    lap, mass = laplacian(vertices, faces)

    # The heat-method time scale t = t_factor * h^2.
    triangles = face_index(vertices, faces)
    edges = triangles - triangles[:, [1, 2, 0], :]
    mean_edge_length = torch.linalg.vector_norm(edges, dim=-1).mean()
    time = t_factor * mean_edge_length.square()

    # Step 1: diffuse a discrete Dirac load from the source vertices.
    delta = vertices.new_zeros(n_vertices)
    delta[source] = 1.0
    temperature = linear_solve(lap * time + mass, delta)[1]

    # Step 2: keep only the direction of the negative heat gradient.
    gradient = grad(vertices, faces, stack=True)
    grad_temperature = sp.matmul(gradient, temperature).reshape(3, -1)
    direction = -grad_temperature / torch.linalg.vector_norm(
        grad_temperature, dim=0, keepdim=True
    ).clamp_min(1e-12)

    # Step 3: recover a scalar potential by solving the Poisson equation.
    areas = triangle_areas(triangles)
    divergence = sp.mul(
        torch.cat(3 * [areas])[None, :], gradient.mT.coalesce()
    )
    rhs = sp.matmul(divergence, direction.flatten())
    distance = linear_solve(lap + 1e-8 * mass, rhs)[1]
    distance = distance - distance[source].mean()

    return distance, temperature, time.item()


mesh_path = Path(__file__).resolve().parent / "assets" / "meshes" / "stanford-bunny.obj"
mesh, _ = Mesh.from_path(mesh_path, dtype=torch.float64, device="cpu")
mesh.geom.normalize()

faces = mesh.topo.faces
vertices = mesh.geom.vertices

# Some OBJ files contain vertices that are not referenced by any triangle.
# Remove and reindex them so all discrete operators have matching dimensions.
used_vertices, inverse = torch.unique(
    faces.flatten(), sorted=True, return_inverse=True
)
vertices = vertices[used_vertices]
faces = inverse.reshape_as(faces)
# Choose a reproducible random source vertex.
torch.manual_seed(42)
source = torch.randint(
    0,
    vertices.shape[0],
    (1,),
    device=vertices.device,
)

distance, temperature, time = heat_method_distance(vertices, faces, source, t_factor = 10.0)

print(f"mesh: {mesh_path}")
print(f"vertices={vertices.shape[0]}, faces={faces.shape[0]}, source={source.tolist()}")
print(f"t={time:.8e}")
print(f"temperature: min={temperature.min().item():.8e}, max={temperature.max().item():.8e}")
print(f"distance: min={distance.min().item():.8e}, max={distance.max().item():.8e}")
print(f"distance at source={distance[source].tolist()}")
print(f"finite values={torch.isfinite(distance).all().item()}")

if "--visualize" in sys.argv:
    # Close the Polyscope window to finish the program.
    import polyscope as ps

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps_mesh = ps.register_surface_mesh(
        "mesh",
        vertices.detach().cpu().numpy(),
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
        vertices[source].detach().cpu().numpy(),
        enabled=True,
        radius=0.02,
    )
    ps.show()
