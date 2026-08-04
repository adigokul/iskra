from pathlib import Path

import igl
import torch

import iskra.sparse as sp
import iskra.sparse_linalg as sparse_linalg
from iskra.dec import laplacian
from iskra.mesh import Mesh
from iskra.profiling import profile_block
from iskra.sparse_linalg import default_solver, linear_solve

# scikit-sparse 0.5 changed its CHOLMOD API, while this Iskra revision expects
# the older API. Use Iskra's installed cholespy backend for this runner.
sparse_linalg._cholmod_available = False


mesh_path = "/Users/huyufan/Downloads/cube386.off"
t = 0.001
alpha = 0.01
learning_rate = 1.0

device = "cpu"

mesh, _ = Mesh.from_path(mesh_path, device=device)
mesh.geom.normalize()

faces = mesh.topo.faces
verts = mesh.geom.vertices

# The optimization variable v. The original vertices remain the fixed target.
verts_var = torch.nn.Parameter(verts.clone())
optim = torch.optim.SGD([verts_var], lr=learning_rate)

# Build and factorize the fixed H1 preconditioner on the reference mesh.
lap, mass = laplacian(verts, faces)
h1_solver = default_solver(mass + alpha * lap)

for i in range(10):
    optim.zero_grad()

    with profile_block("forward"):
        # Rebuild geometry-dependent operators on the current candidate mesh.
        lap, mass = laplacian(verts_var, faces)

        # g(v): one backward-Euler smoothing step.
        faired = linear_solve(
            mass + t * lap,
            sp.matmul(mass, verts_var),
        )[1]

        # Area-weighted squared distance from the fixed target mesh.
        diff = faired - verts
        loss = (
            sp.matmul(
                diff.mT,
                sp.matmul(mass, diff),
            )
            .diagonal()
            .sum()
        )

    with profile_block("backward"):
        loss.backward()

    if verts_var.grad is None:
        raise RuntimeError("verts_var.grad is None!")

    with profile_block("h1"):
        # Replace the raw loss gradient with an H1-smoothed direction.
        verts_var.grad = h1_solver(mass @ verts_var.grad)

    optim.step()
    print(f"iteration={i:02d}, loss={loss.item():.8f}")

# Evaluate the optimized mesh with the reference-mesh operators, matching the
# original Iskra example supplied by the instructor.
lap, mass = laplacian(verts, faces)
faired = linear_solve(mass + t * lap, mass @ verts_var)[1]

output_dir = Path("results/inflate_core")
output_dir.mkdir(parents=True, exist_ok=True)

igl.write_triangle_mesh(
    str(output_dir / "target.obj"),
    verts.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
)
igl.write_triangle_mesh(
    str(output_dir / "inflated.obj"),
    verts_var.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
)
igl.write_triangle_mesh(
    str(output_dir / "faired.obj"),
    faired.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
)

print(f"Results saved to: {output_dir.resolve()}")
