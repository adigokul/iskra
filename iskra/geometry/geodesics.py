# heat method for geodesic distance (Crane et al. 2013)
import torch

import iskra.sparse as sp
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry.volume import triangle_areas
from iskra.sparse_linalg import linear_solve, min_quadratic_energy
from iskra.topology import face_index


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
