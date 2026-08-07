# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.
"""Geodesic distance via the heat method (Crane, Weischedel & Wardetzky 2013)."""

import torch

import iskra.sparse as sp
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry.volume import triangle_areas
from iskra.sparse_linalg import linear_solve, min_quadratic_energy
from iskra.topology import face_index


def heat_method_distance(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source: int | list[int] | torch.Tensor,
    t_factor: float = 1.0,
) -> torch.Tensor:
    """Geodesic distance from `source` to every vertex.

    Three steps: diffuse heat from the source, take the unit direction of the
    heat gradient, then solve a Poisson equation for the distance. `t_factor`
    scales the diffusion time (t = t_factor * mean_edge_length**2); irregular
    meshes like the bunny want ~10, clean grids ~1. Returns a `[V]` tensor.
    """
    n = vertices.shape[0]
    source = torch.as_tensor(source, dtype=torch.long, device=vertices.device).reshape(-1)

    lap, mass = laplacian(vertices, faces)
    tri = face_index(vertices, faces)
    h = torch.linalg.vector_norm(tri - tri[:, [1, 2, 0], :], dim=-1).mean()
    time = t_factor * h.square()

    # 1. diffuse a Dirac heat load from the source
    delta = vertices.new_zeros(n)
    delta[source] = 1.0
    u = linear_solve(lap * time + mass, delta)[1]

    # 2. unit direction of the (negative) heat gradient
    G = grad(vertices, faces, stack=True)
    g = sp.matmul(G, u).reshape(3, -1)
    X = -g / torch.linalg.vector_norm(g, dim=0, keepdim=True).clamp_min(1e-12)

    # 3. Poisson solve L phi = div(X), pinning phi[source] = 0 (works for many sources)
    areas = triangle_areas(tri)
    div = sp.mul(torch.cat(3 * [areas])[None, :], G.mT.coalesce())
    rhs = sp.matmul(div, X.flatten())
    return min_quadratic_energy(lap, rhs, source, rhs.new_zeros(source.numel()))[1]
