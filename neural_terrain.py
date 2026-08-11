# Neural terrain: the heightfield is a small MLP f(x,z) -> height, trained through
# the same differentiable heat method. So instead of a height per vertex, the
# optimization variable is the MLP weights. The mesh resolution is separate from
# the MLP, so we can train on a cheap coarse mesh and evaluate on a fine one.
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
        return self.net(2.0 * xz - 1.0).squeeze(-1) * self.scale


def heights_to_verts(xz, h):
    return torch.stack([xz[:, 0], h, xz[:, 1]], dim=1)


def train(mlp, n, k, iters, lr=1e-2, weight=8e-3):
    # train the MLP on an n x n mesh so the i+j=k diagonal becomes one distance contour
    xz, F = make_grid(n)
    pairs = adjacent_face_pairs(F)
    src = torch.tensor([0, n * n - 1])  # two opposite corners
    diag = diagonal(n, k)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        V = heights_to_verts(xz, mlp(xz))
        phi = heat_method_distance(V, F, src, t_factor=10.0)
        d = phi[diag]
        loss = (d - d.mean()).square().mean() + weight * normal_smoothness_loss(
            V, F, pairs
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        V = heights_to_verts(xz, mlp(xz))
        std = heat_method_distance(V, F, src, 10.0)[diag].std(correction=0).item()
    print(f"  {n}x{n}: {iters} iters, diagonal std={std:.2e}", flush=True)
    return xz, F, src, diag


def sample(mlp, n, k):
    # evaluate the trained MLP at any resolution (no training) -> continuous surface
    xz, F = make_grid(n)
    src = torch.tensor([0, n * n - 1])
    with torch.no_grad():
        V = heights_to_verts(xz, mlp(xz))
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
    ps.screenshot(out)
    print(f"  wrote {out}", flush=True)


def main():
    torch.manual_seed(0)
    mlp = MLP()

    print("train the MLP on a cheap 32x32 mesh:")
    train(mlp, 32, k=8, iters=400)

    print("evaluate the SAME weights at 128x128 (no retraining):")
    xz, F, V, phi, src, diag = sample(mlp, 128, k=32)
    render(V, F, phi, src, diag, "results/geodesics/neural_terrain.png")

    print("analyze the trained MLP with Fourier analysis:")
    analyze_height_spectrum(
        mlp,
        n=128,
        cutoff=0.25,
        out="results/geodesics/fourier_analysis.png",
        show=True,
    )

    print("evaluate the SAME weights at 128x128:")
    xz, F, V, phi, src, diag = sample(mlp, 128, k=33)


if __name__ == "__main__":
    main()
