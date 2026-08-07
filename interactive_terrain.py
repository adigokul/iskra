"""Terrain optimization demo (Polyscope).

Bend a 20x20 heightfield so the geodesic distance from a source matches a target:
either the built-in diagonal made equidistant, or a set of points I click and give
one constant distance. Run with `python interactive_terrain.py`.
"""
import numpy as np
import torch

import iskra.sparse_linalg as sparse_linalg
from iskra.geometry.geodesics import heat_method_distance

sparse_linalg._cholmod_available = False   # use cholespy, not scikit-sparse

NX = NZ = 20
MESH_NAME = "terrain"


def make_grid(nx, nz):
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.double)
    z = torch.linspace(0.0, 1.0, nz, dtype=torch.double)
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    xz = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)
    faces = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a, b, c, d = i * nz + j, i * nz + j + 1, (i + 1) * nz + j, (i + 1) * nz + j + 1
            faces.append([a, b, c])
            faces.append([c, b, d])
    return xz, torch.tensor(faces, dtype=torch.long)


def target_diagonal(nx, nz, k):
    return torch.tensor([i * nz + j for i in range(nx) for j in range(nz) if i + j == k], dtype=torch.long)


def adjacent_face_pairs(F):
    edge2faces = {}
    for fi, (a, b, c) in enumerate(F.tolist()):
        for e in ((a, b), (b, c), (c, a)):
            edge2faces.setdefault(tuple(sorted(e)), []).append(fi)
    return torch.tensor([fs for fs in edge2faces.values() if len(fs) == 2], dtype=torch.long)


def edge_list(F):
    edges = set()
    for a, b, c in F.tolist():
        for e in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted(e)))
    return torch.tensor(sorted(edges), dtype=torch.long)


def boundary_vertices(F):
    # a boundary edge belongs to only one triangle; its endpoints are the rim
    count = {}
    for a, b, c in F.tolist():
        for e in ((a, b), (b, c), (c, a)):
            k = tuple(sorted(e))
            count[k] = count.get(k, 0) + 1
    rim = set()
    for (u, v), c in count.items():
        if c == 1:
            rim.update((u, v))
    return torch.tensor(sorted(rim), dtype=torch.long)


def normal_smoothness_loss(vertices, faces, pairs):
    tri = vertices[faces]
    n = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp_min(1e-12)
    return (n[pairs[:, 0]] - n[pairs[:, 1]]).square().sum(dim=1).mean()


def height_smoothness_loss(heights, edges):
    return (heights[edges[:, 0]] - heights[edges[:, 1]]).square().mean()


class App:
    def __init__(self):
        self.y_init = 1            # 0 flat, 1 bump
        self.optimizer_kind = 0    # 0 Adam, 1 SGD
        self.lr = 1e-2
        self.smooth_kind = 0       # 0 normal, 1 height
        self.smooth_weight = 1e-4
        self.pin_boundary = True
        self.t_factor = 10.0
        self.target_k = (NX + NZ - 2) // 2
        self.running = False
        self.iter = 0
        self.loss = float("nan")
        self.std = float("nan")

        self.manual_targets = []       # vertices I clicked
        self.const_distance = 0.5      # distance I want them all at
        self.use_manual = False
        self.click_mode = 0            # 0 target, 1 source
        self._last_pick = None

        self.xz, self.F = make_grid(NX, NZ)
        self.source = torch.tensor([0])
        self.pairs = adjacent_face_pairs(self.F)
        self.edges = edge_list(self.F)
        self.boundary = boundary_vertices(self.F)
        self.diag = target_diagonal(NX, NZ, self.target_k)
        self.reset()

    def _init_heights(self):
        if self.y_init == 0:
            h = torch.zeros(self.xz.shape[0], dtype=torch.double)   # flat = stuck
        else:
            u = torch.linspace(0, np.pi, NX, dtype=torch.double)
            v = torch.linspace(0, np.pi, NZ, dtype=torch.double)
            h = 1e-3 * (torch.sin(u)[:, None] * torch.sin(v)[None, :]).reshape(-1)
        h[self.source] = 0.0
        return h

    def reset(self):
        self.heights = torch.nn.Parameter(self._init_heights())
        Opt = torch.optim.Adam if self.optimizer_kind == 0 else torch.optim.SGD
        self.opt = Opt([self.heights], lr=self.lr)
        self.iter = 0
        self.diag = target_diagonal(NX, NZ, self.target_k)

    def vertices3d(self, detached=True):
        h = self.heights.detach() if detached else self.heights
        return torch.stack([self.xz[:, 0], h, self.xz[:, 1]], dim=1)

    def current_distance(self):
        with torch.no_grad():
            return heat_method_distance(self.vertices3d(), self.F, self.source, t_factor=self.t_factor)

    def nearest_vertex(self, position):
        p = np.asarray(position, dtype=float)
        return int(((self.vertices3d().numpy() - p) ** 2).sum(axis=1).argmin())

    def step(self):
        self.opt.zero_grad()
        V = self.vertices3d(detached=False)
        phi = heat_method_distance(V, self.F, self.source, t_factor=self.t_factor)

        if self.use_manual and self.manual_targets:
            d = phi[torch.tensor(self.manual_targets)]
            data_loss = (d - self.const_distance).square().mean()
            self.std = (d.detach() - self.const_distance).abs().mean().item()
        else:
            d = phi[self.diag]
            data_loss = (d - d.mean()).square().mean()
            self.std = d.detach().std(correction=0).item()

        if self.smooth_kind == 0:
            smooth = normal_smoothness_loss(V, self.F, self.pairs)
        else:
            smooth = height_smoothness_loss(self.heights, self.edges)
        loss = data_loss + self.smooth_weight * smooth
        loss.backward()

        pin = self.pin_boundary and len(self.boundary) > 0
        if pin and self.heights.grad is not None:
            self.heights.grad[self.boundary] = 0.0     # keep the rim from blowing up
        self.opt.step()
        if not pin:
            with torch.no_grad():
                self.heights.sub_(self.heights[self.source].item())   # anchor the source height
        self.iter += 1
        self.loss = loss.item()
        return phi.detach()

    def register(self, ps):
        ps.register_surface_mesh(MESH_NAME, self.vertices3d().numpy(), self.F.numpy())
        self.update_scalar(ps, self.current_distance().numpy())
        self.draw_markers(ps)

    def draw_markers(self, ps):
        V = self.vertices3d().numpy()
        ps.register_point_cloud("source", V[self.source.numpy()], radius=0.01, enabled=True)
        pts = V[self.diag.numpy()]
        e = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
        ps.register_curve_network("target", pts, e, radius=0.004).set_color((1.0, 0.0, 0.0))
        if self.manual_targets:
            pcm = ps.register_point_cloud("picked targets", V[np.array(self.manual_targets)],
                                          radius=0.014, enabled=True)
            pcm.set_color((0.1, 0.1, 1.0))
        else:
            ps.remove_point_cloud("picked targets", error_if_absent=False)

    def update_scalar(self, ps, phi):
        q = ps.get_surface_mesh(MESH_NAME).add_scalar_quantity(
            "geodesic distance", phi, defined_on="vertices",
            cmap="turbo", isolines_enabled=True, enabled=True)
        span = float(phi.max() - phi.min()) or 1.0
        try:
            q.set_isoline_width(span / 18.0, relative=False)
        except Exception:
            pass

    def refresh_geometry(self, ps):
        ps.get_surface_mesh(MESH_NAME).update_vertex_positions(self.vertices3d().numpy())
        self.draw_markers(ps)


def make_callback(app):
    import polyscope as ps
    import polyscope.imgui as psim

    def callback():
        sel = ps.get_selection()
        if sel.is_hit and sel.structure_name == MESH_NAME:
            key = (sel.structure_name, int(sel.local_index))
            if key != app._last_pick:
                app._last_pick = key
                v = app.nearest_vertex(sel.position)
                if app.click_mode == 0:
                    if not app.manual_targets:
                        app.const_distance = round(float(app.current_distance()[v]), 4)
                    if v not in app.manual_targets:
                        app.manual_targets.append(v)
                    app.use_manual = True
                else:
                    app.source = torch.tensor([v])
                    app.reset()
                app.refresh_geometry(ps)
                app.update_scalar(ps, app.current_distance().numpy())

        metric = "target err" if (app.use_manual and app.manual_targets) else "contour std"
        psim.Text(f"grid {NX}x{NZ}   verts {app.xz.shape[0]}")
        psim.Text(f"iter {app.iter}   loss {app.loss:.3e}   {metric} {app.std:.3e}")

        _, app.click_mode = psim.Combo("click places", app.click_mode, ["target point", "source"])
        psim.Text(f"source: vertex {int(app.source.item())}")

        _, app.use_manual = psim.Checkbox("match picked targets (else auto contour)", app.use_manual)
        _, app.const_distance = psim.InputFloat("constant distance (all targets)", app.const_distance)
        if not app.manual_targets:
            psim.Text("  (set mode to 'target point' and click the surface)")
        else:
            psim.Text(f"  {len(app.manual_targets)} target(s) -> distance {app.const_distance:.3f}")
            if psim.Button("remove last"):
                app.manual_targets.pop()
                app.refresh_geometry(ps)
            psim.SameLine()
            if psim.Button("clear targets"):
                app.manual_targets = []
                app.use_manual = False
                app.refresh_geometry(ps)

        psim.Separator()
        if psim.Button("Run" if not app.running else "Pause"):
            app.running = not app.running
        psim.SameLine()
        step_once = psim.Button("Step")
        psim.SameLine()
        if psim.Button("Reset"):
            app.reset(); app.refresh_geometry(ps); app.update_scalar(ps, app.current_distance().numpy())

        psim.Separator()
        psim.Text("options:")
        ch, app.y_init = psim.Combo("y init", app.y_init, ["flat (stuck)", "bump"])
        if ch:
            app.reset(); app.refresh_geometry(ps); app.update_scalar(ps, app.current_distance().numpy())
        ch, app.optimizer_kind = psim.Combo("optimizer", app.optimizer_kind, ["Adam", "SGD"])
        if ch:
            app.reset()
        _, app.lr = psim.SliderFloat("learning rate", app.lr, 1e-4, 1e-1, format="%.4f")
        if psim.IsItemDeactivatedAfterEdit():
            for g in app.opt.param_groups:
                g["lr"] = app.lr
        _, app.smooth_kind = psim.Combo("smoothness", app.smooth_kind, ["normal", "height"])
        logf = psim.ImGuiSliderFlags_Logarithmic if hasattr(psim, "ImGuiSliderFlags_Logarithmic") else 0
        _, app.smooth_weight = psim.SliderFloat("smoothness weight", app.smooth_weight, 1e-5, 1e-1,
                                                format="%.5f", flags=logf)
        _, app.pin_boundary = psim.Checkbox(f"pin boundary ({len(app.boundary)} verts)", app.pin_boundary)

        if app.running or step_once:
            phi = app.step()
            app.refresh_geometry(ps)
            app.update_scalar(ps, phi.numpy())

    return callback


def main():
    import polyscope as ps
    app = App()
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    app.register(ps)
    ps.set_user_callback(make_callback(app))
    ps.show()


if __name__ == "__main__":
    main()
