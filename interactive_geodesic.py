"""Geodesic distance demo (Polyscope).

Click the mesh to set a source; the surface gets coloured by geodesic distance
with isoline contours. Panel has t_factor, the Step-3 solver (min_quad vs
linear_solve+shift), a multi-source toggle, isoline count, and an OBJ loader.

The mesh is fixed between loads, so the Laplacian / gradient / divergence are
built once and the factorizations are cached -- only new right-hand sides get
solved per click, which keeps it snappy.

    python interactive_geodesic.py [mesh.obj]
"""
import glob
import os
import sys

import numpy as np
import torch

import iskra.sparse as sp
import iskra.sparse_linalg as sparse_linalg
from iskra.dec import laplacian
from iskra.fem import grad
from iskra.geometry.volume import triangle_areas
from iskra.sparse_linalg import default_solver, min_quadratic_energy
from iskra.topology import face_index

sparse_linalg._cholmod_available = False   # use cholespy, not scikit-sparse

MESH_NAME = "surface"
MESH_DIR = os.path.join(os.path.dirname(__file__), "assets", "meshes")
DEFAULT_MESH = os.path.join(MESH_DIR, "bunny.obj")


def load_obj(path):
    import trimesh
    tm = trimesh.load(path, process=True, force="mesh")
    tm.remove_unreferenced_vertices()
    tm = max(tm.split(only_watertight=False), key=lambda c: len(c.faces))
    tm.remove_unreferenced_vertices()
    V = torch.tensor(np.asarray(tm.vertices), dtype=torch.double)
    F = torch.tensor(np.asarray(tm.faces), dtype=torch.long)
    return V, F


class App:
    def __init__(self, path):
        self.sources = [0]
        self.t_factor = 10.0
        self.solver = 0            # 0 min_quad, 1 linear
        self.multi = False
        self.n_isolines = 18
        self.picked_vertex = 0
        self.phi = None
        self._last_pick = None
        self.status = ""
        self.path_buf = path
        self.available = sorted(glob.glob(os.path.join(MESH_DIR, "*.obj")))
        self.combo_idx = 0
        self.load(path)

    def load(self, path):
        self.V, self.F = load_obj(path)
        self.Vnp = self.V.numpy()
        self.n = self.V.shape[0]
        # everything here only depends on the mesh, so build it once
        with torch.no_grad():
            self.lap, self.mass = laplacian(self.V, self.F)
            self.G = grad(self.V, self.F, stack=True)
            tri = face_index(self.V, self.F)
            self.h = torch.linalg.vector_norm(tri - tri[:, [1, 2, 0], :], dim=-1).mean()
            areas = triangle_areas(tri)
            self.div = sp.mul(torch.cat(3 * [areas])[None, :], self.G.mT.coalesce())
            self.linear_solver = default_solver(self.lap + 1e-8 * self.mass)
        self._step1_solver = None
        self._step1_t = None
        self.mesh_path = path

    def reload(self, ps, path):
        try:
            self.load(path)
        except Exception as e:
            self.status = f"load failed: {e}"
            return
        self.sources = [0]
        self.picked_vertex = 0
        self._last_pick = None
        ps.register_surface_mesh(MESH_NAME, self.Vnp, self.F.numpy())
        self.update_field(ps)
        ps.reset_camera_to_home_view()
        self.status = f"loaded {os.path.basename(path)}  ({self.n} verts)"

    def _step1(self, t_factor):
        # (M + tL) only changes when t changes, so cache its factorization
        if self._step1_solver is None or t_factor != self._step1_t:
            t = t_factor * self.h * self.h
            self._step1_solver = default_solver(self.mass + t * self.lap)
            self._step1_t = t_factor
        return self._step1_solver

    def solve(self):
        src = torch.as_tensor(self.sources, dtype=torch.long).reshape(-1)
        with torch.no_grad():
            delta = self.V.new_zeros(self.n)
            delta[src] = 1.0
            u = self._step1(self.t_factor)(delta)

            g = sp.matmul(self.G, u).reshape(3, -1)
            X = -g / torch.linalg.vector_norm(g, dim=0, keepdim=True).clamp_min(1e-12)
            rhs = sp.matmul(self.div, X.flatten())

            if self.solver == 0:
                phi = min_quadratic_energy(self.lap, rhs, src, rhs.new_zeros(src.numel()))[1]
            else:
                phi = self.linear_solver(rhs)
                phi = phi - phi[src].mean()
        self.phi = phi.numpy()

    def nearest_vertex(self, position):
        p = np.asarray(position, dtype=float)
        return int(((self.Vnp - p) ** 2).sum(axis=1).argmin())

    def update_field(self, ps):
        self.solve()
        mesh = ps.get_surface_mesh(MESH_NAME)
        self.q = mesh.add_scalar_quantity("geodesic distance", self.phi, defined_on="vertices",
                                          cmap="turbo", isolines_enabled=True, enabled=True)
        self._set_isolines(self.q)
        ps.register_point_cloud("sources", self.Vnp[self.sources], enabled=True, radius=0.01)

    def _set_isolines(self, q):
        span = float(self.phi.max() - self.phi.min()) or 1.0
        try:
            q.set_isoline_width(span / max(self.n_isolines, 1), relative=False)
        except Exception:
            pass

    def update_isolines_only(self, ps):
        self._set_isolines(self.q)


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
                app.picked_vertex = v
                app.sources = (app.sources + [v]) if (app.multi and v not in app.sources) else [v]
                app.update_field(ps)

        psim.Text(f"mesh: {os.path.basename(app.mesh_path)}   ({app.n} verts)")
        psim.Text(f"sources: {app.sources}")
        if app.phi is not None:
            psim.Text(f"picked vertex {app.picked_vertex}: distance = {app.phi[app.picked_vertex]:.4f}")
            psim.Text(f"max distance = {app.phi.max():.4f}   |   negatives = {int((app.phi < -1e-6).sum())}")

        _, app.t_factor = psim.SliderFloat("t_factor", app.t_factor, 1.0, 40.0)
        if psim.IsItemDeactivatedAfterEdit():          # only re-solve when the drag ends
            app.update_field(ps)

        ch, app.solver = psim.Combo("Step-3 solver", app.solver, ["min_quad", "linear_solve+shift"])
        if ch:
            app.update_field(ps)

        _, app.multi = psim.Checkbox("multi-source (add instead of replace)", app.multi)

        ch, app.n_isolines = psim.SliderInt("# isolines", app.n_isolines, 4, 60)
        if ch:
            app.update_isolines_only(ps)               # just a display change

        if psim.Button("clear sources"):
            app.sources = [app.picked_vertex]
            app.update_field(ps)

        psim.Separator()
        psim.Text("load OBJ:")
        if app.available:
            ch, app.combo_idx = psim.Combo("assets/meshes", app.combo_idx,
                                           [os.path.basename(p) for p in app.available])
            psim.SameLine()
            if psim.Button("Load selected"):
                app.reload(ps, app.available[app.combo_idx])
        _, app.path_buf = psim.InputText("path", app.path_buf)
        psim.SameLine()
        if psim.Button("Load path"):
            app.reload(ps, app.path_buf)
        if app.status:
            psim.Text(app.status)

    return callback


def main():
    import polyscope as ps
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MESH
    app = App(path)
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.register_surface_mesh(MESH_NAME, app.Vnp, app.F.numpy())
    app.update_field(ps)
    ps.set_user_callback(make_callback(app))
    ps.show()


if __name__ == "__main__":
    main()
