"""Human-only MuJoCo scene display for viser. All MuJoCo reads happen on the session thread."""
import math
import time

import mujoco
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from homebody.ui.theme import (
    DOT_3D,
    HEADING,
    LIGHTING,
    LINE_3D,
    MASK,
    MATTE_ROUGHNESS,
    PLAN,
    ROUTE,
    SCENE_BACKGROUND,
    STANCE,
)

SCENE_RATE = 30.  # 3D view updates per wall second

def visual_geometries(model):
    visible = [i for i in range(model.ngeom)
               if model.geom_group[i] < 3 and model.geom_rgba[i, 3] > 0]
    visual_bodies = {int(model.geom_bodyid[i]) for i in visible
                     if model.geom_bodyid[i] != 0 and
                     model.geom_contype[i] == model.geom_conaffinity[i] == 0}
    return [i for i in visible if int(model.geom_bodyid[i]) not in visual_bodies or
            model.geom_contype[i] == model.geom_conaffinity[i] == 0]


def material(model, geom, textures):
    index = int(model.geom_matid[geom])
    rgba = model.geom_rgba[geom]
    image = None
    roughness, metallic = MATTE_ROUGHNESS, 0.
    if index >= 0:
        if model.mat_roughness[index] >= 0:
            roughness = float(model.mat_roughness[index])
        if model.mat_metallic[index] >= 0:
            metallic = float(model.mat_metallic[index])
        if np.allclose(rgba, [.5, .5, .5, 1]):
            rgba = model.mat_rgba[index]
        tex = int(model.mat_texid[index, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        if tex >= 0 and model.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
            if tex not in textures:
                h, w, c = map(int, (model.tex_height[tex], model.tex_width[tex], model.tex_nchannel[tex]))
                start = int(model.tex_adr[tex])
                pixels = model.tex_data[start:start + h * w * c].reshape(h, w, c)
                textures[tex] = Image.fromarray(pixels[::-1].copy())
            image = textures[tex]
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.rint(np.asarray(rgba) * 255).astype(np.uint8),
        baseColorTexture=image, metallicFactor=metallic, roughnessFactor=roughness,
        doubleSided=True, alphaMode="BLEND" if rgba[3] < 1 else "OPAQUE")


def geom_mesh(model, geom, textures):
    kind, size = model.geom_type[geom], model.geom_size[geom]
    uv = None
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        faces = model.mesh_face[start:start + count]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh] + faces].reshape(-1, 3)
        normals = model.mesh_normal[model.mesh_normaladr[mesh] +
                                    model.mesh_facenormal[start:start + count]].reshape(-1, 3)
        result = trimesh.Trimesh(vertices=vertices, faces=np.arange(count * 3).reshape(-1, 3),
                                 vertex_normals=normals, process=False)
        tex_start = int(model.mesh_texcoordadr[mesh])
        if tex_start >= 0:
            uv = model.mesh_texcoord[tex_start + model.mesh_facetexcoord[start:start + count]].reshape(-1, 2)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        result = trimesh.creation.box(extents=2 * size)
    elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        result = trimesh.creation.icosphere(subdivisions=2)
        result.apply_scale(size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        result = trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        result = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    else:
        raise ValueError(f"Unsupported visible MuJoCo geometry: {kind}")
    result.visual = trimesh.visual.TextureVisuals(uv=uv, material=material(model, geom, textures))
    return result


def body_meshes(model):
    """Body-local trimesh scenes by body id, body zero holding the static room."""
    bodies, textures = {}, {}
    for geom in visual_geometries(model):
        body = int(model.geom_bodyid[geom])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(model.geom_quat[geom], scalar_first=True).as_matrix()
        pose[:3, 3] = model.geom_pos[geom]
        bodies.setdefault(body, trimesh.Scene()).add_geometry(
            geom_mesh(model, geom, textures), node_name=f"geom-{geom}",
            geom_name=f"geom-{geom}", transform=pose)
    return bodies


class SceneView:
    def __init__(self, server, backend, rate=SCENE_RATE):
        """RATE caps how often per wall second update() redraws the bodies."""
        self.server, self.backend, self.rate = server, backend, rate
        self.model = None
        self.handles = {}
        self._selection_handles = []
        self._route_handles = []
        self._arm_handles = []
        self.home = None
        self.last = None
        self.shown = -math.inf
        server.scene.set_up_direction("+z")
        server.scene.configure_environment_map(None)
        server.scene.set_background_image(np.full((2, 2, 3), SCENE_BACKGROUND, dtype=np.uint8), format="png")
        server.scene.configure_default_lights(False, cast_shadow=False)
        server.scene.add_light_ambient("/lighting/ambient", **LIGHTING["ambient"])
        server.scene.add_light_directional("/lighting/key", **LIGHTING["key"])
        server.scene.add_light_directional("/lighting/fill", **LIGHTING["fill"])
        server.on_client_connect(self.place_camera)
        self.update()

    def place_camera(self, client):
        if self.home is not None:
            position, target = self.home
            with client.atomic():
                client.camera.up_direction = (0., 0., 1.)
                client.camera.position = position
                client.camera.look_at = target

    def reset_camera(self):
        for client in self.server.get_clients().values():
            self.place_camera(client)

    def _dot(self, name, position, colour, radius=DOT_3D):
        return self.server.scene.add_icosphere(name, radius=radius, color=colour, position=position,
                                               cast_shadow=False, receive_shadow=False)

    def _path(self, name, points, colour):
        return self.server.scene.add_line_segments(
            name, points=np.stack((points[:-1], points[1:]), axis=1), colors=colour, thickness=LINE_3D)

    @staticmethod
    def _drop(handles):
        for handle in handles:
            handle.remove()
        handles.clear()

    def clear_selection(self):
        for handles in (self._selection_handles, self._route_handles, self._arm_handles):
            self._drop(handles)

    def show_selection(self, point):
        self.clear_selection()
        point = np.asarray(point, dtype=float)
        if point.shape == (3,) and np.isfinite(point).all():
            self._selection_handles = [self._dot("/guides/selection", point, MASK)]

    def show_arm_path(self, points):
        """Display the planned palm path with a few waypoints."""
        self._drop(self._arm_handles)
        points = np.asarray(points, dtype=float)
        if points.ndim == 2 and points.shape[1] == 3 and len(points) > 1 and np.isfinite(points).all():
            self._arm_handles = [self._path("/guides/arm-path", points, PLAN),
                                 self._dot("/guides/arm-goal", points[-1], PLAN)]
            interior = np.unique(np.linspace(1, len(points) - 2, min(len(points) - 2, 5), dtype=int))
            self._arm_handles.extend(self._dot(f"/guides/arm-wp-{i}", points[i], PLAN)
                                     for i in interior)

    def clear_arm_path(self):
        self._drop(self._arm_handles)

    def show_route(self, waypoints, yaw=None):
        """Display a base route just above the floor, ending at the stance and, given `yaw`, its heading."""
        self._drop(self._route_handles)
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            return
        model, data = self.backend.model, self.backend.data
        floors = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        if len(floors) != 1:
            return
        floor = int(floors[0])
        origin = data.geom_xpos[floor]
        normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        if not np.isfinite(origin).all() or not np.isfinite(normal).all() or abs(normal[2]) < 1e-6:
            return

        def raised(xy):
            return np.column_stack((xy, origin[2] + (.04 - (xy - origin[:2]) @ normal[:2]) / normal[2]))

        world = raised(points)
        if len(world) > 1:
            self._route_handles.append(self._path("/guides/route", world, ROUTE))
            self._route_handles.extend(self._dot(f"/guides/route-wp-{i}", point, ROUTE, .035)
                                       for i, point in enumerate(world[1:], 1))
        self._route_handles.append(self._dot("/guides/stance", world[-1], STANCE))
        if yaw is not None and np.isfinite(yaw):
            tip = points[-1] + HEADING * np.array([np.cos(yaw), np.sin(yaw)])
            self._route_handles.append(self._path("/guides/stance-heading", raised(np.array([points[-1], tip])),
                                                  STANCE))

    def update(self):
        model, data = self.backend.model, self.backend.data
        stamp = self.backend.epoch, float(data.time)
        if model is not self.model:
            self.model = model
            self.clear_selection()
            for handle in self.handles.values():
                handle.remove()
            self.handles.clear()
            meshes = body_meshes(model)
            for body, scene in meshes.items():
                self.handles[body] = self.server.scene.add_glb(
                    f"/world/body-{body}", scene.export(file_type="glb"),
                    visible=False, cast_shadow=body != 0, receive_shadow=True)
            bounds = meshes[0].bounds
            center = np.mean(bounds, axis=0)
            span = max(float(np.ptp(bounds[:, :2], axis=0).max()), 1.)
            target = (center[0], center[1], bounds[0, 2])
            self.home = ((center[0], center[1] - .9 * span, bounds[0, 2] + .8 * span), target)
            self.reset_camera()
            self.last, self.shown = None, -math.inf
        if self.last == stamp or time.monotonic() - self.shown < 1 / self.rate:
            return
        self.shown = time.monotonic()
        with self.server.atomic():
            for body, handle in self.handles.items():
                if not (np.isfinite(data.xpos[body]).all() and
                        np.isfinite(data.xquat[body]).all()):
                    handle.visible = False
                    continue
                handle.position = data.xpos[body].copy()
                handle.wxyz = data.xquat[body].copy()
                handle.visible = True
        self.last = stamp
