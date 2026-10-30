"""Human-only REAL scene, drawn exclusively from immutable observation packets."""
import json
import math
import time
from pathlib import Path
from threading import RLock

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
    PLAN,
    ROUTE,
    SCENE_BACKGROUND,
    STANCE,
)

FLOOR = (240, 222, 222)
BLOCKED = (135, 135, 145)
TRAVELED = (15, 155, 143)
REJECTED = (230, 104, 40)
MAP_NOTE = ("Planning map and current measured depth. Light red: unoccupied map cells, "
            "not certified reachability. Gray: blocked cells; heights are not inferred. "
            "Teal: measured travel. No reconstruction mesh loaded.")


def map_mesh(navigation, floor_height_m):
    """Raster pixel centres retain the supplied transform, including its handedness."""
    height, width = navigation.occupied.shape
    pixels = np.array([[-.5, -.5, 1], [width - .5, -.5, 1],
                       [width - .5, height - .5, 1], [-.5, height - .5, 1]])
    world = pixels @ navigation.T_map_px.T
    vertices = np.column_stack((world[:, :2] / world[:, 2:], np.full(4, floor_height_m)))
    colours = np.where(navigation.occupied[..., None], BLOCKED, FLOOR).astype(np.uint8)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(colours), metallicFactor=0., roughnessFactor=1.,
        doubleSided=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0., 1.], [1., 1.], [1., 0.], [0., 0.]], material=material)
    return mesh


def measured_cloud(frame, maximum_points=12000):
    """Bounded current RGB-D samples; labels on the robot are excluded."""
    valid = np.isfinite(frame.depth) & (frame.depth > 0) & (frame.labels >= 0)
    v, u = np.nonzero(valid)
    stride = max(1, math.ceil(len(u) / maximum_points))
    u, v = u[::stride], v[::stride]
    rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(frame.intrinsics).T
    points = rays * frame.depth[v, u, None]
    world = points @ frame.world_camera[:3, :3].T + frame.world_camera[:3, 3]
    return world.astype(np.float32), frame.rgb[v, u]


class MeasuredRobot:
    """Pose the visual URDF by measured joints and the observed torso transform."""
    def __init__(self, server, root):
        import yourdfpy
        from holosoma_inference.g1_control.config import amo_profile
        from viser.extras import ViserUrdf

        root = Path(root)
        path = root / "assets/robot/g1_viz/g1_29dof_with_hand_rev_1_0_viz.urdf"
        self.urdf = yourdfpy.URDF.load(str(path), load_meshes=True, build_scene_graph=True,
                                       load_collision_meshes=False)
        self.frame = server.scene.add_frame("/real/robot", show_axes=False, visible=False)
        self.visual = ViserUrdf(server, self.urdf, root_node_name="/real/robot")
        self.names = tuple(self.visual.get_actuated_joint_names())
        self.body = amo_profile().robot.joint_names
        self.hands = json.loads((root / "assets/robot/calibration.json").read_text())["hand_joints"]

    def hide(self):
        self.frame.visible = False

    def show(self, frame):
        values = dict(zip(self.body, frame.joints, strict=True))
        for side, names in self.hands.items():
            if side not in frame.hands:
                self.hide()
                return False
            values.update(zip(names, frame.hands[side].joints, strict=True))
        cfg = np.asarray([values[name] for name in self.names])
        if not np.isfinite(cfg).all() or not np.isfinite(frame.world_torso).all():
            self.hide()
            return False
        self.visual.update_cfg(cfg)
        world_pelvis = frame.world_torso @ np.linalg.inv(self.urdf.get_transform("torso_link", "pelvis"))
        self.frame.position = tuple(world_pelvis[:3, 3])
        self.frame.wxyz = tuple(Rotation.from_matrix(world_pelvis[:3, :3]).as_quat(scalar_first=True))
        self.frame.visible = True
        return True


class RealSceneView:
    """No backend access: every displayed measurement comes from show(frame)."""
    def __init__(self, server, root, *, navigation, floor_height_m,
                 rate=4., clock=time.monotonic):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Scene display rate must be positive")
        if not np.isfinite(floor_height_m):
            raise ValueError("A finite map floor height is required")
        self.server, self.navigation, self.rate, self.clock = server, navigation, rate, clock
        self.floor_height_m = float(floor_height_m)
        self._lock = RLock()
        self._selection_handles, self._route_handles, self._arm_handles = [], [], []
        self._map = self._cloud = self._trail_handle = None
        self._trail, self._epoch, self._last, self._seen = [], None, None, None
        self._shown, self._occupied = -math.inf, None
        scene = server.scene
        scene.set_up_direction("+z")
        scene.configure_environment_map(None)
        scene.configure_default_lights(False, cast_shadow=False)
        scene.set_background_image(np.full((2, 2, 3), SCENE_BACKGROUND, np.uint8), format="png")
        scene.add_light_ambient("/lighting/ambient", **LIGHTING["ambient"])
        scene.add_light_directional("/lighting/key", **LIGHTING["key"])
        scene.add_light_directional("/lighting/fill", **LIGHTING["fill"])
        self.note = server.gui.add_markdown(MAP_NOTE)
        self._robot = MeasuredRobot(server, root)
        mesh = map_mesh(navigation, self.floor_height_m)
        center = mesh.bounds.mean(axis=0)
        span = max(float(np.ptp(mesh.vertices[:, :2], axis=0).max()), 2.)
        self.home = ((center[0], center[1] - .9 * span, center[2] + .8 * span), tuple(center))
        server.on_client_connect(self.place_camera)
        self.reset_camera()
        self._draw_map(navigation)

    def place_camera(self, client):
        with client.atomic():
            client.camera.up_direction = (0., 0., 1.)
            client.camera.position, client.camera.look_at = self.home

    def reset_camera(self):
        for client in self.server.get_clients().values():
            self.place_camera(client)

    @staticmethod
    def _drop(handles):
        for handle in handles:
            handle.remove()
        handles.clear()

    def _path(self, name, points, colour):
        return self.server.scene.add_line_segments(
            name, points=np.stack((points[:-1], points[1:]), axis=1), colors=colour, thickness=LINE_3D)

    def _dot(self, name, point, colour, radius=DOT_3D):
        return self.server.scene.add_icosphere(name, radius=radius, color=colour,
                                               position=point, cast_shadow=False, receive_shadow=False)

    def _draw_map(self, navigation):
        if self._occupied is None or not np.array_equal(self._occupied, navigation.occupied):
            if self._map is not None:
                self._map.remove()
            self._map = self.server.scene.add_mesh_trimesh(
                "/real/navigation", map_mesh(navigation, self.floor_height_m),
                cast_shadow=False, receive_shadow=False)
            self._occupied = navigation.occupied
        self._map.visible = True

    def _same_map(self, navigation):
        original = self.navigation
        return (navigation.source_hash == original.source_hash and navigation.frame_id == original.frame_id
                and navigation.occupied.shape == original.occupied.shape
                and navigation.resolution == original.resolution
                and np.array_equal(navigation.T_map_px, original.T_map_px))

    def _clear_trail(self):
        self._trail.clear()
        if self._trail_handle is not None:
            self._trail_handle.remove()
            self._trail_handle = None

    def show(self, frame):
        with self._lock:
            stamp = frame.epoch, frame.sequence
            if ((self._seen is not None and stamp < self._seen)
                    or (self._last is not None and stamp == self._last)):
                return
            self._seen = stamp
            if not self._same_map(frame.navigation):
                self._robot.hide()
                self._map.visible = False
                if self._cloud is not None:
                    self._cloud.visible = False
                self.clear_selection()
                self._clear_trail()
                self.note.content = "Scene hidden: observation does not match the bound navigation map/frame."
                return
            if frame.epoch != self._epoch:
                self.clear_selection()
                self._clear_trail()
                self._epoch, self._shown = frame.epoch, -math.inf
            now = self.clock()
            if now - self._shown < 1 / self.rate:
                return
            self._shown, self._last = now, stamp
            with self.server.atomic():
                self._draw_map(frame.navigation)
                robot_visible = self._robot.show(frame)
                self.note.content = MAP_NOTE + ("" if robot_visible else " Robot hidden: joint/pose evidence missing.")
                points, colours = measured_cloud(frame)
                if self._cloud is not None:
                    self._cloud.remove()
                    self._cloud = None
                if len(points):
                    self._cloud = self.server.scene.add_point_cloud(
                        "/real/depth", points=points, colors=colours, point_size=.015)
                xy = np.asarray(frame.base_pose[:2], dtype=float)
                if np.isfinite(xy).all() and (not self._trail or np.linalg.norm(xy - self._trail[-1]) > 1e-4):
                    self._trail.append(xy.copy())
                    if len(self._trail) > 1:
                        if self._trail_handle is not None:
                            self._trail_handle.remove()
                        world = np.column_stack((self._trail, np.full(len(self._trail),
                                                                     self.floor_height_m + .025)))
                        self._trail_handle = self._path("/real/traveled", world, TRAVELED)

    def clear_selection(self):
        with self._lock:
            for handles in (self._selection_handles, self._route_handles, self._arm_handles):
                self._drop(handles)

    def show_selection(self, point):
        with self._lock:
            self._drop(self._selection_handles)
            point = np.asarray(point, dtype=float)
            if point.shape == (2,):
                point = np.r_[point, self.floor_height_m + .04]
            if point.shape == (3,) and np.isfinite(point).all():
                self._selection_handles = [self._dot("/real/selection", point, MASK)]

    def show_rejection(self, point):
        with self._lock:
            self._drop(self._route_handles)
            self.show_selection(point)
            if self._selection_handles:
                self._selection_handles[0].color = REJECTED

    def show_route(self, waypoints, yaw=None, *, goal=None):
        with self._lock:
            self._drop(self._route_handles)
            points = np.asarray(waypoints, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                return
            stance = points[-1] if goal is None else np.asarray(goal, dtype=float)
            if stance.shape != (2,) or not np.isfinite(stance).all():
                return
            world = np.column_stack((points, np.full(len(points), self.floor_height_m + .04)))
            stance = np.r_[stance, self.floor_height_m + .04]
            if len(world) > 1:
                self._route_handles.append(self._path("/real/route", world, ROUTE))
                self._route_handles.extend(self._dot(f"/real/route-wp-{i}", point, ROUTE, .035)
                                           for i, point in enumerate(world[1:], 1))
            self._route_handles.append(self._dot("/real/stance", stance, STANCE))
            if yaw is not None and np.isfinite(yaw):
                tip = stance + HEADING * np.array([np.cos(yaw), np.sin(yaw), 0.])
                self._route_handles.append(self._path("/real/heading", np.array([stance, tip]), STANCE))

    def show_arm_path(self, points):
        with self._lock:
            self._drop(self._arm_handles)
            points = np.asarray(points, dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) > 1 and np.isfinite(points).all():
                self._arm_handles = [self._path("/real/arm-path", points, PLAN),
                                     self._dot("/real/arm-goal", points[-1], PLAN)]

    def clear_arm_path(self):
        with self._lock:
            self._drop(self._arm_handles)
