"""The MuJoCo implementation of Robot. Only this object owns simulation state."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from homebody.helpers.geometry import transform
from homebody.primitives.actions import ActionLimits, PhysicsInvalid
from homebody.primitives.observations import Frame, HandState, Measurement, NavigationMap

from .controller import SIDES, Controller
from .scene import DEFAULT_SCENE, load

ARM_LINKS = ("_hand_", "_wrist_", "_elbow_")
SPAWN_CEILING = 1.3  # m above the ground plane
COLLISION_GROUPS = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)  # scene and object hulls


def holding_side(body_name):
    """The side of the arm link a robot body belongs to, or None off the hand, wrist and elbow."""
    return body_name.split("_", 1)[0] if any(link in body_name for link in ARM_LINKS) else None


def hand_state(joints, target, ray, forces, ceilings, names):
    """The HandState of one hand from its joints, closure target, ray and actuator forces."""
    joints, target, ray, forces, ceilings = (
        np.asarray(value, dtype=float) for value in (joints, target, ray, forces, ceilings))
    if (any(value.shape != (7,) or not np.isfinite(value).all()
            for value in (joints, target, ray, forces, ceilings)) or np.any(ceilings <= 0)):
        raise ValueError("Hand sensors require seven finite joints and positive effort ceilings")
    groups = {finger: np.array([i for i, name in enumerate(names) if f"_hand_{finger}_" in name])
              for finger in ("thumb", "index", "middle")}
    if len(names) != 7 or len(set(names)) != 7 or tuple(map(len, groups.values())) != (3, 2, 2):
        raise ValueError("Hand joint names must identify three thumb and two joints per opposing finger")
    gaps, efforts = {}, {}
    for finger, indices in groups.items():
        direction = ray[indices]
        norm = float(direction @ direction)
        if norm <= 0:
            raise ValueError(f"The {finger} closure ray has no closing motion")
        gaps[finger] = max(0., float((target[indices] - joints[indices]) @ direction / norm))
        efforts[finger] = float(np.max(np.abs(forces[indices]) / ceilings[indices]))
    closure = float(np.clip(joints @ ray / (ray @ ray), 0., 1.))
    return HandState(joints, closure, np.array([efforts[finger] for finger in groups]),
                     np.array([gaps[finger] for finger in groups]))


class MujocoBackend:
    def __init__(self, root: Path, settings, *, scene: Path | None = DEFAULT_SCENE, render=True,
                 spawns=None):
        """SCENE is a scene package directory, relative to ROOT or absolute, or None for the
        robot alone on a flat floor. SPAWNS, `{identity: (x, y, yaw)}`, stands those movable
        objects upright on the static surface under (x, y) at every reset."""
        self.root, self.settings = Path(root), settings
        self.scene = None if scene is None else self.root / scene
        self.render = render
        self.spawns = dict(spawns or {})
        self.epoch = 0
        self._renderer = self._observer_renderer = None
        self.reset()

    @property
    def time(self):
        return float(self.data.time)

    def reset(self):
        self.close()
        self._evaluation_sink = None
        self.epoch += 1
        self.sequence = 0
        self._frame_cache = None
        self._valid = True
        self.model, self.data = load(self.root, self.settings, self.scene)
        for identity, spawn in self.spawns.items():
            x, y, yaw = spawn
            self._stand(identity, x, y, self._surface_under(x, y), yaw)
        self.controller = Controller(self.model, self.data, self.root, self.settings.physics,
                                     self.root / ".cache/amo")
        self.limits = ActionLimits(self.settings, self.controller.calibration)
        if 1 + self.settings.grasp.squeeze > 1 / self.controller.calibration["hand_close_fraction"]:
            raise ValueError("grasp.squeeze would close past the hand's full closed pose")
        self._velocity_expiry = 0.
        planes = np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        if len(planes) != 1:
            raise ValueError("Scene must define exactly one ground collision plane")
        floor = int(planes[0])
        self._floor_point = self.data.geom_xpos[floor].copy()
        self._floor_normal = self.data.geom_xmat[floor].reshape(3, 3)[:, 2].copy()
        self._pelvis = self.model.body("pelvis").id
        self._labels = self._make_labels()
        if self.scene is not None:
            with np.load(self.scene / "map.npz") as artifact:
                self.navigation = NavigationMap(artifact["occupancy"], artifact["T_map_px"],
                    float(np.linalg.norm(artifact["T_map_px"][:2, 0])))
            self._objects = json.loads((self.scene / "physics.json").read_text())["objects"]
        else:
            self.navigation = NavigationMap(np.zeros((200, 200), dtype=bool),
                np.array([[.025, 0, -2.5], [0, -.025, 2.5], [0, 0, 1]]), .025)
            self._objects = {}
        self._hulls = {identity: self._hull(identity) for identity in self._objects}
        self._camera = self.model.camera("ego").id
        self._torso = self.model.body("torso_link").id
        c = self.settings.camera
        self.model.vis.map.znear = c.near / self.model.stat.extent
        self.model.vis.map.zfar = c.far / self.model.stat.extent
        focal = c.width / (2 * math.tan(math.radians(c.horizontal_fov) / 2))
        self._intrinsics = np.array([[focal, 0, (c.width - 1)/2],
                                     [0, focal, (c.height - 1)/2], [0, 0, 1]])
        for _ in range(10 * self.settings.physics.control_hz):
            self.controller.velocity()
            self._tick()

    def _hull(self, identity):
        """IDENTITY's collision geom id and its mesh vertices in the geom's frame."""
        name = f"{identity}__collision"
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0 or self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"The scene package gives {identity!r} no mesh collision geom named {name!r}")
        mesh = self.model.geom_dataid[geom]
        start = self.model.mesh_vertadr[mesh]
        return geom, self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]]

    def _stand(self, identity, x, y, z, yaw):
        """Place IDENTITY upright at (x, y, yaw) with its hull's lowest point at height z."""
        geom, vertices = self._hull(identity)
        body = self.model.body(identity)
        if body.jntnum[0] != 1 or self.model.jnt_type[body.jntadr[0]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{identity!r} is not a free body; only a movable object can be stood")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.model.geom_quat[geom])
        bottom = float((vertices @ rotation.reshape(3, 3).T + self.model.geom_pos[geom])[:, 2].min())
        address = self.model.jnt_qposadr[body.jntadr[0]]
        self.data.qpos[address:address + 3] = [x, y, z - bottom + 0.005]
        self.data.qpos[address + 3:address + 7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        self.data.qvel[self.model.jnt_dofadr[body.jntadr[0]]:][:6] = 0.
        mujoco.mj_forward(self.model, self.data)

    def _surface_under(self, x, y):
        """Height of the static surface under map point (x, y), cast down from SPAWN_CEILING."""
        plane = int(np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
        origin = np.array([x, y, self.data.geom_xpos[plane][2] + SPAWN_CEILING])
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]),
                                 COLLISION_GROUPS, 1, -1, hit)
        if distance < 0 or self.model.body_weldid[self.model.geom_bodyid[int(hit[0])]] != 0:
            raise ValueError(f"No static surface under ({x:g}, {y:g}) to stand an object on")
        return float(origin[2] - distance)

    def _make_labels(self):
        """Per-geom labels: one positive integer per scene object, -1 on the robot, 0 elsewhere."""
        labels = np.zeros(self.model.ngeom, dtype=np.int32)
        groups = {}
        for i in range(self.model.ngeom):
            if self.model.body_rootid[self.model.geom_bodyid[i]] == self._pelvis:
                labels[i] = -1
                continue
            name = self.model.geom(i).name
            if "__" in name:
                identity = name.split("__", 1)[0]
                if identity not in ("floor", "structure"):
                    groups.setdefault(identity, len(groups) + 1)
                    labels[i] = groups[identity]
        self.label_ids = groups
        return labels

    def _check_epoch(self, epoch):
        self.limits.epoch(epoch, self.epoch)
        if not self._valid:
            raise PhysicsInvalid("Physics is invalid; reset before another task")

    def _tick(self):
        before = self.time
        self.controller.tick()
        self._check_physics(before)

    def _check_physics(self, before):
        """Raise PhysicsInvalid unless time advanced, state is finite and the robot stands."""
        p = self.settings.physics
        height = float((self.data.qpos[:3] - self._floor_point) @ self._floor_normal)
        self._valid = bool(self._valid and self.time > before and np.isfinite(self.data.qpos).all()
                           and not np.any(self.data.warning.number) and self._upright()
                           and p.min_base_height <= height <= p.max_base_height)
        if not self._valid:
            raise PhysicsInvalid("Robot fell or the physics solver reported invalid state")

    def _upright(self):
        up = self.data.xmat[self._torso].reshape(3, 3)[2, 2]
        return bool(up > math.cos(math.radians(self.settings.physics.max_tilt_degrees)))

    def _base_yaw(self):
        rotation = self.data.xmat[self._pelvis].reshape(3, 3)
        return math.atan2(rotation[1, 0], rotation[0, 0])

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self._check_epoch(epoch)
        values = self.limits.velocity(forward, lateral, yaw_rate)
        self.controller.velocity(*values)
        self._velocity_expiry = self.time + self.settings.physics.command_timeout

    def arm_target(self, epoch, side, joints):
        self._check_epoch(epoch)
        self.controller.arm_target[SIDES[side]] = self.limits.arm(side, joints)

    def grip(self, epoch, side, closure):
        self._check_epoch(epoch)
        closure = self.limits.closure(side, closure)
        self.controller.hand_target[side] = self.controller.closed[side] * closure

    def stop(self, epoch):
        self.limits.epoch(epoch, self.epoch)
        self.controller.velocity()
        self._velocity_expiry = self.time
        self.controller.hold_arms()

    def advance(self, epoch, seconds):
        self._check_epoch(epoch)
        seconds = self.limits.duration(seconds)
        ticks = round(seconds * self.settings.physics.control_hz)
        for _ in range(ticks):
            if self.time >= self._velocity_expiry:
                self.controller.velocity()
            self._tick()
            if self._evaluation_sink is not None and self.time >= self._next_sample:
                self._evaluation_sink(self.evaluation_state())
                self._next_sample = self.time + self.settings.recording.physics_period

    def snapshot(self):
        if not self.render:
            raise RuntimeError("Camera rendering was disabled for this backend")
        if self._frame_cache is not None and self._frame_cache.time == self.time:
            return self._frame_cache
        rgb, depth, segmentation = self._render_ego()
        ids = segmentation[:, :, 0]
        valid = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        labels = np.zeros(ids.shape, dtype=np.int32)
        labels[valid] = self._labels[ids[valid]]
        depth[(depth >= self.settings.camera.far) | ~np.isfinite(depth)] = np.nan
        state = self.measure()
        self._frame_cache = Frame(state.epoch, state.sequence, state.time, rgb, depth, self._intrinsics,
                                 state.world_camera, state.world_torso, state.base_pose, state.joints,
                                 state.hands, labels, self.navigation)
        return self._frame_cache

    def measure(self):
        camera_rotation = self.data.cam_xmat[self._camera].reshape(3, 3) @ np.diag([1, -1, -1])
        self.sequence += 1
        return Measurement(self.epoch, self.sequence, self.time,
                           transform(self.data.cam_xpos[self._camera], camera_rotation),
                           transform(self.data.xpos[self._torso], self.data.xmat[self._torso].reshape(3, 3)),
                           np.r_[self.data.qpos[:2], self._base_yaw()], self.data.qpos[self.controller.port.q],
                           {side: self._hand(side) for side in ("left", "right")})

    def _hand(self, side):
        """SIDE's HandState."""
        controller, names = self.controller, self.controller.calibration["hand_joints"][side]
        joints = [self.model.joint(name).id for name in names]
        return hand_state(self.data.qpos[controller.hand_q[side]], controller.hand_target[side],
                          controller.closed[side], self.data.qfrc_actuator[self.model.jnt_dofadr[joints]],
                          np.max(np.abs(self.model.jnt_actfrcrange[joints]), axis=1), names)

    def _render_ego(self):
        """The ego camera's RGB, depth and segmentation images."""
        renderer = self._render_context()
        renderer.update_scene(self.data, camera="ego")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        return rgb, depth, segmentation

    def set_evaluation_sink(self, callback):
        self._evaluation_sink = callback
        if callback is not None:
            callback(self.evaluation_state())
            self._next_sample = self.time + self.settings.recording.physics_period

    def _render_context(self):
        """The robot camera's renderer, made without multisampling so segmentation ids stay exact."""
        if self._renderer is None:
            c, quality = self.settings.camera, self.model.vis.quality
            samples, quality.offsamples = quality.offsamples, 0
            try:
                self._renderer = mujoco.Renderer(self.model, c.height, c.width)
            finally:
                quality.offsamples = samples
        return self._renderer

    def _observer_context(self):
        """The operator views' renderer."""
        if self._observer_renderer is None:
            c = self.settings.camera
            self._observer_renderer = mujoco.Renderer(self.model, c.observer_height, c.observer_width)
        return self._observer_renderer

    def _observer_render(self, lookat, distance, elevation, azimuth):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        renderer = self._observer_context()
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def observer_frame(self):
        """Operator-only room view from behind the robot, never an observation."""
        return self._observer_render(self.data.qpos[:3] + [0., 0., .15], 1.8, -30.,
                                     math.degrees(self._base_yaw()) + 45.)

    def shoulder_frame(self, side):
        """Operator-only view from behind one shoulder, fixed to the torso's heading."""
        torso = self.data.xmat[self._torso].reshape(3, 3)
        yaw = math.atan2(torso[1, 0], torso[0, 0])
        outward = 1. if side == "left" else -1.
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        shoulder = self.data.xpos[self.model.body(f"{side}_shoulder_pitch_link").id]
        lookat = shoulder + .6 * heading + .05 * outward * left - [0., 0., .4]
        return self._observer_render(lookat, 1.5, -35., math.degrees(yaw) - 25. * outward)

    def evaluation_state(self):
        return {"schema_version": 1, "time": self.time, "epoch": self.epoch,
                "physics_valid": bool(self._valid), "robot_upright": self._upright(),
                "warnings": [mujoco.mjtWarning(int(i)).name
                             for i in np.flatnonzero(self.data.warning.number)],
                "objects": {identity: self._object_state(identity, geom, vertices)
                            for identity, (geom, vertices) in self._hulls.items()}}

    def _object_state(self, identity, geom, vertices):
        body = self.model.body(identity).id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        touching = {"hand": set(), "robot": set(), "support": set(), "object": set()}
        for contact in self.data.contact:
            pair = contact.geom1, contact.geom2
            bodies = self.model.geom_bodyid[list(pair)]
            if body in bodies:
                kind, name = self._toucher(pair[1] if bodies[0] == body else pair[0])
                if kind is not None:
                    touching[kind].add(name)
        corners = vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
        return {"position": self.data.xpos[body].tolist(),
                "quaternion_wxyz": self.data.xquat[body].tolist(), "corners": corners.tolist(),
                "linear_velocity": velocity[3:].tolist(), "angular_velocity": velocity[:3].tolist(),
                "hand_contacts": sorted(touching["hand"]), "robot_contacts": sorted(touching["robot"]),
                "support_contacts": sorted(touching["support"]),
                "object_contacts": sorted(touching["object"])}

    def _toucher(self, geom):
        """(kind, name) of GEOM as a toucher: hand side, robot link, object or support, else (None, None)."""
        body = self.model.geom_bodyid[geom]
        name = self.model.body(body).name
        if self.model.body_rootid[body] == self._pelvis:
            side = holding_side(name)
            return ("hand", side) if side else ("robot", name)
        if name in self._objects:
            return "object", name
        if self.model.body_weldid[body] == 0:
            return "support", self.model.geom(geom).name.split("__", 1)[0]
        return None, None

    def close(self):
        for name in ("_renderer", "_observer_renderer"):
            if getattr(self, name) is not None:
                getattr(self, name).close()
                setattr(self, name, None)
