"""Arm collision geometry from the robot meshes and the collision gate against the other
arm, the sensed depth and a held object's last observed bounds."""
import json
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

from .convex import convex_shape, point_hits
from .geometry import axis_rotation, inverse, joint_origin, points_in, skew, transform, unproject
from .robot_collision import MeshBox, mesh_overlap

PRIMITIVE_LINKS = ("shoulder_pitch_link", "shoulder_roll_link", "hand_thumb_1_link")
BEARING_LINKS = 3
HAND_END_LINKS = 3


class CollisionGeometry:
    """Bounding boxes, with convex hulls where a mesh allows one, for one arm's links and
    fingers."""

    def __init__(self, mesh_directory: Path, side: str, robot_root, links):
        self.side = side
        self.links = tuple(links)
        self.bounds = {}
        self.shapes = {}
        calibration = json.loads((mesh_directory.parents[1] / "calibration.json").read_text())
        self.hand_names = calibration["hand_joints"][side]
        self.finger_joints = [joint for joint in robot_root.findall("joint")
                              if joint.get("name") in self.hand_names]
        self.finger_origins = {joint.get("name"): joint_origin(joint) for joint in self.finger_joints}
        self.finger_cross = {}
        for joint in self.finger_joints:
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            self.finger_cross[joint.get("name")] = skew(axis / np.linalg.norm(axis))
        finger_links = [joint.find("child").get("link") for joint in self.finger_joints]
        for name in (*self.links, *finger_links):
            mesh = trimesh.load_mesh(mesh_directory / (name + ".STL"), process=False)
            low, high = mesh.bounds
            self.bounds[name] = ((low + high) / 2, (high - low) / 2)
            shape = None if name.endswith(PRIMITIVE_LINKS) else convex_shape(mesh.vertices - (low + high) / 2)
            if shape is not None:
                self.shapes[name] = shape

    def posed(self, name, pose):
        center, half = self.bounds[name]
        center, rotation = points_in(pose, center), pose[:3, :3]
        if name in self.shapes:
            return MeshBox(center, rotation, half, self.shapes[name])
        return center, rotation, half

    def boxes(self, arm, joints):
        pose, index, boxes = np.eye(4), 0, []
        for joint, base in zip(arm.chain, arm.origins):
            pose = pose @ base
            if joint.get("type") != "fixed":
                pose = pose @ transform([0, 0, 0], axis_rotation(arm.cross[index], joints[index]))
                index += 1
            name = joint.find("child").get("link")
            if name in self.bounds:
                boxes.append(self.posed(name, pose))
        return boxes

    def hand_boxes(self, arm, joints, fingers):
        palm, _ = arm.forward(joints)
        poses = {self.side + "_hand_palm_link": palm}
        measured = dict(zip(self.hand_names, fingers))
        boxes = []
        for joint in self.finger_joints:
            name = joint.get("name")
            pose = poses[joint.find("parent").get("link")] @ self.finger_origins[name]
            pose = pose @ transform([0, 0, 0], axis_rotation(self.finger_cross[name], measured[name]))
            child = joint.find("child").get("link")
            poses[child] = pose
            boxes.append(self.posed(child, pose))
        return boxes



def contains(box, points, margin=0.0):
    center, rotation, half = box
    local = (points - center) @ rotation
    hits = np.all(np.abs(local) <= half + margin, axis=1)
    if isinstance(box, MeshBox) and np.any(hits):
        hits[hits] = point_hits(box.shape, local[hits], margin)
    return hits


def overlap(first, second, margin=0.0):
    """Separating-axis test for two oriented bounding boxes, including edge axes."""
    ca, ra, ha = first
    cb, rb, hb = second
    ha = ha + margin
    delta = cb - ca
    if delta @ delta > (np.linalg.norm(ha) + np.linalg.norm(hb)) ** 2:
        return False
    rotation = ra.T @ rb
    absolute = np.abs(rotation) + 1e-12
    offset = ra.T @ delta
    if np.any(np.abs(offset) > ha + absolute @ hb):
        return False
    if np.any(np.abs(offset @ rotation) > hb + absolute.T @ ha):
        return False
    following, previous = [1, 2, 0], [2, 0, 1]
    cross_distance = np.abs(offset[previous, None] * rotation[following, :] -
                            offset[following, None] * rotation[previous, :])
    cross_extent = (ha[following, None] * absolute[previous, :] +
                    ha[previous, None] * absolute[following, :] +
                    hb[None, following] * absolute[:, previous] +
                    hb[None, previous] * absolute[:, following])
    return bool(np.all(cross_distance <= cross_extent)) and mesh_overlap(first, second, margin)


def near_pairs(first, second):
    """Bounding-sphere prefilter of overlap, for every pair of boxes at once."""
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=bool)
    centers = [np.array([box[0] for box in boxes]) for boxes in (first, second)]
    radii = [np.linalg.norm(np.array([box[2] for box in boxes]), axis=1) for boxes in (first, second)]
    reach = (radii[0][:, None] + radii[1][None, :]) ** 2
    return np.sum((centers[0][:, None] - centers[1][None, :]) ** 2, axis=2) <= reach


def projected_box_domain(box, margin):
    """Counter-clockwise XY hull of BOX padded by MARGIN."""
    center, rotation, half = box
    corners = np.array(list(product((-1., 1.), repeat=3))) * (half + margin)
    xy = (corners @ rotation.T + center)[:, :2]
    return xy[ConvexHull(xy).vertices]


def under_box(points, box, margin):
    return _support_domain(points, projected_box_domain(box, margin))


def _support_domain(points, vertices):
    """Points whose XY lies in a closed counter-clockwise convex polygon."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    delta = points[:, None, :2] - vertices
    return np.all(edges[:, 0] * delta[:, :, 1] - edges[:, 1] * delta[:, :, 0] >= 0., axis=1)


def _arm_boxes(arm, frame, side):
    """One arm's link and finger boxes at its measured joints."""
    joints = frame.arm_joints(side)
    return [*arm.collision.boxes(arm, joints),
            *arm.collision.hand_boxes(arm, joints, frame.hands[side].joints)]


def _sensed_points(frame, exclude_labels):
    """Sensed depth in the torso frame, less the excluded labels and the robot's own pixels."""
    mask = (frame.labels >= 0) & ~np.isin(frame.labels, exclude_labels)
    points = unproject(frame.depth, frame.intrinsics, mask)
    return points_in(inverse(frame.world_torso) @ frame.world_camera, points)


def first_per_voxel(points, size):
    """Indices of each occupied voxel's first point, voxels in lexicographic order."""
    voxels = np.floor(points / size).astype(int)
    if not len(voxels):
        return np.zeros(0, dtype=int)
    low = voxels.min(axis=0)
    keys = np.ravel_multi_index((voxels - low).T, voxels.max(axis=0) - low + 1)
    runs = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return runs[np.unique(keys[runs], return_index=True)[1]]


class Clearance:
    """One arm's collision gate against the other arm, its own nonadjacent links, the sensed
    scene and, when loaded, its held object's last observed bounds. The torso is not checked."""

    def __init__(self, frame, arms, side, settings, exclude_labels=(), carried=None, exempt_start=False,
                 fingers=None):
        self.settings = settings
        self.initial_joints = frame.arm_joints(side)
        self.arm = arms[side]
        self.carried = carried
        self.geometry = self.arm.collision
        self.fingers = frame.hands[side].joints if fingers is None else fingers
        other = "left" if side == "right" else "right"
        self.other = _arm_boxes(arms[other], frame, other)
        own = _arm_boxes(self.arm, frame, side)
        self.observed_points = _sensed_points(frame, exclude_labels)
        self.points = self.observed_points[first_per_voxel(self.observed_points, settings.voxel_size)]
        if exempt_start:
            inside = np.zeros(len(self.points), dtype=bool)
            for box in own:
                inside |= self.hits(box, settings.cloud_margin)
            self.points = self.points[~inside]
            self.__dict__.pop("_tree", None)
        self._payload_start, self._payload_on_robot = None, False
        if exempt_start and carried is not None and "palm_to_payload" in carried:
            start = self.payload_box(self.initial_joints)
            self._payload_start = self.hits(start, settings.cloud_margin)
            self._payload_on_robot = self._payload_robot(start)

    @cached_property
    def _tree(self):
        return cKDTree(self.points)

    def hits(self, box, margin):
        """contains(box, self.points, margin), testing only points near the box."""
        hits = np.zeros(len(self.points), dtype=bool)
        near = self._tree.query_ball_point(box[0], np.linalg.norm(box[2] + margin) + 1e-9)
        if near:
            near = np.asarray(near)
            hits[near] = contains(box, self.points[near], margin)
        return hits

    def payload_box(self, joints):
        palm, _ = self.arm.forward(joints)
        pose = palm @ self.carried["palm_to_payload"]
        return pose[:3, 3], pose[:3, :3], self.carried["payload_half"]

    def payload_hits(self, box):
        hits = self.hits(box, self.settings.cloud_margin)
        if self._payload_start is not None:
            hits &= ~self._payload_start
        hull = self.carried.get("payload_hull")
        if hull is not None and np.any(hits):
            local = (self.points[hits] - box[0]) @ box[1]
            hits[np.flatnonzero(hits)] = point_hits(hull, local, self.settings.cloud_margin)
        return hits

    def violation(self, joints):
        boxes = self.geometry.boxes(self.arm, joints)
        robot = list(self.other)
        near = near_pairs(boxes, robot) if robot else np.zeros((len(boxes), 0), bool)
        own = near_pairs(boxes, boxes)
        for index, box in enumerate(boxes):
            if any(overlap(box, other) for other, close in zip(self.other, near[index]) if close):
                return f"link {index} intersects other arm"
            if any(overlap(box, boxes[earlier]) for earlier in range(max(0, index - BEARING_LINKS))
                   if own[index, earlier]):
                return f"link {index} intersects nonadjacent arm link"
            if self.blocked(self.hits(box, self.settings.cloud_margin)):
                return f"link {index} intersects visible environment"
        hands = self.geometry.hand_boxes(self.arm, joints, self.fingers)
        near = near_pairs(hands, robot) if robot else np.zeros((len(hands), 0), bool)
        for index, box in enumerate(hands):
            if any(overlap(box, part) for part, close in zip(robot, near[index]) if close):
                return f"finger {index} intersects robot"
            if self.blocked(self.hits(box, self.settings.cloud_margin)):
                return f"finger {index} intersects visible environment"
        if self.carried is not None:
            payload = self.payload_box(joints)
            if self.blocked(self.payload_hits(payload)):
                return "observed payload intersects visible environment"
            if not self._payload_on_robot and self._payload_robot(payload):
                return "observed payload intersects robot"
        return None

    def blocked(self, hits):
        return np.count_nonzero(hits) >= self.settings.min_hit_voxels

    def _payload_robot(self, payload):
        return any(overlap(payload, other) for other in self.other)

    def path_violation(self, start, end):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.settings.joint_step)))
        for fraction in np.linspace(0, 1, count + 1):
            reason = self.violation(start + fraction * (end - start))
            if reason is not None:
                return reason
        return None

    def path(self, start, end):
        return self.path_violation(start, end) is None
