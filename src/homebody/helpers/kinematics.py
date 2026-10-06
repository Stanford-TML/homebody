"""Arm kinematics from the robot URDF: forward kinematics and damped least-squares IK."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .collision import CollisionGeometry
from .geometry import axis_rotation, joint_origin, rotation_error, skew, transform

POSITION_TOLERANCE, ROTATION_TOLERANCE = 0.004, 0.06  # m, rad
ROTATION_WEIGHT, DAMPING_SQUARED, MAX_STEP = 0.25, 0.002, 0.10
LIMIT_MARGIN = 0.02  # rad inside each joint limit
RANDOM_RESTARTS = 6
# Settling the arm's redundancy: a pull toward a preferred pose in the task's null space,
# at most SETTLE_STEP rad per round and re-converged each round, kept only within
# SETTLE_LIMIT rad of the unsettled solution.
SETTLE_ROUNDS, SETTLE_STEP, SETTLE_LIMIT = 3, 0.08, 0.3


def camera_mount(urdf: Path):
    """Head camera optical frame (x right, y down, z forward) in the torso frame."""
    joint = next(j for j in ET.parse(urdf).getroot().findall("joint")
                 if j.find("child").get("link") == "d435_link")
    return joint_origin(joint) @ transform([0, 0, 0], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


class Arm:
    def __init__(self, urdf: Path, side: str):
        if side not in ("left", "right"):
            raise ValueError("Hand must be left or right")
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        link = side + "_hand_palm_link"
        chain = []
        while link != "torso_link":
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.collision = CollisionGeometry(urdf.parent / "g1_grasp" / "meshes", side, root,
                                           [j.find("child").get("link") for j in self.chain])
        self.origins = [joint_origin(j) for j in self.chain]
        self.moving = [j for j in self.chain if j.get("type") != "fixed"]
        self.names = tuple(j.get("name") for j in self.moving)
        self.axes = [np.fromstring(j.find("axis").get("xyz"), sep=" ") for j in self.moving]
        self.cross = [skew(axis / np.linalg.norm(axis)) for axis in self.axes]
        self.lower = np.array([float(j.find("limit").get("lower")) for j in self.moving])
        self.upper = np.array([float(j.find("limit").get("upper")) for j in self.moving])
        first = next(i for i, joint in enumerate(self.chain) if joint.get("type") != "fixed")
        pose = np.eye(4)
        for base in self.origins[:first + 1]:
            pose = pose @ base
        self.shoulder = pose[:3, 3].copy()
        self.reach = sum(np.linalg.norm(base[:3, 3]) for base in self.origins[first + 1:])
        rng = np.random.default_rng(0)
        self.restarts = [(self.lower + self.upper) / 2,
                         *rng.uniform(self.lower, self.upper, (RANDOM_RESTARTS, len(self.lower)))]

    def forward(self, joints):
        """The palm pose in the torso frame and its 6x7 Jacobian (linear rows first)."""
        pose = np.eye(4)
        axes, centers = [], []
        index = 0
        for joint, base in zip(self.chain, self.origins):
            pose = pose @ base
            if joint.get("type") != "fixed":
                centers.append(pose[:3, 3].copy())
                axes.append(pose[:3, :3] @ self.axes[index])
                pose = pose @ transform([0, 0, 0], axis_rotation(self.cross[index], joints[index]))
                index += 1
        jacobian = np.vstack((np.cross(axes, pose[:3, 3] - centers).T, np.array(axes).T))
        return pose, jacobian

    def solve(self, target, seed, *, iterations=150, position_only=False, restarts=True):
        """Joints reaching the torso-frame TARGET from SEED, trying fixed restarts when
        RESTARTS, or None when it does not solve or the pose is beyond reach."""
        if not np.isfinite(target).all() or np.linalg.norm(target[:3, 3] - self.shoulder) > self.reach:
            return None
        for start in (seed, *(self.restarts if restarts else ())):
            result = self._descend(target, start, iterations, position_only)
            if result is not None:
                return result
        return None

    def settle_toward(self, joints, target, prefer, *, position_only=False):
        """JOINTS, a solution for TARGET, moved toward the joint pose PREFER as far as the
        task's null space allows (the elbow's place around a reachable palm); unchanged
        when the move would leave the solution's branch."""
        settled = joints
        for _ in range(SETTLE_ROUNDS):
            _, jacobian = self.forward(settled)
            if position_only:
                jacobian = jacobian[:3]
            inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + DAMPING_SQUARED * np.eye(len(jacobian)))
            pull = (np.eye(len(settled)) - inverse @ jacobian) @ (prefer - settled)
            pulled = settled + pull * min(1., SETTLE_STEP / max(np.abs(pull).max(), 1e-9))
            pulled = self._descend(target, pulled, 60, position_only)
            if pulled is None:
                break
            settled = pulled
        return settled if np.abs(settled - joints).max() <= SETTLE_LIMIT else joints

    def _descend(self, target, seed, iterations, position_only=False):
        joints = np.clip(np.asarray(seed, dtype=float), self.lower + LIMIT_MARGIN, self.upper - LIMIT_MARGIN)
        for _ in range(iterations):
            pose, jacobian = self.forward(joints)
            error = np.r_[target[:3, 3] - pose[:3, 3],
                          rotation_error(target[:3, :3], pose[:3, :3])]
            if np.linalg.norm(error[:3]) < POSITION_TOLERANCE and (
                    position_only or np.linalg.norm(error[3:]) < ROTATION_TOLERANCE):
                return joints
            error[3:] *= 0. if position_only else ROTATION_WEIGHT
            jacobian[3:] *= 0. if position_only else ROTATION_WEIGHT
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + DAMPING_SQUARED * np.eye(6), error)
            joints = np.clip(joints + np.clip(step, -MAX_STEP, MAX_STEP),
                             self.lower + LIMIT_MARGIN, self.upper - LIMIT_MARGIN)
        return None
