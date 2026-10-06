"""Arm motions shared by pick, carry and place: joint streams, palm servos and the one
arm reset."""
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from homebody.helpers.collision import Clearance, overlap
from homebody.helpers.geometry import inverse, points_in, transform
from homebody.helpers.trajectory import joint_rows, retime_path, servo_step
from homebody.primitives.observations import Frame, FreshEvidence
from homebody.skills.contract import SkillFailure

# Left-arm torso-frame reset curve: over, wide, behind the torso, at the hip (right mirrors y).
RESET_POINTS = np.array([[0.291, 0.190, 0.210], [0.130, 0.320, 0.205],
                         [-0.020, 0.340, 0.145], [-0.004, 0.239, -0.010]])
RESET_SAMPLES = 24
RESET_RAISE_M = 0.06  # the first three points, not the hip
RESET_FIRST_RAISE_M = 0.03
RESET_SECOND_RAISE_M = 0.02
NEAR_HOME_M = 0.15  # an arm this near rest skips the curve
REACH_FRACTION = 0.95  # a sample farther out is pulled in to this much of the reach
TORSO_BOX = (np.array([0., 0., 0.17]), np.eye(3), np.array([0.11, 0.13, 0.22]))  # torso frame, m
ARM_LINKS = ("elbow", "wrist", "hand")  # the links a reset sample keeps out of the torso


class Reads:
    """A control loop's reads: a camera frame every servo.perception_period, measurements
    in between."""

    def __init__(self, ctx, epoch):
        self.ctx, self.epoch, self.seen = ctx, epoch, -math.inf

    def next(self):
        if self.ctx.observations.time - self.seen < self.ctx.settings.servo.perception_period:
            return self.ctx.measure(self.epoch)
        frame = self.ctx.observe(self.epoch)
        self.seen = frame.time
        return frame


def live_check(ctx, side, exclude_labels=()):
    """A check of each next reference against segmented depth, EXCLUDE_LABELS and the
    carried load exempt. It returns a reason when the motion is blocked."""
    scene = []

    def violation(state, reference):
        if isinstance(state, Frame):
            scene[:] = [Clearance(state, ctx.arms, side, ctx.settings.collision, exclude_labels,
                                  ctx.carried.get(side), exempt_start=True)]
        return scene[0].path_violation(state.arm_joints(side), reference)
    return violation


def guard(ctx, side, check, state, reference, phase):
    reason = None if check is None else check(state, reference)
    if reason is not None:
        ctx.emit({"type": "arm_blocked", "side": side, "reference": reference.tolist(), "reason": reason})
        raise SkillFailure("COLLISION", f"Arm {phase}: {reason}")


def follow_rows(ctx, epoch, side, rows, check=None, tracked_target=None):
    """Stream precomputed joint rows, one per servo tick, each checked before it is sent."""
    reads = Reads(ctx, epoch)
    for row in rows:
        state = reads.next()
        if tracked_target is not None:
            tracked_target.update(state)
        guard(ctx, side, check, state, row, "approach")
        ctx.check(epoch)
        ctx.actions.arm_target(epoch, side, row)
        ctx.advance(epoch, ctx.settings.servo.tick)


def within_limits(ctx, side, joints):
    """JOINTS clipped to the arm's limits, since a contact can push the arm past one."""
    arm = ctx.arms[side]
    return np.clip(joints, arm.lower, arm.upper)


def step_to(ctx, epoch, side, target, check):
    """Stream a rate-bounded joint ramp from the measured joints to TARGET."""
    rows = joint_rows(within_limits(ctx, side, ctx.measure(epoch).arm_joints(side)), target,
                      ctx.settings.physics.arm_rate, ctx.settings.servo.tick)
    follow_rows(ctx, epoch, side, rows, check)


def move_joints(ctx, epoch, side, target, exclude_labels=()):
    """Step to a joint target against the live scene."""
    step_to(ctx, epoch, side, target, live_check(ctx, side, exclude_labels))


def move_path(ctx, epoch, side, path, tracked_target=None):
    """Stream one shaped trajectory without stopping at its interior samples."""
    rows = retime_path(np.vstack((within_limits(ctx, side, ctx.measure(epoch).arm_joints(side)), path)),
                       ctx.settings.physics.arm_rate, ctx.settings.servo.tick)
    follow_rows(ctx, epoch, side, rows[1:], live_check(ctx, side), tracked_target)


STALL_SECONDS, STALL_PROGRESS_M = 1.0, 0.003
STALL_PROGRESS_RAD, REST_SETTLED_RAD = 0.001, 0.02


def track(ctx, epoch, side, world_target, tracked_target, check, best_effort=False, position_only=False):
    """Servo a world palm pose: the frame it converged on, or None when the deadline passes
    or, with BEST_EFFORT, when the palm stops closing in."""
    reads = Reads(ctx, epoch)
    state = reads.next()
    arm, command = ctx.arms[side], within_limits(ctx, side, state.arm_joints(side))
    start, evidence = state.time, FreshEvidence(epoch)
    closest, closest_time = np.inf, start
    while state.time - start < ctx.settings.motion.arm_timeout:
        target = world_target.copy()
        if tracked_target is not None:
            target[:3, 3] += tracked_target.update(state)
        target = inverse(state.world_torso) @ target
        if position_only:
            target[:3, :3] = (inverse(state.world_torso) @ state.palm(arm, side))[:3, :3]
        command, distance, angle = servo_step(
            arm, state.arm_joints(side), command, target, ctx.settings.physics.arm_rate, ctx.settings.servo)
        if distance < closest - STALL_PROGRESS_M:
            closest, closest_time = distance, state.time
        elif best_effort and state.time - closest_time > STALL_SECONDS:
            ctx.emit({"type": "arm_stalled", "side": side, "reason": f"palm stopped {closest:.3f} m from its target"})
            return None
        stable = evidence.update(state, distance < ctx.settings.motion.palm_tolerance and
                                 angle < ctx.settings.servo.rotation_tolerance)
        if stable is not None and stable >= ctx.settings.servo.stable_frames:
            return ctx.observe(epoch)
        guard(ctx, side, check, state, command, "servo")
        ctx.check(epoch)
        ctx.actions.arm_target(epoch, side, command)
        ctx.advance(epoch, ctx.settings.servo.tick)
        state = reads.next()
    return None


def servo(ctx, epoch, side, world_target, *, tracked_target=None, exclude_labels=()):
    """Servo a world palm pose against the live scene, ARM_NOT_REACHED if it does not converge."""
    frame = track(ctx, epoch, side, world_target, tracked_target, live_check(ctx, side, exclude_labels))
    if frame is None:
        raise SkillFailure("ARM_NOT_REACHED", "Measured palm did not converge before the servo deadline")
    return frame


def servo_best_effort(ctx, epoch, side, world_target, *, exclude_labels=(), position_only=False):
    """Servo a world palm pose (its position alone with POSITION_ONLY) against the live
    scene, stopping where a stall leaves the arm."""
    checked = live_check(ctx, side, exclude_labels)
    return reached_or_stopped(ctx, epoch, track(ctx, epoch, side, world_target, None, checked,
                                                best_effort=True, position_only=position_only))


def servo_unchecked(ctx, epoch, side, world_target):
    """Servo a world palm pose without the live check, stopping where a stall leaves the arm."""
    return reached_or_stopped(ctx, epoch, track(ctx, epoch, side, world_target, None, None, best_effort=True))


def reached_or_stopped(ctx, epoch, frame):
    """FRAME when the servo converged, else the frame after stopping the arm where it is."""
    if frame is not None:
        return frame
    ctx.actions.stop(epoch)
    return ctx.observe(epoch)


def reset_arm(ctx, epoch, side, above=None):
    """The one arm reset, unchecked, for either arm: follow the reset curve to the hip
    unless already near rest, then ramp into the rest joints. Returns whether the arm
    ended within rest_tolerance. ABOVE (torso-frame centre XY, radius, height) keeps the
    curve over a just-released object."""
    arm, rest = ctx.arms[side], ctx.settings.placement.rest_arm(side)
    frame = ctx.observe(epoch)
    joints = within_limits(ctx, side, frame.arm_joints(side))
    start, home = arm.forward(joints)[0], arm.forward(rest)[0]
    if np.linalg.norm(start[:3, 3] - home[:3, 3]) > NEAR_HOME_M:
        positions = within_reach(arm, reset_positions(side, start[:3, 3], above))
        ctx.emit({"type": "arm_return_prepared", "side": side, "frame_id": frame.frame_id,
                  "palm_path": points_in(frame.world_torso, positions[RESET_SAMPLES - 1::RESET_SAMPLES])})
        ctx.stage("Returning arm along the reset curve")
        turn = Slerp([0., 1.], Rotation.from_matrix([start[:3, :3], home[:3, :3]]))
        path, unsolved = [joints], 0
        for position, rotation in zip(positions, turn(np.linspace(0., 1., len(positions))).as_matrix()):
            target = transform(position, rotation)  # no restarts: a restart could flip the arm mid-curve
            solved, position_only = arm.solve(target, path[-1], restarts=False), False
            if solved is None:
                solved, position_only = arm.solve(target, path[-1], position_only=True, restarts=False), True
            if solved is not None:  # the elbow keeps near its rest place
                solved = arm.settle_toward(solved, target, rest, position_only=position_only)
            if solved is None or arm_in_torso(arm, solved):
                unsolved += 1
            else:
                path.append(solved)
        if unsolved:
            ctx.emit({"type": "reset_curve_unsolved", "side": side,
                      "reason": f"{unsolved} of {len(positions)} curve samples did not solve"})
        if len(path) > 1:
            follow_rows(ctx, epoch, side, retime_path(np.array(path), ctx.settings.physics.arm_rate,
                                                      ctx.settings.servo.tick)[1:])
    ctx.stage("Settling into rest")
    step_to(ctx, epoch, side, rest, None)
    joints = settle_joints(ctx, epoch, side, rest)
    error = np.abs(joints - rest)
    if np.max(error) > ctx.settings.placement.rest_tolerance:
        ctx.emit({"type": "arm_rest_missed", "side": side, "joints": joints,
                  "reason": f"joint {int(np.argmax(error))} ended {np.max(error):.2f} rad from rest"})
        return False
    return True


def settle_joints(ctx, epoch, side, target):
    """Hold the joint TARGET until the arm is within REST_SETTLED_RAD of it, stalls for
    STALL_SECONDS, or arm_timeout passes. Returns the measured joints."""
    state = ctx.measure(epoch)
    start = closest_time = state.time
    closest = np.inf
    while state.time - start < ctx.settings.motion.arm_timeout:
        error = np.max(np.abs(state.arm_joints(side) - target))
        if error <= REST_SETTLED_RAD:
            break
        if error < closest - STALL_PROGRESS_RAD:
            closest, closest_time = error, state.time
        elif state.time - closest_time > STALL_SECONDS:
            break
        ctx.check(epoch)
        ctx.actions.arm_target(epoch, side, target)
        ctx.advance(epoch, ctx.settings.servo.tick)
        state = ctx.measure(epoch)
    return state.arm_joints(side)


def arm_in_torso(arm, joints):
    """Whether the elbow, wrist or hand at JOINTS lies inside the torso's box."""
    names = [name for name in arm.collision.links if name in arm.collision.bounds]
    return any(any(part in name for part in ARM_LINKS) and overlap((box[0], box[1], box[2]), TORSO_BOX)
               for name, box in zip(names, arm.collision.boxes(arm, joints)))


def within_reach(arm, positions):
    """POSITIONS with any sample beyond REACH_FRACTION of the arm's reach pulled straight in
    toward the shoulder, so a reset that starts at full stretch still sweeps instead of
    dropping its first samples."""
    offsets = positions - arm.shoulder
    far = np.linalg.norm(offsets, axis=1)
    limit = REACH_FRACTION * arm.reach
    scale = np.where(far > limit, limit / np.maximum(far, 1e-9), 1.0)
    return arm.shoulder + offsets * scale[:, None]


def reset_positions(side, start, above=None):
    """Torso-frame reset curve samples from START, mirrored in y for the right arm, with
    samples inside ABOVE's radius lifted to its height."""
    if side not in ("left", "right"):
        raise ValueError(f"unknown arm side {side!r}")
    mirror = np.array([1., 1. if side == "left" else -1., 1.])
    points = RESET_POINTS.copy()
    points[:3, 2] += RESET_RAISE_M
    points[1, 2] += RESET_SECOND_RAISE_M
    points[0, 2] = max(points[0, 2] + RESET_FIRST_RAISE_M, start[2])
    points *= mirror
    positions = catmull_rom(np.vstack((start, points)), RESET_SAMPLES)
    if above is not None:
        center, radius, height = above
        over = np.linalg.norm(positions[:, :2] - center, axis=1) < radius
        positions[over, 2] = np.maximum(positions[over, 2], height)
    return positions


def catmull_rom(knots, samples):
    """SAMPLES points per leg of the Catmull-Rom curve through KNOTS, excluding the first knot."""
    padded = np.vstack((2 * knots[0] - knots[1], knots, 2 * knots[-1] - knots[-2]))
    points = []
    for p0, p1, p2, p3 in zip(padded, padded[1:], padded[2:], padded[3:]):
        for u in np.linspace(0., 1., samples + 1)[1:]:
            points.append(.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                                + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3))
    return np.array(points)


def close_or_open(ctx, epoch, side, closure, seconds):
    """Command the hand to CLOSURE (0 is open) for SECONDS and return the frame after."""
    for _ in range(int(np.ceil(seconds / ctx.settings.servo.tick))):
        ctx.check(epoch)
        ctx.actions.grip(epoch, side, closure)
        ctx.advance(epoch, ctx.settings.servo.tick)
    return ctx.observe(epoch)
