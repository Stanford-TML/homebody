"""Render only declared camera and static-navigation observations for the agent/UI."""
import json

import numpy as np
from PIL import Image, ImageDraw


def depth_preview(depth: np.ndarray, far: float = 4.0) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    value = np.zeros(depth.shape, dtype=np.uint8)
    value[valid] = np.clip(255 * (1 - depth[valid] / far), 1, 255).astype(np.uint8)
    return np.repeat(value[..., None], 3, axis=2)


def label_preview(labels: np.ndarray) -> np.ndarray:
    """Every visible segment in its own stable colour; label 0 (unlabelled) stays dark."""
    hue = (labels.astype(np.float64) * 0.618034) % 1.0
    k = (hue * 6).astype(int) % 6
    f = hue * 6 - np.floor(hue * 6)
    q, t = 1 - f, f
    r = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [1, q, 0, 0, t, 1])
    g = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [t, 1, 1, q, 0, 0])
    b = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [0, 0, t, 1, 1, q])
    pixels = (np.stack((r, g, b), axis=-1) * 200 + 40).astype(np.uint8)
    pixels[labels <= 0] = 35
    return pixels


def map_preview(frame) -> np.ndarray:
    """The static map with the robot's pose as an arrow."""
    nav = frame.navigation
    image = Image.fromarray(np.where(nav.occupied[..., None], [65, 73, 83], [235, 235, 230]).astype(np.uint8))
    world_to_pixel = np.linalg.inv(nav.T_map_px)

    def px(xy):
        point = world_to_pixel @ np.r_[np.asarray(xy, dtype=float)[:2], 1]
        return point[:2] / point[2]

    pose = frame.base_pose
    (x, y), (tx, ty) = px(pose), px([pose[0] + 0.4 * np.cos(pose[2]), pose[1] + 0.4 * np.sin(pose[2])])
    draw = ImageDraw.Draw(image)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(10, 125, 180))
    draw.line((x, y, tx, ty), fill=(10, 125, 180), width=3)
    return np.asarray(image)


def describe(frame, topdown_settings=None) -> dict:
    result = {
        "epoch": frame.epoch, "sequence": frame.sequence, "simulation_seconds": frame.time,
        "rgb_size": [frame.rgb.shape[1], frame.rgb.shape[0]],
        "map_size": [frame.navigation.occupied.shape[1], frame.navigation.occupied.shape[0]],
        "base_pose_x_y_yaw": frame.base_pose.tolist(),
        "joint_positions": frame.joints.tolist(),
        "hands": {side: {"closure": hand.closure,
                          "effort_fraction": hand.effort_fraction,
                          "closure_gap": hand.closure_gap,
                          "finger_order": ["thumb", "index", "middle"],
                          "finger_effort_fraction": hand.finger_effort_fraction.tolist(),
                          "finger_closure_gap": hand.finger_closure_gap.tolist()}
                  for side, hand in frame.hands.items()},
        "camera_intrinsics": frame.intrinsics.tolist(),
        "world_from_camera": frame.world_camera.tolist(),
        "map_world_xy_from_pixel": frame.navigation.T_map_px.tolist(),
        "map_resolution_metres": frame.navigation.resolution,
        "click_images": {"point_normalized_1000": 1,
                         "goal_map_1000": 3, "facing_map_1000": 3},
        "images": ["current head RGB", "current metric depth preview: white=near, black=invalid",
                   "static occupancy: dark=obstacle, light=free, blue=robot and heading"],
    }
    if topdown_settings is not None:
        from .topdown import metadata
        result["topdown"] = metadata(topdown_settings)
        result["topdown"]["world_from_torso"] = frame.world_torso.tolist()
        result["click_images"]["release_point_1000"] = 4
        result["images"].append("current visible RGB-D top-down in captured torso frame; white may be unobserved")
    return result


def rounded(value, digits=3):
    """VALUE with every float, inside dicts and lists too, rounded to DIGITS decimals."""
    if isinstance(value, float):
        return round(value, digits) + 0.0
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    return value


def model_observation(frame) -> dict:
    """The prompt's numbers: the base pose and the affine map from world [x, y, 1] to a [0, 1000] map click."""
    height, width = frame.navigation.occupied.shape
    click = np.diag([1000 / width, 1000 / height, 1.]) @ np.linalg.inv(frame.navigation.T_map_px)
    return rounded({"base_pose": frame.base_pose.tolist(), "map_click_from_world": click[:2].tolist()})


def annotations(prior) -> str:
    """The initial semantic prior as prompt lines, one per entity."""
    lines = []
    for entity in prior.get("entities", ()):
        name = entity.get("label", "").rstrip(".")
        if entity.get("portable"):
            name += " (portable)"
        if "reference_anchor_ws_map_m" in entity:
            name += " at " + json.dumps(rounded(entity["reference_anchor_ws_map_m"][:2], 2))
        details = " ".join(entity[key] for key in ("visual_description", "location_description") if key in entity)
        lines.append(f"- {name}: {details}" if details else f"- {name}")
    return "\n".join(lines) or "none"


def semantic_prior(scene):
    """The scan's labels, descriptions, anchors and portable flags for every entity."""
    source = json.loads((scene / "semantics.json").read_text())
    keys = ("label", "reference_anchor_ws_map_m", "visual_description", "location_description")
    entries = []
    for item in source.get("entities", []):
        entry = {key: item[key] for key in keys if key in item}
        if item.get("portable_candidate"):
            entry["portable"] = True
        entries.append(entry)
    return {"entities": entries}
