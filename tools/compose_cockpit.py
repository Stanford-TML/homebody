"""Offline cockpit video for a recorded session, laid out like the live console.

Replays events.jsonl through the console's Overlays. A session without frames/ is replayed
from its videos and decision images instead.
"""
import argparse
import json
import re
import textwrap
from dataclasses import MISSING, fields
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from homebody.backends.scene import DEFAULT_SCENE
from homebody.helpers.kinematics import Arm
from homebody.motion.carry import RecoveryPlan
from homebody.primitives.observations import Frame, HandState, NavigationMap
from homebody.session.config import Topdown
from homebody.skills import navigate, pick, place
from homebody.ui.console import Overlays
from homebody.ui.theme import ACCENT, FACING, GOAL, MASK
from homebody.vlm.provider import Decision
from homebody.vlm.topdown import render, size_px

ROOT = Path(__file__).resolve().parents[1]
PLANS = (pick.Plan, pick.ApproachPlan, place.Plan, navigate.Plan, RecoveryPlan)
EGO_HEIGHT = 480
BACKGROUND, INK, MUTED, HIGHLIGHT = (22, 26, 33), (236, 240, 245), (150, 160, 175), (255, 200, 0)


def load_frame(folder, nav):
    """Rebuild a recorded Frame from rgb.png, sensors.npz and state.json."""
    if not (folder / "state.json").is_file():
        raise FileNotFoundError(f"{folder} holds no recorded frame; the session's frames are incomplete")
    state = json.loads((folder / "state.json").read_text())
    with np.load(folder / "sensors.npz") as saved:
        hands = {side: HandState(saved[f"{side}_hand_joints"], hand["closure"],
                                 hand["finger_effort_fraction"], hand["finger_closure_gap"])
                 for side, hand in state["hands"].items()}
        return Frame(state["epoch"], state["sequence"], state["simulation_seconds"],
                     np.asarray(Image.open(folder / "rgb.png").convert("RGB")), saved["depth"],
                     saved["intrinsics"], saved["world_camera"], saved["world_torso"],
                     saved["base_pose"], saved["joints"], hands, saved["labels"], nav)


def plan(record):
    """A recorded plan as its dataclass again, or None if no plan kind matches."""
    def names(kind, required=False):
        return {item.name for item in fields(kind)
                if not required or (item.default is MISSING and item.default_factory is MISSING)}
    kind = next((k for k in PLANS if names(k, required=True) <= set(record)), None)
    if kind is None:
        return None
    return kind(**{key: plan(value) if key == "drive_plan" else
                   (np.asarray(value[0]), *value[1:]) if key == "released" and value else
                   np.asarray(value, dtype=float) if isinstance(value, list) else value
                   for key, value in record.items() if key in names(kind)})


def replay(events, session, nav, overlays, text):
    """Apply the events in recorded order and yield (frame, caption) for each view."""
    for row in events:
        kind, event = row["kind"], row.get("event", {})
        if kind == "task_started":
            text["task"] = row["task"]
        elif kind in ("decision", "scripted_request"):
            record = row.get("decision", row)
            decision = Decision(record.get("text", ""), record["skill"], record["arguments"])
            frame_id = (row["epoch"], row["sequence"]) if kind == "decision" else row["frame"]
            captured = load_frame(session / "frames" / f"{frame_id[0]:03d}-{frame_id[1]:08d}", nav)
            overlays.select(decision, captured)
            text.update(decision=decision, stages=[], topdown=None,
                        who="Agent" if kind == "decision" else "Scripted request")
            yield captured, f"Decision view · captured {captured.time:.1f} s"
        elif event.get("type") == "skill_stage":
            text["stages"].append(event["stage"])
        elif event.get("type") == "skill_prepared":
            prepared = plan(event["plan"])
            if prepared is not None:
                overlays.prepared(prepared)
                text["topdown"] = None
        elif event.get("type") == "arm_return_prepared":
            overlays.returning(event)
        elif event.get("type") == "skill_finished":
            result = event["result"]
            text.update(decision=None, stages=[], result=" · ".join(
                (result["skill"].capitalize(), result["code"], result["message"])))
            overlays.clear()
        elif kind == "frame":
            frame = load_frame(session / row["path"], nav)
            if overlays.observe(frame):
                text["decision"] = None
            yield frame, f"Ego RGB · {frame.time:.1f} s"


def panel(pixels, size, caption, font):
    """PIXELS fitted into SIZE on the cockpit background, captioned top-left."""
    image = Image.new("RGB", size, BACKGROUND)
    picture = ImageOps.contain(Image.fromarray(pixels), size)
    image.paste(picture, ((size[0] - picture.width) // 2, (size[1] - picture.height) // 2))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, font.getlength(caption) + 12, font.size + 8), fill=BACKGROUND)
    draw.text((6, 4), caption, fill=INK, font=font)
    return image


def strip(draw, box, text, font, small):
    """Draw the task, stages, last result and active decision into BOX."""
    left, top, right, bottom = box
    decision, stages = text["decision"], text["stages"]
    paragraphs = [(f"Task · {text['task']}", INK, 2, font),
                  ("Stage · " + (" < ".join(stages[:-5:-1]) if stages else "none"),
                   HIGHLIGHT if stages else MUTED, 1, font),
                  (f"Last result · {text['result'] or 'none yet'}", MUTED, 2, font)]
    if decision is None:
        paragraphs.append(("No active decision", MUTED, 1, font))
    else:
        paragraphs += [(f"{text['who']} · {decision.skill} {json.dumps(decision.arguments)}",
                        INK, 2, font), (decision.text, INK, 99, small)]
    for paragraph, colour, limit, face in paragraphs:
        step, chars = face.size + 4, max(20, int((right - left) / face.getlength("n")))
        room = (bottom - top) // step
        if room < 1:
            return
        for line in textwrap.wrap(paragraph, chars, max_lines=min(limit, room), placeholder=" …"):
            draw.text((left, top), line, fill=colour, font=face)
            top += step


def cockpit(overlays, frame, caption, text):
    """One full cockpit view as an image."""
    scale = max(1, round(EGO_HEIGHT / frame.depth.shape[0]))
    ego_height, ego_width = scale * frame.depth.shape[0], scale * frame.depth.shape[1]
    half = (ego_width // 2, ego_height // 2)
    font = ImageFont.load_default(size=max(13, ego_height // 32))
    small = ImageFont.load_default(size=max(11, ego_height // 40))
    map_rows, map_columns = frame.navigation.occupied.shape
    top_columns, top_rows = size_px(overlays.topdown_settings)
    map_width = round((ego_height + half[1]) * map_columns / map_rows)
    top_width = round(ego_height * top_columns / top_rows)
    rgb, depth, labels = overlays.ego(frame)
    active = overlays.decision is not None
    if active and text["topdown"] is None:
        text["topdown"] = overlays.topdown()
    top = text["topdown"] if active else render(frame, overlays.topdown_settings)
    canvas = Image.new("RGB", (-(-(ego_width + map_width + top_width) // 16) * 16,
                               -(-(ego_height + half[1]) // 16) * 16), BACKGROUND)
    for pixels, size, origin, label in (
            (rgb, (ego_width, ego_height), (0, 0), caption),
            (depth, half, (0, ego_height), "Depth · white is near"),
            (labels, half, (half[0], ego_height), "Segmentation · skills only, not shown to the model"),
            (overlays.map(frame), (map_width, ego_height + half[1]), (ego_width, 0),
             "Map · route, goal, facing, stance"),
            (top, (top_width, ego_height), (ego_width + map_width, 0),
             f"Decision top-down · captured {overlays.captured.time:.1f} s" if active
             else "Top-down · current view, no active decision")):
        canvas.paste(panel(pixels, size, label, font), origin)
    strip(ImageDraw.Draw(canvas), (ego_width + map_width + 12, ego_height + 8,
                                   canvas.width - 12, canvas.height - 4), text, font, small)
    return canvas


# click argument: (index among the decision's four images, ring colour)
CLICKS = {"point_normalized_1000": (0, MASK), "goal_map_1000": (2, GOAL), "facing_map_1000": (2, FACING),
          "release_point_1000": (3, ACCENT)}
WORLD_CLICKS = {"goal_xy_m": "goal_map_1000", "facing_xy_m": "facing_map_1000"}
MAP_TRANSFORM = re.compile(r'"map_click_from_world": (\[\[[^]]*\], \[[^]]*\]\])')
DECISION_IMAGES = ("Decision RGB", "Decision depth", "Decision map", "Decision top-down")


class VideoCursor:
    """Frames of one recorded video, read forward to any index."""

    def __init__(self, path):
        self.reader = imageio_ffmpeg.read_frames(str(path)) if path.is_file() else None
        self.size, self.index, self.pixels = None, -1, None
        if self.reader is not None:
            self.size = next(self.reader)["size"]

    def at(self, index):
        while self.reader is not None and self.index < index:
            data = next(self.reader, None)
            if data is None:
                break
            self.index += 1
            self.pixels = np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)
        return self.pixels


def clicks(decision, folder):
    """The decision's clicks, with world goals converted to map clicks via the prompt's transform."""
    found = dict(decision.arguments)
    prompt = folder / "prompt.txt"
    match = MAP_TRANSFORM.search(prompt.read_text()) if prompt.is_file() else None
    for world, click in WORLD_CLICKS.items():
        point = found.get(world)
        if match and isinstance(point, list) and len(point) == 2 and click not in found:
            found[click] = (np.asarray(json.loads(match.group(1))) @ [*point, 1.]).tolist()
    return found


def ringed(path, arguments, image):
    """Decision image IMAGE with each of the model's clicks on it ringed."""
    picture = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(picture)
    radius = max(6, picture.width // 40)
    for key, (index, colour) in CLICKS.items():
        point = arguments.get(key)
        if index == image and isinstance(point, list) and len(point) == 2:
            x, y = point[0] / 1000 * picture.width, point[1] / 1000 * picture.height
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=3)
    return np.asarray(picture)


def from_videos(events, session):
    """Yield (ego, room, decision views, caption, text) from the session's videos and images."""
    ego, room = VideoCursor(session / "observations.mp4"), VideoCursor(session / "room.mp4")
    text = {"task": "", "decision": None, "who": "Agent", "stages": [], "result": "", "topdown": None}
    shown = None
    for row in events:
        kind, event = row["kind"], row.get("event", {})
        if kind == "task_started":
            text["task"] = row["task"]
        elif kind == "observer_frame" and row.get("stream") == "room":
            room.at(row["video_frame"])
        elif kind == "decision":
            record = row["decision"]
            decision = Decision(record.get("text", ""), record["skill"], record["arguments"])
            folder = next(session.glob(f"tasks/*/decisions/step-{row['step']:03d}"), None)
            images = [] if folder is None else sorted(folder.glob("observation-*.png"))
            arguments = {} if folder is None else clicks(decision, folder)
            shown = [ringed(path, arguments, index) for index, path in enumerate(images)]
            text.update(decision=decision, stages=[])
            if shown:
                yield shown[0], room.pixels, shown[2:], "Decision view · the image the model clicked", text
        elif event.get("type") == "skill_stage":
            text["stages"].append(event["stage"])
        elif event.get("type") == "skill_finished":
            result = event["result"]
            text.update(decision=None, stages=[], result=" · ".join(
                (result["skill"].capitalize(), result["code"], result["message"])))
        elif kind == "frame" and row.get("video_frame") is not None:
            pixels = ego.at(row["video_frame"])
            if pixels is not None:
                yield pixels, room.pixels, shown[2:] if shown else [], \
                    f"Ego RGB · {row['simulation_time']:.1f} s", text


def light_cockpit(ego, room, decision_views, caption, text):
    """One cockpit view built without recorded frames."""
    font, small = ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    ego_size, room_size, lower = (640, EGO_HEIGHT), (854, EGO_HEIGHT), 304
    canvas = Image.new("RGB", (1504, EGO_HEIGHT + lower), BACKGROUND)
    canvas.paste(panel(ego, ego_size, caption, font), (0, 0))
    if room is not None:
        canvas.paste(panel(room, room_size, "Room · 3D view", font), (640, 0))
    for index, pixels in enumerate(decision_views[:2]):
        canvas.paste(panel(pixels, (lower, lower), DECISION_IMAGES[2 + index], font), (index * lower, EGO_HEIGHT))
    strip(ImageDraw.Draw(canvas), (2 * lower + 12, EGO_HEIGHT + 8, canvas.width - 12, canvas.height - 4),
          text, font, small)
    return canvas




def write_video(images, output, fps, folder=None):
    video, count = None, 0
    try:
        for image in images:
            if folder is not None:
                image.save(folder / f"{count:06d}.png", compress_level=1)
            if video is None:
                video = imageio_ffmpeg.write_frames(str(output), image.size, fps=fps, codec="libx264",
                                                    pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                                    ffmpeg_log_level="error")
                video.send(None)
            video.send(np.asarray(image))
            count += 1
    finally:
        if video is not None:
            video.close()
    return count


def compose(session, output=None, fps=2.0, scene=None, frames=False):
    """Write the cockpit video for SESSION; return the video path and frame count."""
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().split("\n")[:-1]]
    output = output or session / "cockpit.mp4"
    folder = output.with_suffix("") if frames else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.png"):
            stale.unlink()
    if not (session / "frames").is_dir():
        if not (session / "observations.mp4").is_file():
            raise FileNotFoundError(f"{session} has neither frames/ nor observations.mp4")
        views = (light_cockpit(*view) for view in from_videos(events, session))
        return output, write_video(views, output, fps, folder)
    recorded = [Path(row["path"]) for row in events if row["kind"] == "scene_package"]
    scene = scene or next((path for path in recorded if (path / "map.npz").is_file()),
                          ROOT / DEFAULT_SCENE)
    with np.load(scene / "map.npz") as saved:
        nav = NavigationMap(saved["occupancy"], saved["T_map_px"],
                            float(np.linalg.norm(saved["T_map_px"][:2, 0])))
    saved = json.loads((session / "provenance.json").read_text())["settings"].get("topdown", {})
    settings = Topdown(**{key: tuple(value) if isinstance(value, list) else value
                          for key, value in saved.items()})
    urdf = ROOT / "assets/robot/g1_29dof_with_hand.urdf"
    overlays = Overlays(settings, {side: Arm(urdf, side) for side in ("left", "right")})
    text = {"task": "", "decision": None, "who": "", "stages": [], "result": "", "topdown": None}
    views = (cockpit(overlays, frame, caption, text)
             for frame, caption in replay(events, session, nav, overlays, text))
    return output, write_video(views, output, fps, folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="A recorded session-* folder")
    parser.add_argument("--output", type=Path, help="Video (default SESSION/cockpit.mp4)")
    parser.add_argument("--fps", type=float, default=2.0, help="2 is real time (0.5 s frames)")
    parser.add_argument("--scene", type=Path, help="Scene package with map.npz (default: recorded)")
    parser.add_argument("--frames", action="store_true", help="Also keep every view as a PNG")
    args = parser.parse_args(argv)
    try:
        video, count = compose(args.session.resolve(), args.output, args.fps, args.scene, args.frames)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"{count} cockpit frames: {video}")


if __name__ == "__main__":
    main()
