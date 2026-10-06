"""Append-only evidence bundles; failures retain their original sensor and provider data."""
import hashlib
import json
import platform
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from homebody.vlm.observations import depth_preview, describe, map_preview
from homebody.vlm.topdown import render


def revision(root):
    """The checkout's git revision, or None outside a git checkout."""
    if not (Path(root) / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def directory_digest(directory: Path, suffixes: set[str] | None = None) -> str:
    """Hash relative paths and file bytes, including local edits not in Git HEAD."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or (suffixes is not None and path.suffix not in suffixes):
            continue
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_digest(source: Path) -> str:
    """Hash the runnable first-party Python and prompt text."""
    return directory_digest(source, {".py", ".md"})


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n")


def write_frame(folder, frame, rgb, depth, nav, topdown):
    """Write a frame's images, sensors and state files."""
    Image.fromarray(frame.rgb).save(rgb)
    Image.fromarray(depth_preview(frame.depth)).save(depth)
    Image.fromarray(map_preview(frame)).save(nav)
    np.savez_compressed(folder / "sensors.npz", depth=frame.depth, labels=frame.labels,
                        intrinsics=frame.intrinsics, world_camera=frame.world_camera,
                        world_torso=frame.world_torso, base_pose=frame.base_pose,
                        joints=frame.joints,
                        **{f"{side}_hand_joints": hand.joints for side, hand in frame.hands.items()})
    write_json(folder / "state.json", describe(frame, topdown))


class Recorder:
    def __init__(self, root: Path, settings, output: Path):
        self.settings = settings
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = output / f"session-{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()
        self._events = (self.path / "events.jsonl").open("a", buffering=1)
        self._physics = None
        self._pending_physics = None
        self._videos = {}
        self._video_failed = set()
        self._stream_frames = Counter()
        self._frames = {}
        self._writes = {}
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-writer")
        self._task_count = 0
        self._task_path = None
        packages = {}
        for name in ("numpy", "mujoco", "torch", "viser", "pillow", "imageio-ffmpeg"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        settings_dict = asdict(settings)
        source = root / "src/homebody"
        if not source.is_dir():
            source = Path(__file__).resolve().parents[1]
        write_json(self.path / "provenance.json", {
            "schema_version": 1, "backend": "mujoco", "created_utc": stamp, "revision": revision(root),
            "source_sha256": source_digest(source),
            "assets_sha256": directory_digest(root / "assets") if (root / "assets").is_dir() else None,
            "settings": settings_dict, "packages": packages, "python": platform.python_version(),
            "model": settings.agent.model, "model_id": settings.agent.model_id,
            "provider": settings.agent.provider,
            "video": ("each video plays its samples at their recording rate (settings.recording); "
                      "every frame's simulation time is in events.jsonl"),
        })

    @property
    def recording(self) -> bool:
        return self._task_path is not None

    def event(self, kind: str, **fields):
        self._events.write(json.dumps({"kind": kind, "wall_time": time.time(), **fields},
                                      default=json_value, allow_nan=False) + "\n")

    def start_task(self, task: str, epoch: int | None) -> Path:
        if self._task_path is not None:
            raise RuntimeError("Another task is already recording")
        path = self.path / "tasks" / f"task-{self._task_count:03d}"
        self._task_count += 1
        path.mkdir(parents=True)
        self._task_path = path
        self._physics = (path / "physics.jsonl").open("a", buffering=1)
        self._pending_physics = None
        write_json(path / "task.json", {"task": task, "epoch": epoch, "started_unix": time.time()})
        self.event("task_started", task=task, epoch=epoch, path=path.relative_to(self.path))
        return path

    def setup_failure(self, task: str, reason: str):
        self.start_task(task, epoch=None)
        self.end_task({"status": "setup_error", "reason": reason, "steps": 0, "model_claim": None})

    def physics(self, state: dict):
        """Coalesce same-instant callbacks without erasing any failure evidence."""
        if self._physics is None:
            return
        current = json.loads(json.dumps(state, default=json_value, allow_nan=False))
        previous = self._pending_physics
        if previous is not None and (previous["epoch"], previous["time"]) == (
                current["epoch"], current["time"]):
            for flag in ("physics_valid", "robot_upright"):
                current[flag] = previous[flag] and current[flag]
            current["warnings"] = previous["warnings"] + [
                warning for warning in current["warnings"] if warning not in previous["warnings"]]
        else:
            self._flush_physics()
        self._pending_physics = current

    def _flush_physics(self):
        if self._pending_physics is not None:
            self._physics.write(json.dumps(self._pending_physics, allow_nan=False) + "\n")
            self._pending_physics = None

    def end_task(self, outcome: dict):
        if self._task_path is None:
            return
        self._flush_physics()
        self._physics.close()
        self._physics = None
        path, self._task_path = self._task_path, None
        write_json(path / "outcome.json", {**outcome, "finished_unix": time.time(),
                                          "independently_scored": False})
        self.event("task_finished", **outcome)

    def frame(self, frame) -> tuple[Path, Path, Path]:
        key = (frame.epoch, frame.sequence)
        self._raise_failed_write()
        if key in self._frames:
            return self._frames[key]
        folder = self.path / "frames" / f"{frame.epoch:03d}-{frame.sequence:08d}"
        folder.mkdir()
        rgb, depth, nav = (folder / f"{name}.png" for name in ("rgb", "depth_preview", "map"))
        self._writes[key] = self._writer.submit(write_frame, folder, frame, rgb, depth, nav,
                                                self.settings.topdown)
        self.event("frame", epoch=frame.epoch, sequence=frame.sequence,
                   simulation_time=frame.time, video_frame=len(self._frames),
                   path=folder.relative_to(self.path))
        self._frames[key] = (rgb, depth, nav)
        self._append_video(frame.rgb, "observations", 1 / self.settings.recording.frame_period)
        return self._frames[key]

    def flush(self):
        """Wait until every recorded frame's files are on disk; a failed write raises."""
        for write in self._writes.values():
            write.result()

    def _raise_failed_write(self):
        """A frame write that already failed fails the next recording, not only close."""
        for write in self._writes.values():
            if write.done() and write.exception() is not None:
                raise write.exception()

    def agent_images(self, frame):
        """Capture the decision's extra depth projection once, from that same frame."""
        images = self.frame(frame)
        self._writes[(frame.epoch, frame.sequence)].result()
        path = images[0].parent / "topdown.png"
        if not path.exists():
            Image.fromarray(render(frame, self.settings.topdown)).save(path)
        return (*images, path)

    def observer(self, rgb, *, epoch: int, simulation_time: float, fps: float, stream="room"):
        """Record a human-only camera frame as an event and a video frame. The room stream also keeps PNGs."""
        index, path = self._stream_frames[stream], None
        self._stream_frames[stream] += 1
        if stream == "room":
            (self.path / "observer").mkdir(exist_ok=True)
            path = Path("observer") / f"{index:08d}.png"
            Image.fromarray(rgb).save(self.path / path)
        self.event("observer_frame", epoch=epoch, simulation_time=simulation_time, stream=stream,
                   video_frame=index, path=path, model_visible=False)
        self._append_video(rgb, stream, fps)

    def _append_video(self, rgb, stream, fps):
        if stream in self._video_failed:
            return
        try:
            if stream not in self._videos:
                video = imageio_ffmpeg.write_frames(
                    str(self.path / f"{stream}.mp4"), (rgb.shape[1], rgb.shape[0]), fps=fps,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2)
                video.send(None)
                self._videos[stream] = video
            self._videos[stream].send(np.ascontiguousarray(rgb))
        except (OSError, RuntimeError) as exc:
            self._video_failed.add(stream)
            self.event("video_unavailable", stream=stream, error=str(exc))
            video = self._videos.pop(stream, None)
            if video is not None:
                video.close()

    def close(self):
        """End an open task, drain the frame writer and close every file. A failed frame write is raised after closing. Idempotent."""
        if self._events.closed:
            return
        failed = []
        try:
            if self._task_path is not None:
                self.end_task({"status": "interrupted", "reason": "Session closed during task"})
            self._writer.shutdown(wait=True)
            failed = [write.exception() for write in self._writes.values() if write.exception() is not None]
            for error in failed:
                self.event("frame_write_failed", error=str(error))
        finally:
            for video in self._videos.values():
                video.close()
            self._events.close()
        if failed:
            raise failed[0]
