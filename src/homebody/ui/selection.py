"""Human overlays derived from the same captured RGB-D selection as the skill."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from homebody.helpers.geometry import inverse, points_in
from homebody.primitives.observations import ObservationError
from homebody.ui.theme import ACCENT, DOT, LINE, MASK, PLAN, RELEASE, RING
from homebody.vlm.topdown import normalized_click, render, to_px


class Pen:
    """Draws on a copy of PIXELS with the theme's marks, sized for the image's width."""
    def __init__(self, pixels):
        self.image = Image.fromarray(np.asarray(pixels, dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.image)
        self.unit = self.image.width / 640

    def size(self, pixels):
        return max(1, round(pixels * self.unit))

    def ring(self, uv, colour):
        u, v, r = *uv, self.size(RING)
        backing = self.size(LINE + 4)
        outer = r + max(1, backing // 2)
        self.draw.ellipse((u - outer, v - outer, u + outer, v + outer),
                          outline=(255, 250, 244), width=backing)
        self.draw.ellipse((u - r, v - r, u + r, v + r), outline=colour, width=self.size(LINE))

    def dot(self, uv, colour):
        u, v, r = *uv, self.size(DOT)
        self.draw.ellipse((u - r, v - r, u + r, v + r), fill=colour)

    def path(self, uvs, colour):
        points = [tuple(uv) for uv in uvs]
        self.draw.line(points, fill=(255, 250, 244), width=self.size(LINE + 4), joint="curve")
        self.draw.line(points, fill=colour, width=self.size(LINE), joint="curve")

    def pose(self, uv, tip, colour):
        self.path((uv, tip), colour)
        self.dot(uv, colour)

    def pixels(self):
        return np.asarray(self.image)


def requested_release(decision, frame, settings):
    """The captured-torso (ahead, left) a place decision asks for, or None for another skill or an invalid proposal."""
    arguments = decision.arguments
    metric = [arguments.get(key) for key in ("x_ahead_m", "y_left_m")]
    forms = ("release_point_1000" in arguments, "point_normalized_1000" in arguments,
             "x_ahead_m" in arguments or "y_left_m" in arguments)
    if decision.skill != "place" or sum(forms) != 1:
        return None
    try:
        if forms[0]:
            return normalized_click(arguments["release_point_1000"], settings)
        if forms[1]:
            point = frame.point(*frame.normalized_pixel(arguments["point_normalized_1000"]))
            return points_in(inverse(frame.world_torso), point)[:2]
    except ObservationError:
        return None
    if all(type(value) in (int, float) and np.isfinite(value) for value in metric):
        return np.array(metric, dtype=float)
    return None


TOPDOWN_UNOBSERVED = (228, 229, 229)


def decision_topdown(decision, frame, settings, resolved=None):
    """The top-down with the requested point as a ring and, once prepared, the resolved world point as a dot."""
    pen = Pen(render(frame, settings, background=TOPDOWN_UNOBSERVED))
    requested = requested_release(decision, frame, settings)
    if requested is not None:
        pen.ring(to_px(*requested, settings), ACCENT)
    if resolved is not None:
        pen.dot(to_px(*points_in(inverse(frame.world_torso), resolved)[:2], settings), RELEASE)
    return pen.pixels()


def draw_path(frame, pixels, points, colour=PLAN):
    """Draw a world polyline on the ego image."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        return pixels.copy()
    camera = np.linalg.inv(frame.world_camera) @ np.column_stack((points, np.ones(len(points)))).T
    if np.any(camera[2] <= 1e-6):
        return pixels.copy()
    uv = (frame.intrinsics @ (camera[:3] / camera[2]))[:2].T
    pen = Pen(pixels)
    pen.path(uv, colour)
    pen.dot(uv[-1], colour)
    return pen.pixels()


@dataclass(frozen=True)
class Selection:
    frame_id: tuple[int, int]
    pixel: tuple[int, int]
    label: int
    point: np.ndarray

    @classmethod
    def from_decision(cls, decision, frame):
        if decision.skill not in ("pick", "place"):
            return None
        try:
            pixel = frame.normalized_pixel(decision.arguments.get("point_normalized_1000"))
            label = frame.select(*pixel)
            point = frame.point(*pixel)
        except ObservationError:
            return None
        return cls(frame.frame_id, pixel, label, point)

    def overlay(self, frame, pixels):
        """Tint the selected segment. The click ring is drawn only on its captured frame."""
        output = pixels.copy()
        if frame.epoch != self.frame_id[0]:
            return output
        mask = frame.labels == self.label
        output[mask] = np.rint(.55 * output[mask] + .45 * np.array(MASK)).astype(np.uint8)
        if frame.frame_id != self.frame_id:
            return output
        pen = Pen(output)
        pen.ring(self.pixel, MASK)
        return pen.pixels()
