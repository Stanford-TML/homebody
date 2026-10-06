"""Visible RGB-D rendered in captured torso coordinates, ahead up and left left. Unobserved space stays white (grey in the cockpit)."""
import numpy as np
from PIL import Image, ImageDraw

from homebody.helpers.geometry import inverse, points_in, unproject
from homebody.primitives.observations import ObservationError


def size_px(settings):
    return tuple(round((bounds[1] - bounds[0]) * settings.pixels_per_metre)
                 for bounds in (settings.left, settings.ahead))


def to_px(x_ahead, y_left, settings):
    width, height = size_px(settings)
    return ((settings.left[1] - np.asarray(y_left)) / np.ptp(settings.left) * (width - 1),
            (settings.ahead[1] - np.asarray(x_ahead)) / np.ptp(settings.ahead) * (height - 1))


def to_metres(u, v, settings):
    width, height = size_px(settings)
    return (settings.ahead[1] - np.asarray(v) / (height - 1) * np.ptp(settings.ahead),
            settings.left[1] - np.asarray(u) / (width - 1) * np.ptp(settings.left))


def normalized_click(point, settings):
    """A [0, 1000] click as captured-torso (ahead, left) metres, or ObservationError when malformed."""
    if (not isinstance(point, (list, tuple)) or len(point) != 2 or
            any(type(value) not in (int, float) or not 0 <= value <= 1000 for value in point)):
        raise ObservationError("Top-down selection requires two finite coordinates in [0,1000]")
    width, height = size_px(settings)
    return np.array(to_metres(point[0] * (width - 1) / 1000,
                             point[1] * (height - 1) / 1000, settings))


def metadata(settings):
    return {"frame": "captured torso: x ahead, y left, z up",
            "image_size": list(size_px(settings)), "ahead_metres": list(settings.ahead),
            "left_metres": list(settings.left), "height_metres": list(settings.height),
            "orientation": "image up is ahead; image left is robot left",
            "coverage": "visible RGB-D points only; white may be unobserved, not free",
            "click": "normalized [0,1000] image [u,v] selects torso [x,y], not height"}


def cloud_image(frame, settings, *, background=(255, 255, 255)):
    """The point cloud in camera colours, highest point per pixel."""
    valid = np.isfinite(frame.depth) & (frame.depth > 0)
    points = points_in(inverse(frame.world_torso) @ frame.world_camera,
                       unproject(frame.depth, frame.intrinsics, valid))
    colors = frame.rgb[valid]
    inside = np.ones(len(points), dtype=bool)
    for axis, bounds in enumerate((settings.ahead, settings.left, settings.height)):
        inside &= (points[:, axis] >= bounds[0]) & (points[:, axis] <= bounds[1])
    points, colors = points[inside], colors[inside]
    width, height = size_px(settings)
    image = np.empty((height, width, 3), np.uint8)
    image[:] = background
    if len(points):
        u, v = to_px(points[:, 0], points[:, 1], settings)
        u, v = np.floor(u).astype(int), np.floor(v).astype(int)
        order = np.argsort(-points[:, 2], kind="stable")
        _, selected = np.unique(v[order] * width + u[order], return_index=True)
        winners = order[selected]
        image[v[winners], u[winners]] = colors[winners]
    return image


def render(frame, settings, *, palm_torso=None, hand=None, background=(255, 255, 255)):
    """The top-down image with grid, chest and heading. `palm_torso` adds a palm marker."""
    image = Image.fromarray(cloud_image(frame, settings, background=background))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    grid = (214, 214, 214) if background == (255, 255, 255) else (190, 190, 190)
    ink, robot = (100, 100, 100), (0, 90, 200)
    for axis, bounds in enumerate((settings.ahead, settings.left)):
        first = np.ceil(bounds[0] / settings.grid_metres) * settings.grid_metres
        for value in np.arange(first, bounds[1] + 1e-9, settings.grid_metres):
            u, v = to_px(value, 0., settings) if axis == 0 else to_px(0., value, settings)
            draw.line(((0, v), (width - 1, v)) if axis == 0 else ((u, 0), (u, height - 1)), fill=grid)
            if np.isclose(value / settings.label_metres, round(value / settings.label_metres)):
                draw.text((4, v - 11) if axis == 0 else (u + 2, height - 14),
                          f"{value:+.1f} m", fill=ink)
    ahead, across = settings.chest_size
    upper = to_px(ahead / 2, across / 2, settings)
    lower = to_px(-ahead / 2, -across / 2, settings)
    draw.rectangle((*upper, *lower), fill=(210, 225, 240), outline=robot, width=3)
    origin, tip = to_px(0., 0., settings), to_px(ahead, 0., settings)
    draw.line((origin, tip), fill=robot, width=4)
    draw.polygon((tip, (tip[0] - 7, tip[1] + 12), (tip[0] + 7, tip[1] + 12)), fill=robot)
    draw.text((tip[0] - 20, tip[1] - 14), "ahead", fill=robot)
    for side, sign in (("L", 1), ("R", -1)):
        u, v = to_px(ahead / 2, sign * across / 2, settings)
        draw.text((u - 3, v - 15), side, fill=robot)
    if palm_torso is not None:
        palm = np.asarray(palm_torso, dtype=float)
        if palm.shape != (3,) or not np.isfinite(palm).all():
            raise ValueError("Measured palm marker requires three finite torso coordinates")
        u, v = to_px(palm[0], palm[1], settings)
        if 0 <= u < width and 0 <= v < height:
            green = (110, 190, 110)
            draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=green, width=3)
            draw.text((min(u + 12, width - 130), v - 8),
                      f"{hand or ''} palm z {palm[2]:+.2f} m", fill=green)
    draw.text((width - 200, 4), "up = ahead; left = robot left", fill=ink)
    return np.asarray(image)
