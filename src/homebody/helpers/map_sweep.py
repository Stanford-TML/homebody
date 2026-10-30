"""Conservative box projections against navigation's vertically extruded blocked cells."""
from itertools import product

import numpy as np

from homebody.helpers.collision import projected_box_domain

_CORNERS = np.array(list(product((-1., 1.), repeat=3)))


def projected_free(navigation, boxes, margin=0.0, sweep_margin=0.0):
    """One bool per grown box, including every touched closed cell and map boundary.

    This uses navigation's full-height extrusion assumption, not a claim of measured
    free 3D volume. SWEEP_MARGIN bounds any point's motion between supplied poses.
    """
    if not np.isfinite([margin, sweep_margin]).all() or min(margin, sweep_margin) < 0:
        raise ValueError("Projection margins must be finite and nonnegative")
    if navigation is None:
        return np.zeros(len(boxes), dtype=bool)
    transform = np.asarray(navigation.T_map_px)
    if not np.isfinite(transform).all() or not np.allclose(transform[2], [0., 0., 1.]):
        return np.zeros(len(boxes), dtype=bool)
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return np.zeros(len(boxes), dtype=bool)
    height, width = navigation.occupied.shape
    clear = []
    for box in boxes:
        if (not all(np.isfinite(part).all() for part in box) or
                np.any(np.asarray(box[2]) < 0)):
            clear.append(False)
            continue
        center, rotation, half = box
        corners = ((_CORNERS * (half + (margin + sweep_margin))) @ rotation.T + center)[:, :2]
        pixels = (np.column_stack((corners, np.ones(len(corners)))) @ inverse.T)[:, :2]
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        if (low <= -.5).any() or (high >= [width - .5, height - .5]).any():
            clear.append(False)
            continue
        lo = np.maximum(0, np.ceil(low - .5).astype(int))
        hi = np.minimum([width, height], np.floor(high + .5).astype(int) + 1)
        cells = np.argwhere(navigation.occupied[lo[1]:hi[1], lo[0]:hi[0]])[:, ::-1] + lo
        if not len(cells):
            clear.append(True)
            continue
        # The convex projection is contained in these closed-cell bounds. Only a
        # rectangle containing occupied cells needs the exact polygon separation.
        polygon = projected_box_domain(box, margin + sweep_margin)
        polygon = (np.column_stack((polygon, np.ones(len(polygon)))) @ inverse.T)[:, :2]
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.vstack((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
        projection, centers = polygon @ axes.T, cells @ axes.T
        half = .5 * np.sum(np.abs(axes), axis=1)
        intersects = np.all((centers + half >= projection.min(axis=0) - 1e-10) &
                            (centers - half <= projection.max(axis=0) + 1e-10), axis=1)
        clear.append(not intersects.any())
    return np.asarray(clear, dtype=bool)
