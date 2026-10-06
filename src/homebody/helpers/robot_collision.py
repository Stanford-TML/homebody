"""Bounding boxes that carry a convex hull, and the GJK overlap test between them."""
from functools import cached_property
from itertools import combinations, product

import numpy as np


class MeshBox(tuple):
    """A (center, rotation, half) bounding box carrying its convex hull."""
    def __new__(cls, center, rotation, half, shape):
        value = super().__new__(cls, (center, rotation, half))
        value.shape = shape
        return value

    @cached_property
    def vertices(self):
        center, rotation, _ = self
        return self.shape["vertices"] @ rotation.T + center


def vertices(box):
    if isinstance(box, MeshBox):
        return box.vertices
    center, rotation, half = box
    corners = np.array(list(product((-1., 1.), repeat=3))) * half
    return corners @ rotation.T + center


def closest_simplex(points):
    """The point of a simplex closest to the origin, and the face it lies on."""
    best, active, distance = None, None, float("inf")
    for size in range(1, min(4, len(points)) + 1):
        for indices in combinations(range(len(points)), size):
            face = points[list(indices)]
            if size == 1:
                weights = np.ones(1)
            else:
                edges = face[1:] - face[0]
                tail = np.linalg.lstsq(edges.T, -face[0], rcond=None)[0]
                weights = np.r_[1 - np.sum(tail), tail]
                if np.min(weights) < -1e-12:
                    continue
                weights = np.maximum(weights, 0.)
                weights /= np.sum(weights)
            point = weights @ face
            squared = point @ point
            if squared < distance:
                best, active, distance = point, face[weights > 0.], squared
    return best, active


def mesh_overlap(first, second, margin=0.):
    """GJK overlap of two boxes' hulls within MARGIN: True unless a separating plane is
    found, and True when neither box carries a hull."""
    if not isinstance(first, MeshBox) and not isinstance(second, MeshBox):
        return True
    va, vb = vertices(first) - first[0], vertices(second) - first[0]
    rounding = 64 * np.finfo(float).eps * max(1., np.max(np.abs(va)), np.max(np.abs(vb)))
    direction = np.array([1., 0., 0.])
    simplex = np.empty((0, 3))
    for _ in range(32):
        length = np.linalg.norm(direction)
        if length <= rounding:
            return True
        direction = direction / length
        support = va[np.argmax(va @ direction)] - vb[np.argmin(vb @ direction)]
        support += margin * direction
        if support @ direction < -rounding:
            return False
        if len(simplex) and np.min(np.linalg.norm(simplex - support, axis=1)) <= rounding:
            return True
        closest, simplex = closest_simplex(np.vstack((simplex, support)))
        direction = -closest
    return True
