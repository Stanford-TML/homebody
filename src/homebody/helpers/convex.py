"""Frozen convex geometry and Euclidean point clearance, without scene state."""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def convex_shape(points):
    """The convex hull of POINTS as read-only arrays, None when they have no 3-D extent."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Convex geometry requires finite Nx3 points")
    if len(points) < 4:
        return None
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    precision = np.finfo(np.float32).eps * max(1., np.max(np.abs(points)))
    if singular[-1] <= np.sqrt(len(points)) * precision:
        return None
    hull = ConvexHull(points)
    indices = np.full(len(points), -1, dtype=int)
    indices[hull.vertices] = np.arange(len(hull.vertices))
    shape = {"vertices": points[hull.vertices].copy(), "faces": indices[hull.simplices],
             "equations": hull.equations.copy()}
    for array in shape.values():
        array.setflags(write=False)
    return shape


def _triangle_distances_squared(triangles, point):
    """Squared distance from POINT to each closed triangle."""
    edges = np.roll(triangles, -1, axis=1) - triangles
    difference = point - triangles
    length_squared = np.sum(edges**2, axis=2)
    along = np.divide(np.sum(difference * edges, axis=2), length_squared,
                      out=np.zeros_like(length_squared), where=length_squared > 0)
    nearest = triangles + np.clip(along, 0., 1.)[:, :, None] * edges
    result = np.min(np.sum((nearest - point)**2, axis=2), axis=1)
    normal = np.cross(edges[:, 0], -edges[:, 2])
    normal_squared = np.sum(normal**2, axis=1)
    inside = np.all(np.sum(np.cross(edges, difference) * normal[:, None, :], axis=2) >= 0., axis=1)
    height_squared = np.divide(np.sum(difference[:, 0] * normal, axis=1)**2, normal_squared,
                               out=np.full(len(triangles), np.inf), where=normal_squared > 0)
    return np.where(inside, np.minimum(result, height_squared), result)


def point_hits(shape, points, margin):
    """Which POINTS lie inside SHAPE or within Euclidean MARGIN of its surface."""
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("Point clearance requires finite Nx3 points and a nonnegative margin")
    if not len(points):
        return np.zeros(0, dtype=bool)
    equations = shape["equations"]
    arithmetic = 64 * np.finfo(float).eps * max(1., np.max(np.abs(shape["vertices"])),
                                              np.max(np.abs(points)))
    hits = cKDTree(shape["vertices"]).query(points)[0] <= margin + arithmetic
    pending = np.flatnonzero(~hits)
    triangles = shape["vertices"][shape["faces"]]
    for start in range(0, len(pending), 128):
        indices = pending[start:start + 128]
        batch = points[indices]
        distances = np.full(len(batch), -np.inf)
        for face_start in range(0, len(equations), 1024):
            faces = equations[face_start:face_start + 1024]
            distances = np.maximum(distances, np.max(batch @ faces[:, :3].T + faces[:, 3], axis=1))
        hits[indices] = distances <= arithmetic
        for index in np.flatnonzero((distances > arithmetic) & (distances <= margin + arithmetic)):
            point = batch[index]
            for face_start in range(0, len(triangles), 1024):
                faces = triangles[face_start:face_start + 1024]
                if np.min(_triangle_distances_squared(faces, point)) <= (margin + arithmetic)**2:
                    hits[indices[index]] = True
                    break
    return hits
