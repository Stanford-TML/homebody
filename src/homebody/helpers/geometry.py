"""Coordinate conventions: metres, Z up; camera X right, Y down, Z forward."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, rotation=None):
    result = np.eye(4)
    result[:3, 3] = position
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def joint_origin(joint):
    """A URDF joint element's origin as a transform; identity when it declares none."""
    node = joint.find("origin")
    if node is None:
        return np.eye(4)
    rpy = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
    return transform(np.fromstring(node.get("xyz", "0 0 0"), sep=" "),
                     Rotation.from_euler("xyz", rpy).as_matrix())


def skew(vector):
    """The cross-product matrix of a 3-vector."""
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(cross, angle):
    """Rotation by `angle` about the unit axis whose cross-product matrix is `cross`."""
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def inverse(matrix):
    rotation = matrix[:3, :3]
    return transform(-rotation.T @ matrix[:3, 3], rotation.T)


def points_in(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unproject(depth, intrinsics, mask):
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    return np.column_stack(((u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                            (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z))

