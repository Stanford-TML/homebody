"""Route planning and square stances on the static occupancy map."""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

EDGE_WINDOW = 0.8
EDGE_BAND = 0.30
EDGE_REFITS = 3
EDGE_FIT_TOLERANCE = 1e-3
STANCE_STEP = 0.02
STANCE_LIMIT = 2.0
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
CROWDING_COST = 10.0  # cost factor of a step at the bare radius


def pixel(navigation, xy):
    """The (row, column) cell of a map point; (-1, -1) for a point no grid can index."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = np.linalg.solve(navigation.T_map_px, np.r_[xy, 1.0])
        coordinates = np.rint(p[:2] / p[2])
    if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= np.iinfo(np.int64).max):
        return -1, -1
    return tuple(coordinates.astype(int)[::-1])


def world(navigation, cell):
    p = navigation.T_map_px @ [cell[1], cell[0], 1.0]
    return p[:2] / p[2]


def inside(grid, cell):
    return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]


def cell_free(free, cell):
    """Whether CELL is a free cell of the inflated map; a cell off the grid never is."""
    return inside(free, cell) and bool(free[cell])


def room(navigation):
    """Each cell's metric distance to the nearest occupied cell or map border, less one
    cell diagonal."""
    padded = np.pad(~navigation.occupied, 1, constant_values=False)
    return (distance_transform_edt(padded)[1:-1, 1:-1] - math.sqrt(2)) * navigation.resolution


def free_space(navigation, radius):
    """Cells with more than RADIUS of room."""
    return room(navigation) > radius


def route(navigation, start, goal, radius, margin=0.0):
    """A* from START to GOAL (map metres) over the map inflated by RADIUS, as straightened
    waypoints. None when either end is not a free cell. Steps with less than MARGIN of
    room beyond the radius cost up to CROWDING_COST times more."""
    space = room(navigation)
    free = space > radius
    origin, end = pixel(navigation, start), pixel(navigation, goal)
    if not (cell_free(free, origin) and cell_free(free, end)):
        return None
    costs, parents, queue = {origin: 0.0}, {}, [(0.0, origin)]
    while queue:
        _, node = heapq.heappop(queue)
        if node == end:
            cells = [end]
            while cells[-1] != origin:
                cells.append(parents[cells[-1]])
            cells.reverse()
            return [world(navigation, cell) for cell in straightened(cells, space, radius + margin)]
        for dy, dx in NEIGHBOURS:
            nxt = node[0] + dy, node[1] + dx
            if not cell_free(free, nxt):
                continue
            if dy and dx and not (cell_free(free, (node[0] + dy, node[1])) and
                                  cell_free(free, (node[0], node[1] + dx))):
                continue
            crowding = max(0.0, radius + margin - space[nxt]) / margin if margin else 0.0
            cost = costs[node] + math.hypot(dx, dy) * (1.0 + (CROWDING_COST - 1.0) * crowding)
            if cost < costs.get(nxt, math.inf):
                costs[nxt], parents[nxt] = cost, node
                heapq.heappush(queue, (cost + math.dist(nxt, end), nxt))
    return None


def line_free(start, end, free):
    count = max(1, int(np.max(np.abs(np.subtract(end, start)))) * 2)
    cells = np.rint(np.linspace(start, end, count + 1)).astype(int)
    if not np.all(free[cells[:, 0], cells[:, 1]]):
        return False
    return bool(np.all(free[cells[:-1, 0], cells[1:, 1]]) and
                np.all(free[cells[1:, 0], cells[:-1, 1]]))


def straightened(cells, space, wanted):
    """CELLS as fewer straight legs, each keeping the room (up to WANTED) of the cells it replaces."""
    if len(cells) <= 2:
        return cells
    result, anchor, least = [cells[0]], 0, math.inf
    for index in range(1, len(cells)):
        least = min(least, space[cells[index]])
        if index - anchor > 1 and line_room(cells[anchor], cells[index], space) < min(wanted, least):
            anchor = index - 1
            result.append(cells[anchor])
            least = min(space[cells[anchor]], space[cells[index]])
    return result + [cells[-1]]


def line_room(start, end, space):
    """The least room of the cells a straight leg from START to END crosses, corners included."""
    count = max(1, int(np.max(np.abs(np.subtract(end, start)))) * 2)
    cells = np.rint(np.linspace(start, end, count + 1)).astype(int)
    return float(min(space[cells[:, 0], cells[:, 1]].min(), space[cells[:-1, 0], cells[1:, 1]].min(),
                     space[cells[1:, 0], cells[:-1, 1]].min()))


def clearance(navigation):
    """Metric distance from every cell centre to the nearest occupied cell."""
    return distance_transform_edt(~navigation.occupied) * navigation.resolution


def edge_normal(navigation, point, toward):
    """Unit normal of the occupied edge nearest POINT, signed toward TOWARD, or None when
    too few occupied cells are near."""
    occupied = np.argwhere(navigation.occupied)
    if not len(occupied):
        return None
    xy = np.column_stack((occupied[:, ::-1], np.ones(len(occupied)))) @ navigation.T_map_px.T
    xy = xy[:, :2] / xy[:, 2, None]
    distance = np.linalg.norm(xy - point, axis=1)
    near = xy[distance <= min(EDGE_WINDOW, distance.min() + EDGE_BAND)]
    if len(near) < 4:
        return None
    centre = near.mean(axis=0)
    normal = np.linalg.eigh(np.cov((near - centre).T))[1][:, 0]
    return normal if normal @ (np.asarray(toward) - centre) >= 0 else -normal


def square_stance(navigation, facing, goal, reach, standoff, lateral, free):
    """A (xy, yaw) stance square to the edge at FACING, GOAL's side of it, with the faced
    point REACH ahead and LATERAL metres to the robot's left, backed out to a free cell with
    STANDOFF clearance. None when there is none."""
    facing, goal = np.asarray(facing, float), np.asarray(goal, float)
    metric = clearance(navigation)

    def clear(xy):
        cell = pixel(navigation, xy)
        return inside(metric, cell) and metric[cell] >= standoff and free[cell]

    def walk(origin, direction, start=STANCE_STEP):
        for distance in np.arange(start, STANCE_LIMIT + STANCE_STEP, STANCE_STEP):
            if clear(origin + distance * direction):
                return origin + distance * direction
        return None

    away = goal - facing
    if np.linalg.norm(away) < 1e-6:
        return None
    normal = away / np.linalg.norm(away)
    stance = walk(facing, normal)
    for _ in range(EDGE_REFITS):
        if stance is None:
            return None
        edge = edge_normal(navigation, stance, stance + normal)
        if edge is None or edge @ normal >= math.cos(EDGE_FIT_TOLERANCE):
            break
        stance, normal = walk(facing, edge), edge
    origin = facing + lateral * np.array([normal[1], -normal[0]])
    stance = walk(origin, normal, reach)
    if stance is None:
        return None
    return stance, math.atan2(-normal[1], -normal[0])
