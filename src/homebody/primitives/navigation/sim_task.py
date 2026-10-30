"""Checked-task navigation evidence from SIM observations and its supplied map."""
import hashlib

import numpy as np

from homebody.helpers.geometry import points_in, unproject
from homebody.helpers.raster import world_to_px
from homebody.primitives.observations import NavigationMap

from .task import NavigationEvidence


class SimNavigationEvidence:
    """Use rendered metric depth and measured SIM pose; never inspect scene object state."""
    def __init__(self, robot, navigation, *, frame_id="simulation-map", floor_height_m=0.0, floor_plane=None):
        self.robot, self.navigation, self.frame_id = robot, navigation, frame_id
        self.floor_plane = floor_plane
        self.floor_height_m = float(floor_height_m)
        if not np.isfinite(self.floor_height_m):
            raise ValueError("The scene floor height must be finite")
        self.map_hash = hashlib.sha256(navigation.occupied.tobytes() + navigation.T_map_px.tobytes()).hexdigest()
        self._frame = None
        self._obstacles = None
        self._labels = None
        self._map = None

    def __call__(self):
        if (self._frame is None or self._frame.epoch != self.robot.epoch or
                self.robot.time - self._frame.time >= .1):
            self._frame = self.robot.snapshot()
            frame = self._frame
            valid = (frame.labels >= 0) & np.isfinite(frame.depth) & (frame.depth > 0)
            points = points_in(frame.world_camera, unproject(frame.depth, frame.intrinsics, valid))
            heights = (points[:, 2] - self.floor_height_m if self.floor_plane is None
                       else points @ np.asarray(self.floor_plane[:3]) + self.floor_plane[3])
            relevant = (heights > .10) & (heights < 1.60)
            self._obstacles = points[relevant]
            self._labels = frame.labels[valid][relevant]
            occupied = self.navigation.occupied.copy()
            if len(self._obstacles):
                pixels = np.rint(world_to_px(self.navigation.T_map_px, self._obstacles[:, :2])).astype(int)
                valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < occupied.shape[1]) &
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < occupied.shape[0]))
                pixels = pixels[valid]
                occupied[pixels[:, 1], pixels[:, 0]] = True
            self._map = NavigationMap(occupied, self.navigation.T_map_px, self.navigation.resolution)
        state = self.robot.measure()
        return NavigationEvidence(state.epoch, state.time, state.time, self._frame.time,
            self._frame.time, state.base_pose, self._map, self._obstacles, 1., self.frame_id,
            self.map_hash, state.epoch, "sim_pose_and_rendered_metric_depth", True, True, self.navigation, self.floor_height_m, self.floor_plane,
            obstacle_state=self._frame, obstacle_labels=self._labels)
