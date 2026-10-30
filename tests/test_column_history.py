"""Bounded acquired-depth retention across skill boundaries, without erasing hits."""
import numpy as np

from homebody.helpers.volume import ColumnHistory


def add(history, time, z=1., label=0, binding=(1, "map")):
    return history.add(time, binding, [[0., 0., z]], [label])


def test_history_expires_by_acquisition_time_and_updates_revision():
    history = ColumnHistory(seconds=10.)
    add(history, 1.)
    revision = history.revision
    np.testing.assert_array_equal(history.points(11., (1, "map")), [[0., 0., 1.]])
    assert history.revision == revision
    assert history.points(11.01, (1, "map")).shape == (0, 3)
    assert history.revision > revision


def test_new_binding_never_reuses_previous_depth():
    history = ColumnHistory()
    add(history, 2.)
    assert history.points(2., (2, "map")).shape == (0, 3)
    add(history, 2.1, z=.4, binding=(2, "map"))
    np.testing.assert_array_equal(history.points(2.1, (2, "map")), [[0., 0., .4]])
    add(history, 2.2, z=.7, binding=(2, "other-map"))
    np.testing.assert_array_equal(history.points(2.2, (2, "other-map")), [[0., 0., .7]])
