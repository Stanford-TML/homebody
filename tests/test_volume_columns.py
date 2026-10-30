"""The docking column rule: parts over occupied cells pass only over a seen support, with nothing
seen at their height."""
import numpy as np

from homebody.helpers.volume import Columns

SPACING = .02


def surface(x, y, z, step=.01):
    """Returns on a horizontal patch spanning X and Y at height Z."""
    xs, ys = np.meshgrid(np.arange(*x, step), np.arange(*y, step))
    return np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, z)))



def test_column_evidence_distinguishes_hit_missing_support_and_clear_support():
    returns = Columns(np.array([[0., 0., .2], [0., 0., 1.], [1., 0., .2],
                                [2., 0., 1.], [3., 0., 1.5]]), SPACING)
    points = np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.],
                       [3., 0., 1.], [4., 0., 1.]])
    seen_at, beneath = returns.evidence(points)
    np.testing.assert_array_equal(seen_at, [True, False, True, False, False])
    np.testing.assert_array_equal(beneath, [True, True, False, False, False])
