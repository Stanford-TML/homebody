"""Release-time pruning preserves acquisition retirement and fresh obstacle evidence."""
from dataclasses import replace

import numpy as np
import pytest
from test_place_withdrawal import (
    placement as placement,  # noqa: PLC0414 -- pytest fixture re-export
)
from test_place_withdrawal import run_placement

from homebody.helpers.volume import ColumnHistory
from homebody.skills import place
from homebody.skills.contract import SkillFailure


@pytest.mark.parametrize("outcome", ["landed", "unseen", "landing_failed", "not_empty"])
def test_runner_forgets_only_after_verified_landing_and_empty_hand(placement, monkeypatch, outcome):
    case = placement
    history = case.ctx.navigation_history = ColumnHistory(period=0.)
    binding = (1, "map")
    history.add(0., binding, [[0., 0., .3], [1., 0., .2]], [1, 4])
    calls, forget = [], history.forget

    def recorded(label, before):
        calls.append((label, before))
        forget(label, before)

    monkeypatch.setattr(history, "forget", recorded)
    if outcome in {"landed", "not_empty"}:
        advance = case.robot.advance

        def visible(*args):
            advance(*args)
            case.robot.frame = replace(case.robot.frame, depth=np.full_like(case.robot.frame.depth, .8),
                                       labels=np.ones_like(case.robot.frame.labels))

        monkeypatch.setattr(case.robot, "advance", visible)
    if outcome == "landing_failed":
        def failed(*args, **kwargs):
            raise SkillFailure("PLACEMENT_NOT_VERIFIED", "visible target has not settled")

        monkeypatch.setattr(place, "verify_settled", failed)
    if outcome == "not_empty":
        complete = place.complete_return

        def occupied(*args, **kwargs):
            hands = dict(case.robot.frame.hands)
            hands["right"] = replace(hands["right"], closure=.8)
            case.robot.frame = replace(case.robot.frame, hands=hands)
            return complete(*args, **kwargs)

        monkeypatch.setattr(place, "complete_return", occupied)
    opened_at = case.ctx.observations.time
    result = run_placement(case)
    assert result.code == {"landed": "OK", "unseen": "OK", "landing_failed": "PLACEMENT_NOT_VERIFIED",
                           "not_empty": "OUTCOME_UNKNOWN"}[outcome]
    assert calls == ([(1, opened_at)] if outcome == "landed" else [])
    expected = [[1., 0., .2]] if outcome == "landed" else [[0., 0., .3], [1., 0., .2]]
    np.testing.assert_array_equal(history.points(case.robot.time, binding), expected)
    if outcome in {"landing_failed", "not_empty"}:
        assert "reset" not in case.order
        assert not any(isinstance(item, tuple) and item[0] == "back" for item in case.order)
