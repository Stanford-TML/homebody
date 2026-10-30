"""Walking heading feedback must check the resulting arm/payload rotation too."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import evidence, loaded_navigation, marked
from test_volume_columns import surface

from homebody.primitives.navigation.actions.lower_body import LowerBody
from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.navigate_real import task as navigation


@pytest.mark.parametrize("direction", [-1., 1.])
def test_translation_clear_but_heading_feedback_swings_payload_into_observed_column(direction):
    robot, ctx, _, _ = loaded_navigation(compactable=True)
    ctx.settings = replace(ctx.settings, collision=replace(ctx.settings.collision, column_margin=.03))
    ctx.carried["right"]["payload_half"] = np.full(3, .01)
    ctx.carried["right"]["palm_to_payload"][0, 3] = 1.
    # The current padded payload ends at y=3 +/- .04. The adjacent occupied
    # row begins at +/- .05, outside its straight translation, but in its turn.
    ys = (3.06, 3.15) if direction > 0 else (2.85, 2.94)
    hits = np.vstack([surface((2.40, 2.85), ys, z)
                      for z in np.arange(.90, 1.11, .01)])
    robot.frame = replace(robot.frame, navigation=marked(robot.frame.navigation, hits))
    ctx.navigation_evidence = lambda: evidence(robot, obstacles=hits)
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)

    bridge._motion_check(.20, 0., 0.)
    assert robot.commands == []
    with pytest.raises(MapObstacleRefused, match="Observed columns") as failure:
        bridge.step("lower_body", LowerBody(vx=.20, vy=0., wz=direction * .25))

    assert failure.value.details["command"] == [.20, 0., direction * .25]
    assert failure.value.details["witness_part"] == "right payload"
    assert failure.value.details["observed_at_height"]
    assert not any(any(command) for command in robot.commands)
