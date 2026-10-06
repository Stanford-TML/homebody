"""Settings remain valid through both file loading and programmatic extension."""
from dataclasses import replace

import pytest

from homebody.session.settings import Camera, Grasp, Motion, Physics, Placement, Servo, Settings


@pytest.mark.parametrize("changes", [
    {"camera": Camera(near=20, far=15)},
    {"camera": Camera(width=True)},
    {"motion": Motion(drive_speed=float("nan"))},
    {"motion": Motion(stop_distance=.2)},
    {"placement": Placement(rest_arm_right=(0.,))},
    {"motion": Motion(drive_speed=10**400)},
    {"physics": Physics(timestep=1)},
    {"physics": Physics(hand_torque_fraction=2)},
    {"grasp": Grasp(spline_out=())},
    {"motion": {"drive_speed": .2}},
    {"physics": Physics(integrator="euler")}, {"physics": Physics(integrator="unsupported")},
    {"physics": Physics(cone="frictionless")}, {"physics": Physics(impratio=0.)},
    {"physics": Physics(impratio=-1.)}, {"physics": Physics(impratio=float("inf"))},
    {"physics": Physics(impratio=True)}, {"physics": Physics(noslip_iterations=-1)},
    {"physics": Physics(noslip_iterations=.5)}, {"physics": Physics(noslip_iterations=True)},
    {"physics": Physics(noslip_iterations=float("inf"))},
])
def test_programmatic_settings_reject_invalid_values(changes):
    with pytest.raises(ValueError):
        replace(Settings(), **changes)


def test_file_and_programmatic_settings_share_validation(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[camera]\nnear = 20\nfar = 15\n")
    with pytest.raises(ValueError, match="near < far"):
        Settings.load(path)
    path.write_text("camera = 3\n")
    with pytest.raises(TypeError, match="settings table"):
        Settings.load(path)


@pytest.mark.parametrize("overlay, error, message", [
    ("stance_clearance = 0.22\n", ValueError, r"Unknown settings sections: \['stance_clearance'\]"),
    ("motion = 0.22\n", TypeError, "motion must be a settings table")])
def test_a_scene_overlay_written_outside_its_table_is_named_like_the_base_file(tmp_path, overlay, error, message):
    """A scene package's settings.toml goes through the same refusals as the base file."""
    (tmp_path / "base.toml").write_text("[motion]\nrobot_radius = 0.2\n")
    (tmp_path / "settings.toml").write_text(overlay)
    with pytest.raises(error, match=message):
        Settings.load(tmp_path / "base.toml", scene=tmp_path)


def test_skill_sample_period_matches_executed_control_clock():
    settings = Settings()
    assert settings.servo.tick * settings.physics.control_hz == 10
    for tick in (.05, .001):
        with pytest.raises(ValueError, match="controller periods"):
            Settings(servo=Servo(tick=tick))
