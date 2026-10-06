"""The MuJoCo backend's own rules, checked on small models: no scene package is loaded."""
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from homebody.backends.mujoco import MujocoBackend, holding_side
from homebody.session.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("physical_evaluation", ROOT / "evaluation/evaluate.py")
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)
CUBE = " ".join(f"{x} {y} {z}" for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05))


def bare_backend(**attributes):
    """A backend that never ran __init__: no scene, no controller, only what a test sets."""
    backend = MujocoBackend.__new__(MujocoBackend)
    for name, value in attributes.items():
        setattr(backend, name, value)
    return backend


def test_the_robot_camera_draws_single_samples_and_the_operator_views_keep_the_models():
    """Segmentation is drawn as id colours, and resolving a multisampled buffer averages
    them: an edge can read as a geom that is not there, or an id past the end of the scene.
    The operator's views stay smooth."""
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="960" offheight="540"/>
      </visual><worldbody><camera name="ego"/><geom type="sphere" size=".1"/></worldbody></mujoco>''')
    backend = bare_backend(model=model, data=mujoco.MjData(model), settings=Settings(),
                           _renderer=None, _observer_renderer=None)
    samples = model.vis.quality.offsamples
    try:
        assert samples > 0 and backend._render_context()._mjr_context.offSamples == 0
        assert backend._observer_context()._mjr_context.offSamples == samples == model.vis.quality.offsamples
        rgb, depth, segmentation = backend._render_ego()
        assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
        assert set(np.unique(segmentation[:, :, 0])) <= {-1, 0}
    finally:
        backend.close()


def micro_model(hull="mesh", toucher="left_wrist_roll_link", touch_height=.11):
    """A free box sunk a millimetre into a floor plane, a static pedestal, and a robot link
    TOUCHER below the pelvis whose sphere sits TOUCH_HEIGHT above the floor: at .11 it
    presses into the box."""
    box = {"mesh": '<geom name="box__collision" type="mesh" mesh="cube"/>',
           "box": '<geom name="box__collision" type="box" size=".05 .05 .05"/>',
           "none": '<geom type="box" size=".05 .05 .05"/>'}[hull]
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <asset><mesh name="cube" vertex="{CUBE}"/></asset>
      <worldbody>
        <geom name="floor__0" type="plane" size="2 2 .1"/>
        <body name="box" pos="0 0 .049"><freejoint/>{box}</body>
        <body name="pedestal" pos="1 0 0"><geom name="pedestal__collision" type="mesh" mesh="cube"/></body>
        <body name="pelvis" pos="0 0 {touch_height}">
          <body name="{toucher}"><geom type="sphere" size=".02"/></body></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_stand_rests_the_hull_bottom_at_the_requested_height():
    model, data = micro_model()
    backend = bare_backend(model=model, data=data)
    backend._stand("box", 1., 2., .7, np.pi / 2)
    np.testing.assert_allclose(data.qpos[:3], [1., 2., .755])
    np.testing.assert_allclose(data.qpos[3:7], [np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)])


@pytest.mark.parametrize("hull, body, reason", [
    ("box", "box", "'box' no mesh collision geom named 'box__collision'"),
    ("none", "box", "'box' no mesh collision geom named 'box__collision'"),
    ("mesh", "pedestal", "'pedestal' is not a free body")])
def test_a_body_without_a_mesh_hull_or_a_free_joint_is_refused_by_name(hull, body, reason):
    model, data = micro_model(hull)
    backend = bare_backend(model=model, data=data)
    with pytest.raises(ValueError, match=reason):
        backend._stand(body, 0., 0., 0., 0.)
    if hull != "mesh":
        with pytest.raises(ValueError, match="'box' no mesh collision geom"):
            backend._hull("box")


def test_every_arm_link_of_the_packaged_robot_counts_as_holding():
    """The robot's bodies below each shoulder all carry _elbow_, _wrist_ or _hand_."""
    xml = ET.parse(ROOT / "assets/robot/g1_grasp/g1_closd_grasp_deploy.xml").getroot()
    bodies = {body.get("name") for body in xml.iter("body")}
    arms = {name for name in bodies if name.startswith(("left_", "right_"))
            and not any(part in name for part in ("shoulder", "hip", "knee", "ankle"))}
    assert arms and {name for name in bodies if holding_side(name)} == arms
    assert {holding_side(name) for name in arms} == {"left", "right"}
    assert holding_side("left_shoulder_pitch_link") is None and holding_side("island") is None


def contact_backend(model, data):
    """A backend over a micro model: its one movable object is the box."""
    backend = bare_backend(model=model, data=data, settings=Settings(), epoch=1, _valid=True,
                           _torso=model.body("pelvis").id, _pelvis=model.body("pelvis").id, _objects={"box": {}})
    backend._hulls = {"box": backend._hull("box")}
    return backend


@pytest.mark.parametrize("toucher, hands, robot, reason", [
    ("left_wrist_roll_link", ["left"], [], "hand_contact"),
    ("right_elbow_link", ["right"], [], "hand_contact"),
    ("left_hand_palm_link", ["left"], [], "hand_contact"),
    ("torso_link", [], ["torso_link"], "robot_contact")])
def test_an_object_any_robot_link_touches_is_not_released(toucher, hands, robot, reason):
    """A wrist or an elbow holds the object as a hand does; an object resting on the torso
    or a leg is not released either."""
    box = contact_backend(*micro_model(toucher=toucher)).evaluation_state()["objects"]["box"]
    assert (box["hand_contacts"], box["robot_contacts"], box["support_contacts"]) == (hands, robot, ["floor"])
    goal = {"bounds": [[-1, -1, -1], [1, 1, 1]], "support_contacts": ["floor"]}
    assert evaluate.object_reason(box, goal) == reason
    released = contact_backend(*micro_model(toucher=toucher, touch_height=.5))
    box = released.evaluation_state()["objects"]["box"]
    assert box["hand_contacts"] == box["robot_contacts"] == [] and evaluate.object_reason(box, goal) is None


def furnished_model():
    """A scene that groups its furniture into bodies: an island top inside a static frame
    body, a thin mesh sheet beside it, and the box resting on the island."""
    sheet = " ".join(f"{x} {y} {z}" for x in (-.2, .2) for y in (-.2, .2) for z in (0, .005))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <asset><mesh name="cube" vertex="{CUBE}"/><mesh name="sheet" vertex="{sheet}"/></asset>
      <worldbody>
        <body name="kitchen"><body name="island_frame" pos="0 0 .5">
          <geom name="island__top" type="box" size=".3 .3 .05" group="2"/>
          <geom name="counter__sheet" type="mesh" mesh="sheet" pos="1 0 .1" group="2"/></body></body>
        <body name="box" pos="0 0 .599"><freejoint/><geom name="box__collision" type="mesh" mesh="cube"/></body>
        <body name="pelvis" pos="5 0 0"><geom type="sphere" size=".02"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_furniture_in_a_static_body_is_as_static_as_world_geometry():
    """The supports an object rests on are every body welded to the world."""
    model, data = furnished_model()
    box = contact_backend(model, data).evaluation_state()["objects"]["box"]
    assert box["support_contacts"] == ["island"] and box["robot_contacts"] == []


def test_a_stop_holds_the_applied_arm_reference_not_the_sagging_measurement():
    """A loaded arm sits below its reference; latching the measured joints on every stop
    would lower a held object a little each time until it sweeps what it is set beside."""
    from types import SimpleNamespace

    from homebody.backends.controller import SIDES, Controller
    controller = Controller.__new__(Controller)
    controller.arm_target = np.zeros(14)
    controller.gates = {side: SimpleNamespace(q=np.full(7, .3 if side == "left" else -.3)) for side in SIDES}
    data = SimpleNamespace(qpos=np.full(40, .25), time=0.)  # measured joints, sagged below the reference
    backend = bare_backend(controller=controller, data=data, epoch=1,
                           limits=SimpleNamespace(epoch=lambda *args: None))
    controller.velocity = lambda *args: None
    backend.stop(1)
    for side, arm in SIDES.items():
        assert np.allclose(controller.arm_target[arm], controller.gates[side].q)
