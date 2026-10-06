"""tools/build_kitchen.py builds any scene package: a small room stands in for a second scan."""
import json
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh
from trimesh.transformations import translation_matrix
from trimesh.visual.material import PBRMaterial

from tools import build_kitchen

FLOOR = trimesh.Trimesh([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], [[0, 1, 2], [0, 2, 3]],
                        process=False)
CUP_BOUNDS = [[.97, -.03, .725], [1.03, .03, .825]]
BOXES = [{"name": "table__support", "position_ws_map_m": [1, 0, .65], "half_size_m": [.5, .3, .05],
          "yaw_rad": 0.0, "contact_solref": [.05, 1.]},
         {"name": "table__leg", "position_ws_map_m": [1, 0, .3], "half_size_m": [.05, .05, .3],
          "yaw_rad": .5}]


def painted(mesh):
    material = PBRMaterial(baseColorFactor=[150, 150, 150, 255])
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def glb_node_indices(data):
    length = struct.unpack("<I", data[12:16])[0]
    nodes = json.loads(data[20:20 + length])["nodes"]
    return {node["name"]: index for index, node in enumerate(nodes)}


@pytest.fixture
def room(tmp_path):
    """A package laid out like assets/real2sim/src_kitchen: a floor sheet, a table, a cup on the table."""
    folder = tmp_path / "small_room"
    folder.mkdir()
    parts = {"floor_sheet": FLOOR.copy(),
             "table_top": trimesh.creation.box((1., .6, .05), translation_matrix([1, 0, .7])),
             "cup_body": trimesh.creation.box((.06, .06, .1), translation_matrix([1, 0, .775]))}
    glb = trimesh.Scene()
    for name, mesh in parts.items():
        glb.add_geometry(painted(mesh), node_name=name, geom_name=name)
    data = glb.export(file_type="glb")
    (folder / "room.glb").write_bytes(data)
    nodes = glb_node_indices(data)
    entities = [{"id": "floor", "glb_node_indices": [nodes["floor_sheet"]]},
                {"id": "table", "glb_node_indices": [nodes["table_top"]]},
                {"id": "cup", "glb_node_indices": [nodes["cup_body"]],
                 "bounds_ws_map_m": CUP_BOUNDS}]
    (folder / "semantics.json").write_text(json.dumps({"coordinate_frame": {"name": "room"},
                                                       "entities": entities}))
    (folder / "task_scene.json").write_text(json.dumps({
        "movable_entities": ["cup"], "collision_boxes": BOXES,
        "object_contact": {"cup": {"friction": "2.0 0.10 0.05", "note": "measured on the cup"}}}))
    return folder


def edit(folder, **changes):
    path = folder / "task_scene.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def test_a_second_scene_package_builds_and_compiles(room):
    report = build_kitchen.build(room)
    scene = ET.parse(room / "scene.xml").getroot()
    geoms = {geom.get("name"): geom.attrib for geom in scene.iter("geom")}
    assert scene.get("model") == "small_room"
    assert [body.get("name") for body in scene.iter("body")] == ["cup"]
    assert geoms["cup__collision"]["friction"] == "2.0 0.10 0.05"
    assert report["objects"]["cup"]["contact_note"] == "measured on the cup"
    assert geoms["table__support"]["solref"] == "0.05 1" and "quat" not in geoms["table__support"]
    assert "solref" not in geoms["table__leg"] and "quat" in geoms["table__leg"]
    model = mujoco.MjModel.from_xml_path(str(room / "scene.xml"))
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert [model.geom(int(plane)).name.split("__")[0] for plane in planes] == ["floor"]
    assert model.body("cup").jntnum[0] == 1


@pytest.mark.parametrize("changes, problem", [
    ({"movable_entities": ["cup", "kettle"]}, "absent from semantics.json"),
    ({"object_contact": {"table": {"friction": "1 1 1", "note": "fixed"}}}, "contact for table"),
    ({"object_contact": {"cup": {"friction": "2.0 0.10", "note": "two only"}}}, "contact for cup"),
    ({"object_contact": {"cup": {"friction": "2.0 0.10 0.05"}}}, "contact for cup"),
    ({"collision_boxes": [{**BOXES[0], "half_size_m": [.5, .3, 0]}]}, "box: table__support"),
    ({"collision_boxes": [{**BOXES[0], "contact_solref": [.05]}]}, "box: table__support"),
])
def test_an_unusable_declaration_is_refused_before_anything_is_written(room, changes, problem):
    edit(room, **changes)
    with pytest.raises(ValueError, match=problem):
        build_kitchen.build(room)
    assert not (room / "scene.xml").exists() and not (room / "meshes").exists()


def test_a_package_needs_exactly_one_semantic_glb(room):
    (room / "spare.glb").write_bytes((room / "room.glb").read_bytes())
    with pytest.raises(ValueError, match="exactly one semantic .glb"):
        build_kitchen.build(room)
