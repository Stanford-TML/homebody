"""Build a scene package's MuJoCo scene from its semantic GLB, without changing its frame.

    build_kitchen.py [--scene DIR]     (default: the packaged scene)

DIR holds one semantic `.glb`, `semantics.json` and `task_scene.json`. The build writes
`scene.xml`, `physics.json` and `meshes/` into DIR.
"""
from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from homebody.session.config import DEFAULT_SCENE

ROOT = Path(__file__).resolve().parents[1]
OBJECT_MASS = 0.15
OBJECT_FRICTION = "4.0 0.20 0.020"  # sliding, torsional, rolling
PART_FRICTION = "1.2 0.005 0.0001"


def numbers(values):
    return " ".join(f"{float(x):.9g}" for x in values)


def positive(values, count):
    """Whether VALUES are COUNT finite numbers above zero."""
    array = np.asarray(values, dtype=float)
    return array.shape == (count,) and bool(np.isfinite(array).all() and (array > 0).all())


def semantic_glb(folder: Path) -> Path:
    sources = sorted(folder.glob("*.glb"))
    if len(sources) != 1:
        raise ValueError(f"A scene package holds exactly one semantic .glb, not {len(sources)}: "
                         f"{folder}")
    return sources[0]


def glb_nodes(source: Path) -> list:
    """The GLB's node table, by node index."""
    with source.open("rb") as stream:
        stream.read(12)
        length, _ = struct.unpack("<II", stream.read(8))
        return json.loads(stream.read(length))["nodes"]


def checked(task: dict, entities: dict) -> dict:
    """TASK, validated against ENTITIES."""
    movable = task["movable_entities"]
    unknown = sorted(set(movable) - set(entities))
    if unknown:
        raise ValueError(f"Movable entities absent from semantics.json: {unknown}")
    for identity, contact in task.get("object_contact", {}).items():
        friction = contact.get("friction")
        if (identity not in movable or not isinstance(friction, str) or
                not positive(friction.split(), 3) or not contact.get("note")):
            raise ValueError(f"Invalid object contact for {identity}")
    for box in task.get("collision_boxes", []):
        position = box["position_ws_map_m"]
        if (len(position) != 3 or not np.isfinite([*position, box["yaw_rad"]]).all() or
                not positive(box["half_size_m"], 3) or
                ("contact_solref" in box and not positive(box["contact_solref"], 2))):
            raise ValueError(f"Invalid collision box: {box['name']}")
    return task


def with_centroid(mesh):
    """A lone triangle as three around its centroid: MuJoCo requires four mesh vertices."""
    uv = mesh.visual.uv
    return trimesh.Trimesh(vertices=np.vstack((mesh.vertices, mesh.vertices.mean(axis=0))),
                           faces=[[0, 1, 3], [1, 2, 3], [2, 0, 3]], process=False,
                           visual=trimesh.visual.TextureVisuals(uv=np.vstack((uv, uv.mean(axis=0))),
                                                                material=mesh.visual.material))


def sheet_collision(parent, identity, index, mesh, axes):
    """A planar static part's collision: the ground plane for the floor, else a thin box."""
    rotation = axes.T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    local = mesh.vertices @ rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    floor = identity == "floor"
    if floor and rotation[2, 2] < 0:
        rotation[:, [0, 2]] *= -1
    ET.SubElement(parent, "geom", name=f"{identity}__{index}_collision",
                  type="plane" if floor else "box",
                  pos=numbers(mesh.vertices.mean(axis=0) if floor else rotation @ ((lo + hi) / 2)),
                  quat=numbers(Rotation.from_matrix(rotation).as_quat(scalar_first=True)),
                  size="0 0 0.05" if floor else numbers(np.maximum((hi - lo) / 2, 0.002)),
                  contype="2", conaffinity="3", rgba="0 0 0 0", group="3", mass="0")


class Scene:
    """One package's MuJoCo scene and physics report."""

    def __init__(self, folder: Path):
        self.folder, self.meshes = folder, folder / "meshes"
        source = semantic_glb(folder)
        self.semantics = json.loads((folder / "semantics.json").read_text())
        self.entities = {entity["id"]: entity for entity in self.semantics["entities"]}
        self.task = checked(json.loads((folder / "task_scene.json").read_text()), self.entities)
        self.nodes = glb_nodes(source)
        self.glb = trimesh.load(source, force="scene", process=False)
        self.owner = {index: entity["id"] for entity in self.semantics["entities"]
                      for index in entity["glb_node_indices"]}
        self.movable = set(self.task["movable_entities"])
        self.xml = ET.Element("mujoco", model=folder.name)
        self.assets = ET.SubElement(self.xml, "asset")
        self.world = ET.SubElement(self.xml, "worldbody")
        self.bodies = {}
        self.report = {"frame": self.semantics["coordinate_frame"],
                       "collision": "convex hull per source mesh", "objects": {}, "parts": []}

    def source_mesh(self, index):
        """GLB node INDEX's mesh, placed in the map frame."""
        transform, key = self.glb.graph[self.nodes[index]["name"]]
        mesh = self.glb.geometry[key].copy()
        mesh.apply_transform(transform)
        return mesh

    def movable_body(self, identity):
        """Add IDENTITY as a free body with one convex collision hull."""
        entity = self.entities[identity]
        bounds = np.array(entity["bounds_ws_map_m"])
        center = bounds.mean(axis=0)
        body = ET.SubElement(self.world, "body", name=identity, pos=numbers(center))
        ET.SubElement(body, "freejoint", name=identity + "_joint")
        self.bodies[identity] = (body, center)
        parts = [self.source_mesh(index) for index in entity["glb_node_indices"]]
        hull = trimesh.util.concatenate(parts).convex_hull
        hull.vertices -= center
        name = identity + "_collision"
        (self.meshes / f"{name}.obj").write_text(trimesh.exchange.obj.export_obj(
            hull, include_texture=False, include_color=False))
        ET.SubElement(self.assets, "mesh", name=name, file=f"meshes/{name}.obj", inertia="convex")
        contact = self.task.get("object_contact", {}).get(identity, {})
        ET.SubElement(body, "geom", name=identity + "__collision", type="mesh", mesh=name,
                      mass=str(OBJECT_MASS), friction=contact.get("friction", OBJECT_FRICTION),
                      condim="6", contype="2", conaffinity="3", group="3", rgba="0 0 0 0")
        record = {"initial_position": center.tolist(), "mass_kg": OBJECT_MASS,
                  "mass_source": "simulation assumption", "source_bounds": bounds.tolist(),
                  "collision": ("single convex hull of source object, "
                                "excluding separate label collisions")}
        if contact:
            record["contact_note"] = contact["note"]
        self.report["objects"][identity] = record

    def material(self, mesh_name, material):
        """Add the part's material asset (and texture) and return its name."""
        rgba = np.asarray(material.baseColorFactor if material.baseColorFactor is not None
                          else [180, 180, 180, 255], dtype=float) / 255
        attributes = {"name": mesh_name + "_material", "rgba": numbers(rgba), "specular": "0.1"}
        if material.baseColorTexture is not None:
            material.baseColorTexture.save(self.meshes / f"{mesh_name}.png")
            ET.SubElement(self.assets, "texture", name=mesh_name, type="2d",
                          file=f"meshes/{mesh_name}.png")
            attributes["texture"] = mesh_name
        ET.SubElement(self.assets, "material", **attributes)
        return attributes["name"]

    def part(self, index):
        """Add GLB node INDEX as a visual mesh under its owner's body, or the world."""
        mesh = self.source_mesh(index)
        if len(mesh.vertices) == 3:
            mesh = with_centroid(mesh)
        identity = self.owner.get(index, "structure")
        parent, origin = self.bodies.get(identity, (self.world, np.zeros(3)))
        mesh.vertices -= origin
        mesh_name = f"part_{index:03d}"
        (self.meshes / f"{mesh_name}.obj").write_text(trimesh.exchange.obj.export_obj(
            mesh, include_texture=True, write_texture=False))
        centered = mesh.vertices - mesh.vertices.mean(axis=0)
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        planar = singular[-1] < 1e-6 * max(singular[0], 1.)
        attributes = {"name": mesh_name, "file": f"meshes/{mesh_name}.obj"}
        if planar:
            attributes.update(inertia="shell", maxhullvert="-1")
        ET.SubElement(self.assets, "mesh", **attributes)
        geom = {"name": f"{identity}__{index}", "type": "mesh", "mesh": mesh_name,
                "material": self.material(mesh_name, mesh.visual.material), "contype": "2",
                "conaffinity": "3", "group": "2", "friction": PART_FRICTION, "mass": "0"}
        if identity in self.movable or planar:
            geom.update(contype="0", conaffinity="0")
        if planar and identity not in self.movable:
            sheet_collision(parent, identity, index, mesh, axes)
        ET.SubElement(parent, "geom", **geom)
        self.report["parts"].append({"name": geom["name"], "bounds": (mesh.bounds + origin).tolist()})

    def proxy_box(self, box):
        """Add a declared solid collision box."""
        attributes = {"name": box["name"], "type": "box", "pos": numbers(box["position_ws_map_m"]),
                      "size": numbers(box["half_size_m"])}
        if box["yaw_rad"]:
            yaw = Rotation.from_euler("z", box["yaw_rad"])
            attributes["quat"] = numbers(yaw.as_quat(scalar_first=True))
        attributes.update(contype="2", conaffinity="3", group="3", mass="0")
        if "contact_solref" in box:
            attributes["solref"] = numbers(box["contact_solref"])
        ET.SubElement(self.world, "geom", **attributes, rgba="0 0 0 0")
        self.report.setdefault("collision_boxes", []).append(box)

    def write(self):
        ET.indent(self.xml)
        ET.ElementTree(self.xml).write(self.folder / "scene.xml", encoding="unicode")
        (self.folder / "physics.json").write_text(json.dumps(self.report, indent=2) + "\n")
        return self.report


def build(folder: Path) -> dict:
    """Write FOLDER's scene.xml, physics.json and meshes/ from its package; return the report."""
    scene = Scene(folder)
    scene.meshes.mkdir(exist_ok=True)
    for identity in sorted(scene.movable):
        scene.movable_body(identity)
    for index, node in enumerate(scene.nodes):
        if "mesh" in node:
            scene.part(index)
    for box in scene.task.get("collision_boxes", []):
        scene.proxy_box(box)
    return scene.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=ROOT / DEFAULT_SCENE,
                        help="Scene package directory")
    result = build(parser.parse_args().scene)
    print(f"Built {len(result['parts'])} parts and {len(result['objects'])} movable objects")
