"""Persistent photo memory survives restarts without losing provenance."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from homebody.memory import MapBinding, MemoryError, NamedMemory
from homebody.primitives.observations import NavigationMap


def frame(pose=(0., 0., 0.), *, value=42, sequence=1):
    transform = np.array([[.1, 0., -4.], [0., -.1, 3.], [0., 0., 1.]])
    navigation = NavigationMap(np.zeros((10, 12), bool), transform, .1)
    return SimpleNamespace(
        epoch=2, sequence=sequence, frame_id=(2, sequence), time=float(sequence),
        navigation=navigation, rgb=np.full((6, 8, 3), value, np.uint8),
        depth=np.full((6, 8), .75, np.float32),
        intrinsics=np.array([[600., 0., 4.], [0., 600., 3.], [0., 0., 1.]]),
        world_camera=np.array([[1., 0., 0., pose[0]], [0., 1., 0., pose[1]],
                               [0., 0., 1., 1.3], [0., 0., 0., 1.]]),
        base_pose=np.array(pose))


def binding(observation=None, **overrides):
    observation = frame() if observation is None else observation
    return MapBinding.from_navigation(
        overrides.get("scene_id", "room-v1"),
        overrides.get("scene_hash", hashlib.sha256(b"verified scene geometry").hexdigest()),
        overrides.get("frame_id", "room_map_z_up_metres"), observation.navigation)


def open_memory(tmp_path, name="room_memory"):
    return NamedMemory.open(tmp_path, name, binding())


def test_two_distinct_views_survive_restart_with_original_pixels_and_geometry(tmp_path):
    store = open_memory(tmp_path)
    first = store.remember(frame((0., 0., math.pi / 3)), "table from the door")
    second_frame = frame((1., .5, -.7), value=193, sequence=2)
    second_frame.depth[0, 0] = np.nan
    second = store.remember(second_frame, "table from the aisle")
    reopened = NamedMemory.open(tmp_path, "room_memory", binding(), create=False)
    found = reopened.recall((1., .5, -.7), k=2)
    assert not found["absent"]
    assert [hit["record_id"] for hit in found["hits"]] == [second["record_id"], first["record_id"]]
    hit = found["hits"][0]
    assert hit["historical"] and hit["pose"] == [1., .5, -.7] and hit["time"] == 2.
    assert hit["frame_id"] == [2, 2] and hit["content"] == "table from the aisle"
    np.testing.assert_array_equal(np.asarray(Image.open(hit["image_path"])), second_frame.rgb)
    np.testing.assert_equal(np.load(reopened.path / hit["depth"]["path"], allow_pickle=False),
                            second_frame.depth)
    np.testing.assert_array_equal(hit["camera"]["world_camera"], second_frame.world_camera)


@pytest.mark.parametrize("change", [
    {"scene_hash": "1" * 64}, {"frame_id": "different_origin"}, {"scene_id": "other_room"},
])
def test_map_or_frame_mismatch_refused_before_recall(tmp_path, change):
    store = open_memory(tmp_path)
    store.remember(frame())
    with pytest.raises(MemoryError, match="binding"):
        NamedMemory.open(tmp_path, "room_memory", binding(**change), create=False)


def test_image_tampering_is_refused_on_reopen_and_live_query(tmp_path):
    store = open_memory(tmp_path)
    hit = store.remember(frame())
    Path(hit["image_path"]).write_bytes(b"tampered")
    with pytest.raises(MemoryError, match="integrity"):
        store.recall((0., 0., 0.))
    with pytest.raises(MemoryError, match="integrity"):
        NamedMemory.open(tmp_path, "room_memory", binding(), create=False)
