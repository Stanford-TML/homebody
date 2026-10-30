"""REAL composition refuses scan context recorded in a different map frame."""
import json

import pytest
from test_real_backend import backend as make_real

from homebody.primitives.observations import ObservationError


def test_real_rejects_scan_context_in_a_different_frame_before_composition(tmp_path):
    (tmp_path / "semantics.json").write_text(json.dumps({
        "coordinate_frame": "different-room", "entities": [{"label": "Wrong room"}]}))
    with pytest.raises(ObservationError, match="different map frames"):
        make_real(tmp_path)
