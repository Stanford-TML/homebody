"""Inspect imports in a separate interpreter; no execution dependency substitutions."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def inspect_cli(arguments):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT / "vendor"))
    script = '''import json, sys
from homebody.__main__ import main
try:
    result = main(json.loads(sys.argv[1]))
except SystemExit as exc:
    result = exc.code
forbidden = [name for name in sys.modules if name.startswith(
    ("homebody.backends", "mujoco", "viser", "torch"))]
print(json.dumps({"result":result, "execution_imports":forbidden}))
'''
    process = subprocess.run([sys.executable, "-c", script, json.dumps(arguments)],
                             env=environment, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.splitlines()[-1])


@pytest.mark.parametrize("arguments, codes", [
    (["--help"], {0}),
    (["--headless"], {2}),
    (["check", "--root", str(ROOT)], {0, 1}),
    (["check", "--root", str(ROOT), "--scene", "{tmp}"], {1}),
])
def test_help_headless_and_the_static_check_never_import_a_backend(tmp_path, arguments, codes):
    """Headless cannot submit an implicit task; the check reports dependency availability
    without initializing anything and inspects the selected scene package."""
    result = inspect_cli([argument.replace("{tmp}", str(tmp_path)) for argument in arguments])
    assert result["result"] in codes and result["execution_imports"] == []
