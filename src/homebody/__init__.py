"""HomeBody. Importing it sets MUJOCO_GL for this machine unless already set."""
import os as _os
import sys as _sys
from pathlib import Path as _Path


def offscreen_gl():
    """The offscreen GL backend: cgl on macOS, egl on Linux with a GPU, else osmesa."""
    if _sys.platform == "darwin":
        return "cgl"
    if _sys.platform.startswith("linux"):
        gpu = _Path("/dev/nvidiactl").exists() or any(_Path("/dev/dri").glob("renderD*"))
        return "egl" if gpu else "osmesa"
    return "glfw"


_os.environ.setdefault("MUJOCO_GL", offscreen_gl())
