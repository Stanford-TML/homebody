"""SIM tasks with checked navigation, strict manipulation and persistent memory."""
from homebody.memory import MapBinding
from homebody.memory.binding import navigation_digest
from homebody.vlm.observations import semantic_prior

from .task import compose as task_session


def compose(backend, provider, settings, recorder, controls, *, root, memory_root,
            memory_name, ui=None, arms=None, jaw_samples=None):
    """Bind a real MuJoCo Robot to the task policy without feeding evaluator state to it.

    The caller constructs the scene and owns any separate scorer. Observations come
    from the normal renderer and robot measurements; scan memory is historical.
    """
    from homebody.backends.mujoco import MujocoBackend

    if not isinstance(backend, MujocoBackend):
        raise TypeError("SIM task qualification requires an actual MuJoCo backend")
    scene_id = "sim-" + (backend.scene.name if backend.scene is not None else "flat-floor")
    frame_id = scene_id + "-world"
    navigation = backend.navigation
    binding = MapBinding.from_navigation(scene_id, navigation_digest(navigation), frame_id, navigation)
    prior = (semantic_prior(backend.scene)
             if backend.scene is not None and (backend.scene / "semantics.json").is_file() else {})
    return task_session(backend, provider, settings, recorder, controls, root=root,
                        memory_root=memory_root, memory_name=memory_name,
                        binding=binding, navigation=navigation, ui=ui, arms=arms,
                        jaw_samples=jaw_samples, semantic_context=prior,
                        navigation_evidence=backend.navigation_evidence)
