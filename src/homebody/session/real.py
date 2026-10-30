"""Compose REAL tasks through the same Session and registered skill lifecycle as SIM."""
from homebody.memory import MapBinding
from homebody.vlm.observations import semantic_prior

from .task import compose as task_session


def compose(backend, provider, settings, recorder, controls, *, root, memory_root,
            memory_name, ui=None, arms=None, jaw_samples=None):
    """Bind capabilities once; neither skills nor Session choose a concrete backend."""
    mapped = backend.map_binding
    binding = MapBinding.from_navigation(mapped.frame_id, mapped.sha256,
                                         mapped.frame_id, mapped.navigation)
    scene = mapped.path.parent
    prior = semantic_prior(scene) if (scene / "semantics.json").is_file() else {}
    return task_session(backend, provider, settings, recorder, controls, root=root,
                        memory_root=memory_root, memory_name=memory_name,
                        binding=binding, navigation=mapped.navigation, ui=ui, arms=arms,
                        jaw_samples=jaw_samples, semantic_context=prior,
                        target_selector=backend.select_target,
                        navigation_evidence=backend.navigation_evidence, readiness=backend.readiness,
                        target_retention=lambda carried: backend.retain_targets(
                            {int(record["label"]) for record in carried.values()}))
