"""Explicit task ownership and passive hand uncertainty at REAL session shutdown."""
import math
import uuid


class RetentionGuard:
    """Retain the command owner until placement or an acknowledged physical handover.

    A missing object ledger is not evidence of empty hands. The handover acknowledgment
    applies only until the next task, which can change the physical grasp state.
    """

    def __init__(self, backend, recorder, controls, *, carried, update,
                 task_active=lambda: False, target_retention=None):
        self.backend, self.recorder, self.controls = backend, recorder, controls
        self.carried, self.update = carried, update
        self.task_active, self.target_retention = task_active, target_retention
        self._handover_confirmed = False
        self._ended = False
        self._task_generation = 0
        self._confirmation = None
        self._confirmation_identity = None

    def begin_task(self):
        with self.controls.task_lock:
            self._task_generation += 1
            self._handover_confirmed = False
            self._ended = False
            self._confirmation = None

    def status(self):
        with self.controls.task_lock:
            status = self._status()
            if self.task_active() or not status["retained_or_unknown"]:
                self._confirmation = None
            else:
                carried = tuple((side, str(record.get("label")), str(record.get("state")))
                                for side, record in sorted(self.carried().items()))
                hands = tuple((side, record.get("retained_or_unknown"), record.get("calibrated_retention"),
                               record.get("contact_state"), record.get("retention_source"),
                               float(record.get("closure", 0.)) > .15)
                              for side, record in sorted(status.get("hands", {}).items()))
                identity = (self.controls.task_generation, self._task_generation, self.backend.epoch,
                            carried, status["retained_or_unknown"], hands)
                if self._confirmation is None or identity != self._confirmation_identity:
                    self._confirmation = uuid.uuid4().hex
                    self._confirmation_identity = identity
            return {**status, "handover_token": self._confirmation}

    def _status(self):
        if self.carried():
            probe = getattr(self.backend, "retention_status", None)
            try:
                status = probe() if probe is not None else {"hands": {}}
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- known carry still requires retention
                status = {"hands": {}, "passive_status_error": str(error)}
            return {**status, "retained_or_unknown": True, "reason": "object_still_carried"}
        if self._ended:
            return {"retained_or_unknown": False, "reason": "session_end_accepted", "hands": {}}
        if self._handover_confirmed:
            return {"retained_or_unknown": False, "reason": "operator_confirmed_object_removed", "hands": {}}
        try:
            probe = getattr(self.backend, "retention_status", None)
            if probe is not None:
                status = probe()
                if type(status.get("retained_or_unknown")) is not bool:
                    raise ValueError("Invalid passive hand-retention status")
                return status
            # The optional REAL-policy SIM composition uses measured hands.
            packet = self.backend.measure()
            if not packet.hands:
                raise ValueError("No measured hand state")
            hands = {side: {"closure": float(hand.closure)} for side, hand in packet.hands.items()}
            if any(not math.isfinite(hand["closure"]) for hand in hands.values()):
                raise ValueError("Invalid measured hand closure")
            unknown = any(hand.closure > .15 for hand in packet.hands.values())
            return {"retained_or_unknown": unknown,
                    "reason": "untracked_closed_hand" if unknown else "measured_open_hands", "hands": hands}
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- failed observations cannot authorize release
            return {"retained_or_unknown": True, "reason": f"hand_state_unavailable: {error}", "hands": {}}

    def end(self):
        self.backend.stop(self.backend.epoch)
        status = self.status()
        if status["retained_or_unknown"]:
            self.recorder.event("end_refused", reason=status["reason"], carried=self.carried(), retention=status)
            self.controls.finish_end(accepted=False)
            self.update("Stopped", "An object may still be held. Place it or confirm a completed handover before ending.")
            return False
        self.recorder.event("session_end", reason="operator", epoch=self.backend.epoch,
                            retention=status)
        self._ended = True
        self.controls.finish_end(accepted=True)
        return True

    def handover(self, confirmation):
        # Submission and acknowledgment are serialized even when UI and Session race.
        with self.controls.task_lock:
            status = self.status()
            if self.task_active() or not confirmation or confirmation != status["handover_token"]:
                self.recorder.event("handover_refused", reason=(
                    "task_active" if self.task_active() else "stale_confirmation"))
                self.update("Stopped", "Task or load state changed. Confirm the completed handover again.")
                return False
            self.backend.stop(self.backend.epoch)
            self.recorder.event("operator_handover", carried=self.carried().copy(),
                                retention=status, verification="operator_confirmed_object_removed")
            self.carried().clear()
            self._handover_confirmed = True
            self._confirmation = None
            if self.target_retention is not None:
                self.target_retention(self.carried())
            self.controls.cancel.clear()
            self.update("Ready", "Handover recorded. The grippers were not commanded to open.")
            return True
