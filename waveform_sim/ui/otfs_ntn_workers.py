"""Bounded, cancellable live adapter for the rectangular OTFS NTN model."""

import threading
import time

from ..simulation.otfs_ntn import simulate_frame


class OtfsNtnRunner:
    """Publish immutable complete frames; retain thread ownership until joined."""

    def __init__(self, config):
        self.config = config.validated()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._snapshot = {}

    def start(self):
        if self._thread is not None:
            raise RuntimeError("An OTFS live runner can only be started once")
        self._thread = threading.Thread(target=self._run, daemon=True, name="otfs-ntn-live")
        self._thread.start()

    def _run(self):
        errors = bits = failed = 0
        started = time.monotonic()
        frame_index = 0
        while not self._stop.is_set():
            try:
                frame = simulate_frame(self.config, frame_index, stop_event=self._stop)
                if self._stop.is_set():
                    break
                errors += int(frame["bit_errors"])
                bits += int(frame["total_bits"])
                failed += int(frame["frame_error"])
                frame_index += 1
                frame.update(
                    cumulative_errors=errors, cumulative_bits=bits,
                    cumulative_ber=errors / bits,
                    cumulative_fer=failed / frame_index, frames=frame_index,
                    elapsed_s=time.monotonic() - started,
                )
                with self._lock:
                    self._snapshot = frame
            except InterruptedError:
                break
            except Exception as exc:
                with self._lock:
                    self._snapshot = {**self._snapshot, "error": f"{type(exc).__name__}: {exc}"}
                break
            self._stop.wait(.08)

    def snapshot(self):
        # Numerical frames are never modified after publication.
        with self._lock:
            return dict(self._snapshot)

    def stop(self):
        self._stop.set()

    def wait(self, timeout=1.5):
        worker = self._thread
        if worker is None or worker.ident is None:
            return
        if worker is threading.current_thread():
            raise RuntimeError("OTFS runner cannot join itself")
        worker.join(timeout)
        if worker.is_alive():
            raise TimeoutError("OTFS 仿真仍在停止，请稍后重试")
