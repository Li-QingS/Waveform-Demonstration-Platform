"""Threaded display adapter for the deterministic NTN OFDM experiment."""

import threading
import time

from ..simulation.ofdm_ntn import simulate_frame


class OfdmNtnRunner:
    def __init__(self, config):
        self.config = config.validated()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._snapshot = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="ofdm-ntn-live")
        self._thread.start()

    def _run(self):
        errors = bits = failed = 0
        started = time.monotonic()
        frame_index = 0
        while not self._stop.is_set():
            try:
                frame = simulate_frame(self.config, frame_index)
                if self._stop.is_set():
                    break
                errors += int(frame["bit_errors"])
                bits += int(frame["total_bits"])
                failed += int(frame["frame_error"])
                frame_index += 1
                frame.update(
                    cumulative_errors=errors, cumulative_bits=bits,
                    cumulative_ber=errors / max(bits, 1),
                    cumulative_fer=failed / frame_index, frames=frame_index,
                    elapsed_s=time.monotonic() - started,
                )
                with self._lock:
                    self._snapshot = frame
            except Exception as exc:
                with self._lock:
                    self._snapshot = {"error": f"{type(exc).__name__}: {exc}"}
                break
            self._stop.wait(.08)

    def snapshot(self):
        # Frame arrays are never mutated after publication.
        with self._lock:
            return dict(self._snapshot)

    def stop(self):
        self._stop.set()

    def wait(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
