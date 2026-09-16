"""FDIDM FEC：卷积编码/交织/校验（从 fdidm_hardtest.py 搬移，阶段 6b）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class FECMixin:
    _conv_incoming_cache = None

    @staticmethod
    def _parity_u32(x: int) -> int:
        return int(int(x).bit_count() & 1)

    @classmethod
    def _conv_next_state_output(cls, memory: int, bit: int) -> Tuple[int, Tuple[int, int]]:
        """K=7, rate-1/2 convolutional encoder trellis.

        The current input bit is bit0 of the shift register and older bits are
        shifted upward.  The Viterbi decoder below uses the exact same trellis,
        so the bit order is self-consistent even though different textbooks may
        print the NASA 171/133 polynomials in the opposite visual direction.
        """
        K = 7
        mask = (1 << (K - 1)) - 1
        reg = (int(bit) & 1) | ((int(memory) & mask) << 1)
        g0, g1 = 0o171, 0o133
        out = (cls._parity_u32(reg & g0), cls._parity_u32(reg & g1))
        return int(reg & mask), out

    @classmethod
    def _conv_encode_bits(cls, bits: np.ndarray, flush: bool = True) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if flush:
            b = np.concatenate([b, np.zeros(6, dtype=np.int8)])
        memory = 0
        out = np.empty(b.size * 2, dtype=np.int8)
        j = 0
        for bit in b:
            memory, pair = cls._conv_next_state_output(memory, int(bit))
            out[j] = pair[0]
            out[j + 1] = pair[1]
            j += 2
        return out

    @classmethod
    def _conv_decode_bits(cls, coded_bits: np.ndarray, decoded_len: int, flushed: bool = True) -> np.ndarray:
        r = np.asarray(coded_bits, dtype=np.int8).reshape(-1)
        if r.size < 2:
            return np.zeros(0, dtype=np.int8)
        if r.size % 2:
            r = r[:-1]
        num_steps = r.size // 2
        num_states = 64
        inf = 10 ** 9
        metrics = np.full(num_states, inf, dtype=np.int32)
        metrics[0] = 0
        prev_state = np.zeros((num_steps, num_states), dtype=np.uint8)
        prev_bit = np.zeros((num_steps, num_states), dtype=np.uint8)
        # Store the two incoming branches for every destination state.  The
        # old implementation walked 64 states x 2 bits in Python for every
        # coded symbol and was called up to four times for QPSK phase trials;
        # on Windows this alone cost ~0.45 s per RX frame and starved the GNU
        # Radio probe.  The trellis and traceback are unchanged, only the
        # per-step add/compare/select is vectorized.
        incoming = getattr(cls, "_conv_incoming_cache", None)
        if incoming is None:
            predecessors = [[] for _ in range(num_states)]
            for s in range(num_states):
                for b in (0, 1):
                    ns, pair = cls._conv_next_state_output(s, b)
                    predecessors[int(ns)].append((int(s), int(b), int(pair[0]), int(pair[1])))
            incoming_states = np.asarray(
                [[branch[0] for branch in branches] for branches in predecessors], dtype=np.uint8
            )
            incoming_bits = np.asarray(
                [[branch[1] for branch in branches] for branches in predecessors], dtype=np.uint8
            )
            incoming_outputs = np.asarray(
                [[[branch[2], branch[3]] for branch in branches] for branches in predecessors], dtype=np.uint8
            )
            incoming = (incoming_states, incoming_bits, incoming_outputs)
            cls._conv_incoming_cache = incoming
        incoming_states, incoming_bits, incoming_outputs = incoming
        state_rows = np.arange(num_states, dtype=np.int64)
        for t in range(num_steps):
            received = r[2 * t:2 * t + 2].astype(np.uint8, copy=False)
            branch_distance = np.count_nonzero(incoming_outputs != received[None, None, :], axis=2)
            candidates = metrics[incoming_states] + branch_distance
            choice = np.argmin(candidates, axis=1)
            metrics = candidates[state_rows, choice].astype(np.int32, copy=False)
            prev_state[t] = incoming_states[state_rows, choice]
            prev_bit[t] = incoming_bits[state_rows, choice]
        state = 0 if flushed else int(np.argmin(metrics))
        decoded = np.zeros(num_steps, dtype=np.int8)
        for t in range(num_steps - 1, -1, -1):
            b = int(prev_bit[t, state])
            decoded[t] = b
            state = int(prev_state[t, state])
        if flushed and decoded.size >= 6:
            decoded = decoded[:-6]
        decoded_len = int(max(0, decoded_len))
        if decoded.size >= decoded_len:
            return decoded[:decoded_len].astype(np.int8, copy=True)
        return np.concatenate([decoded.astype(np.int8), np.zeros(decoded_len - decoded.size, dtype=np.int8)])

    def _coding_permutation(self, n: int) -> np.ndarray:
        n = int(max(0, n))
        if n <= 1:
            return np.arange(n, dtype=np.int64)
        seed = (self.PILOT_SEED ^ 0xC0DEC0DE ^ (int(self.M) << 16) ^ (int(self.N) << 8) ^ n) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return rng.permutation(n).astype(np.int64)

    def _apply_bit_interleaver(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if not bool(getattr(self, "coding_interleaver", True)) or b.size <= 1:
            return b.astype(np.int8, copy=True)
        return b[self._coding_permutation(b.size)].astype(np.int8, copy=True)

    def _remove_bit_interleaver(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.int8).reshape(-1)
        if not bool(getattr(self, "coding_interleaver", True)) or b.size <= 1:
            return b.astype(np.int8, copy=True)
        perm = self._coding_permutation(b.size)
        out = np.zeros_like(b, dtype=np.int8)
        out[perm] = b
        return out

    def _encode_phy_bits(self, app_frame_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(app_frame_bits, dtype=np.int8).reshape(-1)
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return bits.astype(np.int8, copy=True)
        if scheme == "conv12":
            coded = self._conv_encode_bits(bits, flush=True)
            return self._apply_bit_interleaver(coded)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _decode_phy_bits(self, hard_bits: np.ndarray, decoded_len: int) -> np.ndarray:
        bits = np.asarray(hard_bits, dtype=np.int8).reshape(-1)
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            decoded_len = int(max(0, decoded_len))
            if bits.size >= decoded_len:
                return bits[:decoded_len].astype(np.int8, copy=True)
            return np.concatenate([bits.astype(np.int8), np.zeros(decoded_len - bits.size, dtype=np.int8)])
        if scheme == "conv12":
            deint = self._remove_bit_interleaver(bits)
            return self._conv_decode_bits(deint, decoded_len=decoded_len, flushed=True)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _coded_len_for_uncoded_len(self, uncoded_bits: int) -> int:
        uncoded_bits = int(max(0, uncoded_bits))
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return uncoded_bits
        if scheme == "conv12":
            return 2 * (uncoded_bits + 6)
        raise ValueError(f"unsupported coding_scheme={scheme}")

    def _coding_summary(self) -> str:
        scheme = str(getattr(self, "coding_scheme", "none")).lower()
        if scheme == "none":
            return "none"
        if scheme == "conv12":
            return "conv12(rate=1/2,K=7,gens=171/133 octal" + (",interleaved" if self.coding_interleaver else "") + ")"
        return scheme

    def _max_payload_bytes_for_current_phy(self) -> int:
        """Maximum UTF-8 payload bytes that fit after PHY coding."""
        cap = self._max_data_bits_capacity()
        # APP frame adds magic(4)+length(4)+CRC32(4) = 12 bytes.
        best = 0
        for payload_bytes in range(0, max(1, cap // 8) + 1):
            uncoded = 8 * (payload_bytes + 12)
            if self._coded_len_for_uncoded_len(uncoded) <= cap:
                best = payload_bytes
            else:
                break
        return int(max(0, best))
