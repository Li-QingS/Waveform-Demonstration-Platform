"""GNU Radio 流式块工厂（从 fdidm_hardtest.py 搬移，阶段 6c）。

gnuradio 在工厂函数内部懒加载，模块本身可在无 GNU Radio 环境导入。
"""
from __future__ import annotations

import threading

import numpy as np


def make_tdl_channel_block(channel):
    """Create a GNU Radio Python sync_block wrapping NTNTDLChannel."""
    from gnuradio import gr

    class _TDLChannelBlock(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(self, name="ntn_tdl_channel_v34", in_sig=[np.complex64], out_sig=[np.complex64])
            self.channel = channel
            self._channel_lock = threading.RLock()

        def work(self, input_items, output_items):
            # Runtime parameter updates can arrive from the UI thread.  Keep
            # channel.configure()/reset() mutually exclusive with process()
            # so a partial tap-table update cannot leak into one scheduler call.
            with self._channel_lock:
                y = self.channel.process(input_items[0])
            output_items[0][:len(y)] = y
            return len(y)

        def reset_channel(self):
            with self._channel_lock:
                self.channel.reset()

        def configure_channel(self, **kwargs):
            with self._channel_lock:
                self.channel.configure(**kwargs)
                self.channel.reset()

        def channel_summary(self) -> str:
            with self._channel_lock:
                return self.channel.summary()

    return _TDLChannelBlock()


def make_rx_ring_sink(rx_buffer):
    """Fallback bounded RX sink for old GNU Radio builds."""
    from gnuradio import gr

    class _RXRingSink(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(self, name="rx_numpy_ring_sink_fallback", in_sig=[np.complex64], out_sig=[])

        def work(self, input_items, output_items):
            rx_buffer.write(input_items[0])
            return len(input_items[0])

    return _RXRingSink()

