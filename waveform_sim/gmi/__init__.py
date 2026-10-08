"""Receiver-dependent GMI scoring and finite transceiver libraries.

The package is intentionally independent of Qt, GNU Radio and UHD so the
mathematical core can be reproduced and tested without radio hardware.
"""
from .model import (
    ChannelState,
    ExperimentResources,
    GMIExperiment,
    GMIResult,
    ReceiverSpec,
    SelectionDecision,
    TokenScore,
    TransceiverLibrary,
    WaveformToken,
)

__all__ = [
    "ChannelState",
    "ExperimentResources",
    "GMIExperiment",
    "GMIResult",
    "ReceiverSpec",
    "SelectionDecision",
    "TokenScore",
    "TransceiverLibrary",
    "WaveformToken",
]
