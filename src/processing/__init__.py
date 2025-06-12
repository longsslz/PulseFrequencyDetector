# src/processing/__init__.py

from .realtime import (
    RealTimePulseProcessor,
    AsyncPulseProcessor,
    MultiChannelProcessor
)
from .utils import (
    sliding_window,
    compute_instant_frequency,
    signal_quality_check,
    filter_signal_realtime,
    resample_signal
)