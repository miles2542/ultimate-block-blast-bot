"""
Block Blast Bot v2.0 - Capture Module

Frame capture from scrcpy window using mss.
"""

from block_blast_bot.capture.frame_capture import FrameCapture, create_capture
from block_blast_bot.capture.window_detector import (
    WindowDetector,
    set_dpi_awareness,
    is_dpi_aware,
)
from block_blast_bot.config import CaptureConfig
from block_blast_bot.game_types import WindowInfo

__all__ = [
    "FrameCapture",
    "create_capture",
    "WindowDetector",
    "CaptureConfig",
    "WindowInfo",
    "set_dpi_awareness",
    "is_dpi_aware",
]
