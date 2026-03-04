"""
Block Blast Bot v2.0 - Calibration Module

Interactive calibration system for game detection.
"""

from block_blast_bot.calibration.data import (
    CalibrationData,
    GridCalibration,
    SlotCalibration,
    DetectionCalibration,
    BoardCalibration,  # Legacy alias
    PieceSlotCalibration,  # Legacy alias
)
from block_blast_bot.calibration.calibrator import Calibrator

__all__ = [
    "CalibrationData",
    "GridCalibration",
    "SlotCalibration",
    "DetectionCalibration",
    "BoardCalibration",
    "PieceSlotCalibration",
    "Calibrator",
]
