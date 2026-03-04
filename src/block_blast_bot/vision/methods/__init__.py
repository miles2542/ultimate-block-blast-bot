"""
Block Blast Bot v2.0 - Detection Methods

Various detection methods for board and piece detection.

Based on v1.0 lessons:
- Multi-method cascade REQUIRED - no single method works universally
- LAB color distance + Otsu WORKS - primary method
- Edge-based as fallback
"""

from block_blast_bot.vision.methods.cascade import (
    DetectionCascade,
    DetectionError,
    detect_board_cascade,
)
from block_blast_bot.vision.methods.edge_based import (
    EdgeBasedDetector,
    detect_board_edge_based,
)
from block_blast_bot.vision.methods.lab_distance import (
    LABDistanceDetector,
    detect_board_lab_hard,
    detect_board_lab_otsu,
)

__all__ = [
    # Cascade
    "DetectionCascade",
    "DetectionError",
    "detect_board_cascade",
    # LAB methods
    "LABDistanceDetector",
    "detect_board_lab_otsu",
    "detect_board_lab_hard",
    # Edge-based method
    "EdgeBasedDetector",
    "detect_board_edge_based",
]
