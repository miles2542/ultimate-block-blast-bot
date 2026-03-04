"""
Block Blast Bot v2.0 - Vision Module

Vision pipeline for game state detection.

Based on v1.0 lessons:
- Multi-method cascade REQUIRED - no single method works universally
- LAB color distance + Otsu WORKS - primary method
- Edge-based as fallback
"""

from block_blast_bot.game_types import DetectionResult, BoardDetectionResult
from block_blast_bot.vision.board_detector import (
    BoardDetector,
    create_board_detector,
)
from block_blast_bot.vision.piece_detector import (
    PieceDetector,
    create_piece_detector,
)
from block_blast_bot.vision.methods.cascade import (
    DetectionCascade,
    DetectionError,
    detect_board_cascade,
)
from block_blast_bot.vision.methods.lab_distance import (
    LABDistanceDetector,
    detect_board_lab_otsu,
    detect_board_lab_hard,
)
from block_blast_bot.vision.methods.edge_based import (
    EdgeBasedDetector,
    detect_board_edge_based,
)

__all__ = [
    # Types
    "DetectionResult",
    "BoardDetectionResult",
    # High-level interface
    "BoardDetector",
    "create_board_detector",
    "PieceDetector",
    "create_piece_detector",
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
