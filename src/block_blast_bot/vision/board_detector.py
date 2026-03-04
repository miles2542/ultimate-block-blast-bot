"""
Block Blast Bot v2.0 - Board Detector

High-level interface for board detection using multi-method cascade.

Based on v1.0 lessons:
- Multi-method cascade REQUIRED - no single method works universally
- LAB color distance + Otsu WORKS - primary method
- Edge-based as fallback
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.game_types import (
    BoardDetectionResult,
    BoardState,
    Rect,
)
from block_blast_bot.vision.methods.cascade import (
    DetectionCascade,
)


class BoardDetector:
    """
    Board detector using multi-method cascade.

    Provides high-level interface for detecting the 8x8 game board.

    Usage:
        calibration = CalibrationData.load("calibration.json")
        detector = BoardDetector(calibration)
        result = detector.detect(frame)
        print(f"Found {result.filled_count} filled cells")
    """

    def __init__(
        self,
        calibration: CalibrationData,
        lab_threshold: int = 24,
        morph_kernel_size: int = 5,
    ) -> None:
        """
        Initialize board detector.

        Args:
            calibration: Calibration data with grid bounds and cell centers
            lab_threshold: Hard threshold for LAB fallback method
            morph_kernel_size: Kernel size for morphological operations
        """
        if calibration.grid is None:
            raise ValueError("Calibration must include grid data")

        self._calibration = calibration
        self._cascade = DetectionCascade(
            lab_threshold=lab_threshold,
            morph_kernel_size=morph_kernel_size,
        )

    def detect(self, frame: NDArray[np.uint8]) -> BoardDetectionResult:
        """
        Detect board state from frame.

        Args:
            frame: BGR frame from capture

        Returns:
            BoardDetectionResult with board bitmask and filled cells list

        Raises:
            DetectionError: If all detection methods fail
        """
        # Run cascade detection
        result = self._cascade.detect_board(frame, self._calibration)

        # Convert to high-level result
        filled_cells = self._bitmask_to_cells(result.board)

        return BoardDetectionResult(
            board=result.board,
            filled_cells=filled_cells,
            method=result.method,
            confidence=result.confidence,
        )

    def detect_with_debug(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[BoardDetectionResult, NDArray[np.uint8]]:
        """
        Detect board state with debug visualization.

        Args:
            frame: BGR frame from capture

        Returns:
            Tuple of (BoardDetectionResult, debug frame)

        Raises:
            DetectionError: If all detection methods fail
        """
        # Run cascade detection
        result = self._cascade.detect_board(frame, self._calibration)

        # Convert to high-level result
        filled_cells = self._bitmask_to_cells(result.board)

        detection_result = BoardDetectionResult(
            board=result.board,
            filled_cells=filled_cells,
            method=result.method,
            confidence=result.confidence,
        )

        # Get debug frame
        debug_frame = result.debug_frame if result.debug_frame is not None else frame

        return detection_result, debug_frame

    def detect_board_state(self, frame: NDArray[np.uint8]) -> BoardState:
        """
        Detect board state and return as BoardState object.

        This is a convenience method for compatibility with existing code.

        Args:
            frame: BGR frame from capture

        Returns:
            BoardState with 8x8 boolean grid

        Raises:
            DetectionError: If all detection methods fail
        """
        result = self.detect(frame)
        return result.to_board_state()

    def _bitmask_to_cells(self, bitmask: int) -> List[Tuple[int, int]]:
        """
        Convert u64 bitmask to list of (row, col) tuples.

        Args:
            bitmask: u64 bitmask where bit = 1 if cell is filled

        Returns:
            List of (row, col) tuples for filled cells
        """
        cells = []
        for row in range(8):
            for col in range(8):
                bit_position = row * 8 + col
                if bitmask & (1 << bit_position):
                    cells.append((row, col))
        return cells

    @property
    def calibration(self) -> CalibrationData:
        """Get the calibration data used by this detector."""
        return self._calibration

    @property
    def grid_bounds(self) -> Rect:
        """Get the grid bounds as a Rect."""
        if self._calibration.grid is None:
            raise ValueError("Calibration missing grid data")
        x, y, w, h = self._calibration.grid.bounds
        return Rect(x=x, y=y, width=w, height=h)


def create_board_detector(
    calibration_path: str,
) -> BoardDetector:
    """
    Factory function to create a BoardDetector from calibration file.

    Args:
        calibration_path: Path to calibration JSON file

    Returns:
        Configured BoardDetector instance

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        ValueError: If calibration is invalid
    """
    calibration = CalibrationData.load(calibration_path)
    return BoardDetector(calibration)


__all__ = [
    "BoardDetector",
    "create_board_detector",
]
