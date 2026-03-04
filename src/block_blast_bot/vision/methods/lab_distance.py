"""
Block Blast Bot v2.0 - LAB Distance Detection Method

Primary detection method using LAB color space distance.

Based on v1.0 lessons:
- HSV thresholding FAILED - theme-dependent
- Otsu alone PARTIAL - assumes bimodal histogram
- LAB color distance + Otsu WORKS - primary method
"""

from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.game_types import DetectionResult


class LABDistanceDetector:
    """
    Board detection using Variance-Anchored LAB Classification.

    Algorithm (from v1.0 success):
    1. Extract 64 cell regions based on calibration
    2. Compute variance of a small inner square for each cell
       (Empty cells are 2D flat color with near 0 variance.
        Filled cells are 3D with bevels, highlights, shadows = high variance)
    3. Find all cells with variance < threshold (definitively empty cells)
    4. Compute median LAB color of those empty cells -> True Background
    5. Compute distance of ALL cells to True Background
    6. Classify cells as filled if distance > threshold
    """

    def __init__(
        self,
        use_otsu: bool = False,  # Kept for API compatibility, but ignored by this method
        hard_threshold: int = 24,
        variance_threshold: float = 15.0,  # Threshold for flat color vs 3D texture
        morph_kernel_size: int = 5,  # Kept for API compatibility, unused
    ) -> None:
        """
        Initialize Variance-Anchored LAB detector.

        Args:
            use_otsu: Ignored, kept for API compatibility.
            hard_threshold: LAB distance threshold.
            variance_threshold: Variance threshold below which a cell is considered completely empty.
            morph_kernel_size: Ignored, kept for API compatibility.
        """
        self._hard_threshold = hard_threshold
        self._variance_threshold = variance_threshold

    def detect_board(
        self,
        frame: NDArray[np.uint8],
        calibration: CalibrationData,
    ) -> DetectionResult:
        """
        Detect board state using Variance-Anchored LAB Classification.

        Args:
            frame: BGR frame from capture
            calibration: Calibration data with grid bounds and cell centers

        Returns:
            DetectionResult with board bitmask and debug visualization
        """
        if calibration.grid is None:
            raise ValueError("Calibration missing grid data")

        grid = calibration.grid
        cell_size = grid.cell_size[0]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        cell_variances = []
        cell_center_labs = []

        # Inset for variance calculation: look at the inner part of cell, avoiding edges
        inset = max(2, int(cell_size * 0.15))
        half_cell = cell_size // 2

        for row in range(8):
            for col in range(8):
                cx, cy = grid.get_cell_center(row, col)

                # Region A: Inner cell (for structural variance)
                y1 = max(0, cy - half_cell + inset)
                y2 = min(frame.shape[0], cy + half_cell - inset)
                x1 = max(0, cx - half_cell + inset)
                x2 = min(frame.shape[1], cx + half_cell - inset)

                inner_gray = frame_gray[y1:y2, x1:x2]
                var = np.var(inner_gray) if inner_gray.size > 0 else 0
                cell_variances.append(var)

                # Region B: Exact center patch (for pure face color, ignoring bevels)
                patch_lab = frame_lab[max(0, cy - 4) : cy + 5, max(0, cx - 4) : cx + 5]
                if patch_lab.size > 0:
                    cell_center_labs.append(np.median(patch_lab, axis=(0, 1)))
                else:
                    cell_center_labs.append(np.array([0, 0, 0], dtype=np.float32))

        # Discover the true background dynamically
        # Empty cells are flat (variance near 0)
        empty_cell_labs = []
        for i in range(64):
            if cell_variances[i] < self._variance_threshold:
                empty_cell_labs.append(cell_center_labs[i])

        board_bitmask = 0
        debug_frame = frame.copy()

        # Handle extreme edge cases
        if len(empty_cell_labs) == 0:
            # No flat cells -> Board is 100% full
            board_bitmask = 0xFFFFFFFFFFFFFFFF

            # Draw debug for all filled
            for row in range(8):
                for col in range(8):
                    cx, cy = grid.get_cell_center(row, col)
                    cv2.circle(debug_frame, (cx, cy), 6, (0, 255, 0), -1)

            return DetectionResult(
                board=board_bitmask,
                method="variance_lab",
                confidence=1.0,
                debug_frame=debug_frame,
            )

        elif len(empty_cell_labs) == 64:
            # All cells flat -> Board is 100% empty
            board_bitmask = 0

            # Draw debug for all empty
            for row in range(8):
                for col in range(8):
                    cx, cy = grid.get_cell_center(row, col)
                    cv2.circle(debug_frame, (cx, cy), 3, (80, 80, 80), 1)

            return DetectionResult(
                board=board_bitmask,
                method="variance_lab",
                confidence=1.0,
                debug_frame=debug_frame,
            )

        # Establish the background baseline
        true_bg_lab = np.median(empty_cell_labs, axis=0)

        # Classify all 64 cells using LAB distance threshold
        for row in range(8):
            for col in range(8):
                i = row * 8 + col
                cx, cy = grid.get_cell_center(row, col)

                dist = np.linalg.norm(cell_center_labs[i] - true_bg_lab)

                if dist > self._hard_threshold:
                    board_bitmask |= 1 << i
                    cv2.circle(
                        debug_frame, (cx, cy), 6, (0, 0, 255), -1
                    )  # Red for filled
                    cv2.circle(debug_frame, (cx, cy), 8, (0, 255, 255), 1)
                else:
                    cv2.circle(
                        debug_frame, (cx, cy), 3, (255, 255, 255), 1
                    )  # White for empty

        # Calculate confidence metric based on variance separability
        confidence = self._calculate_confidence(cell_variances, board_bitmask)

        return DetectionResult(
            board=board_bitmask,
            method="variance_lab",
            confidence=confidence,
            debug_frame=debug_frame,
        )

    def _calculate_confidence(
        self,
        variances: list[float],
        board_bitmask: int,
    ) -> float:
        """
        Calculate confidence score based on how well the variances separate.
        """
        # A simple confidence metric: if we had clear empty cells, confidence is high
        empty_vars = []
        filled_vars = []

        for i in range(64):
            if board_bitmask & (1 << i):
                filled_vars.append(variances[i])
            else:
                empty_vars.append(variances[i])

        # If we have both, check how well separated they are
        if empty_vars and filled_vars:
            max_empty = max(empty_vars)
            min_filled = min(filled_vars)
            if min_filled > max_empty:
                return 1.0
            else:
                return 0.8  # Some overlap, but classification used LAB color, so still okay

        return 1.0


def detect_board_lab_otsu(
    frame: NDArray[np.uint8],
    calibration: CalibrationData,
) -> DetectionResult:
    """
    Legacy convenience function, redirects to variance detector.
    """
    detector = LABDistanceDetector()
    return detector.detect_board(frame, calibration)


def detect_board_lab_hard(
    frame: NDArray[np.uint8],
    calibration: CalibrationData,
    hard_threshold: int = 24,
) -> DetectionResult:
    """
    Legacy convenience function, redirects to variance detector.
    """
    detector = LABDistanceDetector(hard_threshold=hard_threshold)
    return detector.detect_board(frame, calibration)


__all__ = [
    "LABDistanceDetector",
    "detect_board_lab_otsu",
    "detect_board_lab_hard",
]
