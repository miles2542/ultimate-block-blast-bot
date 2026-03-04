"""
Block Blast Bot v2.0 - Cascade Detection Method

Multi-method cascade for robust detection.

Based on v1.0 lessons:
- Multi-method cascade REQUIRED - no single method works universally
- Try methods in order: LAB+Otsu, LAB+Hard, Edge-based
- Validate each result before returning
"""

from typing import Optional, List


import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.game_types import DetectionResult
from block_blast_bot.vision.methods.lab_distance import LABDistanceDetector
from block_blast_bot.vision.methods.edge_based import EdgeBasedDetector


class DetectionCascade:
    """
    Cascade of detection methods.

    Tries methods in order until one succeeds with valid result.

    Order:
    1. LAB + Otsu (primary) - best for most themes
    2. LAB + Hard threshold (fallback) - for themes where Otsu fails
    3. Edge-based (last resort) - less reliable but works in edge cases
    """

    # Minimum confidence threshold for accepting a result
    MIN_CONFIDENCE = 0.3

    def __init__(
        self,
        lab_threshold: int = 24,
        morph_kernel_size: int = 5,
    ) -> None:
        """
        Initialize detection cascade.

        Args:
            lab_threshold: Hard threshold for LAB fallback method
            morph_kernel_size: Kernel size for morphological operations
        """
        self._lab_threshold = lab_threshold
        self._morph_kernel_size = morph_kernel_size

        # Initialize detectors
        self._lab_otsu = LABDistanceDetector(
            use_otsu=True,
            morph_kernel_size=morph_kernel_size,
        )
        self._lab_hard = LABDistanceDetector(
            use_otsu=False,
            hard_threshold=lab_threshold,
            morph_kernel_size=morph_kernel_size,
        )
        self._edge_based = EdgeBasedDetector(
            morph_kernel_size=morph_kernel_size,
        )

    def detect_board(
        self,
        frame: NDArray[np.uint8],
        calibration: CalibrationData,
    ) -> DetectionResult:
        """
        Run detection cascade on frame.

        Args:
            frame: BGR frame from capture
            calibration: Calibration data

        Returns:
            DetectionResult from first successful method

        Raises:
            DetectionError: If all methods fail
        """
        if calibration.grid is None:
            raise ValueError("Calibration missing grid data")

        # Try each method in order
        methods = [
            ("LAB + Otsu", self._lab_otsu.detect_board),
            ("LAB + Hard", self._lab_hard.detect_board),
            ("Edge-based", self._edge_based.detect_board),
        ]

        last_error: Optional[Exception] = None

        for method_name, detect_func in methods:
            try:
                result = detect_func(frame, calibration)

                # Validate result
                if self._validate_result(result):
                    return result

            except Exception as e:
                last_error = e
                continue

        # All methods failed
        raise DetectionError(f"All detection methods failed. Last error: {last_error}")

    def _validate_result(self, result: DetectionResult) -> bool:
        """
        Validate detection result.

        Args:
            result: Detection result to validate

        Returns:
            True if result is valid, False otherwise
        """
        # Check confidence threshold
        if result.confidence < self.MIN_CONFIDENCE:
            return False

        # Check that board has valid number of filled cells (0-64)
        filled_count = result.filled_count
        if filled_count < 0 or filled_count > 64:
            return False

        # Additional validation: check for impossible states
        # (e.g., all cells filled is technically possible but suspicious)
        # For now, we accept any result with reasonable confidence

        return True


class DetectionError(Exception):
    """Exception raised when all detection methods fail."""

    pass


def detect_board_cascade(
    frame: NDArray[np.uint8],
    calibration: CalibrationData,
) -> DetectionResult:
    """
    Convenience function for cascade detection.

    Args:
        frame: BGR frame
        calibration: Calibration data

    Returns:
        DetectionResult

    Raises:
        DetectionError: If all methods fail
    """
    cascade = DetectionCascade()
    return cascade.detect_board(frame, calibration)


__all__ = [
    "DetectionCascade",
    "DetectionError",
    "detect_board_cascade",
]
