"""
Block Blast Bot v2.0 - Edge-Based Detection Method

Fallback detection using multi-channel edge detection.

Based on v1.0 lessons:
- Edge detection PARTIAL - loose bounding boxes
- Use as fallback when LAB methods fail
- Multi-channel Canny on B, G, R channels
"""

from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import CalibrationData, GridCalibration
from block_blast_bot.game_types import DetectionResult


class EdgeBasedDetector:
    """
    Board detection using multi-channel edge detection.

    Algorithm:
    1. Run Canny on each B, G, R channel
    2. Combine edges with OR
    3. Dilate to connect edges
    4. Fill contours
    5. Sample at cell centers

    Note: This method produces "loose" bounding boxes and is used
    as a fallback when LAB methods fail.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        dilate_iterations: int = 2,
        morph_kernel_size: int = 5,
    ) -> None:
        """
        Initialize edge-based detector.

        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            dilate_iterations: Number of dilation iterations
            morph_kernel_size: Kernel size for morphological operations
        """
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._dilate_iterations = dilate_iterations
        self._morph_kernel_size = morph_kernel_size

    def detect_board(
        self,
        frame: NDArray[np.uint8],
        calibration: CalibrationData,
    ) -> DetectionResult:
        """
        Detect board state using edge detection.

        Args:
            frame: BGR frame from capture
            calibration: Calibration data with grid bounds and cell centers

        Returns:
            DetectionResult with board bitmask and debug visualization
        """
        if calibration.grid is None:
            raise ValueError("Calibration missing grid data")

        grid = calibration.grid

        # Step 1: Run Canny on each channel
        edges = self._multi_channel_canny(frame)

        # Step 2: Dilate to connect edges
        kernel = np.ones((self._morph_kernel_size, self._morph_kernel_size), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=self._dilate_iterations)

        # Step 3: Find and fill contours
        filled = self._fill_contours(dilated)

        # Step 4: Apply morphological close to bridge gaps
        closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

        # Step 5: Sample at cell centers
        board_bitmask = self._sample_cell_centers(closed, grid)

        # Step 6: Create debug visualization
        debug_frame = self._create_debug_visualization(
            frame, closed, grid, board_bitmask
        )

        # Calculate confidence (edge-based is less reliable)
        confidence = self._calculate_confidence(closed, grid)

        return DetectionResult(
            board=board_bitmask,
            method="edge_based",
            confidence=confidence,
            debug_frame=debug_frame,
        )

    def _multi_channel_canny(
        self,
        frame: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """
        Run Canny edge detection on each B, G, R channel and combine.

        Args:
            frame: BGR frame

        Returns:
            Combined edge map
        """
        b, g, r = cv2.split(frame)

        edges_b = cv2.Canny(b, self._canny_low, self._canny_high)
        edges_g = cv2.Canny(g, self._canny_low, self._canny_high)
        edges_r = cv2.Canny(r, self._canny_low, self._canny_high)

        # Combine with OR
        combined = cv2.bitwise_or(edges_b, edges_g)
        combined = cv2.bitwise_or(combined, edges_r)

        return combined

    def _fill_contours(
        self,
        edges: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """
        Find contours and fill them.

        Args:
            edges: Edge map

        Returns:
            Filled mask
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filled = np.zeros_like(edges)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        return filled

    def _sample_cell_centers(
        self,
        mask: NDArray[np.uint8],
        grid: GridCalibration,
    ) -> int:
        """
        Sample mask at cell centers to build board bitmask.

        Args:
            mask: Binary mask of detected regions
            grid: Grid calibration with cell centers

        Returns:
            u64 bitmask where bit = 1 if cell is filled
        """
        board = 0

        for row in range(8):
            for col in range(8):
                cx, cy = grid.get_cell_center(row, col)

                # Ensure coordinates are within mask bounds
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx] > 0:
                        bit_position = row * 8 + col
                        board |= 1 << bit_position

        return board

    def _create_debug_visualization(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        grid: GridCalibration,
        board_bitmask: int,
    ) -> NDArray[np.uint8]:
        """
        Create debug visualization showing detection result.

        Args:
            frame: Original BGR frame
            mask: Binary detection mask
            grid: Grid calibration
            board_bitmask: Detected board state

        Returns:
            BGR debug frame
        """
        # Create overlay
        debug = frame.copy()

        # Draw mask as semi-transparent overlay
        mask_colored = np.zeros_like(debug)
        mask_colored[:, :, 2] = mask  # Red channel
        cv2.addWeighted(debug, 0.7, mask_colored, 0.3, 0, debug)

        # Draw grid bounds
        x, y, w, h = grid.bounds
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw cell centers with filled/unfilled indication
        for row in range(8):
            for col in range(8):
                cx, cy = grid.get_cell_center(row, col)
                is_filled = bool(board_bitmask & (1 << (row * 8 + col)))

                color = (0, 0, 255) if is_filled else (255, 255, 255)
                cv2.circle(debug, (cx, cy), 3, color, -1)

        return debug

    def _calculate_confidence(
        self,
        mask: NDArray[np.uint8],
        grid: GridCalibration,
    ) -> float:
        """
        Calculate confidence score for detection.

        Edge-based detection is less reliable, so confidence is lower.

        Args:
            mask: Binary detection mask
            grid: Grid calibration

        Returns:
            Confidence score (0.0 to 0.7 for edge-based)
        """
        # Count how many cell centers have clear values
        clear_count = 0
        total_count = 0

        for row in range(8):
            for col in range(8):
                cx, cy = grid.get_cell_center(row, col)

                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    total_count += 1

                    # Check if the area around the center is uniform
                    y_start = max(0, cy - 3)
                    y_end = min(mask.shape[0], cy + 4)
                    x_start = max(0, cx - 3)
                    x_end = min(mask.shape[1], cx + 4)

                    local_area = mask[y_start:y_end, x_start:x_end]
                    if local_area.size > 0:
                        if np.all(local_area == local_area[0, 0]):
                            clear_count += 1

        if total_count == 0:
            return 0.0

        # Edge-based has max confidence of 0.7 (less reliable)
        return min(0.7, clear_count / total_count)


def detect_board_edge_based(
    frame: NDArray[np.uint8],
    calibration: CalibrationData,
) -> DetectionResult:
    """
    Convenience function for edge-based detection.

    Args:
        frame: BGR frame
        calibration: Calibration data

    Returns:
        DetectionResult
    """
    detector = EdgeBasedDetector()
    return detector.detect_board(frame, calibration)


__all__ = [
    "EdgeBasedDetector",
    "detect_board_edge_based",
]
