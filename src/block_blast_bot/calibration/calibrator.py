"""
Block Blast Bot v2.0 - Calibrator

Auto-detection and validation of game regions for calibration.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import (
    CalibrationData,
    GridCalibration,
    SlotCalibration,
    DetectionCalibration,
)
from block_blast_bot.capture.frame_capture import FrameCapture

logger = logging.getLogger(__name__)


class Calibrator:
    """
    Calibration logic for auto-detecting game regions.

    Provides methods for automatic detection of grid and slots,
    as well as validation of calibration data.
    """

    def __init__(self, capture: Optional[FrameCapture] = None) -> None:
        """
        Initialize calibrator.

        Args:
            capture: FrameCapture instance for live preview
        """
        self._capture = capture
        self._calibration_data: Optional[CalibrationData] = None

    def auto_detect_grid(
        self,
        frame: NDArray[np.uint8],
        min_grid_size: int = 200,
    ) -> Optional[GridCalibration]:
        """
        Auto-detect the 8x8 game grid using edge detection.

        Uses Canny edge detection and Hough line transform to find
        the grid boundaries, then validates the 8x8 structure.

        Args:
            frame: BGR frame from scrcpy window
            min_grid_size: Minimum expected grid size in pixels

        Returns:
            GridCalibration if detected, None otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Look for the largest square-ish contour that could be the grid
        best_contour = None
        best_score = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if too small
            if w < min_grid_size or h < min_grid_size:
                continue

            # Skip if not roughly square (allow 20% aspect ratio deviation)
            aspect_ratio = w / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue

            # Score based on size and squareness
            area = w * h
            squareness = 1.0 - abs(1.0 - aspect_ratio)
            score = area * squareness

            if score > best_score:
                best_score = score
                best_contour = (x, y, w, h)

        if best_contour is None:
            logger.warning("Could not detect grid - no suitable contour found")
            return None

        x, y, w, h = best_contour

        # Calculate cell size (8x8 grid)
        cell_w = w // 8
        cell_h = h // 8

        # Validate cell size is reasonable
        if cell_w < 20 or cell_h < 20:
            logger.warning(f"Cell size too small: {cell_w}x{cell_h}")
            return None

        # Adjust bounds to be exactly 8 cells
        grid_w = cell_w * 8
        grid_h = cell_h * 8

        grid = GridCalibration(
            bounds=(x, y, grid_w, grid_h),
            cell_size=(cell_w, cell_h),
        )

        logger.info(
            f"Detected grid at ({x}, {y}) size {grid_w}x{grid_h}, cell size {cell_w}x{cell_h}"
        )
        return grid

    def auto_detect_slots(
        self,
        frame: NDArray[np.uint8],
        grid: GridCalibration,
        expected_slots: int = 3,
    ) -> List[SlotCalibration]:
        """
        Auto-detect piece slots below the grid.

        Slots are typically aligned horizontally below the grid.
        This method looks for rectangular regions that could be slots.

        Args:
            frame: BGR frame from scrcpy window
            grid: Detected grid calibration
            expected_slots: Number of slots to detect (default 3)

        Returns:
            List of SlotCalibration objects
        """
        height, width = frame.shape[:2]
        grid_x, grid_y, grid_w, grid_h = grid.bounds
        cell_w, cell_h = grid.cell_size

        # Slots are typically below the grid, centered horizontally
        # They should be roughly the same width as grid cells
        slots: List[SlotCalibration] = []

        # Estimate slot region (below grid, with some margin)
        slot_region_y = grid_y + grid_h + int(cell_h * 0.5)
        slot_region_h = min(int(cell_h * 2.5), height - slot_region_y)

        if slot_region_h <= 0:
            logger.warning("Not enough space below grid for slots")
            return slots

        # Extract slot region
        slot_region = frame[slot_region_y : slot_region_y + slot_region_h, :]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(slot_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in slot region
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours that could be slots
        slot_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Slots should be roughly square to slightly rectangular
            aspect_ratio = w / h if h > 0 else 0

            # Expected slot size is roughly 2-4 cells wide
            min_slot_size = cell_w * 1.5
            max_slot_size = cell_w * 4

            if min_slot_size < w < max_slot_size and min_slot_size < h < max_slot_size:
                if 0.5 < aspect_ratio < 2.0:
                    # Convert to full frame coordinates
                    abs_x = x
                    abs_y = y + slot_region_y
                    slot_candidates.append((abs_x, abs_y, w, h))

        # Sort by x position (left to right)
        slot_candidates.sort(key=lambda s: s[0])

        # If we found more than expected, take the most likely ones
        if len(slot_candidates) >= expected_slots:
            # Take the first 'expected_slots' that are roughly evenly spaced
            slots = []
            for i, (x, y, w, h) in enumerate(slot_candidates[:expected_slots]):
                center_x = x + w // 2
                center_y = y + h // 2
                slots.append(
                    SlotCalibration(
                        center=(center_x, center_y),
                        bounds=(x, y, w, h),
                    )
                )
        elif len(slot_candidates) > 0:
            # Use what we found
            for x, y, w, h in slot_candidates:
                center_x = x + w // 2
                center_y = y + h // 2
                slots.append(
                    SlotCalibration(
                        center=(center_x, center_y),
                        bounds=(x, y, w, h),
                    )
                )

        # If we didn't find enough slots, estimate positions based on grid
        if len(slots) < expected_slots:
            logger.info(f"Only found {len(slots)} slots, estimating remaining")
            slots = self._estimate_slot_positions(grid, expected_slots, frame.shape[:2])

        logger.info(f"Detected {len(slots)} slots")
        return slots

    def _estimate_slot_positions(
        self,
        grid: GridCalibration,
        num_slots: int,
        frame_size: Tuple[int, int],
    ) -> List[SlotCalibration]:
        """
        Estimate slot positions based on grid geometry.

        This is a fallback when auto-detection fails.
        Slots are assumed to be evenly spaced below the grid.

        Args:
            grid: Grid calibration
            num_slots: Number of slots to estimate
            frame_size: (height, width) of the frame

        Returns:
            List of estimated SlotCalibration objects
        """
        height, width = frame_size
        grid_x, grid_y, grid_w, grid_h = grid.bounds
        cell_w, cell_h = grid.cell_size

        slots = []

        # Estimate slot size (roughly 2x2 cells)
        slot_w = cell_w * 2
        slot_h = cell_h * 2

        # Position slots below grid, evenly spaced
        total_slots_width = num_slots * slot_w + (num_slots - 1) * cell_w
        start_x = grid_x + (grid_w - total_slots_width) // 2

        # Y position: below grid with some margin
        slot_y = grid_y + grid_h + int(cell_h * 1.5)

        for i in range(num_slots):
            slot_x = start_x + i * (slot_w + cell_w)
            center_x = slot_x + slot_w // 2
            center_y = slot_y + slot_h // 2

            slots.append(
                SlotCalibration(
                    center=(center_x, center_y),
                    bounds=(slot_x, slot_y, slot_w, slot_h),
                )
            )

        return slots

    def validate_calibration(self, cal: CalibrationData) -> List[str]:
        """
        Validate calibration data for correctness.

        Checks for common issues and returns a list of error messages.

        Args:
            cal: CalibrationData to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []

        # Check version
        if cal.version != "2.0":
            errors.append(f"Unsupported calibration version: {cal.version}")

        # Check resolutions
        if cal.device_resolution[0] <= 0 or cal.device_resolution[1] <= 0:
            errors.append("Device resolution not set or invalid")

        if cal.window_resolution[0] <= 0 or cal.window_resolution[1] <= 0:
            errors.append("Window resolution not set or invalid")

        # Check grid
        if cal.grid is None:
            errors.append("Grid calibration missing")
        else:
            x, y, w, h = cal.grid.bounds
            cell_w, cell_h = cal.grid.cell_size

            # Check grid bounds are within window
            if cal.window_resolution[0] > 0 and cal.window_resolution[1] > 0:
                if x < 0 or y < 0:
                    errors.append(
                        "Grid bounds extend outside window (negative position)"
                    )
                if x + w > cal.window_resolution[0]:
                    errors.append("Grid bounds extend beyond window width")
                if y + h > cal.window_resolution[1]:
                    errors.append("Grid bounds extend beyond window height")

            # Check cell size consistency (allow up to 8px float rounding tolerance)
            if abs(cell_w * 8 - w) > 8:
                errors.append(
                    f"Grid width ({w}) differs too much from 8 * cell_width ({cell_w * 8})"
                )
            if abs(cell_h * 8 - h) > 8:
                errors.append(
                    f"Grid height ({h}) differs too much from 8 * cell_height ({cell_h * 8})"
                )

            # Check cell centers
            if len(cal.grid.cell_centers) != 8:
                errors.append(
                    f"Expected 8 rows of cell centers, got {len(cal.grid.cell_centers)}"
                )
            else:
                for i, row in enumerate(cal.grid.cell_centers):
                    if len(row) != 8:
                        errors.append(f"Row {i} has {len(row)} cells, expected 8")

        # Check slots
        if len(cal.slots) != 3:
            errors.append(f"Expected 3 slots, got {len(cal.slots)}")
        else:
            # Check slots are roughly aligned horizontally
            if cal.grid is not None:
                grid_bottom = cal.grid.bounds[1] + cal.grid.bounds[3]
                for i, slot in enumerate(cal.slots):
                    # Slots should be below the grid
                    if slot.center[1] < grid_bottom:
                        errors.append(
                            f"Slot {i} appears to be above or inside the grid"
                        )

            # Check slots are roughly evenly spaced
            if len(cal.slots) >= 2:
                centers_x = [s.center[0] for s in cal.slots]
                gaps = [
                    centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)
                ]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    for i, gap in enumerate(gaps):
                        if abs(gap - avg_gap) > avg_gap * 0.3:
                            errors.append(f"Slots {i} and {i + 1} have uneven spacing")

        # Check detection parameters
        if cal.detection.lab_threshold < 1 or cal.detection.lab_threshold > 100:
            errors.append(
                f"LAB threshold ({cal.detection.lab_threshold}) should be between 1 and 100"
            )

        if cal.detection.morph_kernel_size < 1 or cal.detection.morph_kernel_size > 15:
            errors.append(
                f"Morph kernel size ({cal.detection.morph_kernel_size}) should be between 1 and 15"
            )

        return errors

    def create_calibration_from_frame(
        self,
        frame: NDArray[np.uint8],
        device_resolution: Tuple[int, int],
    ) -> Optional[CalibrationData]:
        """
        Create complete calibration from a single frame.

        This is a convenience method that combines grid detection,
        slot detection, and validation.

        Args:
            frame: BGR frame from scrcpy window
            device_resolution: (width, height) of the Android device

        Returns:
            CalibrationData if successful, None otherwise
        """
        height, width = frame.shape[:2]

        # Detect grid
        grid = self.auto_detect_grid(frame)
        if grid is None:
            logger.error("Failed to detect grid")
            return None

        # Detect slots
        slots = self.auto_detect_slots(frame, grid)
        if len(slots) != 3:
            logger.warning(f"Detected {len(slots)} slots instead of 3")

        # Create calibration data
        cal = CalibrationData(
            version="2.0",
            device_resolution=device_resolution,
            window_resolution=(width, height),
            grid=grid,
            slots=slots,
            detection=DetectionCalibration(),
        )

        # Validate
        errors = self.validate_calibration(cal)
        if errors:
            logger.warning(f"Calibration validation warnings: {errors}")

        self._calibration_data = cal
        return cal

    def load_calibration(self, path: str) -> Optional[CalibrationData]:
        """
        Load calibration data from file.

        Args:
            path: Path to calibration JSON file

        Returns:
            CalibrationData if successful, None otherwise
        """
        try:
            cal = CalibrationData.load(path)
            errors = self.validate_calibration(cal)
            if errors:
                logger.warning(f"Loaded calibration has validation warnings: {errors}")
            self._calibration_data = cal
            return cal
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None

    def save_calibration(self, data: CalibrationData, path: str) -> bool:
        """
        Save calibration data to file.

        Args:
            data: CalibrationData to save
            path: Path to save to

        Returns:
            True if successful
        """
        try:
            data.save(path)
            self._calibration_data = data
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    @property
    def calibration_data(self) -> Optional[CalibrationData]:
        """Get the current calibration data."""
        return self._calibration_data


__all__ = ["Calibrator"]
