"""
Block Blast Bot v2.0 - Stability Gate

Detects when the game screen has stopped animating using fast perceptual analysis.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple

from block_blast_bot.calibration.data import CalibrationData

logger = logging.getLogger(__name__)


class StabilityGate:
    """
    Monitors a Region of Interest (ROI) for visual stability.
    Useful for waiting until block drop animations or line clears are finished.
    """

    def __init__(
        self, calib: CalibrationData, threshold: float = 1.0, required_ms: int = 300
    ):
        """
        Args:
            calib: Calibration data to determine ROI
            threshold: Sensitivity threshold (lower = more sensitive)
            required_ms: Milliseconds of stability required
        """
        self.calib = calib
        self.threshold = threshold
        self.required_ms = required_ms

        self._last_thumb: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None

        self._setup_roi()

    def _setup_roi(self):
        """Calculate the minimal bounding box containing both board and pieces."""
        if not self.calib.is_valid():
            return

        bx, by, bw, bh = self.calib.grid.bounds

        # Take all slots and find their combined vertical extent
        slot_y_max = max(s.bounds[1] + s.bounds[3] for s in self.calib.slots)

        # ROI spans full board width and from board top to slot bottom
        self._roi_rect = (bx, by, bw, slot_y_max - by)

    def reset(self):
        """Reset the stability counter."""
        self._stable_since = None
        self._last_thumb = None

    def check_stability(self, frame: np.ndarray) -> bool:
        """
        Checks if the ROI is visually stable compared to the previous call.

        Args:
            frame: Current BGR frame

        Returns:
            True if ROI has been stable for 'required_frames' consecutive checks.
        """
        if self._roi_rect is None:
            self._setup_roi()
            if self._roi_rect is None:
                return True  # Fallback

        x, y, w, h = self._roi_rect
        roi = frame[y : y + h, x : x + w]

        # 1. Downsample for speed and noise filtering (32x32 is perfect for O(1) comparison)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

        if self._last_thumb is None:
            self._last_thumb = thumb
            self._stable_since = time.time()
            return False

        # 2. Compute Mean Absolute Difference against the original REFERENCE FRAME
        # This catches slow-drifting animations that consecutive-diffing misses.
        diff = cv2.absdiff(thumb, self._last_thumb)
        score = np.mean(diff)

        # 3. Apply threshold
        if score < self.threshold:
            # Within tolerance of the reference frame
            if self._stable_since is None:
                self._stable_since = time.time()
        else:
            # Moving! Target is changing.
            # Reset reference to the latest frame and clear stable timer.
            self._last_thumb = thumb
            self._stable_since = None
            return False

        if self._stable_since is None:
            return False

        elapsed_ms = (time.time() - self._stable_since) * 1000.0
        return elapsed_ms >= self.required_ms
