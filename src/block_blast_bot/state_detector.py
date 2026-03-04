"""
Block Blast Bot v2.0 - Game State Detector

Ultra-fast heuristics to determine the current screen (menu, playing, game over).
"""

from enum import Enum

import numpy as np
import cv2

from block_blast_bot.calibration.data import CalibrationData


class GameScreen(Enum):
    PLAYING = "PLAYING"
    GAME_OVER = "GAME_OVER"
    MAIN_MENU = "MAIN_MENU"
    UNKNOWN = "UNKNOWN"


class StateDetector:
    """
    Rapidly evaluates what phase the Block Blast app is currently in.
    Uses O(1) targeted pixel lookups rather than heavy Vision scanning.
    """

    def __init__(self, calib_data: CalibrationData):
        self._calib = calib_data

    def detect_screen(self, frame: np.ndarray) -> GameScreen:
        """
        Runs extremely lightweight heuristic checks to determine the active screen.

        Args:
            frame: Raw BGR window frame

        Returns:
            The current GameScreen enum.
        """
        # We need the user to map these coordinates once using the calibration tool later,
        # but for now, we will perform a basic structural analysis.

        # 1. Are we playing?
        # A quick heuristic to verify the "Playing" state is if the board area
        # is visible and not heavily darkened (which happens during Game Over/Pause overlays).
        # We can sample the corners of the calibrated board bounding box.

        if not self._calib.is_valid():
            return GameScreen.UNKNOWN

        board_rect = self._calib.grid.bounds
        if board_rect is None:
            return GameScreen.UNKNOWN

        bx, by, bw, bh = board_rect

        # Ensure rect is within frame
        f_h, f_w = frame.shape[:2]
        if bx < 0 or by < 0 or bx + bw > f_w or by + bh > f_h:
            import logging

            logging.getLogger(__name__).warning(
                f"Board rect {(bx, by, bw, bh)} out of frame bounds {f_w}x{f_h}"
            )
            return GameScreen.UNKNOWN

        # Extract board region and convert to grayscale to check brightness
        board_roi = frame[by : by + bh, bx : bx + bw]
        gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Log brightness if it's near the threshold to diagnose false positives
        # if mean_brightness < 50:
        #     import logging

        #     logging.getLogger(__name__).info(
        #         f"[State] Mean Brightness: {mean_brightness:.2f}"
        #     )

        # 1. CAPTURE ERROR CHECK: If screen is pure black (or nearly so), it's a capture failure.
        # PrintWindow can return zeros. Real Game Over screens are dark but NOT pure black.
        if mean_brightness < 2.0:
            return GameScreen.UNKNOWN

        # 2. VERIFY PLAYING STATE:
        # Check if we see characteristic features of a "live" game board.
        if mean_brightness > 20:
            return GameScreen.PLAYING

        # 3. GAME OVER CHECK:
        # Game Over significantly darkens the board area (usually < 25 brightness).
        # We assume GAME_OVER if it's dark but NOT pure black.
        if 2.0 <= mean_brightness <= 20:
            return GameScreen.GAME_OVER

        return GameScreen.UNKNOWN


__all__ = ["GameScreen", "StateDetector"]
