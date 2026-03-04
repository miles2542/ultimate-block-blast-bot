"""
Block Blast Bot v2.0 - State Detector

Detects game state (normal, game over, popup, etc.) and visual stability.
"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from block_blast_bot.game_types import GameState
from block_blast_bot.config import VisionStabilityConfig


class StateDetector:
    """
    Game state detector.

    Detects:
    - Normal gameplay
    - Game over screen
    - Popup overlays
    - Loading states
    """

    def __init__(self, config: Optional[VisionStabilityConfig] = None) -> None:
        """Initialize state detector."""
        self._config = config or VisionStabilityConfig()

    def detect_state(self, frame: NDArray[np.uint8]) -> GameState:
        """
        Detect the current game state.

        Args:
            frame: BGR frame

        Returns:
            GameState enum value
        """
        # TODO(Phase 3): Implement robust state detection using template
        # matching against known Game Over and Popup UI elements.
        # For Phase 2, assume NORMAL to let the pipeline proceed.
        return GameState.NORMAL

    def is_stable(
        self, frames: list[NDArray[np.uint8]], threshold: Optional[float] = None
    ) -> bool:
        """
        Check if the game state is stable (no animations) across recent frames.

        Uses normalized cross-correlation / structural similarity on a
        downscaled grayscale version of the frames for speed and robustness against
        minor compression artifacts or capture jitter.

        Args:
            frames: List of recent frames (at least 2 required)
            threshold: Similarity threshold (defaults to config)

        Returns:
            True if all consecutive frame pairs are similar enough
        """
        if len(frames) < 2:
            return False

        sim_threshold = (
            threshold if threshold is not None else self._config.hash_threshold
        )

        # Downscale size for fast comparison
        compare_size = (32, 64)

        # Convert and downscale all frames first
        small_frames = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, compare_size, interpolation=cv2.INTER_AREA)
            # Normalize to account for global brightness shifts
            small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
            small_frames.append(small.astype(np.float32))

        # Check all consecutive pairs
        for i in range(len(small_frames) - 1):
            f1 = small_frames[i]
            f2 = small_frames[i + 1]

            # Use matchTemplate with TM_CCOEFF_NORMED for structural similarity
            # Since images are the same size, result is a 1x1 matrix
            result = cv2.matchTemplate(f1, f2, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]

            if similarity < sim_threshold:
                return False

        return True


__all__ = ["StateDetector"]
