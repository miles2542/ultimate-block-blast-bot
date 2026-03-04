"""
Block Blast Bot v2.0 - Vision Pipeline

Main vision pipeline that orchestrates board and piece detection.
"""

from typing import Optional
import time

import numpy as np
from numpy.typing import NDArray

from block_blast_bot.game_types import GameSnapshot, GameState
from block_blast_bot.config import VisionConfig
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.vision.board_detector import BoardDetector
from block_blast_bot.vision.piece_detector import PieceDetector
from block_blast_bot.vision.state_detector import StateDetector


class VisionPipeline:
    """
    Vision pipeline for game state detection.

    Coordinates board detection, piece detection, and game state detection.
    """

    def __init__(
        self,
        calibration: CalibrationData,
        config: Optional[VisionConfig] = None,
    ) -> None:
        """
        Initialize vision pipeline with calibration and optional config.
        """
        self._calibration = calibration
        self._config = config or VisionConfig()

        self._state_detector = StateDetector(self._config.stability)
        self._board_detector = BoardDetector(
            self._calibration,
            lab_threshold=self._config.board.lab_threshold,
            morph_kernel_size=self._config.board.morph_kernel_size,
        )
        self._piece_detector = PieceDetector(
            self._calibration,
            use_otsu=(self._config.pieces.method == "lab_otsu"),
            hard_threshold=self._config.board.lab_threshold,  # Reuse board lab threshold for consistency
        )

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[GameSnapshot]:
        """
        Process a captured frame and extract game state.

        Args:
            frame: BGR frame from capture module

        Returns:
            GameSnapshot with detected state, or None if detection failed
        """
        # 1. State Detection
        state = self._state_detector.detect_state(frame)

        # Depending on state, we might skip full detection
        # But for NORMAL, we need board and pieces
        if state != GameState.NORMAL:
            return GameSnapshot(
                board=self._board_detector.detect(
                    frame
                ).to_board_state(),  # Need an empty board state easily
                pieces=[],
                state=state,
                frame=frame,
                timestamp=time.perf_counter(),
            )

        # 2. Board Detection
        board_result = self._board_detector.detect(frame)
        board_state = board_result.to_board_state()

        # 3. Piece Detection
        pieces_result = self._piece_detector.detect_pieces(frame)

        # Return combined snapshot
        return GameSnapshot(
            board=board_state,
            pieces=pieces_result.pieces,
            state=state,
            frame=frame,
            timestamp=time.perf_counter(),
        )


__all__ = ["VisionPipeline"]
