"""
Block Blast Bot v2.0 - Vision Pipeline

Coordinates the execution of frame capture, board detection,
and piece detection into a unified, thread-safe game state.
"""

from typing import Optional, List, Tuple
import numpy as np
from pydantic import BaseModel


from block_blast_bot.config import VisionConfig
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.vision.board_detector import BoardDetector
from block_blast_bot.vision.piece_detector import PieceDetector


import logging

logger = logging.getLogger(__name__)


class EngineInputState(BaseModel):
    """
    The unified physical state of the Block Blast game at a specific frame.
    This is passed directly into the Solver Engine.
    """

    # 8x8 boolean matrix (True = cell is filled)
    board_cells: List[List[bool]]

    # List of (slot_index, PieceCells array)
    available_pieces: List[Tuple[int, List[Tuple[int, int]]]]


class VisionPipeline:
    """
    Main aggregator for the computer vision subsystem.
    Takes raw BGR frames and outputs structured, mathematical game states.
    """

    def __init__(self, calib_data: CalibrationData, vision_config: VisionConfig):
        """
        Initialize the pipeline utilizing pre-calibrated bounding geometry.
        """
        self._calib = calib_data
        self._config = vision_config

        if not self._calib.is_valid():
            raise ValueError("VisionPipeline requires strictly valid CalibrationData")

        self.board_detector = BoardDetector(
            calibration=self._calib,
            lab_threshold=self._config.board.lab_threshold,
            morph_kernel_size=self._config.board.morph_kernel_size,
        )

        self.piece_detector = PieceDetector(
            calibration=self._calib,
            # use_otsu=self._config.pieces.use_otsu # Assuming config mapping if needed, else using defaults
        )

    def process(self, frame: np.ndarray) -> Optional[EngineInputState]:
        """
        Processes a full BGR frame to extract the mathematical game state.

        Args:
            frame: a BGR numpy array cropped to window size (or raw device size)

        Returns:
            GameState object containing the board matrix and available pieces,
            or None if the frame could not be confidently parsed (e.g. obscured).
        """
        # 1. Detect Board State (8x8)
        try:
            board_result = self.board_detector.detect(frame)
            board_matrix = board_result.to_board_state().grid.tolist()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Board detection failed: {e}")
            # Game is likely obscured, on a menu, or watching an ad
            return None

        # 2. Detect Available Tray Pieces
        try:
            pieces = self.piece_detector.detect_pieces(frame, region="slots")
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Piece detection failed: {e}")
            pieces = []

        # 3. Format into the Engine structures
        available_pieces = []
        if isinstance(pieces, list):
            piece_list = pieces
        else:
            piece_list = pieces.pieces

        for piece in piece_list:
            if piece.slot_index != -1:
                available_pieces.append((piece.slot_index, piece.cells))

        # logger.debug(
        #     f"Vision: Detected {len(board_result.filled_cells)} cells, {len(available_pieces)} pieces"
        # )

        return EngineInputState(
            board_cells=board_matrix,
            available_pieces=available_pieces,
        )


__all__ = ["VisionPipeline", "EngineInputState"]
