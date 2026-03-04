"""
Block Blast Bot v2.0 - Solver Interface

Python interface to the Rust solver engine via FFI.
"""

from typing import Optional

from block_blast_bot.game_types import BoardState, MoveChain, PieceShape


class SolverInterface:
    """
    Interface to the Rust solver engine.

    Provides Python-friendly API for the high-performance Rust solver.
    """

    def __init__(
        self,
        time_limit_ms: int = 3000,
        beam_width: int = 150,
        depth2_samples: int = 60,
    ) -> None:
        """
        Initialize solver interface.

        Args:
            time_limit_ms: Maximum time per solve in milliseconds
            beam_width: Beam search width
            depth2_samples: Number of depth-2 samples
        """
        self._time_limit_ms = time_limit_ms
        self._beam_width = beam_width
        self._depth2_samples = depth2_samples

    def find_best_chain(
        self,
        board: BoardState,
        pieces: list[PieceShape],
    ) -> Optional[MoveChain]:
        """
        Find the best move chain for the given board and pieces.

        Args:
            board: Current board state
            pieces: List of 3 pieces to place

        Returns:
            MoveChain with optimal moves, or None if no valid moves
        """
        # TODO: Implement FFI call to Rust solver
        raise NotImplementedError("Solver interface not yet implemented")

    def set_time_limit(self, time_limit_ms: int) -> None:
        """Update the time limit for solving."""
        self._time_limit_ms = time_limit_ms


__all__ = ["SolverInterface"]
