"""
Block Blast Bot v2.0 - Test Configuration

Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_board_state():
    """Provide a sample empty board state for testing."""
    from block_blast_bot.game_types import BoardState

    return BoardState.empty()


@pytest.fixture
def sample_piece_shapes():
    """Provide sample piece shapes for testing."""
    from block_blast_bot.game_types import PieceShape

    return [
        # Single cell
        PieceShape(cells=[(0, 0)]),
        # 2x1 horizontal
        PieceShape(cells=[(0, 0), (0, 1)]),
        # 2x2 square
        PieceShape(cells=[(0, 0), (0, 1), (1, 0), (1, 1)]),
        # L-shape
        PieceShape(cells=[(0, 0), (1, 0), (2, 0), (2, 1)]),
    ]
