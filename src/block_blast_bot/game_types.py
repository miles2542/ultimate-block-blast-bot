"""
Block Blast Bot v2.0 - Type Definitions

Core type definitions used throughout the application.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class WindowInfo:
    """Information about a detected window."""

    title: str  # Window title
    left: int  # Left edge position (screen coordinates)
    top: int  # Top edge position (screen coordinates)
    width: int  # Window width in pixels
    height: int  # Window height in pixels
    hwnd: int  # Window handle (Windows)


class GameState(Enum):
    """Game state enumeration."""

    NORMAL = "normal"
    GAME_OVER = "game_over"
    POPUP = "popup"
    LOADING = "loading"
    UNKNOWN = "unknown"


class AppMode(str, Enum):
    MANUAL = "Manual"
    SEMI_AUTO = "Semi-Auto"
    FULL_AUTO = "Full-Auto"


class SolverMode(str, Enum):
    AGGRESSIVE = "Aggressive"
    SAFE = "Safe"
    CUSTOM = "Custom"


class DisplayMode(str, Enum):
    LEAN = "Lean"
    STUDIO = "Studio"


@dataclass
class Position:
    """2D position with integer coordinates."""

    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)


@dataclass
class Rect:
    """Rectangle defined by top-left position and size."""

    x: int
    y: int
    width: int
    height: int

    @property
    def top_left(self) -> Position:
        return Position(self.x, self.y)

    @property
    def bottom_right(self) -> Position:
        return Position(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Position:
        return Position(self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, pos: Position) -> bool:
        """Check if a position is inside this rectangle."""
        return (
            self.x <= pos.x < self.x + self.width
            and self.y <= pos.y < self.y + self.height
        )


@dataclass
class Cell:
    """A single cell on the game board."""

    row: int
    col: int
    filled: bool = False

    @property
    def position(self) -> tuple[int, int]:
        return (self.row, self.col)


@dataclass
class PieceShape:
    """Shape of a game piece defined by relative cell positions."""

    cells: list[tuple[int, int]]  # List of (row_offset, col_offset) from origin

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def bounding_box(self) -> tuple[int, int, int, int]:
        """Return (min_row, max_row, min_col, max_col)."""
        if not self.cells:
            return (0, 0, 0, 0)
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (min(rows), max(rows), min(cols), max(cols))


@dataclass
class Move:
    """A single move placing a piece on the board."""

    piece_index: int  # Index of piece in the current set (0, 1, or 2)
    target_row: int  # Target row for piece origin
    target_col: int  # Target column for piece origin
    piece_shape: Optional[PieceShape] = None  # Shape of the piece being placed


@dataclass
class MoveChain:
    """A sequence of moves for a 3-piece set."""

    moves: list[Move]
    final_score: float = 0.0
    line_clears: int = 0


@dataclass
class BoardState:
    """State of the game board."""

    grid: NDArray[np.bool_]  # 8x8 boolean array, True = filled
    timestamp: float = 0.0  # Capture timestamp

    @classmethod
    def empty(cls) -> "BoardState":
        """Create an empty board state."""
        return cls(grid=np.zeros((8, 8), dtype=np.bool_))

    def is_filled(self, row: int, col: int) -> bool:
        """Check if a cell is filled."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row, col]
        return False

    def set_filled(self, row: int, col: int, filled: bool = True) -> None:
        """Set a cell's filled state."""
        if 0 <= row < 8 and 0 <= col < 8:
            self.grid[row, col] = filled


@dataclass
class DetectedPiece:
    """A detected piece from vision pipeline."""

    # Updated format: cells is now List of (row, col) relative to piece origin
    cells: List[Tuple[int, int]]  # List of (row, col) relative to piece origin
    bbox: Tuple[int, int, int, int]  # x, y, width, height in pixels
    center: Tuple[int, int]  # center x, y in pixels
    cell_size: int  # detected cell size in pixels
    confidence: float = 1.0
    # Legacy support - keep for compatibility
    shape: Optional[PieceShape] = None
    bounding_rect: Optional[Rect] = None
    slot_index: int = -1

    def __post_init__(self) -> None:
        """Build legacy fields from new format for compatibility."""
        if self.shape is None and self.cells:
            self.shape = PieceShape(cells=self.cells)
        if self.bounding_rect is None:
            x, y, w, h = self.bbox
            self.bounding_rect = Rect(x=x, y=y, width=w, height=h)


@dataclass
class PieceDetectionResult:
    """Result from piece detection."""

    pieces: List[DetectedPiece]
    method: str  # Detection method used
    debug_frame: Optional[NDArray[np.uint8]] = None  # Debug visualization


# Legacy DetectedPiece kept for compatibility
@dataclass
class _LegacyDetectedPiece:
    """Legacy format - kept for compatibility."""

    shape: PieceShape
    bounding_rect: Rect
    slot_index: int  # Which slot (0, 1, 2) the piece is in
    confidence: float = 1.0


@dataclass
class GameSnapshot:
    """Complete snapshot of game state at a point in time."""

    board: BoardState
    pieces: list[DetectedPiece]
    state: GameState
    frame: Optional[NDArray[np.uint8]] = None  # Original captured frame
    timestamp: float = 0.0


@dataclass
class DetectionResult:
    """Result from a single detection method."""

    board: int  # u64 bitmask (bit = 1 if cell is filled)
    method: str  # Name of the detection method that succeeded
    confidence: float  # Confidence score (0.0 to 1.0)
    debug_frame: Optional[NDArray[np.uint8]] = None  # Debug visualization

    @property
    def filled_count(self) -> int:
        """Count the number of filled cells."""
        return bin(self.board).count("1")

    def is_cell_filled(self, row: int, col: int) -> bool:
        """Check if a specific cell is filled."""
        if 0 <= row < 8 and 0 <= col < 8:
            return bool(self.board & (1 << (row * 8 + col)))
        return False


@dataclass
class BoardDetectionResult:
    """High-level board detection result with convenience methods."""

    board: int  # u64 bitmask
    filled_cells: list[tuple[int, int]]  # List of (row, col) for filled cells
    method: str  # Detection method used
    confidence: float  # Confidence score

    @property
    def filled_count(self) -> int:
        """Count the number of filled cells."""
        return len(self.filled_cells)

    def to_board_state(self) -> BoardState:
        """Convert to BoardState for compatibility."""
        grid = np.zeros((8, 8), dtype=np.bool_)
        for row, col in self.filled_cells:
            grid[row, col] = True
        return BoardState(grid=grid)


__all__ = [
    "WindowInfo",
    "GameState",
    "Position",
    "Rect",
    "Cell",
    "PieceShape",
    "Move",
    "MoveChain",
    "BoardState",
    "DetectedPiece",
    "PieceDetectionResult",
    "GameSnapshot",
    "DetectionResult",
    "BoardDetectionResult",
    "AppMode",
    "SolverMode",
    "DisplayMode",
]
