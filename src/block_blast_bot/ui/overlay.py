"""
Block Blast Bot v2.0 - Overlay Renderer

Handles all OpenCV drawing operations for the Studio Mode GUI.
Separates rendering logic from the main orchestration loop.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from block_blast_bot.calibration.data import CalibrationData


class OverlayRenderer:
    """
    Modular renderer for game state overlays.
    Draws grid, pieces, and move projections onto a frame.
    """

    def __init__(self, calibration: CalibrationData):
        self.calib = calibration

        # Colors (BGR)
        self.COLOR_GRID = (100, 100, 100)  # Grey
        self.COLOR_FILLED = (0, 255, 0)  # Green
        self.COLOR_EMPTY = (200, 200, 200)  # Light Grey
        self.COLOR_PIECE = (0, 165, 255)  # Orange
        self.COLOR_PROJECTION = (255, 255, 0)  # Cyan
        self.COLOR_TEXT = (255, 255, 255)  # White

    def draw_studio_overlay(
        self,
        frame: np.ndarray,
        board_cells: Optional[List[List[bool]]],
        live_pieces: List[Tuple[int, List[Tuple[int, int]]]],
        best_move: Optional[Dict],
        inplace: bool = True,
    ) -> np.ndarray:
        """
        Draws the full suite of Studio Mode overlays onto the frame.
        """
        overlay = frame if inplace else frame.copy()

        # 1. Draw Board Grid & Cells
        if board_cells:
            self._draw_grid(overlay, board_cells)

        # 2. Draw Detected Pieces in Trays
        self._draw_pieces(overlay, live_pieces)

        # 3. Draw Move Projection
        if best_move:
            self._draw_best_move(overlay, best_move, live_pieces)

        return overlay

    def _draw_grid(self, frame: np.ndarray, board_cells: List[List[bool]]):
        """Draws the 8x8 grid and filled cell status."""
        if not self.calib.grid:
            return

        # Draw outer bounds
        x, y, w, h = self.calib.grid.bounds
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.COLOR_GRID, 1)

        # Draw cells
        for r in range(8):
            for c in range(8):
                cx, cy = self.calib.grid.get_cell_center(r, c)
                if board_cells[r][c]:
                    # Draw a filled box or diamond for filled cells
                    cv2.circle(frame, (cx, cy), 10, self.COLOR_FILLED, -1)
                else:
                    # Small dot for empty
                    cv2.circle(frame, (cx, cy), 2, self.COLOR_EMPTY, -1)

    def _draw_pieces(
        self, frame: np.ndarray, live_pieces: List[Tuple[int, List[Tuple[int, int]]]]
    ):
        """Draws the shapes of pieces in their tray slots."""
        if not self.calib.slots:
            return

        for slot_idx, cells in live_pieces:
            if not cells or slot_idx < 0 or slot_idx >= len(self.calib.slots):
                continue

            slot = self.calib.slots[slot_idx]
            center_x, center_y = slot.center

            # Use the precise cell size matching the game's idle piece scaling (44%)
            grid_cell_w = self.calib.grid.cell_size[0] if self.calib.grid else 64
            cell_size = int(grid_cell_w * 0.44)

            # CALC CENTERING:
            # Pieces are normalized to (0,0) top-left.
            # We need to find the bounding box size to center it.
            rows = max(r for r, c in cells) + 1
            cols = max(c for r, c in cells) + 1

            # Offset piece top-left so the whole shape is centered in slot
            top_left_x = center_x - (cols * cell_size) // 2
            top_left_y = center_y - (rows * cell_size) // 2

            for r, c in cells:
                px = top_left_x + (c * cell_size)
                py = top_left_y + (r * cell_size)
                cv2.rectangle(
                    frame,
                    (px, py),
                    (px + cell_size, py + cell_size),
                    self.COLOR_PIECE,
                    2,
                )

    def _draw_best_move(
        self,
        frame: np.ndarray,
        best_move: Dict,
        live_pieces: List[Tuple[int, List[Tuple[int, int]]]],
    ):
        """Draws a projection of where the piece will land."""
        slot_idx = best_move.get("slot_index")
        target_row = best_move.get("target_row")
        target_col = best_move.get("target_col")

        if slot_idx is None or target_row is None or target_col is None:
            return

        # Find the shape list for this slot
        piece_cells = None
        for s_idx, cells in live_pieces:
            if s_idx == slot_idx:
                piece_cells = cells
                break

        if not piece_cells or not self.calib.grid:
            return

        # Draw the piece shape projected onto the grid
        for dr, dc in piece_cells:
            r, c = target_row + dr, target_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                cx, cy = self.calib.grid.get_cell_center(r, c)
                # Draw a thicker highlight for the projection
                size = self.calib.grid.bounds[2] // 16  # Approx half cell size
                cv2.rectangle(
                    frame,
                    (cx - size, cy - size),
                    (cx + size, cy + size),
                    self.COLOR_PROJECTION,
                    2,
                )

    def draw_lean_overlay(self, frame: np.ndarray, status: str) -> np.ndarray:
        """Draws minimal info for Lean mode."""
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"STATUS: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return overlay
