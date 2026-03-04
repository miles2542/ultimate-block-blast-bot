"""
Block Blast Bot v2.0 - Visualizer

Lightweight OpenCV-based visualization for debugging the vision pipeline.
"""

import cv2


from block_blast_bot.game_types import GameSnapshot
from block_blast_bot.calibration.data import CalibrationData


class VisionVisualizer:
    """
    OpenCV-based visualizer for vision pipeline debugging.

    Provides realtime visualization of:
    - Captured frames
    - Board detection with cell overlays
    - Piece detection with bounding boxes
    """

    def __init__(self, calibration: CalibrationData) -> None:
        """Initialize visualizer."""
        self._running = False
        self._calibration = calibration
        self._window_name = "Block Blast Bot v2 - Vision Debug"

    def start(self) -> None:
        """Start the visualization window."""
        self._running = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def stop(self) -> None:
        """Stop the visualization window."""
        if self._running:
            self._running = False
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def show_snapshot(self, snapshot: GameSnapshot) -> None:
        """
        Display a complete GameSnapshot in the visualizer.

        Args:
            snapshot: Evaluated game state
        """
        if not self._running or snapshot.frame is None:
            return

        debug_frame = snapshot.frame.copy()
        grid = self._calibration.grid

        # 1. Draw Board State
        if grid:
            # Draw board bounds
            gx, gy, gw, gh = grid.bounds
            cv2.rectangle(debug_frame, (gx, gy), (gx + gw, gy + gh), (255, 0, 0), 2)

            # Draw cell states
            for r in range(8):
                for c in range(8):
                    cx, cy = grid.get_cell_center(r, c)
                    if snapshot.board.is_filled(r, c):
                        # Filled cell (Green)
                        cv2.circle(debug_frame, (cx, cy), 8, (0, 255, 0), -1)
                    else:
                        # Empty cell (Small dot)
                        cv2.circle(debug_frame, (cx, cy), 2, (200, 200, 200), -1)

        # 2. Draw Pieces
        for i, piece in enumerate(snapshot.pieces):
            x, y, w, h = piece.bbox
            cv2.rectangle(
                debug_frame, (x, y), (x + w, y + h), (0, 165, 255), 1
            )  # Orange bounding box (thin)

            # Draw piece cells reconstructed around center
            if piece.cells:
                s = piece.cell_size
                max_r = max(r for r, c in piece.cells)
                max_c = max(c for r, c in piece.cells)

                # Calculate the top-left origin for drawing relative to center
                origin_x = piece.center[0] - (max_c * s) // 2
                origin_y = piece.center[1] - (max_r * s) // 2

                for r, c in piece.cells:
                    px = origin_x + (c * s) - s // 2
                    py = origin_y + (r * s) - s // 2
                    cv2.rectangle(
                        debug_frame, (px, py), (px + s, py + s), (0, 255, 255), 2
                    )
                    cv2.circle(
                        debug_frame,
                        (px + s // 2, py + s // 2),
                        max(2, s // 8),
                        (0, 165, 255),
                        -1,
                    )

            # Label piece
            cv2.putText(
                debug_frame,
                f"P{i}: {len(piece.cells)} cells",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

            # Draw piece center point
            cv2.circle(debug_frame, piece.center, 3, (0, 0, 255), -1)

        # 3. Draw Game State Overlay
        state_text = f"State: {snapshot.state.value.upper()}"
        cv2.putText(
            debug_frame,
            state_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        # Render
        cv2.imshow(self._window_name, debug_frame)

        # Pump events (returns ESC if pressed)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            self.stop()

    def is_running(self) -> bool:
        """Check if visualizer is running."""
        return self._running


__all__ = ["VisionVisualizer"]
