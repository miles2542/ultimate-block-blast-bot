"""
Block Blast Bot v2.0 - Input Executor

High-level wrapper for executing piece placements.
"""

from block_blast_bot.input.adb_controller import ADBController
from block_blast_bot.input.coordinate import InputCalibration, calculate_swipe
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.config import InputConfig


class InputExecutor:
    """Executes physical screen interactions required for piece placement."""

    def __init__(
        self, adb: ADBController, calib_data: CalibrationData, input_cfg: InputConfig
    ):
        self._adb = adb
        self._calib_data = calib_data
        self._input_calib = InputCalibration(
            popup_y_offset=input_cfg.popup_y_offset,
            scale_x=input_cfg.scale_x,
            scale_y=input_cfg.scale_y,
        )
        self.offset_x = 0
        self.offset_y = 0

    def execute_move(
        self,
        slot_index: int,
        target_row: int,
        target_col: int,
        piece_cells: list[tuple[int, int]],
        duration_ms: int = 400,
    ) -> bool:
        """
        Executes an ADB drag operation to place a piece from a tray slot onto the grid.

        Args:
            slot_index: 0, 1, or 2 (Left, Center, Right slot)
            target_row: 0-7 target row for the piece's origin block
            target_col: 0-7 target column for the piece's origin block
            piece_cells: List of (row, col) relative coordinates representing the piece shape
            duration_ms: How long the physical swipe should take in milliseconds

        Returns:
            bool: True if execution was dispatched successfully
        """
        if not self._calib_data.is_valid():
            return False

        if not (0 <= slot_index < len(self._calib_data.slots)):
            return False

        # Get slot pixel coordinate (Center of the tray slot)
        slot_cx, slot_cy = self._calib_data.slots[slot_index].center

        # Get target grid pixel coordinate
        if not (0 <= target_row < 8 and 0 <= target_col < 8):
            return False

        # The mathematical grid target maps to where (0, 0) of the piece lands.
        # However, the physical drag from the bottom tray is anchored to the
        # CENTER of the piece's bounding box, not its origin. Let's calculate the offset.
        min_r = min(c[0] for c in piece_cells)
        max_r = max(c[0] for c in piece_cells)
        min_c = min(c[1] for c in piece_cells)
        max_c = max(c[1] for c in piece_cells)

        cell_w, cell_h = self._calib_data.grid.cell_size

        # Distance from origin (0, 0) cell center to actual bounding box center
        pw = (max_c - min_c + 1) * cell_w
        ph = (max_r - min_r + 1) * cell_h
        offset_c_px = pw / 2.0 - (0 - min_c) * cell_w - cell_w / 2.0
        offset_r_px = ph / 2.0 - (0 - min_r) * cell_h - cell_h / 2.0

        origin_dest_x, origin_dest_y = self._calib_data.grid.cell_centers[target_row][
            target_col
        ]

        # The actual physical pixel we need to snap the crosshair onto
        dest_x = int(origin_dest_x + offset_c_px)
        dest_y = int(origin_dest_y + offset_r_px)

        # Calculate perfect geometric finger drag logic using calibrated scalars
        # Coordinates return in raw Window Space.
        w_x1, w_y1, w_x2, w_y2 = calculate_swipe(
            slot_center_x=slot_cx,
            slot_center_y=slot_cy,
            dest_center_x=dest_x,
            dest_center_y=dest_y,
            calibration=self._input_calib,
        )

        # Convert the calculated logical window space sweeps to Device screen space
        # Apply the same manual offsets that were used to warp the vision frame
        w_x1 -= self.offset_x
        w_y1 -= self.offset_y
        w_x2 -= self.offset_x
        w_y2 -= self.offset_y

        dev_x1, dev_y1 = self._calib_data.window_to_device(w_x1, w_y1)
        dev_x2, dev_y2 = self._calib_data.window_to_device(w_x2, w_y2)

        # Dispatch
        try:
            self._adb.swipe(dev_x1, dev_y1, dev_x2, dev_y2, duration_ms)
            return True
        except Exception as e:
            print(f"Failed to execute ADB move: {e}")
            return False


__all__ = ["InputExecutor"]
