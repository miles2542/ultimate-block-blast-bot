#!/usr/bin/env python3
"""
Block Blast Bot v2.0 - Visual Placement Tester

Interactive Dear PyGui tool to visually pick up pieces and test ADB grid-snapping drops.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_blast_bot.config import load_config
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.capture.frame_capture import FrameCapture
from block_blast_bot.vision.board_detector import BoardDetector
from block_blast_bot.vision.piece_detector import PieceDetector
from block_blast_bot.input.adb_controller import ADBController
from block_blast_bot.input.execution import InputExecutor
from block_blast_bot.game_types import DetectedPiece


class VisualPlacementTester:
    def __init__(self):
        self._config = load_config()
        self._calib_data: Optional[CalibrationData] = None
        self._capture: Optional[FrameCapture] = None
        self._board_detector: Optional[BoardDetector] = None
        self._piece_detector: Optional[PieceDetector] = None
        self._adb: Optional[ADBController] = None
        self._executor: Optional[InputExecutor] = None

        self._texture_id = None
        self._current_frame: Optional[np.ndarray] = None

        # State
        self._state = "LIVE"  # LIVE, PLACING, EXECUTING
        self._active_piece: Optional[DetectedPiece] = None
        self._active_slot_idx: int = -1
        self._available_pieces: List[DetectedPiece] = []

        # Overlay caches
        self._board_top_left = (0, 0)
        self._cell_w = 0
        self._cell_h = 0

    def _frame_to_texture(self, frame: np.ndarray) -> list[float]:
        try:
            code = cv2.COLOR_BGR2RGBA if frame.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
            rgb = cv2.cvtColor(frame, code)
            return (rgb.astype(np.float32) / 255.0).flatten().tolist()
        except Exception:
            # Return blank red texture on fail
            return [1.0, 0.0, 0.0, 1.0] * (frame.shape[0] * frame.shape[1])

    def run(self) -> bool:
        calib_path = Path("calibration.json")
        if not calib_path.exists():
            print("ERROR: calibration.json missing. Run `tools/calibrate.py` first.")
            return False

        import json

        with open(calib_path, "r", encoding="utf-8") as f:
            self._calib_data = CalibrationData.from_dict(json.load(f))

        if not self._calib_data.is_valid():
            print("ERROR: Calibration data invalid.")
            return False

        self._capture = FrameCapture(self._config.capture)
        if not self._capture.initialize():
            print("ERROR: Could not init frame capture.")
            return False

        self._board_detector = BoardDetector(self._calib_data)
        self._piece_detector = PieceDetector(self._calib_data)

        self._adb = ADBController()

        self._executor = InputExecutor(self._adb, self._calib_data, self._config.input)

        # Cache calibration layout
        c00 = self._calib_data.grid.cell_centers[0][0]
        self._cell_w, self._cell_h = self._calib_data.grid.cell_size
        self._board_top_left = (
            int(c00[0] - self._cell_w / 2),
            int(c00[1] - self._cell_h / 2),
        )

        dpg.create_context()
        self._current_frame = self._capture.capture()
        if self._current_frame is None:
            print("ERROR: Failed initial capture.")
            return False

        height, width = self._current_frame.shape[:2]

        with dpg.texture_registry():
            self._texture_id = dpg.add_dynamic_texture(
                width,
                height,
                self._frame_to_texture(self._current_frame),
                tag="frame_texture",
            )

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left, callback=self._cb_mouse_click
            )
            dpg.add_key_press_handler(callback=self._cb_key_press)

        with dpg.window(tag="main_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=width + 20, height=height + 20):
                    dpg.add_image("frame_texture", tag="frame_image")

                with dpg.child_window(width=350):
                    dpg.add_text("Visual Drop Tester", color=[100, 200, 255])
                    dpg.add_separator()
                    dpg.add_separator()
                    dpg.add_text(
                        "STATUS: LIVE MAP", tag="text_status", color=[0, 255, 0]
                    )
                    dpg.add_text("Click any slot piece to pick it up.")
                    dpg.add_text("Press ESC to cancel a selection.")
                    dpg.add_separator()

                    dpg.add_text("Target Row: None", tag="t_row")
                    dpg.add_text("Target Col: None", tag="t_col")
                    dpg.add_separator()

                    dpg.add_text("Live Tuning", color=[255, 200, 100])
                    dpg.add_slider_float(
                        label="Scale X",
                        default_value=self._config.input.scale_x,
                        tag="slider_scx",
                        min_value=1.0,
                        max_value=2.5,
                        callback=self._cb_update_config,
                    )
                    dpg.add_slider_float(
                        label="Scale Y",
                        default_value=self._config.input.scale_y,
                        tag="slider_scy",
                        min_value=1.0,
                        max_value=2.5,
                        callback=self._cb_update_config,
                    )
                    dpg.add_slider_int(
                        label="Popup Y Offset",
                        default_value=self._config.input.popup_y_offset,
                        tag="slider_popy",
                        min_value=-500,
                        max_value=100,
                        callback=self._cb_update_config,
                    )
                    dpg.add_button(
                        label="Save to config.yaml", callback=self._cb_save_config
                    )

        dpg.create_viewport(
            title="Visual Placement Tester", width=width + 400, height=height + 60
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        while dpg.is_dearpygui_running():
            try:
                frame = self._capture.capture()
                if frame is not None:
                    display_frame = frame.copy()

                    if self._state == "LIVE":
                        # Detect pieces in slots to draw their bboxes so user knows where to click
                        res = self._piece_detector.detect_pieces(frame)
                        self._available_pieces = res.pieces if res else []

                        # Draw slots
                        for p in self._available_pieces:
                            x, y, w, h = p.bbox
                            cv2.rectangle(
                                display_frame,
                                (x, y),
                                (x + w, y + h),
                                (200, 200, 200),
                                2,
                            )

                        # Draw grid bounds
                        gw = self._cell_w * 8
                        gh = self._cell_h * 8
                        bx, by = self._board_top_left
                        cv2.rectangle(
                            display_frame,
                            (bx, by),
                            (bx + gw, by + gh),
                            (100, 100, 255),
                            2,
                        )

                        dpg.set_value(
                            "frame_texture", self._frame_to_texture(display_frame)
                        )

                    elif self._state == "PLACING" and self._active_piece is not None:
                        # Draw board with current target cell highlighted
                        gw = self._cell_w * 8
                        gh = self._cell_h * 8
                        bx, by = self._board_top_left
                        cv2.rectangle(
                            display_frame,
                            (bx, by),
                            (bx + gw, by + gh),
                            (100, 100, 255),
                            1,
                        )

                        mx, my = dpg.get_mouse_pos(local=False)
                        rect_min = dpg.get_item_rect_min("frame_image")
                        img_x = int(mx - rect_min[0])
                        img_y = int(my - rect_min[1])

                        # Draw the piece centered on cursor
                        overlay = display_frame.copy()
                        cells = self._active_piece.cells
                        if not cells:
                            cells = [(0, 0)]

                        min_r = min(c[0] for c in cells)
                        max_r = max(c[0] for c in cells)
                        min_c = min(c[1] for c in cells)
                        max_c = max(c[1] for c in cells)

                        pw = (max_c - min_c + 1) * self._cell_w
                        ph = (max_r - min_r + 1) * self._cell_h
                        top_x = img_x - pw // 2
                        top_y = img_y - ph // 2

                        for r, c in cells:
                            rx = int(top_x + (c - min_c) * self._cell_w)
                            ry = int(top_y + (r - min_r) * self._cell_h)
                            cv2.rectangle(
                                overlay,
                                (rx, ry),
                                (rx + int(self._cell_w), ry + int(self._cell_h)),
                                (0, 255, 0),
                                -1,
                            )

                        cv2.addWeighted(
                            overlay, 0.4, display_frame, 0.6, 0, display_frame
                        )

                        # If hovering over grid, identify origin target
                        if bx <= img_x < bx + gw and by <= img_y < by + gh:
                            # The origin block of a piece is visually represented by cell (0, 0)
                            # Let's map the cursor to a specific cell, and trace back to where origin would land
                            # To be intuitive, the user's cursor is pointing at the literal center of the piece bound.

                            # Center offset to origin:
                            offset_r_px = (
                                ph // 2 - (0 - min_r) * self._cell_h - self._cell_h // 2
                            )
                            offset_c_px = (
                                pw // 2 - (0 - min_c) * self._cell_w - self._cell_w // 2
                            )

                            origin_px_x = img_x - offset_c_px
                            origin_px_y = img_y - offset_r_px

                            target_col = int((origin_px_x - bx) // self._cell_w)
                            target_row = int((origin_px_y - by) // self._cell_h)

                            dpg.set_value("t_row", f"Target Row: {target_row}")
                            dpg.set_value("t_col", f"Target Col: {target_col}")

                            # Draw outline of actual snapped origin drop
                            if 0 <= target_row < 8 and 0 <= target_col < 8:
                                # Draw full snapped piece
                                snp_x = bx + target_col * self._cell_w
                                snp_y = by + target_row * self._cell_h

                                for r, c in cells:
                                    rx = int(snp_x + c * self._cell_w)
                                    ry = int(snp_y + r * self._cell_h)
                                    cv2.rectangle(
                                        display_frame,
                                        (rx, ry),
                                        (
                                            rx + int(self._cell_w),
                                            ry + int(self._cell_h),
                                        ),
                                        (255, 255, 0),
                                        2,
                                    )
                        else:
                            dpg.set_value("t_row", "Target Row: None")
                            dpg.set_value("t_col", "Target Col: None")

                        dpg.set_value(
                            "frame_texture", self._frame_to_texture(display_frame)
                        )

            except Exception as e:
                time.sleep(0.01)

            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        return True

    def _cb_key_press(self, sender, app_data):
        if app_data == dpg.mvKey_Escape and self._state == "PLACING":
            self._state = "LIVE"
            self._active_piece = None
            dpg.set_value("text_status", "STATUS: LIVE MAP")
            dpg.configure_item("text_status", color=[0, 255, 0])

    def _cb_mouse_click(self):
        if self._state == "EXECUTING":
            return

        mx, my = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("frame_image")
        rect_max = dpg.get_item_rect_max("frame_image")

        if not (rect_min[0] <= mx <= rect_max[0] and rect_min[1] <= my <= rect_max[1]):
            return

        img_x = int(mx - rect_min[0])
        img_y = int(my - rect_min[1])

        if self._state == "LIVE":
            # Check if clicked on a piece
            for p in self._available_pieces:
                x, y, w, h = p.bbox
                if x <= img_x <= x + w and y <= img_y <= y + h:
                    # Determine slot (left, center, right -> 0, 1, 2)
                    self._active_slot_idx = 0
                    if x > self._current_frame.shape[1] * 0.66:
                        self._active_slot_idx = 2
                    elif x > self._current_frame.shape[1] * 0.33:
                        self._active_slot_idx = 1

                    self._active_piece = p
                    self._state = "PLACING"
                    dpg.set_value(
                        "text_status",
                        f"STATUS: PLACING (Slot {self._active_slot_idx + 1})",
                    )
                    dpg.configure_item("text_status", color=[255, 255, 0])
                    break

        elif self._state == "PLACING":
            # Attempt to execute drop
            # Recalculate snapped coords
            cells = self._active_piece.cells
            if not cells:
                cells = [(0, 0)]
            min_r = min(c[0] for c in cells)
            max_r = max(c[0] for c in cells)
            min_c = min(c[1] for c in cells)
            max_c = max(c[1] for c in cells)

            pw = (max_c - min_c + 1) * self._cell_w
            ph = (max_r - min_r + 1) * self._cell_h
            offset_r_px = ph // 2 - (0 - min_r) * self._cell_h - self._cell_h // 2
            offset_c_px = pw // 2 - (0 - min_c) * self._cell_w - self._cell_w // 2

            origin_px_x = img_x - offset_c_px
            origin_px_y = img_y - offset_r_px

            bx, by = self._board_top_left
            target_col = int((origin_px_x - bx) // self._cell_w)
            target_row = int((origin_px_y - by) // self._cell_h)

            if 0 <= target_row < 8 and 0 <= target_col < 8:
                self._state = "EXECUTING"
                dpg.set_value("text_status", f"STATUS: EXECUTING DROP...")
                dpg.configure_item("text_status", color=[255, 0, 0])

                def _do_swipe():
                    print(
                        f"Executing: Slot {self._active_slot_idx} -> Row {target_row}, Col {target_col}"
                    )
                    self._executor.execute_move(
                        self._active_slot_idx,
                        target_row,
                        target_col,
                        self._active_piece.cells,
                        duration_ms=500,
                    )
                    time.sleep(1.0)  # Wait for animation to finish visually
                    self._state = "LIVE"
                    self._active_piece = None
                    dpg.set_value("text_status", "STATUS: LIVE MAP")
                    dpg.configure_item("text_status", color=[0, 255, 0])

                threading.Thread(target=_do_swipe).start()

    def _cb_update_config(self, sender, app_data):
        self._config.input.scale_x = dpg.get_value("slider_scx")
        self._config.input.scale_y = dpg.get_value("slider_scy")
        self._config.input.popup_y_offset = dpg.get_value("slider_popy")

        self._executor = InputExecutor(self._adb, self._calib_data, self._config.input)
        dpg.set_value("text_status", "STATUS: SCALES UPDATED (Try dropping!)")
        dpg.configure_item("text_status", color=[255, 255, 100])

    def _cb_save_config(self, sender, app_data):
        from block_blast_bot.config import save_config

        save_config(self._config)
        dpg.set_value("text_status", "STATUS: SAVED TO config.yaml")
        dpg.configure_item("text_status", color=[0, 255, 0])


if __name__ == "__main__":
    app = VisualPlacementTester()
    sys.exit(0 if app.run() else 1)
