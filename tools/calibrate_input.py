"""
Block Blast Bot v2.0 - Input Calibration Tool

Interactive Dear PyGui tool for calibrating ADB swipe physics using manual alignment.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_blast_bot.capture import FrameCapture
from block_blast_bot.config import load_config
from block_blast_bot.input.adb_controller import ADBController
from block_blast_bot.vision.piece_detector import PieceDetector
from block_blast_bot.calibration import CalibrationData
from block_blast_bot.game_types import DetectedPiece


class InputCalibratorTool:
    def __init__(self) -> None:
        self._capture: Optional[FrameCapture] = None
        self._adb = ADBController()
        self._config = load_config()
        self._calibration_data: Optional[CalibrationData] = None

        # Load vision calibration
        calib_path = Path("calibration.json")
        if calib_path.exists():
            self._calibration_data = CalibrationData.load(calib_path)

        self._detector = PieceDetector(
            calibration=self._calibration_data,
            hard_threshold=self._config.vision.board.lab_threshold,
        )

        self._current_frame: Optional[np.ndarray] = None
        self._texture_id: Optional[int] = None

        self._slot_centers = []
        if self._calibration_data:
            self._slot_centers = [s.center for s in self._calibration_data.slots]

        # Results
        self._popup_y = 0
        self._scale_x = 1.0
        self._scale_y = 1.0

        # State Machine for Manual UI Alignment
        self._state = (
            "LIVE"  # LIVE, POPUP_WAIT, POPUP_SELECT, SWEEP_WAIT, SWEEP_SELECT, *CONFIRM
        )
        self._frozen_frame: Optional[np.ndarray] = None
        self._active_piece: Optional[DetectedPiece] = None
        self._active_slot_idx = 0
        self._locked_img_x = 0
        self._locked_img_y = 0
        self._target_dx = 200
        self._target_dy = -300
        self._duration_ms = 3000

    def run(self) -> bool:
        """Run the input calibration tool."""
        if not self._calibration_data:
            print(
                "ERROR: calibration.json missing. Run `uv run tools/calibrate.py` first."
            )
            return False

        self._capture = FrameCapture(self._config.capture)
        if not self._capture.initialize():
            print("ERROR: Could not initialize frame capture.")
            return False

        self._current_frame = self._capture.capture()
        height, width = self._current_frame.shape[:2]

        dpg.create_context()

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
            # Add global key handler for Enter (Confirm) and ESC (Cancel)
            dpg.add_key_press_handler(callback=self._cb_key_press)

        with dpg.window(tag="main_window"):
            with dpg.group(horizontal=True):
                # Left side: Live view
                with dpg.child_window(width=width + 20, height=height + 20):
                    dpg.add_image("frame_texture", tag="frame_image")

                # Right side: Controls
                with dpg.child_window(width=400):
                    dpg.add_text("Input Calibration", color=[100, 200, 255])
                    dpg.add_text(
                        "STATUS: LIVE MAP", tag="text_status", color=[0, 255, 0]
                    )
                    dpg.add_separator()

                    dpg.add_text("Target Slot:")
                    dpg.add_combo(
                        items=["Slot 1", "Slot 2", "Slot 3"],
                        default_value="Slot 1",
                        tag="combo_slot_idx",
                    )
                    dpg.add_separator()

                    dpg.add_text("1. Popup Y-Offset Calibration")
                    dpg.add_button(
                        label="Capture Popup (Manual Align)",
                        callback=self._cb_capture_popup,
                    )
                    dpg.add_text("Y-Offset: 0px", tag="text_popup_y")
                    dpg.add_separator()

                    dpg.add_text("2. Drag Scale Calibration")
                    dpg.add_button(
                        label="Capture Sweep X/Y (Manual Align)",
                        callback=self._cb_capture_sweep,
                    )
                    dpg.add_text("Scale X: 1.0", tag="text_scale_x")
                    dpg.add_text("Scale Y: 1.0", tag="text_scale_y")
                    dpg.add_separator()

                    dpg.add_text("3. Verification Test")
                    dpg.add_button(
                        label="Run Diagonal Test Sweep", callback=self._cb_test_sweep
                    )
                    dpg.add_separator()

                    dpg.add_text("4. Save System")
                    dpg.add_button(label="Save to config.yaml", callback=self._cb_save)

        dpg.create_viewport(
            title="Input Calibration", width=width + 450, height=height + 60
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        while dpg.is_dearpygui_running():
            try:
                if self._state == "LIVE":
                    frame = self._capture.capture()
                    if frame is not None:
                        # Draw detection overlays for debugging
                        result = self._detector.detect_pieces(frame)
                        if result and result.pieces:
                            for idx, p in enumerate(result.pieces):
                                x, y, w, h = p.bbox
                                cv2.rectangle(
                                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                                )
                                cv2.putText(
                                    frame,
                                    f"Slot {idx + 1}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )
                        self._current_frame = frame.copy()
                        dpg.set_value("frame_texture", self._frame_to_texture(frame))

                elif self._state in [
                    "POPUP_SELECT",
                    "SWEEP_SELECT",
                    "POPUP_CONFIRM",
                    "SWEEP_CONFIRM",
                ]:
                    # Manual alignment mode: draw the piece anchored to the mouse or locked
                    if (
                        self._frozen_frame is not None
                        and self._active_piece is not None
                    ):
                        display_frame = self._frozen_frame.copy()

                        if "CONFIRM" in self._state:
                            img_x, img_y = self._locked_img_x, self._locked_img_y
                        else:
                            # Get mouse pos relative to the image
                            mx, my = dpg.get_mouse_pos(local=False)
                            rect_min = dpg.get_item_rect_min("frame_image")
                            img_x = int(mx - rect_min[0])
                            img_y = int(my - rect_min[1])

                        # Draw the semi-transparent piece shape
                        if 0 <= img_x < width and 0 <= img_y < height:
                            overlay = display_frame.copy()
                            cells = self._active_piece.cells
                            cell_w, cell_h = self._calibration_data.grid.cell_size

                            if not cells:
                                cells = [(0, 0)]  # Fallback

                            min_r = min(c[0] for c in cells)
                            max_r = max(c[0] for c in cells)
                            min_c = min(c[1] for c in cells)
                            max_c = max(c[1] for c in cells)

                            pw = (max_c - min_c + 1) * cell_w
                            ph = (max_r - min_r + 1) * cell_h
                            top_x = img_x - pw // 2
                            top_y = img_y - ph // 2

                            for r, c in cells:
                                rx = int(top_x + (c - min_c) * cell_w)
                                ry = int(top_y + (r - min_r) * cell_h)
                                cv2.rectangle(
                                    overlay,
                                    (rx, ry),
                                    (rx + int(cell_w), ry + int(cell_h)),
                                    (0, 255, 0),
                                    -1,
                                )

                            # Blend transparently
                            cv2.addWeighted(
                                overlay, 0.4, display_frame, 0.6, 0, display_frame
                            )

                            # Draw center crosshair
                            is_confirm = "CONFIRM" in self._state
                            cross_color = (0, 0, 255) if is_confirm else (0, 255, 255)
                            cv2.drawMarker(
                                display_frame,
                                (img_x, img_y),
                                cross_color,
                                cv2.MARKER_CROSS,
                                10,
                                2,
                            )

                        dpg.set_value(
                            "frame_texture", self._frame_to_texture(display_frame)
                        )

            except Exception:
                time.sleep(0.01)

            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        return True

    def _frame_to_texture(self, frame: np.ndarray) -> list[float]:
        """Convert BGR numpy array to flat RGBA float list for Dear PyGui."""
        code = cv2.COLOR_BGR2RGBA if frame.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
        rgb = cv2.cvtColor(frame, code)
        return (rgb.astype(np.float32) / 255.0).flatten().tolist()

    def _get_active_slot_index(self) -> int:
        val = dpg.get_value("combo_slot_idx")
        if val == "Slot 2":
            return 1
        if val == "Slot 3":
            return 2
        return 0

    def _cb_cancel_state(self):
        """Cancel the current calibration step"""
        if self._state != "LIVE":
            self._state = "LIVE"
            dpg.set_value("text_status", "STATUS: CANCELLED -> LIVE MAP")
            dpg.configure_item("text_status", color=[255, 100, 100])

    def _cb_key_press(self, sender, app_data):
        """Global key handler for Confirm, Cancel, and Nudging"""
        if app_data == dpg.mvKey_Escape:
            self._cb_cancel_state()

        elif self._state in ["POPUP_CONFIRM", "SWEEP_CONFIRM"]:
            # Confirm
            if app_data in [dpg.mvKey_Return, dpg.mvKey_Spacebar]:
                self._confirm_calibration()
            # Nudge up/down/left/right by 1 px
            elif app_data == dpg.mvKey_Up:
                self._locked_img_y -= 1
            elif app_data == dpg.mvKey_Down:
                self._locked_img_y += 1
            elif app_data == dpg.mvKey_Left:
                self._locked_img_x -= 1
            elif app_data == dpg.mvKey_Right:
                self._locked_img_x += 1

    def _cb_mouse_click(self):
        """Handle clicks during the _SELECT states to lock in the calibration"""
        if self._state not in ["POPUP_SELECT", "SWEEP_SELECT"]:
            return

        mx, my = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("frame_image")
        rect_max = dpg.get_item_rect_max("frame_image")

        # Ensure click is inside the image bounds
        if not (rect_min[0] <= mx <= rect_max[0] and rect_min[1] <= my <= rect_max[1]):
            return

        self._locked_img_x = int(mx - rect_min[0])
        self._locked_img_y = int(my - rect_min[1])

        if self._state == "POPUP_SELECT":
            self._state = "POPUP_CONFIRM"
        elif self._state == "SWEEP_SELECT":
            self._state = "SWEEP_CONFIRM"

        dpg.set_value("text_status", "STATUS: PRESS ENTER TO CONFIRM OR ESC TO RETRY")
        dpg.configure_item("text_status", color=[255, 255, 100])

    def _confirm_calibration(self):
        """Finalize the calibration after user presses Enter in _CONFIRM state"""
        cx, cy = self._slot_centers[self._active_slot_idx]
        img_x, img_y = self._locked_img_x, self._locked_img_y

        if self._state == "POPUP_CONFIRM":
            self._popup_y = img_y - cy
            dpg.set_value("text_popup_y", f"Y-Offset: {self._popup_y}px")
            print(f"Manual Popup Y-Offset Confirmed: {self._popup_y}")
            self._state = "LIVE"
            dpg.set_value("text_status", "STATUS: POPUP CONFIRMED -> LIVE MAP")
            dpg.configure_item("text_status", color=[0, 255, 0])

        elif self._state == "SWEEP_CONFIRM":
            origin_hover_x = cx
            origin_hover_y = cy + self._popup_y

            observed_dx = img_x - origin_hover_x
            observed_dy = img_y - origin_hover_y

            # Ideal expected finger travel at 1.5s (50% of 3s)
            finger_dx = self._target_dx * 0.5
            finger_dy = self._target_dy * 0.5

            self._scale_x = observed_dx / finger_dx if finger_dx != 0 else 1.0
            self._scale_y = observed_dy / finger_dy if finger_dy != 0 else 1.0

            dpg.set_value("text_scale_x", f"Scale X: {self._scale_x:.3f}")
            dpg.set_value("text_scale_y", f"Scale Y: {self._scale_y:.3f}")
            print(
                f"Manual Scales Confirmed -> X: {self._scale_x:.3f}, Y: {self._scale_y:.3f}"
            )

            self._state = "LIVE"
            dpg.set_value("text_status", "STATUS: SWEEP CONFIRMED -> LIVE MAP")
            dpg.configure_item("text_status", color=[0, 255, 0])

    def _freeze_and_wait(self, delay: float, next_state: str):
        """Helper to sleep, capture frozen frame, and update UI state."""
        time.sleep(delay)
        frame = None
        for _ in range(5):
            try:
                frame = self._capture.capture()
                if frame is not None:
                    break
            except Exception:
                time.sleep(0.1)

        if frame is not None:
            self._frozen_frame = frame
            self._state = next_state
            dpg.set_value("text_status", "STATUS: CLICK TO ALIGN (Press ESC to Cancel)")
            dpg.configure_item("text_status", color=[255, 255, 0])
        else:
            self._state = "LIVE"
            dpg.set_value("text_status", "STATUS: FRAME CAPTURE FAILED")
            dpg.configure_item("text_status", color=[255, 0, 0])

    def _extract_active_piece(self) -> bool:
        """Extract the exact piece structure currently sitting in the targeted slot"""
        self._active_slot_idx = self._get_active_slot_index()

        # We need the live frame to pull the grid template
        frame = None
        for _ in range(3):
            try:
                frame = self._capture.capture()
                if frame is not None:
                    break
            except Exception:
                pass

        if frame is None:
            return False

        # Find the piece in the selected slot
        result = self._detector.detect_pieces(frame)
        if result and result.pieces:
            # Match the center of the bounding box to the nearest slot center
            target_cx, target_cy = self._slot_centers[self._active_slot_idx]
            closest_dist = float("inf")
            closest_piece = None

            for p in result.pieces:
                dist = (p.center[0] - target_cx) ** 2 + (p.center[1] - target_cy) ** 2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_piece = p

            if closest_piece:
                self._active_piece = closest_piece
                return True

        return False

    def _cb_capture_popup(self):
        """Extrapolate the piece, tap and hold, freeze, let user align."""
        if not self._slot_centers:
            return

        if not self._extract_active_piece():
            print("Failed to detect piece in the targeted slot.")
            dpg.set_value("text_status", "STATUS: FAILED TO EXTRACT PIECE FROM SLOT")
            dpg.configure_item("text_status", color=[255, 0, 0])
            return

        self._state = "POPUP_WAIT"
        dpg.set_value("text_status", "STATUS: HOLDING... DO NOT TOUCH")
        dpg.configure_item("text_status", color=[255, 100, 0])

        cx, cy = self._slot_centers[self._active_slot_idx]
        adb_cx, adb_cy = self._calibration_data.window_to_device(cx, cy)

        def _exec():
            # Tap and hold on slot for 2 seconds (using physical device coords)
            threading.Thread(
                target=self._adb.swipe, args=(adb_cx, adb_cy, adb_cx, adb_cy, 2000)
            ).start()
            self._freeze_and_wait(1.2, "POPUP_SELECT")

        threading.Thread(target=_exec).start()

    def _cb_capture_sweep(self):
        """Extrapolate the piece, sweep, freeze partway through, let user align."""
        if not self._slot_centers:
            return

        if not self._extract_active_piece():
            print("Failed to detect piece in the targeted slot.")
            dpg.set_value("text_status", "STATUS: FAILED TO EXTRACT PIECE FROM SLOT")
            dpg.configure_item("text_status", color=[255, 0, 0])
            return

        self._state = "SWEEP_WAIT"
        dpg.set_value("text_status", "STATUS: SWEEPING... DO NOT TOUCH")
        dpg.configure_item("text_status", color=[255, 100, 0])

        cx, cy = self._slot_centers[self._active_slot_idx]

        # Convert window coordinates to physical device coordinates
        adb_cx, adb_cy = self._calibration_data.window_to_device(cx, cy)
        adb_dest_x, adb_dest_y = self._calibration_data.window_to_device(
            cx + self._target_dx, cy + self._target_dy
        )

        def _exec():
            # Start slow swipe
            threading.Thread(
                target=self._adb.swipe,
                args=(adb_cx, adb_cy, adb_dest_x, adb_dest_y, self._duration_ms),
            ).start()
            self._freeze_and_wait(2.0, "SWEEP_SELECT")

        threading.Thread(target=_exec).start()

    def _cb_test_sweep(self):
        """Run a test sweep applying the newly found scalars to drop on the grid center"""
        if not self._slot_centers or not self._calibration_data:
            return

        self._active_slot_idx = self._get_active_slot_index()
        cx, cy = self._slot_centers[self._active_slot_idx]

        # Target Grid Cell: Row 4, Col 4 (Center of the board)
        try:
            target_cell = self._calibration_data.grid.cell_centers[4][4]
            dest_x, dest_y = target_cell
        except IndexError:
            return

        from block_blast_bot.input.coordinate import InputCalibration, calculate_swipe

        calib = InputCalibration(
            popup_y_offset=int(self._popup_y),
            scale_x=self._scale_x,
            scale_y=self._scale_y,
        )

        x1, y1, x2, y2 = calculate_swipe(
            slot_center_x=cx,
            slot_center_y=cy,
            dest_center_x=dest_x,
            dest_center_y=dest_y,
            calibration=calib,
        )

        adb_x1, adb_y1 = self._calibration_data.window_to_device(x1, y1)
        adb_x2, adb_y2 = self._calibration_data.window_to_device(x2, y2)

        dpg.set_value("text_status", "STATUS: DROPPING ON GRID [4][4]")
        dpg.configure_item("text_status", color=[100, 100, 255])

        threading.Thread(
            target=self._adb.swipe, args=(adb_x1, adb_y1, adb_x2, adb_y2, 200)
        ).start()

        # Reset text back to live
        def _reset_txt():
            time.sleep(1.0)
            if self._state == "LIVE":
                dpg.set_value("text_status", "STATUS: LIVE MAP")
                dpg.configure_item("text_status", color=[0, 255, 0])

        threading.Thread(target=_reset_txt).start()

    def _cb_save(self):
        """Save the configured popup and scale parameters back to config.yaml"""
        from block_blast_bot.config import save_config

        self._config.input.popup_y_offset = int(self._popup_y)
        self._config.input.scale_x = float(self._scale_x)
        self._config.input.scale_y = float(self._scale_y)

        if save_config(self._config):
            dpg.set_value("text_status", "STATUS: SAVED TO CONFIG.YAML")
            dpg.configure_item("text_status", color=[0, 255, 0])
            print("Successfully saved input calibration to config.yaml")
        else:
            dpg.set_value("text_status", "STATUS: SAVE FAILED")
            dpg.configure_item("text_status", color=[255, 0, 0])


def main() -> int:
    app = InputCalibratorTool()
    return 0 if app.run() else 1


if __name__ == "__main__":
    sys.exit(main())
