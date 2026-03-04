"""
Block Blast Bot v2.0 - Calibration Tool

Interactive Dear PyGui calibration tool for setting up the bot.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dearpygui.dearpygui as dpg

from block_blast_bot.calibration import (
    Calibrator,
    CalibrationData,
    GridCalibration,
    SlotCalibration,
)
from block_blast_bot.capture import FrameCapture
from block_blast_bot.config import CaptureConfig, load_config
from block_blast_bot.vision.board_detector import BoardDetector
from block_blast_bot.vision.piece_detector import PieceDetector


class CalibrationTool:
    """
    Interactive calibration tool with Dear PyGui.

    Provides a visual interface for:
    - Live frame preview from scrcpy
    - Auto-detection of grid and slots
    - Manual adjustment of regions
    - Threshold tuning
    - Save/load calibration files
    """

    def __init__(self) -> None:
        """Initialize the calibration tool."""
        self._capture: Optional[FrameCapture] = None
        self._calibrator = Calibrator()
        self._calibration: Optional[CalibrationData] = None
        self._current_frame: Optional[NDArray[np.uint8]] = None
        self._display_frame: Optional[NDArray[np.uint8]] = None
        self._texture_id: Optional[int] = None
        self._running = False
        self._device_resolution = (1080, 1920)  # Default device resolution

        # Manual adjustment state
        self._grid_bounds = [100, 200, 400, 400]  # x, y, w, h
        self._slot_centers = [[270, 850], [270, 950], [270, 1050]]
        self._lab_threshold = 24
        self._morph_kernel = 5

        # UI state
        self._show_grid = True
        self._show_slots = True
        self._show_cell_centers = False
        self._show_live_detection = False

        self._config = load_config()
        self._auto_detect_enabled = True

    def run(self) -> bool:
        """
        Run the calibration tool.

        Returns:
            True if calibration was successful, False otherwise
        """
        try:
            # Initialize frame capture
            self._capture = FrameCapture(CaptureConfig())
            if not self._capture.initialize():
                print("ERROR: Could not initialize frame capture.")
                print("Make sure scrcpy is running with the device connected.")
                return False

            # Get initial frame
            self._current_frame = self._capture.capture()
            height, width = self._current_frame.shape[:2]

            # Initialize Dear PyGui
            dpg.create_context()

            # Create texture for frame display
            with dpg.texture_registry():
                self._texture_id = dpg.add_dynamic_texture(
                    width,
                    height,
                    self._frame_to_texture(self._current_frame),
                    tag="frame_texture",
                )

            # Create viewport
            dpg.create_viewport(
                title="Block Blast Bot - Calibration Tool", width=1200, height=800
            )
            dpg.set_viewport_small_icon(None)
            dpg.set_viewport_large_icon(None)

            # Build UI
            self._build_ui(width, height)

            # Setup viewport
            dpg.setup_dearpygui()
            dpg.show_viewport()

            # Run main loop
            self._running = True
            last_frame_time = time.perf_counter()
            frame_interval = 1.0 / 15  # 15 FPS for UI

            while dpg.is_dearpygui_running() and self._running:
                current_time = time.perf_counter()
                if current_time - last_frame_time >= frame_interval:
                    self._update_frame()
                    last_frame_time = current_time
                dpg.render_dearpygui_frame()

            # Cleanup
            dpg.destroy_context()
            if self._capture:
                self._capture.close()

            return self._calibration is not None and self._calibration.is_valid()

        except Exception as e:
            print(f"ERROR: Calibration tool failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _build_ui(self, frame_width: int, frame_height: int) -> None:
        """Build the Dear PyGui interface."""
        with dpg.window(label="Calibration Tool", tag="main_window"):
            # Main layout: image on left, controls on right
            with dpg.group(horizontal=True):
                # Left side: Frame display
                with dpg.child_window(width=600, height=700):
                    dpg.add_text("Live Preview")

                    # Scale image to fit within roughly 580x650
                    scale = min(580 / frame_width, 650 / frame_height)
                    scaled_w = int(frame_width * scale)
                    scaled_h = int(frame_height * scale)

                    dpg.add_image(
                        "frame_texture",
                        tag="frame_display",
                        width=scaled_w,
                        height=scaled_h,
                    )

                    # Status text
                    dpg.add_text("", tag="status_text")
                    dpg.add_text("", tag="validation_text")

                # Right side: Controls
                with dpg.child_window(width=550, height=700):
                    # Device settings
                    with dpg.collapsing_header(
                        label="Device Settings", default_open=True
                    ):
                        dpg.add_input_int(
                            label="Device Width",
                            default_value=self._device_resolution[0],
                            tag="device_width",
                            callback=self._on_device_resolution_change,
                            width=150,
                        )
                        dpg.add_input_int(
                            label="Device Height",
                            default_value=self._device_resolution[1],
                            tag="device_height",
                            callback=self._on_device_resolution_change,
                            width=150,
                        )
                        dpg.add_text(
                            f"Window: {frame_width}x{frame_height}",
                            tag="window_resolution_text",
                        )

                    # Auto-detection
                    with dpg.collapsing_header(
                        label="Auto-Detection", default_open=True
                    ):
                        dpg.add_button(
                            label="Auto Detect All",
                            callback=self._on_auto_detect,
                            width=200,
                        )
                        dpg.add_button(
                            label="Auto Detect Grid Only",
                            callback=self._on_auto_detect_grid,
                            width=200,
                        )
                        dpg.add_button(
                            label="Auto Detect Slots Only",
                            callback=self._on_auto_detect_slots,
                            width=200,
                        )

                    # Grid settings
                    with dpg.collapsing_header(label="Grid Settings"):
                        dpg.add_checkbox(
                            label="Show Grid Overlay",
                            default_value=self._show_grid,
                            callback=self._on_show_grid_change,
                            tag="show_grid",
                        )
                        dpg.add_checkbox(
                            label="Show Cell Centers",
                            default_value=self._show_cell_centers,
                            callback=self._on_show_cell_centers_change,
                            tag="show_cell_centers",
                        )
                        dpg.add_checkbox(
                            label="Show Live Detection",
                            default_value=self._show_live_detection,
                            callback=self._on_show_live_detection_change,
                            tag="show_live_detection",
                        )
                        dpg.add_text("Grid Bounds (x, y, width, height):")
                        with dpg.group(horizontal=True):
                            dpg.add_input_int(
                                label="X",
                                default_value=self._grid_bounds[0],
                                tag="grid_x",
                                callback=self._on_grid_change,
                                width=80,
                            )
                            dpg.add_input_int(
                                label="Y",
                                default_value=self._grid_bounds[1],
                                tag="grid_y",
                                callback=self._on_grid_change,
                                width=80,
                            )
                            dpg.add_input_int(
                                label="W",
                                default_value=self._grid_bounds[2],
                                tag="grid_w",
                                callback=self._on_grid_change,
                                width=80,
                            )
                            dpg.add_input_int(
                                label="H",
                                default_value=self._grid_bounds[3],
                                tag="grid_h",
                                callback=self._on_grid_change,
                                width=80,
                            )

                    # Slot settings
                    with dpg.collapsing_header(
                        label="Slot Settings", default_open=True
                    ):
                        dpg.add_checkbox(
                            label="Show Slot Overlays",
                            default_value=self._show_slots,
                            callback=self._on_show_slots_change,
                            tag="show_slots",
                        )
                        dpg.add_text("Slot Size (Width / Height):")
                        with dpg.group(horizontal=True):
                            dpg.add_input_int(
                                label="W",
                                default_value=150,
                                tag="slot_w",
                                callback=self._on_slot_change,
                                width=100,
                            )
                            dpg.add_input_int(
                                label="H",
                                default_value=150,
                                tag="slot_h",
                                callback=self._on_slot_change,
                                width=100,
                            )
                        dpg.add_text("Slot Centers (click to set):")
                        for i in range(3):
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"Slot {i}:")
                                dpg.add_input_int(
                                    label="X",
                                    default_value=self._slot_centers[i][0],
                                    tag=f"slot_{i}_x",
                                    callback=self._on_slot_change,
                                    width=80,
                                )
                                dpg.add_input_int(
                                    label="Y",
                                    default_value=self._slot_centers[i][1],
                                    tag=f"slot_{i}_y",
                                    callback=self._on_slot_change,
                                    width=80,
                                )

                    # Detection parameters
                    with dpg.collapsing_header(label="Detection Parameters"):
                        dpg.add_slider_int(
                            label="LAB Threshold",
                            default_value=self._lab_threshold,
                            min_value=1,
                            max_value=100,
                            tag="lab_threshold",
                            callback=self._on_threshold_change,
                            width=200,
                        )
                        dpg.add_slider_int(
                            label="Morph Kernel Size",
                            default_value=self._morph_kernel,
                            min_value=1,
                            max_value=15,
                            tag="morph_kernel",
                            callback=self._on_threshold_change,
                            width=200,
                        )

                    # File operations
                    with dpg.collapsing_header(
                        label="File Operations", default_open=True
                    ):
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Save Calibration",
                                callback=self._on_save,
                                width=150,
                            )
                            dpg.add_button(
                                label="Load Calibration",
                                callback=self._on_load,
                                width=150,
                            )
                        dpg.add_input_text(
                            label="File Path",
                            default_value="calibration.json",
                            tag="calibration_path",
                            width=200,
                        )

                    # Validation
                    with dpg.collapsing_header(label="Validation"):
                        dpg.add_button(
                            label="Validate Calibration",
                            callback=self._on_validate,
                            width=200,
                        )
                        dpg.add_text("", tag="validation_result")

                    # Actions
                    with dpg.collapsing_header(label="Actions", default_open=True):
                        dpg.add_button(
                            label="Apply & Close", callback=self._on_apply, width=200
                        )
                        dpg.add_button(
                            label="Cancel", callback=self._on_cancel, width=200
                        )

    def _frame_to_texture(self, frame: NDArray[np.uint8]) -> list:
        """Convert BGR frame to RGBA texture data."""
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba = rgba.flatten().astype(np.float32) / 255.0
        return rgba.tolist()

    def _update_frame(self) -> None:
        """Update the frame display."""
        if not self._capture:
            return

        try:
            # Capture new frame
            self._current_frame = self._capture.capture()

            # Draw overlays
            display = self._current_frame.copy()
            self._draw_overlays(display)

            # Update texture
            if self._texture_id:
                height, width = display.shape[:2]
                texture_data = self._frame_to_texture(display)
                dpg.set_value("frame_texture", texture_data)

            # Update status
            if self._calibration:
                dpg.set_value(
                    "status_text",
                    f"Calibration ready - Grid: {self._calibration.grid.bounds if self._calibration.grid else 'None'}",
                )
            else:
                dpg.set_value(
                    "status_text", "No calibration - Use Auto Detect or manual settings"
                )

        except Exception as e:
            dpg.set_value("status_text", f"Error: {e}")

    def _draw_overlays(self, frame: NDArray[np.uint8]) -> None:
        """Draw calibration overlays on the frame."""
        if self._show_grid and self._calibration and self._calibration.grid:
            grid = self._calibration.grid
            x, y, w, h = grid.bounds

            # Draw grid outline
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw cell grid lines
            cell_w, cell_h = grid.cell_size
            for i in range(1, 8):
                # Vertical lines
                cv2.line(
                    frame, (x + i * cell_w, y), (x + i * cell_w, y + h), (0, 255, 0), 1
                )
                # Horizontal lines
                cv2.line(
                    frame, (x, y + i * cell_h), (x + w, y + i * cell_h), (0, 255, 0), 1
                )

            # Draw cell centers or live detection if enabled
            if self._show_live_detection:
                # Update config thresholds to match UI sliders
                self._config.vision.board.lab_threshold = getattr(
                    self._calibration.detection, "lab_threshold", self._lab_threshold
                )
                self._config.vision.board.morph_kernel_size = getattr(
                    self._calibration.detection, "morph_kernel_size", self._morph_kernel
                )

                try:
                    detector = BoardDetector(
                        self._calibration,
                        self._config.vision.board.lab_threshold,
                        self._config.vision.board.morph_kernel_size,
                    )
                    board_state = detector.detect(self._current_frame).to_board_state()

                    for row in range(8):
                        for col in range(8):
                            cx, cy = grid.cell_centers[row][col]
                            if board_state.grid[row, col]:
                                # Filled cell (Green)
                                cv2.circle(
                                    frame,
                                    (cx, cy),
                                    max(2, cell_w // 6),
                                    (0, 255, 0),
                                    -1,
                                )
                            else:
                                # Empty cell (Red)
                                cv2.circle(
                                    frame,
                                    (cx, cy),
                                    max(2, cell_w // 10),
                                    (0, 0, 255),
                                    -1,
                                )
                except Exception as e:
                    # Ignore detection errors during dynamic resizing
                    import traceback

                    print(f"Board Detection Error: {e}")
                    traceback.print_exc()
            elif self._show_cell_centers:
                for row in grid.cell_centers:
                    for cx, cy in row:
                        cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

        if self._show_slots and self._calibration:
            for i, slot in enumerate(self._calibration.slots):
                cx, cy = slot.center
                sx, sy, sw, sh = slot.bounds

                # Draw slot bounds
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)

                # Draw slot label
                cv2.putText(
                    frame,
                    f"S{i}",
                    (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )

            if self._show_live_detection:
                try:
                    p_detector = PieceDetector(self._calibration)
                    pieces = p_detector.detect_pieces(self._current_frame).pieces

                    for piece in pieces:
                        s = piece.cell_size
                        if not piece.cells:
                            continue

                        max_r = max(r for r, c in piece.cells)
                        max_c = max(c for r, c in piece.cells)

                        # Calculate the top-left origin for drawing relative to center
                        origin_x = piece.center[0] - (max_c * s) // 2
                        origin_y = piece.center[1] - (max_r * s) // 2

                        for r, c in piece.cells:
                            px = origin_x + (c * s) - s // 2
                            py = origin_y + (r * s) - s // 2
                            cv2.rectangle(
                                frame, (px, py), (px + s, py + s), (0, 255, 255), 2
                            )
                            cv2.circle(
                                frame,
                                (px + s // 2, py + s // 2),
                                max(2, s // 8),
                                (0, 165, 255),
                                -1,
                            )
                except Exception as e:
                    import traceback

                    print(f"Piece Detection Error: {e}")
                    traceback.print_exc()
            else:
                for i, slot in enumerate(self._calibration.slots):
                    cx, cy = slot.center
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

    def _on_device_resolution_change(self, sender, app_data) -> None:
        """Handle device resolution change."""
        self._device_resolution = (
            dpg.get_value("device_width"),
            dpg.get_value("device_height"),
        )

    def _on_auto_detect(self, sender, app_data) -> None:
        """Handle auto-detect all button."""
        if self._current_frame is None:
            return

        self._calibration = self._calibrator.create_calibration_from_frame(
            self._current_frame, self._device_resolution
        )

        if self._calibration:
            self._update_ui_from_calibration()
            dpg.set_value("status_text", "Auto-detection complete!")
        else:
            dpg.set_value(
                "status_text", "Auto-detection failed - try manual adjustment"
            )

    def _on_auto_detect_grid(self, sender, app_data) -> None:
        """Handle auto-detect grid button."""
        if self._current_frame is None:
            return

        grid = self._calibrator.auto_detect_grid(self._current_frame)
        if grid:
            if not self._calibration:
                self._calibration = CalibrationData(
                    device_resolution=self._device_resolution,
                    window_resolution=self._current_frame.shape[:2][::-1],
                )
            self._calibration.grid = grid
            self._update_ui_from_calibration()
            dpg.set_value("status_text", "Grid detected!")
        else:
            dpg.set_value("status_text", "Grid detection failed")

    def _on_auto_detect_slots(self, sender, app_data) -> None:
        """Handle auto-detect slots button."""
        if (
            self._current_frame is None
            or not self._calibration
            or not self._calibration.grid
        ):
            dpg.set_value("status_text", "Detect grid first!")
            return

        slots = self._calibrator.auto_detect_slots(
            self._current_frame, self._calibration.grid
        )
        if slots:
            self._calibration.slots = slots
            self._update_ui_from_calibration()
            dpg.set_value("status_text", f"Detected {len(slots)} slots!")
        else:
            dpg.set_value("status_text", "Slot detection failed")

    def _on_grid_change(self, sender, app_data) -> None:
        """Handle manual grid bounds change."""
        x = dpg.get_value("grid_x")
        y = dpg.get_value("grid_y")
        w = dpg.get_value("grid_w")
        h = dpg.get_value("grid_h")

        if w > 0 and h > 0:
            cell_w = w // 8
            cell_h = h // 8

            if not self._calibration:
                self._calibration = CalibrationData(
                    device_resolution=self._device_resolution,
                    window_resolution=self._current_frame.shape[:2][::-1]
                    if self._current_frame is not None
                    else (0, 0),
                )

            self._calibration.grid = GridCalibration(
                bounds=(x, y, w, h),
                cell_size=(cell_w, cell_h),
            )

    def _on_slot_change(self, sender, app_data) -> None:
        """Handle manual slot center change."""
        if not self._calibration:
            self._calibration = CalibrationData(
                device_resolution=self._device_resolution,
                window_resolution=self._current_frame.shape[:2][::-1]
                if self._current_frame is not None
                else (0, 0),
            )

        slots = []

        sw = dpg.get_value("slot_w")
        sh = dpg.get_value("slot_h")
        if sw <= 0:
            sw = 100
        if sh <= 0:
            sh = 100

        for i in range(3):
            cx = dpg.get_value(f"slot_{i}_x")
            cy = dpg.get_value(f"slot_{i}_y")

            slots.append(
                SlotCalibration(
                    center=(cx, cy),
                    bounds=(cx - sw // 2, cy - sh // 2, sw, sh),
                )
            )

        self._calibration.slots = slots

    def _on_threshold_change(self, sender, app_data) -> None:
        """Handle detection threshold change."""
        self._lab_threshold = dpg.get_value("lab_threshold")
        self._morph_kernel = dpg.get_value("morph_kernel")

        if self._calibration:
            self._calibration.detection.lab_threshold = self._lab_threshold
            self._calibration.detection.morph_kernel_size = self._morph_kernel

    def _on_show_grid_change(self, sender, app_data) -> None:
        """Handle show grid checkbox change."""
        self._show_grid = dpg.get_value("show_grid")

    def _on_show_slots_change(self, sender, app_data) -> None:
        """Handle show slots checkbox change."""
        self._show_slots = dpg.get_value("show_slots")

    def _on_show_cell_centers_change(self, sender, app_data) -> None:
        """Handle show cell centers checkbox change."""
        self._show_cell_centers = dpg.get_value("show_cell_centers")

    def _on_show_live_detection_change(self, sender, app_data) -> None:
        """Handle show live detection checkbox change."""
        self._show_live_detection = dpg.get_value("show_live_detection")

    def _on_save(self, sender, app_data) -> None:
        """Handle save calibration button."""
        if not self._calibration:
            dpg.set_value("status_text", "No calibration to save!")
            return

        path = dpg.get_value("calibration_path")
        if self._calibrator.save_calibration(self._calibration, path):
            dpg.set_value("status_text", f"Calibration saved to {path}")
        else:
            dpg.set_value("status_text", "Failed to save calibration")

    def _on_load(self, sender, app_data) -> None:
        """Handle load calibration button."""
        path = dpg.get_value("calibration_path")
        cal = self._calibrator.load_calibration(path)
        if cal:
            self._calibration = cal
            self._update_ui_from_calibration()
            dpg.set_value("status_text", f"Calibration loaded from {path}")
        else:
            dpg.set_value("status_text", "Failed to load calibration")

    def _on_validate(self, sender, app_data) -> None:
        """Handle validate calibration button."""
        if not self._calibration:
            dpg.set_value("validation_result", "No calibration to validate!")
            return

        errors = self._calibrator.validate_calibration(self._calibration)
        if errors:
            error_text = "Errors:\n" + "\n".join(f"• {e}" for e in errors)
            dpg.set_value("validation_result", error_text)
        else:
            dpg.set_value("validation_result", "✓ Calibration is valid!")

    def _on_apply(self, sender, app_data) -> None:
        """Handle apply and close button."""
        if self._calibration and self._calibration.is_valid():
            self._running = False
        else:
            dpg.set_value("status_text", "Calibration is not valid - check all fields")

    def _on_cancel(self, sender, app_data) -> None:
        """Handle cancel button."""
        self._calibration = None
        self._running = False

    def _update_ui_from_calibration(self) -> None:
        """Update UI controls from calibration data."""
        if not self._calibration:
            return

        # Update device resolution
        dpg.set_value("device_width", self._calibration.device_resolution[0])
        dpg.set_value("device_height", self._calibration.device_resolution[1])
        self._device_resolution = self._calibration.device_resolution

        # Update grid
        if self._calibration.grid:
            x, y, w, h = self._calibration.grid.bounds
            dpg.set_value("grid_x", x)
            dpg.set_value("grid_y", y)
            dpg.set_value("grid_w", w)
            dpg.set_value("grid_h", h)

        # Update slots
        if self._calibration.slots:
            sw = self._calibration.slots[0].bounds[2]
            sh = self._calibration.slots[0].bounds[3]
            dpg.set_value("slot_w", sw)
            dpg.set_value("slot_h", sh)

            for i, slot in enumerate(self._calibration.slots):
                if i < 3:
                    dpg.set_value(f"slot_{i}_x", slot.center[0])
                    dpg.set_value(f"slot_{i}_y", slot.center[1])

        # Update detection parameters
        dpg.set_value("lab_threshold", self._calibration.detection.lab_threshold)
        dpg.set_value("morph_kernel", self._calibration.detection.morph_kernel_size)


def main() -> int:
    """Run interactive calibration."""
    print("Block Blast Bot - Calibration Tool")
    print("===================================")
    print()
    print("This tool will guide you through calibrating the bot for your device.")
    print()
    print("Prerequisites:")
    print("  1. Connect your Android device via USB")
    print("  2. Enable USB debugging on the device")
    print("  3. Run scrcpy to mirror the device screen")
    print("  4. Open Block Blast game on the device")
    print()

    tool = CalibrationTool()
    result = tool.run()

    if result:
        print()
        print("Calibration completed successfully!")
        print("Calibration data saved to: calibration.json")
        return 0
    else:
        print()
        print("Calibration was cancelled or failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
