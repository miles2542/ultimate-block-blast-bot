"""
Block Blast Bot v2.0 - Main Entry Point

The master orchestrator defining the UI, control loop, and system integration.
"""

import sys
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from block_blast_bot.config import load_config
from block_blast_bot.capture.frame_capture import create_capture
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.pipeline import VisionPipeline
from block_blast_bot.state_detector import StateDetector, GameScreen
from block_blast_bot.capture.stability import StabilityGate
from block_blast_bot.input.adb_controller import ADBController
from block_blast_bot.input.execution import InputExecutor
from block_blast_bot.ui.overlay import OverlayRenderer
from block_blast_bot.ui.controller import UIController
from block_blast_bot.game_types import AppMode, SolverMode, DisplayMode

from block_blast_bot.rust_core import Solver, SolverProfile

logger = logging.getLogger(__name__)


# Enums moved to game_types.py


@dataclass
class SharedState:
    """Thread-safe data shared between the backend engine and the GUI frontend."""

    # Settings
    mode: str = AppMode.MANUAL
    display: str = DisplayMode.LEAN
    lookahead_depth: int = 1
    lookahead_samples: int = 30
    solver_mode: str = SolverMode.AGGRESSIVE

    # Custom weights (matching config.yaml / SolverWeightsConfig)
    custom_hole_penalty: float = -88.0
    custom_density_penalty: float = -3.08
    custom_score_reward: float = 5.1
    custom_combo_preservation_reward: float = 32.7
    custom_bumpiness_penalty: float = -0.31
    custom_well_depth_penalty: float = -21.9

    time_limit_ms: int = 1500
    pre_drop_ms: int = 200
    post_drop_ms: int = 400
    post_turn_ms: int = 600
    auto_restart: bool = False

    # Dynamic UI state for Smart Resizing
    view_width: int = 450
    view_height: int = 1000

    # Internal Trigger
    trigger_semi_auto: bool = False

    # Telemetry for GUI
    current_bgr: Optional[np.ndarray] = None
    fps: float = 0.0
    status_text: str = "INITIALIZING..."
    board_cells: Optional[list] = None  # 8x8 list of bools
    live_pieces: list = field(default_factory=list)  # list of (slot_index, cells)
    best_move: Optional[dict] = None
    gui_texture_buffer: Optional[np.ndarray] = None  # float32 RGBA buffer
    loop_heartbeat: int = 0


class BotOrchestrator:
    def __init__(self):
        self.config = load_config()
        self.state = SharedState()

        # Initialize custom weights from config as default baseline
        w = self.config.solver.weights
        self.state.custom_hole_penalty = w.hole_penalty
        self.state.custom_density_penalty = w.density_penalty
        self.state.custom_score_reward = w.score_reward
        self.state.custom_combo_preservation_reward = w.combo_preservation_reward
        self.state.custom_bumpiness_penalty = w.bumpiness_penalty
        self.state.custom_well_depth_penalty = w.well_depth_penalty
        self.state.lookahead_samples = self.config.solver.depth2_samples

        # Load delays from config
        self.state.pre_drop_ms = self.config.input.pre_drop_ms
        self.state.post_drop_ms = self.config.input.post_drop_ms
        self.state.post_turn_ms = self.config.input.post_turn_ms

        self.calib_data = CalibrationData.load("calibration.json")

        if not self.calib_data.is_valid():
            logger.error(
                "Calibration data is invalid or missing. Run Phase 4 calibration first."
            )
            sys.exit(1)

        # CRITICAL: Initialize capture immediately to lock window and dimensions
        self.capture = create_capture(self.config.capture)
        if not self.capture.initialize():
            logger.error("Failed to initialize capture. Ensure scrcpy is running.")
            sys.exit(1)

        self.adb = ADBController()

        self.vision = VisionPipeline(self.calib_data, self.config.vision)
        self.state_detector = StateDetector(self.calib_data)
        self.executor = InputExecutor(self.adb, self.calib_data, self.config.input)
        self.renderer = OverlayRenderer(self.calib_data)

        # Stability Gate (ROI-based animation check)
        self.stability_gate = StabilityGate(
            self.calib_data, threshold=1.0, required_ms=200
        )
        self._dead_board_count = 0

        # UI Controller
        self.ui = UIController(
            self.state,
            capture_res_provider=self.capture.get_resolution,
            refresh_cb=self._on_manual_refresh,
            semi_auto_cb=self._on_semi_auto_trigger,
        )

        # Start background reasoning thread
        self._running = False
        self._bot_thread = None
        self._execution_lock = threading.Lock()
        self._executing = False
        self._last_pieces = None

        self._running = True
        self._bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self._bot_thread.start()

    def _on_manual_refresh(self):
        self._last_pieces = None
        self._executing = False
        self._execution_start_time = 0.0
        self.state.best_move = None
        self.state.status_text = "FORCED REFRESH (SYNC)"
        logger.info("Manual sync: Resetting piece hash and execution flags.")

    def _on_semi_auto_trigger(self):
        self.state.trigger_semi_auto = True

    def _get_preset_weights(self) -> SolverProfile:
        """Returns the base profile to use before applying overrides."""
        if self.state.solver_mode == SolverMode.SAFE:
            return SolverProfile.SafeEndurance
        if self.state.solver_mode == SolverMode.CUSTOM:
            return SolverProfile.SafeEndurance
        return SolverProfile.Aggressive

    def _board_to_mask(self, board_cells: list[list[bool]]) -> int:
        """Converts an 8x8 boolean board to a 64-bit integer mask."""
        board_mask = 0
        for r in range(8):
            for c in range(8):
                if board_cells[r][c]:
                    board_mask |= 1 << (r * 8 + c)
        return board_mask

    def _bot_loop(self):
        """The Master Game Loop."""
        last_time = time.time()
        frames = 0

        while self._running:
            try:
                # 0. Heartbeat
                self.state.loop_heartbeat = (self.state.loop_heartbeat + 1) % 1000

                # Timing
                current_time = time.time()
                frames += 1
                if current_time - last_time >= 1.0:
                    self.state.fps = frames
                    frames = 0
                    last_time = current_time

                # 1. Capture
                frame = self.capture.capture()
                if frame is None:
                    self.state.status_text = "ERROR: Scrcpy Feed Lost"
                    time.sleep(0.1)
                    continue

                self.state.current_bgr = frame

                # 2. State Detection
                screen_state = self.state_detector.detect_screen(frame)

                # 3. Decision Logic (Structured to allow GUI update on every frame)
                game_state = None
                if screen_state == GameScreen.GAME_OVER:
                    self.state.status_text = "GAME OVER"
                    self.stability_gate.reset()
                    time.sleep(1.0)
                elif screen_state != GameScreen.PLAYING:
                    self.state.status_text = f"IDLE ({screen_state.value})"
                    self.stability_gate.reset()
                    time.sleep(0.1)
                else:
                    # 1. WATCHDOG: Reset execution if it takes too long (> 15s)
                    if self._executing and (
                        time.time() - self._execution_start_time > 15.0
                    ):
                        logger.warning("Watchdog: Execution stalled! Forcing reset.")
                        self._executing = False
                        self.state.status_text = "WATCHDOG RESET"

                    # 2. STABILITY GATE: Wait for animations to settle
                    if self._executing:
                        # Still moving, reset stability
                        self.stability_gate.reset()
                    else:
                        is_stable = self.stability_gate.check_stability(frame)
                        if not is_stable:
                            self.state.status_text = "STABILIZING FEED..."
                        else:
                            # 2. CAPTURE & VISION (Only if stable)
                            game_state = self.vision.process(frame)
                            if not game_state:
                                self.state.status_text = "WAITING FOR BOARD..."
                                self.state.board_cells = None
                                self.state.live_pieces = []
                                time.sleep(0.1)
                            else:
                                self.state.board_cells = game_state.board_cells
                                self.state.live_pieces = game_state.available_pieces

                                # Turn Detection
                                current_pieces_hash = str(
                                    [
                                        (s, c)
                                        for s, c in game_state.available_pieces
                                        if s != -1
                                    ]
                                )
                                is_new_turn = current_pieces_hash != self._last_pieces

                                if not game_state.available_pieces:
                                    self.state.status_text = "WAITING FOR PIECES..."
                                    time.sleep(0.1)
                                else:
                                    # 3. Reasoning/Solver (Only if stable and it's a new turn)
                                    if is_new_turn and not self._executing:
                                        self.state.best_move = (
                                            None  # Clear stale projection while solving
                                        )
                                        self.state.status_text = (
                                            "CONSIDERING OPTIONS..."
                                        )
                                        self._last_pieces = current_pieces_hash

                                        # Process Solver...
                                        solver = Solver(self._get_preset_weights())
                                        if self.state.solver_mode == SolverMode.CUSTOM:
                                            solver.set_weights(
                                                hole_penalty=self.state.custom_hole_penalty,
                                                density_penalty=self.state.custom_density_penalty,
                                                score_reward=self.state.custom_score_reward,
                                                combo_preservation_reward=self.state.custom_combo_preservation_reward,
                                                bumpiness_penalty=self.state.custom_bumpiness_penalty,
                                                well_depth_penalty=self.state.custom_well_depth_penalty,
                                            )

                                        optimal_turn_dict = solver.solve(
                                            board_mask=self._board_to_mask(
                                                game_state.board_cells
                                            ),
                                            pieces=[
                                                c
                                                for _, c in game_state.available_pieces
                                            ],
                                            combo=0,
                                            time_limit_ms=self.state.time_limit_ms,
                                            lookahead_depth=self.state.lookahead_depth,
                                            lookahead_samples=self.state.lookahead_samples,
                                        )

                                        if (
                                            not optimal_turn_dict
                                            or not optimal_turn_dict.get("placements")
                                        ):
                                            self.state.status_text = "NO VALID MOVES?"
                                            self._dead_board_count += 1
                                            self.state.best_move = None

                                            # PHASE 5.11: DEAD BOARD RECOVERY
                                            if self._dead_board_count == 1:
                                                # First attempt failed: Wait 1.5s then force a refresh
                                                logger.warning(
                                                    "No moves found. Retrying in 1.5s..."
                                                )
                                                time.sleep(1.5)
                                                self._on_manual_refresh()
                                            else:
                                                # Still no moves? Beep!
                                                self.state.status_text = (
                                                    "DEAD BOARD (STUCK)"
                                                )
                                                try:
                                                    import winsound

                                                    winsound.Beep(1000, 3000)
                                                except Exception:
                                                    pass
                                        else:
                                            self._dead_board_count = 0
                                            placements = optimal_turn_dict["placements"]
                                            piece_seq = optimal_turn_dict[
                                                "piece_sequence"
                                            ]
                                            original_slot_indices = [
                                                s
                                                for s, _ in game_state.available_pieces
                                            ]
                                            pure_pieces = [
                                                c
                                                for _, c in game_state.available_pieces
                                            ]
                                            first_move_slot = original_slot_indices[
                                                piece_seq[0]
                                            ]

                                            self.state.best_move = {
                                                "slot_index": first_move_slot,
                                                "target_row": placements[0][
                                                    "target_row"
                                                ],
                                                "target_col": placements[0][
                                                    "target_col"
                                                ],
                                                "full_placements": placements,
                                                "piece_seq": piece_seq,
                                                "original_slot_indices": original_slot_indices,
                                                "pure_pieces": pure_pieces,
                                            }
                                            self.state.status_text = (
                                                f"READY (Slot {first_move_slot})"
                                            )

                            # 5. Execution Trigger (STRICTLY gated by stability and non-executing state)
                            should_execute = False
                            if self.state.mode == AppMode.FULL_AUTO:
                                should_execute = True
                            elif (
                                self.state.mode == AppMode.SEMI_AUTO
                                and self.state.trigger_semi_auto
                            ):
                                should_execute = True
                                self.state.trigger_semi_auto = False

                            # WATCHDOG: Reset execution if it takes too long (> 15s)
                            if self._executing and (
                                time.time() - self._execution_start_time > 15.0
                            ):
                                logger.warning(
                                    "Watchdog: Execution stalled! Forcing reset."
                                )
                                self._executing = False
                                self.state.status_text = "WATCHDOG RESET"

                            if (
                                should_execute
                                and self.state.best_move
                                and not self._executing
                            ):
                                self._executing = True
                                self._execution_start_time = time.time()
                                bm = self.state.best_move
                                self.state.best_move = None  # CLEAR IMMEDIATELY

                                threading.Thread(
                                    target=self._run_execution_sequence,
                                    args=(
                                        bm["original_slot_indices"],
                                        bm["piece_seq"],
                                        bm["full_placements"],
                                        bm["pure_pieces"],
                                    ),
                                    daemon=True,
                                ).start()

                # --- ALWAYS REFRESH GUI ---
                if self.state.display != DisplayMode.LEAN:
                    if self.state.display == DisplayMode.STUDIO and game_state:
                        display_frame = self.renderer.draw_studio_overlay(
                            frame,
                            game_state.board_cells,
                            game_state.available_pieces,
                            self.state.best_move,
                        )
                    else:
                        # Fallback for IDLE, GAME_OVER, or WAITING
                        display_frame = self.renderer.draw_lean_overlay(
                            frame, self.state.status_text
                        )

                    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
                    self.state.gui_texture_buffer = (
                        rgb.reshape(-1) * (1.0 / 255.0)
                    ).astype(np.float32)

            except Exception as e:
                logger.error(f"LOOP ERROR: {e}")
                self.state.status_text = f"ERROR: {str(e)[:15]}"
                time.sleep(0.5)

            # Small yield
            time.sleep(0.005)

    def _run_execution_sequence(self, slot_indices, piece_seq, placements, pieces):
        """Executes a 3-piece turn sequence in a background thread."""
        try:
            self.state.status_text = "EXECUTING DROPS..."
            for move_idx, placement in enumerate(placements):
                if not self._running:
                    break

                slot_index = slot_indices[piece_seq[move_idx]]

                # Pre-drop delay
                time.sleep(self.state.pre_drop_ms / 1000.0)

                # Execute drag
                self.executor.execute_move(
                    slot_index=slot_index,
                    target_row=placement["target_row"],
                    target_col=placement["target_col"],
                    piece_cells=pieces[piece_seq[move_idx]],
                    duration_ms=self.config.input.drag_duration_ms,
                )

                # Post-drop delay (allow animation to settle)
                time.sleep(self.state.post_drop_ms / 1000.0)

            # Post-turn buffer
            if self.state.post_turn_ms > 0:
                self.state.status_text = "COOLING DOWN..."
                time.sleep(self.state.post_turn_ms / 1000.0)

        finally:
            self._executing = False
            self.state.status_text = "FINISHED TURN"

    def run(self):
        """Entry point for the application."""
        self.ui.setup()
        self.ui.run(loop_check_fn=lambda: self._running)
        self._running = False


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = BotOrchestrator()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
