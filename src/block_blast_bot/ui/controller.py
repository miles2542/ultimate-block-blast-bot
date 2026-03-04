"""
Block Blast Bot v2.0 - UI Controller

Manages the Dear PyGui lifecycle, texture updates, and user interactions.
"""

import logging
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Callable
import time

from block_blast_bot.game_types import DisplayMode, AppMode, SolverMode

logger = logging.getLogger(__name__)


class UIController:
    """
    Handles all Dear PyGui setup and event loops.
    Communicates with BotOrchestrator through the SharedState and callbacks.
    """

    def __init__(
        self,
        state,
        capture_res_provider: Callable,
        refresh_cb: Callable,
        semi_auto_cb: Callable,
    ):
        self.state = state
        self.get_capture_res = capture_res_provider
        self.on_refresh = refresh_cb
        self.on_semi_auto = semi_auto_cb
        self._running = True

    def _cb_mode(self, s, a):
        self.state.mode = a

    def _cb_display(self, s, a):
        self.state.display = a

    def _cb_solver_mode(self, s, a):
        self.state.solver_mode = a
        if a == SolverMode.CUSTOM:
            dpg.show_item("custom_weights_group")
        else:
            dpg.hide_item("custom_weights_group")

    def _cb_resize(self, s, a):
        vw, vh = a[0], a[1]
        tex_width, tex_height = self.get_capture_res()

        avail_w = max(100, vw - 390)
        aspect = tex_height / tex_width
        new_w, new_h = avail_w, int(avail_w * aspect)

        if vh > 100 and new_h > vh - 60:
            new_h = vh - 60
            new_w = int(new_h / aspect)

        self.state.view_width, self.state.view_height = new_w, new_h

        if dpg.does_item_exist("video_img"):
            dpg.configure_item("video_img", width=new_w, height=new_h)
        if dpg.does_item_exist("view_panel"):
            dpg.configure_item("view_panel", width=new_w + 20)

    def setup(self):
        """Initial DPG context and window creation."""
        dpg.create_context()

        tex_width, tex_height = self.get_capture_res()
        blank_data = np.zeros((tex_height, tex_width, 4), dtype=np.float32)

        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                tex_width, tex_height, blank_data.reshape(-1), tag="scrcpy_feed"
            )

        with dpg.window(tag="Main Window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(
                    width=self.state.view_width + 20, tag="view_panel"
                ):
                    dpg.add_image(
                        "scrcpy_feed",
                        tag="video_img",
                        width=self.state.view_width,
                        height=self.state.view_height,
                    )

                with dpg.child_window(width=350):
                    dpg.add_text("Block Blast Bot v2.0", color=[100, 200, 255])
                    with dpg.group(horizontal=True):
                        dpg.add_text("FPS: 0.0", tag="text_fps")
                        dpg.add_text("|")
                        dpg.add_text("HB: 0", tag="text_hb")
                    dpg.add_text(
                        "Status: INITIALIZING...",
                        tag="text_status",
                        color=[200, 200, 200],
                    )
                    dpg.add_separator()

                    dpg.add_combo(
                        [AppMode.MANUAL, AppMode.SEMI_AUTO, AppMode.FULL_AUTO],
                        default_value=self.state.mode,
                        callback=self._cb_mode,
                    )
                    dpg.add_button(
                        label="EXECUTE OPTIMAL TURN (Semi-Auto)",
                        callback=self.on_semi_auto,
                    )
                    dpg.add_button(
                        label="FORCE REFRESH (SYNC)",
                        callback=self.on_refresh,
                    )
                    dpg.add_separator()

                    dpg.add_combo(
                        [DisplayMode.LEAN, DisplayMode.STUDIO],
                        default_value=self.state.display,
                        callback=self._cb_display,
                    )
                    dpg.add_separator()

                    dpg.add_combo(
                        [SolverMode.SAFE, SolverMode.AGGRESSIVE, SolverMode.CUSTOM],
                        default_value=self.state.solver_mode,
                        callback=self._cb_solver_mode,
                    )

                    with dpg.group(
                        tag="custom_weights_group",
                        show=(self.state.solver_mode == SolverMode.CUSTOM),
                    ):
                        # Slider helpers
                        def add_weight_slider(label, attr, min_v, max_v):
                            dpg.add_slider_float(
                                label=label,
                                default_value=getattr(self.state, attr),
                                min_value=min_v,
                                max_value=max_v,
                                callback=lambda s, a: setattr(self.state, attr, a),
                            )

                        add_weight_slider(
                            "Hole Penalty", "custom_hole_penalty", -200.0, 0.0
                        )
                        add_weight_slider(
                            "Density Penalty", "custom_density_penalty", -20.0, 0.0
                        )
                        add_weight_slider(
                            "Score Reward", "custom_score_reward", 0.0, 20.0
                        )
                        add_weight_slider(
                            "Combo Reward",
                            "custom_combo_preservation_reward",
                            0.0,
                            100.0,
                        )
                        add_weight_slider(
                            "Bumpiness Penalty", "custom_bumpiness_penalty", -10.0, 0.0
                        )
                        add_weight_slider(
                            "Well Penalty", "custom_well_depth_penalty", -50.0, 0.0
                        )

                    dpg.add_slider_int(
                        label="Lookahead Depth",
                        default_value=self.state.lookahead_depth,
                        min_value=0,
                        max_value=3,
                        callback=lambda s, a: setattr(
                            self.state, "lookahead_depth", int(a)
                        ),
                    )
                    dpg.add_slider_int(
                        label="Lookahead Samples",
                        default_value=self.state.lookahead_samples,
                        min_value=1,
                        max_value=100,
                        callback=lambda s, a: setattr(
                            self.state, "lookahead_samples", int(a)
                        ),
                    )
                    dpg.add_slider_int(
                        label="Search Time (ms)",
                        default_value=self.state.time_limit_ms,
                        min_value=100,
                        max_value=5000,
                        callback=lambda s, a: setattr(
                            self.state, "time_limit_ms", int(a)
                        ),
                    )
                    dpg.add_separator()

                    def add_delay_slider(label, attr, max_v):
                        dpg.add_slider_int(
                            label=label,
                            default_value=getattr(self.state, attr),
                            min_value=0,
                            max_value=max_v,
                            callback=lambda s, a: setattr(self.state, attr, int(a)),
                        )

                    add_delay_slider("Pre-Drop Delay", "pre_drop_ms", 1000)
                    add_delay_slider("Post-Drop Delay", "post_drop_ms", 1500)
                    add_delay_slider("Post-Turn Delay", "post_turn_ms", 3000)

        dpg.create_viewport(title="Block Blast v2.0 AI", width=850, height=1050)
        dpg.set_viewport_resize_callback(self._cb_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main Window", True)

    def run(self, loop_check_fn: Callable):
        """Standard DPG render loop."""
        while dpg.is_dearpygui_running() and loop_check_fn():
            dpg.set_value("text_fps", f"Engine Loop FPS: {self.state.fps:.1f}")
            dpg.set_value("text_hb", f"HB: {self.state.loop_heartbeat}")
            dpg.set_value("text_status", self.state.status_text)

            if self.state.display == DisplayMode.LEAN:
                if dpg.is_item_shown("view_panel"):
                    dpg.hide_item("view_panel")
            else:
                if not dpg.is_item_shown("view_panel"):
                    dpg.show_item("view_panel")
                if self.state.gui_texture_buffer is not None:
                    try:
                        dpg.set_value("scrcpy_feed", self.state.gui_texture_buffer)
                    except Exception:
                        pass

            dpg.render_dearpygui_frame()
            time.sleep(0.005)

        self._running = False
        dpg.destroy_context()

    def is_running(self) -> bool:
        return self._running
