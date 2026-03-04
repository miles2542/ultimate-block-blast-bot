"""
Block Blast Bot v2.0 - Frame Capture

Captures frames from scrcpy window using mss.
Handles DPI awareness and window detection.
"""

import logging
import time
from typing import Optional, Tuple

import win32gui
import win32ui

import ctypes

import numpy as np
from numpy.typing import NDArray

from block_blast_bot.config import CaptureConfig
from block_blast_bot.game_types import WindowInfo
from block_blast_bot.capture.window_detector import (
    WindowDetector,
    set_dpi_awareness,
    is_dpi_aware,
)

logger = logging.getLogger(__name__)


class FrameCapture:
    """
    Frame capture from scrcpy window.

    Uses native Windows API (PrintWindow) for capture.
    Successfully captures window even if it's occluded or on another virtual desktop.
    """

    def __init__(
        self, config: Optional[CaptureConfig] = None, title_pattern: str = "scrcpy"
    ) -> None:
        """
        Initialize frame capture.

        Args:
            config: Capture configuration. Uses defaults if None.
            title_pattern: Window title pattern to capture.
        """
        # Ensure DPI awareness is set before any operations
        set_dpi_awareness()

        self._config = config or CaptureConfig()
        self._detector = WindowDetector(title_pattern)
        self._window_info: Optional[WindowInfo] = None
        self._capture_region: Optional[dict] = None

        # Win32 capture resources
        self._hwnd: int = 0
        self._hwndDC = None
        self._mfcDC = None
        self._saveDC = None
        self._saveBitMap = None

        self._frame_interval = 1.0 / self._config.fps
        self._last_capture_time = 0.0
        self._is_initialized = False

        # Phase 5.9: Resilience features
        self._use_bitblt = False
        self._black_frame_count = 0
        self._last_toggle_time = 0.0

    def initialize(self) -> bool:
        """
        Initialize the capture system.

        Finds the scrcpy window and prepares for capture.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Find the scrcpy window
            self._window_info = self._detector.find_scrcpy_window()
            if not self._window_info or not self._window_info.hwnd:
                logger.error(
                    f"Could not find window matching: {self._detector._title_pattern}"
                )
                return False

            self._hwnd = self._window_info.hwnd

            # Get client rect for capture dimensions
            client_rect = self._detector.get_client_rect()
            if client_rect:
                self._capture_region = {
                    "width": client_rect.width,
                    "height": client_rect.height,
                }
            else:
                self._capture_region = {
                    "width": self._window_info.width,
                    "height": self._window_info.height,
                }

            # Pre-allocate DCs for performance
            self._setup_win32_resources()

            self._is_initialized = True
            logger.info(
                f"Frame capture initialized: {self._capture_region['width']}x{self._capture_region['height']} "
                f"(HWND: {self._hwnd})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize frame capture: {e}")
            return False

    def _setup_win32_resources(self) -> None:
        """Create and cache the Win32 DCs and Bitmaps for rapid capture."""
        self._release_win32_resources()

        width = self._capture_region["width"]
        height = self._capture_region["height"]

        self._hwndDC = win32gui.GetWindowDC(self._hwnd)
        self._mfcDC = win32ui.CreateDCFromHandle(self._hwndDC)
        self._saveDC = self._mfcDC.CreateCompatibleDC()

        self._saveBitMap = win32ui.CreateBitmap()
        self._saveBitMap.CreateCompatibleBitmap(self._mfcDC, width, height)
        self._saveDC.SelectObject(self._saveBitMap)

    def _release_win32_resources(self) -> None:
        """Properly delete all GDI objects to prevent memory leaks."""
        try:
            if self._saveBitMap is not None:
                win32gui.DeleteObject(self._saveBitMap.GetHandle())
                self._saveBitMap = None
            if self._saveDC is not None:
                self._saveDC.DeleteDC()
                self._saveDC = None
            if self._mfcDC is not None:
                self._mfcDC.DeleteDC()
                self._mfcDC = None
            if self._hwndDC is not None and self._hwnd != 0:
                win32gui.ReleaseDC(self._hwnd, self._hwndDC)
                self._hwndDC = None
        except Exception as e:
            logger.warning(f"Error releasing Win32 resources: {e}")

    def capture(self) -> NDArray[np.uint8]:
        """
        Capture a single frame from the scrcpy window.
        Bypasses occlusion using PrintWindow(PW_RENDERFULLCONTENT).

        Returns:
            BGR frame as numpy array with shape (height, width, 3)

        Raises:
            RuntimeError: If capture fails or window not found
        """
        if not self._is_initialized:
            if not self.initialize():
                raise RuntimeError("Frame capture not initialized - window not found")

        # Check if window is still valid
        if not self._detector.is_window_visible() or not win32gui.IsWindow(self._hwnd):
            # Try to re-find the window
            self._window_info = self._detector.find_scrcpy_window()
            if not self._window_info or not self._window_info.hwnd:
                raise RuntimeError("scrcpy window not visible or not found")

            # Update capture region and resources
            self._hwnd = self._window_info.hwnd
            client_rect = self._detector.get_client_rect()
            if client_rect:
                self._capture_region = {
                    "width": client_rect.width,
                    "height": client_rect.height,
                }
                self._setup_win32_resources()

        # FPS limiting
        elapsed = time.perf_counter() - self._last_capture_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        frame = None
        last_err = None

        # Try preferred method
        for _ in range(3):
            try:
                if self._use_bitblt:
                    frame = self._capture_bitblt()
                else:
                    frame = self._capture_printwindow()

                if frame is not None:
                    # Check for black frame
                    if np.mean(frame) < 1.0:
                        self._black_frame_count += 1
                        # If we get too many black frames, toggle method
                        if self._black_frame_count >= 10 and (
                            time.time() - self._last_toggle_time > 5.0
                        ):
                            self._use_bitblt = not self._use_bitblt
                            self._black_frame_count = 0
                            self._last_toggle_time = time.time()
                            method_name = (
                                "BitBlt" if self._use_bitblt else "PrintWindow"
                            )
                            logger.warning(
                                f"Detection: Persistent black frames! Toggling to {method_name}"
                            )
                            self._setup_win32_resources()
                    else:
                        self._black_frame_count = 0

                    self._last_capture_time = time.perf_counter()
                    return frame
            except Exception as e:
                last_err = e
                time.sleep(0.01)

        # Final retry with fallback if not already used
        try:
            if frame is None:
                # Force swap for one last try
                self._use_bitblt = not self._use_bitblt
                self._setup_win32_resources()
                if self._use_bitblt:
                    frame = self._capture_bitblt()
                else:
                    frame = self._capture_printwindow()

                if frame is not None:
                    self._last_capture_time = time.perf_counter()
                    return frame
        except Exception:
            pass

        # If we exhausted retries:
        logger.error(f"Frame capture failed after retries: {last_err}")
        raise RuntimeError(f"Frame capture failed: {last_err}") from last_err

    def _capture_printwindow(self) -> Optional[NDArray[np.uint8]]:
        """Implementation using PrintWindow API (Occlusion-safe)."""
        # PW_RENDERFULLCONTENT = 2
        result = ctypes.windll.user32.PrintWindow(
            self._hwnd, self._saveDC.GetSafeHdc(), 2
        )
        if result != 1:
            return None

        bmpstr = self._saveBitMap.GetBitmapBits(True)
        frame = np.frombuffer(bmpstr, dtype=np.uint8)
        bmpinfo = self._saveBitMap.GetInfo()
        frame.shape = (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        return np.ascontiguousarray(frame[:, :, :3])

    def _capture_bitblt(self) -> Optional[NDArray[np.uint8]]:
        """Implementation using BitBlt (Fast, direct pixel copy)."""
        # SRCCOPY = 0x00CC0020
        self._saveDC.BitBlt(
            (0, 0),
            (self._capture_region["width"], self._capture_region["height"]),
            self._mfcDC,
            (0, 0),
            0x00CC0020,
        )
        bmpstr = self._saveBitMap.GetBitmapBits(True)
        frame = np.frombuffer(bmpstr, dtype=np.uint8)
        bmpinfo = self._saveBitMap.GetInfo()
        frame.shape = (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        return np.ascontiguousarray(frame[:, :, :3])

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current capture resolution.

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            RuntimeError: If not initialized
        """
        if not self._capture_region:
            raise RuntimeError("Frame capture not initialized")

        return (self._capture_region["width"], self._capture_region["height"])

    def get_window_info(self) -> Optional[WindowInfo]:
        """
        Get information about the detected window.

        Returns:
            WindowInfo if window was found, None otherwise
        """
        return self._window_info

    def update_window_position(self) -> bool:
        """
        Update the window position (call if window was moved).

        Returns:
            True if window was found and position updated, False otherwise
        """
        self._window_info = self._detector.find_scrcpy_window()
        if self._window_info and self._window_info.hwnd:
            self._hwnd = self._window_info.hwnd
            client_rect = self._detector.get_client_rect()
            if client_rect:
                self._capture_region = {
                    "width": client_rect.width,
                    "height": client_rect.height,
                }
                self._setup_win32_resources()
            return True
        return False

    def verify_dpi_awareness(self) -> bool:
        """
        Verify that DPI awareness is correctly set.

        Returns:
            True if DPI awareness is set, False otherwise
        """
        return is_dpi_aware()

    def close(self) -> None:
        """Release resources."""
        self._release_win32_resources()
        self._is_initialized = False
        logger.debug("Frame capture closed")

    def __enter__(self) -> "FrameCapture":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def create_capture(
    config: Optional[CaptureConfig] = None, title_pattern: str = "scrcpy"
) -> FrameCapture:
    """
    Factory function to create a FrameCapture instance.

    Args:
        config: Capture configuration.
        title_pattern: Window title pattern to capture.

    Returns:
        A new FrameCapture instance.
    """
    return FrameCapture(config, title_pattern)


__all__ = ["FrameCapture", "create_capture"]
