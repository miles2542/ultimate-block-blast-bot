"""
Block Blast Bot v2.0 - Window Detector

Detects and tracks scrcpy window position and size.
Handles Windows DPI awareness for correct coordinate mapping.
"""

import ctypes
import logging
import re
from typing import Optional

import win32gui

from block_blast_bot.game_types import Rect, WindowInfo

logger = logging.getLogger(__name__)

# DPI awareness constants
PROCESS_DPI_UNAWARE = 0
PROCESS_SYSTEM_DPI_AWARE = 1
PROCESS_PER_MONITOR_DPI_AWARE = 2

# Track if DPI awareness has been set
_dpi_awareness_set = False


def set_dpi_awareness() -> bool:
    """
    Set process DPI awareness to per-monitor aware.

    This MUST be called before any window operations to ensure
    correct coordinate mapping on high-DPI displays.

    Returns:
        True if DPI awareness was set successfully, False otherwise
    """
    global _dpi_awareness_set

    if _dpi_awareness_set:
        return True

    try:
        # Try to set per-monitor DPI awareness (Windows 8.1+)
        result = ctypes.windll.shcore.SetProcessDpiAwareness(
            PROCESS_PER_MONITOR_DPI_AWARE
        )
        if result == 0:  # S_OK
            _dpi_awareness_set = True
            logger.debug("DPI awareness set to PROCESS_PER_MONITOR_DPI_AWARE")
            return True
        else:
            logger.warning(f"SetProcessDpiAwareness returned {result}")
    except AttributeError:
        # shcore not available (older Windows version)
        logger.warning("shcore not available, trying SetProcessDPIAware")
        try:
            result = ctypes.windll.user32.SetProcessDPIAware()
            if result:
                _dpi_awareness_set = True
                logger.debug("DPI awareness set via SetProcessDPIAware")
                return True
        except Exception as e:
            logger.error(f"Failed to set DPI awareness: {e}")

    return _dpi_awareness_set


def is_dpi_aware() -> bool:
    """Check if DPI awareness has been set."""
    return _dpi_awareness_set


class WindowDetector:
    """
    Detects scrcpy window on Windows using Win32 API.

    Handles DPI scaling to ensure correct coordinate mapping.
    """

    def __init__(self, title_pattern: str = "scrcpy"):
        """
        Initialize window detector.

        Args:
            title_pattern: Regex pattern to match window title (default: "scrcpy")
        """
        # Ensure DPI awareness is set before any window operations
        set_dpi_awareness()

        self._title_pattern = title_pattern
        self._cached_info: Optional[WindowInfo] = None

    def find_scrcpy_window(self) -> Optional[WindowInfo]:
        """
        Find the scrcpy window and return its information.

        Returns:
            WindowInfo with window details, or None if not found
        """
        try:
            # Fast-path: Try to find window exactly by title first
            # Many scrcpy windows are named exactly "scrcpy"
            hwnd = win32gui.FindWindow(None, self._title_pattern)
            if not hwnd:
                # Try common variations if pattern is just "scrcpy"
                if self._title_pattern.lower() == "scrcpy":
                    hwnd = win32gui.FindWindow(None, "scrcpy")

            if hwnd and win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if re.search(self._title_pattern, title, re.IGNORECASE):
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    result = WindowInfo(
                        title=title,
                        left=left,
                        top=top,
                        width=right - left,
                        height=bottom - top,
                        hwnd=hwnd,
                    )
                    logger.debug(f"Found window via FindWindow fast-path: {title}")
                    return result
        except Exception as e:
            logger.debug(f"FindWindow fast-path failed: {e}")

        def enum_callback(hwnd: int, _) -> bool:
            nonlocal result

            try:
                # Skip invisible windows
                if not win32gui.IsWindowVisible(hwnd):
                    return True

                # Get window title
                title = win32gui.GetWindowText(hwnd)
                if not title:
                    return True

                # Check if title matches pattern
                if re.search(self._title_pattern, title, re.IGNORECASE):
                    # Get window rect (includes non-client area)
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top

                    result = WindowInfo(
                        title=title,
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                        hwnd=hwnd,
                    )
                    return False  # Stop enumeration
            except Exception:
                # Continue enumeration on individual window errors
                pass

            return True  # Continue enumeration

        try:
            logger.debug(f"Starting EnumWindows with pattern: {self._title_pattern}")
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            # We already tried FindWindow, so just log this error
            logger.error(f"Error enumerating windows: {e}")

        if result:
            self._cached_info = result
            logger.debug(
                f"Found window: {result.title} at ({result.left}, {result.top}) "
                f"size {result.width}x{result.height}"
            )
        else:
            logger.warning(f"No window found matching pattern: {self._title_pattern}")

        return result

    def get_window_rect(self, title_pattern: Optional[str] = None) -> Optional[Rect]:
        """
        Find window and return its bounds as a Rect.

        Args:
            title_pattern: Optional pattern override (uses instance pattern if None)

        Returns:
            Rect with window bounds, or None if not found
        """
        if title_pattern:
            old_pattern = self._title_pattern
            self._title_pattern = title_pattern
            info = self.find_scrcpy_window()
            self._title_pattern = old_pattern
        else:
            info = self.find_scrcpy_window()

        if info:
            return Rect(
                x=info.left,
                y=info.top,
                width=info.width,
                height=info.height,
            )
        return None

    def is_window_visible(self) -> bool:
        """
        Check if the cached scrcpy window is visible and not minimized.

        Returns:
            True if window is visible, False otherwise
        """
        if not self._cached_info:
            return False

        try:
            hwnd = self._cached_info.hwnd
            if not win32gui.IsWindow(hwnd):
                return False
            if not win32gui.IsWindowVisible(hwnd):
                return False
            # Check if minimized (iconic)
            if win32gui.IsIconic(hwnd):
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking window visibility: {e}")
            return False

    def get_client_rect(self) -> Optional[Rect]:
        """
        Get the client area rect of the cached window.

        The client area excludes window decorations (title bar, borders).

        Returns:
            Rect with client area bounds in screen coordinates, or None
        """
        if not self._cached_info:
            return None

        try:
            hwnd = self._cached_info.hwnd
            # Get client rect (relative to window)
            client_rect = win32gui.GetClientRect(hwnd)
            left, top, right, bottom = client_rect

            # Convert to screen coordinates
            point_left_top = win32gui.ClientToScreen(hwnd, (left, top))
            point_right_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

            return Rect(
                x=point_left_top[0],
                y=point_left_top[1],
                width=point_right_bottom[0] - point_left_top[0],
                height=point_right_bottom[1] - point_left_top[1],
            )
        except Exception as e:
            logger.error(f"Error getting client rect: {e}")
            return None

    def verify_dpi_match(self, captured_width: int, captured_height: int) -> bool:
        """
        Verify that captured dimensions match reported window dimensions.

        This detects DPI scaling issues where Windows reports scaled
        coordinates but capture gets actual pixels.

        Args:
            captured_width: Width of captured frame
            captured_height: Height of captured frame

        Returns:
            True if dimensions match, False if there's a mismatch
        """
        if not self._cached_info:
            logger.warning("No cached window info for DPI verification")
            return False

        # Get client rect for comparison (capture is usually client area)
        client_rect = self.get_client_rect()
        if not client_rect:
            return False

        if captured_width != client_rect.width or captured_height != client_rect.height:
            logger.warning(
                f"DPI mismatch detected: captured {captured_width}x{captured_height}, "
                f"expected {client_rect.width}x{client_rect.height}"
            )
            return False

        return True


__all__ = ["WindowDetector", "WindowInfo", "set_dpi_awareness", "is_dpi_aware"]
