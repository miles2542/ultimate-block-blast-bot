"""
Block Blast Bot v2.0 - ADB Controller

Controls Android device via ADB shell commands.
"""

from typing import Optional


import subprocess


class ADBController:
    """
    ADB controller for Android device input.

    Executes gestures via ADB shell input commands.
    """

    def __init__(self, device_id: Optional[str] = None) -> None:
        """
        Initialize ADB controller.

        Args:
            device_id: Optional specific device ID for multi-device scenarios
        """
        self._device_id = device_id
        self._connected = True

    def _build_cmd(self, shell_cmd: str) -> list[str]:
        base = ["adb"]
        if self._device_id:
            base.extend(["-s", self._device_id])
        base.append("shell")
        base.extend(shell_cmd.split())
        return base

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        """
        Executes a screen swipe. Also used for tap-and-hold (where x1,y1 == x2,y2).
        Returns True if successful, False if the command failed.
        """
        cmd = self._build_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
        try:
            # We don't want to block indefinitely if ADB hangs
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("ADB swipe command timed out.")
            return False
        except Exception as e:
            print(f"Error executing ADB swipe: {e}")
            return False

    def tap(self, x: int, y: int) -> bool:
        """
        Executes a quick tap.
        """
        cmd = self._build_cmd(f"input tap {x} {y}")
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            print(f"Error executing ADB tap: {e}")
            return False


__all__ = ["ADBController"]
