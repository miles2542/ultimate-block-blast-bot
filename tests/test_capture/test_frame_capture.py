"""
Tests for frame capture module.

Tests DPI awareness, window detection, and frame capture functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from block_blast_bot.capture import (
    FrameCapture,
    WindowDetector,
    set_dpi_awareness,
    is_dpi_aware,
)
from block_blast_bot.config import CaptureConfig
from block_blast_bot.game_types import WindowInfo


class TestDPIAwareness:
    """Tests for DPI awareness functionality."""

    def test_set_dpi_awareness_returns_bool(self):
        """Test that set_dpi_awareness returns a boolean."""
        result = set_dpi_awareness()
        assert isinstance(result, bool)

    def test_dpi_awareness_is_set_after_call(self):
        """Test that is_dpi_aware returns True after setting."""
        set_dpi_awareness()
        assert is_dpi_aware() is True

    def test_dpi_awareness_idempotent(self):
        """Test that calling set_dpi_awareness multiple times is safe."""
        result1 = set_dpi_awareness()
        result2 = set_dpi_awareness()
        result3 = set_dpi_awareness()
        # All should succeed (idempotent)
        assert is_dpi_aware() is True


class TestWindowDetector:
    """Tests for window detection functionality."""

    def test_init_sets_dpi_awareness(self):
        """Test that WindowDetector initialization sets DPI awareness."""
        detector = WindowDetector()
        assert is_dpi_aware() is True

    def test_default_title_pattern(self):
        """Test default title pattern is 'scrcpy'."""
        detector = WindowDetector()
        assert detector._title_pattern == "scrcpy"

    def test_custom_title_pattern(self):
        """Test custom title pattern."""
        detector = WindowDetector(title_pattern="my_custom_pattern")
        assert detector._title_pattern == "my_custom_pattern"

    def test_find_scrcpy_window_returns_window_info_or_none(self):
        """Test that find_scrcpy_window returns WindowInfo or None."""
        detector = WindowDetector()
        result = detector.find_scrcpy_window()
        # Result is either WindowInfo or None (depends on whether scrcpy is running)
        assert result is None or isinstance(result, WindowInfo)

    def test_get_window_rect_returns_rect_or_none(self):
        """Test that get_window_rect returns Rect or None."""
        detector = WindowDetector()
        result = detector.get_window_rect()
        # Result is either Rect or None
        from block_blast_bot.game_types import Rect

        assert result is None or isinstance(result, Rect)

    def test_is_window_visible_returns_false_without_window(self):
        """Test that is_window_visible returns False when no window cached."""
        detector = WindowDetector()
        # Without finding a window first
        detector._cached_info = None
        assert detector.is_window_visible() is False


class TestCaptureConfig:
    """Tests for CaptureConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CaptureConfig()
        assert config.method == "mss"
        assert config.fps == 10
        assert config.timeout_ms == 3000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CaptureConfig(
            method="d3d",
            fps=30,
            timeout_ms=5000,
        )
        assert config.method == "d3d"
        assert config.fps == 30
        assert config.timeout_ms == 5000


class TestWindowInfo:
    """Tests for WindowInfo dataclass."""

    def test_window_info_creation(self):
        """Test creating WindowInfo instance."""
        info = WindowInfo(
            title="Test Window",
            left=100,
            top=200,
            width=800,
            height=600,
            hwnd=12345,
        )
        assert info.title == "Test Window"
        assert info.left == 100
        assert info.top == 200
        assert info.width == 800
        assert info.height == 600
        assert info.hwnd == 12345


class TestFrameCapture:
    """Tests for FrameCapture class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        capture = FrameCapture()
        assert capture._config.method == "mss"
        assert capture._config.fps == 10

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = CaptureConfig(fps=30)
        capture = FrameCapture(config, title_pattern="custom")
        assert capture._config.fps == 30
        assert capture._detector._title_pattern == "custom"

    def test_initialize_without_window_raises(self):
        """Test that initialize returns False when window not found."""
        config = CaptureConfig()
        capture = FrameCapture(config, title_pattern="nonexistent_window_xyz123")
        result = capture.initialize()
        assert result is False

    def test_capture_without_initialize_raises(self):
        """Test that capture raises RuntimeError when not initialized."""
        config = CaptureConfig()
        capture = FrameCapture(config, title_pattern="nonexistent_window_xyz123")
        with pytest.raises(RuntimeError, match="not initialized"):
            capture.capture()

    def test_get_resolution_without_initialize_raises(self):
        """Test that get_resolution raises RuntimeError when not initialized."""
        capture = FrameCapture()
        with pytest.raises(RuntimeError, match="not initialized"):
            capture.get_resolution()

    def test_close_is_safe(self):
        """Test that close() can be called safely."""
        capture = FrameCapture()
        capture.close()  # Should not raise

    def test_context_manager(self):
        """Test context manager usage."""
        config = CaptureConfig()
        with FrameCapture(config, title_pattern="nonexistent_window_xyz123") as capture:
            # Initialize will fail but context manager should work
            pass

    def test_verify_dpi_awareness(self):
        """Test DPI awareness verification."""
        capture = FrameCapture()
        assert capture.verify_dpi_awareness() is True


class TestFrameCaptureIntegration:
    """Integration tests that require scrcpy window.

    These tests are skipped if scrcpy is not running.
    Run with: pytest -m integration tests/test_capture/
    """

    @pytest.mark.integration
    def test_find_real_scrcpy_window(self):
        """Test finding real scrcpy window (requires scrcpy running)."""
        detector = WindowDetector()
        result = detector.find_scrcpy_window()
        if result is None:
            pytest.skip("scrcpy window not found - start scrcpy to run this test")

        assert isinstance(result, WindowInfo)
        assert "scrcpy" in result.title.lower()
        assert result.width > 0
        assert result.height > 0

    @pytest.mark.integration
    def test_capture_real_frame(self):
        """Test capturing real frame (requires scrcpy running)."""
        capture = FrameCapture()
        if not capture.initialize():
            pytest.skip("scrcpy window not found - start scrcpy to run this test")

        try:
            frame = capture.capture()

            # Verify frame properties
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # BGR
            assert frame.dtype == np.uint8

            # Verify resolution matches
            width, height = capture.get_resolution()
            assert frame.shape[1] == width
            assert frame.shape[0] == height

        finally:
            capture.close()

    @pytest.mark.integration
    def test_multiple_captures(self):
        """Test multiple consecutive captures (requires scrcpy running)."""
        capture = FrameCapture()
        if not capture.initialize():
            pytest.skip("scrcpy window not found - start scrcpy to run this test")

        try:
            frames = []
            for _ in range(5):
                frame = capture.capture()
                frames.append(frame)

            # All frames should have same shape
            shapes = [f.shape for f in frames]
            assert all(s == shapes[0] for s in shapes)

        finally:
            capture.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
