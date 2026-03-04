"""
Block Blast Bot v2.0 - Board Detector Tests

Tests for the board detection system.
"""

import numpy as np
import pytest

from block_blast_bot.calibration.data import (
    CalibrationData,
    DetectionCalibration,
    GridCalibration,
    SlotCalibration,
)
from block_blast_bot.game_types import BoardDetectionResult, DetectionResult
from block_blast_bot.vision import (
    BoardDetector,
    DetectionCascade,
    LABDistanceDetector,
    EdgeBasedDetector,
)


def create_test_calibration() -> CalibrationData:
    """Create a test calibration with known grid bounds."""
    # Create a 400x400 grid at position (100, 100) with 50x50 cells
    grid = GridCalibration(
        bounds=(100, 100, 400, 400),
        cell_size=(50, 50),
    )

    # Create 3 empty slots
    slots = [
        SlotCalibration(
            center=(200, 550),
            bounds=(150, 500, 100, 100),
        ),
        SlotCalibration(
            center=(300, 550),
            bounds=(250, 500, 100, 100),
        ),
        SlotCalibration(
            center=(400, 550),
            bounds=(350, 500, 100, 100),
        ),
    ]

    return CalibrationData(
        version="2.0",
        device_resolution=(1080, 1920),
        window_resolution=(540, 960),
        grid=grid,
        slots=slots,
        detection=DetectionCalibration(),
    )


def create_test_frame() -> np.ndarray:
    """Create a test frame with a simulated board."""
    # Create a 540x960 BGR frame (simulating scrcpy window)
    frame = np.zeros((960, 540, 3), dtype=np.uint8)

    # Fill with a dark gray background
    frame[:, :] = (30, 30, 30)

    # Draw the grid area with a slightly different color
    frame[100:500, 100:500] = (40, 40, 40)

    # Draw some filled cells (simulating blocks)
    # Cell 0,0 at position (125, 125)
    frame[125:175, 125:175] = (100, 150, 200)  # Blue-ish block
    # Cell 2,3 at position (275, 225)
    frame[225:275, 275:325] = (150, 100, 200)  # Purple-ish block

    return frame


class TestLABDistanceDetector:
    """Tests for LABDistanceDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized with default parameters."""
        detector = LABDistanceDetector()
        assert detector is not None
        assert detector._variance_threshold == 15.0
        assert detector._hard_threshold == 24

    def test_detector_with_custom_threshold(self):
        """Test detector can be initialized with custom threshold."""
        detector = LABDistanceDetector(variance_threshold=10.0, hard_threshold=30)
        assert detector._variance_threshold == 10.0
        assert detector._hard_threshold == 30

    def test_detect_board_returns_detection_result(self):
        """Test detect_board returns a DetectionResult."""
        detector = LABDistanceDetector()
        calibration = create_test_calibration()
        frame = create_test_frame()

        result = detector.detect_board(frame, calibration)

        assert isinstance(result, DetectionResult)
        assert result.method in ["variance_lab", "lab_hard"]
        assert 0.0 <= result.confidence <= 1.0
        assert 0 <= result.board <= 0xFFFFFFFFFFFFFFFF  # u64

    def test_detect_board_with_missing_grid(self):
        """Test detect_board raises error with missing grid calibration."""
        detector = LABDistanceDetector()
        calibration = CalibrationData()
        frame = create_test_frame()

        with pytest.raises(ValueError, match="Calibration missing grid"):
            detector.detect_board(frame, calibration)


class TestEdgeBasedDetector:
    """Tests for EdgeBasedDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized with default parameters."""
        detector = EdgeBasedDetector()
        assert detector is not None
        assert detector._canny_low == 50
        assert detector._canny_high == 150

    def test_detect_board_returns_detection_result(self):
        """Test detect_board returns a DetectionResult."""
        detector = EdgeBasedDetector()
        calibration = create_test_calibration()
        frame = create_test_frame()

        result = detector.detect_board(frame, calibration)

        assert isinstance(result, DetectionResult)
        assert result.method == "edge_based"
        assert 0.0 <= result.confidence <= 0.7  # Edge-based has max 0.7
        assert 0 <= result.board <= 0xFFFFFFFFFFFFFFFF


class TestDetectionCascade:
    """Tests for DetectionCascade class."""

    def test_cascade_initialization(self):
        """Test cascade can be initialized."""
        cascade = DetectionCascade()
        assert cascade is not None

    def test_cascade_with_custom_threshold(self):
        """Test cascade can be initialized with custom threshold."""
        cascade = DetectionCascade(lab_threshold=30)
        assert cascade is not None

    def test_cascade_detects_board(self):
        """Test cascade can detect board."""
        cascade = DetectionCascade()
        calibration = create_test_calibration()
        frame = create_test_frame()

        result = cascade.detect_board(frame, calibration)

        assert isinstance(result, DetectionResult)
        assert result.method in ["variance_lab", "lab_hard", "edge_based"]
        assert 0 <= result.board <= 0xFFFFFFFFFFFFFFFF

    def test_cascade_validates_result(self):
        """Test cascade validates results."""
        # Create a cascade with high threshold to force low confidence
        cascade = DetectionCascade(lab_threshold=100)
        calibration = create_test_calibration()
        frame = create_test_frame()

        # Should still return a result even with low confidence
        result = cascade.detect_board(frame, calibration)
        assert result is not None


class TestBoardDetector:
    """Tests for BoardDetector class."""

    def test_board_detector_initialization(self):
        """Test board detector can be initialized."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        assert detector is not None

    def test_board_detector_with_missing_grid(self):
        """Test board detector raises error with missing grid."""
        calibration = CalibrationData()
        with pytest.raises(ValueError, match="Calibration must include grid"):
            BoardDetector(calibration)

    def test_detect_returns_board_detection_result(self):
        """Test detect returns BoardDetectionResult."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        frame = create_test_frame()

        result = detector.detect(frame)

        assert isinstance(result, BoardDetectionResult)
        assert result.method in ["variance_lab", "lab_hard", "edge_based"]
        assert 0 <= result.board <= 0xFFFFFFFFFFFFFFFF
        assert isinstance(result.filled_cells, list)

    def test_detect_filled_count(self):
        """Test filled_count property."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        frame = create_test_frame()

        result = detector.detect(frame)

        assert isinstance(result.filled_count, int)
        assert 0 <= result.filled_count <= 64

    def test_detect_to_board_state(self):
        """Test to_board_state conversion."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        frame = create_test_frame()

        result = detector.detect(frame)
        board_state = result.to_board_state()

        assert board_state.grid.shape == (8, 8)
        assert board_state.grid.dtype == np.bool_


class TestDetectionResult:
    """Tests for DetectionResult class."""

    def test_filled_count_property(self):
        """Test filled_count counts bits correctly."""
        result = DetectionResult(
            board=0b00000001,  # 1 filled cell
            method="test",
            confidence=1.0,
        )
        assert result.filled_count == 1

        result = DetectionResult(
            board=0b00000011,  # 2 filled cells
            method="test",
            confidence=1.0,
        )
        assert result.filled_count == 2

    def test_is_cell_filled(self):
        """Test is_cell_filled method."""
        result = DetectionResult(
            board=0b00000001,  # Cell 0,0 filled
            method="test",
            confidence=1.0,
        )
        assert result.is_cell_filled(0, 0) is True
        assert result.is_cell_filled(0, 1) is False

    def test_is_cell_filled_out_of_bounds(self):
        """Test is_cell_filled returns False for out of bounds."""
        result = DetectionResult(
            board=0,
            method="test",
            confidence=1.0,
        )
        assert result.is_cell_filled(-1, 0) is False
        assert result.is_cell_filled(0, 8) is False
        assert result.is_cell_filled(8, 0) is False


class TestBoardDetectionResult:
    """Tests for BoardDetectionResult class."""

    def test_filled_count_property(self):
        """Test filled_count returns length of filled_cells."""
        result = BoardDetectionResult(
            board=0,
            filled_cells=[(0, 0), (1, 1)],
            method="test",
            confidence=1.0,
        )
        assert result.filled_count == 2

    def test_to_board_state(self):
        """Test to_board_state conversion."""
        result = BoardDetectionResult(
            board=0b00000011,  # Cells 0,0 and 0,1
            filled_cells=[(0, 0), (0, 1)],
            method="test",
            confidence=1.0,
        )
        board_state = result.to_board_state()

        assert board_state.grid.shape == (8, 8)
        assert board_state.grid[0, 0]
        assert board_state.grid[0, 1]
        assert not board_state.grid[1, 0]


class TestIntegration:
    """Integration tests for the full detection pipeline."""

    def test_full_pipeline(self):
        """Test the full detection pipeline end-to-end."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        frame = create_test_frame()

        # Should not raise any exceptions
        result = detector.detect(frame)

        assert result is not None
        assert result.filled_count >= 0
        assert result.confidence > 0

    def test_detect_with_debug(self):
        """Test detection with debug visualization."""
        calibration = create_test_calibration()
        detector = BoardDetector(calibration)
        frame = create_test_frame()

        result, debug_frame = detector.detect_with_debug(frame)

        assert result is not None
        assert debug_frame is not None
        assert debug_frame.shape == frame.shape
