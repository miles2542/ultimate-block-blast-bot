"""
Block Blast Bot v2.0 - Piece Detector Tests

Tests for piece detection functionality.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
import cv2

from block_blast_bot.calibration.data import (
    CalibrationData,
    DetectionCalibration,
    GridCalibration,
    SlotCalibration,
)
from block_blast_bot.game_types import DetectedPiece, PieceDetectionResult
from block_blast_bot.vision.piece_detector import (
    MIN_CELLS,
    MAX_CELLS,
    PieceDetector,
    SLOT_CELL_SCALE,
)


def create_test_frame(width: int = 400, height: int = 600) -> NDArray[np.uint8]:
    """Create a test frame with a dark background."""
    frame = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    return frame


def create_mock_calibration(
    window_width: int = 400,
    window_height: int = 600,
) -> CalibrationData:
    """Create mock calibration data for testing."""
    # Create a mock grid (8x8)
    grid = GridCalibration(
        bounds=(20, 20, 360, 360),
        cell_size=(45, 45),
    )

    # Create 3 slot positions - these are within a 600x400 frame
    # Slots are typically below the grid
    slots = [
        SlotCalibration(
            center=(100, 500),
            bounds=(50, 450, 100, 80),
        ),
        SlotCalibration(
            center=(200, 500),
            bounds=(150, 450, 100, 80),
        ),
        SlotCalibration(
            center=(300, 500),
            bounds=(250, 450, 100, 80),
        ),
    ]

    return CalibrationData(
        version="2.0",
        device_resolution=(1080, 1920),
        window_resolution=(window_width, window_height),
        grid=grid,
        slots=slots,
        detection=DetectionCalibration(),
    )


def create_piece_in_frame(
    frame: NDArray[np.uint8],
    position: tuple[int, int],
    cells: list[tuple[int, int]],
    cell_size: int,
    color: tuple[int, int, int] = (200, 100, 50),  # Orange piece
) -> None:
    """
    Draw a piece in the test frame.

    Args:
        frame: Frame to draw on
        position: Top-left position (x, y)
        cells: List of (row, col) for cells
        cell_size: Size of each cell
        color: BGR color
    """
    x, y = position
    for row, col in cells:
        cell_x = x + col * cell_size
        cell_y = y + row * cell_size
        # Draw filled rectangle with slight padding
        padding = 2
        cv2.rectangle(
            frame,
            (cell_x + padding, cell_y + padding),
            (cell_x + cell_size - padding, cell_y + cell_size - padding),
            color,
            -1,
        )


class TestPieceDetector:
    """Tests for PieceDetector class."""

    def test_initialization(self):
        """Test detector can be initialized."""
        detector = PieceDetector()
        assert detector is not None

    def test_initialization_with_calibration(self):
        """Test detector can be initialized with calibration."""
        calibration = create_mock_calibration()
        detector = PieceDetector(calibration=calibration)
        assert detector is not None

    def test_detect_requires_calibration(self):
        """Test that detection raises error without calibration."""
        detector = PieceDetector()
        frame = create_test_frame()

        with pytest.raises(ValueError, match="Calibration data required"):
            detector.detect_pieces(frame, region="slots")

    def test_detect_slots_region(self):
        """Test slot region detection."""
        calibration = create_mock_calibration()
        detector = PieceDetector(calibration=calibration)
        frame = create_test_frame()

        result = detector.detect_pieces(frame, region="slots")

        assert isinstance(result, PieceDetectionResult)
        assert isinstance(result.pieces, list)
        assert result.method in ["variance_lab", "lab_hard", "canny", "none"]
        assert result.debug_frame is not None

    def test_detect_popup_region(self):
        """Test popup region detection."""
        calibration = create_mock_calibration()
        detector = PieceDetector(calibration=calibration)
        frame = create_test_frame()

        result = detector.detect_pieces(frame, region="popup")

        assert isinstance(result, PieceDetectionResult)
        assert isinstance(result.pieces, list)
        assert result.debug_frame is not None

    def test_invalid_region_raises_error(self):
        """Test that invalid region raises error."""
        calibration = create_mock_calibration()
        detector = PieceDetector(calibration=calibration)
        frame = create_test_frame()

        with pytest.raises(ValueError, match="Unknown region"):
            detector.detect_pieces(frame, region="invalid")


class TestDetectedPiece:
    """Tests for DetectedPiece dataclass."""

    def test_create_detected_piece(self):
        """Test creating a DetectedPiece."""
        cells = [(0, 0), (0, 1), (1, 0)]  # L-shape
        bbox = (100, 100, 90, 90)
        center = (145, 145)
        cell_size = 30
        confidence = 0.95

        piece = DetectedPiece(
            cells=cells,
            bbox=bbox,
            center=center,
            cell_size=cell_size,
            confidence=confidence,
        )

        assert piece.cells == cells
        assert piece.bbox == bbox
        assert piece.center == center
        assert piece.cell_size == cell_size
        assert piece.confidence == confidence

    def test_legacy_shape_compatibility(self):
        """Test that legacy shape field is built automatically."""
        cells = [(0, 0), (1, 0), (2, 0)]  # 3 vertical
        piece = DetectedPiece(
            cells=cells,
            bbox=(100, 100, 30, 90),
            center=(115, 145),
            cell_size=30,
        )

        assert piece.shape is not None
        assert piece.shape.cells == cells

    def test_legacy_bounding_rect_compatibility(self):
        """Test that legacy bounding_rect field is built automatically."""
        cells = [(0, 0)]
        piece = DetectedPiece(
            cells=cells,
            bbox=(100, 100, 30, 30),
            center=(115, 115),
            cell_size=30,
        )

        assert piece.bounding_rect is not None
        assert piece.bounding_rect.x == 100
        assert piece.bounding_rect.y == 100
        assert piece.bounding_rect.width == 30
        assert piece.bounding_rect.height == 30


class TestPieceValidation:
    """Tests for piece validation logic."""

    def test_min_max_cells_constant(self):
        """Test MIN_CELLS and MAX_CELLS constants."""
        assert MIN_CELLS == 1
        assert MAX_CELLS == 9

    def test_slot_cell_scale(self):
        """Test SLOT_CELL_SCALE constant."""
        assert SLOT_CELL_SCALE == 0.44


class TestConnectedShapeValidation:
    """Tests for connected shape validation."""

    def test_single_cell_is_connected(self):
        """Test that a single cell is considered connected."""
        cells = [(0, 0)]
        detector = PieceDetector()

        # Access private method for testing
        is_connected = detector._is_connected_shape(cells)

        assert is_connected is True

    def test_horizontal_line_is_connected(self):
        """Test that a horizontal line is connected."""
        cells = [(0, 0), (0, 1), (0, 2)]
        detector = PieceDetector()

        is_connected = detector._is_connected_shape(cells)

        assert is_connected is True

    def test_vertical_line_is_connected(self):
        """Test that a vertical line is connected."""
        cells = [(0, 0), (1, 0), (2, 0)]
        detector = PieceDetector()

        is_connected = detector._is_connected_shape(cells)

        assert is_connected is True

    def test_l_shape_is_connected(self):
        """Test that L-shape is connected."""
        cells = [(0, 0), (1, 0), (2, 0), (2, 1)]
        detector = PieceDetector()

        is_connected = detector._is_connected_shape(cells)

        assert is_connected is True

    def test_disconnected_cells(self):
        """Test that disconnected cells are detected."""
        # Two separate single cells
        cells = [(0, 0), (2, 2)]
        detector = PieceDetector()

        is_connected = detector._is_connected_shape(cells)

        assert is_connected is False

    def test_empty_cells(self):
        """Test that empty cells list is not connected."""
        cells = []
        detector = PieceDetector()

        is_connected = detector._is_connected_shape(cells)

        assert is_connected is False


class TestBBoxValidation:
    """Tests for bounding box validation."""

    def test_valid_bbox(self):
        """Test that valid bbox passes validation."""
        detector = PieceDetector()
        cell_size = 30

        # Valid: ~1 cell
        is_valid = detector._is_valid_piece_bbox(cell_size, cell_size, cell_size)

        assert is_valid is True

    def test_too_small_bbox(self):
        """Test that too small bbox fails validation."""
        detector = PieceDetector()
        cell_size = 30

        # Too small: less than 0.5 cell
        is_valid = detector._is_valid_piece_bbox(10, 10, cell_size)

        assert is_valid is False

    def test_too_large_bbox(self):
        """Test that too large bbox fails validation."""
        detector = PieceDetector()
        cell_size = 30

        # Too large: more than 5 cells
        is_valid = detector._is_valid_piece_bbox(200, 200, cell_size)

        assert is_valid is False

    def test_extreme_aspect_ratio(self):
        """Test that extreme aspect ratio fails validation."""
        detector = PieceDetector()
        cell_size = 30

        # Very elongated: 10:1 ratio
        is_valid = detector._is_valid_piece_bbox(300, 30, cell_size)

        assert is_valid is False
