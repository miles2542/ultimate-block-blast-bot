import numpy as np

from unittest.mock import patch

from block_blast_bot.game_types import (
    GameSnapshot,
    GameState,
    BoardState,
    DetectedPiece,
)
from block_blast_bot.calibration.data import CalibrationData, GridCalibration
from block_blast_bot.debug.visualizer import VisionVisualizer


def create_test_calibration() -> CalibrationData:
    grid = GridCalibration(bounds=(100, 100, 400, 400), cell_size=(50, 50))
    return CalibrationData(
        version="2.0",
        device_resolution=(1080, 1920),
        window_resolution=(540, 960),
        grid=grid,
        slots=[],
    )


class TestVisionVisualizer:
    def test_init(self):
        cal = create_test_calibration()
        vis = VisionVisualizer(cal)
        assert vis.is_running() is False

    @patch("cv2.namedWindow")
    def test_start(self, mock_namedWindow):
        cal = create_test_calibration()
        vis = VisionVisualizer(cal)
        vis.start()
        assert vis.is_running() is True
        mock_namedWindow.assert_called_once()

    @patch("cv2.destroyWindow")
    def test_stop(self, mock_destroyWindow):
        cal = create_test_calibration()
        vis = VisionVisualizer(cal)
        vis.start()
        vis.stop()
        assert vis.is_running() is False
        mock_destroyWindow.assert_called_once()

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    def test_show_snapshot(self, mock_waitKey, mock_imshow):
        cal = create_test_calibration()
        vis = VisionVisualizer(cal)
        vis.start()

        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        board = BoardState(grid=np.zeros((8, 8), dtype=np.bool_))

        piece = DetectedPiece(
            cells=[(0, 0)], bbox=(0, 0, 10, 10), center=(5, 5), cell_size=10, shape=None
        )

        snapshot = GameSnapshot(
            board=board,
            pieces=[piece],
            state=GameState.NORMAL,
            frame=frame,
            timestamp=0.0,
        )

        vis.show_snapshot(snapshot)

        mock_imshow.assert_called_once()
        mock_waitKey.assert_called_once()
