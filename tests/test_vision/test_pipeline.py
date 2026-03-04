import pytest
import numpy as np

from block_blast_bot.game_types import GameSnapshot, GameState
from block_blast_bot.calibration.data import CalibrationData, GridCalibration
from block_blast_bot.vision.pipeline import VisionPipeline


def create_test_calibration() -> CalibrationData:
    grid = GridCalibration(bounds=(100, 100, 400, 400), cell_size=(50, 50))
    return CalibrationData(
        version="2.0",
        device_resolution=(1080, 1920),
        window_resolution=(540, 960),
        grid=grid,
        slots=[],
    )


class TestVisionPipeline:
    def test_process_frame_returns_snapshot(self):
        cal = create_test_calibration()
        pipeline = VisionPipeline(calibration=cal)

        frame = np.zeros((960, 540, 3), dtype=np.uint8)
        # Add basic grid background to please variance_lab
        frame[100:500, 100:500] = (40, 40, 40)

        snapshot = pipeline.process_frame(frame)

        assert isinstance(snapshot, GameSnapshot)
        assert snapshot.state == GameState.NORMAL
        assert snapshot.board is not None
        assert snapshot.pieces is not None
