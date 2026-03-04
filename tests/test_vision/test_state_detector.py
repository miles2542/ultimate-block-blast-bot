import numpy as np
from block_blast_bot.game_types import GameState
from block_blast_bot.vision.state_detector import StateDetector
from block_blast_bot.config import VisionStabilityConfig


class TestStateDetector:
    def test_detect_state_defaults_to_normal(self):
        detector = StateDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.detect_state(frame) == GameState.NORMAL

    def test_is_stable_requires_two_frames(self):
        detector = StateDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.is_stable([frame]) is False

    def test_is_stable_with_identical_frames(self):
        detector = StateDetector()
        f1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Identical frame should be 100% stable
        assert detector.is_stable([f1, f1]) is True

    def test_is_stable_with_different_frames(self):
        detector = StateDetector()
        f1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        f2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # two different random frames should be unstable
        assert detector.is_stable([f1, f2]) is False
