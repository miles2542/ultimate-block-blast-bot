"""
Block Blast Bot v2.0 - Vision Debug Tool

Debug tool for testing vision pipeline with a live capture feed.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from block_blast_bot.config import load_config
from block_blast_bot.calibration.data import CalibrationData
from block_blast_bot.capture import FrameCapture
from block_blast_bot.vision.pipeline import VisionPipeline
from block_blast_bot.debug.visualizer import VisionVisualizer


def main() -> int:
    """Run vision debugging."""
    print("Block Blast Bot - Vision Debug Tool")
    print("====================================")

    # Load configuration
    config = load_config()

    # Load calibration data
    # Try to find calibration data in default locations
    cal_path = Path("calibration.json")
    if not cal_path.exists():
        cal_path = Path("config/calibration.json")

    if not cal_path.exists():
        print(
            "Error: Calibration data not found. Please run the calibration tool first:"
        )
        print("uv run tools/calibrate.py")
        return 1

    calibration = CalibrationData.load(cal_path)

    # Initialize components
    pipeline = VisionPipeline(calibration=calibration, config=config.vision)
    visualizer = VisionVisualizer(calibration=calibration)

    print("\nStarting live vision feed...")
    print(
        "Ensure your phone is plugged in, scrcpy is running, and Block Blast is open."
    )
    print("Press ESC in the visualizer window or Ctrl+C in terminal to exit.\n")

    try:
        with FrameCapture(config.capture) as capture:
            visualizer.start()

            while visualizer.is_running():
                # 1. Capture frame
                frame = capture.capture()

                # 2. Process through pipeline
                start_time = time.perf_counter()
                snapshot = pipeline.process_frame(frame)
                process_time = (time.perf_counter() - start_time) * 1000

                # 3. Show in visualizer
                if snapshot:
                    visualizer.show_snapshot(snapshot)

                    # Print stats occasionally or if desired
                    # print(f"\rProcess time: {process_time:.1f}ms | Board filled: {snapshot.board.grid.sum()}/64", end="")

                # Small sleep to yield CPU and maintain loop pace
                time.sleep(0.01)

    except RuntimeError as e:
        print(f"\nError initializing capture: {e}")
        print("Make sure scrcpy is running and visible!")
        return 1
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        visualizer.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
