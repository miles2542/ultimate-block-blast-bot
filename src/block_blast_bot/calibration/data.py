"""
Block Blast Bot v2.0 - Calibration Data

Data structures for calibration information with JSON serialization support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class GridCalibration:
    """Grid region calibration data for the 8x8 game board."""

    bounds: Tuple[int, int, int, int]  # x, y, width, height
    cell_size: Tuple[int, int]  # width, height of each cell
    cell_centers: List[List[Tuple[int, int]]] = field(
        default_factory=list
    )  # 8x8 grid of (x, y) centers

    def __post_init__(self) -> None:
        """Compute cell centers if not provided."""
        if not self.cell_centers:
            self._compute_cell_centers()

    def _compute_cell_centers(self) -> None:
        """Compute the center coordinates for each cell in the grid."""
        x, y, w, h = self.bounds
        cell_w, cell_h = self.cell_size

        self.cell_centers = []
        for row in range(8):
            row_centers = []
            for col in range(8):
                center_x = x + col * cell_w + cell_w // 2
                center_y = y + row * cell_h + cell_h // 2
                row_centers.append((center_x, center_y))
            self.cell_centers.append(row_centers)

    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        """Get the center coordinates for a specific cell."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.cell_centers[row][col]
        raise IndexError(f"Cell ({row}, {col}) out of bounds")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bounds": list(self.bounds),
            "cell_size": list(self.cell_size),
            "cell_centers": [[list(c) for c in row] for row in self.cell_centers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridCalibration":
        """Create from dictionary."""
        cell_centers = [[tuple(c) for c in row] for row in data.get("cell_centers", [])]
        return cls(
            bounds=tuple(data["bounds"]),
            cell_size=tuple(data["cell_size"]),
            cell_centers=cell_centers,
        )


@dataclass
class SlotCalibration:
    """Piece slot calibration data."""

    center: Tuple[int, int]  # x, y center of the slot
    bounds: Tuple[int, int, int, int]  # x, y, width, height

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "center": list(self.center),
            "bounds": list(self.bounds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SlotCalibration":
        """Create from dictionary."""
        return cls(
            center=tuple(data["center"]),
            bounds=tuple(data["bounds"]),
        )


@dataclass
class DetectionCalibration:
    """Detection parameters calibration."""

    lab_threshold: int = 24  # LAB color distance threshold
    morph_kernel_size: int = 5  # Morphological operation kernel size

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lab_threshold": self.lab_threshold,
            "morph_kernel_size": self.morph_kernel_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionCalibration":
        """Create from dictionary."""
        return cls(
            lab_threshold=data.get("lab_threshold", 24),
            morph_kernel_size=data.get("morph_kernel_size", 5),
        )


@dataclass
class CalibrationData:
    """Complete calibration data for the game."""

    version: str = "2.0"
    timestamp: str = ""
    device_resolution: Tuple[int, int] = (0, 0)  # width, height of device
    window_resolution: Tuple[int, int] = (0, 0)  # width, height of scrcpy window
    grid: Optional[GridCalibration] = None
    slots: List[SlotCalibration] = field(default_factory=list)  # 3 slots
    detection: DetectionCalibration = field(default_factory=DetectionCalibration)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_valid(self) -> bool:
        """Check if calibration data is valid and complete."""
        return (
            self.grid is not None
            and len(self.slots) == 3
            and self.device_resolution[0] > 0
            and self.device_resolution[1] > 0
            and self.window_resolution[0] > 0
            and self.window_resolution[1] > 0
        )

    def get_scale_factor(self) -> Tuple[float, float]:
        """Get the scale factor from window to device coordinates."""
        if self.window_resolution[0] > 0 and self.window_resolution[1] > 0:
            return (
                self.device_resolution[0] / self.window_resolution[0],
                self.device_resolution[1] / self.window_resolution[1],
            )
        return (1.0, 1.0)

    def window_to_device(self, x: int, y: int) -> Tuple[int, int]:
        """Convert window coordinates to device coordinates."""
        scale_x, scale_y = self.get_scale_factor()
        return (int(x * scale_x), int(y * scale_y))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "device_resolution": list(self.device_resolution),
            "window_resolution": list(self.window_resolution),
            "grid": self.grid.to_dict() if self.grid else None,
            "slots": [s.to_dict() for s in self.slots],
            "detection": self.detection.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationData":
        """Create from dictionary."""
        grid = GridCalibration.from_dict(data["grid"]) if data.get("grid") else None
        slots = [SlotCalibration.from_dict(s) for s in data.get("slots", [])]
        detection = DetectionCalibration.from_dict(data.get("detection", {}))

        return cls(
            version=data.get("version", "2.0"),
            timestamp=data.get("timestamp", ""),
            device_resolution=tuple(data.get("device_resolution", [0, 0])),
            window_resolution=tuple(data.get("window_resolution", [0, 0])),
            grid=grid,
            slots=slots,
            detection=detection,
        )

    def save(self, path: str) -> None:
        """
        Save calibration data to JSON file.

        Args:
            path: Path to save the calibration file
        """
        # Update timestamp on save
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationData":
        """
        Load calibration data from JSON file.

        Args:
            path: Path to the calibration file

        Returns:
            CalibrationData instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Legacy compatibility aliases
BoardCalibration = GridCalibration
PieceSlotCalibration = SlotCalibration


__all__ = [
    "GridCalibration",
    "SlotCalibration",
    "DetectionCalibration",
    "CalibrationData",
    "BoardCalibration",  # Legacy alias
    "PieceSlotCalibration",  # Legacy alias
]
