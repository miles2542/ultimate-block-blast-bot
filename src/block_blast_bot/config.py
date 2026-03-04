"""
Block Blast Bot v2.0 - Configuration Module

Handles loading and validation of user configuration.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    """Frame capture configuration."""

    method: str = Field(
        default="mss", description="Capture method: mss or scrcpy_direct"
    )
    fps: int = Field(default=10, ge=1, le=60, description="Target frames per second")
    timeout_ms: int = Field(
        default=3000, ge=100, description="Capture timeout in milliseconds"
    )


class VisionBoardConfig(BaseModel):
    """Board detection configuration."""

    method: str = Field(default="cascade", description="Detection method")
    lab_threshold: int = Field(
        default=24, ge=1, le=100, description="LAB color distance threshold"
    )
    morph_kernel_size: int = Field(
        default=5, ge=1, le=15, description="Morphological kernel size"
    )


class VisionPiecesConfig(BaseModel):
    """Piece detection configuration."""

    method: str = Field(default="cascade", description="Detection method")
    min_area: int = Field(
        default=100, ge=10, description="Minimum piece area in pixels"
    )
    max_cells: int = Field(default=9, ge=1, le=9, description="Maximum cells per piece")


class VisionStabilityConfig(BaseModel):
    """Vision stability configuration."""

    buffer_size: int = Field(
        default=3, ge=1, le=10, description="Frame buffer size for stability"
    )
    hash_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Hash similarity threshold"
    )


class VisionConfig(BaseModel):
    """Vision pipeline configuration."""

    board: VisionBoardConfig = Field(default_factory=VisionBoardConfig)
    pieces: VisionPiecesConfig = Field(default_factory=VisionPiecesConfig)
    stability: VisionStabilityConfig = Field(default_factory=VisionStabilityConfig)


class SolverWeightsConfig(BaseModel):
    """Solver evaluation weights (matching Rust core Weights struct)."""

    hole_penalty: float = Field(
        default=-88.0, description="Penalty for 1x1 trapped holes"
    )
    density_penalty: float = Field(
        default=-3.08, description="Penalty for total board density"
    )
    score_reward: float = Field(
        default=5.1, description="Reward for raw game points generated"
    )
    combo_preservation_reward: float = Field(
        default=32.7, description="Reward for clearing at least one line"
    )
    bumpiness_penalty: float = Field(
        default=-0.31, description="Penalty for jagged board surface"
    )
    well_depth_penalty: float = Field(
        default=-21.9, description="Penalty for deep 1-cell wide trenches"
    )


class SolverConfig(BaseModel):
    """Solver configuration."""

    time_limit_ms: int = Field(
        default=3000, ge=100, le=30000, description="Time limit per piece set"
    )
    beam_width: int = Field(default=150, ge=10, le=500, description="Beam search width")
    depth2_samples: int = Field(
        default=60, ge=10, le=200, description="Depth-2 sample count"
    )
    weights: SolverWeightsConfig = Field(default_factory=SolverWeightsConfig)


class InputConfig(BaseModel):
    """Input execution configuration."""

    drag_duration_ms: int = Field(
        default=300, ge=50, le=1000, description="Drag gesture duration"
    )
    pre_drop_ms: int = Field(
        default=200, ge=0, le=1000, description="Delay before picking up a piece"
    )
    post_drop_ms: int = Field(
        default=400, ge=0, le=2000, description="Delay after dropping a piece"
    )
    post_turn_ms: int = Field(
        default=600, ge=0, le=3000, description="Delay after finishing a complete turn"
    )
    tap_duration_ms: int = Field(
        default=50, ge=10, le=200, description="Tap gesture duration"
    )
    verify_placement: bool = Field(
        default=True, description="Verify piece placement after move"
    )
    popup_y_offset: int = Field(
        default=-150, description="Y-offset of piece when picked up"
    )
    scale_x: float = Field(default=0.75, description="Horizontal drag scalar")
    scale_y: float = Field(default=0.75, description="Vertical drag scalar")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR"
    )
    save_frames: bool = Field(default=False, description="Save debug frames")
    debug_dir: str = Field(default="./debug", description="Debug output directory")


class Config(BaseModel):
    """Root configuration model."""

    version: str = Field(default="2.0", description="Configuration version")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to ./config.yaml

    Returns:
        Validated Config object
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        # Return defaults if file doesn't exist
        return Config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                return Config()
            return Config(**data)
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"Failed to load config from {config_path}: {e}"
        )
        return Config()


def save_config(config: Config, config_path: Optional[Path] = None) -> bool:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to config.yaml. Defaults to ./config.yaml

    Returns:
        bool: True if successful
    """
    if config_path is None:
        config_path = Path("config.yaml")

    try:
        data = config.model_dump()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"Failed to save config to {config_path}: {e}"
        )
        return False


__all__ = ["Config", "load_config", "save_config"]
