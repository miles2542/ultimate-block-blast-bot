"""
Block Blast Bot v2.0 - Logger

Logging utilities for debugging and diagnostics.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class BotLogger:
    """
    Logger for Block Blast Bot.

    Provides structured logging with file output and debug frame saving.
    """

    def __init__(
        self,
        name: str = "block_blast_bot",
        level: str = "INFO",
        log_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directory for log files
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._log_dir = log_dir or Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def setup_file_handler(self, filename: Optional[str] = None) -> None:
        """
        Set up file handler for logging.

        Args:
            filename: Log filename, defaults to timestamped name
        """
        if filename is None:
            filename = f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        handler = logging.FileHandler(self._log_dir / filename)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(handler)

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self._logger.error(msg)


__all__ = ["BotLogger"]