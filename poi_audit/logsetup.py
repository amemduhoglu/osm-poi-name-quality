"""Logging setup shared by every script.

Logs go to stdout and to a rotating file under the configured log directory, so
that a long run leaves a record even when its terminal is gone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from poi_audit.config import path

_configured = False


def setup(name: str, level: str = "INFO") -> Any:
    """Configure loguru for one entry point and return the logger.

    The log file is named after the program that is running, not after the
    caller. A script that imports a helper from another script would otherwise
    write its run into the helper's log, because the helper configures logging
    first as it is imported.

    Args:
        name: Short name of the caller, used when the running program cannot be
            identified.
        level: Minimum level written to stdout and to the file.

    Returns:
        The configured loguru logger.
    """
    global _configured
    if _configured:
        return logger
    entry_point = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    if entry_point and entry_point not in {
        "pytest",
        "-c",
        "-",
        "__main__",
        "python",
        "python3",
    }:
        name = entry_point
    log_dir = path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=level, format="{time:HH:mm:ss} {level:<7} {message}")
    logger.add(
        log_dir / f"{name}.log",
        level=level,
        rotation="10 MB",
        retention=10,
        encoding="utf-8",
    )
    _configured = True
    return logger
