"""Loader for the single configuration file.

Every experiment parameter is read through this module. A script that needs a
number asks for it by its dotted key; it never carries a literal copy.

Example:
    >>> from poi_audit.config import get, seed_everything
    >>> get("injected_set.classes.M1.items")
    300
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

_MISSING = object()


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """Read the configuration file once and cache it.

    Returns:
        The parsed configuration as nested dictionaries.

    Raises:
        FileNotFoundError: If the configuration file is absent.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get(key: str, default: Any = _MISSING) -> Any:
    """Return a configuration value addressed by a dotted key.

    Args:
        key: Dotted path into the configuration, such as ``analysis.confidence``.
        default: Value to return when the key is absent. Without it, a missing
            key raises, because a silently defaulted parameter is a parameter
            that is no longer under the configuration file's control.

    Returns:
        The value stored at that key.

    Raises:
        KeyError: If the key is absent and no default was supplied.
    """
    node: Any = load()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"missing configuration key: {key}")
            return default
        node = node[part]
    return node


def path(key: str) -> Path:
    """Return a configured path resolved against the repository root.

    Args:
        key: Key under ``paths``, such as ``figures``.

    Returns:
        The absolute path. The directory is not created here.
    """
    return REPO_ROOT / str(get(f"paths.{key}"))


def secret(key: str) -> str:
    """Return the value of the environment variable named by a configuration key.

    Args:
        key: Dotted key whose value is the name of an environment variable, such
            as ``database.dsn_env``.

    Returns:
        The environment variable's value.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    load_dotenv(REPO_ROOT / ".env")
    variable = str(get(key))
    value = os.environ.get(variable, "")
    if not value:
        raise RuntimeError(
            f"environment variable {variable} is unset; copy .env.example to .env"
        )
    return value


def seed_everything() -> int:
    """Seed every random source from the configured seed.

    Returns:
        The seed that was applied, so that a run can log it.
    """
    value = int(get("project.seed"))
    random.seed(value)
    try:
        import numpy as np
    except ImportError:  # numpy is not needed by every entry point
        return value
    np.random.seed(value)
    return value
