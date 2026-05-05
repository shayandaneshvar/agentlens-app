"""File discovery, loading helpers, and validation.

This module handles locating trajectory files inside agent-run folders and
converting them into :class:`~swe_trace_sdk.models.Trace` objects.

The evaluation platform format uses ``chat-export-logs.json`` as its trajectory filename.
The openhands format uses ``trajectory_openhands.json``.
The atif format uses ``trajectory.json`` inside an ``atif/<agent>/`` directory.
Other formats may use different filenames or non-JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .models import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "find_trajectory_file",
    "find_openhands_trajectory_file",
    "find_atif_trajectory_files",
    "load_saved_trace",
    "load_trajectory",
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_trajectory_file(instance_dir: str | Path) -> Optional[Path]:
    """Locate a evaluation platform trajectory (``chat-export-logs.json``) inside *instance_dir*.

    Checks well-known sub-paths first, then falls back to a recursive glob.
    This helper is evaluation platform-specific; other formats may store trajectories
    under different names or in non-JSON files.

    Parameters
    ----------
    instance_dir:
        Path to an instance folder (e.g.
        ``run-12345-instance-task-logs/``).

    Returns
    -------
    Path | None
        Path to the trajectory file, or *None* if not found.
    """
    instance_dir = Path(instance_dir)
    candidates = [
        instance_dir / "output" / "vsc-output" / "chat-export-logs.json",
        instance_dir / "vsc-output" / "chat-export-logs.json",
        instance_dir / "chat-export-logs.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: recursive search
    for p in instance_dir.rglob("chat-export-logs.json"):
        return p
    return None


def find_openhands_trajectory_file(instance_dir: str | Path) -> Optional[Path]:
    """Locate an OpenHands trajectory (``trajectory_openhands.json``) inside *instance_dir*.

    Checks well-known sub-paths first, then falls back to a recursive glob.

    Parameters
    ----------
    instance_dir:
        Path to an instance folder (e.g.
        ``openhands-swebench/astropy__astropy-12907/``).

    Returns
    -------
    Path | None
        Path to the trajectory file, or *None* if not found.
    """
    instance_dir = Path(instance_dir)
    candidates = [
        instance_dir / "output" / "trajectories" / "trajectory_openhands.json",
        instance_dir / "trajectories" / "trajectory_openhands.json",
        instance_dir / "trajectory_openhands.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: recursive search
    for p in instance_dir.rglob("trajectory_openhands.json"):
        return p
    return None


def find_atif_trajectory_files(instance_dir: str | Path) -> List[Path]:
    """Locate ATIF trajectory files inside *instance_dir*.

    An ATIF session stores one trajectory per agent under
    ``atif/<agent>/trajectory.json``.  This function returns **all**
    trajectory files found (typically one for copilot and one for claude).

    Also handles the case where *instance_dir* already points to an
    agent directory (e.g. ``atif/copilot/``) containing a direct
    ``trajectory.json``.

    Parameters
    ----------
    instance_dir:
        Path to a session directory (e.g.
        ``sessions/20260302-225121-830822/``) or an agent directory.

    Returns
    -------
    list[Path]
        Sorted list of trajectory file paths found.  Empty if none.
    """
    instance_dir = Path(instance_dir)
    found: list[Path] = []

    # 1. Check direct trajectory.json (when instance_dir IS the agent dir)
    direct = instance_dir / "trajectory.json"
    if direct.exists():
        found.append(direct)
        return found

    # 2. Check standard atif/<agent>/trajectory.json paths
    atif_dir = instance_dir / "atif"
    if atif_dir.is_dir():
        for agent_dir in sorted(atif_dir.iterdir()):
            if agent_dir.is_dir():
                traj = agent_dir / "trajectory.json"
                if traj.exists():
                    found.append(traj)

    if found:
        return found

    # 3. Fallback: recursive search for trajectory.json files that
    #    look like ATIF (contain schema_version starting with 'ATIF').
    for p in instance_dir.rglob("trajectory.json"):
        try:
            # Quick peek at the file to check for ATIF schema
            with open(p, "r", encoding="utf-8") as fh:
                # Read only the first 500 bytes to check schema_version
                head = fh.read(500)
            if '"ATIF' in head:
                found.append(p)
        except (OSError, UnicodeDecodeError):
            pass

    return sorted(found)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_saved_trace(path: str | Path) -> Trace:
    """Load a :class:`Trace` previously saved by the SDK via :meth:`Trace.save`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as a Trace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    try:
        return Trace.load(str(path))
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Cannot parse {path} as a Trace: {exc}") from exc


def load_trajectory(path: str | Path, *, format: str) -> Trace:
    """Load a trajectory file and return a :class:`Trace`.

    Parameters
    ----------
    path : str | Path
        Path to the file.
    format : str
        ``"chatlog"``   — parse a raw evaluation platform ``chat-export-logs.json``.
        ``"openhands"``  — parse an OpenHands ``trajectory_openhands.json``.
        ``"atif"``       — parse an ATIF v1.6 ``trajectory.json``.
        ``"trace"``      — load a Trace JSON previously saved by the SDK.

    Raises
    ------
    ValueError
        If *format* is not supported.
    """
    _SUPPORTED_FORMATS = {"chatlog", "openhands", "atif", "trace"}
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported trajectory format: {format!r}. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    path = Path(path)

    if format == "trace":
        logger.debug("Loading saved Trace: %s", path)
        return load_saved_trace(path)

    if format == "atif":
        from ._generator_atif import ATIFTraceGenerator

        logger.debug("Generating Trace from ATIF trajectory: %s", path)
        gen = ATIFTraceGenerator()
        return gen.generate(str(path))

    if format == "openhands":
        from ._generator_openhands import OpenHandsTraceGenerator

        logger.debug("Generating Trace from OpenHands trajectory: %s", path)
        gen = OpenHandsTraceGenerator()
        return gen.generate(str(path))

    # format == "chatlog"
    from ._generator import TraceGenerator

    logger.debug("Generating Trace from evaluation platform trajectory: %s", path)
    gen = TraceGenerator()
    return gen.generate(str(path))
