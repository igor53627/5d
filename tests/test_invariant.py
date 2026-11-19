"""The invariant helper is always on, also where Python removes `assert`."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import invariant

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
SUBPROCESS_TIMEOUT_S = 60


def test_invariant_passes_when_the_condition_holds() -> None:
    """Goal: a true condition has no effect. Method: call it and check the return value."""
    assert invariant(True, "never shown") is None


def test_invariant_raises_with_the_message() -> None:
    """Goal: a false condition raises AssertionError. Method: check the type and the text."""
    with pytest.raises(AssertionError, match="mass is negative"):
        invariant(False, "mass is negative")


def test_invariant_survives_optimized_mode() -> None:
    """Goal: `python -O` does not remove the check. Method: run a child with -O.

    The child first runs a plain `assert False`. Under -O that statement is removed, so
    the only AssertionError that can end the child is the one from invariant().
    """
    code = "from helpers import invariant\nassert False\ninvariant(False, 'always on')\n"
    environment = dict(os.environ, PYTHONPATH=str(SOURCE_DIRECTORY))
    child = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert child.returncode == 1
    assert "AssertionError: always on" in child.stderr
