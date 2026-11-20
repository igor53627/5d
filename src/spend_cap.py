"""Worst-case spend of a Modal launch, computed before a container starts.

A launch starts ``call_count`` containers and Modal stops each one when its timeout ends. Without
a preemption, the GPU charge of the launch is thus at most::

    call_count * timeout_seconds / 3600 * dollars_per_hour

``modal_app.spawn_h100`` calls ``require_spend_under_cap`` before it starts a container.

The bound covers the GPU charge for the time that the job code runs, when no container is
preempted. It does not cover:

- the CPU and memory charge;
- the start of a container, or the idle seconds before a container scales down;
- the seconds that a job can run after its timeout before Modal stops it;
- a preemption. Modal restarts a preempted Function on the same input, so the time before
  the preemption is charged in addition to the restarted run. The number of preemptions is
  not known before a launch.

It applies to one launch: two launches under the cap can add to more than the cap.

This module imports neither Modal nor JAX.
"""

from __future__ import annotations

SECONDS_PER_HOUR: int = 3600
H100_DOLLARS_PER_HOUR: float = 3.95


def _require_positive(name: str, value: float) -> None:
    """Raise ValueError unless the value is greater than zero. A NaN is not greater than zero."""
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")


def worst_case_dollars(call_count: int, timeout_seconds: int, dollars_per_hour: float) -> float:
    """Return the spend of a launch in which each call runs until its timeout.

    Raises ValueError when the call count, the timeout or the rate is not positive.
    """
    _require_positive("call_count", call_count)
    _require_positive("timeout_seconds", timeout_seconds)
    _require_positive("dollars_per_hour", dollars_per_hour)
    return call_count * timeout_seconds / SECONDS_PER_HOUR * dollars_per_hour


def require_spend_under_cap(
    call_count: int, timeout_seconds: int, dollars_per_hour: float, cap_dollars: float
) -> float:
    """Return the worst-case spend of a launch, or raise ValueError when it exceeds the cap.

    A worst case equal to the cap passes. Raises ValueError also when the cap, the call count,
    the timeout or the rate is not positive.
    """
    _require_positive("cap_dollars", cap_dollars)
    worst_dollars: float = worst_case_dollars(call_count, timeout_seconds, dollars_per_hour)
    if not worst_dollars <= cap_dollars:
        raise ValueError(
            f"worst-case GPU spend {worst_dollars:.4f} dollars exceeds the cap of "
            f"{cap_dollars:.4f} dollars: {call_count} calls x {timeout_seconds} s "
            f"x {dollars_per_hour} dollars per hour"
        )
    return worst_dollars
