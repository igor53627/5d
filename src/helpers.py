"""Shared helper functions used across experiments."""

from math import gamma, pi


def ball_vol(N, R):
    """Volume of an N-dimensional ball with radius R."""
    return pi ** (N / 2) / gamma(N / 2 + 1) * R**N


def invariant(condition: bool, message: str) -> None:
    """Raise AssertionError with the message when an internal invariant does not hold.

    Use this in place of the `assert` statement, which `python -O` removes. The check
    is always on. Pass a plain bool: a traced JAX value has no truth value inside
    `jax.jit`, so check static facts such as shapes there and check numbers outside.
    Do not catch the error and continue with the state that failed the check.
    See docs/style-profile.md, "Mechanisms".
    """
    if not condition:
        raise AssertionError(message)
