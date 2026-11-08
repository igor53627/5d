"""Shared helper functions used across experiments."""
from math import gamma, pi


def ball_vol(N, R):
    """Volume of an N-dimensional ball with radius R."""
    return pi ** (N / 2) / gamma(N / 2 + 1) * R ** N
