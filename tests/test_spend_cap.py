"""The spend cap arithmetic of a Modal launch, on values derived by hand. No Modal call."""

import math

import pytest

import spend_cap

RATE = spend_cap.H100_DOLLARS_PER_HOUR
PART19_CAP_DOLLARS = 15.0


def test_h100_rate_is_the_rate_of_the_notes() -> None:
    """Goal: the constant is the rate of Parts 18g to 19d. Method: compare with the notes."""
    assert RATE == 3.95


def test_worst_case_matches_hand_arithmetic() -> None:
    """Goal: the formula is calls x hours x rate. Method: three launches computed by hand.

    8 x 14400 s = 32 h, and 32 x 3.95 = 126.40.
    8 x 1700 s = 13600 s = 3.7778 h, and 3.7778 x 3.95 = 14.9222.
    8 x 1710 s = 13680 s = 3.8 h, and 3.8 x 3.95 = 15.01.
    """
    assert spend_cap.worst_case_dollars(8, 14400, RATE) == pytest.approx(126.40, abs=1e-9)
    assert spend_cap.worst_case_dollars(8, 1700, RATE) == pytest.approx(14.9222, abs=1e-4)
    assert spend_cap.worst_case_dollars(8, 1710, RATE) == pytest.approx(15.01, abs=1e-9)


def test_worst_case_is_linear_in_each_factor() -> None:
    """Goal: no factor is dropped. Method: 1 call for 1 hour costs the rate; then double each."""
    assert spend_cap.worst_case_dollars(1, 3600, 2.5) == 2.5
    assert spend_cap.worst_case_dollars(2, 3600, 2.5) == 5.0
    assert spend_cap.worst_case_dollars(1, 7200, 2.5) == 5.0
    assert spend_cap.worst_case_dollars(1, 3600, 5.0) == 5.0


def test_launch_under_the_cap_returns_the_worst_case() -> None:
    """Goal: a launch under the cap passes. Method: 8 calls at 1700 s, 14.92 against 15."""
    worst = spend_cap.require_spend_under_cap(8, 1700, RATE, PART19_CAP_DOLLARS)
    assert worst == pytest.approx(14.9222, abs=1e-4)


def test_launch_over_the_cap_is_refused_with_the_numbers() -> None:
    """Goal: a launch over the cap is refused. Method: 8 calls at 1710 s, 15.01 against 15."""
    with pytest.raises(ValueError) as refusal:
        spend_cap.require_spend_under_cap(8, 1710, RATE, PART19_CAP_DOLLARS)
    message = str(refusal.value)
    for number in ("15.01", "15.00", "8 calls", "1710 s", "3.95"):
        assert number in message, message


def test_four_hour_default_of_the_pilot_is_refused() -> None:
    """Goal: the old 4 h timeout cannot pass. Method: 8 calls at 14400 s, 126.40 against 15."""
    with pytest.raises(ValueError, match="126.40"):
        spend_cap.require_spend_under_cap(8, 14400, RATE, PART19_CAP_DOLLARS)


def test_cap_boundary_is_inclusive() -> None:
    """Goal: both sides of the boundary. Method: 4 calls x 1 h x 2.50 is 10 with no rounding."""
    assert spend_cap.require_spend_under_cap(4, 3600, 2.5, 10.0) == 10.0
    with pytest.raises(ValueError, match="exceeds the cap"):
        spend_cap.require_spend_under_cap(4, 3600, 2.5, 9.99)
    with pytest.raises(ValueError, match="exceeds the cap"):
        spend_cap.require_spend_under_cap(4, 3601, 2.5, 10.0)


@pytest.mark.parametrize("timeout_seconds", [0, -1, -1700])
def test_timeout_that_is_not_positive_is_rejected(timeout_seconds: int) -> None:
    """Goal: a launch needs a timeout. Method: zero and negative values, large cap."""
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        spend_cap.worst_case_dollars(8, timeout_seconds, RATE)
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        spend_cap.require_spend_under_cap(8, timeout_seconds, RATE, 1000.0)


@pytest.mark.parametrize("cap_dollars", [0.0, -15.0])
def test_cap_that_is_not_positive_is_rejected(cap_dollars: float) -> None:
    """Goal: a cap of zero or less is an error, not a pass. Method: a valid launch, bad cap."""
    with pytest.raises(ValueError, match="cap_dollars must be positive"):
        spend_cap.require_spend_under_cap(8, 1700, RATE, cap_dollars)


def test_other_inputs_that_would_void_the_check_are_rejected() -> None:
    """Goal: no input makes the check pass by accident. Method: zero counts and rates, NaN."""
    with pytest.raises(ValueError, match="call_count must be positive"):
        spend_cap.require_spend_under_cap(0, 1700, RATE, PART19_CAP_DOLLARS)
    with pytest.raises(ValueError, match="dollars_per_hour must be positive"):
        spend_cap.require_spend_under_cap(8, 1700, 0.0, PART19_CAP_DOLLARS)
    with pytest.raises(ValueError, match="cap_dollars must be positive"):
        spend_cap.require_spend_under_cap(8, 1700, RATE, math.nan)
    with pytest.raises(ValueError, match="dollars_per_hour must be positive"):
        spend_cap.require_spend_under_cap(8, 1700, math.nan, PART19_CAP_DOLLARS)
