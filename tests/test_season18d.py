"""Part 18d: bias noise at init, mut_bias override, and committed specs."""
import json
from pathlib import Path

import numpy as np
import pytest

import fl_season as Z
import init18c as I18c
import init18d as I
import run18d as R

ROOT = Path(__file__).resolve().parents[1]


def test_mut_bias_default_is_off():
    assert "mut_bias" in Z.BKEYS
    assert Z.BDEF["mut_bias"] < 0.0
    P = Z.params({})
    assert float(P["mut_bias"]) < 0.0


def test_mut_bias_override_in_params():
    P = Z.params({"mut_bias": 0.02, "mutw": 0.004})
    assert float(P["mut_bias"]) == pytest.approx(0.02)


def test_bias_indices_not_in_new_channels():
    new = set(Z.new_channel_flat())
    for i in I.bias_indices():
        assert i not in new


@pytest.fixture(scope="module")
def pops():
    return R.populations()


def test_bias_cue_moves_biases_and_new_channels(pops):
    for seed, founder in I.SEED_OF_FOUNDER.items():
        rows, _, p0 = pops[seed]["bias_cue"]
        rec = I.genome_report(rows, p0)
        assert rec["std_delta_bias"] > 0.15
        assert rec["max_abs_delta_bias"] > 0.0
        # New channels still noisy (same as 18c inputs).
        assert rec["std_delta_new"] > 0.25
        assert np.allclose(rows, pops[seed]["bias_nocue"][0])


def test_inputs_cue_keeps_biases_frozen_at_init(pops):
    for seed in I.SEED_OF_FOUNDER:
        rows, _, p0 = pops[seed]["inputs_cue"]
        rec = I.genome_report(rows, p0)
        assert rec["max_abs_delta_bias"] == 0.0
        assert rec["n_switch"] == 0
        # Old non-new non-bias weights unchanged.
        assert rec["max_abs_delta_old"] == 0.0


def test_planted_and_nobrain(pops):
    for seed in I.SEED_OF_FOUNDER:
        pl, _, _ = pops[seed]["planted"]
        assert I.genome_report(pl)["n_switch"] == I18c.N_PLANTED
        nb = pops[seed]["nobrain"][0]
        assert nb.shape[0] == I18c.N_INIT


def test_committed_18d_specs():
    for name, n_runs, steps in (("b18d_pilot.json", 5, 12000), ("b18d_full.json", 20, 20000)):
        path = ROOT / "archive" / "specs" / name
        assert path.exists(), f"missing {path}; run run18d.py write"
        spec = json.loads(path.read_text())
        assert len(spec) == n_runs
        tags = [item["runs"][0]["tag"] for item in spec]
        for arm in I.ARMS:
            assert any(t.startswith(arm + "_") for t in tags)
        for item in spec:
            r = item["runs"][0]
            e = r["ecos"][0]
            assert r["steps"] == steps
            assert e["krep"] == R.KREP
            assert e["Ts"] == R.TS
            assert e["season_rule"] == 1.0
            tag = r["tag"]
            if tag.startswith("bias_"):
                assert e["mut_bias"] == R.MUT_BIAS
            else:
                assert "mut_bias" not in e
            if tag.startswith("bias_cue") or tag.startswith("inputs_cue") or tag.startswith("planted"):
                assert e["cue"] == 1.0
            if tag.startswith("bias_nocue") or tag.startswith("nobrain"):
                assert e["cue"] == 0.0


def test_18c_specs_omit_mut_bias():
    """Old parts stay bit-identical: no mut_bias key → BDEF -1 → amp uses mutw."""
    for name in ("b18c_pilot.json", "b18c_full.json"):
        spec = json.loads((ROOT / "archive" / "specs" / name).read_text())
        for item in spec:
            assert "mut_bias" not in item["runs"][0]["ecos"][0]
