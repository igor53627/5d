"""Season sign, reward, founder init, and the within-body versus replacement split."""
import json
from pathlib import Path

import numpy as np
import pytest

import fl_season as Z
import glider_eco as GE
import init18c as I
import metrics18c as M
import run18c as R

ROOT = Path(__file__).resolve().parents[1]


def test_season_sign_and_full_run_switches():
    import jax.numpy as jnp
    Ts = 8000.0
    assert float(Z.season_half(0, Ts, np)) == 1.0
    assert float(Z.season_half(Ts / 4, Ts, np)) == 1.0
    assert float(Z.season_half(Ts / 2, Ts, np)) == 1.0
    assert float(Z.season_half(Ts / 2 + 1, Ts, np)) == -1.0
    assert float(Z.season_half(3 * Ts / 4, Ts, np)) == -1.0
    assert float(Z.season_half(Ts, Ts, np)) == 1.0
    # Historical sine, kept for 18/18b. It really does disagree across dtypes at a logged boundary.
    assert float(Z.season(jnp.float32(Ts / 2), jnp.float32(Ts))) == -1.0
    assert float(Z.season(Ts / 2, Ts, np)) == 1.0
    assert Z.BDEF["season_rule"] == 0.0
    for t in range(0, 20001, 200):
        assert float(Z.season_half(t, Ts, np)) == float(Z.season_half(jnp.float32(t), jnp.float32(Ts)))
    n_full, sg = R.sign_changes(20000)
    n_pilot, _ = R.sign_changes(12000)
    assert sg[0] == 1.0 and n_full == 4 and n_pilot == 2
    # The logged phenotype was written by the step that read t-1, so the switch is visible one frame later.
    logged = [Z.snapshot_season(t, Ts) for t in range(0, 20001, 200)]
    assert sum(logged[i] != logged[i - 1] for i in range(1, len(logged))) == 4
    assert Z.snapshot_season(4000, Ts) == 1.0 and Z.snapshot_season(4200, Ts) == -1.0
    assert Z.snapshot_season(8000, Ts) == -1.0 and Z.snapshot_season(8200, Ts) == 1.0


def test_targets_and_food_gain_sign():
    cc = np.array([-1.0, -0.2, 0.0, 0.4, 1.0])
    smooth = Z.targets(cc, {"F0": 0.08, "smooth": 1.0}, np)
    assert np.allclose(smooth[0] + smooth[1], 0.08)
    assert np.allclose(smooth[0], 0.08 * (1 + cc) / 2)
    hard = Z.targets(cc, {"F0": 0.08, "smooth": 0.0}, np)
    assert np.allclose(hard[0], 0.08 * np.maximum(cc, 0))
    assert np.allclose(hard[1], 0.08 * np.maximum(-cc, 0))
    assert float(Z.food_gain(1.0, 0.0, 1.0, 0.5)) > 0
    assert float(Z.food_gain(0.0, 1.0, 1.0, 0.5)) < 0
    # cp=0.5: B-season gain is negative only past poison fraction 2/3.
    assert float(Z.food_gain(0.7, 0.3, -1.0, 0.5)) < 0
    assert float(Z.food_gain(0.6, 0.4, -1.0, 0.5)) > 0
    assert float(Z.food_gain(0.0, 1.0, -1.0, 0.5)) > 0


def test_part18b_new_channels_are_zero():
    founders = I.load_founders()
    assert set(founders) == {621, 622, 623, 624}
    new = list(Z.new_channel_flat())
    for p in founders.values():
        assert p.shape == (Z.KB,)
        assert np.all(p[Z.K0 + np.array(new)] == 0.0)
        assert not I.is_switcher(p[Z.K0:Z.iM])


@pytest.fixture(scope="module")
def pops():
    return R.populations()


def test_inputs_keep_old_weights_and_do_not_switch(pops):
    for seed, founder in I.SEED_OF_FOUNDER.items():
        rows, _, p0 = pops[seed]["inputs"]
        rec = I.genome_report(rows, p0)
        assert rec["n_switch"] == 0
        assert rec["max_abs_delta_old"] == 0.0
        assert rec["std_new"] > 0.25
        assert np.allclose(rows, pops[seed]["cue"][0])
        assert founder == I.SEED_OF_FOUNDER[seed]


def test_lo_is_small_noise_and_nobrain_matches(pops):
    for seed in I.SEED_OF_FOUNDER:
        rows, _, p0 = pops[seed]["lo"]
        delta = rows - p0
        assert 0.03 < float(delta.std()) < 0.08
        assert I.genome_report(rows, p0)["n_switch"] == 0
        assert np.allclose(rows, pops[seed]["nobrain"][0])
        assert not np.allclose(rows, pops[seed]["inputs"][0])


def test_planted_switchers_edit_only_cue_and_output_bias(pops):
    protected_cols = [Z.IN_A, Z.IN_B, Z.IN_R, Z.IN_H, Z.IN_UB, Z.IN_UFAR]
    for seed in I.SEED_OF_FOUNDER:
        rows, info, p0 = pops[seed]["planted"]
        rec = I.genome_report(rows, p0)
        assert rec["n_switch"] == I.N_PLANTED
        assert rec["switch"][:I.N_INIT - I.N_PLANTED] == [False] * (I.N_INIT - I.N_PLANTED)
        assert rec["switch"][I.N_INIT - I.N_PLANTED:] == [True] * I.N_PLANTED
        w0 = p0[Z.K0:Z.iM]
        W10, b10, W20, b20 = [np.array(x) for x in Z.unpack(w0.astype(np.float32), 1.0)]
        for i, row in enumerate(rows):
            if i < I.N_INIT - I.N_PLANTED:
                assert np.allclose(row, p0)
                assert info[i] is None
                continue
            w = row[Z.K0:Z.iM]
            W1, b1, W2, b2 = [np.array(x) for x in Z.unpack(w.astype(np.float32), 1.0)]
            assert np.allclose(W1[:, protected_cols], W10[:, protected_cols])
            assert np.allclose(W2, W20)
            assert np.allclose(b1, b10)
            assert np.allclose(b2[Z.OUT_GR:], b20[Z.OUT_GR:])
            assert info[i]["rms"] < 0.35
            assert not np.allclose(W1[:, Z.IN_CUE], 0.0)
        # The six switchers are not copies of each other.
        tail = rows[I.N_INIT - I.N_PLANTED:]
        assert np.unique(np.round(tail, 5), axis=0).shape[0] == I.N_PLANTED


def test_specs_match_arms_and_18b_ecology(pops):
    c, eco, _ = R.base_world()
    for kind, n_seed in (("pilot", 1), ("full", 4)):
        payloads = R.build(kind, pops)
        assert len(payloads) == n_seed * len(I.ARMS)
        by = {}
        for item in payloads:
            run = item["runs"][0]
            by[run["tag"]] = run
            assert run["model"] == "season" and run["K"] == Z.KB and run["L"] == 48
            assert run["steps"] == R.STEPS[kind] and run["every"] == 200
            assert run["p_spread"] == 0.0 and run["min_gap"] == 4 and run["min_sep"] == 1
            assert run["opts"] == {"coarse": True, "fnoise": True, "m": 5}
            assert run["c"]["mode"] == c["mode"]
            e = run["ecos"][0]
            for key in ("keat", "dec", "F0", "cp", "rG", "lam", "smooth", "krep", "hom", "mutw"):
                assert e[key] == eco[key]
            assert e["Ts"] == 8000.0 and e["season_rule"] == 1.0
        assert by["inputs_s731"]["p_inits"] == by["cue_s731"]["p_inits"]
        assert by["lo_s731"]["p_inits"] == by["nobrain_s731"]["p_inits"]
        assert by["inputs_s731"]["p_inits"] != by["planted_s731"]["p_inits"]
        assert by["cue_s731"]["ecos"][0]["cue"] == 1.0 and by["inputs_s731"]["ecos"][0]["cue"] == 0.0
        assert by["cue_s731"]["ecos"][0]["mem"] == by["inputs_s731"]["ecos"][0]["mem"] == 1.0
        assert by["nobrain_s731"]["ecos"][0]["kb"] == 0.0 and by["nobrain_s731"]["ecos"][0]["kr"] == 0.0
        assert by["lo_s731"]["ecos"][0]["kb"] == 1.0
    full = R.build("full", pops)
    seeds = {item["runs"][0]["seed"] for item in full}
    assert seeds == {731, 732, 733, 734}
    genomes = {}
    for item in full:
        run = item["runs"][0]
        if run["tag"].startswith("inputs_"):
            genomes[run["seed"]] = run["p_inits"]
    assert len({json.dumps(v) for v in genomes.values()}) == 4


def test_min_gap_and_p_inits_do_not_move_placement():
    L, N, K = 18, 2, 4
    A = np.zeros((L, L), np.float32)
    A[2:6, 2:6] = 0.6
    st = {"A": A, "x": np.zeros_like(A), "y": np.zeros_like(A), "z": np.zeros_like(A), "c": np.full_like(A, 2.6)}
    kw = dict(n_init=4, L=L, N=N, K=K, p0=np.zeros(K), p_spread=0.2, min_sep=1, min_gap=None)
    b1, _, p1 = GE.init_multi(st, np.random.default_rng(3), p_inits=None, **kw)
    drawn = np.zeros((4, K))
    b2, _, p2 = GE.init_multi(st, np.random.default_rng(3), p_inits=drawn, **kw)
    assert np.allclose(b1["A"], b2["A"])
    assert np.allclose(p2, 0.0)
    assert not np.allclose(p1, p2)
    old = np.zeros((L, L), np.float32)
    old[0, 0] = 1.0
    near = np.zeros_like(old)
    near[0, 2] = 1.0
    far = np.zeros_like(old)
    far[0, 4] = 1.0
    assert GE.masks_ok(near, old, 1, None)          # adjacent cells do not overlap; min_sep is not a distance
    assert not GE.masks_ok(near, old, 1, 2)         # Chebyshev distance 2 is inside the gap
    assert GE.masks_ok(far, old, 1, 2)              # distance 4 is outside a gap of 2
    with pytest.raises(ValueError):
        GE.init_multi(st, np.random.default_rng(1), p_inits=np.zeros((2, K)), **kw)


def _obj(mass, center, gA, gB, eA, eB):
    q = [0.0] * (Z.KB + Z.ND)
    q[Z.KB], q[Z.KB + 1], q[Z.KB + 3], q[Z.KB + 4] = eA, eB, gA, gB
    return (mass, list(center), q)


def test_switch_split_behavior_versus_replacement():
    L = 48
    behav = [
        (0, [_obj(1, [0, 0, 0, 0], 1, -1, 0.9, 0.1), _obj(1, [20, 0, 0, 0], 1, -1, 0.9, 0.1)]),
        (200, [_obj(1, [0.4, 0, 0, 0], -1, 1, 0.2, 0.8), _obj(1, [20.4, 0, 0, 0], -1, 1, 0.2, 0.8)]),
    ]
    got = M.decompose_switch(behav, L, [1.0, -1.0], Z.KB + 3, Z.KB + 4)[0]
    assert got["within"] == pytest.approx(-4.0)
    assert got["replacement"] == pytest.approx(0.0)
    assert got["sign_flip_mass_frac"] == pytest.approx(1.0)
    repl = [
        (0, [_obj(1, [0, 0, 0, 0], 1, -1, 0, 0), _obj(1, [30, 0, 0, 0], -1, 1, 0, 0)]),
        (200, [_obj(0.1, [0.5, 0, 0, 0], 1, -1, 0, 0), _obj(1.9, [30.5, 0, 0, 0], -1, 1, 0, 0)]),
    ]
    got = M.decompose_switch(repl, L, [1.0, -1.0], Z.KB + 3, Z.KB + 4)[0]
    assert got["within"] == pytest.approx(0.0)
    assert got["delta_mean"] == pytest.approx(-1.8)
    assert got["replacement"] == pytest.approx(-1.8)
    assert got["sign_flip_mass_frac"] == pytest.approx(0.0)
    # A body that appears far from both parents is an entrant, not a within-body flip.
    born = [
        (0, [_obj(1, [0, 0, 0, 0], 1, -1, 0, 0)]),
        (200, [_obj(1, [0, 0, 0, 0], 1, -1, 0, 0), _obj(1, [40, 0, 0, 0], -1, 1, 0, 0)]),
    ]
    got = M.decompose_switch(born, L, [1.0, -1.0], Z.KB + 3, Z.KB + 4)[0]
    assert got["within"] == pytest.approx(0.0)
    assert got["n_matched"] == 1 and got["entrant_mass_frac"] == pytest.approx(0.5)
    diet = M.diet_around(behav, [1.0, -1.0], Z.KB, Z.KB + 1, half=1)[0]
    assert diet["poison_before"] == pytest.approx(0.1)
    assert diet["poison_after"] == pytest.approx(0.2)
    series = M.population_series([(0, [_obj(0.2, [0, 0, 0, 0], 0, 0, 0, 0), _obj(0.8, [5, 0, 0, 0], 0, 0, 0, 0)])])
    assert M.fusion_time(series) == 0.0
    tie = M.population_series([(0, [_obj(0.5, [0, 0, 0, 0], 0, 0, 0, 0), _obj(0.5, [5, 0, 0, 0], 0, 0, 0, 0)])])
    assert M.fusion_time(tie) is None


def test_repulsion_is_smaller_than_food_taxis():
    r = M.taxis_repulsion_ratio()
    assert r["coarse"] is True and r["krep"] == 10.0
    assert 0.01 < r["ratio"] < 0.25
    assert r["v_food"] > r["v_rep"]


def test_live_step_reward_and_cue_signs():
    R.smoke_step()


def test_committed_specs_match_the_phenotype_contract():
    pilot = json.loads((ROOT / "archive" / "specs" / "b18c_pilot.json").read_text())
    full = json.loads((ROOT / "archive" / "specs" / "b18c_full.json").read_text())
    assert len(pilot) == 5 and len(full) == 20
    def one(spec, tag):
        return next(item["runs"][0] for item in spec if item["runs"][0]["tag"] == tag)
    assert one(pilot, "planted_s731")["p_inits"] == one(full, "planted_s731")["p_inits"]
    assert one(pilot, "cue_s731")["steps"] == 12000 and one(full, "cue_s731")["steps"] == 20000
    assert one(full, "lo_s731")["ecos"][0]["season_rule"] == 1.0
    for item in full:
        run = item["runs"][0]
        rows = np.asarray(run["p_inits"], float)
        n_sw = sum(I.is_switcher(row[Z.K0:Z.iM]) for row in rows)
        if run["tag"].startswith("planted"):
            assert n_sw == 6
        else:
            assert n_sw == 0
        assert len(rows) == 12 and rows.shape[1] == Z.KB


def test_analyze18c_reads_the_modal_ecolist_pickle(tmp_path):
    import pickle
    import analyze18c as A
    log = [
        (0, [_obj(1, [0, 0, 0, 0], 1, -1, 0.9, 0.1)]),
        (200, [_obj(1, [0.2, 0, 0, 0], 1, -1, 0.9, 0.1)]),
    ]
    eco = dict(Ts=8000.0, season_rule=1.0)
    result = dict(runs=[dict(log=log, eco=eco)], p_init=np.zeros((2, Z.KB)).tolist())
    blob = dict(kind="ecolist", out=[dict(planted_s731=result, summary=[("planted_s731", None)])])
    path = tmp_path / "b18c_0.pkl"
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    runs = A.load(str(path))
    assert [r["tag"] for r in runs] == ["planted_s731"]
    summary = A.summarize(runs[0], 48)
    assert summary["n_switch"] == 0 and summary["init"]["n"] == 2
    with pytest.raises(ValueError):
        A.summarize(dict(runs[0], eco=dict(Ts=8000.0)), 48)


def test_season_jit_matches_numpy():
    import jax
    import jax.numpy as jnp
    fn = jax.jit(lambda t, Ts: Z.season_half(t, Ts))
    for t in (0, 200, 4000, 4001, 8000, 12000, 16000, 20000):
        got = float(fn(jnp.float32(t), jnp.float32(8000.0)))
        assert got == float(Z.season_half(t, 8000.0, np))
    choose = jax.jit(lambda t, Ts, rule: jnp.where(rule > 0, Z.season_half(t, Ts), Z.season(t, Ts)))
    # Half-period: historical float32 sine is already B; the 18c rule is still A.
    assert float(choose(jnp.float32(4000), jnp.float32(8000), jnp.float32(0))) == -1.0
    assert float(choose(jnp.float32(4000), jnp.float32(8000), jnp.float32(1))) == 1.0
