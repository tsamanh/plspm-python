"""Tests for bootstrap_with_seed and sign_change.

Covers: seeded reproducibility across processes, deterministic replicate
ordering, iteration counts / collection alignment, failure reporting,
sign-change getters, and correction of artificially sign-flipped constructs.
"""
import numpy as np
import pandas as pd
import pytest

import plspm.config as c
import plspm.inner_model as im
import plspm.sign_change as sgch
import plspm.weights as w
from plspm.estimator import Estimator
from plspm.mode import Mode
from plspm.plspm_custom import Plspm_Custom
from plspm.scheme import Scheme

ITERATIONS = 30
PROCESSES = 2
N = 100
X_ITEMS = ["x1", "x2", "x3"]


def _synthetic_data():
    rng = np.random.default_rng(42)
    lv1 = rng.normal(size=N)
    lv2 = 0.6 * lv1 + rng.normal(scale=0.8, size=N)
    return pd.DataFrame({
        "x1": lv1 + rng.normal(scale=0.5, size=N),
        "x2": lv1 + rng.normal(scale=0.5, size=N),
        "x3": lv1 + rng.normal(scale=0.5, size=N),
        "y1": lv2 + rng.normal(scale=0.5, size=N),
        "y2": lv2 + rng.normal(scale=0.5, size=N),
        "y3": lv2 + rng.normal(scale=0.5, size=N)})


def _config(data):
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), default_scale=c.Scale.NUM)
    config.add_lv_with_columns_named("X", Mode.A, data, "x")
    config.add_lv_with_columns_named("Y", Mode.A, data, "y")
    return config


def _run_bootstrap(seed, processes=PROCESSES, sign_change=False, iterations=ITERATIONS):
    data = _synthetic_data()
    model = Plspm_Custom(data, _config(data), Scheme.PATH, bootstrap=True,
                         bootstrap_iterations=iterations, processes=processes,
                         seed=seed, sign_change=sign_change)
    return model.bootstrap()


# ---------------------------------------------------------------- bootstrap

def test_same_seed_same_processes_is_reproducible():
    b1 = _run_bootstrap(seed=123)
    b2 = _run_bootstrap(seed=123)
    pd.testing.assert_frame_equal(b1.paths(), b2.paths())
    pd.testing.assert_frame_equal(b1.weights(), b2.weights())
    pd.testing.assert_frame_equal(b1.loading(), b2.loading())
    pd.testing.assert_frame_equal(b1.r_squared(), b2.r_squared())
    pd.testing.assert_frame_equal(b1.total_effects(), b2.total_effects())


def test_replicate_order_is_deterministic():
    b1 = _run_bootstrap(seed=123)
    b2 = _run_bootstrap(seed=123)
    s1, s2 = b1.boot_scores(), b2.boot_scores()
    assert list(s1.keys()) == list(s2.keys())
    for key in s1:
        pd.testing.assert_frame_equal(s1[key], s2[key])


def test_different_seeds_give_different_results():
    b1 = _run_bootstrap(seed=123)
    b2 = _run_bootstrap(seed=456)
    assert not b1.paths().equals(b2.paths())


def test_all_iterations_returned_and_collections_aligned():
    b = _run_bootstrap(seed=123)
    assert b.failures() == 0
    assert len(b.boot_raw_weights()) == ITERATIONS
    assert len(b.boot_raw_loadings()) == ITERATIONS
    assert len(b.r_squared_list()) == ITERATIONS
    assert len(b.boot_scores()) == ITERATIONS
    assert len(b.boot_final_data()) == ITERATIONS
    assert len(b.boot_inner_model()) == ITERATIONS


def test_uneven_iteration_split_across_processes():
    # 31 iterations over 2 processes: one worker gets the extra iteration
    b = _run_bootstrap(seed=123, iterations=31)
    assert len(b.boot_raw_weights()) == 31


def test_seed_getter_returns_seed():
    assert _run_bootstrap(seed=123).seed() == 123


def test_random_seed_generated_when_none():
    b = _run_bootstrap(seed=None)
    assert isinstance(b.seed(), int)


def test_sign_change_outputs_have_expected_lengths_and_keys():
    b = _run_bootstrap(seed=123, sign_change=True)
    for getter in (b.boot_nc_weights, b.boot_nc_loadings, b.boot_nc_path_coef,
                   b.boot_cl_scores, b.boot_cl_weights, b.boot_cl_loadings,
                   b.boot_cl_path_coef, b.boot_cl_effects,
                   b.boot_cl_path_coef_recalc, b.boot_cl_effects_recalc,
                   b.boot_di_scores, b.boot_di_weights, b.boot_di_loadings,
                   b.boot_di_path_coef, b.boot_di_effects,
                   b.boot_di_path_coef_recalc, b.boot_di_effects_recalc,
                   b.boot_cs_scores, b.boot_cs_weights, b.boot_cs_loadings,
                   b.boot_cs_path_coef, b.boot_cs_effects,
                   b.boot_cs_path_coef_recalc, b.boot_cs_effects_recalc):
        assert len(getter()) == ITERATIONS
    assert "cl_effects_recalc_0" in b.boot_cl_effects_recalc()
    assert "di_effects_recalc_0" in b.boot_di_effects_recalc()


def test_sign_change_getters_raise_when_disabled():
    b = _run_bootstrap(seed=123, sign_change=False)
    with pytest.raises(ValueError):
        b.boot_nc_weights()
    with pytest.raises(ValueError):
        b.boot_cl_path_coef()
    with pytest.raises(ValueError):
        b.boot_cs_scores()


# ---------------------------------------------------------------- sign change

@pytest.fixture(scope="module")
def flipped_model():
    """Fit the model once, then artificially flip construct X's weights,
    loadings, and scores, as an arbitrary PLS sign indeterminacy would."""
    data = _synthetic_data()
    config = _config(data)
    model = Plspm_Custom(data, config, Scheme.PATH)
    outer = model.outer_model()
    inner = model.inner_model_raw()

    estimator = Estimator(config)
    calculator = w.WeightsCalculatorFactory(config, 100, 1e-6, np.sqrt(N / (N - 1)), Scheme.PATH)
    final_data, scores, weights = estimator.estimate(calculator, config.filter(data))
    cfg = estimator.config()
    boot_loadings = (scores.apply(lambda s: final_data.corrwith(s))
                     * cfg.odm(cfg.path())).sum(axis=1).to_frame().T

    flipped_weights = weights.copy()
    flipped_weights.loc[X_ITEMS] = -flipped_weights.loc[X_ITEMS]
    flipped_loadings = boot_loadings.copy()
    flipped_loadings[X_ITEMS] = -flipped_loadings[X_ITEMS]
    flipped_scores = scores.copy()
    flipped_scores["X"] = -flipped_scores["X"]
    flipped_inner = im.InnerModel(cfg.path(), flipped_scores)

    return {"cfg": cfg, "outer": outer, "orig_path": inner.path_coefficients(),
            "orig_weights": outer["weight"], "orig_loadings": outer["loading"],
            "scores": scores, "final_data": final_data,
            "f_weights": flipped_weights, "f_loadings": flipped_loadings,
            "f_scores": flipped_scores, "f_inner": flipped_inner}


def test_flip_produces_negative_path(flipped_model):
    fm = flipped_model
    assert np.sign(fm["f_inner"].path_coefficients().loc["Y", "X"]) \
        != np.sign(fm["orig_path"].loc["Y", "X"])


def test_naive_sign_change_corrects_flip(flipped_model):
    fm = flipped_model
    nw, nl, npath = sgch._boot_naive_sign_change(
        fm["f_weights"].squeeze(), fm["f_loadings"].squeeze(),
        fm["f_inner"].path_coefficients(),
        fm["orig_weights"], fm["orig_loadings"], fm["orig_path"])
    assert (np.sign(nw.loc[X_ITEMS]) == np.sign(fm["orig_weights"].loc[X_ITEMS])).all()
    assert (np.sign(nl.loc[X_ITEMS]) == np.sign(fm["orig_loadings"].loc[X_ITEMS])).all()
    assert np.sign(npath.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])


def test_construct_level_change_corrects_flip(flipped_model):
    fm = flipped_model
    scores, weights, loadings, path, effects, path_recalc, effects_recalc = \
        sgch._boot_construct_level_change(fm["cfg"], fm["f_inner"], fm["f_weights"].T,
                                          fm["f_loadings"], fm["f_scores"], fm["outer"])
    assert np.sign(scores["X"].iloc[0]) == np.sign(fm["scores"]["X"].iloc[0])
    assert (np.sign(weights[X_ITEMS].iloc[0]).values
            == np.sign(fm["orig_weights"].loc[X_ITEMS]).values).all()
    assert np.sign(path.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])
    assert np.sign(path_recalc.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])


def test_dominant_indicator_change_corrects_flip(flipped_model):
    fm = flipped_model
    scores, weights, loadings, path, effects, path_recalc, effects_recalc = \
        sgch._boot_dominant_indicator_change(fm["cfg"], fm["f_inner"], fm["f_weights"].T,
                                             fm["f_loadings"], fm["f_scores"], fm["outer"])
    assert np.sign(scores["X"].iloc[0]) == np.sign(fm["scores"]["X"].iloc[0])
    assert np.sign(path.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])
    assert np.sign(path_recalc.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])


def test_construct_scores_change_corrects_flip(flipped_model):
    fm = flipped_model
    scores, weights, loadings, path, effects, path_recalc, effects_recalc = \
        sgch._boot_construct_scores_change(fm["cfg"], fm["f_inner"], fm["f_weights"].T,
                                           fm["f_loadings"], fm["f_scores"],
                                           fm["final_data"], fm["outer"])
    assert not (scores.astype(str) == "E").any().any()
    assert np.sign(scores["X"].iloc[0]) == np.sign(fm["scores"]["X"].iloc[0])
    assert (np.sign(weights[X_ITEMS].iloc[0]).values
            == np.sign(fm["orig_weights"].loc[X_ITEMS]).values).all()
    assert np.sign(path.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])
    assert np.sign(path_recalc.loc["Y", "X"]) == np.sign(fm["orig_path"].loc["Y", "X"])


def test_sign_change_leaves_unflipped_sample_unchanged(flipped_model):
    """A sample with no flip should pass through every method unchanged."""
    fm = flipped_model
    data = _synthetic_data()
    config = _config(data)
    estimator = Estimator(config)
    calculator = w.WeightsCalculatorFactory(config, 100, 1e-6, np.sqrt(N / (N - 1)), Scheme.PATH)
    final_data, scores, weights = estimator.estimate(calculator, config.filter(data))
    cfg = estimator.config()
    inner = im.InnerModel(cfg.path(), scores)
    boot_loadings = (scores.apply(lambda s: final_data.corrwith(s))
                     * cfg.odm(cfg.path())).sum(axis=1).to_frame().T

    for func, args in [
            (sgch._boot_construct_level_change,
             (cfg, inner, weights.T, boot_loadings, scores, fm["outer"])),
            (sgch._boot_dominant_indicator_change,
             (cfg, inner, weights.T, boot_loadings, scores, fm["outer"])),
            (sgch._boot_construct_scores_change,
             (cfg, inner, weights.T, boot_loadings, scores, final_data, fm["outer"]))]:
        result = func(*args)
        pd.testing.assert_frame_equal(result[0], scores)
        pd.testing.assert_frame_equal(result[1], weights.T)
