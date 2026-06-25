"""
Tests for the import-safe helpers in Training/metrics.py (no SB3 / Gymnasium).
The RichEvalCallback glue itself is exercised by the Colab smoke run.

Run with:  pytest Training/test_metrics.py -v
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from Training.metrics import (
    CSV_COLUMNS,
    aggregate_episode_metrics,
    finalize_run_csv,
    iqm,
    steps_to_fraction_of_asymptote,
    upsert_run_row,
)


def _ep(ret, length, success, collision, wait_freq, room):
    return {"return": ret, "length": length, "success": success,
            "collision": collision, "wait_freq": wait_freq, "target_room": room}


# ---------------------------------------------------------------------------
# iqm
# ---------------------------------------------------------------------------

def test_iqm_trims_tails():
    # 0..7: drop bottom 25% (0,1) and top 25% (6,7) -> mean(2,3,4,5)=3.5
    assert iqm(range(8)) == 3.5


def test_iqm_robust_to_outliers():
    base = [10.0] * 10
    assert iqm(base) == 10.0
    assert iqm(base + [1e6]) == pytest.approx(10.0, abs=1.0)  # outlier mostly trimmed


def test_iqm_empty_is_nan():
    assert math.isnan(iqm([]))


# ---------------------------------------------------------------------------
# aggregate_episode_metrics
# ---------------------------------------------------------------------------

def test_aggregate_basic_and_per_room():
    eps = [
        _ep(27.0, 11, True, False, 0.0, "kitchen"),
        _ep(27.0, 11, True, False, 0.1, "bedroom"),
        _ep(-5.0, 8, False, True, 0.0, "bathroom"),   # collision, fail
        _ep(20.0, 22, True, False, 0.2, "bathroom"),
    ]
    agg = aggregate_episode_metrics(eps)
    assert agg["success_rate"] == 0.75
    assert agg["collision_rate"] == 0.25
    assert agg["mean_ep_len"] == pytest.approx((11 + 11 + 8 + 22) / 4)
    assert agg["wait_freq"] == pytest.approx((0.0 + 0.1 + 0.0 + 0.2) / 4)
    # per-room
    assert agg["sr_kitchen"] == 1.0
    assert agg["sr_bedroom"] == 1.0
    assert agg["sr_bathroom"] == 0.5            # one success, one collision
    assert agg["len_bathroom"] == pytest.approx((8 + 22) / 2)


def test_aggregate_missing_room_is_nan():
    eps = [_ep(27.0, 11, True, False, 0.0, "kitchen")]
    agg = aggregate_episode_metrics(eps)
    assert math.isnan(agg["sr_bedroom"])
    assert math.isnan(agg["len_bathroom"])


def test_aggregate_empty():
    assert aggregate_episode_metrics([]) == {}


# ---------------------------------------------------------------------------
# sample efficiency
# ---------------------------------------------------------------------------

def test_steps_to_fraction_basic():
    # asymptote = mean(last 3) = mean(28,30,29) = 29; 90% = 26.1; first >= 26.1 is step 60k
    hist = [(20_000, 10.0), (40_000, 20.0), (60_000, 27.0),
            (80_000, 28.0), (100_000, 30.0), (120_000, 29.0)]
    assert steps_to_fraction_of_asymptote(hist, frac=0.9) == 60_000


def test_steps_to_fraction_nonpositive_asymptote_returns_none():
    hist = [(20_000, -10.0), (40_000, -6.0), (60_000, -5.0)]
    assert steps_to_fraction_of_asymptote(hist) is None


def test_steps_to_fraction_empty_returns_none():
    assert steps_to_fraction_of_asymptote([]) is None


# ---------------------------------------------------------------------------
# CSV upsert + finalize
# ---------------------------------------------------------------------------

def test_upsert_creates_header_and_dedupes_by_seed(tmp_path):
    p = tmp_path / "p1_ppo_exp4.csv"
    upsert_run_row(p, {"config_id": "ppo_exp4", "seed": 0, "success_rate": 0.9})
    upsert_run_row(p, {"config_id": "ppo_exp4", "seed": 1, "success_rate": 0.8})
    # re-run seed 0 -> overwrite, not duplicate
    upsert_run_row(p, {"config_id": "ppo_exp4", "seed": 0, "success_rate": 0.95})

    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == CSV_COLUMNS
    seeds = sorted(r["seed"] for r in rows)
    assert seeds == ["0", "1"]                 # exactly two rows
    s0 = next(r for r in rows if r["seed"] == "0")
    assert s0["success_rate"] == "0.95"        # overwritten value


def test_nan_serialized_as_blank(tmp_path):
    p = tmp_path / "x.csv"
    upsert_run_row(p, {"config_id": "c", "seed": 0, "sr_bedroom": float("nan")})
    with open(p, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["sr_bedroom"] == ""


class _FakeCallback:
    """Stand-in for RichEvalCallback to test finalize_run_csv without SB3."""
    def __init__(self):
        self.last_agg = {"eval_return_iqm": 27.0, "eval_return_mean": 26.5,
                         "success_rate": 1.0, "collision_rate": 0.0,
                         "mean_ep_len": 14.7, "wait_freq": 0.05,
                         "sr_kitchen": 1.0, "sr_bedroom": 1.0, "sr_bathroom": 1.0,
                         "len_kitchen": 11.0, "len_bedroom": 11.0, "len_bathroom": 22.0,
                         "n_eval_episodes": 30.0}

    def sample_efficiency(self):
        return 80_000


def test_finalize_run_csv_writes_full_row(tmp_path):
    p = tmp_path / "p1_ppo_exp4.csv"
    row = finalize_run_csv(p, phase="p1", config_id="ppo_exp4", algo="ppo",
                           seed=3, env_steps=200_000, callback=_FakeCallback(),
                           wandb_run_name="ppo_ppo_exp4_seed3")
    assert row["sample_eff_steps_90"] == 80_000
    with open(p, newline="") as f:
        written = next(csv.DictReader(f))
    assert written["success_rate"] == "1.0"
    assert written["len_bathroom"] == "22.0"
    assert written["seed"] == "3"
    assert written["wandb_run_name"] == "ppo_ppo_exp4_seed3"
