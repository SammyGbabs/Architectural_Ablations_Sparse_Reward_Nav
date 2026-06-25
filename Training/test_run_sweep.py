"""
Tests for Training/run_sweep.py — queue construction, ordering, resumability.
No subprocess is launched.

Run with:  pytest Training/test_run_sweep.py -v
"""

from __future__ import annotations

from Training.run_sweep import (
    BASELINES,
    TRAINER_MODULE,
    build_command,
    build_queue,
    is_run_complete,
    mark_complete,
)
from Training.seeds import EXPLORATORY_SEEDS, MAIN_SEEDS


def test_queue_total_is_105():
    q = build_queue()
    assert len(q) == 105                       # 9*10 main + 3*5 baselines


def test_front_loaded_h1_comparison_first():
    q = build_queue()
    # first 20 runs are exactly ppo_exp1 and ppo_exp4, all 10 seeds each
    first20 = q[:20]
    assert {s.config_id for s in first20} == {"ppo_exp1", "ppo_exp4"}
    assert [s.config_id for s in q[:10]] == ["ppo_exp1"] * 10
    assert [s.config_id for s in q[10:20]] == ["ppo_exp4"] * 10
    assert sorted(s.seed for s in q[:10]) == list(MAIN_SEEDS)


def test_baselines_are_last_and_exploratory():
    q = build_queue()
    tail = q[-15:]
    assert {s.config_id for s in tail} == set(BASELINES)
    for cid in BASELINES:
        seeds = sorted(s.seed for s in q if s.config_id == cid)
        assert seeds == list(EXPLORATORY_SEEDS)      # 5 seeds each


def test_main_configs_have_ten_seeds():
    q = build_queue()
    for cid in ("ppo_exp2", "ppo_exp3", "dqn_exp1", "dqn_exp5"):
        assert sum(1 for s in q if s.config_id == cid) == 10


def test_every_algo_has_a_trainer_module():
    q = build_queue()
    for spec in q:
        assert spec.algo in TRAINER_MODULE


def test_build_command_uses_deterministic_id_no_fresh():
    q = build_queue()
    spec = q[0]
    import argparse
    args = argparse.Namespace(configs_dir="configs", output_dir="runs",
                              wandb_mode="online")
    cmd = build_command(spec, args)
    assert "--fresh" not in cmd                 # deterministic ids -> resume="allow"
    assert "-m" in cmd and TRAINER_MODULE[spec.algo] in cmd
    assert "--seed" in cmd and str(spec.seed) in cmd
    assert f"configs/{spec.config_id}.yaml".replace("/", "\\") in cmd or \
           f"configs/{spec.config_id}.yaml" in cmd


def test_resumability_marker_roundtrip(tmp_path):
    rn = "ppo_ppo_exp1_seed0"
    assert not is_run_complete(tmp_path, rn)
    mark_complete(tmp_path, rn)
    assert is_run_complete(tmp_path, rn)
