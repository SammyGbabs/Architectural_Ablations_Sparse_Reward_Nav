"""
main.py — Visualise a trained RL agent on the Residential Grid-World
=====================================================================
Loads a saved Stable-Baselines3 model (DQN or PPO) and rolls it out on
the custom environment for a configurable number of episodes. Per-step
PNG frames are saved under ``docs/rollouts/ep{N}/step_{t:03d}.png`` so
you can stitch them into a GIF/MP4 for the report.

Example
-------
    python main.py --model-path ppo_models/ppo_final_model.zip \\
                   --model-type ppo --episodes 3

Dependencies
------------
    gymnasium, stable-baselines3, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Gymnasium (the maintained fork of OpenAI Gym). SB3 >= 2.0 uses this.
import gymnasium as gym  # noqa: F401  (imported for type clarity / future use)

# Stable-Baselines3 is imported lazily inside visualize_agent() so that
# `python main.py --help` works even in environments where SB3 isn't
# installed yet (useful for quick README/docs checks).

# Our custom env + renderer
from Environment.custom_env import ResidentialGridEnv, ACTION_NAMES
from Environment.rendering import render_environment


def visualize_agent(
    model_path: str,
    model_type: str = "ppo",
    num_episodes: int = 3,
    step_delay: float = 0.0,
    save_frames: bool = True,
    frames_dir: str = "docs/rollouts",
    max_steps: int = 200,
    seed: int = 0,
) -> None:
    """
    Run `num_episodes` rollouts of a trained model and (optionally) save
    a 300-DPI PNG for each step.

    Parameters
    ----------
    model_path : str
        Path to a Stable-Baselines3 .zip model file.
    model_type : {"dqn", "ppo"}
        Which SB3 algorithm the zip was saved with.
    num_episodes : int
        Number of evaluation episodes.
    step_delay : float
        Optional sleep between steps (useful for watching live in a notebook).
    save_frames : bool
        If True, write a PNG per step under frames_dir/ep{N}/.
    frames_dir : str
        Root directory for saved frames.
    max_steps : int
        Episode step cap (passed to the env).
    seed : int
        Base RNG seed; each episode uses seed + episode_index.
    """
    # ---- Load model (SB3 imported lazily) -------------------------------
    from stable_baselines3 import DQN, PPO

    model_type = model_type.lower()
    if model_type == "dqn":
        model = DQN.load(model_path)
    elif model_type == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError("model_type must be 'dqn' or 'ppo'")

    # ---- Build env ------------------------------------------------------
    # render_mode="rgb_array" lets us grab a frame via env.render() if we want
    # to build a video later. We save publication-quality PNGs separately
    # below, which is usually what RL papers actually need.
    env = ResidentialGridEnv(max_steps=max_steps, render_mode="rgb_array")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)  # Gymnasium: (obs, info) tuple

        # Per-episode output directory
        if save_frames:
            ep_dir = os.path.join(frames_dir, f"ep{ep + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            # Save the initial state as step 000. The renderer takes a
            # single cell as the "goal marker" — we pass the target-room
            # centroid, which is the most informative visual proxy.
            render_environment(
                agent_pos=info["agent_pos"],
                goal_pos=info["target_centroid"],
                save_path=os.path.join(ep_dir, "step_000.png"),
                dpi=200,            # 200 DPI for frames (300 is heavy for many frames)
                show_legend=False,  # trim legend on frames for readability
            )

        total_reward = 0.0
        terminated = truncated = False
        step_count = 0
        t0 = time.time()

        while not (terminated or truncated):
            # SB3's predict() returns (action, state); deterministic=True is
            # standard for evaluation — stops the policy from exploring.
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Gymnasium: 5-tuple (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if save_frames:
                render_environment(
                    agent_pos=info["agent_pos"],
                    goal_pos=info["target_centroid"],
                    save_path=os.path.join(ep_dir, f"step_{step_count:03d}.png"),
                    dpi=200,
                    show_legend=False,
                )

            if step_delay > 0:
                time.sleep(step_delay)

        # ---- Episode summary --------------------------------------------
        # terminated=True now covers TWO outcomes — target reached OR
        # collision — so we disambiguate via the info-dict flags set by
        # step() on the final transition.
        duration = time.time() - t0
        reached_target = info.get("reached_target", False)
        collided       = info.get("collision", False)
        if reached_target:
            outcome = "Target reached"
        elif collided:
            outcome = "Collision (episode terminated)"
        elif truncated:
            outcome = "Timeout"
        else:
            outcome = "Unknown"

        print(f"\nEpisode {ep + 1} Results:")
        print(f"  Target room      : {info['target_room']}")
        print(f"  Final position   : {info['agent_pos']}")
        print(f"  Last action      : {ACTION_NAMES[action]}")
        print(f"  Dist to target   : {info['distance_to_target']:.2f}")
        print(f"  Doorways passed  : {info.get('doorways_passed', 0)}")
        print(f"  Total reward     : {total_reward:+.3f}")
        print(f"  Duration (wall)  : {duration:.1f}s")
        print(f"  Steps            : {step_count}")
        print(f"  Outcome          : {outcome}")
        if save_frames:
            print(f"  Frames saved to  : {ep_dir}/")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL agent on the residential grid-world."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained SB3 model (.zip).")
    parser.add_argument("--model-type", type=str, choices=["dqn", "ppo"],
                        default="ppo", help="RL algorithm used for training.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes.")
    parser.add_argument("--step-delay", type=float, default=0.0,
                        help="Extra seconds to sleep between steps (for live viewing).")
    parser.add_argument("--no-frames", action="store_true",
                        help="Disable per-step PNG frame saving.")
    parser.add_argument("--frames-dir", type=str, default="docs/rollouts",
                        help="Directory to write per-step PNG frames into.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Per-episode step cap.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; episode i uses seed + i.")
    args = parser.parse_args()

    visualize_agent(
        model_path=args.model_path,
        model_type=args.model_type,
        num_episodes=args.episodes,
        step_delay=args.step_delay,
        save_frames=not args.no_frames,
        frames_dir=args.frames_dir,
        max_steps=args.max_steps,
        seed=args.seed,
    )
