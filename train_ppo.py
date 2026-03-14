"""PPO Training Script for SmartRoad AI — Day 2 / Day 3.

Day 2: DriverEnv (FakePipeline), 10k timesteps -> ppo_v1.zip
Day 3: 50k timesteps, ent_coef=0.005  -> ppo_v2.zip
       training_curve.png, eval_results_v2.csv, v1 vs v2 comparison

Usage:
    python train_ppo.py          # runs Day 3 by default
    python train_ppo.py --day2   # re-runs Day 2
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl_environment import DriverEnv


# ---------------------------------------------------------------------------
# StubDriverEnv kept for reference — no longer used in training
# ---------------------------------------------------------------------------

import gymnasium as gym
from gymnasium import spaces

class StubDriverEnv(gym.Env):
    """Legacy stub from Day 1 — kept for reference only."""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= 200
        return self.observation_space.sample(), float(np.random.uniform(-2.0, 1.0)), terminated, False, {}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(model, env, n_episodes=20):
    """Run n_episodes with the trained model and collect results.

    Args:
        model: Trained PPO model.
        env: DriverEnv instance (will be reset each episode).
        n_episodes (int): Number of evaluation episodes.

    Returns:
        list[dict]: One dict per step with keys:
            episode, step, action, reward, obs_vector
    """
    records = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep * 7)
        terminated = False
        step = 0

        while not terminated:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)

            records.append({
                "episode":    ep,
                "step":       step,
                "action":     action,
                "reward":     round(float(reward), 4),
                "obs_vector": obs.tolist(),
            })

            if truncated:
                break

        print(f"  Eval episode {ep:>2d}/{n_episodes} — "
              f"{step} steps, last action={action}, "
              f"violations={info.get('violations', '?')}")

    return records


def save_eval_results(records, path="eval_results.csv"):
    """Write evaluation records to a CSV file.

    Args:
        records (list[dict]): Output of evaluate_agent().
        path (str): Destination CSV path.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "action", "reward", "obs_vector"])
        for r in records:
            writer.writerow([
                r["episode"],
                r["step"],
                r["action"],
                r["reward"],
                str(r["obs_vector"]),
            ])
    print(f"Saved {path} ({len(records)} rows)")


def action_distribution(records, label=""):
    """Print action distribution for a set of eval records."""
    actions = [r["action"] for r in records]
    total = len(actions)
    print(f"\nAction distribution — {label} ({total} steps):")
    for a, name in [(0, "ALL_CLEAR"), (1, "MONITOR"), (2, "VIOLATION")]:
        count = actions.count(a)
        print(f"  {name:<12s}: {count:>5d}  ({100 * count / total:.1f}%)")
    return actions


def plot_training_curve(monitor_csv="monitor_log_v2.monitor.csv",
                        out_path="training_curve.png"):
    """Read the SB3 Monitor CSV and save an episode reward plot.

    Args:
        monitor_csv (str): Path written by the Monitor wrapper.
        out_path (str): Output PNG path.
    """
    if not os.path.exists(monitor_csv):
        print(f"[WARN] Monitor CSV not found: {monitor_csv}")
        return

    episodes, rewards = [], []
    with open(monitor_csv, newline="") as f:
        # SB3 Monitor CSV has 1 comment line then a header line then data
        lines = f.readlines()
    # Find the header row (contains 'r,l,t')
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("r,"))
    reader = csv.DictReader(lines[header_idx:])
    for i, row in enumerate(reader, start=1):
        episodes.append(i)
        rewards.append(float(row["r"]))

    if not rewards:
        print("[WARN] Monitor CSV is empty — no plot generated.")
        return

    # Smooth with a 20-episode rolling mean
    window = 20
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i + 1]))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(episodes, smoothed, color="steelblue", linewidth=2,
             label=f"Rolling mean (n={window})")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SmartRoad AI — PPO v2 Training Curve (50 000 timesteps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Training curve saved -> {out_path}  ({len(rewards)} episodes)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("SmartRoad AI — PPO Training  (Day 2)")
    print("=" * 60)

    # ---- 1. Build env ----
    raw_env = DriverEnv(pipeline_fn=None, max_steps=200)

    print("\nRunning check_env() on DriverEnv...")
    check_env(raw_env.unwrapped)
    print("check_env() passed.\n")

    # Wrap with Monitor so SB3 logs per-episode reward/length to monitor.csv
    env = Monitor(raw_env, filename="monitor_log")

    # ---- 2. Quick 2 000-step sanity run ----
    print("--- Quick sanity run (2 000 timesteps) ---")
    probe = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        device="cpu",
    )
    probe.learn(total_timesteps=2_000)
    print("Sanity run complete.\n")

    # ---- 3. Full Day 2 training — 10 000 timesteps ----
    print("--- Full training (10 000 timesteps) ---")
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        device="cpu",
    )
    model.learn(total_timesteps=10_000)

    model.save("ppo_v1")
    print("\nModel saved -> ppo_v1.zip")

    # ---- 4. Evaluate ----
    print("\n--- Evaluating ppo_v1 over 20 episodes ---")
    eval_env = DriverEnv(pipeline_fn=None, max_steps=200)
    loaded = PPO.load("ppo_v1")

    records = evaluate_agent(loaded, eval_env, n_episodes=20)
    save_eval_results(records, path="eval_results.csv")


    # ---- 5. Action distribution summary ----
    actions = [r["action"] for r in records]
    total = len(actions)
    print(f"\nAction distribution over {total} steps:")
    for a, label in [(0, "ALL_CLEAR"), (1, "MONITOR"), (2, "VIOLATION")]:
        count = actions.count(a)
        print(f"  {label:<12s}: {count:>5d}  ({100*count/total:.1f}%)")

    # ---- 6. Inference smoke-test ----
    obs, _ = eval_env.reset(seed=0)
    action, _ = loaded.predict(obs, deterministic=True)
    print(f"\nInference check — obs shape: {obs.shape}, action: {int(action)}  (expected 0/1/2)")

    eval_env.close()
    env.close()
    print("\nDay 2 complete.")


# ---------------------------------------------------------------------------
# Day 3 Training
# ---------------------------------------------------------------------------

def train_day3():
    print("=" * 60)
    print("SmartRoad AI — PPO Training  (Day 3)")
    print("=" * 60)

    # ---- 1. Build env with hardened max_steps=300 ----
    raw_env = DriverEnv(pipeline_fn=None, max_steps=300)

    print("\nRunning check_env() on hardened DriverEnv (max_steps=300)...")
    check_env(raw_env.unwrapped)
    print("check_env() passed.\n")

    # Monitor wrapper logs every episode reward to CSV
    monitor_path = "monitor_log_v2"
    env = Monitor(raw_env, filename=monitor_path)

    # ---- 2. Train 50 000 timesteps ----
    print("--- Training ppo_v2  (50 000 timesteps) ---")
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.005,
        device="cpu",
    )
    model.learn(total_timesteps=50_000)

    model.save("ppo_v2")
    print("\nModel saved -> ppo_v2.zip")
    env.close()

    # ---- 3. Plot training curve ----
    print("\n--- Generating training_curve.png ---")
    plot_training_curve(
        monitor_csv=monitor_path + ".monitor.csv",
        out_path="training_curve.png"
    )

    # ---- 4. Evaluate ppo_v2 over 50 episodes ----
    print("\n--- Evaluating ppo_v2 over 50 episodes ---")
    eval_env_v2 = DriverEnv(pipeline_fn=None, max_steps=300)
    loaded_v2 = PPO.load("ppo_v2")
    records_v2 = evaluate_agent(loaded_v2, eval_env_v2, n_episodes=50)
    save_eval_results(records_v2, path="eval_results_v2.csv")
    eval_env_v2.close()

    # ---- 5. Load v1 results and compare ----
    print("\n--- v1 vs v2 action distribution comparison ---")

    # Re-read v1 records from CSV
    records_v1 = []
    if os.path.exists("eval_results.csv"):
        with open("eval_results.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records_v1.append({"action": int(row["action"])})
        action_distribution(records_v1, label="ppo_v1")
    else:
        print("  eval_results.csv not found — skipping v1 comparison")

    action_distribution(records_v2, label="ppo_v2")

    # ---- 6. Inference smoke-test ----
    eval_env_check = DriverEnv(pipeline_fn=None, max_steps=300)
    obs, _ = eval_env_check.reset(seed=0)
    action, _ = loaded_v2.predict(obs, deterministic=True)
    print(f"\nInference check -- obs shape: {obs.shape}, action: {int(action)}  (0/1/2)")
    eval_env_check.close()

    print("\nDay 3 complete.")


if __name__ == "__main__":
    if "--day2" in sys.argv:
        train()
    else:
        train_day3()
