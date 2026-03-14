# PPO Hyperparameter Notes — SmartRoad AI

Reference for Day 2 tuning when StubDriverEnv is swapped for real DriverEnv.

---

## Key Parameters

### `learning_rate` (default `3e-4`)
Controls the step size of each gradient update.
- **Day 1 stub value:** `3e-4` — fine for random env warm-up.
- **Day 2 suggestion:** Drop to `1e-4` once real env is connected.  Slower learning
  stabilises training when reward variance is high (mixed phone/no-phone frames).

### `n_steps` (default `2048`, our Day 1: `512`)
Number of steps collected per environment before each update.
- Smaller `n_steps` = more frequent updates, noisier gradient estimates.
- **Day 2 suggestion:** Try `1024` to balance sample efficiency vs. stability.
- If reward curve is spiky, increase to `2048`.

### `batch_size` (default `64`)
Mini-batch size drawn from the collected rollout.
- Must be ≤ `n_steps`.  Our `n_steps=512`, `batch_size=64` → 8 mini-batches per update.
- **Day 2 suggestion:** `128` with `n_steps=1024` keeps the ratio the same.

### `n_epochs` (default `10`)
How many passes over the collected rollout each update cycle.
- More epochs = stronger signal extraction but risk of over-fitting old data.
- **Keep at `10`** — standard safe default for most tasks.

### `clip_range` (default `0.2`)
PPO's clipping parameter — limits how much the policy can change per update.
- Larger value = bolder updates.  Smaller = more conservative.
- **Day 2 suggestion:** Start at `0.2`.  If policy collapses (all same action), try `0.1`.

### `ent_coef` (default `0.0`)
Entropy coefficient — encourages exploration by penalising deterministic policies.
- **Day 2 suggestion:** `0.01` to keep some exploration early in real training.
  If agent gets stuck outputting only ALL_CLEAR, raise to `0.05`.

---

## Day 2 Recommended Config

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
)
model.learn(total_timesteps=10_000)
model.save("ppo_v1")
```

## Day 3 Extended Training Config

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.005,
)
model.learn(total_timesteps=50_000)
model.save("ppo_v2")
```

---

## Diagnosing Training Problems

| Symptom | Likely cause | Fix |
|---|---|---|
| Reward stays at ~-2 every step | Agent always misses violations | Raise `ent_coef` to 0.05 |
| Reward spiky, no trend | `n_steps` too small | Increase to 2048 |
| Policy collapses to one action | `clip_range` too large | Lower to 0.1 |
| Training very slow | `batch_size` too small | Raise to 256 |

---

## Monitoring with SB3 Monitor Wrapper

Wrap the env before passing to PPO to get automatic reward logging:

```python
from stable_baselines3.common.monitor import Monitor

env = Monitor(DriverEnv(), filename="monitor_log")
model = PPO("MlpPolicy", env, ...)
```

After training, `monitor_log.monitor.csv` contains per-episode reward/length.
Use this to generate `training_curve.png` on Day 3.
