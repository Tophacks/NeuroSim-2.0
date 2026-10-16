"""Reproducible held-out evaluations; output per-episode CSV and summary JSON."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from neurosim.env import RocketLandingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/rocket_ppo.zip")
    ap.add_argument("--planet", choices=["earth","mars","moon"], default="earth")
    ap.add_argument("--vehicle", choices=["falcon9_inspired","custom"], default="falcon9_inspired")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=10000,
                    help="Use a fixed evaluation seed range not used in training")
    ap.add_argument("--output", default="results/eval")
    args = ap.parse_args()
    model = PPO.load(args.model, device="auto")
    env = RocketLandingEnv(planet=args.planet, vehicle=args.vehicle)
    rows = []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_start + i)
        total = 0.0
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        rows.append({
            "seed": args.seed_start+i, "reward": total,
            "outcome": info["outcome"], "landing_error_m": info["landing_error_m"],
            "fuel_used_kg": info["fuel_used_kg"],
        })
    errors = [r["landing_error_m"] for r in rows if r["landing_error_m"] is not None]
    report = {
        "model": args.model, "planet": args.planet, "vehicle": args.vehicle,
        "episodes": len(rows), "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "success_rate": sum(r["outcome"] == "success" for r in rows)/len(rows),
        "crash_rate": sum(r["outcome"] == "crash" for r in rows)/len(rows),
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows)/len(rows),
        "mean_touchdown_error_m": float(np.mean(errors)) if errors else None,
        "mean_fuel_used_kg": float(np.mean([r["fuel_used_kg"] for r in rows])),
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
