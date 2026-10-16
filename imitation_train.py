"""Behavioural cloning: initialize a PPO policy from successful manual flights.

This is imitation pretraining, not PPO learning. Follow with train.py --resume.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from neurosim.env import RocketLandingEnv


def load_samples(directory, planet, vehicle, include_failures=False):
    observations, actions = [], []
    episodes = 0
    for path in sorted(Path(directory).glob("manual_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2 or data.get("source") != "human":
            continue
        if data.get("planet") != planet or data.get("vehicle") != vehicle:
            continue
        if data.get("outcome") != "success" and not include_failures:
            continue
        steps = data.get("steps", [])
        for step in steps:
            observations.append(step["observation"])
            actions.append(step["action"])
        episodes += 1
    if not observations:
        raise SystemExit("No matching demonstration steps. Complete successful manual flights first, "
                         "or pass --include-failures for experimental use.")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demonstrations")
    ap.add_argument("--planet", choices=["earth", "mars", "moon"], default="earth")
    ap.add_argument("--vehicle", choices=["falcon9_inspired", "custom"], default="falcon9_inspired")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output", default="models/rocket_imitation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y, episodes = load_samples(args.data, args.planet, args.vehicle, args.include_failures)
    print(f"Loaded {len(x)} steps from {episodes} flights.")
    env = RocketLandingEnv(planet=args.planet, vehicle=args.vehicle)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu", seed=args.seed)
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    obs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
    actions = torch.as_tensor(y, dtype=torch.float32, device=model.device)
    # Maximize log probability of recorded continuous actions under PPO's
    # Gaussian actor. SB3's standard continuous action prediction clips to Box.
    for epoch in range(args.epochs):
        order = torch.randperm(len(obs), device=model.device)
        losses = []
        for idx in order.split(args.batch_size):
            _, log_prob, _ = model.policy.evaluate_actions(obs[idx], actions[idx])
            loss = -log_prob.mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            losses.append(float(loss.detach()))
        print(f"Imitation epoch {epoch+1}/{args.epochs}: NLL {np.mean(losses):.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"Saved imitation policy to {output}.zip")
    print("Evaluate with: python evaluate.py --model " + str(output) + ".zip")
    print("Then fine-tune with: python train.py --resume " + str(output) + ".zip --output models/rocket_finetuned")
    env.close()


if __name__ == "__main__":
    main()
