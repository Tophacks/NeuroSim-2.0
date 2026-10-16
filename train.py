"""Train a new NeuroSim 2.0 rocket policy."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from neurosim.env import RocketLandingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--planet", choices=["earth","mars","moon"], default="earth")
    ap.add_argument("--vehicle", choices=["falcon9_inspired","custom"], default="falcon9_inspired")
    ap.add_argument("--resume", default=None, help="Existing PPO .zip with matching spaces")
    ap.add_argument("--output", default="models/rocket_ppo")
    args = ap.parse_args()
    env = Monitor(RocketLandingEnv(planet=args.planet, vehicle=args.vehicle))
    if args.resume:
        model = PPO.load(args.resume, env=env, device="auto")
    else:
        model = PPO("MlpPolicy", env, verbose=1, device="auto",
                    n_steps=2048, batch_size=64, tensorboard_log=None)
    Path("checkpoints").mkdir(exist_ok=True)
    cb = CheckpointCallback(save_freq=50_000, save_path="checkpoints", name_prefix="rocket")
    model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume), callback=cb)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved policy to {args.output}.zip")
    env.close()


if __name__ == "__main__":
    main()
