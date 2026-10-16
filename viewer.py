"""Interactive 2D NeuroSim flight viewer. Manual demonstrations use JSONL."""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame

from neurosim.env import RocketLandingEnv

WIDTH, HEIGHT = 1120, 760
SKY = (13, 20, 35)
WHITE = (226, 235, 245)
MUTED = (143, 164, 186)
PAD = (85, 201, 168)


def main():
    parser = argparse.ArgumentParser(description="Manual or PPO-controlled rocket viewer")
    parser.add_argument("--mode", choices=["manual", "ai"], default="manual")
    parser.add_argument("--model", default="models/rocket_ppo.zip")
    parser.add_argument("--planet", choices=["earth", "mars", "moon"], default="earth")
    parser.add_argument("--vehicle", choices=["falcon9_inspired", "custom"], default="falcon9_inspired")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-dir", default="demonstrations")
    args = parser.parse_args()

    model = None
    if args.mode == "ai":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroSim 2.0 | Mission Control")
    font = pygame.font.SysFont("consolas", 21)
    small = pygame.font.SysFont("consolas", 17)
    clock = pygame.time.Clock()
    env = RocketLandingEnv(planet=args.planet, vehicle=args.vehicle)
    obs, _ = env.reset(seed=args.seed)
    throttle = 0.0
    gimbal = 0.0
    episode = []
    outcome = "flying"
    paused = False
    last_file = ""
    episode_seed = args.seed

    def save_episode():
        nonlocal episode, last_file
        if args.mode != "manual" or not episode:
            return
        directory = Path(args.record_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        destination = directory / f"manual_{stamp}.json"
        payload = {
            "schema_version": 1, "source": "human",
            "planet": args.planet, "vehicle": args.vehicle,
            "seed": episode_seed, "dt": env.dt, "outcome": outcome,
            "steps": episode,
        }
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        last_file = str(destination)
        episode = []

    def restart():
        nonlocal obs, throttle, gimbal, outcome, paused, episode_seed
        save_episode()
        episode_seed = None  # new randomized initial state
        obs, _ = env.reset()
        throttle, gimbal = 0.0, 0.0
        outcome, paused = "flying", False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    restart()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_F5 and args.mode == "manual":
                    save_episode()
        keys = pygame.key.get_pressed()
        if not paused and outcome == "flying":
            if args.mode == "manual":
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    throttle += 0.025
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    throttle -= 0.025
                throttle = float(np.clip(throttle, 0, 1))
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    gimbal -= 0.045
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    gimbal += 0.045
                if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    gimbal *= 0.78
                gimbal = float(np.clip(gimbal, -1, 1))
                action = np.array([throttle, gimbal], dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
                throttle, gimbal = map(float, action)
            before = obs.copy()
            obs, reward, done, truncated, info = env.step(action)
            if args.mode == "manual":
                episode.append({"observation": before.tolist(), "action": np.asarray(action).tolist(),
                                "reward": float(reward), "next_observation": obs.tolist(),
                                "terminated": bool(done), "truncated": bool(truncated)})
            if done or truncated:
                outcome = info["outcome"]
                save_episode()

        screen.fill(SKY)
        # Ground and landing pad. Scale focuses on low-altitude landing phase.
        ground_y = HEIGHT - 85
        center_x = 390
        scale = 2.5
        pygame.draw.rect(screen, (43, 60, 70), (0, ground_y, 785, 85))
        pygame.draw.rect(screen, PAD, (center_x - 8 * scale, ground_y - 5, 16 * scale, 6))
        pygame.draw.line(screen, (80, 108, 129), (785, 0), (785, HEIGHT), 2)
        x, z, vx, vz, theta, omega, fuel = env.state
        px = center_x + x * scale
        # Rocket center at z = vehicle.height/2 on touchdown.
        py = ground_y - z * scale
        length = max(35, min(90, env.vehicle.height * scale))
        width = 14
        direction = pygame.Vector2(math.sin(theta), -math.cos(theta))
        sideways = pygame.Vector2(direction.y, -direction.x)
        nose = pygame.Vector2(px, py) + direction * length / 2
        tail = pygame.Vector2(px, py) - direction * length / 2
        pts = [nose, tail + sideways * width/2, tail - sideways * width/2]
        pygame.draw.polygon(screen, (218, 226, 234), pts)
        pygame.draw.lines(screen, (86, 143, 178), True, pts, 2)
        if throttle > 0.02 and outcome == "flying":
            flame = tail - direction * (12 + 35 * throttle)
            pygame.draw.polygon(screen, (255, 171, 58),
                                [tail + sideways*5, tail - sideways*5, flame])
        pygame.draw.line(screen, (79, 91, 109), (center_x, 0), (center_x, ground_y), 1)
        panel_x = 810
        lines = [
            "NEUROSIM 2.0", f"MODE     {args.mode.upper()}",
            f"PLANET   {args.planet.upper()}", f"VEHICLE  {args.vehicle}",
            "", f"ALTITUDE {z:8.1f} m", f"X OFFSET {x:8.1f} m",
            f"VERT SPD {vz:8.2f} m/s", f"HORIZ SPD{vx:8.2f} m/s",
            f"PITCH    {math.degrees(theta):8.2f} deg",
            f"FUEL     {fuel:8.1f} kg",
            f"THROTTLE {throttle*100:8.1f} %",
            f"GIMBAL   {gimbal:8.2f}",
            f"STEP     {env.steps:8d}", "", f"STATUS   {outcome.upper()}",
            "PAUSED" if paused else "",
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, PAD if i in (0, 15) else WHITE), (panel_x, 26 + i*33))
        instructions = "UP/DOWN: throttle | LEFT/RIGHT: gimbal | F5: save | SPACE: pause | R: reset | ESC: quit"
        if args.mode == "ai":
            instructions = "AI flight | SPACE: pause | R: new flight | ESC: quit"
        screen.blit(small.render(instructions, True, WHITE), (18, HEIGHT - 35))
        if last_file:
            screen.blit(small.render("Saved: " + Path(last_file).name, True, MUTED), (18, 18))
        pygame.display.flip()
        clock.tick(round(1 / env.dt))
    save_episode()
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
