"""NeuroSim 2.0 Mission Control: stylized Falcon 9-inspired 2D flight display.

Uses the real RocketLandingEnv state; no visual-only fake telemetry.
Manual episodes are recorded in the same JSON format as before.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame

from neurosim.env import RocketLandingEnv

W, H = 1400, 820
LEFT, RIGHT = 245, 1085
GROUND = 730
NAVY = (9, 16, 28)
PANEL = (15, 25, 40)
EDGE = (38, 59, 79)
TEXT = (224, 234, 244)
MUTED = (140, 164, 187)
BLUE = (38, 145, 235)
GREEN = (64, 205, 152)
AMBER = (248, 177, 68)


def label(screen, font, message, pos, color=TEXT):
    screen.blit(font.render(str(message), True, color), pos)


def rotated_polygon(screen, color, cx, cy, points, angle):
    # Local +y points down, positive simulation angle leans right.
    ca, sa = math.cos(angle), math.sin(angle)
    transformed = [(int(cx + x*ca + y*sa), int(cy - x*sa + y*ca)) for x, y in points]
    pygame.draw.polygon(screen, color, transformed)
    return transformed


def rocket_sprite(screen, x, y, height, theta, throttle, rcs, vehicle):
    """Procedural Falcon-9-inspired silhouette with fins, legs, engine and RCS plumes."""
    height = max(50, int(height))
    width = max(12, int(height * 0.095))
    # Local coordinates relative to center.
    body = [(-width/2, -height*.44), (-width*.36, -height*.49),
            (width*.36, -height*.49), (width/2, -height*.44),
            (width/2, height*.46), (-width/2, height*.46)]
    rotated_polygon(screen, (226, 230, 229), x, y, body, theta)
    rotated_polygon(screen, (36, 43, 50), x, y,
                    [(-width/2, height*.23), (width/2, height*.23),
                     (width/2, height*.46), (-width/2, height*.46)], theta)
    rotated_polygon(screen, (27, 38, 48), x, y,
                    [(-width*.50, -height*.31), (-width*1.3, -height*.39),
                     (-width*1.25, -height*.19), (-width*.50, -height*.17)], theta)
    rotated_polygon(screen, (27, 38, 48), x, y,
                    [(width*.50, -height*.31), (width*1.3, -height*.39),
                     (width*1.25, -height*.19), (width*.50, -height*.17)], theta)
    # Landing legs
    for side in (-1, 1):
        rotated_polygon(screen, (141, 151, 159), x, y,
                        [(side*width*.4, height*.18), (side*width*.62, height*.17),
                         (side*width*1.6, height*.51), (side*width*1.45, height*.53)], theta)
    # Body detail / decorative identification (only drawn at adequate scale).
    if height > 85:
        rotated_polygon(screen, (52, 65, 76), x, y,
                        [(-width*.42, -height*.08), (width*.42, -height*.08),
                         (width*.42, -height*.055), (-width*.42, -height*.055)], theta)
    if throttle > .015:
        flame_length = height*(.14 + .30*throttle)
        rotated_polygon(screen, (255, 139, 45), x, y,
                        [(-width*.24, height*.46), (width*.24, height*.46),
                         (width*.10, height*.46+flame_length*.72),
                         (0, height*.46+flame_length),
                         (-width*.10, height*.46+flame_length*.72)], theta)
        rotated_polygon(screen, (255, 225, 146), x, y,
                        [(-width*.10, height*.46), (width*.10, height*.46),
                         (0, height*.46+flame_length*.65)], theta)
    if abs(rcs) > .01:
        # Opposed gas jets indicate commanded attitude torque.
        side = 1 if rcs > 0 else -1
        rotated_polygon(screen, (115, 208, 245), x, y,
                        [(side*width*.55, -height*.30),
                         (side*(width*.55 + 10 + 16*abs(rcs)), -height*.35),
                         (side*width*.55, -height*.26)], theta)
        rotated_polygon(screen, (115, 208, 245), x, y,
                        [(-side*width*.55, height*.29),
                         (-side*(width*.55 + 10 + 16*abs(rcs)), height*.34),
                         (-side*width*.55, height*.25)], theta)


def bar(screen, x, y, width, value, color):
    pygame.draw.rect(screen, EDGE, (x, y, width, 7), border_radius=3)
    pygame.draw.rect(screen, color, (x, y, int(width*max(0, min(1, value))), 7), border_radius=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manual", "ai"], default="manual")
    ap.add_argument("--model", default="models/rocket_ppo.zip")
    ap.add_argument("--planet", choices=["earth", "mars", "moon"], default="earth")
    ap.add_argument("--vehicle", choices=["falcon9_inspired", "custom"], default="falcon9_inspired")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record-dir", default="demonstrations")
    args = ap.parse_args()

    model = None
    if args.mode == "ai":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        if model.action_space.shape != (3,):
            raise SystemExit("This checkpoint uses the older 2-control action space. "
                             "Retrain with the new 3-control RCS environment.")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("NeuroSim 2.0 — Mission Control")
    title_font = pygame.font.SysFont("segoeui", 26, bold=True)
    font = pygame.font.SysFont("consolas", 17)
    tiny = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    env = RocketLandingEnv(planet=args.planet, vehicle=args.vehicle)
    obs, _ = env.reset(seed=args.seed)
    throttle, gimbal, rcs = 0., 0., 0.
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
        payload = {"schema_version": 2, "source": "human", "planet": args.planet,
                   "vehicle": args.vehicle, "seed": episode_seed, "dt": env.dt,
                   "outcome": outcome, "steps": episode}
        destination.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
        last_file = destination.name
        episode = []

    def restart():
        nonlocal obs, throttle, gimbal, rcs, outcome, paused, episode_seed
        save_episode()
        episode_seed = None
        obs, _ = env.reset()
        throttle, gimbal, rcs = 0., 0., 0.
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
                elif event.key == pygame.K_F5:
                    save_episode()

        keys = pygame.key.get_pressed()
        if not paused and outcome == "flying":
            if args.mode == "manual":
                throttle += .025 * (bool(keys[pygame.K_UP] or keys[pygame.K_w])
                                    - bool(keys[pygame.K_DOWN] or keys[pygame.K_s]))
                throttle = float(np.clip(throttle, 0, 1))
                gimbal += .045 * (bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])
                                  - bool(keys[pygame.K_LEFT] or keys[pygame.K_a]))
                if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or
                        keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    gimbal *= .76
                gimbal = float(np.clip(gimbal, -1, 1))
                # Q/E: cold-gas RCS jets. Return to zero when released.
                rcs = float(bool(keys[pygame.K_e]) - bool(keys[pygame.K_q]))
                action = np.array([throttle, gimbal, rcs], dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
                throttle, gimbal, rcs = map(float, action)
            before = obs.copy()
            obs, reward, done, truncated, info = env.step(action)
            if args.mode == "manual":
                episode.append({"observation": before.tolist(), "action": np.asarray(action).tolist(),
                                "reward": float(reward), "next_observation": obs.tolist(),
                                "terminated": bool(done), "truncated": bool(truncated)})
            if done or truncated:
                outcome = info["outcome"]
                save_episode()

        screen.fill(NAVY)
        pygame.draw.rect(screen, PANEL, (0, 0, LEFT, H))
        pygame.draw.rect(screen, PANEL, (RIGHT, 0, W-RIGHT, H))
        pygame.draw.line(screen, EDGE, (LEFT, 0), (LEFT, H), 2)
        pygame.draw.line(screen, EDGE, (RIGHT, 0), (RIGHT, H), 2)
        label(screen, title_font, "NeuroSim 2.0", (16, 20))
        label(screen, tiny, "Autonomous Landing Simulator", (16, 59), MUTED)
        pygame.draw.rect(screen, BLUE, (15, 95, 100, 35), border_radius=5)
        label(screen, font, args.mode.upper(), (27, 103))
        label(screen, font, "VEHICLE", (16, 157), MUTED)
        label(screen, font, "Falcon 9-inspired" if args.vehicle == "falcon9_inspired" else "Custom booster", (16, 190))
        rocket_sprite(screen, 120, 333, 210, 0, .3, 0, args.vehicle)
        for i, line in enumerate([f"Height   {env.vehicle.height:.1f} m",
                                  f"Dry mass {env.vehicle.dry_mass:,.0f} kg",
                                  f"Fuel     {env.vehicle.fuel_mass:,.0f} kg",
                                  f"Thrust   {env.vehicle.max_thrust/1000:,.0f} kN",
                                  "RCS      2D attitude jets"]):
            label(screen, tiny, line, (16, 465+i*27), MUTED)
        pygame.draw.line(screen, EDGE, (16, 620), (225, 620))
        label(screen, font, "ENVIRONMENT", (16, 636), MUTED)
        label(screen, font, args.planet.upper(), (16, 671))
        label(screen, tiny, f"Gravity: {env.planet.gravity:.2f} m/s2", (16, 708), MUTED)

        # World: descending camera follows rocket once it enters the lower altitude range.
        world_left = LEFT+12
        world_right = RIGHT-12
        world_mid = (world_left+world_right)//2
        x, z, vx, vz, theta, omega, fuel = env.state
        scale = 2.7
        rocket_h = max(48, env.vehicle.height*scale)
        ground_y = GROUND
        rocket_x = world_mid + x*scale
        rocket_y = ground_y - z*scale
        # Follow-camera keeps the rocket visible during higher-altitude approach.
        if rocket_y < 105:
            camera_offset = 105 - rocket_y
            rocket_y += camera_offset
            ground_y += camera_offset
        for i in range(34):
            sx = world_left + ((i*193 + 57) % (world_right-world_left))
            sy = 40 + ((i*113 + 31) % 590)
            pygame.draw.circle(screen, (55, 79, 104), (sx, sy), 1)
        if ground_y < H:
            pygame.draw.rect(screen, (33, 49, 56), (world_left, ground_y, world_right-world_left, H-ground_y))
            pygame.draw.rect(screen, (44, 72, 83), (world_mid-35, ground_y-5, 70, 5))
            pygame.draw.ellipse(screen, AMBER, (world_mid-55, ground_y-14, 110, 24), 2)
        rocket_sprite(screen, rocket_x, rocket_y, rocket_h, theta, throttle, rcs, args.vehicle)
        label(screen, tiny, f"{args.vehicle.upper()}  |  {args.planet.upper()}  |  {args.mode.upper()}", (LEFT+23, 24), TEXT)
        label(screen, tiny, f"T+ {env.steps*env.dt:06.1f}s", (RIGHT-142, 24), GREEN)
        if paused:
            label(screen, title_font, "PAUSED", (world_mid-48, 84), AMBER)
        elif outcome != "flying":
            label(screen, title_font, outcome.upper(), (world_mid-70, 84),
                  GREEN if outcome == "success" else AMBER)

        label(screen, font, "TELEMETRY", (RIGHT+18, 30))
        values = [
            ("Altitude", f"{z:.1f} m", z/200),
            ("Vertical speed", f"{vz:.2f} m/s", abs(vz)/40),
            ("Horizontal speed", f"{vx:.2f} m/s", abs(vx)/20),
            ("Pitch", f"{math.degrees(theta):.2f} deg", abs(theta)/.5),
            ("Throttle", f"{throttle*100:.0f} %", throttle),
            ("Gimbal", f"{gimbal:.2f}", abs(gimbal)),
            ("RCS thrusters", f"{rcs:+.0f}", abs(rcs)),
            ("Fuel", f"{fuel:.0f} kg", fuel/max(1, env.vehicle.fuel_mass)),
            ("Landing offset", f"{abs(x):.1f} m", abs(x)/100),
        ]
        for i, (name, value, fraction) in enumerate(values):
            yy = 82+i*67
            label(screen, tiny, name, (RIGHT+18, yy), MUTED)
            label(screen, tiny, value, (RIGHT+188, yy))
            bar(screen, RIGHT+18, yy+27, 268, fraction,
                GREEN if name in ("Fuel", "RCS thrusters") else BLUE)
        label(screen, font, "ATTITUDE", (RIGHT+18, 705), MUTED)
        label(screen, tiny, f"omega: {math.degrees(omega):+.1f} deg/s", (RIGHT+18, 737))
        label(screen, tiny, f"Status: {outcome}", (RIGHT+18, 768), GREEN if outcome=="success" else TEXT)

        pygame.draw.rect(screen, (9, 18, 31), (LEFT+12, H-52, RIGHT-LEFT-24, 40), border_radius=4)
        label(screen, tiny, "W/S throttle   A/D gimbal   Q/E side jets   SPACE pause   R reset   F5 save   ESC exit",
              (LEFT+21, H-40), TEXT)
        if last_file:
            label(screen, tiny, "Recorded: "+last_file, (LEFT+22, H-73), GREEN)
        pygame.display.flip()
        clock.tick(round(1/env.dt))

    save_episode()
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
