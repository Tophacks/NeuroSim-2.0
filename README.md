# NeuroSim 2.0 🚀

A research-oriented **2D vertical rocket landing simulator** for reinforcement learning, based on a *fictional Falcon 9-inspired booster* or an editable custom vehicle. This is not flight software or a validated model of any SpaceX vehicle.

## Goals
- Planet presets: Earth, Mars, Moon (gravity and simplified atmospheric density).
- Configurable booster: mass, fuel, thrust, geometry, drag and control limits.
- Continuous actions: main-engine throttle and gimbal.
- Closed-loop PPO training and repeatable, held-out episode evaluation.
- Future: demonstrations from [NeuroSim 1.0](https://github.com/Tophacks/NeuroSim), failure replay, training feedback, mission-control dashboard.

## Setup (Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --steps 200000
.\.venv\Scripts\python.exe evaluate.py --model models/rocket_ppo.zip --episodes 100
```
If `py` is not recognized, use the full path to your existing Python 3.11 executable.

## Vehicles
- `falcon9_inspired`: simplified **fictional single-engine landing** booster. It does **not** use proprietary engine data, control laws, or full Falcon 9 dynamics.
- `custom`: sample smaller research booster; edit values in `neurosim/config.py`.

Select with `--vehicle falcon9_inspired|custom` and `--planet earth|mars|moon`.

## Limitations
This is an intentionally simplified 2D point-mass + pitch model. Approximate control torque, aerodynamic drag, and terrain contact are for testing learning pipelines, **not for designing or flying rockets**. The original LunarLander PPO weights cannot load directly because the action/observation spaces differ. Transfer of demonstrations will be a separate, explicitly tested experiment.

## Validation
Evaluate on held-out random seeds. Compare landing rate, landing error, crash count, fuel usage, and average reward, not only training reward. Preserve frozen baseline results before retraining.
