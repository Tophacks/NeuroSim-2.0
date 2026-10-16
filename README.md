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


## Interactive flight viewer and manual demonstrations

Install the updated dependencies and run from the repository root:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe viewer.py --mode manual --vehicle falcon9_inspired --planet earth
```

Manual flight: Up/W increases throttle; Down/S decreases it; Left/A and Right/D gimbal the engine. Space pauses, R resets the flight, F5 saves the current recording, and Esc exits. Completed flights are automatically saved as JSON files in `demonstrations/` with state/action/reward/next-state records and the final outcome. Partial recordings can be saved with F5 or Esc, but are not marked successful.

```powershell
# Pretrain a PPO-compatible actor by imitating successful manual flights
.\.venv\Scripts\python.exe imitation_train.py --epochs 30 --output models/rocket_imitation

# Evaluate the imitation-only policy
.\.venv\Scripts\python.exe evaluate.py --model models/rocket_imitation.zip --episodes 100

# Continue improving via PPO (saves a NEW model; does not overwrite imitation model)
.\.venv\Scripts\python.exe train.py --resume models/rocket_imitation.zip --steps 200000 --output models/rocket_finetuned

# Watch the trained agent
.\.venv\Scripts\python.exe viewer.py --mode ai --model models/rocket_finetuned.zip
```

Only successful flights matching the selected planet and vehicle are used for imitation by default. For exploratory training on all manual actions, pass `--include-failures` (this may teach bad actions). The viewer uses the simplified 2D physics environment, not a flight-qualified rocket model. Behavioural cloning provides an initialization, not a guarantee of landing. Compare performance with a fresh PPO baseline on the same held-out evaluation seeds. Manual control is intentionally challenging; a hover/landing controller or easier curriculum may be needed to collect successful demonstrations.


## Falcon 9-inspired Mission Control and side thrusters

The upgraded viewer draws a stylized white/black booster with a nose profile, grid fins,
landing legs, engine plume, and opposing reaction-control thruster jets. The right
panel shows live telemetry. The model is a **2D educational approximation**, not an
accurate SpaceX vehicle or a photorealistic 3D simulator.

Controls: **W/S or Up/Down** throttle, **A/D or Left/Right** engine gimbal,
**Q/E** side thrusters (opposite attitude torque), **Space** pause, **R** reset,
**F5** save partial flight, **Esc** quit. Completed manual flights auto-save.

The action vector is now `[throttle, gimbal, rcs]` (3 continuous controls).
**Older 2-action NeuroSim 2.0 checkpoints cannot be loaded** into this version:
retrain PPO and recollect manual demonstrations with the updated viewer.
Imitation training uses schema-version-2 demonstration files. Earlier
2-action recordings are ignored to avoid training with incompatible actions.

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe viewer.py --mode manual
```
