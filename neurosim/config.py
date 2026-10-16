"""Illustrative physical presets; not real vehicle engineering specifications."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Planet:
    name: str
    gravity: float
    air_density: float


PLANETS = {
    "earth": Planet("earth", 9.81, 1.225),
    "mars": Planet("mars", 3.71, 0.020),
    "moon": Planet("moon", 1.62, 0.0),
}


@dataclass(frozen=True)
class Vehicle:
    name: str
    dry_mass: float
    fuel_mass: float
    max_thrust: float
    burn_rate: float  # kg/s at full throttle
    height: float
    diameter: float
    max_gimbal_deg: float
    drag_coefficient: float


VEHICLES = {
    "falcon9_inspired": Vehicle(
        "falcon9_inspired", 22000, 8500, 750000, 210, 35.0, 3.6, 8.0, 0.50
    ),
    "custom": Vehicle("custom", 500, 200, 15000, 5.0, 8.0, 1.2, 10.0, 0.55),
}


def get_vehicle(name: str, **overrides) -> Vehicle:
    return replace(VEHICLES[name], **overrides)
