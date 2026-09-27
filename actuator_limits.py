import numpy as np


COMMAND_NAMES = ["F1", "F2", "alpha2", "F3", "alpha3"]
COMMAND_MIN = np.array([-30000., -30000., -np.pi, -30000., -np.pi])
COMMAND_MAX = np.array([30000., 60000., np.pi, 60000., np.pi])
MAX_RATES = np.array([1000., 1000., np.deg2rad(10.), 1000., np.deg2rad(10.)])
FORBIDDEN_ANGLES = np.deg2rad([80., 100.])
FORBIDDEN_SECTORS = np.array([-FORBIDDEN_ANGLES[::-1], FORBIDDEN_ANGLES])
COMMAND_TOLERANCE = np.array([1e-3, 1e-3, 1e-8, 1e-3, 1e-8])


def in_forbidden_sector(angles, tolerance=0.0):
    angles = np.abs(np.asarray(angles))
    lower, upper = FORBIDDEN_ANGLES
    return (angles > lower + tolerance) & (angles < upper - tolerance)


def limit_command(network_command, previous_command, dt):
    network_command = np.asarray(network_command, dtype=float)
    previous_command = np.asarray(previous_command, dtype=float)
    for name, command in (("network_command", network_command),
                          ("previous_command", previous_command)):
        if command.shape != (5,) or not np.isfinite(command).all():
            raise ValueError(f"{name} must contain five finite values in N and radians")
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive seconds")
    if np.any(previous_command < COMMAND_MIN) or np.any(previous_command > COMMAND_MAX):
        raise ValueError("previous_command must be within the actuator magnitude bounds")

    lower = np.maximum(COMMAND_MIN, previous_command - MAX_RATES * dt)
    upper = np.minimum(COMMAND_MAX, previous_command + MAX_RATES * dt)
    return np.clip(network_command, lower, upper)
