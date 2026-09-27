import casadi as ca
import numpy as np

from actuator_limits import (
    COMMAND_MIN,
    COMMAND_MAX,
    COMMAND_TOLERANCE,
    MAX_RATES,
    FORBIDDEN_ANGLES,
    in_forbidden_sector,
)


FORCE_SCALE = 30000
FORCE_ERROR_WEIGHT = 1e6
ANGLE_CHANGE_WEIGHT = 0.01
FORCE_TOLERANCE = np.array([1e-3, 1e-3, 1e-2])
IPOPT_OPTIONS = {
    "print_level": 0,
    "sb": "yes",
    "tol": 1e-9,
    "constr_viol_tol": 1e-10,
    "bound_relax_factor": 0.0,
    "max_iter": 300,
}


class ControlAllocator:
    def __init__(self, thrusters):
        self.thrusters = thrusters

    def allocate(self, tau_desired, previous_command, dt, *, allow_slack=False):
        tau_desired = np.asarray(tau_desired, dtype=float)
        previous = np.asarray(previous_command, dtype=float)

        if tau_desired.shape != (3,) or not np.isfinite(tau_desired).all():
            raise ValueError("tau_desired must be [sway, surge, yaw]")
        if previous.shape != (5,) or not np.isfinite(previous).all():
            raise ValueError("previous_command must be [F1, F2, alpha2, F3, alpha3]")
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive number of seconds")
        if not isinstance(allow_slack, (bool, np.bool_)):
            raise ValueError("allow_slack must be True or False")

        if (np.any(previous < COMMAND_MIN - COMMAND_TOLERANCE)
                or np.any(previous > COMMAND_MAX + COMMAND_TOLERANCE)
                or in_forbidden_sector(previous[[2, 4]], tolerance=1e-8).any()):
            raise ValueError("previous_command violates a thruster limit")

        opti = ca.Opti()
        command = opti.variable(5)
        F1, F2, alpha2, F3, alpha3 = ca.vertsplit(command)
        opti.set_linear_scale(command, [FORCE_SCALE, FORCE_SCALE, 1, FORCE_SCALE, 1])

        tau = self.thrusters.symbolic_generalized_forces(command)

        opti.subject_to(opti.bounded(COMMAND_MIN, command, COMMAND_MAX))
        max_change = MAX_RATES * dt
        opti.subject_to(opti.bounded(-max_change, command - previous, max_change))

        # These products exclude the open intervals (80, 100) and (-100, -80).
        angle80, angle100 = FORBIDDEN_ANGLES
        for angle in (alpha2, alpha3):
            opti.subject_to((angle - angle80) * (angle - angle100) >= 0)
            opti.subject_to((angle + angle100) * (angle + angle80) >= 0)

        # Scale moments by a lever arm so all three tracking errors are comparable.
        length = max(1.0, np.linalg.norm(self.thrusters.positions, axis=1).max())
        scale = ca.DM([FORCE_SCALE, FORCE_SCALE, FORCE_SCALE * length])
        force_error = (tau_desired - tau) / scale
        forces = ca.vertcat(F1, F2, F3) / FORCE_SCALE
        power = ca.sum1((forces**2 + 1e-12)**0.75)  # smooth |F|^(3/2) near zero
        angle_change = (alpha2 - previous[2])**2 + (alpha3 - previous[4])**2

        opti.minimize(FORCE_ERROR_WEIGHT * ca.sumsqr(force_error)
                      + power + ANGLE_CHANGE_WEIGHT * angle_change)
        if not allow_slack:
            opti.subject_to(force_error == 0)

        opti.set_initial(command, previous)
        opti.solver("ipopt", {"print_time": False}, IPOPT_OPTIONS)
        try:
            solution = opti.solve()
        except RuntimeError as error:
            raise RuntimeError(
                "Allocation failed: request may be unreachable, or IPOPT failed locally."
            ) from error
        answer = np.asarray(solution.value(command)).reshape(5)

        lower = np.maximum(COMMAND_MIN, previous - max_change)
        upper = np.minimum(COMMAND_MAX, previous + max_change)
        if (not np.isfinite(answer).all()
                or np.any(answer < lower - COMMAND_TOLERANCE)
                or np.any(answer > upper + COMMAND_TOLERANCE)
                or in_forbidden_sector(answer[[2, 4]], tolerance=1e-8).any()):
            raise RuntimeError("Returned command violates a thruster constraint")
        actual = self.thrusters.generalized_forces(answer)
        if not allow_slack and np.any(np.abs(tau_desired - actual) > FORCE_TOLERANCE):
            raise RuntimeError("Returned command does not match the requested force")
        return answer
