import argparse
from pathlib import Path

import numpy as np
import torch

from ControlAllocator import ControlAllocator
from NeuralAllocator import NeuralAllocator
from ShipDynamics import ShipDynamics
from ThrusterModel import ThrusterModel
from allocation_output import save_comparison


DT = 0.05
N_STEPS = 240
ALLOW_SLACK = False
LIMIT_NEURAL_COMMANDS = True
INITIAL_STATE = np.zeros(6)
INITIAL_COMMAND = np.zeros(5)
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "neural_results" / "model.pt"
DEFAULT_OUTPUT = ROOT / "allocation_comparison_results"


def example_requests(time):
    desired = np.zeros((len(time), 3))
    desired[:, 0] = 100 * np.sin(0.5 * time)
    desired[:, 1] = 1000 * np.sin(0.25 * time)
    desired[:, 2] = 1000 * np.sin(0.5 * time)
    return desired


def main(output=DEFAULT_OUTPUT, checkpoint=DEFAULT_MODEL, limit_neural_commands=LIMIT_NEURAL_COMMANDS):
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint).expanduser()
    ship = ShipDynamics.gunnerus()
    thrusters = ThrusterModel.from_paper()
    nlp = ControlAllocator(thrusters)
    neural = NeuralAllocator(checkpoint, limit_commands=limit_neural_commands)
    if not np.isclose(neural.dt_s, DT, rtol=1e-8, atol=1e-12):
        raise ValueError(f"Simulation DT={DT} must match the trained network's dt={neural.dt_s}")
    if not np.allclose(neural.model.thrusters.positions, thrusters.positions,
                       rtol=0., atol=1e-9):
        raise ValueError("The neural checkpoint and simulated ship use different thruster geometry")
    neural.reset(INITIAL_COMMAND)

    times = np.arange(N_STEPS + 1) * DT
    desired = example_requests(times[:-1])
    states = {name: np.zeros((N_STEPS + 1, 6)) for name in ("direct", "nlp", "neural")}
    forces = {"direct": desired, "nlp": np.zeros_like(desired), "neural": np.zeros_like(desired)}
    commands = {name: np.zeros((N_STEPS, 5)) for name in ("nlp", "neural")}
    raw_neural_commands = np.zeros((N_STEPS, 5))
    for values in states.values():
        values[0] = INITIAL_STATE
    previous_command = INITIAL_COMMAND.copy()

    for step in range(N_STEPS):
        commands["nlp"][step] = nlp.allocate(
            desired[step], previous_command, DT, allow_slack=ALLOW_SLACK,
        )
        commands["neural"][step] = neural.allocate(desired[step])
        raw_neural_commands[step] = neural.last_raw_command
        previous_command = commands["nlp"][step].copy()

        # Apply the physical forces produced by each allocator's commands.
        for name in commands:
            forces[name][step] = thrusters.generalized_forces(commands[name][step])
        for name in states:
            states[name][step + 1] = ship.step(states[name][step], forces[name][step], DT)

    summary = save_comparison(
        output, times, states, forces, commands, INITIAL_COMMAND,
        checkpoint=checkpoint, allow_slack=ALLOW_SLACK,
        raw_neural_commands=raw_neural_commands,
        neural_limiter_enabled=limit_neural_commands,
    )
    for name in ("nlp", "neural"):
        result = summary[name]
        print(f"{name.upper()} force RMSE [sway N, surge N, yaw N m]:",
              np.array(result["force_rmse"]))
        print(f"{name.upper()} maximum position difference from direct (m):",
              result["max_position_error_m"])
        print(f"{name.upper()} rate violations [F1, F2, alpha2, F3, alpha3]:",
              result["rate_violation_samples_per_command"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--raw-neural", action="store_true", help="Disable neural magnitude/rate limiting")
    args = parser.parse_args()
    main(args.output, args.model, LIMIT_NEURAL_COMMANDS and not args.raw_neural)
