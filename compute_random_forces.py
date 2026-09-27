import argparse
from pathlib import Path

import numpy as np

from ThrusterModel import ThrusterModel
from signal_output import save_force_results


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "random_signal_results" / "signals.npz"
DEFAULT_OUTPUT = ROOT / "random_force_results"


def load_commands(input_file):
    """Read the generator's saved time samples and five command columns."""
    with np.load(Path(input_file).expanduser()) as data:
        time = np.asarray(data["time_s"], dtype=float)
        commands = np.asarray(data["commands"], dtype=float)

    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be an increasing array with at least two samples")
    if commands.shape != (len(time), 5):
        raise ValueError("commands must have five columns and one row per time sample")
    if not np.isfinite(time).all() or not np.isfinite(commands).all():
        raise ValueError("time_s and commands must contain finite values")
    return time, commands


def main(input_file=DEFAULT_INPUT, output=DEFAULT_OUTPUT):
    time, commands = load_commands(input_file)
    thrusters = ThrusterModel.from_paper()
    forces = thrusters.generalized_forces(commands)

    save_force_results(output, time, commands, forces, thrusters.positions)
    print(f"Converted {len(commands):,} commands from {input_file}")
    print(f"Saved forces and plots to {Path(output).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.input, args.output)
