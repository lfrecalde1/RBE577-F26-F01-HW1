import argparse
from pathlib import Path

import numpy as np

from actuator_limits import COMMAND_NAMES, MAX_RATES


N_SAMPLES = 100_000
DT = 0.05
SEED = 577
TARGET_BINS = 20
MIN_SPEED_FRACTION = 0.35
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "random_signal_results"

LOWER = np.array([-10000.0, -5000.0, -np.pi, -5000.0, -np.pi])
UPPER = np.array([10000.0, 5000.0, np.pi, 5000.0, np.pi])


def _random_targets(rng, lower, upper, bins=TARGET_BINS):
    while True:
        fractions = (np.arange(bins) + rng.random(bins)) / bins
        targets = np.concatenate(([lower, upper], lower + fractions * (upper - lower)))
        yield from rng.permutation(targets)


def _smooth_channel(n_samples, dt, lower, upper, max_rate, rng):
    signal = np.empty(n_samples)
    signal[0] = 0.0
    index = 0

    for target in _random_targets(rng, lower, upper):
        start = signal[index]
        peak_rate = rng.uniform(MIN_SPEED_FRACTION, 1.0) * max_rate

        required_steps = (15 / 8) * abs(target - start) / (peak_rate * dt)
        steps = max(1, int(np.ceil(required_steps)))
        count = min(steps, n_samples - 1 - index)
        progress = np.arange(1, count + 1, dtype=float) / steps
        blend = progress**3 * (10 + progress * (-15 + 6 * progress))
        signal[index + 1:index + count + 1] = start + (target - start) * blend
        index += count

        if count == steps:
            signal[index] = target  # Keep exact endpoints despite roundoff.
        if index == n_samples - 1:
            break

    return signal


def generate_commands(n_samples=N_SAMPLES, dt=DT, seed=SEED):
    if (isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer)) or n_samples < 2):
        raise ValueError("n_samples must be an integer of at least 2")
    if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be a finite positive number of seconds")

    streams = np.random.SeedSequence(seed).spawn(len(COMMAND_NAMES))
    commands = np.empty((n_samples, len(COMMAND_NAMES)))
    for column, stream in enumerate(streams):
        commands[:, column] = _smooth_channel(
            n_samples, dt, LOWER[column], UPPER[column], MAX_RATES[column],
            np.random.default_rng(stream),
        )
    return commands


def main(n_samples=N_SAMPLES, dt=DT, seed=SEED, output=DEFAULT_OUTPUT):
    from signal_output import save_random_commands

    commands = generate_commands(n_samples, dt, seed)
    summary = save_random_commands(output, commands, dt, seed, LOWER, UPPER, MAX_RATES)

    print(f"Saved {n_samples:,} samples to {Path(output).resolve()}")
    for name, stats in summary["channels"].items():
        print(f"{name}: {stats['observed_min']:.3f} to {stats['observed_max']:.3f} "
              f"{stats['unit']}; peak rate {stats['max_abs_rate_per_s']:.3f} "
              f"{stats['unit']}/s")
    print("Angles cover the full range, including the allocator's forbidden sectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.samples, args.dt, args.seed, args.output)
