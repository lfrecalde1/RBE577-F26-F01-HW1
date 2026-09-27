import json
from pathlib import Path

import matplotlib
import numpy as np

from actuator_limits import COMMAND_NAMES as NAMES, FORBIDDEN_SECTORS, in_forbidden_sector

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def command_summary(commands, dt, seed, lower, upper, max_rates):
    observed_rates = np.max(np.abs(np.diff(commands, axis=0)), axis=0) / dt
    channels = {}
    for j, name in enumerate(NAMES):
        # Report angles in degrees for readability; arrays/CSV remain radians.
        scale = 180. / np.pi if j in (2, 4) else 1.
        counts, _ = np.histogram(commands[:, j], bins=20, range=(lower[j], upper[j]))
        channels[name] = {
            "unit": "deg" if j in (2, 4) else "N",
            "limits": [float(lower[j] * scale), float(upper[j] * scale)],
            "observed_min": float(commands[:, j].min() * scale),
            "observed_max": float(commands[:, j].max() * scale),
            "max_abs_rate_per_s": float(observed_rates[j] * scale),
            "rate_limit_per_s": float(max_rates[j] * scale),
            "occupied_amplitude_bins_of_20": int(np.count_nonzero(counts)),
            "amplitude_bin_counts": counts.tolist(),
        }
    normalized = (commands - lower) / (upper - lower)
    joint_indices = np.clip(np.floor(normalized * 5).astype(int), 0, 4)
    joint_cells = len(np.unique(joint_indices, axis=0))
    forbidden = in_forbidden_sector(commands[:, [2, 4]]).any(axis=1)
    return {
        "samples": len(commands), "dt_s": dt,
        "time_span_s": (len(commands) - 1) * dt, "seed": seed,
        "command_order": NAMES, "array_units": ["N", "N", "rad", "N", "rad"],
        "initial_command": commands[0].tolist(),
        "method": "Stratified random waypoints with quintic smooth transitions",
        "angle_domain": "Full Eq. (7) interval; forbidden sectors are included",
        "samples_in_allocator_forbidden_sectors": int(forbidden.sum()),
        "bounds_satisfied": bool(np.all(commands >= lower - 1e-9)
                                 and np.all(commands <= upper + 1e-9)),
        "rates_satisfied": bool(np.all(observed_rates <= max_rates * (1 + 1e-10))),
        "channels": channels,
        "joint_coverage": {"bins_per_dimension": 5, "occupied_cells": joint_cells,
                           "total_cells": 5**5, "fraction": joint_cells / 5**5},
        "coverage_note": "Samples are correlated and not uniformly distributed. "
                         "Marginal range coverage does not imply all 5-D combinations "
                         "or uniform coverage of temporal frequencies.",
    }


def save_random_commands(output, commands, dt, seed, lower, upper, max_rates):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    time = np.arange(len(commands)) * dt
    summary = command_summary(commands, dt, seed, lower, upper, max_rates)
    if not summary["bounds_satisfied"] or not summary["rates_satisfied"]:
        raise RuntimeError("Generated commands failed the magnitude or rate check")
    np.savez_compressed(output / "signals.npz", time_s=time, commands=commands,
                        dt_s=dt, seed=seed, command_names=np.array(NAMES),
                        lower=lower, upper=upper, max_rates=max_rates)
    np.savetxt(output / "commands.csv", np.column_stack((time, commands)),
               delimiter=",", comments="", fmt="%.12g",
               header="time_s,F1_N,F2_N,alpha2_rad,F3_N,alpha3_rad")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_commands(output, time, commands, lower, upper)
    return summary


def plot_commands(output, time, commands, lower, upper):
    labels = ["F1 (N)", "F2 (N)", "alpha2 (deg)", "F3 (N)", "alpha3 (deg)"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    display = commands.copy()
    display[:, [2, 4]] *= 180. / np.pi
    bounds = np.column_stack((lower, upper))
    bounds[[2, 4]] *= 180. / np.pi

    fig, axes = plt.subplots(5, 2, figsize=(14, 11), constrained_layout=True)
    stride = max(1, len(time) // 12000)
    zoom = time <= min(120., time[-1])
    for j in range(5):
        axes[j, 0].plot(time[::stride] / 60., display[::stride, j], color=colors[j], lw=.7)
        axes[j, 1].plot(time[zoom], display[zoom, j], color=colors[j], lw=1.2)
        for axis in axes[j]:
            for limit in bounds[j]:
                axis.axhline(limit, color="black", ls="--", lw=.7, alpha=.5)
            axis.set_ylabel(labels[j])
            axis.grid(alpha=.2)
        if j in (2, 4):
            for axis in axes[j]:
                for low, high in np.rad2deg(FORBIDDEN_SECTORS):
                    axis.axhspan(low, high, color="tab:red", alpha=.10)
    axes[0, 0].set_title("Full record")
    axes[0, 1].set_title("First 120 seconds (or available duration)")
    axes[-1, 0].set_xlabel("Time (min)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"{len(time):,} smooth random commands | full Eq. (7) ranges\n"
                 "Shaded angle bands are forbidden by the current allocator")
    fig.savefig(output / "signals.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for j, axis in enumerate(axes.flat):
        if j == 5:
            axis.axis("off")
            continue
        axis.hist(display[:, j], bins=20, range=bounds[j], color=colors[j], rwidth=.9)
        axis.set(xlabel=labels[j], ylabel="Samples")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Amplitude coverage | time samples need not be uniformly distributed")
    fig.savefig(output / "coverage.png", dpi=150)
    plt.close(fig)


def save_force_results(output, time, commands, forces, positions):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "forces.npz", time_s=time, commands=commands,
                        generalized_forces=forces, thruster_positions_m=positions,
                        force_frame="body_origin",
                        command_names=["F1_N", "F2_N", "alpha2_rad", "F3_N", "alpha3_rad"],
                        force_names=["sway_N", "surge_N", "yaw_Nm"])
    np.savetxt(output / "commands_and_forces.csv",
               np.column_stack((time, commands, forces)), delimiter=",", comments="",
               fmt="%.12g",
               header="time_s,F1_N,F2_N,alpha2_rad,F3_N,alpha3_rad,sway_N,surge_N,yaw_Nm")

    plot_forces(output, time, forces)


def plot_forces(output, time, forces):
    elapsed = time - time[0]
    zoom = elapsed <= 120.
    labels = ["Sway force (N)", "Surge force (N)", "Yaw moment (N m)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col", constrained_layout=True)
    for j, label in enumerate(labels):
        axes[j, 0].plot(elapsed / 60., forces[:, j], color=colors[j], linewidth=.7)
        axes[j, 1].plot(elapsed[zoom], forces[zoom, j], color=colors[j], linewidth=1.2)
        for axis in axes[j]:
            axis.set_ylabel(label)
            axis.axhline(0., color="black", linewidth=.5, alpha=.5)
            axis.grid(alpha=.25)
    axes[0, 0].set_title("Full record")
    axes[0, 1].set_title("First 120 seconds (or available duration)")
    axes[-1, 0].set_xlabel("Elapsed time (min)")
    axes[-1, 1].set_xlabel("Elapsed time (s)")
    fig.suptitle(f"Forces from {len(forces):,} saved thruster commands | moment about body origin")
    fig.savefig(output / "forces.png", dpi=160)
    plt.close(fig)
