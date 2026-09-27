import json
from pathlib import Path

import matplotlib
import numpy as np

from actuator_limits import (
    COMMAND_MIN,
    COMMAND_MAX,
    COMMAND_TOLERANCE,
    MAX_RATES,
    FORBIDDEN_ANGLES,
    in_forbidden_sector,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"direct": "Direct forces", "nlp": "NLP + thrusters", "neural": "Neural + thrusters"}
STYLES = {"direct": dict(color="black", ls="--", lw=1.5),
          "nlp": dict(color="tab:blue", ls=":", lw=2.),
          "neural": dict(color="tab:orange", lw=1.4)}
FORCE_NAMES = ["sway_N", "surge_N", "yaw_Nm"]
COMMAND_NAMES = ["F1_N", "F2_N", "alpha2_rad", "F3_N", "alpha3_rad"]
STATE_NAMES = ["north_m", "east_m", "heading_rad", "u_m_s", "v_m_s", "r_rad_s"]


def command_rates(commands, times, initial_command):
    """Include the first issued command relative to the initial actuator setting."""
    changes = np.diff(np.vstack((initial_command, commands)), axis=0)
    return changes / np.diff(times)[:, None]


def method_summary(states, forces, commands, times, initial_command,
                   direct_states, desired):
    force_error = forces - desired
    state_error = states - direct_states
    position_error = np.linalg.norm(state_error[:, :2], axis=1)
    heading_error = np.arctan2(np.sin(state_error[:, 2]), np.cos(state_error[:, 2]))
    rates = command_rates(commands, times, initial_command)
    magnitude_violation = ((commands < COMMAND_MIN - COMMAND_TOLERANCE)
                           | (commands > COMMAND_MAX + COMMAND_TOLERANCE))
    # Use a per-step tolerance, matching the NLP's command constraint check.
    rate_violation = (np.abs(rates) - MAX_RATES) * np.diff(times)[:, None] > COMMAND_TOLERANCE
    forbidden = in_forbidden_sector(commands[:, [2, 4]], tolerance=1e-8)
    return {
        "force_rmse": np.sqrt(np.mean(force_error**2, axis=0)).tolist(),
        "max_abs_force_error": np.abs(force_error).max(axis=0).tolist(),
        "max_abs_state_difference": np.abs(state_error).max(axis=0).tolist(),
        "max_position_error_m": float(position_error.max()),
        "final_position_error_m": float(position_error[-1]),
        "max_heading_error_deg": float(np.rad2deg(np.abs(heading_error).max())),
        "magnitude_violation_samples_per_command": magnitude_violation.sum(axis=0).tolist(),
        "rate_violation_samples_per_command": rate_violation.sum(axis=0).tolist(),
        "max_abs_rate_per_s": np.abs(rates).max(axis=0).tolist(),
        "samples_in_forbidden_sectors": int(forbidden.any(axis=1).sum()),
        "mean_power_proxy_N_to_1_5": float(
            np.mean(np.sum(np.abs(commands[:, [0, 1, 3]])**1.5, axis=1))),
    }


def save_comparison(output, times, states, forces, commands, initial_command,
                    *, checkpoint, allow_slack, raw_neural_commands=None,
                    neural_limiter_enabled=False):
    """Save N command/force intervals and N+1 states for the three methods."""
    count = len(times) - 1
    if count < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Need increasing time edges for at least one control interval")
    for name in LABELS:
        for values, shape in ((states[name], (count + 1, 6)), (forces[name], (count, 3))):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name}: expected finite array of shape {shape}")
    for values in commands.values():
        if values.shape != (count, 5) or not np.isfinite(values).all():
            raise ValueError("Commands must be finite with shape (N, 5)")
    if raw_neural_commands is not None:
        if (raw_neural_commands.shape != (count, 5)
                or not np.isfinite(raw_neural_commands).all()):
            raise ValueError("Raw neural commands must be finite with shape (N, 5)")

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    intervals = np.column_stack((times[:-1], times[1:]))
    data = dict(time_s=times, initial_command=initial_command,
                force_names=FORCE_NAMES, command_names=COMMAND_NAMES,
                state_names=STATE_NAMES, force_frame="body_origin")
    summary = {
        "samples": count, "state_samples": len(times),
        "duration_s": float(times[-1] - times[0]), "dt_s": float(times[1] - times[0]),
        "initial_state": states["direct"][0].tolist(),
        "initial_command": initial_command.tolist(),
        "neural_checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "nlp_allow_slack": bool(allow_slack),
        "neural_limiter_enabled": bool(neural_limiter_enabled),
        "neural_limiter_note": "Limits magnitude and rate using the previous applied command. "
                               "Forbidden angle sectors are checked separately, not enforced by clipping.",
        "force_order": FORCE_NAMES, "command_order": COMMAND_NAMES, "state_order": STATE_NAMES,
        "command_min": COMMAND_MIN.tolist(), "command_max": COMMAND_MAX.tolist(),
        "rate_limits_per_s": MAX_RATES.tolist(),
        "rate_note": "Includes the first command relative to initial_command for both allocators.",
        "comparison_note": "Same open-loop force requests, ship dynamics, initial state, and timestep. "
                           "Direct forces bypass allocation. Neural physical forces are calculated "
                           "from the applied commands after any limiting; the decoder is not used.",
    }
    for name in LABELS:
        data[f"{name}_states"] = states[name]
        data[f"{name}_forces"] = forces[name]
        np.savetxt(output / f"states_{name}.csv", np.column_stack((times, states[name])),
                   delimiter=",", comments="", header=",".join(["time_s"] + STATE_NAMES))
        np.savetxt(output / f"forces_{name}.csv", np.column_stack((intervals, forces[name])),
                   delimiter=",", comments="", header=",".join(["t_start_s", "t_end_s"] + FORCE_NAMES))
        if name in commands:
            data[f"{name}_commands"] = commands[name]
            data[f"{name}_force_error"] = forces[name] - forces["direct"]
            data[f"{name}_state_error"] = states[name] - states["direct"]
            np.savetxt(output / f"commands_{name}.csv", np.column_stack((intervals, commands[name])),
                       delimiter=",", comments="", header=",".join(["t_start_s", "t_end_s"] + COMMAND_NAMES))
            summary[name] = method_summary(states[name], forces[name], commands[name], times,
                                           initial_command, states["direct"], forces["direct"])
    if raw_neural_commands is not None:
        data["neural_raw_commands"] = raw_neural_commands
        difference = commands["neural"] - raw_neural_commands
        data["neural_limiter_adjustment"] = difference
        summary["neural_samples_limited"] = int(np.any(np.abs(difference) > 1e-9, axis=1).sum())
        summary["max_abs_neural_limiter_adjustment"] = np.abs(difference).max(axis=0).tolist()
        np.savetxt(output / "commands_neural_raw.csv",
                   np.column_stack((intervals, raw_neural_commands)), delimiter=",", comments="",
                   header=",".join(["t_start_s", "t_end_s"] + COMMAND_NAMES))
    np.savez_compressed(output / "rollout.npz", **data)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_states(output, times, states)
    plot_forces(output, times, forces)
    plot_trajectory(output, times, states)
    plot_commands(output, times, commands, initial_command,
                  raw_neural_commands if neural_limiter_enabled else None)
    print(f"Saved the three-method comparison to {output}")
    return summary


def plot_states(output, times, states):
    labels = ["North (m)", "East (m)", "Heading (deg)",
              "Surge speed (m/s)", "Sway speed (m/s)", "Yaw rate (deg/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for axis, index in zip(axes.flat, [0, 3, 1, 4, 2, 5]):
        scale = 180. / np.pi if index in (2, 5) else 1.
        for name in LABELS:
            axis.plot(times, states[name][:, index] * scale, label=LABELS[name], **STYLES[name])
        axis.set_ylabel(labels[index])
        axis.grid(alpha=.25)
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Ship motion under the same force requests")
    fig.savefig(output / "state_comparison.png", dpi=160)
    plt.close(fig)


def plot_forces(output, times, forces):
    labels = ["Sway (N)", "Surge (N)", "Yaw (N m)"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for j, label in enumerate(labels):
        for name in LABELS:
            axes[j, 0].stairs(forces[name][:, j], times, baseline=None,
                               label=LABELS[name], **STYLES[name])
            if name != "direct":
                axes[j, 1].stairs(forces[name][:, j] - forces["direct"][:, j], times,
                                   baseline=None, label=LABELS[name], **STYLES[name])
        axes[j, 0].set_ylabel(label)
        axes[j, 1].set_ylabel("Error " + label)
        axes[j, 1].axhline(0., color="black", ls="--", lw=.7)
        for axis in axes[j]:
            axis.grid(alpha=.25)
    axes[0, 0].set_title("Force applied to the ship")
    axes[0, 1].set_title("Applied minus requested force")
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Body-origin forces: direct input, NLP, and neural allocation")
    fig.savefig(output / "force_comparison.png", dpi=160)
    plt.close(fig)


def plot_trajectory(output, times, states):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name in LABELS:
        values = states[name]
        axes[0].plot(values[:, 1], values[:, 0], label=LABELS[name], **STYLES[name])
        axes[0].plot(values[-1, 1], values[-1, 0], "o", color=STYLES[name]["color"], ms=4)
        if name != "direct":
            error = np.linalg.norm(values[:, :2] - states["direct"][:, :2], axis=1)
            axes[1].plot(times, error, label=LABELS[name], **STYLES[name])
    axes[0].plot(states["direct"][0, 1], states["direct"][0, 0], "ks", ms=5, label="Start")
    axes[0].set(xlabel="East (m)", ylabel="North (m)", title="Ship path; dots mark final positions")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(xlabel="Time (s)", ylabel="Position difference (m)", title="Distance from direct-force trajectory")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(output / "trajectory.png", dpi=160)
    plt.close(fig)


def plot_commands(output, times, commands, initial_command, raw_neural_commands=None):
    labels = ["F1 (N)", "F2 (N)", "alpha2 (deg)", "F3 (N)", "alpha3 (deg)"]
    scale = np.array([1., 1., 180. / np.pi, 1., 180. / np.pi])
    rates = {name: command_rates(values, times, initial_command) for name, values in commands.items()}
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True, constrained_layout=True)
    for j, label in enumerate(labels):
        if raw_neural_commands is not None:
            axes[j, 0].stairs(raw_neural_commands[:, j] * scale[j], times, baseline=None,
                               label="Neural prediction before limits", color="gray", ls="--", lw=1.)
        for name, values in commands.items():
            axes[j, 0].stairs(values[:, j] * scale[j], times, baseline=None,
                               label=LABELS[name], **STYLES[name])
            axes[j, 1].stairs(rates[name][:, j] * scale[j], times, baseline=None, **STYLES[name])
        for sign in (-1, 1):
            axes[j, 1].axhline(sign * MAX_RATES[j] * scale[j], color="black", ls="--", lw=.6)
        if j in (2, 4):
            for limit in (COMMAND_MIN[j], COMMAND_MAX[j]):
                axes[j, 0].axhline(limit * scale[j], color="black", ls="--", lw=.6)
            lower, upper = np.rad2deg(FORBIDDEN_ANGLES)
            axes[j, 0].axhspan(lower, upper, color="red", alpha=.1)
            axes[j, 0].axhspan(-upper, -lower, color="red", alpha=.1)
        else:
            bounds = f"Bounds: {COMMAND_MIN[j]:g} to {COMMAND_MAX[j]:g} N"
            axes[j, 0].text(.02, .04, bounds, transform=axes[j, 0].transAxes,
                            fontsize=8, bbox=dict(facecolor="white", alpha=.7, edgecolor="none"))
        axes[j, 0].set_ylabel(label)
        axes[j, 1].set_ylabel(label.replace(")", "/s)"))
        for axis in axes[j]:
            axis.grid(alpha=.25)
    axes[0, 0].set_title("Thruster commands (force axes zoomed to the data)")
    axes[0, 1].set_title("Applied rates, including change from initial command")
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Applied NLP and neural commands | shaded sectors are forbidden")
    fig.savefig(output / "thruster_commands.png", dpi=160)
    plt.close(fig)
