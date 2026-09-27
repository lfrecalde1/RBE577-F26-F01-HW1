import json

import matplotlib
import numpy as np

from actuator_limits import COMMAND_MIN, COMMAND_MAX

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORCE_NAMES = ["sway_N", "surge_N", "yaw_Nm"]
COMMAND_NAMES = ["F1_N", "F2_N", "alpha2_rad", "F3_N", "alpha3_rad"]


def save_training_results(output, time, desired, predicted, achieved, commands,
                          history, saved, elapsed_s):
    dt = saved["dt_s"]
    limits = np.asarray(saved["command_max"])
    max_step = np.asarray(saved["max_step"])
    tolerance = np.array([1e-3, 1e-3, 1e-6, 1e-3, 1e-6])
    changes = np.abs(np.diff(commands, axis=0))
    magnitude_excess = np.maximum(np.abs(commands) - limits, 0.)
    rate_excess = np.maximum(changes - max_step, 0.)
    forbidden = np.zeros((len(commands), 2), dtype=bool)
    angles = commands[:, [2, 4]]
    for lower, upper in saved["forbidden_sectors"]:
        forbidden |= (angles > lower) & (angles < upper)
    std = saved["model_state"]["tau_std"].cpu().numpy()

    # Table 1's reverse-thrust limit is stricter than the paper's symmetric L2.
    hardware_violation = ((commands < COMMAND_MIN - tolerance)
                          | (commands > COMMAND_MAX + tolerance))
    metrics = {
        "selected_epoch": saved["epoch"], "test_samples": len(time), "dt_s": dt,
        "training_seconds": elapsed_s, "training_settings": saved["training_settings"],
        "initial_validation_loss": saved["initial_validation_loss"],
        "selected_validation_loss": saved["validation_loss"],
        "force_order": FORCE_NAMES, "command_order": COMMAND_NAMES,
        "loss_weights": saved["loss_weights"],
        "symmetric_command_limits": limits.tolist(),
        "rate_limits_per_s": (max_step / dt).tolist(),
        "forbidden_sectors_rad": saved["forbidden_sectors"],
        "physical_rmse": np.sqrt(np.mean((achieved - desired)**2, axis=0)).tolist(),
        "decoder_rmse": np.sqrt(np.mean((predicted - desired)**2, axis=0)).tolist(),
        "physical_normalized_rmse": np.sqrt(np.mean(((achieved - desired) / std)**2, axis=0)).tolist(),
        "decoder_normalized_rmse": np.sqrt(np.mean(((predicted - desired) / std)**2, axis=0)).tolist(),
        "magnitude_violation_samples_per_command": (magnitude_excess > tolerance).sum(axis=0).tolist(),
        "rate_violation_pairs_per_command": (rate_excess > tolerance).sum(axis=0).tolist(),
        "max_magnitude_excess_per_command": magnitude_excess.max(axis=0).tolist(),
        "max_abs_rate_per_s": (changes.max(axis=0) / dt).tolist(),
        "max_rate_excess_per_s": (rate_excess.max(axis=0) / dt).tolist(),
        "samples_in_forbidden_sectors": int(forbidden.any(axis=1).sum()),
        "sector_violation_samples_per_angle": forbidden.sum(axis=0).tolist(),
        "hardware_bound_violation_samples_per_command": hardware_violation.sum(axis=0).tolist(),
        "mean_power_proxy_N_to_1_5": float(
            np.mean(np.sum(np.abs(commands[:, [0, 1, 3]])**1.5, axis=1))),
        "constraint_note": "Soft penalties, no clipping. Rates include every test pair; "
                           "test memory is preserved for the full sequence. Angles/rates use radians.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output / "test_predictions.npz", time_s=time,
                        tau_desired=desired, tau_decoder=predicted,
                        tau_command=achieved, commands=commands,
                        force_names=FORCE_NAMES, command_names=COMMAND_NAMES,
                        thruster_positions_m=saved["thruster_positions_m"],
                        force_frame="body_origin")
    header = (["time_s"] + COMMAND_NAMES
              + [f"desired_{n}" for n in FORCE_NAMES]
              + [f"command_{n}" for n in FORCE_NAMES]
              + [f"decoder_{n}" for n in FORCE_NAMES])
    np.savetxt(output / "test_predictions.csv",
               np.column_stack((time, commands, desired, achieved, predicted)),
               delimiter=",", comments="", header=",".join(header), fmt="%.10g")
    plot_losses(output, history, saved["epoch"])
    plot_forces(output, time, desired, predicted, achieved)
    plot_commands(output, time, commands, limits, max_step / dt, saved["forbidden_sectors"])
    return metrics


def plot_losses(output, history, selected_epoch):
    epochs = [row["epoch"] for row in history]
    terms = ["total", "ground", "reconstruction", "magnitude", "rate", "power", "sector"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes.flat, terms)):
        values = []
        for split in ("train", "validation"):
            series = [row[f"{split}_{name}"] for row in history]
            values.extend(series)
            axis.plot(epochs, series, label=split)
        axis.axvline(selected_epoch, color="gray", ls=":", lw=1)
        title = name if index == 0 else f"L{index - 1}: {name}"
        axis.set(title=title, xlabel="Epoch", ylabel="Weighted mean loss")
        if min(values) > 0 and max(values) / min(values) > 100:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0, top=1 if max(values) == 0 else None)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            if max(values) == 0:
                axis.text(.5, .5, "Zero throughout", ha="center", transform=axis.transAxes)
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    axes[-1, -1].axis("off")
    fig.suptitle("Training and validation losses | dotted line: saved model")
    fig.savefig(output / "losses.png", dpi=150)
    plt.close(fig)


def plot_forces(output, time, desired, predicted, achieved):
    elapsed = time - time[0]
    zoom = elapsed <= 120.
    labels = ["Sway (N)", "Surge (N)", "Yaw (N m)"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col", constrained_layout=True)
    for j, label in enumerate(labels):
        for column, selected in enumerate((slice(None), zoom)):
            axis = axes[j, column]
            axis.plot(elapsed[selected], desired[selected, j], color="black", lw=1.3, label="Desired")
            axis.plot(elapsed[selected], achieved[selected, j], lw=1., label="From encoder commands")
            axis.plot(elapsed[selected], predicted[selected, j], ls="--", lw=1., label="Decoder")
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
    axes[0, 0].set_title("Entire held-out test sequence")
    axes[0, 1].set_title("First 120 seconds")
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Test elapsed time (s)")
    fig.suptitle("Desired and reconstructed forces | physical units, body origin")
    fig.savefig(output / "forces.png", dpi=150)
    plt.close(fig)


def plot_commands(output, time, commands, limits, rates, sectors):
    elapsed = time - time[0]
    scale = np.array([1., 1., 180. / np.pi, 1., 180. / np.pi])
    values = commands * scale
    measured_rates = np.diff(values, axis=0) / np.diff(time)[:, None]
    labels = ["F1 (N)", "F2 (N)", "alpha2 (deg)", "F3 (N)", "alpha3 (deg)"]
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True, constrained_layout=True)
    for j, label in enumerate(labels):
        axes[j, 0].plot(elapsed, values[:, j], lw=.9)
        axes[j, 1].plot(elapsed[1:], measured_rates[:, j], lw=.9)
        for sign in (-1, 1):
            axes[j, 0].axhline(sign * limits[j] * scale[j], color="black", ls="--", lw=.7)
            axes[j, 1].axhline(sign * rates[j] * scale[j], color="black", ls="--", lw=.7)
        if j in (2, 4):
            for lower, upper in np.rad2deg(sectors):
                axes[j, 0].axhspan(lower, upper, color="tab:red", alpha=.15)
        axes[j, 0].set_ylabel(label)
        axes[j, 1].set_ylabel(label.replace(")", "/s)"))
        for axis in axes[j]:
            axis.grid(alpha=.2)
    axes[0, 0].set_title("Encoder commands and symmetric training limits")
    axes[0, 1].set_title("Measured rates and limits")
    for axis in axes[-1]:
        axis.set_xlabel("Test elapsed time (s)")
    fig.suptitle("Raw neural allocation | shaded angle sectors are forbidden")
    fig.savefig(output / "commands.png", dpi=150)
    plt.close(fig)
