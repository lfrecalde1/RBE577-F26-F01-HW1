import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from NeuralAllocator import AutoEncoder, ConstrainedControlLoss, LOSS_WEIGHTS, load_model
from ThrusterModel import ThrusterModel
from neural_output import save_training_results


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "random_force_results" / "forces.npz"
DEFAULT_OUTPUT = ROOT / "neural_results"

HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 1024
EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1  # Reserve the final 10% for a separate test.
SEED = 577
CPU_THREADS = 1  # Small recurrent networks often slow down with many CPU threads.


def load_force_data(input_file):
    with np.load(Path(input_file).expanduser()) as data:
        time = np.asarray(data["time_s"], dtype=float)
        forces = np.asarray(data["generalized_forces"], dtype=float)
        thrusters = ThrusterModel(data["thruster_positions_m"])
        if data["force_names"].tolist() != ["sway_N", "surge_N", "yaw_Nm"]:
            raise ValueError("force_names must be [sway_N, surge_N, yaw_Nm]")
        if str(data["force_frame"]) != "body_origin":
            raise ValueError("Forces and thruster positions must use the body origin")
    if time.ndim != 1 or len(time) < 20 or forces.shape != (len(time), 3):
        raise ValueError("Need at least 20 samples, time_s (N,), and forces (N, 3)")
    if not np.isfinite(time).all() or not np.isfinite(forces).all():
        raise ValueError("Time and force samples must be finite")
    steps = np.diff(time)
    dt = float(steps[0])
    if dt <= 0 or not np.allclose(steps, dt, rtol=1e-5, atol=1e-9):
        raise ValueError("time_s must increase with a fixed positive timestep")
    return time, forces, thrusters, dt


def split_forces(forces):
    train_end = int(len(forces) * TRAIN_FRACTION)
    validation_end = int(len(forces) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    training = forces[:train_end]
    validation = forces[train_end:validation_end]
    test = forces[validation_end:]
    mean, std = training.mean(axis=0), training.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError("Training data must vary in all three force components")
    return training, validation, test, mean, std


def run_epoch(model, criterion, forces, sequence_length, optimizer=None, rng=None):
    model.train(optimizer is not None)
    starts = np.arange(0, len(forces), sequence_length)
    if optimizer is not None:
        starts = rng.permutation(starts)
    accumulated = dict.fromkeys(LOSS_WEIGHTS, 0.)

    with torch.set_grad_enabled(optimizer is not None):
        for start in starts:
            # Shape: (one sequence, timesteps, three force components).
            tau = forces[start:start + sequence_length].unsqueeze(0)
            tau_pred, commands = model(tau)
            loss, _, parts = criterion(tau, tau_pred, commands)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; check data and loss weights")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
                optimizer.step()
            for name, value in parts.items():
                accumulated[name] += value.item() * tau.shape[1]

    # Weight by sample count so the shorter final window contributes correctly.
    result = {name: total / len(forces) * criterion.weights[name]
              for name, total in accumulated.items()}
    return {"total": sum(result.values()), **result}


def write_history(output, history):
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train(input_file=DEFAULT_INPUT, output=DEFAULT_OUTPUT, epochs=EPOCHS,
          sequence_length=SEQUENCE_LENGTH, device="cpu"):
    if epochs < 1 or sequence_length < 2:
        raise ValueError("epochs must be positive and sequence_length at least two")
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    time, forces, thrusters, dt = load_force_data(input_file)
    training, validation, test, mean, std = split_forces(forces)
    train_tensor = torch.tensor(training, dtype=torch.float32, device=device)
    validation_tensor = torch.tensor(validation, dtype=torch.float32, device=device)
    model = AutoEncoder(mean, std, thrusters, HIDDEN_SIZE).to(device)
    criterion = ConstrainedControlLoss(std, thrusters, dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    settings = {
        "input_file": str(Path(input_file).resolve()), "seed": SEED,
        "epochs": epochs, "sequence_length": sequence_length,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "train_samples": len(training), "validation_samples": len(validation),
        "test_samples": len(test), "device": str(device),
        "torch_version": str(torch.__version__),
    }
    initial = run_epoch(model, criterion, validation_tensor, sequence_length)
    print(f"Data: {len(training):,} train / {len(validation):,} validation / "
          f"{len(test):,} test; dt={dt:g} s; device={device}", flush=True)
    print(f"Initial validation loss: {initial['total']:.6g}", flush=True)
    history, best_loss = [], float("inf")
    start_time = perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, criterion, train_tensor, sequence_length, optimizer, rng)
        validation_loss = run_epoch(model, criterion, validation_tensor, sequence_length)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in train_loss.items()},
                        **{f"validation_{k}": v for k, v in validation_loss.items()}})
        write_history(output, history)

        if validation_loss["total"] < best_loss:
            best_loss = validation_loss["total"]
            torch.save({
                "model_state": model.state_dict(),
                "hidden_size": HIDDEN_SIZE,
                "thruster_positions_m": thrusters.positions.tolist(), "dt_s": dt,
                "force_order": ["sway", "surge", "yaw"],
                "command_order": ["F1", "F2", "alpha2", "F3", "alpha3"],
                "epoch": epoch, "validation_loss": best_loss,
                "loss_weights": criterion.weights,
                "command_max": criterion.command_max.cpu().tolist(),
                "max_step": criterion.max_step.cpu().tolist(),
                "forbidden_sectors": criterion.sectors.cpu().tolist(),
                "initial_validation_loss": initial["total"],
                "training_settings": settings,
            }, output / "model.pt")
        print(f"Epoch {epoch:3d}/{epochs}: train={train_loss['total']:.6g}  "
              f"validation={validation_loss['total']:.6g}", flush=True)

    # Evaluate the held-out test sequence after validation selects the model.
    model, saved = load_model(output / "model.pt", device)
    with torch.no_grad():
        test_tensor = torch.tensor(test, dtype=torch.float32, device=device).unsqueeze(0)
        tau_pred, commands = model(test_tensor)
        _, tau_command, _ = criterion(test_tensor, tau_pred, commands)
    metrics = save_training_results(
        output, time[-len(test):], test, tau_pred[0].cpu().numpy(),
        tau_command[0].cpu().numpy(), commands[0].cpu().numpy(),
        history, saved, perf_counter() - start_time)
    print(f"Selected epoch {saved['epoch']}; saved model, test data, and plots to {output}")
    print("Test physical-force RMSE [sway N, surge N, yaw N m]:",
          np.round(metrics["physical_rmse"], 2))
    print("Test samples in forbidden sectors:", metrics["samples_in_forbidden_sectors"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.input, args.output, args.epochs, args.sequence_length, args.device)
