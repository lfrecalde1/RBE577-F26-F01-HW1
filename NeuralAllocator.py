from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ThrusterModel import ThrusterModel
from actuator_limits import MAX_RATES, FORBIDDEN_SECTORS, limit_command


COMMAND_SCALE = [10000., 5000., np.pi, 5000., np.pi]

COMMAND_MAX = [30000., 60000., np.pi, 60000., np.pi]

LOSS_WEIGHTS = {
    "ground": 100.,       
    "reconstruction": 100., 
    "magnitude": 1e-1,  
    "rate": 1e-7,       
    "power": 1e-7,      
    "sector": 1e-1,
}


class AutoEncoder(nn.Module):
    def __init__(self, tau_mean, tau_std, thrusters=None, hidden_size=64):
        super().__init__()
        self.thrusters = thrusters if thrusters is not None else ThrusterModel.from_paper()
        mean = torch.as_tensor(tau_mean, dtype=torch.float32)
        std = torch.as_tensor(tau_std, dtype=torch.float32)
        if (mean.shape != (3,) or std.shape != (3,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or torch.any(std <= 0)):
            raise ValueError("tau_mean and tau_std must have three finite values; std > 0")
        self.register_buffer("tau_mean", mean.clone())
        self.register_buffer("tau_std", std.clone())
        self.register_buffer("command_scale", torch.tensor(COMMAND_SCALE, dtype=torch.float32))

        self.encoder = nn.LSTM(3, hidden_size, num_layers=2, batch_first=True)
        self.command_head = nn.Linear(hidden_size, 5)
        self.decoder = nn.LSTM(5, hidden_size, num_layers=2, batch_first=True)
        self.force_head = nn.Linear(hidden_size, 3)

    def encode(self, tau_desired, state=None):
        if tau_desired.ndim != 3 or tau_desired.shape[-1] != 3:
            raise ValueError("tau_desired must have shape (batch, time, 3)")
        normalized = (tau_desired - self.tau_mean) / self.tau_std
        features, state = self.encoder(normalized, state)
        commands = self.command_head(features) * self.command_scale
        return commands, state

    def forward(self, tau_desired):
        commands, _ = self.encode(tau_desired)
        features, _ = self.decoder(commands / self.command_scale)
        tau_pred = self.force_head(features) * self.tau_std + self.tau_mean
        return tau_pred, commands


class ConstrainedControlLoss(nn.Module):
    def __init__(self, tau_std, thrusters, dt, command_max=COMMAND_MAX,
                 max_rates=MAX_RATES, weights=None):
        super().__init__()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive seconds")
        self.thrusters = thrusters
        self.dt = float(dt)
        self.weights = dict(LOSS_WEIGHTS if weights is None else weights)
        self.register_buffer("tau_std", torch.as_tensor(tau_std, dtype=torch.float32).clone())
        self.register_buffer("command_max", torch.tensor(command_max, dtype=torch.float32))
        self.register_buffer("max_step", torch.tensor(max_rates, dtype=torch.float32) * dt)
        self.register_buffer("sectors", torch.tensor(FORBIDDEN_SECTORS, dtype=torch.float32))

    def sector_loss(self, commands):
        angles = commands[..., [2, 4]]
        penalty = torch.zeros_like(angles)
        for lower, upper in self.sectors:
            midpoint = (lower + upper) / 2
            distance = torch.where(angles < midpoint, angles - lower, upper - angles)
            penalty = penalty + F.relu(distance) / ((upper - lower) / 2)
        return penalty.sum(dim=-1).mean()

    def forward(self, tau_desired, tau_pred, commands):
        tau_command = self.thrusters.tensor_generalized_forces(commands)

        L0 = ((tau_command - tau_desired) / self.tau_std).square().mean()
        L1 = ((tau_pred - tau_desired) / self.tau_std).square().mean()
        L2 = F.relu(commands.abs() - self.command_max).sum(dim=-1).mean()

        # Rate penalties compare adjacent timesteps within each sequence.
        changes = commands[:, 1:] - commands[:, :-1]
        samples = commands.shape[0] * commands.shape[1]
        L3 = F.relu(changes.abs() - self.max_step).sum() / samples

        L4 = commands[..., [0, 1, 3]].abs().pow(1.5).sum(dim=-1).mean()
        L5 = self.sector_loss(commands)
        losses = {"ground": L0, "reconstruction": L1, "magnitude": L2,
                  "rate": L3, "power": L4, "sector": L5}
        total = sum(self.weights[name] * value for name, value in losses.items())
        return total, tau_command, losses


def load_model(checkpoint_file, device="cpu"):
    saved = torch.load(Path(checkpoint_file), map_location=device, weights_only=True)
    state = saved["model_state"]
    model = AutoEncoder(state["tau_mean"], state["tau_std"],
                        ThrusterModel(saved["thruster_positions_m"]),
                        hidden_size=saved["hidden_size"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, saved


class NeuralAllocator:
    def __init__(self, checkpoint_file, device="cpu", *, limit_commands=True):
        self.model, saved = load_model(checkpoint_file, device)
        self.dt_s = saved["dt_s"]
        self.limit_commands = limit_commands
        self.reset()

    def reset(self, initial_command=None):
        initial = np.zeros(5) if initial_command is None else initial_command
        # Passing the initial setting twice validates it without changing it.
        previous = limit_command(initial, initial, self.dt_s)
        self.state = None
        self.previous_command = previous.copy()
        self.last_raw_command = None

    @torch.no_grad()
    def allocate(self, tau_desired):
        tau = torch.as_tensor(tau_desired, dtype=torch.float32,
                              device=self.model.tau_mean.device)
        if tau.shape != (3,) or not torch.isfinite(tau).all():
            raise ValueError("tau_desired must be finite [sway, surge, yaw]")
        commands, self.state = self.model.encode(tau.reshape(1, 1, 3), self.state)
        self.last_raw_command = commands[0, 0].cpu().numpy().copy()
        command = self.last_raw_command.copy()
        if self.limit_commands:
            command = limit_command(command, self.previous_command, self.dt_s)
        # The next rate limit starts from the command actually applied.
        self.previous_command = command.copy()
        return command
