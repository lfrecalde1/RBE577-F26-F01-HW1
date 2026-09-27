import numpy as np

class ThrusterModel:
    def __init__(self, positions):
        positions = np.array(positions, dtype=float, copy=True)
        if positions.shape != (3, 2) or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite (3, 2) array for T1, T2, T3")
        self.positions = positions
        self.positions.setflags(write=False)

    @classmethod
    def from_paper(cls):
        l1 = -14.0  
        l2 = 14.5   
        l3 = -2.7
        l4 = 2.7
        return cls([
            [l2, 0.0],
            [l1, l3],
            [l1, l4],
        ])

    def generalized_forces(self, command):
        command = np.asarray(command, dtype=float)
        if command.ndim == 0 or command.shape[-1] != 5 or not np.isfinite(command).all():
            raise ValueError("command must have shape (..., 5) with finite values")
        components = self._force_components(
            np.moveaxis(command, -1, 0), sin=np.sin, cos=np.cos
        )
        return np.stack(components, axis=-1)

    def symbolic_generalized_forces(self, command):
        import casadi as ca
        components = self._force_components(ca.vertsplit(command), sin=ca.sin, cos=ca.cos)
        return ca.vertcat(*components)

    def tensor_generalized_forces(self, commands):
        import torch
        components = self._force_components(
            commands.unbind(dim=-1), sin=torch.sin, cos=torch.cos)
        return torch.stack(components, dim=-1)

    def _force_components(self, command, *, sin, cos):
        F1, F2, alpha2, F3, alpha3 = command
        l2, _ = self.positions[0]
        l1_port, l3 = self.positions[1]
        l1_starboard, l4 = self.positions[2]
        # Both stern arms are l1 for this boat; custom layouts may differ.
        s2, c2 = sin(alpha2), cos(alpha2)
        s3, c3 = sin(alpha3), cos(alpha3)

        tau_surge = F2 * c2 + F3 * c3
        tau_sway = F1 + F2 * s2 + F3 * s3
        # Moment = x*Fy - y*Fx for each thruster.
        tau_yaw = (l2 * F1
                   + (l1_port * s2 - l3 * c2) * F2
                   + (l1_starboard * s3 - l4 * c3) * F3)
        return tau_sway, tau_surge, tau_yaw
