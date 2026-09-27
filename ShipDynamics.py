import numpy as np


class ShipDynamics:
    def __init__(self, mass_matrix, damping_matrix, *, include_coriolis=True):
        self.M = self._matrix(mass_matrix, "mass_matrix")
        self.D = self._matrix(damping_matrix, "damping_matrix")
        if np.linalg.eigvalsh(self.M).min() <= 0:
            raise ValueError("mass_matrix must be positive definite")
        if np.linalg.eigvalsh(self.D).min() < 0:
            raise ValueError("damping_matrix must be positive semidefinite")
        self.include_coriolis = include_coriolis

        self.cg_in_body_frame = None

        self.M.setflags(write=False)
        self.D.setflags(write=False)

    @classmethod
    def gunnerus(cls, *, include_coriolis=True):
        mass = 418_061.0
        x_g = 13.202
        radius_of_gyration = 7.225
        yaw_inertia = mass * radius_of_gyration**2

        m11 = mass - 70000.0
        m22 = mass - 340000.0
        m23 = mass * x_g - 5749241.32
        m33 = yaw_inertia - 20240000.51
        mass_matrix = [
            [m11, 0.0, 0.0],
            [0.0, m22, m23],
            [0.0, m23, m33],
        ]
        damping_matrix = np.diag([17400.0, 80200.0, 4971500.0])
        model = cls(mass_matrix, damping_matrix,
                    include_coriolis=include_coriolis)
        model.cg_in_body_frame = (x_g, 0.0)
        return model

    def coriolis_matrix(self, velocity):
        velocity = self._vector(velocity, 3, "velocity")
        p_surge, p_sway, _ = self.M @ velocity
        return np.array([
            [0.0, 0.0, -p_sway],
            [0.0, 0.0, p_surge],
            [p_sway, -p_surge, 0.0],
        ])

    def derivative(self, state, control, *, disturbance=None):
        state = self._vector(state, 6, "state")
        control = self._vector(control, 3, "control")
        _, _, heading, u, v, r = state
        velocity = state[3:]

        tsway, tsurge, tyaw = control
        tau = np.array([tsurge, tsway, tyaw])
        if disturbance is not None:
            disturbance = self._vector(disturbance, 3, "disturbance")
            tau += disturbance[[1, 0, 2]]

        cosine, sine = np.cos(heading), np.sin(heading)
        position_rate = np.array([
            u * cosine - v * sine,
            u * sine + v * cosine,
            r,
        ])

        net_force = tau - self.D @ velocity
        if self.include_coriolis:
            net_force -= self.coriolis_matrix(velocity) @ velocity

        acceleration = np.linalg.solve(self.M, net_force)
        return np.concatenate((position_rate, acceleration))

    def step(self, state, control, dt, *, disturbance=None):
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        state = self._vector(state, 6, "state")
        k1 = self.derivative(state, control, disturbance=disturbance)
        k2 = self.derivative(state + 0.5 * dt * k1, control,
                             disturbance=disturbance)
        k3 = self.derivative(state + 0.5 * dt * k2, control,
                             disturbance=disturbance)
        k4 = self.derivative(state + dt * k3, control,
                             disturbance=disturbance)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def kinetic_energy(self, state):
        velocity = self._vector(state, 6, "state")[3:]
        return float(0.5 * velocity @ self.M @ velocity)

    @staticmethod
    def _matrix(values, name):
        matrix = np.array(values, dtype=float, copy=True)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite 3 by 3 matrix")
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        return matrix

    @staticmethod
    def _vector(values, length, name):
        vector = np.asarray(values, dtype=float)
        if vector.shape != (length,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite vector of length {length}")
        return vector
