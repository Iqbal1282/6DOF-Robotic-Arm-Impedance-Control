# gym_robot/envs/robot_env.py
import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from matplotlib import animation

from robot.model import RobotModel
from robot.jacobian import compute_jacobian
from robot.kinematics import forward_kinematics
from robot.dynamics import mass_matrix, coriolis_matrix, gravity_vector
from simulation.external_force import external_force


class RobotImpedanceEnv(gym.Env):
    """
    RL-guided impedance control environment (stable + renderable)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, dt=0.01):
        super().__init__()

        self.robot = RobotModel()
        self.dt = dt
        self.n = 6  # 6-DOF

        # Desired end-effector position
        self.xd = np.array([0.45, 0.15, 0.3])

        # Nominal impedance gains
        self.K_nom = np.array([100.0, 100.0, 100.0])
        self.D_nom = np.array([30.0, 30.0, 30.0])

        # Action: ΔK, ΔD scaling factors
        self.action_space = spaces.Box(
            low=-0.2, high=0.2, shape=(6,), dtype=np.float32
        )

        # Observation: q, qdot, position error, external force
        obs_dim = 6 + 6 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # For rendering
        self.fig = None
        self.ax = None
        self.line = None

        self.reset()

    # --------------------------------------------------
    def reset(self):
        self.q = np.zeros(6)
        self.qdot = np.zeros(6)
        self.t = 0.0
        self.done = False
        return self._get_obs()

    # --------------------------------------------------
    def step(self, action):

        if self.done:
            return self._get_obs(), 0.0, True, {}

        # ----------------------------
        # Clip RL actions
        # ----------------------------
        action = np.clip(action, -0.2, 0.2)

        K = self.K_nom * (1.0 + action[:3])
        D = self.D_nom * (1.0 + action[3:])

        # ----------------------------
        # Forward kinematics
        # ----------------------------
        T, _ = forward_kinematics(self.robot.get_dh(), self.q)
        x = T[0:3, 3]

        # Jacobian
        J = compute_jacobian(self.robot.get_dh(), self.q)
        Jv = J[0:3, :]

        # End-effector velocity
        xdot = Jv @ self.qdot

        # External force
        F_ext = external_force(self.t, self.q)

        # Impedance control
        e = x - self.xd
        edot = xdot

        # Clip errors to prevent explosion
        e = np.clip(e, -1.0, 1.0)
        edot = np.clip(edot, -1.0, 1.0)

        # Compute torque
        tau = Jv.T @ (-K * e - D * edot + F_ext)
        tau += coriolis_matrix(self.q, self.qdot) @ self.qdot
        tau += gravity_vector(self.q)

        # Clip torque for stability
        tau = np.clip(tau, -200, 200)

        # Dynamics integration (Euler)
        M = mass_matrix(self.q)
        try:
            qddot = np.linalg.inv(M) @ tau
        except np.linalg.LinAlgError:
            qddot = np.zeros_like(self.q)  # fail-safe

        self.qdot += qddot * self.dt
        self.q += self.qdot * self.dt

        # Clip joint angles and velocities
        self.q = np.clip(self.q, -np.pi, np.pi)
        self.qdot = np.clip(self.qdot, -5, 5)

        # Update time
        self.t += self.dt

        # Check for NaN / Inf
        if np.any(np.isnan(self.q)) or np.any(np.isnan(self.qdot)):
            self.done = True
            reward = -1000
            return self._get_obs(), reward, True, {}

        # ----------------------------
        # Observation
        # ----------------------------
        obs = self._get_obs()

        # ----------------------------
        # Reward: tracking + energy + force
        # ----------------------------
        reward = -np.linalg.norm(e) - 0.001 * np.linalg.norm(tau) - 0.01 * np.linalg.norm(F_ext)

        # Episode termination
        self.done = self.t >= 5.0

        info = {"x_error": np.linalg.norm(e), "force": np.linalg.norm(F_ext)}

        return obs, reward, self.done, info

    # --------------------------------------------------
    def _get_obs(self):
        T, _ = forward_kinematics(self.robot.get_dh(), self.q)
        x = T[0:3, 3]
        F_ext = external_force(self.t, self.q)
        e = x - self.xd

        # Clip to avoid NaNs
        e = np.clip(e, -1.0, 1.0)
        F_ext = np.clip(F_ext, -50, 50)

        obs = np.concatenate([self.q, self.qdot, e, F_ext])
        return obs.astype(np.float32)

    # --------------------------------------------------
    def render(self, mode="human"):
        """
        3D robot visualization using matplotlib
        """
        T_list = forward_kinematics(self.robot.get_dh(), self.q)[1]

        xs = [0]
        ys = [0]
        zs = [0]
        for T in T_list:
            pos = T[0:3, 3]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

        if self.fig is None:
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.line, = self.ax.plot(xs, ys, zs, "o-", lw=3)
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-1, 1])
            self.ax.set_zlim([0, 1.2])
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("6-DOF Robot")
            plt.ion()
            plt.show()
        else:
            self.line.set_data(xs, ys)
            self.line.set_3d_properties(zs)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
