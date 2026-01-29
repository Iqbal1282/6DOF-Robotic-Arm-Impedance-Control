import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

from robot.model import RobotModel
from robot.jacobian import compute_jacobian
from robot.kinematics import forward_kinematics
from robot.dynamics import mass_matrix, coriolis_matrix, gravity_vector
from simulation.external_force import external_force


class RobotImpedanceEnv(gym.Env):
    """
    RL-guided adaptive impedance control environment for a 6-DOF robot arm.

    The RL agent does NOT control torques.
    It adaptively modulates stiffness (K) and damping (D)
    to balance tracking accuracy and safe interaction force.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, dt=0.01):
        super().__init__()

        # ----------------------------
        # Robot model
        # ----------------------------
        self.robot = RobotModel()
        self.n = 6
        self.dt = dt

        # ----------------------------
        # Desired end-effector position
        # ----------------------------
        self.xd = np.array([0.45, 0.15, 0.30])

        # ----------------------------
        # Nominal impedance parameters
        # ----------------------------
        self.K_nom = np.array([100.0, 100.0, 100.0])
        self.D_nom = np.array([30.0, 30.0, 30.0])

        # ----------------------------
        # Action space
        # ΔK and ΔD scaling factors
        # ----------------------------
        self.action_space = spaces.Box(
            low=-0.2,
            high=0.2,
            shape=(6,),
            dtype=np.float32,
        )

        # ----------------------------
        # Observation space
        # q, qdot, position error, external force
        # ----------------------------
        obs_dim = 6 + 6 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Rendering
        self.fig = None
        self.ax = None
        self.line = None

        self.reset()

    # ==================================================
    # Reset
    # ==================================================
    def reset(self):
        # Randomized environment stiffness (unknown to agent)
        self.k_env = np.random.uniform(300, 2000)

        self.q = np.zeros(6)
        self.qdot = np.zeros(6)

        self.t = 0.0
        self.done = False

        # initialize force
        self.F_ext = np.zeros(3)

        return self._get_obs()

    # ==================================================
    # Step
    # ==================================================
    def step(self, action):

        if self.done:
            return self._get_obs(), 0.0, True, {}

        # ----------------------------
        # RL action → impedance tuning
        # ----------------------------
        action = np.clip(action, -0.2, 0.2)

        K = self.K_nom * (1.0 + action[:3])
        D = self.D_nom * (1.0 + action[3:])
        self.current_kz = K[2]

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

        # ----------------------------
        # External interaction force
        # ----------------------------
        self.F_ext = external_force(
            self.t,
            self.q,
            k_env=self.k_env,
        )

        self.current_force_z = self.F_ext[2]

        self.current_z = x[2]
        # ----------------------------
        # Impedance control law
        # ----------------------------
        e = x - self.xd
        edot = xdot

        e = np.clip(e, -1.0, 1.0)
        edot = np.clip(edot, -1.0, 1.0)

        F_cmd = -K * e - D * edot + self.F_ext

        tau = Jv.T @ F_cmd
        tau += coriolis_matrix(self.q, self.qdot) @ self.qdot
        tau += gravity_vector(self.q)

        tau = np.clip(tau, -200, 200)

        # ----------------------------
        # Robot dynamics
        # ----------------------------
        M = mass_matrix(self.q)

        try:
            qddot = np.linalg.solve(M, tau)
        except np.linalg.LinAlgError:
            qddot = np.zeros_like(self.q)

        self.qdot += qddot * self.dt
        self.q += self.qdot * self.dt

        self.q = np.clip(self.q, -np.pi, np.pi)
        self.qdot = np.clip(self.qdot, -5.0, 5.0)

        self.t += self.dt

        # ----------------------------
        # Safety termination
        # ----------------------------
        if np.any(np.isnan(self.q)) or np.any(np.isnan(self.qdot)):
            self.done = True
            return self._get_obs(), -1000.0, True, {}

        # ----------------------------
        # Reward
        # ----------------------------
        reward = (
            -2.0 * np.linalg.norm(e)
            -0.001 * np.linalg.norm(tau)
            -0.05 * np.linalg.norm(self.F_ext)
        )

        # Episode horizon
        self.done = self.t >= 5.0

        info = {
            "tracking_error": np.linalg.norm(e),
            "force": np.linalg.norm(self.F_ext),
        }

        return self._get_obs(), reward, self.done, info

    # ==================================================
    # Observation
    # ==================================================
    def _get_obs(self):

        T, _ = forward_kinematics(self.robot.get_dh(), self.q)
        x = T[0:3, 3]

        e = x - self.xd

        e = np.clip(e, -1.0, 1.0)
        F_ext = np.clip(self.F_ext, -50.0, 50.0)

        obs = np.concatenate([
            self.q,
            self.qdot,
            e,
            F_ext
        ])

        return obs.astype(np.float32)
    
    def get_end_effector_position(self):
        T, _ = forward_kinematics(self.robot.get_dh(), self.q)
        return T[0:3, 3].copy()


    # ==================================================
    # Rendering
    # ==================================================
    def render(self, mode="human"):

        T_list = forward_kinematics(self.robot.get_dh(), self.q)[1]

        xs, ys, zs = [0], [0], [0]

        for T in T_list:
            p = T[0:3, 3]
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])

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
            self.ax.set_title("RL-Guided Impedance Control")

            plt.ion()
            plt.show()

        else:
            self.line.set_data(xs, ys)
            self.line.set_3d_properties(zs)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
