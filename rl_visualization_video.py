import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from stable_baselines3 import PPO

# Import your modules (adjust paths as needed)
try:
    from gym_robot.envs.robot_env import RobotImpedanceEnv
except ImportError:
    # If gym_robot is not in path, try direct import
    import sys
    sys.path.append('.')
    from robot_env import RobotImpedanceEnv

try:
    from robot.kinematics import forward_kinematics
except ImportError:
    # Placeholder if module not available
    def forward_kinematics(dh, q):
        # Will use env's method instead
        return None, None


class RobotVideoGenerator:
    """Generate animated video of 6-DOF robot RL impedance control"""
    
    def __init__(self, env, model, output_path='outputs/robot_control_video.gif', use_rl=True):
        self.env = env
        self.model = model
        self.output_path = output_path
        self.use_rl = use_rl
        
        # Data storage
        self.trajectories = []  # End-effector trajectory
        self.joint_angles = []  # Joint configurations
        self.joint_velocities = []
        self.forces = []  # External forces
        self.stiffness = []  # Impedance stiffness (K)
        self.damping = []  # Impedance damping (D)
        self.tracking_errors = []
        self.actions = []  # RL actions
        self.timestamps = []
        self.robot_frames = []  # Full robot arm frames for 3D visualization
        
    def collect_trajectory_data(self, max_steps=500, initial_q=None):
        """Run simulation and collect data"""
        print(f"Collecting trajectory data {'WITH RL' if self.use_rl else 'WITHOUT RL'}...")
        
        obs = self.env.reset()
        
        # Optional: Set specific initial joint configuration
        if initial_q is not None:
            self.env.q = np.array(initial_q)
            self.env.qdot = np.zeros(6)
            obs = self.env._get_obs()
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Select action
            if self.use_rl:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = np.zeros(6)  # No RL control
            
            # Store action
            self.actions.append(action.copy())
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Get end-effector position
            ee_pos = self.env.get_end_effector_position()
            
            # Get robot arm frames for visualization
            T, T_list = forward_kinematics(self.env.robot.get_dh(), self.env.q)
            if T_list is None:
                # Fallback if forward_kinematics not available
                T_list = []
            
            # Store data
            self.trajectories.append(ee_pos.copy())
            self.joint_angles.append(self.env.q.copy())
            self.joint_velocities.append(self.env.qdot.copy())
            self.forces.append(self.env.F_ext.copy())
            self.tracking_errors.append(np.linalg.norm(ee_pos - self.env.xd))
            self.timestamps.append(self.env.t)
            self.robot_frames.append(T_list)
            
            # Calculate current impedance parameters
            K = self.env.K_nom * (1.0 + action[:3]) if self.use_rl else self.env.K_nom
            D = self.env.D_nom * (1.0 + action[3:]) if self.use_rl else self.env.D_nom
            self.stiffness.append(K.copy())
            self.damping.append(D.copy())
            
            step_count += 1
        
        # Convert to numpy arrays
        self.trajectories = np.array(self.trajectories)
        self.joint_angles = np.array(self.joint_angles)
        self.joint_velocities = np.array(self.joint_velocities)
        self.forces = np.array(self.forces)
        self.stiffness = np.array(self.stiffness)
        self.damping = np.array(self.damping)
        self.tracking_errors = np.array(self.tracking_errors)
        self.actions = np.array(self.actions)
        self.timestamps = np.array(self.timestamps)
        
        print(f"✅ Collected {len(self.trajectories)} data points")
        return len(self.trajectories)
    
    def generate_video(self, fps=30, skip_frames=2):
        """Generate animated video with comprehensive visualization"""
        print("Generating video animation...")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(22, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.4,
                      left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # Create subplots
        ax_3d = fig.add_subplot(gs[:, :2], projection='3d')  # Robot 3D view
        ax_trajectory = fig.add_subplot(gs[0, 2])  # End-effector trajectory
        ax_status = fig.add_subplot(gs[0, 3:])  # Status panel
        
        ax_tracking = fig.add_subplot(gs[1, 2])  # Tracking error
        ax_force = fig.add_subplot(gs[1, 3])  # External force
        ax_impedance = fig.add_subplot(gs[1, 4])  # Impedance parameters
        
        ax_joints = fig.add_subplot(gs[2, 2])  # Joint angles
        ax_actions = fig.add_subplot(gs[2, 3])  # RL actions
        ax_physics = fig.add_subplot(gs[2, 4])  # Physics panel
        
        # Turn off axes for info panels
        for ax in [ax_status, ax_physics]:
            ax.axis('off')
        
        # Initialize 3D robot plot
        self._setup_3d_robot_plot(ax_3d)
        
        # Initialize robot arm visualization
        robot_line, = ax_3d.plot([], [], [], 'o-', linewidth=4, markersize=8, 
                                 color='#2E86AB', markerfacecolor='#A23B72', 
                                 markeredgewidth=2, markeredgecolor='white',
                                 label='Robot Arm')
        ee_dot, = ax_3d.plot([], [], [], 'r*', markersize=20, label='End-Effector')
        target_dot, = ax_3d.plot([self.env.xd[0]], [self.env.xd[1]], [self.env.xd[2]], 
                                 'g^', markersize=15, label='Target')
        
        # Trajectory trace
        traj_line, = ax_3d.plot([], [], [], 'r-', linewidth=1.5, alpha=0.6, 
                               label='EE Trajectory')
        
        # End-effector trajectory plot (XY view)
        ax_trajectory.set_xlabel('X (m)', fontsize=9)
        ax_trajectory.set_ylabel('Y (m)', fontsize=9)
        ax_trajectory.set_title('End-Effector Path (Top View)', fontsize=10, fontweight='bold')
        ax_trajectory.grid(True, alpha=0.3)
        ax_trajectory.plot(self.env.xd[0], self.env.xd[1], 'g^', markersize=12, label='Target')
        traj_xy_line, = ax_trajectory.plot([], [], 'r-', linewidth=2, alpha=0.7)
        ee_xy_dot, = ax_trajectory.plot([], [], 'ro', markersize=8)
        ax_trajectory.legend(fontsize=8)
        ax_trajectory.set_aspect('equal', adjustable='box')
        
        # Tracking error plot
        ax_tracking.set_xlabel('Time (s)', fontsize=9)
        ax_tracking.set_ylabel('Error (m)', fontsize=9)
        ax_tracking.set_title('Tracking Error', fontsize=10, fontweight='bold')
        ax_tracking.grid(True, alpha=0.3)
        ax_tracking.set_xlim(0, self.timestamps[-1])
        ax_tracking.set_ylim(0, max(self.tracking_errors) * 1.1)
        tracking_line, = ax_tracking.plot([], [], 'b-', linewidth=2)
        ax_tracking.axhline(y=0.01, color='g', linestyle='--', linewidth=1.5, 
                           alpha=0.7, label='Target (10mm)')
        ax_tracking.legend(fontsize=8)
        
        # Force plot
        ax_force.set_xlabel('Time (s)', fontsize=9)
        ax_force.set_ylabel('Force (N)', fontsize=9)
        ax_force.set_title('External Force (Z-axis)', fontsize=10, fontweight='bold')
        ax_force.grid(True, alpha=0.3)
        ax_force.set_xlim(0, self.timestamps[-1])
        force_range = max(abs(self.forces[:, 2].min()), abs(self.forces[:, 2].max())) * 1.2
        ax_force.set_ylim(-force_range, force_range)
        force_line, = ax_force.plot([], [], 'r-', linewidth=2)
        ax_force.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Impedance parameters plot
        ax_impedance.set_xlabel('Time (s)', fontsize=9)
        ax_impedance.set_ylabel('Stiffness Kz (N/m)', fontsize=9)
        ax_impedance.set_title('Adaptive Impedance (Z-axis)', fontsize=10, fontweight='bold')
        ax_impedance.grid(True, alpha=0.3)
        ax_impedance.set_xlim(0, self.timestamps[-1])
        ax_impedance.set_ylim(self.stiffness[:, 2].min() * 0.9, 
                             self.stiffness[:, 2].max() * 1.1)
        stiffness_line, = ax_impedance.plot([], [], 'b-', linewidth=2, label='Kz')
        ax_impedance.axhline(y=self.env.K_nom[2], color='gray', linestyle='--', 
                            linewidth=1.5, alpha=0.7, label='Nominal')
        ax_impedance.legend(fontsize=8)
        
        # Joint angles plot
        ax_joints.set_xlabel('Time (s)', fontsize=9)
        ax_joints.set_ylabel('Angle (rad)', fontsize=9)
        ax_joints.set_title('Joint Angles', fontsize=10, fontweight='bold')
        ax_joints.grid(True, alpha=0.3)
        ax_joints.set_xlim(0, self.timestamps[-1])
        ax_joints.set_ylim(self.joint_angles.min() * 1.1, self.joint_angles.max() * 1.1)
        joint_lines = [ax_joints.plot([], [], linewidth=1.5, alpha=0.7, 
                                     label=f'q{i+1}')[0] for i in range(6)]
        ax_joints.legend(fontsize=7, ncol=2)
        
        # RL Actions plot
        if self.use_rl:
            ax_actions.set_xlabel('Time (s)', fontsize=9)
            ax_actions.set_ylabel('Action', fontsize=9)
            ax_actions.set_title('RL Actions (ΔK, ΔD)', fontsize=10, fontweight='bold')
            ax_actions.grid(True, alpha=0.3)
            ax_actions.set_xlim(0, self.timestamps[-1])
            ax_actions.set_ylim(self.actions.min() * 1.2, self.actions.max() * 1.2)
            action_lines = [ax_actions.plot([], [], linewidth=1.5, alpha=0.7,
                                          label=f'a{i+1}')[0] for i in range(6)]
            ax_actions.legend(fontsize=7, ncol=2)
        else:
            ax_actions.text(0.5, 0.5, 'NO RL\n(Zero Actions)', 
                          transform=ax_actions.transAxes, ha='center', va='center',
                          fontsize=14, fontweight='bold', color='gray')
            ax_actions.axis('off')
        
        # Animation function
        def update(frame):
            idx = frame * skip_frames
            if idx >= len(self.trajectories):
                idx = len(self.trajectories) - 1
            
            # Update 3D robot arm
            if len(self.robot_frames[idx]) > 0:
                xs, ys, zs = [0], [0], [0]
                for T in self.robot_frames[idx]:
                    p = T[0:3, 3]
                    xs.append(p[0])
                    ys.append(p[1])
                    zs.append(p[2])
                robot_line.set_data(xs, ys)
                robot_line.set_3d_properties(zs)
            
            # Update end-effector
            ee_pos = self.trajectories[idx]
            ee_dot.set_data([ee_pos[0]], [ee_pos[1]])
            ee_dot.set_3d_properties([ee_pos[2]])
            
            # Update trajectory trace
            traj_line.set_data(self.trajectories[:idx, 0], self.trajectories[:idx, 1])
            traj_line.set_3d_properties(self.trajectories[:idx, 2])
            
            # Update XY trajectory
            traj_xy_line.set_data(self.trajectories[:idx, 0], self.trajectories[:idx, 1])
            ee_xy_dot.set_data([ee_pos[0]], [ee_pos[1]])
            
            # Update tracking error
            tracking_line.set_data(self.timestamps[:idx], self.tracking_errors[:idx])
            
            # Update force
            force_line.set_data(self.timestamps[:idx], self.forces[:idx, 2])
            
            # Update impedance
            stiffness_line.set_data(self.timestamps[:idx], self.stiffness[:idx, 2])
            
            # Update joint angles
            for i, line in enumerate(joint_lines):
                line.set_data(self.timestamps[:idx], self.joint_angles[:idx, i])
            
            # Update RL actions
            if self.use_rl:
                for i, line in enumerate(action_lines):
                    line.set_data(self.timestamps[:idx], self.actions[:idx, i])
            
            # Update info panels
            self._update_status_panel(ax_status, idx)
            self._update_physics_panel(ax_physics, idx)
            
            return [robot_line, ee_dot, traj_line, tracking_line]
        
        # Add title
        mode_str = "WITH RL Adaptive Impedance" if self.use_rl else "WITHOUT RL (Nominal Impedance)"
        fig.suptitle(f'6-DOF Robot Impedance Control - {mode_str}', 
                    fontsize=16, fontweight='bold')
        
        # Add legend to 3D plot
        ax_3d.legend(loc='upper left', fontsize=9)
        
        # Create animation
        total_frames = len(self.trajectories) // skip_frames
        print(f"Creating animation with {total_frames} frames...")
        
        anim = FuncAnimation(fig, update, frames=total_frames,
                           interval=1000/fps, blit=False, repeat=True)
        
        # Save animation
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', 
                   exist_ok=True)
        
        print(f"Saving video to {self.output_path}...")
        if self.output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(self.output_path, writer=writer)
        else:
            # For MP4
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(self.output_path, writer=writer)
            except Exception as e:
                print(f"⚠️ Could not save as MP4: {e}")
                print("Saving as GIF instead...")
                gif_path = self.output_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                self.output_path = gif_path
        
        print(f"✅ Video saved to: {self.output_path}")
        plt.close()
    
    def _setup_3d_robot_plot(self, ax):
        """Setup 3D robot plot axes and labels"""
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('Robot Arm Configuration', fontsize=11, fontweight='bold')
        
        # Set workspace limits
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 0.8])
        
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    def _update_status_panel(self, ax, idx):
        """Update status information panel"""
        ax.clear()
        ax.axis('off')
        
        current_time = self.timestamps[idx]
        current_error = self.tracking_errors[idx]
        avg_error = np.mean(self.tracking_errors[:idx+1])
        max_error = np.max(self.tracking_errors[:idx+1])
        progress = (idx / len(self.trajectories)) * 100
        
        ee_pos = self.trajectories[idx]
        target = self.env.xd
        
        # Determine status
        if current_error < 0.01:
            status = "PRECISE"
            status_color = '#00FF00'
        elif current_error < 0.05:
            status = "TRACKING"
            status_color = '#FFD700'
        else:
            status = "CONVERGING"
            status_color = '#FF6B6B'
        
        # Mode indicator
        mode = "RL ACTIVE" if self.use_rl else "NO RL"
        mode_color = '#00BFFF' if self.use_rl else '#808080'
        
        # Create text
        text = f"""PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIME: {current_time:.3f} s
STEP: {idx}/{len(self.trajectories)}
MODE: {mode}

TRACKING ERROR:
Current: {current_error:.5f} m ({current_error*1000:.2f} mm)
Average: {avg_error:.5f} m ({avg_error*1000:.2f} mm)
Maximum: {max_error:.5f} m ({max_error*1000:.2f} mm)

POSITION:
Current:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]
Target:   [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]
Error:    [{ee_pos[0]-target[0]:.3f}, {ee_pos[1]-target[1]:.3f}, {ee_pos[2]-target[2]:.3f}]

STATUS: {status}
Progress: {progress:.1f}%
"""
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=8.5, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add status indicator
        status_patch = mpatches.Rectangle((0.75, 0.88), 0.22, 0.08,
                                         facecolor=status_color, alpha=0.7,
                                         transform=ax.transAxes)
        ax.add_patch(status_patch)
        ax.text(0.86, 0.92, status, transform=ax.transAxes,
               fontsize=10, fontweight='bold', ha='center', va='center')
        
        # Add mode indicator
        mode_patch = mpatches.Rectangle((0.75, 0.78), 0.22, 0.08,
                                       facecolor=mode_color, alpha=0.7,
                                       transform=ax.transAxes)
        ax.add_patch(mode_patch)
        ax.text(0.86, 0.82, mode, transform=ax.transAxes,
               fontsize=9, fontweight='bold', ha='center', va='center')
    
    def _update_physics_panel(self, ax, idx):
        """Update physics information panel"""
        ax.clear()
        ax.axis('off')
        
        force = self.forces[idx]
        force_mag = np.linalg.norm(force)
        
        K = self.stiffness[idx]
        D = self.damping[idx]
        
        q = self.joint_angles[idx]
        qdot = self.joint_velocities[idx]
        
        if self.use_rl:
            action = self.actions[idx]
            k_change = action[:3] * 100  # Percentage change
            d_change = action[3:] * 100
        
        # Create text
        text = f"""IMPEDANCE CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STIFFNESS K (N/m):
Kx: {K[0]:7.1f}  (Nom: {self.env.K_nom[0]:.1f})
Ky: {K[1]:7.1f}  (Nom: {self.env.K_nom[1]:.1f})
Kz: {K[2]:7.1f}  (Nom: {self.env.K_nom[2]:.1f})

DAMPING D (N·s/m):
Dx: {D[0]:7.1f}  (Nom: {self.env.D_nom[0]:.1f})
Dy: {D[1]:7.1f}  (Nom: {self.env.D_nom[1]:.1f})
Dz: {D[2]:7.1f}  (Nom: {self.env.D_nom[2]:.1f})

EXTERNAL FORCE (N):
Fx: {force[0]:8.3f}
Fy: {force[1]:8.3f}
Fz: {force[2]:8.3f}
|F|: {force_mag:7.3f}

JOINT STATE:
Max |q|:    {np.max(np.abs(q)):.3f} rad
Max |qdot|: {np.max(np.abs(qdot)):.3f} rad/s
"""
        
        if self.use_rl:
            text += f"""
RL ACTIONS (% change):
ΔKx: {k_change[0]:+6.1f}%  ΔDx: {d_change[0]:+6.1f}%
ΔKy: {k_change[1]:+6.1f}%  ΔDy: {d_change[1]:+6.1f}%
ΔKz: {k_change[2]:+6.1f}%  ΔDz: {d_change[2]:+6.1f}%
"""
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=8.5, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))


def generate_comparison_video(env, model, output_path='outputs/robot_comparison.gif', 
                             fps=30, skip_frames=2):
    """Generate side-by-side comparison video of RL vs No-RL"""
    print("=" * 80)
    print("GENERATING COMPARISON VIDEO: RL vs No-RL")
    print("=" * 80)
    
    # Collect data for both scenarios
    print("\n1. Collecting NO-RL data...")
    gen_no_rl = RobotVideoGenerator(env, model, use_rl=False)
    gen_no_rl.collect_trajectory_data(max_steps=500)
    
    print("\n2. Collecting RL data...")
    gen_rl = RobotVideoGenerator(env, model, use_rl=True)
    gen_rl.collect_trajectory_data(max_steps=500)
    
    print("\n3. Creating comparison animation...")
    
    # Create figure with side-by-side layout
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.04, right=0.96, top=0.92, bottom=0.05)
    
    # Left side: No-RL
    ax_3d_norl = fig.add_subplot(gs[:, 0], projection='3d')
    ax_track_norl = fig.add_subplot(gs[0, 1])
    ax_force_norl = fig.add_subplot(gs[1, 1])
    
    # Right side: RL
    ax_3d_rl = fig.add_subplot(gs[:, 2], projection='3d')
    ax_track_rl = fig.add_subplot(gs[0, 3])
    ax_force_rl = fig.add_subplot(gs[1, 3])
    
    # Setup plots
    for ax, title in [(ax_3d_norl, 'WITHOUT RL'), (ax_3d_rl, 'WITH RL')]:
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 0.8])
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    # Initialize robot lines
    robot_line_norl, = ax_3d_norl.plot([], [], [], 'o-', linewidth=3, markersize=6, color='gray')
    ee_norl, = ax_3d_norl.plot([], [], [], 'r*', markersize=15)
    traj_norl, = ax_3d_norl.plot([], [], [], 'r-', linewidth=1.5, alpha=0.6)
    
    robot_line_rl, = ax_3d_rl.plot([], [], [], 'o-', linewidth=3, markersize=6, color='#2E86AB')
    ee_rl, = ax_3d_rl.plot([], [], [], 'r*', markersize=15)
    traj_rl, = ax_3d_rl.plot([], [], [], 'r-', linewidth=1.5, alpha=0.6)
    
    # Add targets
    for ax in [ax_3d_norl, ax_3d_rl]:
        ax.plot([env.xd[0]], [env.xd[1]], [env.xd[2]], 'g^', markersize=12)
    
    # Tracking error plots
    for ax, title, data in [(ax_track_norl, 'Tracking Error (No-RL)', gen_no_rl),
                             (ax_track_rl, 'Tracking Error (RL)', gen_rl)]:
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Error (m)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(gen_no_rl.timestamps[-1], gen_rl.timestamps[-1]))
        ax.set_ylim(0, max(gen_no_rl.tracking_errors.max(), gen_rl.tracking_errors.max()) * 1.1)
        ax.axhline(y=0.01, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    track_line_norl, = ax_track_norl.plot([], [], 'b-', linewidth=2)
    track_line_rl, = ax_track_rl.plot([], [], 'b-', linewidth=2)
    
    # Force plots
    force_range = max(abs(gen_no_rl.forces[:, 2].min()), abs(gen_rl.forces[:, 2].min()),
                     abs(gen_no_rl.forces[:, 2].max()), abs(gen_rl.forces[:, 2].max())) * 1.2
    
    for ax, title in [(ax_force_norl, 'External Force Fz (No-RL)'),
                      (ax_force_rl, 'External Force Fz (RL)')]:
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Force (N)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(gen_no_rl.timestamps[-1], gen_rl.timestamps[-1]))
        ax.set_ylim(-force_range, force_range)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    force_line_norl, = ax_force_norl.plot([], [], 'r-', linewidth=2)
    force_line_rl, = ax_force_rl.plot([], [], 'r-', linewidth=2)
    
    # Animation function
    def update(frame):
        idx = frame * skip_frames
        idx_norl = min(idx, len(gen_no_rl.trajectories) - 1)
        idx_rl = min(idx, len(gen_rl.trajectories) - 1)
        
        # Update No-RL robot
        if len(gen_no_rl.robot_frames[idx_norl]) > 0:
            xs, ys, zs = [0], [0], [0]
            for T in gen_no_rl.robot_frames[idx_norl]:
                p = T[0:3, 3]
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            robot_line_norl.set_data(xs, ys)
            robot_line_norl.set_3d_properties(zs)
        
        ee_pos = gen_no_rl.trajectories[idx_norl]
        ee_norl.set_data([ee_pos[0]], [ee_pos[1]])
        ee_norl.set_3d_properties([ee_pos[2]])
        traj_norl.set_data(gen_no_rl.trajectories[:idx_norl, 0], 
                          gen_no_rl.trajectories[:idx_norl, 1])
        traj_norl.set_3d_properties(gen_no_rl.trajectories[:idx_norl, 2])
        
        # Update RL robot
        if len(gen_rl.robot_frames[idx_rl]) > 0:
            xs, ys, zs = [0], [0], [0]
            for T in gen_rl.robot_frames[idx_rl]:
                p = T[0:3, 3]
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            robot_line_rl.set_data(xs, ys)
            robot_line_rl.set_3d_properties(zs)
        
        ee_pos = gen_rl.trajectories[idx_rl]
        ee_rl.set_data([ee_pos[0]], [ee_pos[1]])
        ee_rl.set_3d_properties([ee_pos[2]])
        traj_rl.set_data(gen_rl.trajectories[:idx_rl, 0], 
                        gen_rl.trajectories[:idx_rl, 1])
        traj_rl.set_3d_properties(gen_rl.trajectories[:idx_rl, 2])
        
        # Update tracking errors
        track_line_norl.set_data(gen_no_rl.timestamps[:idx_norl], 
                                gen_no_rl.tracking_errors[:idx_norl])
        track_line_rl.set_data(gen_rl.timestamps[:idx_rl], 
                              gen_rl.tracking_errors[:idx_rl])
        
        # Update forces
        force_line_norl.set_data(gen_no_rl.timestamps[:idx_norl], 
                                gen_no_rl.forces[:idx_norl, 2])
        force_line_rl.set_data(gen_rl.timestamps[:idx_rl], 
                              gen_rl.forces[:idx_rl, 2])
        
        return [robot_line_norl, robot_line_rl]
    
    # Title
    fig.suptitle('6-DOF Robot Impedance Control: Comparison', 
                fontsize=16, fontweight='bold')
    
    # Create animation
    total_frames = max(len(gen_no_rl.trajectories), len(gen_rl.trajectories)) // skip_frames
    print(f"Creating animation with {total_frames} frames...")
    
    anim = FuncAnimation(fig, update, frames=total_frames,
                       interval=1000/fps, blit=False, repeat=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
               exist_ok=True)
    print(f"Saving comparison video to {output_path}...")
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"✅ Comparison video saved to: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)
    print(f"\nNO-RL:")
    print(f"  Mean error: {np.mean(gen_no_rl.tracking_errors):.5f} m ({np.mean(gen_no_rl.tracking_errors)*1000:.2f} mm)")
    print(f"  Final error: {gen_no_rl.tracking_errors[-1]:.5f} m ({gen_no_rl.tracking_errors[-1]*1000:.2f} mm)")
    print(f"  Max force: {np.max(np.abs(gen_no_rl.forces[:, 2])):.2f} N")
    
    print(f"\nWITH RL:")
    print(f"  Mean error: {np.mean(gen_rl.tracking_errors):.5f} m ({np.mean(gen_rl.tracking_errors)*1000:.2f} mm)")
    print(f"  Final error: {gen_rl.tracking_errors[-1]:.5f} m ({gen_rl.tracking_errors[-1]*1000:.2f} mm)")
    print(f"  Max force: {np.max(np.abs(gen_rl.forces[:, 2])):.2f} N")
    
    improvement = ((np.mean(gen_no_rl.tracking_errors) - np.mean(gen_rl.tracking_errors)) / 
                   np.mean(gen_no_rl.tracking_errors) * 100)
    print(f"\nRL Improvement: {improvement:+.1f}%")
    
    plt.close()


def main():
    """Main function to generate robot control video"""
    print("=" * 80)
    print("6-DOF ROBOT RL IMPEDANCE CONTROL VIDEO GENERATOR")
    print("=" * 80)
    
    # Initialize environment
    env = RobotImpedanceEnv(dt=0.01)
    
    # Load trained model
    model_path = 'ppo_robot_impedance.zip'
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded trained model from: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first by running rl_main.py")
        return
    
    print("\n" + "=" * 80)
    print("VIDEO GENERATION OPTIONS")
    print("=" * 80)
    print("1. Single video (RL-controlled)")
    print("2. Single video (No-RL baseline)")
    print("3. Comparison video (Side-by-side)")
    
    choice = input("\nSelect option (1/2/3) [default: 3]: ").strip() or "3"
    
    if choice == "1":
        # Generate RL video
        video_gen = RobotVideoGenerator(env, model, 
                                        output_path='outputs/robot_rl_control.gif',
                                        use_rl=True)
        video_gen.collect_trajectory_data(max_steps=500)
        
        print("\n" + "=" * 80)
        print("TRAJECTORY STATISTICS (WITH RL)")
        print("=" * 80)
        print(f"Duration: {video_gen.timestamps[-1]:.2f} seconds")
        print(f"Mean error: {np.mean(video_gen.tracking_errors):.5f} m ({np.mean(video_gen.tracking_errors)*1000:.2f} mm)")
        print(f"Final error: {video_gen.tracking_errors[-1]:.5f} m ({video_gen.tracking_errors[-1]*1000:.2f} mm)")
        print(f"Max force: {np.max(np.abs(video_gen.forces[:, 2])):.2f} N")
        
        video_gen.generate_video(fps=30, skip_frames=2)
        
    elif choice == "2":
        # Generate No-RL video
        video_gen = RobotVideoGenerator(env, model, 
                                        output_path='outputs/robot_norl_control.gif',
                                        use_rl=False)
        video_gen.collect_trajectory_data(max_steps=500)
        
        print("\n" + "=" * 80)
        print("TRAJECTORY STATISTICS (WITHOUT RL)")
        print("=" * 80)
        print(f"Duration: {video_gen.timestamps[-1]:.2f} seconds")
        print(f"Mean error: {np.mean(video_gen.tracking_errors):.5f} m ({np.mean(video_gen.tracking_errors)*1000:.2f} mm)")
        print(f"Final error: {video_gen.tracking_errors[-1]:.5f} m ({video_gen.tracking_errors[-1]*1000:.2f} mm)")
        print(f"Max force: {np.max(np.abs(video_gen.forces[:, 2])):.2f} N")
        
        video_gen.generate_video(fps=30, skip_frames=2)
        
    else:
        # Generate comparison video
        generate_comparison_video(env, model, 
                                 output_path='outputs/robot_comparison.gif',
                                 fps=30, skip_frames=2)
    
    print("\n" + "=" * 80)
    print("VIDEO GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
