# rl_main.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_robot.envs.robot_env import RobotImpedanceEnv

# ------------------------------
# PARAMETERS
# ------------------------------
TRAINING_STEPS = 200_000
TEST_EPISODES = 5
MODEL_PATH = "ppo_robot_impedance"

# ------------------------------
# CREATE ENVIRONMENT
# ------------------------------
def make_env():
    return RobotImpedanceEnv(dt=0.01)


# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train():
    env = make_env()

    # Save checkpoints every 50k steps
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="ppo_model")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4
    )

    print("Starting training...")
    model.learn(total_timesteps=TRAINING_STEPS, callback=checkpoint_callback)
    print(f"Training completed. Saving model to {MODEL_PATH}.zip")
    model.save(MODEL_PATH)
    env.close()


# ------------------------------
# TEST FUNCTION
# ------------------------------
def test(model_path=f"{MODEL_PATH}.zip", visualize=True):
    env = make_env()
    model = PPO.load(model_path)

    for ep in range(TEST_EPISODES):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            if visualize:
                env.render()

        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    env.close()


# ------------------------------
# MAIN ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    # Create checkpoints folder
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # Train the model
    train()

    # Test the model
    test(visualize=True)
