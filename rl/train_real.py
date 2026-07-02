import os
import sys

# Make sure the project root (parent of this rl/ directory) is on
# sys.path so `env.real_congestion_env` resolves regardless of the
# working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv

# Create environment (live Mininet network -- each step takes real
# wall-clock time, so this uses a much smaller timestep budget than
# train.py's synthetic-env run, and a smaller rollout buffer so PPO
# actually gets multiple policy updates within that budget).
env = RealCongestionEnv()

try:
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        verbose=1,
    )

    # Train (~3000 steps * 0.5s/step + reset overhead ~= 25-30 minutes)
    model.learn(total_timesteps=3000)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model
    model.save("models/ppo_real_congestion")

    print("Training Complete (real network)")
finally:
    env.close()
