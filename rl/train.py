from stable_baselines3 import PPO
from env.congestion_env import CongestionEnv
import os

# Create environment
env = CongestionEnv()

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

# Train
model.learn(total_timesteps=10000)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/ppo_congestion")

print("Training Complete")
