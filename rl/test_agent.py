from stable_baselines3 import PPO
from env.congestion_env import CongestionEnv

env = CongestionEnv()

model = PPO.load("models/ppo_congestion")

obs, _ = env.reset()

for i in range(20):
    action, _ = model.predict(obs)

    obs, reward, _, _, _ = env.step(action)

    print(
        f"Step {i+1}:",
        "Action =", action,
        "State =", obs,
        "Reward =", reward
    )
