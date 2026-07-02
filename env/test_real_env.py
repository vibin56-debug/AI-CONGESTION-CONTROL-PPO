from real_congestion_env import RealCongestionEnv

env = RealCongestionEnv()

obs, _ = env.reset()
print("Initial state:", obs)

for i in range(10):

    action = 2 if i % 2 == 0 else 0  # alternate increase/decrease

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        "Step", i + 1,
        "Action:", action,
        "State:", obs,
        "Reward:", round(reward, 3),
        "Rate:", info["rate_mbps"],
    )

env.close()
