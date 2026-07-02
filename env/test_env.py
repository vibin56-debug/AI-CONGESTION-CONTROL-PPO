from congestion_env import CongestionEnv

env = CongestionEnv()

obs, _ = env.reset()

for i in range(5):

    obs, reward, _, _, _ = env.step(2)

    print(
        "State:",
        obs,
        "Reward:",
        reward
    )
