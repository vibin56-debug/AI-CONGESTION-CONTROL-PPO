import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CongestionEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # State:
        # [RTT(ms), CWND, Throughput(Mbps)]

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([5000, 5000, 100]),
            dtype=np.float32
        )

        # Actions:
        # 0 = decrease
        # 1 = keep
        # 2 = increase

        self.action_space = spaces.Discrete(3)

        self.state = np.array(
            [100, 1000, 5],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        self.state = np.array(
            [100, 1000, 5],
            dtype=np.float32
        )

        return self.state, {}

    def step(self, action):

        rtt, cwnd, throughput = self.state

        if action == 0:
            cwnd -= 50

        elif action == 2:
            cwnd += 50

        throughput = min(10, cwnd / 150)

        rtt = 80 + max(0, cwnd - 1000) * 0.5

        reward = throughput - (rtt / 1000)

        self.state = np.array(
            [rtt, cwnd, throughput],
            dtype=np.float32
        )

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(self.state)
