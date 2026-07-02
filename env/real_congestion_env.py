import os
import re
import time
from functools import partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo


class BottleneckTopo(Topo):
    def build(self, bw=10, delay="20ms"):
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")
        h3 = self.addHost("h3")
        s1 = self.addSwitch("s1")
        s2 = self.addSwitch("s2")
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)
        self.addLink(s1, s2, bw=bw, delay=delay)


def parse_ss_output(output):
    """Return (rtt_ms, cwnd, throughput_mbps, loss_pct) for the ESTAB
    socket actually carrying data (largest bytes_sent). ss -ti reports
    more than one socket per iperf3 run (control channel + data
    stream); mixing fields across sockets pairs the wrong values
    together, so every field here comes from the same block. Same
    parser as scripts/collect_dataset.py, validated against a live
    Mininet bottleneck topology."""
    blocks = re.split(r"\n(?=\S)", output)
    estab_blocks = [b for b in blocks if b.startswith("ESTAB")]
    if not estab_blocks:
        return None

    def bytes_sent_of(block):
        m = re.search(r"bytes_sent:(\d+)", block)
        return int(m.group(1)) if m else -1

    block = max(estab_blocks, key=bytes_sent_of)
    if bytes_sent_of(block) <= 0:
        return None

    rtt_match = re.search(r"rtt:(\d+\.\d+)", block)
    cwnd_match = re.search(r"cwnd:(\d+)", block)
    rate_match = re.search(r"delivery_rate\s+([\d.]+)([KMGkmg]?)bps", block)
    segs_match = re.search(r"segs_out:(\d+)", block)
    retrans_match = re.search(r"retrans:\d+/(\d+)", block)

    unit_to_mbps = {"": 1e-6, "K": 1e-3, "M": 1, "G": 1e3}

    rtt = float(rtt_match.group(1)) if rtt_match else 0
    cwnd = int(cwnd_match.group(1)) if cwnd_match else 0
    throughput = (
        float(rate_match.group(1)) * unit_to_mbps[rate_match.group(2).upper()]
        if rate_match else 0
    )

    loss_pct = 0.0
    if retrans_match and segs_match:
        segs_out = int(segs_match.group(1))
        total_retrans = int(retrans_match.group(1))
        if segs_out > 0:
            loss_pct = 100.0 * total_retrans / segs_out

    return rtt, cwnd, throughput, loss_pct


class RealCongestionEnv(gym.Env):
    """Gym env driven by a live Mininet network. Actions adjust h1's
    own uplink rate (via tc, same mechanism validated in
    scripts/collect_dataset.py); state comes from real ss -ti stats
    on the resulting iperf3 flow crossing the s1<->s2 bottleneck.

    One Mininet network is kept running for the lifetime of the env
    (created in __init__, torn down in close()) since rebuilding it
    every episode would dominate wall-clock time. reset() only
    restarts the iperf3 server/client, which is fast.
    """

    MIN_RATE_MBPS = 1
    MAX_RATE_MBPS = 20
    RATE_STEP_MBPS = 1
    START_RATE_MBPS = 5
    STEP_INTERVAL_S = 0.5
    MAX_STEPS_PER_EPISODE = 40
    SERVER_PORT = 5201

    def __init__(self, start_rate_mbps=None):
        super().__init__()

        # Overridable per-episode starting rate. Used to run a plain
        # TCP Cubic baseline (start high, e.g. 100, well above the
        # 10Mbps bottleneck, then never touch it -- only the real
        # bottleneck link governs, same as any other Cubic flow) for
        # comparison against the agent's normal 5Mbps start.
        self.start_rate_mbps = (
            start_rate_mbps if start_rate_mbps is not None else self.START_RATE_MBPS
        )

        if os.geteuid() != 0:
            raise SystemExit(
                "RealCongestionEnv must run as root, e.g.: sudo python3 rl/train_real.py"
            )

        # State: [RTT(ms), CWND, Throughput(Mbps)] -- same shape as
        # env/congestion_env.py so it's a drop-in swap for train.py.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([5000, 5000, 100]),
            dtype=np.float32,
        )

        # 0 = decrease, 1 = maintain, 2 = increase (uplink rate)
        self.action_space = spaces.Discrete(3)

        self.net = Mininet(
            topo=BottleneckTopo(),
            link=TCLink,
            switch=partial(OVSSwitch, failMode="standalone"),
            controller=None,
        )
        self.net.start()

        loss = self.net.pingAll()
        if loss >= 100:
            self.net.stop()
            raise RuntimeError("No connectivity in RealCongestionEnv topology.")

        self.h1 = self.net.get("h1")
        self.h3 = self.net.get("h3")

        self.rate_mbps = float(self.START_RATE_MBPS)
        self.step_count = 0

    def _restart_server(self):
        self.h3.cmd("pkill -9 iperf3")
        time.sleep(0.2)
        self.h3.cmd(f"iperf3 -s -p {self.SERVER_PORT} -D")
        time.sleep(0.3)

    def _restart_client(self):
        self.h1.cmd("pkill -9 -f 'iperf3 -c'")
        time.sleep(0.2)
        duration = int(self.MAX_STEPS_PER_EPISODE * self.STEP_INTERVAL_S) + 10
        self.h1.cmd(
            f"iperf3 -c {self.h3.IP()} -p {self.SERVER_PORT} "
            f"-t {duration} > /tmp/real_env_client.log 2>&1 &"
        )
        time.sleep(1)  # let TCP handshake complete before the first sample

    def _apply_rate(self):
        intf = self.h1.intfList()[0]
        intf.config(bw=self.rate_mbps)

    def _sample_state(self):
        parsed = parse_ss_output(self.h1.cmd("ss -ti"))
        if parsed is None:
            return np.array([0, 0, 0], dtype=np.float32), 0.0
        rtt, cwnd, throughput, loss_pct = parsed
        return np.array([rtt, cwnd, throughput], dtype=np.float32), loss_pct

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rate_mbps = float(self.start_rate_mbps)
        self.step_count = 0

        self._restart_server()
        self._apply_rate()
        self._restart_client()

        state, _ = self._sample_state()
        return state, {}

    def step(self, action):
        if action == 0:
            self.rate_mbps = max(self.MIN_RATE_MBPS, self.rate_mbps - self.RATE_STEP_MBPS)
        elif action == 2:
            self.rate_mbps = min(self.MAX_RATE_MBPS, self.rate_mbps + self.RATE_STEP_MBPS)
        self._apply_rate()

        time.sleep(self.STEP_INTERVAL_S)  # let the network react before sampling
        state, loss_pct = self._sample_state()
        rtt, cwnd, throughput = state

        # Reward = throughput - loss penalty - latency penalty
        # (Section 8 of the project plan).
        reward = float(throughput) - 2.0 * loss_pct - (float(rtt) / 100.0)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.MAX_STEPS_PER_EPISODE

        return state, reward, terminated, truncated, {"rate_mbps": self.rate_mbps}

    def render(self):
        print(self.rate_mbps)

    def close(self):
        self.h1.cmd("pkill -9 -f 'iperf3 -c'")
        self.h3.cmd("pkill -9 iperf3")
        self.net.stop()
