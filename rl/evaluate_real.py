import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv

RESULTS_CSV = "results/ppo_vs_cubic_eval.csv"


def run_episode(env, writer, run_name, policy_fn):
    obs, _ = env.reset()
    rows = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rtt, cwnd, throughput = obs
        row = [run_name, step, rtt, cwnd, throughput, reward, info["rate_mbps"]]
        writer.writerow(row)
        rows.append(row)
        step += 1
    return rows


def summarize(name, rows):
    n = len(rows)
    avg_rtt = sum(r[2] for r in rows) / n
    avg_thr = sum(r[4] for r in rows) / n
    avg_reward = sum(r[5] for r in rows) / n
    print(f"{name:<12} n={n:<4} avg_rtt={avg_rtt:>8.1f}ms "
          f"avg_throughput={avg_thr:>6.2f}Mbps avg_reward={avg_reward:>8.2f}")


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load("models/ppo_real_congestion")

    # start_rate_mbps for the agent doesn't matter here (constructor
    # default is used for the "agent" run); the "cubic" run below gets
    # its own env instance started high and never adjusted.
    agent_env = RealCongestionEnv()
    try:
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "step", "rtt_ms", "cwnd", "throughput_mbps", "reward", "rate_mbps"])

            print("=== Running trained PPO agent ===")
            agent_policy = lambda obs: int(model.predict(obs, deterministic=True)[0])
            agent_rows = run_episode(agent_env, writer, "ppo_agent", agent_policy)
            f.flush()
    finally:
        agent_env.close()

    # Fresh network for the baseline run -- start rate high (100Mbps,
    # well above the 10Mbps bottleneck) and always "maintain" (action
    # 1), so no external rate shaping ever kicks in: only the real
    # bottleneck link and TCP Cubic's own congestion control govern
    # the flow, same as any plain Cubic connection.
    cubic_env = RealCongestionEnv(start_rate_mbps=100)
    try:
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)

            print("=== Running plain TCP Cubic baseline (no rate shaping) ===")
            cubic_policy = lambda obs: 1
            cubic_rows = run_episode(cubic_env, writer, "cubic_baseline", cubic_policy)
            f.flush()
    finally:
        cubic_env.close()

    print()
    print("=== Summary ===")
    summarize("ppo_agent", agent_rows)
    summarize("cubic_baseline", cubic_rows)
    print(f"\nFull per-step results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
