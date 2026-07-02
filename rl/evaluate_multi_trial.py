import os
import sys
import csv
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv

N_TRIALS = 5
RESULTS_CSV = "results/multi_trial_eval.csv"


def run_episode(env, policy_fn):
    obs, _ = env.reset()
    rtts, thrs, rewards = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rtt, cwnd, throughput = obs
        rtts.append(rtt)
        thrs.append(throughput)
        rewards.append(reward)
    return rtts, thrs, rewards


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ci95(xs):
    # Normal approximation, not an exact small-sample t-interval
    # (no scipy in this environment) -- fine for a rough estimate at
    # n=5, but should be labeled as approximate in the report.
    m, s, n = mean(xs), stdev(xs), len(xs)
    margin = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
    return m, m - margin, m + margin


def welch_t_test(a, b):
    ma, mb = mean(a), mean(b)
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), 0.0
    t = (ma - mb) / se
    p_approx = 2 * (1 - _norm_cdf(abs(t)))  # normal approximation, see ci95() note
    return t, p_approx


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load("models/ppo_real_congestion")
    agent_policy = lambda obs: int(model.predict(obs, deterministic=True)[0])
    cubic_policy = lambda obs: 1

    agent_trial_rtts, agent_trial_thrs = [], []
    cubic_trial_rtts, cubic_trial_thrs = [], []

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "trial", "avg_rtt_ms", "avg_throughput_mbps", "avg_reward"])

        print(f"=== PPO agent: {N_TRIALS} trials ===")
        env = RealCongestionEnv()
        try:
            for trial in range(N_TRIALS):
                rtts, thrs, rewards = run_episode(env, agent_policy)
                writer.writerow(["ppo_agent", trial, mean(rtts), mean(thrs), mean(rewards)])
                agent_trial_rtts.append(mean(rtts))
                agent_trial_thrs.append(mean(thrs))
                print(f"  trial {trial}: avg_rtt={mean(rtts):.1f}ms avg_thr={mean(thrs):.2f}Mbps")
        finally:
            env.close()

        print(f"=== TCP Cubic baseline: {N_TRIALS} trials ===")
        env = RealCongestionEnv(start_rate_mbps=100)
        try:
            for trial in range(N_TRIALS):
                rtts, thrs, rewards = run_episode(env, cubic_policy)
                writer.writerow(["cubic_baseline", trial, mean(rtts), mean(thrs), mean(rewards)])
                cubic_trial_rtts.append(mean(rtts))
                cubic_trial_thrs.append(mean(thrs))
                print(f"  trial {trial}: avg_rtt={mean(rtts):.1f}ms avg_thr={mean(thrs):.2f}Mbps")
        finally:
            env.close()

    print()
    print(f"=== Statistical comparison (n={N_TRIALS} trials each, normal approximation) ===")
    for name, agent_vals, cubic_vals, unit in [
        ("RTT", agent_trial_rtts, cubic_trial_rtts, "ms"),
        ("Throughput", agent_trial_thrs, cubic_trial_thrs, "Mbps"),
    ]:
        am, alo, ahi = ci95(agent_vals)
        cm, clo, chi = ci95(cubic_vals)
        t, p = welch_t_test(agent_vals, cubic_vals)
        print(f"{name}: PPO={am:.1f}{unit} (95% CI [{alo:.1f}, {ahi:.1f}])  "
              f"Cubic={cm:.1f}{unit} (95% CI [{clo:.1f}, {chi:.1f}])  "
              f"t={t:.2f} p~={p:.4f}")

    print(f"\nPer-trial results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
