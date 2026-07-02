import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_CSV = "results/ppo_vs_cubic_eval.csv"

runs = defaultdict(lambda: {"step": [], "rtt_ms": [], "throughput_mbps": [], "reward": []})

with open(RESULTS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = runs[row["run"]]
        r["step"].append(int(row["step"]))
        r["rtt_ms"].append(float(row["rtt_ms"]))
        r["throughput_mbps"].append(float(row["throughput_mbps"]))
        r["reward"].append(float(row["reward"]))

LABELS = {"ppo_agent": "PPO Agent", "cubic_baseline": "TCP Cubic (baseline)"}
COLORS = {"ppo_agent": "#2E75B6", "cubic_baseline": "#C0504D"}

# Throughput vs time
plt.figure(figsize=(8, 4.5))
for run, data in runs.items():
    plt.plot(data["step"], data["throughput_mbps"], label=LABELS.get(run, run), color=COLORS.get(run))
plt.xlabel("Step")
plt.ylabel("Throughput (Mbps)")
plt.title("Throughput vs Time: PPO Agent vs TCP Cubic")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/throughput_vs_time.png", dpi=150)
plt.close()

# RTT vs time
plt.figure(figsize=(8, 4.5))
for run, data in runs.items():
    plt.plot(data["step"], data["rtt_ms"], label=LABELS.get(run, run), color=COLORS.get(run))
plt.xlabel("Step")
plt.ylabel("RTT (ms)")
plt.title("RTT vs Time: PPO Agent vs TCP Cubic")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/rtt_vs_time.png", dpi=150)
plt.close()

# Summary bar chart: avg throughput / avg RTT / avg reward
run_names = list(runs.keys())
avg_thr = [sum(runs[r]["throughput_mbps"]) / len(runs[r]["throughput_mbps"]) for r in run_names]
avg_rtt = [sum(runs[r]["rtt_ms"]) / len(runs[r]["rtt_ms"]) for r in run_names]
avg_reward = [sum(runs[r]["reward"]) / len(runs[r]["reward"]) for r in run_names]
labels = [LABELS.get(r, r) for r in run_names]
colors = [COLORS.get(r, "#888888") for r in run_names]

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].bar(labels, avg_thr, color=colors)
axes[0].set_title("Avg Throughput (Mbps)")
axes[0].tick_params(axis="x", rotation=15)
axes[1].bar(labels, avg_rtt, color=colors)
axes[1].set_title("Avg RTT (ms)")
axes[1].tick_params(axis="x", rotation=15)
axes[2].bar(labels, avg_reward, color=colors)
axes[2].set_title("Avg Reward")
axes[2].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("results/summary_comparison.png", dpi=150)
plt.close()

print("Saved: results/throughput_vs_time.png")
print("Saved: results/rtt_vs_time.png")
print("Saved: results/summary_comparison.png")
