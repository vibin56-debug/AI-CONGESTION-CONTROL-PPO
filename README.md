# AI-Based Adaptive Congestion Control using PPO + Mininet

An RL agent (PPO) learns to control a TCP flow's sending rate over a live
Mininet network, trading a small amount of throughput for a large reduction
in queueing delay compared to plain TCP Cubic.

## Environment setup

```bash
sudo apt -y install mininet openvswitch-switch iperf3 wireshark python3 python3-pip
sudo pip3 install numpy pandas matplotlib gymnasium stable-baselines3 torch pyshark
```

Packages must be installed with `sudo pip3 install` (system-wide), not
`pip3 install --user` -- every script here needs root (for Mininet), and
root's Python doesn't see per-user `--user` installs.

## Project structure

```
data/raw/network_metrics.csv    Real network metrics collected via scripts/collect_dataset.py
env/congestion_env.py           Synthetic (formula-based) Gym env -- used for the first PPO pass
env/real_congestion_env.py      Live Gym env: actions adjust h1's real uplink rate (tc), state
                                 comes from real `ss -ti` stats on a live iperf3 flow
models/                         Saved PPO models
results/                        Plots and evaluation CSVs
rl/train.py                     PPO training on the synthetic env
rl/train_real.py                PPO training on the live Mininet env
rl/evaluate_real.py             Single-episode PPO-agent-vs-Cubic comparison
rl/evaluate_multi_trial.py      5-trial comparison with 95% CI and a significance test
rl/plot_results.py              Throughput/RTT/summary bar charts from evaluate_real.py's output
rl/plot_training_curve.py       Training reward curve
scripts/monitor.py              Live ss -ti monitor for a given Mininet host PID (manual use)
scripts/collect_dataset.py      Automated multi-scenario dataset collector (baseline/medium/
                                 high load + bursty), fully self-contained (own Mininet network)
src/topology.py, src/bottleneck.py   Manual Mininet topologies (interactive CLI)
```

## How the live environment works

`RealCongestionEnv` (`env/real_congestion_env.py`) owns a Mininet network for
its whole lifetime (topology: `h1`, `h2` on switch `s1`; `h3` on switch `s2`;
`s1<->s2` is the bottleneck link, `bw=10Mbps, delay=20ms`). Each episode runs
a fresh iperf3 flow from `h1` to `h3`. The agent's 3 actions (decrease /
maintain / increase) adjust h1's own uplink rate via `tc` (`TCIntf.config`).
State is `[rtt_ms, cwnd, throughput_mbps]`, read from `ss -ti` on h1, parsed
to pick out the actual data-carrying socket (iperf3 opens a second, mostly
idle control-channel socket that must not be mixed into the same reading).
Reward is `throughput - 2*loss_pct - rtt_ms/100` (Section 8 of the original
project plan).

Since each step takes real wall-clock time (~0.5s to let the network react),
training uses a much smaller timestep budget (3000) than the original plan's
100k-200k target, with a correspondingly smaller PPO rollout buffer
(`n_steps=256`) so multiple policy updates still happen within that budget.

## Running things

All scripts that touch Mininet need root and a clean prior state:

```bash
sudo mn -c
cd ~/AI_Congestion_Control
sudo python3 <script>
```

- Collect a labeled dataset: `sudo python3 scripts/collect_dataset.py`
  (writes `data/raw/network_metrics.csv`; current scenario durations are set
  for a quick smoke test -- raise `SAMPLE_INTERVAL`/`duration` in the file
  for a full-scale collection, see the comment at the bottom of the file)
- Train on the live network: `sudo python3 rl/train_real.py` (~25-30 min)
- Compare agent vs Cubic (1 episode each): `sudo python3 rl/evaluate_real.py`
- Compare agent vs Cubic (5 trials, with stats): `sudo python3 rl/evaluate_multi_trial.py`
- Plots (no root needed): `python3 rl/plot_results.py`, `python3 rl/plot_training_curve.py`

## Results (5-trial evaluation, `rl/evaluate_multi_trial.py`)

| Metric | PPO agent | TCP Cubic (no rate shaping) |
|---|---|---|
| RTT | 209.2ms (95% CI [141.5, 276.9]) | 436.7ms (95% CI [200.6, 672.8]) |
| Throughput | 9.2Mbps (95% CI [9.1, 9.2]) | 9.5Mbps (95% CI [9.4, 9.6]) |

RTT reduction (~52%) trends strongly in the agent's favor but doesn't reach
significance at n=5 (t=-1.82, p~=0.069) given high trial-to-trial variance;
the throughput difference (~3%) is small but statistically significant
(t=-5.06, p~=0.0000). In short: the agent learned to trade a small,
consistent amount of throughput for a large but noisier reduction in queueing
delay -- more trials would tighten the RTT confidence interval.

Plots: `results/throughput_vs_time.png`, `results/rtt_vs_time.png`,
`results/summary_comparison.png`, `results/training_reward_curve.png`.

## Known limitations / notes for anyone continuing this

- iperf3's own `-b` bitrate flag is not reliably enforced in this
  environment; real load control comes from `tc`-shaping each sender's own
  uplink (`set_uplink_bw()` in `scripts/collect_dataset.py`,
  `_apply_rate()` in `env/real_congestion_env.py`), not from `-b`.
- Mininet hosts share one PID namespace by default -- `pkill` run "on" one
  host kills matching processes on *every* host. Client-side cleanup must
  use a name-scoped pattern (`pkill -9 -f 'iperf3 -c'`); broad `pkill -9
  iperf3` is only safe when nothing else is meant to survive it.
- OVSSwitch's default `failMode` is `secure` (drops everything with no
  controller). Passing `controller=None` to `Mininet()` does not fix this by
  itself -- switches must be created with `failMode='standalone'`.
- `sudo -n` / any non-interactive `sudo` will not work in this setup (no
  cached credentials, no NOPASSWD rule); everything must be run from an
  interactive terminal where the password can be typed once per session.
- The full ~100k-sample dataset target from the original plan has not been
  collected yet -- `scripts/collect_dataset.py` is validated end-to-end but
  currently runs at smoke-test scale (30s/scenario). Bumping
  `SAMPLE_INTERVAL` to 0.1 and scenario `duration` to ~900s gets close to
  that target in under an hour (see the comment at the bottom of the file).
