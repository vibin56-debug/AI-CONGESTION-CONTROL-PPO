import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ep_rew_mean printed by PPO's own logger during train_real.py's run
# (rl/train_real.py, 3072 timesteps / n_steps=256 -> 12 rollout
# iterations). Recorded here since no tensorboard_log was configured
# for that run.
timesteps = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072]
ep_rew_mean = [52.2, 90.3, 133, 163, 185, 196, 205, 216, 223, 228, 230, 234]

plt.figure(figsize=(8, 4.5))
plt.plot(timesteps, ep_rew_mean, marker="o", color="#2E75B6")
plt.xlabel("Training timestep")
plt.ylabel("Mean episode reward")
plt.title("PPO Training Reward Curve (real Mininet network)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_reward_curve.png", dpi=150)
plt.close()

print("Saved: results/training_reward_curve.png")
