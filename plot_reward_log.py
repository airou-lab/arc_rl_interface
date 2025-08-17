import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="Logs", help="Directory containing episode_log CSVs")
parser.add_argument("--threshold", choices=["mean", "median", "percentile"], default="percentile",
                    help="Threshold type for comparison line")
args = parser.parse_args()

log_dir = args.log_dir

# Load logs
log_files = sorted([
    f for f in os.listdir(log_dir)
    if f.endswith(".csv") and f.startswith("episode_log_")
])
if not log_files:
    raise FileNotFoundError(f"No CSV logs found in {log_dir}/.")

plt.figure(figsize=(13, 7))
colors = cm.get_cmap("tab10", len(log_files))

# Track for dynamic threshold
all_smoothed_rewards = []
best_run_label = None
best_run_reward = -float("inf")
best_run_data = None

for i, log_file in enumerate(log_files):
    path = os.path.join(log_dir, log_file)
    df = pd.read_csv(path)

    df.dropna(inplace=True)
    df = df.drop_duplicates(subset="step_idx")
    df = df.sort_values(by="step_idx").reset_index(drop=True)

    df["smoothed_reward"] = df["reward"].rolling(window=5).mean()
    all_smoothed_rewards.extend(df["smoothed_reward"].dropna().values)

    run_id = log_file.replace("episode_log_", "Run ").replace(".csv", "")
    mean_smoothed = df["smoothed_reward"].dropna().mean()
    label = f"{run_id} (Mean: {mean_smoothed:.2f})"

    if mean_smoothed > best_run_reward:
        best_run_reward = mean_smoothed
        best_run_label = label
        best_run_data = df.copy()

    color = "gold" if mean_smoothed == best_run_reward else colors(i)
    linewidth = 3 if mean_smoothed == best_run_reward else 2
    plt.plot(df["step_idx"], df["smoothed_reward"], label=label, color=color, linewidth=linewidth)

# --- Dynamic Threshold Line ---
if all_smoothed_rewards:
    if args.threshold == "mean":
        threshold = np.mean(all_smoothed_rewards)
        thresh_label = f"Mean: {threshold:.2f}"
    elif args.threshold == "median":
        threshold = np.median(all_smoothed_rewards)
        thresh_label = f"Median: {threshold:.2f}"
    else:
        threshold = np.percentile(all_smoothed_rewards, 90)
        thresh_label = f"90th Percentile: {threshold:.2f}"
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.2, label=thresh_label)
else:
    threshold = 0.8
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.2, label="Default Threshold: 0.8")

# --- Time Decay Start Marker ---
if best_run_data is not None:
    halfway = best_run_data["step_idx"].max() // 2
    plt.axvline(x=halfway, color='gray', linestyle='--', linewidth=1.2, label="Time Decay Start")

# --- Highlight Best Peak ---
if best_run_data is not None:
    max_idx = best_run_data["smoothed_reward"].idxmax()
    best_step = best_run_data.loc[max_idx, "step_idx"]
    best_val = best_run_data.loc[max_idx, "smoothed_reward"]
    plt.plot(best_step, best_val, 'k*', markersize=12, label="Best Run Peak")
    plt.annotate("Best Run", (best_step, best_val), textcoords="offset points", xytext=(0,10), ha='center')

# Plot setup
plt.title("Smoothed Total Reward Comparison Across Runs")
plt.xlabel("Step")
plt.ylabel("Smoothed Reward")
plt.grid(True)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
plt.tight_layout()

# Save
plot_path = os.path.join(log_dir, "reward_comparison_all_runs.png")
plt.savefig(plot_path)
print(f"[Saved] {plot_path}")

plt.show()