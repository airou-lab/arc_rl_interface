import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_csv", type=str, required=True, help="Path to SB3 monitor.csv")
    p.add_argument("--out", type=str, default=None, help="Path to save PNG (optional)")
    args = p.parse_args()

    df = pd.read_csv(args.log_csv, skiprows=1)  # SB3 monitor: first row is a header comment
    if df.empty:
        print("No data in monitor.csv")
        return

    # Plot episode reward vs episode index
    plt.figure()
    plt.plot(df['r'].values, label='episode_reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, bbox_inches='tight')
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()