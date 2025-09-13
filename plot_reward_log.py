"""
Plot SB3 monitor.csv or our CSV run log.
"""
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="monitor.csv or run.csv")
    p.add_argument("--out", type=str, default=None, help="save figure path")
    args = p.parse_args()

    df = pd.read_csv(args.path, comment="#")
    if {"t", "r"}.issubset(df.columns): # SB3 monitor schema
        x = df["t"].values
        y = df["r"].values
        title = "Monitor Reward"
    elif {"step", "reward"}.issubset(df.columns):
        x = df["step"].values
        y = df["reward"].values
        title = "Run Reward"
    else:
        print("Unknown CSV format:", df.columns.tolist())
        return

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, lw=1)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if args.out:
        plt.savefig(args.out, bbox_inches="tight")
        print("saved:", args.out)
    else:
        plt.show()

if __name__ == "__main__":
    main()