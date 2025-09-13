"""
Simple success report from a run CSV file (produced by record_run_from_model.py).
Success = episode ended with done==1 (not truncated).
"""
from __future__ import annotations

import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    final = df.tail(1).iloc[0]
    done = bool(final["done"])
    trunc = bool(final["truncated"])
    ret = df["reward"].sum()

    print(f"return={ret:+.3f}  done={done}  truncated={trunc}")
    print("SUCCESS" if done and not trunc else "FAIL/INCOMPLETE")

if __name__ == "__main__":
    main()