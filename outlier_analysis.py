import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\takit\Downloads\f1")
TRAIN_PATH = DATA_DIR / "train.csv"

TARGET = "PitNextLap"
ID_COL = "id"

OUTLIER_QUANTILES = [0.01, 0.05, 0.95, 0.99]
MIN_COUNT = 30

def analyze_outliers(train):
    base_rate = train[TARGET].mean()
    results = []

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [TARGET, ID_COL]]

    print("=" * 80)
    print("OUTLIER ANALYSIS REPORT")
    print("=" * 80)
    print(f"Rows: {len(train)}")
    print(f"Base PitNextLap rate: {base_rate:.4f}")
    print(f"Numeric columns analyzed: {len(numeric_cols)}")
    print("=" * 80)

    for col in numeric_cols:
        s = train[col].replace([np.inf, -np.inf], np.nan).dropna()

        if s.nunique() <= 2:
            continue

        q01 = s.quantile(0.01)
        q05 = s.quantile(0.05)
        q95 = s.quantile(0.95)
        q99 = s.quantile(0.99)

        conditions = [
            ("LOW_1%", train[col] <= q01, q01),
            ("LOW_5%", train[col] <= q05, q05),
            ("HIGH_95%", train[col] >= q95, q95),
            ("HIGH_99%", train[col] >= q99, q99),
        ]

        for label, mask, threshold in conditions:
            group = train[mask]

            if len(group) < MIN_COUNT:
                continue

            pit_rate = group[TARGET].mean()
            diff = pit_rate - base_rate
            lift = pit_rate / base_rate if base_rate > 0 else np.nan

            results.append({
                "feature": col,
                "outlier_type": label,
                "threshold": threshold,
                "count": len(group),
                "pit_rate": pit_rate,
                "base_rate": base_rate,
                "diff_vs_base": diff,
                "lift": lift,
            })

    report = pd.DataFrame(results)

    if report.empty:
        print("No outlier groups found.")
        return report

    report = report.sort_values(
        ["lift", "count"],
        ascending=[False, False]
    )

    print("\nTOP OUTLIERS THAT INCREASE PIT PROBABILITY")
    print("-" * 80)
    print(
        report[report["diff_vs_base"] > 0]
        .head(30)
        .to_string(index=False)
    )

    print("\nTOP OUTLIERS THAT DECREASE PIT PROBABILITY")
    print("-" * 80)
    print(
        report[report["diff_vs_base"] < 0]
        .sort_values(["lift", "count"], ascending=[True, False])
        .head(30)
        .to_string(index=False)
    )

    output_path = DATA_DIR / "outlier_analysis_report.csv"
    report.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved full report to: {output_path}")
    print("=" * 80)

    return report


def main():
    print("Loading train.csv...")
    train = pd.read_csv(TRAIN_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    analyze_outliers(train)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:")
        print(e)

    input("\nPress Enter to close...")
