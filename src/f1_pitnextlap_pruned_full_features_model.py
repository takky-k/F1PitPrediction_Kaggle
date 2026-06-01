import re
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================

SEED = 42

N_SPLITS = 3
N_TRIALS = 10
N_ESTIMATORS = 20000
EARLY_STOPPING_ROUNDS = 300

# ============================================================
# Project paths
# ============================================================
# このファイルを C:\Projects\F1PitPrediction\src に置く前提。
# PROJECT_ROOT = C:\Projects\F1PitPrediction
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT

# Pruning uses the previous full_features_model importance file.
DEFAULT_IMPORTANCE_PATH = PROJECT_ROOT / "feature_importance_latest.csv"

# Global selected feature list.
# None means use all features.
SELECTED_FEATURES = None


TARGET = "PitNextLap"
ID_COL = "id"

BASE_CATEGORICAL_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Year",
]

# 0.939〜0.941付近を出していた元コードと同じ構造を維持
TARGET_ENCODING_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Race_Compound",
    "Driver_Compound",
    "Race_Year",
    "Compound_Stint_bin",
]

# 今回追加する安全寄りの数値特徴量
ADDED_NUMERIC_FEATURES = [
    "TyreLife_pct",
    "RemainingRace",
    "RemainingRace_x_TyreLife",
    "abs_delta_x_TyreLife",
    "Degradation_per_TyreLife",
    "LapTime_over_TyreLife",
    "PositionChange_abs",
    "RaceProgress_sq",
    "LapDelta_x_Degradation",
    "Position_x_LapDelta",
    "Position_x_Degradation",
    "PitStop_x_RaceProgress",
]

# 外れ値分析から追加するflag特徴量
ADDED_OUTLIER_FLAG_FEATURES = [
    "flag_TyreLife_high95",
    "flag_TyreLife_high99",
    "flag_TyreLife_low05",
    "flag_TyreLife_low01",
    "flag_Cumulative_Degradation_low05",
    "flag_Cumulative_Degradation_low01",
    "flag_Position_Change_high95",
    "flag_Position_Change_high99",
    "flag_LapNumber_high95",
    "flag_LapTime_Delta_high95",
    "flag_RaceProgress_low05",
    "flag_RaceProgress_high99",
    "flag_Stint_low05",
    "flag_old_tyre_and_slow",
    "flag_old_tyre_late_race",
    "flag_position_loss_and_slow",
]


# lag / rolling / stint trend features
# ここは自動生成されるので、実際の列数はログで確認する。
ADDED_TIME_SERIES_FEATURES = []


# ============================================================
# Utility
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)


def safe_col_name(x):
    x = str(x)
    x = re.sub(r"[^0-9a-zA-Z_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def safe_divide(a, b):
    return a / (b + 1e-6)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Logger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Run started: {now_str()}\n")

    def log(self, msg=""):
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="pruned_full_features_model",
    )

    parser.add_argument(
        "--baseline-cv",
        type=float,
        default=0.944658,
        help="Previous baseline CV AUC for comparison.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
    )

    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Disable Optuna and use default parameters.",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick test mode: fewer trials and fewer estimators.",
    )

    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause console at the end.",
    )

    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable feature pruning and use all generated features.",
    )

    parser.add_argument(
        "--importance-path",
        type=str,
        default=str(DEFAULT_IMPORTANCE_PATH),
        help="Path to previous feature_importance_latest.csv used for pruning.",
    )

    parser.add_argument(
        "--keep-top-n",
        type=int,
        default=150,
        help="Keep top N features by previous gain importance.",
    )

    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="Minimum gain_importance_mean required to keep a feature.",
    )

    return parser.parse_args()


# ============================================================
# Load data
# ============================================================

def load_data(data_dir: Path, logger):
    logger.log(f"Path to competition files: {data_dir}")

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    missing_files = [
        str(p)
        for p in [train_path, test_path]
        if not p.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing_files)
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if sample_path.exists():
        sample_submission = pd.read_csv(sample_path)
    else:
        sample_submission = None

    return train, test, sample_submission


# ============================================================
# EDA
# ============================================================

def print_data_overview(train, test, logger):
    logger.log("\n========== Data overview ==========")
    logger.log(f"train shape: {train.shape}")
    logger.log(f"test shape : {test.shape}")

    logger.log("\nTrain columns:")
    logger.log(train.columns.tolist())

    logger.log("\nMissing values:")
    logger.log(
        train.isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )

    logger.log("\nTarget distribution:")
    logger.log(train[TARGET].value_counts().to_string())


# ============================================================
# Outlier parameters
# ============================================================

def fit_outlier_params(train):
    clip_cols = [
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
    ]

    # 外れ値flag用の分位点
    quantile_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "Position_Change",
        "LapNumber",
        "LapTime_Delta",
        "RaceProgress",
        "Stint",
    ]

    params = {
        "clip": {},
        "q": {},
        "compound_levels": sorted(
            train["Compound"].astype(str).unique().tolist()
        ),
    }

    for col in clip_cols:
        params["clip"][col] = {
            "low": float(train[col].quantile(0.005)),
            "high": float(train[col].quantile(0.995)),
        }

    for col in quantile_cols:
        params["q"][col] = {
            "q01": float(train[col].quantile(0.01)),
            "q05": float(train[col].quantile(0.05)),
            "q95": float(train[col].quantile(0.95)),
            "q99": float(train[col].quantile(0.99)),
        }

    return params


# ============================================================
# Time-series / lag / rolling features
# ============================================================

def add_time_series_features(df):
    """
    Race x Driver x LapNumber の順番で、過去ラップ情報を特徴量化する。
    重要：
    - rolling特徴量は shift(1) してから作るので、現在行より未来の情報は使わない。
    - train/testは別々に処理する。targetは使わない。
    - 元の行順に戻して返す。
    """
    global ADDED_TIME_SERIES_FEATURES

    df = df.copy()
    df["__orig_index"] = np.arange(len(df))

    sort_cols = ["Race", "Driver", "LapNumber"]
    if "Stint" in df.columns:
        sort_cols = ["Race", "Driver", "Stint", "LapNumber"]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    group_rd = df.groupby(["Race", "Driver"], observed=False, sort=False)

    base_cols = [
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "Position",
        "Position_Change",
        "TyreLife",
        "RaceProgress",
        "PitStop",
    ]

    base_cols = [c for c in base_cols if c in df.columns]

    created = []

    # ----------------------------
    # Lag features
    # ----------------------------
    for col in base_cols:
        safe = safe_col_name(col)

        for lag in [1, 2, 3]:
            new_col = f"{safe}_lag{lag}_by_driver_race"
            df[new_col] = group_rd[col].shift(lag)
            created.append(new_col)

        new_col = f"{safe}_diff_lag1_by_driver_race"
        df[new_col] = df[col] - group_rd[col].shift(1)
        created.append(new_col)

        new_col = f"{safe}_pct_change_lag1_by_driver_race"
        df[new_col] = safe_divide(
            df[col] - group_rd[col].shift(1),
            np.abs(group_rd[col].shift(1)) + 1
        )
        created.append(new_col)

    # ----------------------------
    # Rolling features
    # ----------------------------
    rolling_cols = [
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "Position_Change",
        "TyreLife",
        "Position",
    ]

    rolling_cols = [c for c in rolling_cols if c in df.columns]

    for col in rolling_cols:
        safe = safe_col_name(col)
        shifted = group_rd[col].shift(1)

        for window in [3, 5]:
            roll = shifted.groupby(
                [df["Race"], df["Driver"]],
                observed=False,
                sort=False
            ).rolling(window=window, min_periods=1)

            mean_col = f"{safe}_roll{window}_mean_by_driver_race"
            max_col = f"{safe}_roll{window}_max_by_driver_race"
            min_col = f"{safe}_roll{window}_min_by_driver_race"
            std_col = f"{safe}_roll{window}_std_by_driver_race"

            df[mean_col] = roll.mean().reset_index(level=[0, 1], drop=True)
            df[max_col] = roll.max().reset_index(level=[0, 1], drop=True)
            df[min_col] = roll.min().reset_index(level=[0, 1], drop=True)
            df[std_col] = roll.std().reset_index(level=[0, 1], drop=True).fillna(0)

            created.extend([mean_col, max_col, min_col, std_col])

            # current vs recent average
            diff_col = f"{safe}_minus_roll{window}_mean_by_driver_race"
            df[diff_col] = df[col] - df[mean_col]
            created.append(diff_col)

    # ----------------------------
    # Stint-level trend features
    # ----------------------------
    group_rds = df.groupby(["Race", "Driver", "Stint"], observed=False, sort=False)

    df["stint_lap_index"] = group_rds.cumcount() + 1
    created.append("stint_lap_index")

    df["stint_lap_index_pct_of_lap"] = safe_divide(
        df["stint_lap_index"],
        df["LapNumber"] + 1
    )
    created.append("stint_lap_index_pct_of_lap")

    stint_cols = [
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "Position",
        "Position_Change",
        "TyreLife",
    ]

    stint_cols = [c for c in stint_cols if c in df.columns]

    for col in stint_cols:
        safe = safe_col_name(col)

        start_col = f"{safe}_stint_start"
        since_col = f"{safe}_since_stint_start"
        diff_prev_col = f"{safe}_stint_diff_prev"
        cummean_col = f"{safe}_stint_expanding_mean_prev"
        cumsum_col = f"{safe}_stint_cumsum_prev"

        df[start_col] = group_rds[col].transform("first")
        df[since_col] = df[col] - df[start_col]
        df[diff_prev_col] = df[col] - group_rds[col].shift(1)

        shifted = group_rds[col].shift(1)
        df[cummean_col] = (
            shifted.groupby(
                [df["Race"], df["Driver"], df["Stint"]],
                observed=False,
                sort=False
            )
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )

        df[cumsum_col] = (
            shifted.groupby(
                [df["Race"], df["Driver"], df["Stint"]],
                observed=False,
                sort=False
            )
            .cumsum()
        )

        created.extend([start_col, since_col, diff_prev_col, cummean_col, cumsum_col])

    # ----------------------------
    # Domain interaction features using lag/rolling/trend
    # ----------------------------
    if "LapTime_Delta_lag1_by_driver_race" in df.columns:
        df["TyreLife_x_LapDelta_lag1"] = (
            df["TyreLife"] * df["LapTime_Delta_lag1_by_driver_race"]
        )
        created.append("TyreLife_x_LapDelta_lag1")

    if "LapTime_Delta_roll3_mean_by_driver_race" in df.columns:
        df["TyreLife_x_LapDelta_roll3_mean"] = (
            df["TyreLife"] * df["LapTime_Delta_roll3_mean_by_driver_race"]
        )
        created.append("TyreLife_x_LapDelta_roll3_mean")

    if "Cumulative_Degradation_since_stint_start" in df.columns:
        df["TyreLife_x_Degradation_since_stint_start"] = (
            df["TyreLife"] * df["Cumulative_Degradation_since_stint_start"]
        )
        created.append("TyreLife_x_Degradation_since_stint_start")

    if "LapTime_Delta_since_stint_start" in df.columns:
        df["RaceProgress_x_LapDelta_since_stint_start"] = (
            df["RaceProgress"] * df["LapTime_Delta_since_stint_start"]
        )
        created.append("RaceProgress_x_LapDelta_since_stint_start")

    if "Position_Change_roll3_mean_by_driver_race" in df.columns:
        df["Position_x_PositionChange_roll3_mean"] = (
            df["Position"] * df["Position_Change_roll3_mean_by_driver_race"]
        )
        created.append("Position_x_PositionChange_roll3_mean")

    # restore original order
    df = df.sort_values("__orig_index").drop(columns=["__orig_index"]).reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)

    # keep unique feature names
    ADDED_TIME_SERIES_FEATURES = sorted(list(dict.fromkeys(created)))

    return df


# ============================================================
# Feature engineering
# ============================================================

def add_basic_features(df, params):
    df = df.copy()

    # ----------------------------
    # Categorical
    # ----------------------------

    for col in BASE_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")

    # ----------------------------
    # Clipping
    # ----------------------------

    for col, lim in params["clip"].items():
        new_col = safe_col_name(f"clipped_{col}")
        df[new_col] = df[col].clip(lim["low"], lim["high"])

    # ----------------------------
    # Original stable features
    # ----------------------------

    df["log1p_TyreLife"] = np.log1p(
        df["TyreLife"].clip(lower=0)
    )

    df["late_race"] = (
        df["RaceProgress"] > 0.70
    ).astype("int8")

    df["early_race"] = (
        df["RaceProgress"] < 0.20
    ).astype("int8")

    df["top3"] = (
        df["Position"] <= 3
    ).astype("int8")

    df["backmarker"] = (
        df["Position"] >= 16
    ).astype("int8")

    df["abs_lap_delta"] = np.abs(
        df["LapTime_Delta"]
    )

    df["TyreLife_x_RaceProgress"] = (
        df["TyreLife"] * df["RaceProgress"]
    )

    df["TyreLife_x_Stint"] = (
        df["TyreLife"] * df["Stint"]
    )

    # ----------------------------
    # New safe numeric features
    # ----------------------------

    df["TyreLife_pct"] = safe_divide(
        df["TyreLife"],
        df["LapNumber"] + 1,
    )

    df["RemainingRace"] = 1 - df["RaceProgress"]

    df["RemainingRace_x_TyreLife"] = (
        df["RemainingRace"] * df["TyreLife"]
    )

    df["abs_delta_x_TyreLife"] = (
        df["abs_lap_delta"] * df["TyreLife"]
    )

    df["Degradation_per_TyreLife"] = safe_divide(
        df["Cumulative_Degradation"],
        df["TyreLife"] + 1,
    )

    df["LapTime_over_TyreLife"] = safe_divide(
        df["LapTime (s)"],
        df["TyreLife"] + 1,
    )

    df["PositionChange_abs"] = np.abs(
        df["Position_Change"]
    )

    df["RaceProgress_sq"] = (
        df["RaceProgress"] ** 2
    )

    df["LapDelta_x_Degradation"] = (
        df["LapTime_Delta"] * df["Cumulative_Degradation"]
    )

    df["Position_x_LapDelta"] = (
        df["Position"] * df["LapTime_Delta"]
    )

    df["Position_x_Degradation"] = (
        df["Position"] * df["Cumulative_Degradation"]
    )

    df["PitStop_x_RaceProgress"] = (
        df["PitStop"] * df["RaceProgress"]
    )

    # ----------------------------
    # Outlier flag features
    # trainでfitした分位点をtestにもそのまま使う
    # ----------------------------

    q = params["q"]

    df["flag_TyreLife_high95"] = (
        df["TyreLife"] >= q["TyreLife"]["q95"]
    ).astype("int8")

    df["flag_TyreLife_high99"] = (
        df["TyreLife"] >= q["TyreLife"]["q99"]
    ).astype("int8")

    df["flag_TyreLife_low05"] = (
        df["TyreLife"] <= q["TyreLife"]["q05"]
    ).astype("int8")

    df["flag_TyreLife_low01"] = (
        df["TyreLife"] <= q["TyreLife"]["q01"]
    ).astype("int8")

    df["flag_Cumulative_Degradation_low05"] = (
        df["Cumulative_Degradation"] <= q["Cumulative_Degradation"]["q05"]
    ).astype("int8")

    df["flag_Cumulative_Degradation_low01"] = (
        df["Cumulative_Degradation"] <= q["Cumulative_Degradation"]["q01"]
    ).astype("int8")

    df["flag_Position_Change_high95"] = (
        df["Position_Change"] >= q["Position_Change"]["q95"]
    ).astype("int8")

    df["flag_Position_Change_high99"] = (
        df["Position_Change"] >= q["Position_Change"]["q99"]
    ).astype("int8")

    df["flag_LapNumber_high95"] = (
        df["LapNumber"] >= q["LapNumber"]["q95"]
    ).astype("int8")

    df["flag_LapTime_Delta_high95"] = (
        df["LapTime_Delta"] >= q["LapTime_Delta"]["q95"]
    ).astype("int8")

    df["flag_RaceProgress_low05"] = (
        df["RaceProgress"] <= q["RaceProgress"]["q05"]
    ).astype("int8")

    df["flag_RaceProgress_high99"] = (
        df["RaceProgress"] >= q["RaceProgress"]["q99"]
    ).astype("int8")

    df["flag_Stint_low05"] = (
        df["Stint"] <= q["Stint"]["q05"]
    ).astype("int8")

    # 組み合わせflag：単体よりF1の文脈を持たせる
    df["flag_old_tyre_and_slow"] = (
        (df["TyreLife"] >= q["TyreLife"]["q95"])
        & (df["LapTime_Delta"] >= q["LapTime_Delta"]["q95"])
    ).astype("int8")

    df["flag_old_tyre_late_race"] = (
        (df["TyreLife"] >= q["TyreLife"]["q95"])
        & (df["RaceProgress"] >= 0.70)
    ).astype("int8")

    df["flag_position_loss_and_slow"] = (
        (df["Position_Change"] >= q["Position_Change"]["q95"])
        & (df["LapTime_Delta"] >= q["LapTime_Delta"]["q95"])
    ).astype("int8")

    # ----------------------------
    # Original categorical interactions
    # Keep stable structure
    # ----------------------------

    df["Race_Compound"] = (
        df["Race"].astype(str)
        + "__"
        + df["Compound"].astype(str)
    )

    df["Driver_Compound"] = (
        df["Driver"].astype(str)
        + "__"
        + df["Compound"].astype(str)
    )

    df["Race_Year"] = (
        df["Race"].astype(str)
        + "__"
        + df["Year"].astype(str)
    )

    df["Stint_bin"] = pd.cut(
        df["Stint"],
        bins=[0, 1, 2, 3, 4, 99],
        labels=["1", "2", "3", "4", "5plus"],
        include_lowest=True,
    ).astype(str)

    df["Compound_Stint_bin"] = (
        df["Compound"].astype(str)
        + "__"
        + df["Stint_bin"].astype(str)
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# Frequency encoding
# ============================================================

def add_frequency_encoding(train_fe, test_fe, cols):
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()

    all_df = pd.concat(
        [train_fe[cols], test_fe[cols]],
        axis=0,
        ignore_index=True,
    )

    for col in cols:
        freq = (
            all_df[col]
            .astype(str)
            .value_counts(normalize=True)
        )

        train_fe[f"{col}_frequency"] = (
            train_fe[col]
            .astype(str)
            .map(freq)
            .fillna(0)
            .astype("float32")
        )

        test_fe[f"{col}_frequency"] = (
            test_fe[col]
            .astype(str)
            .map(freq)
            .fillna(0)
            .astype("float32")
        )

    return train_fe, test_fe


# ============================================================
# Align categorical dtypes
# ============================================================

def align_categorical_dtypes(train_fe, test_fe, cat_cols):
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()

    for col in cat_cols:
        both = pd.concat([
            train_fe[col].astype(str),
            test_fe[col].astype(str),
        ])

        categories = sorted(
            both.fillna("missing").unique().tolist()
        )

        dtype = pd.CategoricalDtype(categories=categories)

        train_fe[col] = (
            train_fe[col]
            .astype(str)
            .fillna("missing")
            .astype(dtype)
        )

        test_fe[col] = (
            test_fe[col]
            .astype(str)
            .fillna("missing")
            .astype(dtype)
        )

    return train_fe, test_fe


# ============================================================
# Target encoding
# ============================================================

def apply_target_encoding(
    train_part,
    valid_part,
    test_fe,
    cols,
    target=TARGET,
    smoothing=30.0,
):
    train_part = train_part.copy()
    valid_part = valid_part.copy()
    test_part = test_fe.copy()

    global_mean = float(train_part[target].mean())

    for col in cols:
        stats = (
            train_part
            .groupby(col, observed=False)[target]
            .agg(["mean", "count"])
        )

        smooth = (
            (stats["count"] * stats["mean"])
            + (smoothing * global_mean)
        ) / (
            stats["count"] + smoothing
        )

        mapping = smooth.to_dict()
        new_col = f"{col}_target_encoded"

        train_part[new_col] = (
            train_part[col]
            .map(mapping)
            .astype(float)
            .fillna(global_mean)
        )

        valid_part[new_col] = (
            valid_part[col]
            .map(mapping)
            .astype(float)
            .fillna(global_mean)
        )

        test_part[new_col] = (
            test_part[col]
            .map(mapping)
            .astype(float)
            .fillna(global_mean)
        )

    return train_part, valid_part, test_part


# ============================================================
# Feature columns
# ============================================================

def get_feature_columns(train_fe):
    """
    Return feature columns.

    If SELECTED_FEATURES is set, only use those features.
    This pruning is applied after fold target encoding as well, so
    target-encoded columns can be kept when they exist in tr_part/va_part.
    """
    exclude = {
        ID_COL,
        TARGET,
    }

    all_features = [
        c for c in train_fe.columns
        if c not in exclude
    ]

    global SELECTED_FEATURES

    if SELECTED_FEATURES is None:
        return all_features

    selected_existing = [
        c for c in SELECTED_FEATURES
        if c in train_fe.columns and c not in exclude
    ]

    return selected_existing


# ============================================================
# Feature pruning
# ============================================================

def load_pruned_feature_list(
    importance_path,
    keep_top_n,
    min_gain,
    logger,
):
    """
    Previous full_features_modelのfeature_importance_latest.csvを使って、
    重要度が低すぎる特徴量を削る。

    方針：
    - gain_importance_mean が高い順に並べる
    - keep_top_n まで残す
    - gain_importance_mean <= min_gain は除外
    - target encoded featuresも普通に残す
    - ファイルがない場合は pruning せず None を返す
    """
    importance_path = Path(importance_path)

    logger.log("\n========== Feature pruning ==========")
    logger.log(f"importance_path: {importance_path}")
    logger.log(f"keep_top_n     : {keep_top_n}")
    logger.log(f"min_gain       : {min_gain}")

    if not importance_path.exists():
        logger.log("Importance file not found. Pruning disabled.")
        return None, pd.DataFrame()

    imp = pd.read_csv(importance_path)

    required_cols = {"feature", "gain_importance_mean"}

    if not required_cols.issubset(set(imp.columns)):
        logger.log("Importance file does not have required columns. Pruning disabled.")
        logger.log(f"columns: {imp.columns.tolist()}")
        return None, imp

    imp = imp.copy()
    imp["gain_importance_mean"] = pd.to_numeric(
        imp["gain_importance_mean"],
        errors="coerce",
    ).fillna(0.0)

    if "gain_rank" in imp.columns:
        imp["gain_rank"] = pd.to_numeric(
            imp["gain_rank"],
            errors="coerce",
        )
        imp = imp.sort_values(
            ["gain_rank", "gain_importance_mean"],
            ascending=[True, False],
        )
    else:
        imp = imp.sort_values(
            "gain_importance_mean",
            ascending=False,
        )
        imp["gain_rank"] = np.arange(1, len(imp) + 1)

    before_count = len(imp)

    keep = imp[
        imp["gain_importance_mean"] > float(min_gain)
    ].copy()

    keep = keep.head(int(keep_top_n)).copy()

    selected = keep["feature"].astype(str).tolist()

    logger.log(f"features in importance file : {before_count}")
    logger.log(f"selected feature count      : {len(selected)}")

    logger.log("\nTop selected features:")
    logger.log(
        keep[
            [
                "feature",
                "gain_rank",
                "gain_importance_mean",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    logger.log("\nBottom selected features:")
    logger.log(
        keep[
            [
                "feature",
                "gain_rank",
                "gain_importance_mean",
            ]
        ]
        .tail(20)
        .to_string(index=False)
    )

    dropped = imp[~imp["feature"].astype(str).isin(selected)].copy()

    logger.log("\nTop dropped features:")
    if dropped.empty:
        logger.log("No dropped features.")
    else:
        logger.log(
            dropped[
                [
                    "feature",
                    "gain_rank",
                    "gain_importance_mean",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    return selected, imp


def save_selected_features(
    selected_features,
    output_path,
):
    output_path = Path(output_path)

    payload = {
        "selected_feature_count": None if selected_features is None else len(selected_features),
        "selected_features": selected_features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Prepare features
# ============================================================

def prepare_base_features(train, test, params, logger):
    train_fe = add_basic_features(train, params)
    test_fe = add_basic_features(test, params)

    # lag / rolling / Race-Driver-Stint trend features
    train_fe = add_time_series_features(train_fe)
    test_fe = add_time_series_features(test_fe)

    freq_cols = TARGET_ENCODING_COLS.copy()

    train_fe, test_fe = add_frequency_encoding(
        train_fe,
        test_fe,
        freq_cols,
    )

    categorical_cols = BASE_CATEGORICAL_COLS + [
        "Stint_bin",
        "Race_Compound",
        "Driver_Compound",
        "Race_Year",
        "Compound_Stint_bin",
    ]

    train_fe, test_fe = align_categorical_dtypes(
        train_fe,
        test_fe,
        categorical_cols,
    )

    logger.log("\n========== Feature summary ==========")
    logger.log(f"train_fe shape: {train_fe.shape}")
    logger.log(f"test_fe shape : {test_fe.shape}")
    logger.log(f"feature cols before fold target encoding: {len(get_feature_columns(train_fe))}")
    logger.log(f"categorical cols: {len(categorical_cols)}")
    logger.log(f"target encoding cols: {len(TARGET_ENCODING_COLS)}")

    logger.log("\nAdded numeric features:")
    for f in ADDED_NUMERIC_FEATURES:
        logger.log(f"- {f}")

    logger.log("\nAdded outlier flag features:")
    for f in ADDED_OUTLIER_FLAG_FEATURES:
        logger.log(f"- {f}")

    logger.log("\nAdded time-series features:")
    logger.log(f"count: {len(ADDED_TIME_SERIES_FEATURES)}")
    for f in ADDED_TIME_SERIES_FEATURES[:120]:
        logger.log(f"- {f}")
    if len(ADDED_TIME_SERIES_FEATURES) > 120:
        logger.log(f"... and {len(ADDED_TIME_SERIES_FEATURES) - 120} more")

    return train_fe, test_fe, categorical_cols


# ============================================================
# LightGBM params
# ============================================================

def make_default_lgbm_params(scale_pos_weight):
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",

        "n_estimators": N_ESTIMATORS,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,

        # 直近でCV/LBが良かったparamsをdefaultに変更
        "learning_rate": 0.015484014792532953,
        "num_leaves": 192,
        "max_depth": 5,
        "min_child_samples": 122,
        "subsample": 0.8646188322585528,
        "colsample_bytree": 0.6501186041193369,
        "reg_alpha": 3.643035463653499e-05,
        "reg_lambda": 1.91211447920584,
        "scale_pos_weight": 4.58378394553113,
    }


def make_lgbm_params(trial, scale_pos_weight):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",

        "n_estimators": N_ESTIMATORS,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.005,
            0.08,
            log=True,
        ),

        "num_leaves": trial.suggest_int(
            "num_leaves",
            31,
            255,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            4,
            12,
        ),

        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            20,
            400,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            30.0,
            log=True,
        ),

        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(0.5, scale_pos_weight * 0.5),
            scale_pos_weight * 1.5,
        ),
    }

    return params


def get_enqueue_trial_params():
    """
    直近で良かったtrialに近い設定を最初に試す。
    """
    return {
        "learning_rate": 0.015484014792532953,
        "num_leaves": 192,
        "max_depth": 5,
        "min_child_samples": 122,
        "subsample": 0.8646188322585528,
        "colsample_bytree": 0.6501186041193369,
        "reg_alpha": 3.643035463653499e-05,
        "reg_lambda": 1.91211447920584,
        "scale_pos_weight": 4.58378394553113,
    }


# ============================================================
# CV training
# ============================================================

def train_one_cv(
    train_fe,
    test_fe,
    categorical_cols,
    lgbm_params,
    logger,
    return_test_preds=False,
    return_importance=False,
    trial_number=None,
):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = train_fe[TARGET].astype(int).values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(len(train_fe), dtype=np.float32)

    if return_test_preds:
        test_preds = np.zeros(len(test_fe), dtype=np.float32)
    else:
        test_preds = None

    aucs = []
    best_iters = []
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(train_fe, y),
        start=1,
    ):
        fold_start = time.time()

        tr_part = train_fe.iloc[tr_idx].copy()
        va_part = train_fe.iloc[va_idx].copy()

        tr_part, va_part, test_part_fold = apply_target_encoding(
            tr_part,
            va_part,
            test_fe,
            cols=TARGET_ENCODING_COLS,
        )

        feature_cols = get_feature_columns(tr_part)

        X_tr = tr_part[feature_cols]
        y_tr = tr_part[TARGET]

        X_va = va_part[feature_cols]
        y_va = va_part[TARGET]

        cats = [
            c for c in categorical_cols
            if c in feature_cols
        ]

        model = lgb.LGBMClassifier(**lgbm_params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cats,
            callbacks=[
                lgb.early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            ],
        )

        va_pred = model.predict_proba(
            X_va,
            num_iteration=model.best_iteration_,
        )[:, 1]

        oof[va_idx] = va_pred

        auc = roc_auc_score(y_va, va_pred)
        aucs.append(float(auc))

        best_iter = int(model.best_iteration_)
        best_iters.append(best_iter)

        elapsed = time.time() - fold_start

        prefix = f"Trial {trial_number} | " if trial_number is not None else ""

        logger.log(
            f"{prefix}Fold {fold}: AUC = {auc:.6f} | "
            f"best_iter = {best_iter} | "
            f"time = {elapsed:.1f}s"
        )

        if return_importance:
            split_imp = model.feature_importances_
            gain_imp = model.booster_.feature_importance(
                importance_type="gain"
            )

            imp = pd.DataFrame({
                "feature": feature_cols,
                f"split_importance_fold_{fold}": split_imp,
                f"gain_importance_fold_{fold}": gain_imp,
            })

            importances.append(imp)

        if return_test_preds:
            pred = model.predict_proba(
                test_part_fold[feature_cols],
                num_iteration=model.best_iteration_,
            )[:, 1]

            test_preds += pred / N_SPLITS

    mean_auc = float(np.mean(aucs))

    logger.log(f"Mean CV AUC: {mean_auc:.6f}")
    logger.log(f"Best iterations: {best_iters}")

    if return_importance and importances:
        importance_df = importances[0]

        for imp in importances[1:]:
            importance_df = importance_df.merge(
                imp,
                on="feature",
                how="outer",
            )

        split_cols = [
            c for c in importance_df.columns
            if c.startswith("split_importance_fold_")
        ]

        gain_cols = [
            c for c in importance_df.columns
            if c.startswith("gain_importance_fold_")
        ]

        importance_df["split_importance_mean"] = (
            importance_df[split_cols].mean(axis=1)
        )

        importance_df["gain_importance_mean"] = (
            importance_df[gain_cols].mean(axis=1)
        )

        importance_df["gain_importance_pct"] = (
            importance_df["gain_importance_mean"]
            / importance_df["gain_importance_mean"].sum()
            * 100
        )

        importance_df = importance_df.sort_values(
            "gain_importance_mean",
            ascending=False,
        )

        importance_df["gain_rank"] = np.arange(1, len(importance_df) + 1)

    else:
        importance_df = None

    return mean_auc, oof, test_preds, importance_df, best_iters


# ============================================================
# Feature effect reports
# ============================================================

def analyze_binary_flag_effects(train_fe, flag_cols, logger):
    """
    各flagが1のときと0のときで、実際のPitNextLap率がどれくらい違うかを見る。
    これは「特徴量として入れる前の実データ上の効果」を見る表。
    """
    rows = []
    base_rate = float(train_fe[TARGET].mean())
    n_total = len(train_fe)

    for col in flag_cols:
        if col not in train_fe.columns:
            continue

        x = train_fe[col].fillna(0).astype(int)
        mask1 = x == 1
        mask0 = x == 0

        n1 = int(mask1.sum())
        n0 = int(mask0.sum())

        if n1 == 0 or n0 == 0:
            continue

        rate1 = float(train_fe.loc[mask1, TARGET].mean())
        rate0 = float(train_fe.loc[mask0, TARGET].mean())

        rows.append({
            "feature": col,
            "count_flag_1": n1,
            "count_flag_0": n0,
            "coverage_pct": n1 / n_total * 100,
            "pit_rate_when_1": rate1,
            "pit_rate_when_0": rate0,
            "base_rate": base_rate,
            "diff_1_vs_0": rate1 - rate0,
            "diff_1_vs_base": rate1 - base_rate,
            "lift_vs_base": rate1 / base_rate if base_rate > 0 else np.nan,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["lift_vs_base", "coverage_pct"],
            ascending=[False, False],
        )

    logger.log("\n========== Outlier flag effect report ==========")

    if df.empty:
        logger.log("No flag effect rows created.")
    else:
        logger.log(
            df[
                [
                    "feature",
                    "count_flag_1",
                    "coverage_pct",
                    "pit_rate_when_1",
                    "pit_rate_when_0",
                    "base_rate",
                    "diff_1_vs_base",
                    "lift_vs_base",
                ]
            ]
            .to_string(index=False)
        )

    return df


def analyze_oof_by_flags(train_fe, oof, flag_cols, logger):
    """
    flag別にOOF予測が実際のPitNextLap率をどれくらい再現できているかを見る。
    actual_rateとpred_rateが近いほど、そのflag領域でcalibrationが良い。
    """
    rows = []

    temp = train_fe[[TARGET] + [c for c in flag_cols if c in train_fe.columns]].copy()
    temp["oof_pred"] = oof

    for col in flag_cols:
        if col not in temp.columns:
            continue

        x = temp[col].fillna(0).astype(int)

        for value in [0, 1]:
            mask = x == value
            n = int(mask.sum())

            if n == 0:
                continue

            actual_rate = float(temp.loc[mask, TARGET].mean())
            pred_rate = float(temp.loc[mask, "oof_pred"].mean())

            rows.append({
                "feature": col,
                "flag_value": value,
                "count": n,
                "actual_pit_rate": actual_rate,
                "mean_oof_pred": pred_rate,
                "pred_minus_actual": pred_rate - actual_rate,
            })

    df = pd.DataFrame(rows)

    logger.log("\n========== OOF calibration by outlier flags ==========")

    if df.empty:
        logger.log("No OOF flag calibration rows created.")
    else:
        logger.log(
            df[df["flag_value"] == 1]
            .sort_values("actual_pit_rate", ascending=False)
            .to_string(index=False)
        )

    return df


def create_feature_effect_report(
    train_fe,
    oof,
    importance_df,
    flag_cols,
    output_dir,
    logger,
):
    """
    最終的に以下を保存する：
    1. outlier_flag_effects.csv
       - flag=1の実際のPitNextLap率、lift、coverageを見る
    2. oof_calibration_by_flags.csv
       - flag領域でモデル予測が高め/低めに出ているかを見る
    3. feature_effect_report.csv
       - LightGBM gain importanceとflag効果を結合した総合表
    """
    output_dir = Path(output_dir)

    flag_effect_df = analyze_binary_flag_effects(
        train_fe=train_fe,
        flag_cols=flag_cols,
        logger=logger,
    )

    oof_flag_df = analyze_oof_by_flags(
        train_fe=train_fe,
        oof=oof,
        flag_cols=flag_cols,
        logger=logger,
    )

    flag_effect_path = output_dir / "outlier_flag_effects.csv"
    oof_flag_path = output_dir / "oof_calibration_by_flags.csv"
    feature_effect_path = output_dir / "feature_effect_report.csv"

    flag_effect_df.to_csv(flag_effect_path, index=False)
    oof_flag_df.to_csv(oof_flag_path, index=False)

    if importance_df is not None:
        cols = [
            "feature",
            "gain_rank",
            "gain_importance_mean",
            "gain_importance_pct",
            "split_importance_mean",
        ]

        report = importance_df[cols].copy()

        if not flag_effect_df.empty:
            report = report.merge(
                flag_effect_df,
                on="feature",
                how="left",
            )

        report["is_outlier_flag"] = report["feature"].isin(flag_cols).astype("int8")

        report.to_csv(feature_effect_path, index=False)

        logger.log("\n========== Outlier flags in model importance ==========")
        outlier_imp = report[report["is_outlier_flag"] == 1].copy()

        if outlier_imp.empty:
            logger.log("No outlier flags found in importance report.")
        else:
            logger.log(
                outlier_imp[
                    [
                        "feature",
                        "gain_rank",
                        "gain_importance_mean",
                        "gain_importance_pct",
                        "split_importance_mean",
                        "coverage_pct",
                        "pit_rate_when_1",
                        "base_rate",
                        "lift_vs_base",
                    ]
                ]
                .sort_values("gain_rank")
                .to_string(index=False)
            )

        logger.log("\n========== Top 30 total feature effect report ==========")
        logger.log(
            report.head(30).to_string(index=False)
        )
    else:
        report = pd.DataFrame()

    logger.log(f"\nSaved outlier flag effects      : {flag_effect_path}")
    logger.log(f"Saved OOF calibration by flags  : {oof_flag_path}")
    logger.log(f"Saved feature effect report     : {feature_effect_path}")

    return flag_effect_df, oof_flag_df, report


# ============================================================
# Optuna
# ============================================================

def run_optuna(
    train_fe,
    test_fe,
    categorical_cols,
    output_dir,
    logger,
):
    import optuna

    y = train_fe[TARGET].astype(int)

    neg = int((y == 0).sum())
    pos = int((y == 1).sum())

    scale_pos_weight = neg / max(pos, 1)

    logger.log("\n========== Optuna tuning ==========")
    logger.log(f"scale_pos_weight base: {scale_pos_weight:.4f}")

    trial_records = []

    def objective(trial):
        params = make_lgbm_params(
            trial,
            scale_pos_weight,
        )

        auc, _, _, _, best_iters = train_one_cv(
            train_fe,
            test_fe,
            categorical_cols,
            params,
            logger,
            return_test_preds=False,
            return_importance=False,
            trial_number=trial.number,
        )

        record = {
            "trial": trial.number,
            "cv_auc": auc,
            "best_iter_mean": float(np.mean(best_iters)),
            "best_iter_min": int(np.min(best_iters)),
            "best_iter_max": int(np.max(best_iters)),
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "max_depth": params["max_depth"],
            "min_child_samples": params["min_child_samples"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "scale_pos_weight": params["scale_pos_weight"],
        }

        trial_records.append(record)

        trial_df = (
            pd.DataFrame(trial_records)
            .sort_values("cv_auc", ascending=False)
        )

        trial_df.to_csv(
            Path(output_dir) / "optuna_trials_live.csv",
            index=False,
        )

        logger.log("\nCurrent best trials:")
        logger.log(
            trial_df[
                [
                    "trial",
                    "cv_auc",
                    "best_iter_mean",
                    "learning_rate",
                    "num_leaves",
                    "max_depth",
                    "min_child_samples",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                    "scale_pos_weight",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

        return auc

    study = optuna.create_study(
        direction="maximize"
    )

    # 最近良かった設定を最初に試す
    study.enqueue_trial(
        get_enqueue_trial_params()
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    trials_df = (
        pd.DataFrame(trial_records)
        .sort_values("cv_auc", ascending=False)
    )

    trials_path = Path(output_dir) / "optuna_trials.csv"

    trials_df.to_csv(
        trials_path,
        index=False,
    )

    logger.log("\nBest Optuna score:")
    logger.log(study.best_value)

    logger.log("\nBest Optuna params:")
    logger.log(study.best_params)

    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": N_ESTIMATORS,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        **study.best_params,
    }

    return best_params, float(study.best_value), study.best_params


# ============================================================
# Experiment summary
# ============================================================

def append_experiment_summary(
    summary_path,
    record,
):
    summary_path = Path(summary_path)

    if summary_path.exists():
        old = pd.read_csv(summary_path)
        new = pd.concat(
            [old, pd.DataFrame([record])],
            axis=0,
            ignore_index=True,
        )
    else:
        new = pd.DataFrame([record])

    new.to_csv(
        summary_path,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main():
    global N_TRIALS
    global N_ESTIMATORS
    global EARLY_STOPPING_ROUNDS
    global SELECTED_FEATURES

    args = parse_args()

    data_dir = Path(args.data_dir)

    output_base_dir = Path(args.output_dir)

    output_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fast:
        N_TRIALS = 3
        N_ESTIMATORS = 5000
        EARLY_STOPPING_ROUNDS = 150
    else:
        N_TRIALS = int(args.n_trials)

    run_id = f"{args.run_name}_{timestamp_str()}"
    run_dir = output_base_dir / "runs" / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = Logger(
        run_dir / "run_log.txt"
    )

    logger.log("========== CONFIG ==========")
    logger.log(f"run_id: {run_id}")
    logger.log(f"N_SPLITS: {N_SPLITS}")
    logger.log(f"N_TRIALS: {N_TRIALS}")
    logger.log(f"N_ESTIMATORS: {N_ESTIMATORS}")
    logger.log(f"EARLY_STOPPING_ROUNDS: {EARLY_STOPPING_ROUNDS}")
    logger.log(f"baseline_cv: {args.baseline_cv}")
    logger.log(f"data_dir: {data_dir}")
    logger.log(f"run_dir: {run_dir}")

    set_seed(SEED)

    train, test, sample_submission = load_data(
        data_dir,
        logger,
    )

    print_data_overview(
        train,
        test,
        logger,
    )

    outlier_params = fit_outlier_params(
        train
    )

    outlier_params_path = run_dir / "outlier_params.json"
    with open(outlier_params_path, "w", encoding="utf-8") as f:
        json.dump(outlier_params, f, indent=2)

    train_fe, test_fe, categorical_cols = prepare_base_features(
        train,
        test,
        outlier_params,
        logger,
    )

    # ----------------------------
    # Prune features using previous full_features_model importance
    # ----------------------------

    if args.disable_pruning:
        logger.log("\n========== Feature pruning disabled by argument ==========")
        SELECTED_FEATURES = None
        previous_importance_for_pruning = pd.DataFrame()
    else:
        SELECTED_FEATURES, previous_importance_for_pruning = load_pruned_feature_list(
            importance_path=args.importance_path,
            keep_top_n=args.keep_top_n,
            min_gain=args.min_gain,
            logger=logger,
        )

    selected_features_path = run_dir / "selected_features.json"
    save_selected_features(
        selected_features=SELECTED_FEATURES,
        output_path=selected_features_path,
    )

    latest_selected_features_path = output_base_dir / "selected_features_latest.json"
    save_selected_features(
        selected_features=SELECTED_FEATURES,
        output_path=latest_selected_features_path,
    )

    logger.log(f"Saved selected features       : {selected_features_path}")
    logger.log(f"Saved latest selected features: {latest_selected_features_path}")

    logger.log("\nFeature count currently available before target encoding:")
    logger.log(len(get_feature_columns(train_fe)))

    if args.no_optuna:
        logger.log("\nFeature count currently available before target encoding:")

        y = train_fe[TARGET].astype(int)
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        scale_pos_weight = neg / max(pos, 1)

        best_params = make_default_lgbm_params(
            scale_pos_weight
        )

        best_optuna_score = None
        best_optuna_params = None

    else:
        best_params, best_optuna_score, best_optuna_params = run_optuna(
            train_fe,
            test_fe,
            categorical_cols,
            run_dir,
            logger,
        )

    # Optuna終了時点でbest paramsを保存
    best_params_path = run_dir / "best_params.json"

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_optuna_cv_auc": best_optuna_score,
                "best_optuna_params_only": best_optuna_params,
                "best_params_full": best_params,
                "added_numeric_features": ADDED_NUMERIC_FEATURES,
                "added_outlier_flag_features": ADDED_OUTLIER_FLAG_FEATURES,
                "added_time_series_features": ADDED_TIME_SERIES_FEATURES,
                "selected_features": SELECTED_FEATURES,
                "selected_features_path": str(selected_features_path),
                "target_encoding_cols": TARGET_ENCODING_COLS,
                "categorical_cols": categorical_cols,
                "outlier_params_path": str(outlier_params_path),
            },
            f,
            indent=2,
        )

    latest_best_params_path = output_base_dir / "best_params_latest.json"

    with open(latest_best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "best_optuna_cv_auc": best_optuna_score,
                "best_optuna_params_only": best_optuna_params,
                "best_params_full": best_params,
                "added_numeric_features": ADDED_NUMERIC_FEATURES,
                "added_outlier_flag_features": ADDED_OUTLIER_FLAG_FEATURES,
                "added_time_series_features": ADDED_TIME_SERIES_FEATURES,
                "selected_features": SELECTED_FEATURES,
                "selected_features_path": str(selected_features_path),
                "target_encoding_cols": TARGET_ENCODING_COLS,
                "categorical_cols": categorical_cols,
                "outlier_params_path": str(outlier_params_path),
            },
            f,
            indent=2,
        )

    logger.log(f"\nSaved best params: {best_params_path}")
    logger.log(f"Saved latest best params: {latest_best_params_path}")
    logger.log(f"Saved outlier params: {outlier_params_path}")

    logger.log("\n========== Final training ==========")

    final_auc, oof, test_preds, importance_df, best_iters = train_one_cv(
        train_fe,
        test_fe,
        categorical_cols,
        best_params,
        logger,
        return_test_preds=True,
        return_importance=True,
        trial_number=None,
    )

    # ----------------------------
    # Feature effect reports
    # ----------------------------

    flag_effect_df, oof_flag_df, feature_effect_report = create_feature_effect_report(
        train_fe=train_fe,
        oof=oof,
        importance_df=importance_df,
        flag_cols=ADDED_OUTLIER_FLAG_FEATURES,
        output_dir=run_dir,
        logger=logger,
    )

    # 最新版としても保存
    flag_effect_latest_path = output_base_dir / "outlier_flag_effects_latest.csv"
    oof_flag_latest_path = output_base_dir / "oof_calibration_by_flags_latest.csv"
    feature_effect_latest_path = output_base_dir / "feature_effect_report_latest.csv"

    flag_effect_df.to_csv(flag_effect_latest_path, index=False)
    oof_flag_df.to_csv(oof_flag_latest_path, index=False)
    feature_effect_report.to_csv(feature_effect_latest_path, index=False)

    # ----------------------------
    # Save submission
    # ----------------------------

    if sample_submission is not None:
        submission = sample_submission.copy()
        submission[TARGET] = test_preds
    else:
        submission = pd.DataFrame({
            ID_COL: test[ID_COL].values,
            TARGET: test_preds,
        })

    submission[TARGET] = np.clip(
        submission[TARGET],
        0.0,
        1.0,
    )

    submission_path = run_dir / "submission.csv"
    submission.to_csv(
        submission_path,
        index=False,
    )

    # すぐKaggleに出せる最新ファイルも保存
    latest_submission_path = output_base_dir / "submission.csv"
    submission.to_csv(
        latest_submission_path,
        index=False,
    )

    # ----------------------------
    # Save OOF
    # ----------------------------

    oof_path = run_dir / "oof_predictions.csv"

    pd.DataFrame({
        ID_COL: train[ID_COL].values,
        TARGET: train[TARGET].values,
        "oof_pred": oof,
    }).to_csv(
        oof_path,
        index=False,
    )

    # ----------------------------
    # Save feature importance
    # ----------------------------

    importance_path = run_dir / "feature_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    latest_importance_path = output_base_dir / "feature_importance_latest.csv"

    importance_df.to_csv(
        latest_importance_path,
        index=False,
    )

    # ----------------------------
    # Save final result JSON
    # ----------------------------

    delta_vs_baseline = final_auc - float(args.baseline_cv)

    if delta_vs_baseline > 0.0005:
        comparison_status = "improved"
    elif delta_vs_baseline < -0.0005:
        comparison_status = "worse"
    else:
        comparison_status = "roughly_same"

    final_result = {
        "run_id": run_id,
        "timestamp": now_str(),
        "final_cv_auc": final_auc,
        "baseline_cv_auc": float(args.baseline_cv),
        "delta_vs_baseline": delta_vs_baseline,
        "comparison_status": comparison_status,
        "best_optuna_cv_auc": best_optuna_score,
        "best_iters_final": best_iters,
        "n_trials": N_TRIALS,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feature_count_before_target_encoding": len(get_feature_columns(train_fe)),
        "added_numeric_features": ADDED_NUMERIC_FEATURES,
        "added_outlier_flag_features": ADDED_OUTLIER_FLAG_FEATURES,
        "added_time_series_features": ADDED_TIME_SERIES_FEATURES,
        "selected_features": SELECTED_FEATURES,
        "selected_features_path": str(selected_features_path),
        "submission_path": str(submission_path),
        "oof_path": str(oof_path),
        "importance_path": str(importance_path),
        "flag_effect_path": str(run_dir / "outlier_flag_effects.csv"),
        "oof_flag_calibration_path": str(run_dir / "oof_calibration_by_flags.csv"),
        "feature_effect_report_path": str(run_dir / "feature_effect_report.csv"),
        "best_params_path": str(best_params_path),
    }

    result_path = run_dir / "final_result.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            final_result,
            f,
            indent=2,
        )

    # ----------------------------
    # Append experiment summary
    # ----------------------------

    summary_path = output_base_dir / "experiment_summary.csv"

    summary_record = {
        "run_id": run_id,
        "timestamp": now_str(),
        "final_cv_auc": final_auc,
        "baseline_cv_auc": float(args.baseline_cv),
        "delta_vs_baseline": delta_vs_baseline,
        "comparison_status": comparison_status,
        "best_optuna_cv_auc": best_optuna_score,
        "n_trials": N_TRIALS,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feature_count_before_target_encoding": len(get_feature_columns(train_fe)),
        "n_outlier_flags": len(ADDED_OUTLIER_FLAG_FEATURES),
        "n_time_series_features": len(ADDED_TIME_SERIES_FEATURES),
        "n_selected_features": None if SELECTED_FEATURES is None else len(SELECTED_FEATURES),
        "keep_top_n": args.keep_top_n,
        "min_gain": args.min_gain,
        "run_dir": str(run_dir),
    }

    append_experiment_summary(
        summary_path,
        summary_record,
    )

    # ----------------------------
    # Print final logs
    # ----------------------------

    logger.log("\n========== Top 50 features by gain ==========")
    logger.log(
        importance_df[
            [
                "feature",
                "gain_rank",
                "gain_importance_mean",
                "gain_importance_pct",
                "split_importance_mean",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    logger.log("\n========== DONE ==========")
    logger.log(f"Final CV AUC                 : {final_auc:.6f}")
    logger.log(f"Baseline CV AUC              : {float(args.baseline_cv):.6f}")
    logger.log(f"Delta vs baseline            : {delta_vs_baseline:+.6f}")
    logger.log(f"Comparison status            : {comparison_status}")
    logger.log(f"Best Optuna CV AUC           : {best_optuna_score}")
    logger.log(f"Saved submission             : {submission_path}")
    logger.log(f"Saved latest submission      : {latest_submission_path}")
    logger.log(f"Saved OOF                    : {oof_path}")
    logger.log(f"Saved importance             : {importance_path}")
    logger.log(f"Saved latest importance      : {latest_importance_path}")
    logger.log(f"Saved flag effects           : {run_dir / 'outlier_flag_effects.csv'}")
    logger.log(f"Saved latest flag effects    : {flag_effect_latest_path}")
    logger.log(f"Saved feature effect report  : {run_dir / 'feature_effect_report.csv'}")
    logger.log(f"Saved latest effect report   : {feature_effect_latest_path}")
    logger.log(f"Saved selected features      : {selected_features_path}")
    logger.log(f"Saved latest selected features: {latest_selected_features_path}")
    logger.log(f"Saved best params            : {best_params_path}")
    logger.log(f"Saved latest params          : {latest_best_params_path}")
    logger.log(f"Saved result JSON            : {result_path}")
    logger.log(f"Saved experiment summary     : {summary_path}")
    logger.log(f"Saved log                    : {logger.log_path}")

    if args.pause:
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
