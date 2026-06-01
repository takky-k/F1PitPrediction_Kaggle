#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1 PitNextLap prediction pipeline
=================================

What this script does
---------------------
1. Loads train.csv / test.csv / sample_submission.csv.
2. Adds F1 race metadata: circuit name, circuit length, race date, calendar round, season timing.
3. Creates many domain-driven features for pit-stop prediction.
4. Bins Stint as 1,2,3,4,5,6plus.
5. Runs a solid LightGBM baseline with ROC AUC CV.
6. Tests engineered features one-by-one and keeps only features that improve CV AUC.
7. Optionally tunes LightGBM hyperparameters with Optuna.
8. Trains final CV ensemble and writes:
   - submission.csv
   - oof_predictions.csv
   - selected_features.txt
   - feature_selection_report.csv
   - feature_importance.csv
   - run_summary.json
   - console log and optional plots

Python compatibility: Python 3.9+

Example
-------
py f1_pitnextlap_lgbm_feature_selection_pipeline.py --data-dir C:\\Users\\takit\\Downloads\\f1 --n-trials 30

Quick smoke test
----------------
py f1_pitnextlap_lgbm_feature_selection_pipeline.py --data-dir . --quick

Notes
-----
- This is designed for Kaggle Playground Series S6E5: Predicting F1 Pit Stops.
- The metadata dictionaries are intentionally embedded so the script works offline.
- The competition data is synthetic; external race metadata may help, but always trust local CV more than intuition.
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False

try:
    import lightgbm as lgb
except Exception as e:
    lgb = None
    LIGHTGBM_IMPORT_ERROR = e
else:
    LIGHTGBM_IMPORT_ERROR = None

try:
    import optuna
except Exception:
    optuna = None

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

TARGET = "PitNextLap"
ID_COL = "id"
EPS = 1e-9

# ---------------------------------------------------------------------------
# Race metadata
# ---------------------------------------------------------------------------
# Circuit lengths are approximate current F1 GP layout lengths in kilometers.
# For Singapore / Spain there have been layout changes in recent years; the script
# also has year-specific overrides where useful.
CIRCUIT_META = {
    "Bahrain Grand Prix": {"circuit": "Bahrain International Circuit", "country": "Bahrain", "length_km": 5.412, "laps": 57, "track_type": "permanent"},
    "Saudi Arabian Grand Prix": {"circuit": "Jeddah Corniche Circuit", "country": "Saudi Arabia", "length_km": 6.174, "laps": 50, "track_type": "street"},
    "Australian Grand Prix": {"circuit": "Albert Park Circuit", "country": "Australia", "length_km": 5.278, "laps": 58, "track_type": "street_park"},
    "Chinese Grand Prix": {"circuit": "Shanghai International Circuit", "country": "China", "length_km": 5.451, "laps": 56, "track_type": "permanent"},
    "Japanese Grand Prix": {"circuit": "Suzuka Circuit", "country": "Japan", "length_km": 5.807, "laps": 53, "track_type": "permanent"},
    "Emilia-Romagna Grand Prix": {"circuit": "Autodromo Internazionale Enzo e Dino Ferrari", "country": "Italy", "length_km": 4.909, "laps": 63, "track_type": "permanent"},
    "Miami Grand Prix": {"circuit": "Miami International Autodrome", "country": "United States", "length_km": 5.412, "laps": 57, "track_type": "street"},
    "Spanish Grand Prix": {"circuit": "Circuit de Barcelona-Catalunya", "country": "Spain", "length_km": 4.657, "laps": 66, "track_type": "permanent"},
    "Monaco Grand Prix": {"circuit": "Circuit de Monaco", "country": "Monaco", "length_km": 3.337, "laps": 78, "track_type": "street"},
    "Azerbaijan Grand Prix": {"circuit": "Baku City Circuit", "country": "Azerbaijan", "length_km": 6.003, "laps": 51, "track_type": "street"},
    "Canadian Grand Prix": {"circuit": "Circuit Gilles Villeneuve", "country": "Canada", "length_km": 4.361, "laps": 70, "track_type": "street_park"},
    "British Grand Prix": {"circuit": "Silverstone Circuit", "country": "United Kingdom", "length_km": 5.891, "laps": 52, "track_type": "permanent"},
    "Austrian Grand Prix": {"circuit": "Red Bull Ring", "country": "Austria", "length_km": 4.318, "laps": 71, "track_type": "permanent"},
    "French Grand Prix": {"circuit": "Circuit Paul Ricard", "country": "France", "length_km": 5.842, "laps": 53, "track_type": "permanent"},
    "Hungarian Grand Prix": {"circuit": "Hungaroring", "country": "Hungary", "length_km": 4.381, "laps": 70, "track_type": "permanent"},
    "Belgian Grand Prix": {"circuit": "Circuit de Spa-Francorchamps", "country": "Belgium", "length_km": 7.004, "laps": 44, "track_type": "permanent"},
    "Dutch Grand Prix": {"circuit": "Circuit Zandvoort", "country": "Netherlands", "length_km": 4.259, "laps": 72, "track_type": "permanent"},
    "Italian Grand Prix": {"circuit": "Autodromo Nazionale Monza", "country": "Italy", "length_km": 5.793, "laps": 53, "track_type": "permanent"},
    "Singapore Grand Prix": {"circuit": "Marina Bay Street Circuit", "country": "Singapore", "length_km": 4.940, "laps": 62, "track_type": "street"},
    "United States Grand Prix": {"circuit": "Circuit of the Americas", "country": "United States", "length_km": 5.513, "laps": 56, "track_type": "permanent"},
    "Mexico City Grand Prix": {"circuit": "Autodromo Hermanos Rodriguez", "country": "Mexico", "length_km": 4.304, "laps": 71, "track_type": "permanent"},
    "Sao Paulo Grand Prix": {"circuit": "Autodromo Jose Carlos Pace / Interlagos", "country": "Brazil", "length_km": 4.309, "laps": 71, "track_type": "permanent"},
    "Qatar Grand Prix": {"circuit": "Lusail International Circuit", "country": "Qatar", "length_km": 5.419, "laps": 57, "track_type": "permanent"},
    "Las Vegas Grand Prix": {"circuit": "Las Vegas Strip Circuit", "country": "United States", "length_km": 6.201, "laps": 50, "track_type": "street"},
    "Abu Dhabi Grand Prix": {"circuit": "Yas Marina Circuit", "country": "United Arab Emirates", "length_km": 5.281, "laps": 58, "track_type": "permanent"},
    "Pre-Season Testing": {"circuit": "Bahrain International Circuit", "country": "Bahrain", "length_km": 5.412, "laps": 57, "track_type": "testing"},
}

# Year/layout specific overrides.
CIRCUIT_LENGTH_OVERRIDE = {
    (2022, "Spanish Grand Prix"): 4.675,
    (2022, "Singapore Grand Prix"): 5.063,
}

# Calendar windows taken from official F1 race calendars. race_date is the final day of the race weekend.
# For 2024 Bahrain/Saudi races were held on Saturday, so race_date is Saturday.
CALENDAR_BY_YEAR = {
    2022: [
        (0, "Pre-Season Testing", "2022-03-10", "2022-03-12"),
        (1, "Bahrain Grand Prix", "2022-03-18", "2022-03-20"),
        (2, "Saudi Arabian Grand Prix", "2022-03-25", "2022-03-27"),
        (3, "Australian Grand Prix", "2022-04-08", "2022-04-10"),
        (4, "Emilia-Romagna Grand Prix", "2022-04-22", "2022-04-24"),
        (5, "Miami Grand Prix", "2022-05-06", "2022-05-08"),
        (6, "Spanish Grand Prix", "2022-05-20", "2022-05-22"),
        (7, "Monaco Grand Prix", "2022-05-27", "2022-05-29"),
        (8, "Azerbaijan Grand Prix", "2022-06-10", "2022-06-12"),
        (9, "Canadian Grand Prix", "2022-06-17", "2022-06-19"),
        (10, "British Grand Prix", "2022-07-01", "2022-07-03"),
        (11, "Austrian Grand Prix", "2022-07-08", "2022-07-10"),
        (12, "French Grand Prix", "2022-07-22", "2022-07-24"),
        (13, "Hungarian Grand Prix", "2022-07-29", "2022-07-31"),
        (14, "Belgian Grand Prix", "2022-08-26", "2022-08-28"),
        (15, "Dutch Grand Prix", "2022-09-02", "2022-09-04"),
        (16, "Italian Grand Prix", "2022-09-09", "2022-09-11"),
        (17, "Singapore Grand Prix", "2022-09-30", "2022-10-02"),
        (18, "Japanese Grand Prix", "2022-10-07", "2022-10-09"),
        (19, "United States Grand Prix", "2022-10-21", "2022-10-23"),
        (20, "Mexico City Grand Prix", "2022-10-28", "2022-10-30"),
        (21, "Sao Paulo Grand Prix", "2022-11-11", "2022-11-13"),
        (22, "Abu Dhabi Grand Prix", "2022-11-18", "2022-11-20"),
    ],
    2023: [
        (0, "Pre-Season Testing", "2023-02-23", "2023-02-25"),
        (1, "Bahrain Grand Prix", "2023-03-03", "2023-03-05"),
        (2, "Saudi Arabian Grand Prix", "2023-03-17", "2023-03-19"),
        (3, "Australian Grand Prix", "2023-03-31", "2023-04-02"),
        (4, "Azerbaijan Grand Prix", "2023-04-28", "2023-04-30"),
        (5, "Miami Grand Prix", "2023-05-05", "2023-05-07"),
        (6, "Emilia-Romagna Grand Prix", "2023-05-19", "2023-05-21"),
        (7, "Monaco Grand Prix", "2023-05-26", "2023-05-28"),
        (8, "Spanish Grand Prix", "2023-06-02", "2023-06-04"),
        (9, "Canadian Grand Prix", "2023-06-16", "2023-06-18"),
        (10, "Austrian Grand Prix", "2023-06-30", "2023-07-02"),
        (11, "British Grand Prix", "2023-07-07", "2023-07-09"),
        (12, "Hungarian Grand Prix", "2023-07-21", "2023-07-23"),
        (13, "Belgian Grand Prix", "2023-07-28", "2023-07-30"),
        (14, "Dutch Grand Prix", "2023-08-25", "2023-08-27"),
        (15, "Italian Grand Prix", "2023-09-01", "2023-09-03"),
        (16, "Singapore Grand Prix", "2023-09-15", "2023-09-17"),
        (17, "Japanese Grand Prix", "2023-09-22", "2023-09-24"),
        (18, "Qatar Grand Prix", "2023-10-06", "2023-10-08"),
        (19, "United States Grand Prix", "2023-10-20", "2023-10-22"),
        (20, "Mexico City Grand Prix", "2023-10-27", "2023-10-29"),
        (21, "Sao Paulo Grand Prix", "2023-11-03", "2023-11-05"),
        (22, "Las Vegas Grand Prix", "2023-11-16", "2023-11-18"),
        (23, "Abu Dhabi Grand Prix", "2023-11-24", "2023-11-26"),
    ],
    2024: [
        (0, "Pre-Season Testing", "2024-02-21", "2024-02-23"),
        (1, "Bahrain Grand Prix", "2024-02-29", "2024-03-02"),
        (2, "Saudi Arabian Grand Prix", "2024-03-07", "2024-03-09"),
        (3, "Australian Grand Prix", "2024-03-22", "2024-03-24"),
        (4, "Japanese Grand Prix", "2024-04-05", "2024-04-07"),
        (5, "Chinese Grand Prix", "2024-04-19", "2024-04-21"),
        (6, "Miami Grand Prix", "2024-05-03", "2024-05-05"),
        (7, "Emilia-Romagna Grand Prix", "2024-05-17", "2024-05-19"),
        (8, "Monaco Grand Prix", "2024-05-24", "2024-05-26"),
        (9, "Canadian Grand Prix", "2024-06-07", "2024-06-09"),
        (10, "Spanish Grand Prix", "2024-06-21", "2024-06-23"),
        (11, "Austrian Grand Prix", "2024-06-28", "2024-06-30"),
        (12, "British Grand Prix", "2024-07-05", "2024-07-07"),
        (13, "Hungarian Grand Prix", "2024-07-19", "2024-07-21"),
        (14, "Belgian Grand Prix", "2024-07-26", "2024-07-28"),
        (15, "Dutch Grand Prix", "2024-08-23", "2024-08-25"),
        (16, "Italian Grand Prix", "2024-08-30", "2024-09-01"),
        (17, "Azerbaijan Grand Prix", "2024-09-13", "2024-09-15"),
        (18, "Singapore Grand Prix", "2024-09-20", "2024-09-22"),
        (19, "United States Grand Prix", "2024-10-18", "2024-10-20"),
        (20, "Mexico City Grand Prix", "2024-10-25", "2024-10-27"),
        (21, "Sao Paulo Grand Prix", "2024-11-01", "2024-11-03"),
        (22, "Las Vegas Grand Prix", "2024-11-21", "2024-11-23"),
        (23, "Qatar Grand Prix", "2024-11-29", "2024-12-01"),
        (24, "Abu Dhabi Grand Prix", "2024-12-06", "2024-12-08"),
    ],
    2025: [
        (0, "Pre-Season Testing", "2025-02-26", "2025-02-28"),
        (1, "Australian Grand Prix", "2025-03-14", "2025-03-16"),
        (2, "Chinese Grand Prix", "2025-03-21", "2025-03-23"),
        (3, "Japanese Grand Prix", "2025-04-04", "2025-04-06"),
        (4, "Bahrain Grand Prix", "2025-04-11", "2025-04-13"),
        (5, "Saudi Arabian Grand Prix", "2025-04-18", "2025-04-20"),
        (6, "Miami Grand Prix", "2025-05-02", "2025-05-04"),
        (7, "Emilia-Romagna Grand Prix", "2025-05-16", "2025-05-18"),
        (8, "Monaco Grand Prix", "2025-05-23", "2025-05-25"),
        (9, "Spanish Grand Prix", "2025-05-30", "2025-06-01"),
        (10, "Canadian Grand Prix", "2025-06-13", "2025-06-15"),
        (11, "Austrian Grand Prix", "2025-06-27", "2025-06-29"),
        (12, "British Grand Prix", "2025-07-04", "2025-07-06"),
        (13, "Belgian Grand Prix", "2025-07-25", "2025-07-27"),
        (14, "Hungarian Grand Prix", "2025-08-01", "2025-08-03"),
        (15, "Dutch Grand Prix", "2025-08-29", "2025-08-31"),
        (16, "Italian Grand Prix", "2025-09-05", "2025-09-07"),
        (17, "Azerbaijan Grand Prix", "2025-09-19", "2025-09-21"),
        (18, "Singapore Grand Prix", "2025-10-03", "2025-10-05"),
        (19, "United States Grand Prix", "2025-10-17", "2025-10-19"),
        (20, "Mexico City Grand Prix", "2025-10-24", "2025-10-26"),
        (21, "Sao Paulo Grand Prix", "2025-11-07", "2025-11-09"),
        (22, "Las Vegas Grand Prix", "2025-11-20", "2025-11-22"),
        (23, "Qatar Grand Prix", "2025-11-28", "2025-11-30"),
        (24, "Abu Dhabi Grand Prix", "2025-12-05", "2025-12-07"),
    ],
}

COMPOUND_META = {
    "SOFT": {"durability": 1, "speed": 3, "dry": 1, "wet": 0},
    "MEDIUM": {"durability": 2, "speed": 2, "dry": 1, "wet": 0},
    "HARD": {"durability": 3, "speed": 1, "dry": 1, "wet": 0},
    "INTERMEDIATE": {"durability": 2, "speed": 1, "dry": 0, "wet": 1},
    "WET": {"durability": 2, "speed": 0, "dry": 0, "wet": 1},
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print("[{0}] {1}".format(datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def ensure_lightgbm() -> None:
    if lgb is None:
        raise ImportError(
            "lightgbm is not installed. Install it with: py -m pip install lightgbm\n"
            "Original import error: {0}".format(repr(LIGHTGBM_IMPORT_ERROR))
        )


def safe_div(a: Any, b: Any) -> Any:
    return a / (b + EPS)


def normalize_race_name(x: Any) -> str:
    s = str(x).strip().lower()
    s2 = s.replace("-", " ").replace("_", " ")
    if "pre" in s2 and "season" in s2:
        return "Pre-Season Testing"
    if "bahrain" in s2 or "sakhir" in s2:
        return "Bahrain Grand Prix"
    if "saudi" in s2 or "jeddah" in s2:
        return "Saudi Arabian Grand Prix"
    if "austral" in s2 or "melbourne" in s2 or "albert park" in s2:
        return "Australian Grand Prix"
    if "china" in s2 or "chinese" in s2 or "shanghai" in s2:
        return "Chinese Grand Prix"
    if "japan" in s2 or "japanese" in s2 or "suzuka" in s2:
        return "Japanese Grand Prix"
    if "emilia" in s2 or "romagna" in s2 or "imola" in s2:
        return "Emilia-Romagna Grand Prix"
    if "miami" in s2:
        return "Miami Grand Prix"
    if "spanish" in s2 or "spain" in s2 or "barcelona" in s2:
        return "Spanish Grand Prix"
    if "monaco" in s2 or "monte carlo" in s2:
        return "Monaco Grand Prix"
    if "azerbaijan" in s2 or "baku" in s2:
        return "Azerbaijan Grand Prix"
    if "canad" in s2 or "montreal" in s2 or "gilles" in s2:
        return "Canadian Grand Prix"
    if "british" in s2 or "great britain" in s2 or "silverstone" in s2:
        return "British Grand Prix"
    if "austria" in s2 or "red bull ring" in s2 or "osterreich" in s2 or "österreich" in s2:
        return "Austrian Grand Prix"
    if "french" in s2 or "france" in s2 or "paul ricard" in s2:
        return "French Grand Prix"
    if "hungar" in s2 or "budapest" in s2:
        return "Hungarian Grand Prix"
    if "belg" in s2 or "spa" in s2:
        return "Belgian Grand Prix"
    if "dutch" in s2 or "netherlands" in s2 or "zandvoort" in s2:
        return "Dutch Grand Prix"
    if "italian" in s2 or "monza" in s2:
        return "Italian Grand Prix"
    if "singapore" in s2 or "marina bay" in s2:
        return "Singapore Grand Prix"
    if "united states" in s2 or "usa" in s2 or "cota" in s2 or "austin" in s2:
        return "United States Grand Prix"
    if "mexico" in s2:
        return "Mexico City Grand Prix"
    if "sao paulo" in s2 or "são paulo" in s2 or "brazil" in s2 or "interlagos" in s2:
        return "Sao Paulo Grand Prix"
    if "qatar" in s2 or "lusail" in s2 or "losail" in s2:
        return "Qatar Grand Prix"
    if "las vegas" in s2 or "vegas" in s2:
        return "Las Vegas Grand Prix"
    if "abu dhabi" in s2 or "yas marina" in s2:
        return "Abu Dhabi Grand Prix"
    return str(x).strip()


def build_calendar_frame() -> pd.DataFrame:
    rows = []
    for year, events in CALENDAR_BY_YEAR.items():
        season_events = [e for e in events if e[0] > 0]
        max_round = max([e[0] for e in season_events]) if season_events else 1
        season_start = pd.to_datetime(season_events[0][2]) if season_events else pd.Timestamp(year=year, month=1, day=1)
        season_end = pd.to_datetime(season_events[-1][3]) if season_events else pd.Timestamp(year=year, month=12, day=31)
        prev_end = None
        for rd, race, start, end in events:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            gap_prev = np.nan if prev_end is None else (start_dt - prev_end).days
            prev_end = end_dt
            total_days = max((season_end - season_start).days, 1)
            rows.append({
                "Year": year,
                "RaceKey": race,
                "CalendarRound": rd,
                "CalendarStartDate": start_dt,
                "RaceDate": end_dt,
                "CalendarEndDate": end_dt,
                "CalendarRoundNorm": rd / max_round if rd > 0 else 0.0,
                "RaceMonth": int(end_dt.month),
                "RaceDayOfYear": int(end_dt.dayofyear),
                "RaceWeekOfYear": int(end_dt.isocalendar().week),
                "DaysSinceSeasonStart": float((end_dt - season_start).days),
                "DaysToSeasonEnd": float((season_end - end_dt).days),
                "SeasonDateProgress": float((end_dt - season_start).days) / total_days,
                "DaysSincePrevEvent": gap_prev,
                "IsPreSeason": 1 if rd == 0 else 0,
                "IsSeasonOpener": 1 if rd == 1 else 0,
                "IsSeasonFinale": 1 if rd == max_round else 0,
            })
    return pd.DataFrame(rows)


def add_race_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Race" not in df.columns:
        df["Race"] = "Unknown"
    if "Year" not in df.columns:
        df["Year"] = np.nan

    df["RaceKey"] = df["Race"].apply(normalize_race_name)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Static circuit metadata.
    for col in ["CircuitName", "CircuitCountry", "CircuitLengthKM", "OfficialRaceLaps", "TrackType"]:
        df[col] = np.nan

    def get_meta(row: pd.Series, key: str) -> Any:
        race = row["RaceKey"]
        year_val = row["Year"]
        meta = CIRCUIT_META.get(race, {})
        if key == "length_km" and not pd.isna(year_val):
            override = CIRCUIT_LENGTH_OVERRIDE.get((int(year_val), race))
            if override is not None:
                return override
        return meta.get(key, np.nan)

    df["CircuitName"] = df.apply(lambda r: get_meta(r, "circuit"), axis=1)
    df["CircuitCountry"] = df.apply(lambda r: get_meta(r, "country"), axis=1)
    df["CircuitLengthKM"] = df.apply(lambda r: get_meta(r, "length_km"), axis=1)
    df["OfficialRaceLaps"] = df.apply(lambda r: get_meta(r, "laps"), axis=1)
    df["TrackType"] = df.apply(lambda r: get_meta(r, "track_type"), axis=1)

    cal = build_calendar_frame()
    df = df.merge(cal, on=["Year", "RaceKey"], how="left")

    # Fallbacks for unknown races/years.
    if "CalendarRound" in df.columns:
        df["CalendarRound"] = df["CalendarRound"].fillna(-1)
    for col in ["CalendarRoundNorm", "SeasonDateProgress", "DaysSinceSeasonStart", "DaysToSeasonEnd", "DaysSincePrevEvent"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
    for col in ["RaceMonth", "RaceDayOfYear", "RaceWeekOfYear", "IsPreSeason", "IsSeasonOpener", "IsSeasonFinale"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def canonical_col(df: pd.DataFrame, col: str) -> Optional[str]:
    if col in df.columns:
        return col
    # Common variants.
    lowered = {c.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", ""): c for c in df.columns}
    key = col.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")
    return lowered.get(key)


def load_data(data_dir: Path, original_train_csv: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not train_path.exists():
        raise FileNotFoundError("train.csv not found in {0}".format(data_dir))
    if not test_path.exists():
        raise FileNotFoundError("test.csv not found in {0}".format(data_dir))

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None

    train["DataSource"] = "competition_train"
    test["DataSource"] = "competition_test"

    if original_train_csv is not None:
        if not original_train_csv.exists():
            raise FileNotFoundError("original_train_csv not found: {0}".format(original_train_csv))
        orig = pd.read_csv(original_train_csv)
        if TARGET not in orig.columns:
            raise ValueError("original train file must contain target column {0}".format(TARGET))
        orig["DataSource"] = "original_external"
        if ID_COL not in orig.columns:
            orig[ID_COL] = -np.arange(1, len(orig) + 1)
        # Align columns: keep union, missing columns become NaN.
        all_cols = sorted(set(train.columns).union(set(orig.columns)))
        train = pd.concat([train.reindex(columns=all_cols), orig.reindex(columns=all_cols)], axis=0, ignore_index=True)
        log("Original/external data appended: {0:,} rows".format(len(orig)))

    log("Loaded train: {0:,} rows, {1} columns".format(len(train), train.shape[1]))
    log("Loaded test : {0:,} rows, {1} columns".format(len(test), test.shape[1]))
    if TARGET in train.columns:
        rate = pd.to_numeric(train[TARGET], errors="coerce").mean()
        log("Target mean PitNextLap=1: {0:.5f}".format(rate))
    return train, test, sample

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_group_stats(all_df: pd.DataFrame, value_col: str, group_cols: List[str], prefix: str) -> pd.DataFrame:
    df = all_df.copy()
    out = pd.DataFrame(index=df.index)
    if value_col not in df.columns:
        return out
    for gcol in group_cols:
        if gcol not in df.columns:
            continue
        grp = df.groupby(gcol, observed=True)[value_col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        med = grp.transform("median")
        out["{0}_{1}_mean_diff".format(prefix, gcol)] = df[value_col] - mean
        out["{0}_{1}_z".format(prefix, gcol)] = (df[value_col] - mean) / (std + EPS)
        out["{0}_{1}_median_diff".format(prefix, gcol)] = df[value_col] - med
    return out


def add_compound_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comp = df["Compound"].astype(str).str.upper() if "Compound" in df.columns else pd.Series("UNKNOWN", index=df.index)
    df["CompoundNorm"] = comp
    df["CompoundDurability"] = comp.map(lambda x: COMPOUND_META.get(x, {}).get("durability", np.nan))
    df["CompoundSpeed"] = comp.map(lambda x: COMPOUND_META.get(x, {}).get("speed", np.nan))
    df["IsDryCompound"] = comp.map(lambda x: COMPOUND_META.get(x, {}).get("dry", 0)).fillna(0)
    df["IsWetCompound"] = comp.map(lambda x: COMPOUND_META.get(x, {}).get("wet", 0)).fillna(0)
    df["IsSoft"] = (comp == "SOFT").astype(int)
    df["IsMedium"] = (comp == "MEDIUM").astype(int)
    df["IsHard"] = (comp == "HARD").astype(int)
    df["IsIntermediate"] = (comp == "INTERMEDIATE").astype(int)
    df["IsWet"] = (comp == "WET").astype(int)
    return df


def add_engineered_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """Return processed train/test plus base features, candidate features, categorical features."""
    n_train = len(train)
    all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

    # Standardize core columns.
    lap_col = canonical_col(all_df, "LapNumber") or "LapNumber"
    tyre_col = canonical_col(all_df, "TyreLife") or "TyreLife"
    stint_col = canonical_col(all_df, "Stint") or "Stint"
    pos_col = canonical_col(all_df, "Position") or "Position"
    lap_time_col = canonical_col(all_df, "LapTime (s)") or "LapTime (s)"
    lap_delta_col = canonical_col(all_df, "LapTime_Delta") or "LapTime_Delta"
    degr_col = canonical_col(all_df, "Cumulative_Degradation") or "Cumulative_Degradation"
    progress_col = canonical_col(all_df, "RaceProgress") or "RaceProgress"
    pos_change_col = canonical_col(all_df, "Position_Change") or "Position_Change"
    pitstop_col = canonical_col(all_df, "PitStop") or "PitStop"

    for c in [lap_col, tyre_col, stint_col, pos_col, lap_time_col, lap_delta_col, degr_col, progress_col, pos_change_col, pitstop_col, "Year"]:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = add_race_metadata(all_df)
    all_df = add_compound_meta(all_df)

    # Clean basic names for easier feature formulas.
    if lap_time_col in all_df.columns:
        all_df["LapTime_s"] = all_df[lap_time_col]
    if lap_delta_col in all_df.columns:
        all_df["LapTimeDelta"] = all_df[lap_delta_col]
    if degr_col in all_df.columns:
        all_df["CumulativeDegradation"] = all_df[degr_col]
    if progress_col in all_df.columns:
        all_df["RaceProgressClean"] = all_df[progress_col].clip(0, 1)
    else:
        all_df["RaceProgressClean"] = np.nan

    # Stint binning: 1..5 stay separate, 6+ is collapsed.
    if stint_col in all_df.columns:
        stint_num = pd.to_numeric(all_df[stint_col], errors="coerce")
        all_df["StintBin"] = np.where(stint_num >= 6, "6plus", stint_num.fillna(-1).astype(int).astype(str))
        all_df["StintClipped"] = np.minimum(stint_num, 6)
        all_df["IsStint6Plus"] = (stint_num >= 6).astype(int)
    else:
        all_df["StintBin"] = "unknown"
        all_df["StintClipped"] = np.nan
        all_df["IsStint6Plus"] = 0

    # Race-year and driver-race identifiers. These help tree models learn event-level patterns.
    all_df["RaceYear"] = all_df["RaceKey"].astype(str) + "_" + all_df["Year"].astype(str)
    if "Driver" in all_df.columns:
        all_df["DriverRace"] = all_df["Driver"].astype(str) + "_" + all_df["RaceKey"].astype(str)
        all_df["DriverYear"] = all_df["Driver"].astype(str) + "_" + all_df["Year"].astype(str)
    else:
        all_df["Driver"] = "unknown"
        all_df["DriverRace"] = "unknown"
        all_df["DriverYear"] = "unknown"
    if "Compound" not in all_df.columns:
        all_df["Compound"] = "unknown"

    # Estimate total laps from RaceProgress when available. This recreates a normalized race position signal.
    # It is useful, but check CV carefully because the competition intentionally removed Normalized_TyreLife.
    lap_num = all_df[lap_col] if lap_col in all_df.columns else pd.Series(np.nan, index=all_df.index)
    tyre_life = all_df[tyre_col] if tyre_col in all_df.columns else pd.Series(np.nan, index=all_df.index)
    progress = all_df["RaceProgressClean"]
    official_laps = pd.to_numeric(all_df["OfficialRaceLaps"], errors="coerce")
    est_total_laps_from_progress = safe_div(lap_num, progress.replace(0, np.nan))
    all_df["EstimatedTotalLaps"] = est_total_laps_from_progress.replace([np.inf, -np.inf], np.nan)
    all_df["EstimatedTotalLaps"] = all_df["EstimatedTotalLaps"].fillna(official_laps)
    all_df["EstimatedTotalLaps"] = all_df["EstimatedTotalLaps"].clip(lower=1, upper=200)
    all_df["OfficialRaceLapsFilled"] = official_laps.fillna(all_df["EstimatedTotalLaps"])
    all_df["LapsRemaining_Est"] = (all_df["EstimatedTotalLaps"] - lap_num).clip(lower=0)
    all_df["OfficialLapsRemaining"] = (all_df["OfficialRaceLapsFilled"] - lap_num).clip(lower=0)

    # Distance features.
    circuit_len = pd.to_numeric(all_df["CircuitLengthKM"], errors="coerce")
    all_df["RaceDistanceKM_Official"] = circuit_len * all_df["OfficialRaceLapsFilled"]
    all_df["DistanceCompletedKM"] = lap_num * circuit_len
    all_df["DistanceRemainingKM_Est"] = all_df["LapsRemaining_Est"] * circuit_len
    all_df["TyreDistanceKM"] = tyre_life * circuit_len
    all_df["TyreLifePerCircuitKM"] = safe_div(tyre_life, circuit_len)
    all_df["TyreDistanceShareOfRace"] = safe_div(all_df["TyreDistanceKM"], all_df["RaceDistanceKM_Official"])

    # Ratios and interactions around tyre age.
    all_df["TyreLife_to_LapNumber"] = safe_div(tyre_life, lap_num)
    all_df["TyreLife_to_EstTotalLaps"] = safe_div(tyre_life, all_df["EstimatedTotalLaps"])
    all_df["TyreLife_to_LapsRemaining"] = safe_div(tyre_life, all_df["LapsRemaining_Est"] + 1)
    all_df["TyreLife_to_OfficialLaps"] = safe_div(tyre_life, all_df["OfficialRaceLapsFilled"])
    all_df["LapNumber_to_OfficialLaps"] = safe_div(lap_num, all_df["OfficialRaceLapsFilled"])
    all_df["LapNumber_minus_TyreLife"] = lap_num - tyre_life
    all_df["TyreLife_minus_LapNumber"] = tyre_life - lap_num
    all_df["TyreLife_x_RaceProgress"] = tyre_life * progress
    all_df["TyreLife_x_SeasonProgress"] = tyre_life * all_df["SeasonDateProgress"]
    all_df["TyreLife_x_CalendarRoundNorm"] = tyre_life * all_df["CalendarRoundNorm"]
    all_df["TyreLife_x_CircuitLengthKM"] = tyre_life * circuit_len
    all_df["TyreLife_x_CompoundDurability"] = tyre_life * all_df["CompoundDurability"]
    all_df["TyreLife_div_CompoundDurability"] = safe_div(tyre_life, all_df["CompoundDurability"])
    all_df["TyreLife_x_IsSoft"] = tyre_life * all_df["IsSoft"]
    all_df["TyreLife_x_IsMedium"] = tyre_life * all_df["IsMedium"]
    all_df["TyreLife_x_IsHard"] = tyre_life * all_df["IsHard"]
    all_df["TyreLife_x_IsWetCompound"] = tyre_life * all_df["IsWetCompound"]

    # Non-linear race progress. This addresses non-monotonic RaceProgress effects.
    all_df["RaceProgress_sq"] = progress ** 2
    all_df["RaceProgress_cube"] = progress ** 3
    all_df["RaceProgress_sqrt"] = np.sqrt(progress.clip(lower=0))
    all_df["RaceProgress_logit"] = np.log(safe_div(progress.clip(EPS, 1 - EPS), 1 - progress.clip(EPS, 1 - EPS)))
    all_df["RaceProgress_mid_peak"] = progress * (1 - progress)
    all_df["RaceProgress_late_pressure"] = np.maximum(progress - 0.70, 0)
    all_df["RaceProgress_very_late"] = np.maximum(progress - 0.85, 0)
    all_df["RaceProgress_early"] = np.maximum(0.25 - progress, 0)
    all_df["RaceProgress_sin"] = np.sin(2 * np.pi * progress)
    all_df["RaceProgress_cos"] = np.cos(2 * np.pi * progress)
    all_df["RaceProgress_Bin"] = pd.cut(progress, bins=[-0.001, 0.10, 0.25, 0.50, 0.75, 0.90, 1.001], labels=["p00_10", "p10_25", "p25_50", "p50_75", "p75_90", "p90_100"]).astype(str)
    all_df["IsEarlyRace"] = (progress <= 0.25).astype(int)
    all_df["IsMidRace"] = ((progress > 0.25) & (progress <= 0.75)).astype(int)
    all_df["IsLateRace"] = (progress > 0.75).astype(int)
    all_df["IsFinal10Percent"] = (progress > 0.90).astype(int)

    # Lap time, degradation, position interactions.
    lap_time = all_df["LapTime_s"] if "LapTime_s" in all_df.columns else pd.Series(np.nan, index=all_df.index)
    lap_delta = all_df["LapTimeDelta"] if "LapTimeDelta" in all_df.columns else pd.Series(np.nan, index=all_df.index)
    degr = all_df["CumulativeDegradation"] if "CumulativeDegradation" in all_df.columns else pd.Series(np.nan, index=all_df.index)
    pos = all_df[pos_col] if pos_col in all_df.columns else pd.Series(np.nan, index=all_df.index)
    pos_chg = all_df[pos_change_col] if pos_change_col in all_df.columns else pd.Series(np.nan, index=all_df.index)

    all_df["LapTime_log"] = np.log1p(lap_time.clip(lower=0))
    all_df["LapTime_inv"] = safe_div(1.0, lap_time)
    all_df["LapTimeDelta_abs"] = lap_delta.abs()
    all_df["LapTimeDelta_sq"] = lap_delta ** 2
    all_df["LapTimeDelta_pos"] = lap_delta.clip(lower=0)
    all_df["LapTimeDelta_neg_abs"] = (-lap_delta.clip(upper=0))
    all_df["LapTimeDelta_x_TyreLife"] = lap_delta * tyre_life
    all_df["LapTimeDelta_x_RaceProgress"] = lap_delta * progress
    all_df["LapTimeDelta_x_IsLateRace"] = lap_delta * all_df["IsLateRace"]
    all_df["CumulativeDeg_abs"] = degr.abs()
    all_df["CumulativeDeg_sq"] = degr ** 2
    all_df["CumulativeDeg_per_TyreLife"] = safe_div(degr, tyre_life)
    all_df["CumulativeDeg_per_Lap"] = safe_div(degr, lap_num)
    all_df["CumulativeDeg_x_TyreLife"] = degr * tyre_life
    all_df["CumulativeDeg_x_Progress"] = degr * progress
    all_df["CumulativeDeg_x_IsSoft"] = degr * all_df["IsSoft"]
    all_df["CumulativeDeg_x_IsHard"] = degr * all_df["IsHard"]

    all_df["Position_inv"] = safe_div(1.0, pos)
    all_df["Position_sq"] = pos ** 2
    all_df["PositionChange_abs"] = pos_chg.abs()
    all_df["PositionChange_pos"] = pos_chg.clip(lower=0)
    all_df["PositionChange_neg_abs"] = (-pos_chg.clip(upper=0))
    all_df["Position_x_RaceProgress"] = pos * progress
    all_df["Position_x_TyreLife"] = pos * tyre_life
    all_df["PositionChange_x_LapTimeDelta"] = pos_chg * lap_delta
    all_df["PositionChange_x_RaceProgress"] = pos_chg * progress
    all_df["IsTop3"] = (pos <= 3).astype(int)
    all_df["IsTop5"] = (pos <= 5).astype(int)
    all_df["IsTop10"] = (pos <= 10).astype(int)
    all_df["IsBackmarker"] = (pos >= 16).astype(int)
    all_df["FrontRunningLateRace"] = all_df["IsTop5"] * all_df["IsLateRace"]
    all_df["BackmarkerLateRace"] = all_df["IsBackmarker"] * all_df["IsLateRace"]

    if pitstop_col in all_df.columns:
        pitstop = all_df[pitstop_col].fillna(0)
        all_df["PitStop_x_LapNumber"] = pitstop * lap_num
        all_df["PitStop_x_TyreLife"] = pitstop * tyre_life
        all_df["PitStop_x_RaceProgress"] = pitstop * progress
        all_df["PitStop_recent_proxy"] = ((pitstop == 1) | (tyre_life <= 2)).astype(int)
    else:
        all_df["PitStop_x_LapNumber"] = 0
        all_df["PitStop_x_TyreLife"] = 0
        all_df["PitStop_x_RaceProgress"] = 0
        all_df["PitStop_recent_proxy"] = (tyre_life <= 2).astype(int)

    # Calendar/time strategy features.
    all_df["SeasonEndPressure"] = np.maximum(all_df["SeasonDateProgress"] - 0.75, 0)
    all_df["SeasonOpeningPhase"] = np.maximum(0.25 - all_df["SeasonDateProgress"], 0)
    all_df["RoundLatePressure"] = np.maximum(all_df["CalendarRoundNorm"] - 0.75, 0)
    all_df["RoundEarlyPhase"] = np.maximum(0.25 - all_df["CalendarRoundNorm"], 0)
    all_df["RaceProgress_x_SeasonEndPressure"] = progress * all_df["SeasonEndPressure"]
    all_df["TyreLife_x_SeasonEndPressure"] = tyre_life * all_df["SeasonEndPressure"]
    all_df["LateRace_x_SeasonEndPressure"] = all_df["IsLateRace"] * all_df["SeasonEndPressure"]
    all_df["IsBackToBack"] = (all_df["DaysSincePrevEvent"] <= 8).astype(int)
    all_df["LongCalendarGapBefore"] = (all_df["DaysSincePrevEvent"] >= 20).astype(int)

    # Threshold flags. These are useful for tree models and for interpretability.
    for threshold in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50]:
        all_df["TyreLife_ge_{0}".format(threshold)] = (tyre_life >= threshold).astype(int)
    for threshold in [0.25, 0.50, 0.75, 0.90]:
        all_df["RaceProgress_ge_{0}".format(str(threshold).replace('.', '_'))] = (progress >= threshold).astype(int)

    # Binned numeric features.
    all_df["TyreLifeBin"] = pd.cut(tyre_life, bins=[-1, 1, 3, 5, 10, 15, 20, 30, 40, 60, 999], labels=["0_1", "2_3", "4_5", "6_10", "11_15", "16_20", "21_30", "31_40", "41_60", "60plus"]).astype(str)
    all_df["LapNumberBin"] = pd.cut(lap_num, bins=[-1, 5, 10, 20, 30, 40, 50, 60, 80, 999], labels=["0_5", "6_10", "11_20", "21_30", "31_40", "41_50", "51_60", "61_80", "80plus"]).astype(str)
    all_df["PositionBin"] = pd.cut(pos, bins=[0, 3, 5, 10, 15, 20, 999], labels=["p1_3", "p4_5", "p6_10", "p11_15", "p16_20", "p20plus"]).astype(str)
    all_df["CircuitLengthBin"] = pd.cut(circuit_len, bins=[0, 4.0, 5.0, 5.7, 6.2, 10], labels=["very_short", "short", "medium", "long", "very_long"]).astype(str)

    # ------------------------------------------------------------------
    # V2: richer domain-driven features.
    # These are not target encodings. They use race logic and unsupervised
    # train+test distribution statistics only.
    # ------------------------------------------------------------------
    # Finer race phase / remaining-lap buckets.
    all_df["RaceProgressBin10"] = pd.cut(
        progress,
        bins=[-0.001, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001],
        labels=["p00_10", "p10_20", "p20_30", "p30_40", "p40_50", "p50_60", "p60_70", "p70_80", "p80_90", "p90_100"],
    ).astype(str)
    all_df["LapsRemainingBin"] = pd.cut(
        all_df["LapsRemaining_Est"],
        bins=[-1, 0, 3, 5, 10, 15, 20, 30, 50, 999],
        labels=["0", "1_3", "4_5", "6_10", "11_15", "16_20", "21_30", "31_50", "50plus"],
    ).astype(str)
    all_df["TyreDistanceBin"] = pd.cut(
        all_df["TyreDistanceKM"],
        bins=[-1, 5, 15, 30, 50, 75, 100, 150, 220, 9999],
        labels=["0_5", "5_15", "15_30", "30_50", "50_75", "75_100", "100_150", "150_220", "220plus"],
    ).astype(str)

    # Compound-specific rough pit windows. The exact values are intentionally
    # approximate; CV decides whether they help.
    expected_life = all_df["CompoundNorm"].astype(str).map({
        "SOFT": 18.0, "MEDIUM": 28.0, "HARD": 38.0, "INTERMEDIATE": 16.0, "WET": 14.0,
    }).astype(float)
    # Street tracks often punish tyres differently and safety-car risk is higher;
    # use a soft adjustment only as a candidate signal.
    track_life_adj = all_df["TrackType"].astype(str).map({
        "street": 0.92, "street_park": 0.95, "permanent": 1.00, "testing": 1.00,
    }).fillna(1.0).astype(float)
    all_df["ExpectedCompoundLife"] = expected_life
    all_df["ExpectedCompoundTrackLife"] = expected_life * track_life_adj
    all_df["TyreLife_to_ExpectedCompoundLife"] = safe_div(tyre_life, all_df["ExpectedCompoundLife"])
    all_df["TyreLife_to_ExpectedCompoundTrackLife"] = safe_div(tyre_life, all_df["ExpectedCompoundTrackLife"])
    all_df["TyreLife_minus_ExpectedCompoundLife"] = tyre_life - all_df["ExpectedCompoundLife"]
    all_df["TyreLife_minus_ExpectedCompoundTrackLife"] = tyre_life - all_df["ExpectedCompoundTrackLife"]
    all_df["ProjectedTyreLifeAtFinish"] = tyre_life + all_df["LapsRemaining_Est"]
    all_df["ProjectedTyreDistanceAtFinishKM"] = all_df["ProjectedTyreLifeAtFinish"] * circuit_len
    all_df["ProjectedTyreLife_to_Expected"] = safe_div(all_df["ProjectedTyreLifeAtFinish"], all_df["ExpectedCompoundTrackLife"])
    all_df["CanFinishOnExpectedLife"] = (all_df["ProjectedTyreLifeAtFinish"] <= all_df["ExpectedCompoundTrackLife"]).astype(int)
    all_df["OverExpectedLifeNow"] = (tyre_life > all_df["ExpectedCompoundTrackLife"]).astype(int)
    all_df["NearExpectedPitWindow"] = ((all_df["TyreLife_to_ExpectedCompoundTrackLife"] >= 0.75) & (all_df["TyreLife_to_ExpectedCompoundTrackLife"] <= 1.25)).astype(int)
    all_df["PastExpectedPitWindow"] = (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 1.15).astype(int)

    # Strategy pressure candidates: old tyres, not enough laps to justify another stop, and race phase.
    all_df["TyreLife_x_LapsRemaining"] = tyre_life * all_df["LapsRemaining_Est"]
    all_df["TyreLife_minus_LapsRemaining"] = tyre_life - all_df["LapsRemaining_Est"]
    all_df["LapsRemaining_to_TyreLife"] = safe_div(all_df["LapsRemaining_Est"], tyre_life)
    all_df["TyrePressure_mid_old"] = all_df["IsMidRace"] * (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 0.65).astype(int)
    all_df["TyrePressure_late_old"] = all_df["IsLateRace"] * (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 0.85).astype(int)
    all_df["TyrePressure_final_old"] = all_df["IsFinal10Percent"] * (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 0.90).astype(int)
    all_df["OldTyreButCanFinish"] = all_df["CanFinishOnExpectedLife"] * (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 0.70).astype(int)
    all_df["OldTyreCannotFinish"] = (1 - all_df["CanFinishOnExpectedLife"]) * (all_df["TyreLife_to_ExpectedCompoundTrackLife"] > 0.70).astype(int)
    all_df["Top10_OldTyre_MidRace"] = all_df["IsTop10"] * all_df["TyrePressure_mid_old"]
    all_df["Top5_OldTyre_LateRace"] = all_df["IsTop5"] * all_df["TyrePressure_late_old"]
    all_df["Backmarker_OldTyre_LateRace"] = all_df["IsBackmarker"] * all_df["TyrePressure_late_old"]

    # Compound-specific threshold flags. These often help trees because SOFT 18 laps
    # and HARD 18 laps do not have the same meaning.
    compound_thresholds = {
        "SOFT": [8, 10, 12, 15, 18, 20, 25],
        "MEDIUM": [10, 15, 18, 22, 25, 30, 35],
        "HARD": [15, 20, 25, 30, 35, 40, 50],
        "INTERMEDIATE": [5, 8, 10, 12, 15, 20, 25],
        "WET": [5, 8, 10, 12, 15, 20, 25],
    }
    comp_norm_str = all_df["CompoundNorm"].astype(str)
    for comp_name, thresholds in compound_thresholds.items():
        comp_flag = (comp_norm_str == comp_name).astype(int)
        for threshold in thresholds:
            all_df["Is{0}_TyreLife_ge_{1}".format(comp_name.title().replace("_", ""), threshold)] = comp_flag * (tyre_life >= threshold).astype(int)

    # Lap-position periodicity / window candidates. Pit stops often cluster around
    # lap windows; sine/cosine allow smooth non-linear phase signals.
    all_df["LapNumber_sin_10"] = np.sin(2 * np.pi * safe_div(lap_num, 10.0))
    all_df["LapNumber_cos_10"] = np.cos(2 * np.pi * safe_div(lap_num, 10.0))
    all_df["LapNumber_sin_20"] = np.sin(2 * np.pi * safe_div(lap_num, 20.0))
    all_df["LapNumber_cos_20"] = np.cos(2 * np.pi * safe_div(lap_num, 20.0))
    all_df["LapNumber_sin_total"] = np.sin(2 * np.pi * safe_div(lap_num, all_df["OfficialRaceLapsFilled"]))
    all_df["LapNumber_cos_total"] = np.cos(2 * np.pi * safe_div(lap_num, all_df["OfficialRaceLapsFilled"]))

    # Group percentile features: relative position in race/compound distribution.
    # These can catch outliers without target leakage.
    percentile_specs = [
        (tyre_col, "TyreLifePct", ["RaceYear", "RaceKey", "CompoundNorm", "TrackType", "RaceYear_CompoundNorm"]),
        (lap_col, "LapNumberPct", ["RaceYear", "RaceKey", "TrackType"]),
        ("LapTime_s", "LapTimePct", ["RaceYear", "RaceKey", "CompoundNorm", "TrackType"]),
        ("LapTimeDelta", "LapDeltaPct", ["RaceYear", "RaceKey", "CompoundNorm", "TrackType"]),
        ("CumulativeDegradation", "CumDegPct", ["RaceYear", "RaceKey", "CompoundNorm", "TrackType"]),
    ]
    all_df["RaceYear_CompoundNorm"] = all_df["RaceYear"].astype(str) + "__" + all_df["CompoundNorm"].astype(str)
    for value_col, out_prefix, groups_for_value in percentile_specs:
        if isinstance(value_col, str) and value_col in all_df.columns:
            for gcol in groups_for_value:
                if gcol in all_df.columns:
                    try:
                        all_df["{0}_within_{1}".format(out_prefix, gcol)] = all_df.groupby(gcol, observed=True)[value_col].rank(pct=True)
                    except Exception:
                        pass

    # Categorical interaction keys. These are useful when two features only work
    # together, e.g., Compound x TyreLifeBin or Race x Compound.
    cat_combo_pairs = [
        ("CompoundNorm", "StintBin"),
        ("CompoundNorm", "RaceProgress_Bin"),
        ("CompoundNorm", "RaceProgressBin10"),
        ("CompoundNorm", "TyreLifeBin"),
        ("CompoundNorm", "LapsRemainingBin"),
        ("CompoundNorm", "PositionBin"),
        ("RaceKey", "CompoundNorm"),
        ("RaceKey", "TyreLifeBin"),
        ("RaceKey", "RaceProgress_Bin"),
        ("RaceYear", "CompoundNorm"),
        ("RaceYear", "StintBin"),
        ("TrackType", "CompoundNorm"),
        ("TrackType", "TyreLifeBin"),
        ("TrackType", "RaceProgress_Bin"),
        ("StintBin", "TyreLifeBin"),
        ("StintBin", "RaceProgress_Bin"),
        ("TyreLifeBin", "RaceProgress_Bin"),
        ("TyreLifeBin", "LapsRemainingBin"),
        ("PositionBin", "RaceProgress_Bin"),
        ("PositionBin", "TyreLifeBin"),
        ("CircuitLengthBin", "CompoundNorm"),
        ("CircuitLengthBin", "TyreLifeBin"),
        ("Driver", "CompoundNorm"),
        ("Driver", "StintBin"),
        ("Driver", "RaceProgress_Bin"),
    ]
    combo_cat_cols = []
    for a, b in cat_combo_pairs:
        if a in all_df.columns and b in all_df.columns:
            cname = "Cat_{0}__x__{1}".format(a, b)
            all_df[cname] = all_df[a].astype(str) + "__" + all_df[b].astype(str)
            combo_cat_cols.append(cname)
    # A few high-signal three-way keys. Keep them limited to avoid huge cardinality.
    three_way = [
        ("CompoundNorm", "TyreLifeBin", "RaceProgress_Bin"),
        ("CompoundNorm", "StintBin", "TyreLifeBin"),
        ("RaceKey", "CompoundNorm", "RaceProgress_Bin"),
        ("TrackType", "CompoundNorm", "TyreLifeBin"),
        ("PositionBin", "CompoundNorm", "RaceProgress_Bin"),
    ]
    for a, b, c in three_way:
        if a in all_df.columns and b in all_df.columns and c in all_df.columns:
            cname = "Cat_{0}__x__{1}__x__{2}".format(a, b, c)
            all_df[cname] = all_df[a].astype(str) + "__" + all_df[b].astype(str) + "__" + all_df[c].astype(str)
            combo_cat_cols.append(cname)

    # Unsupervised frequency/count encodings from combined train+test only. No target leakage.
    freq_cols = ["Driver", "RaceKey", "CompoundNorm", "StintBin", "RaceYear", "DriverRace", "TrackType", "CircuitCountry", "TyreLifeBin", "RaceProgress_Bin", "RaceProgressBin10", "LapsRemainingBin", "TyreDistanceBin"] + combo_cat_cols
    for c in freq_cols:
        if c in all_df.columns:
            vc = all_df[c].astype(str).value_counts(dropna=False)
            all_df["freq_{0}".format(c)] = all_df[c].astype(str).map(vc).astype(float)
            all_df["freq_{0}_norm".format(c)] = all_df["freq_{0}".format(c)] / len(all_df)

    # Group-level z-score features. These are very useful for outlier behavior.
    stats_pieces = []
    for vcol, prefix in [("LapTime_s", "LapTime"), ("LapTimeDelta", "LapDelta"), ("CumulativeDegradation", "CumDeg"), (tyre_col, "TyreLife")]:
        if isinstance(vcol, str) and vcol in all_df.columns:
            stats_pieces.append(make_group_stats(all_df, vcol, ["RaceYear", "RaceKey", "CompoundNorm", "TrackType"], prefix))
    if stats_pieces:
        all_df = pd.concat([all_df] + stats_pieces, axis=1)
    for c in list(all_df.columns):
        if c.endswith("_z"):
            all_df[c + "_abs"] = all_df[c].abs()
            all_df[c + "_outlier2"] = (all_df[c].abs() >= 2).astype(int)
            all_df[c + "_outlier3"] = (all_df[c].abs() >= 3).astype(int)

    # Replace infs and large impossible values.
    num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan)

    # Categorical columns for LightGBM.
    categorical_features = [
        "Driver", "Compound", "CompoundNorm", "Race", "RaceKey", "CircuitName", "CircuitCountry", "TrackType",
        "StintBin", "RaceYear", "DriverRace", "DriverYear", "RaceYear_CompoundNorm",
        "RaceProgress_Bin", "RaceProgressBin10", "TyreLifeBin", "LapNumberBin", "PositionBin",
        "CircuitLengthBin", "LapsRemainingBin", "TyreDistanceBin", "DataSource"
    ]
    # Add all categorical interaction keys created above.
    try:
        categorical_features = list(dict.fromkeys(categorical_features + combo_cat_cols + [c for c in all_df.columns if str(c).startswith("Cat_")]))
    except NameError:
        categorical_features = list(dict.fromkeys(categorical_features + [c for c in all_df.columns if str(c).startswith("Cat_")]))
    categorical_features = [c for c in categorical_features if c in all_df.columns and c != TARGET]
    for c in categorical_features:
        all_df[c] = all_df[c].astype("category")

    # Baseline features: raw columns + safe metadata + core categorical features.
    raw_numeric = [
        pitstop_col, lap_col, stint_col, tyre_col, pos_col, "LapTime_s", "LapTimeDelta", "CumulativeDegradation",
        progress_col, "RaceProgressClean", pos_change_col, "Year"
    ]
    raw_numeric = [c for c in raw_numeric if c in all_df.columns and c != TARGET]
    base_features = []
    for c in raw_numeric + ["Driver", "Compound", "RaceKey", "StintBin", "CircuitLengthKM", "OfficialRaceLapsFilled", "CalendarRound", "CalendarRoundNorm", "SeasonDateProgress"]:
        if c in all_df.columns and c not in base_features and c != TARGET:
            base_features.append(c)

    # Candidate features: everything engineered, excluding IDs, target, dates, and baseline columns.
    excluded = set([TARGET, ID_COL, "CalendarStartDate", "CalendarEndDate", "RaceDate"])
    excluded.update(base_features)
    candidate_features = []
    for c in all_df.columns:
        if c in excluded:
            continue
        if c.startswith("Unnamed"):
            continue
        if c in ["Race"]:  # raw Race can be noisier than RaceKey; RaceKey is baseline.
            continue
        if c == "Year" and c in base_features:
            continue
        # Dates are handled via numeric date features.
        if str(all_df[c].dtype).startswith("datetime"):
            continue
        if c == TARGET:
            continue
        candidate_features.append(c)

    # Keep at least ~100 candidate features but avoid duplicate-like columns.
    candidate_features = [c for c in candidate_features if c not in base_features]

    train_out = all_df.iloc[:n_train].copy()
    test_out = all_df.iloc[n_train:].copy()

    log("Base features: {0}".format(len(base_features)))
    log("Candidate engineered features: {0}".format(len(candidate_features)))
    log("Categorical features available: {0}".format(len(categorical_features)))
    return train_out, test_out, base_features, candidate_features, categorical_features

# ---------------------------------------------------------------------------
# CV / model
# ---------------------------------------------------------------------------

def make_folds(train: pd.DataFrame, y: pd.Series, n_splits: int, mode: str = "stratified_group") -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = int(max(2, n_splits))
    if mode in ["stratified_group", "group"]:
        groups = None
        for c in ["RaceYear", "RaceKey", "Race"]:
            if c in train.columns:
                groups = train[c].astype(str).values
                break
        if groups is not None and len(np.unique(groups)) >= n_splits:
            if mode == "stratified_group" and HAS_STRATIFIED_GROUP_KFOLD:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                return list(splitter.split(train, y, groups=groups))
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(train, y, groups=groups))
        log("Not enough groups for group CV; falling back to StratifiedKFold.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(train, y))


def default_lgbm_params(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.035 if not quick else 0.06,
        "n_estimators": 5000 if not quick else 1000,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "class_weight": None,
        "verbosity": -1,
        "metric": "auc",
    }
    return params


def train_cv_predict(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    features: List[str],
    target_col: str,
    categorical_features: List[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    early_stopping_rounds: int = 150,
    verbose_eval: int = 0,
    return_importance: bool = True,
) -> Dict[str, Any]:
    ensure_lightgbm()
    y = pd.to_numeric(train[target_col], errors="coerce").astype(int).values
    X = train[features].copy()
    X_test = test[features].copy() if test is not None else None
    cat_in_features = [c for c in categorical_features if c in features]

    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float) if test is not None else None
    fold_scores = []
    importances = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(**params)
        callbacks = []
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if verbose_eval and verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(verbose_eval))

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_in_features if cat_in_features else "auto",
            callbacks=callbacks,
        )
        pred_va = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred_va
        auc = roc_auc_score(y_va, pred_va)
        fold_scores.append(float(auc))
        best_it = getattr(model, "best_iteration_", None) or params.get("n_estimators")
        best_iterations.append(int(best_it))
        if test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / len(folds)
        if return_importance:
            try:
                imp = pd.DataFrame({
                    "feature": features,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                    "importance_split": model.booster_.feature_importance(importance_type="split"),
                    "fold": fold,
                })
                importances.append(imp)
            except Exception:
                pass
        log("Fold {0}/{1} AUC={2:.6f}, best_iter={3}".format(fold, len(folds), auc, best_it))

    overall_auc = roc_auc_score(y, oof)
    result = {
        "auc": float(overall_auc),
        "fold_scores": fold_scores,
        "oof": oof,
        "test_pred": test_pred,
        "best_iterations": best_iterations,
        "importance": pd.concat(importances, axis=0, ignore_index=True) if importances else pd.DataFrame(),
    }
    return result


def rank_candidates_quick(train: pd.DataFrame, candidates: List[str], target_col: str, max_candidates: int) -> List[str]:
    """Light pre-ranking so selection does not waste time on obviously useless columns."""
    y = pd.to_numeric(train[target_col], errors="coerce").astype(float)
    scores = []
    for c in candidates:
        try:
            if str(train[c].dtype) == "category" or train[c].dtype == object:
                # Use target-rate spread as a cheap unsmoothed signal. This is only for ordering, not as a feature.
                tmp = pd.DataFrame({"x": train[c].astype(str), "y": y})
                grp = tmp.groupby("x")["y"].mean()
                score = float(grp.max() - grp.min()) if len(grp) > 1 else 0.0
            else:
                x = pd.to_numeric(train[c], errors="coerce")
                if x.notna().sum() < 10 or x.nunique(dropna=True) <= 1:
                    score = 0.0
                else:
                    corr = np.corrcoef(x.fillna(x.median()).values, y.values)[0, 1]
                    score = abs(float(corr)) if np.isfinite(corr) else 0.0
            scores.append((c, score))
        except Exception:
            scores.append((c, 0.0))
    scores = sorted(scores, key=lambda t: t[1], reverse=True)
    ranked = [c for c, _ in scores]
    if max_candidates and max_candidates > 0:
        ranked = ranked[:max_candidates]
    return ranked


def forward_select_features(
    train: pd.DataFrame,
    base_features: List[str],
    candidates: List[str],
    target_col: str,
    categorical_features: List[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    min_delta: float,
    max_candidates: int,
    selection_rounds: int,
    output_dir: Path,
) -> Tuple[List[str], pd.DataFrame]:
    log("Running baseline CV before forward selection...")
    current_features = list(base_features)
    baseline = train_cv_predict(
        train=train,
        test=None,
        features=current_features,
        target_col=target_col,
        categorical_features=categorical_features,
        params=params,
        folds=folds,
        early_stopping_rounds=100,
        verbose_eval=0,
        return_importance=False,
    )
    current_auc = baseline["auc"]
    log("Baseline AUC = {0:.6f}".format(current_auc))

    ranked = rank_candidates_quick(train, candidates, target_col, max_candidates=max_candidates)
    log("Candidate features to test one-by-one: {0}".format(len(ranked)))

    report_rows = []
    selected = []
    tried_global = set()

    for round_i in range(1, selection_rounds + 1):
        log("Forward selection round {0}/{1}".format(round_i, selection_rounds))
        kept_this_round = 0
        for idx, feat in enumerate(ranked, 1):
            if feat in current_features or feat in tried_global:
                continue
            tried_global.add(feat)
            test_features = current_features + [feat]
            start = time.time()
            try:
                res = train_cv_predict(
                    train=train,
                    test=None,
                    features=test_features,
                    target_col=target_col,
                    categorical_features=categorical_features,
                    params=params,
                    folds=folds,
                    early_stopping_rounds=100,
                    verbose_eval=0,
                    return_importance=False,
                )
                new_auc = res["auc"]
                delta = new_auc - current_auc
                keep = bool(delta > min_delta)
                if keep:
                    current_features.append(feat)
                    selected.append(feat)
                    current_auc = new_auc
                    kept_this_round += 1
                status = "KEEP" if keep else "drop"
                log("[{0:03d}/{1}] {2}: AUC {3:.6f} -> {4:.6f} Δ={5:+.6f} ({6}, {7:.1f}s)".format(
                    idx, len(ranked), feat, current_auc - delta, new_auc, delta, status, time.time() - start
                ))
                report_rows.append({
                    "round": round_i,
                    "candidate_rank": idx,
                    "feature": feat,
                    "auc_before": current_auc - delta,
                    "auc_after": new_auc,
                    "delta": delta,
                    "kept": keep,
                    "n_features_after": len(current_features),
                    "seconds": time.time() - start,
                })
            except Exception as e:
                log("ERROR testing feature {0}: {1}".format(feat, repr(e)))
                report_rows.append({
                    "round": round_i,
                    "candidate_rank": idx,
                    "feature": feat,
                    "auc_before": current_auc,
                    "auc_after": np.nan,
                    "delta": np.nan,
                    "kept": False,
                    "n_features_after": len(current_features),
                    "seconds": time.time() - start,
                    "error": repr(e),
                })

            # Save progress frequently so interruption does not lose work.
            if len(report_rows) % 5 == 0:
                pd.DataFrame(report_rows).to_csv(output_dir / "feature_selection_report_live.csv", index=False)
                (output_dir / "selected_features_live.txt").write_text("\n".join(current_features), encoding="utf-8")
        if kept_this_round == 0:
            log("No features kept this round; stopping selection.")
            break

    report = pd.DataFrame(report_rows)
    return current_features, report


def pair_select_features(
    train: pd.DataFrame,
    selected_features: List[str],
    candidates: List[str],
    target_col: str,
    categorical_features: List[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    min_delta: float,
    pair_top_k: int,
    max_pairs: int,
    output_dir: Path,
) -> Tuple[List[str], pd.DataFrame]:
    """Search for two-feature bundles that may only help when used together.

    Forward selection can miss interactions: feature A alone may not improve AUC,
    feature B alone may not improve AUC, but A+B can improve it because the model
    needs both pieces of information. This routine tests pairs among the highest
    pre-ranked remaining candidates.
    """
    current_features = list(dict.fromkeys(selected_features))
    remaining = [c for c in candidates if c not in current_features]
    if not remaining:
        log("Pair search skipped: no remaining candidate features.")
        return current_features, pd.DataFrame()

    ranked = rank_candidates_quick(train, remaining, target_col, max_candidates=pair_top_k)
    if len(ranked) < 2:
        log("Pair search skipped: fewer than 2 remaining candidates.")
        return current_features, pd.DataFrame()

    log("Running current CV before pair search...")
    base_res = train_cv_predict(
        train=train,
        test=None,
        features=current_features,
        target_col=target_col,
        categorical_features=categorical_features,
        params=params,
        folds=folds,
        early_stopping_rounds=100,
        verbose_eval=0,
        return_importance=False,
    )
    current_auc = base_res["auc"]
    log("Pair-search starting AUC = {0:.6f}".format(current_auc))

    # Build pair list from the top ranked remaining features.
    pair_list = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            pair_list.append((ranked[i], ranked[j]))
    if max_pairs and max_pairs > 0:
        pair_list = pair_list[:max_pairs]

    log("Candidate feature pairs to test: {0} from top_k={1}".format(len(pair_list), len(ranked)))
    report_rows = []
    tested = set()
    kept_pairs = 0

    for idx, (feat_a, feat_b) in enumerate(pair_list, 1):
        if feat_a in current_features or feat_b in current_features:
            continue
        key = tuple(sorted([feat_a, feat_b]))
        if key in tested:
            continue
        tested.add(key)
        test_features = current_features + [feat_a, feat_b]
        start = time.time()
        try:
            res = train_cv_predict(
                train=train,
                test=None,
                features=test_features,
                target_col=target_col,
                categorical_features=categorical_features,
                params=params,
                folds=folds,
                early_stopping_rounds=100,
                verbose_eval=0,
                return_importance=False,
            )
            new_auc = res["auc"]
            delta = new_auc - current_auc
            keep = bool(delta > min_delta)
            if keep:
                current_features.extend([feat_a, feat_b])
                current_features = list(dict.fromkeys(current_features))
                current_auc = new_auc
                kept_pairs += 1
            status = "KEEP_PAIR" if keep else "drop_pair"
            log("[PAIR {0:03d}/{1}] {2} + {3}: AUC {4:.6f} -> {5:.6f} Δ={6:+.6f} ({7}, {8:.1f}s)".format(
                idx, len(pair_list), feat_a, feat_b, current_auc - delta, new_auc, delta, status, time.time() - start
            ))
            report_rows.append({
                "pair_rank": idx,
                "feature_a": feat_a,
                "feature_b": feat_b,
                "auc_before": current_auc - delta,
                "auc_after": new_auc,
                "delta": delta,
                "kept": keep,
                "n_features_after": len(current_features),
                "seconds": time.time() - start,
            })
        except Exception as e:
            log("ERROR testing pair {0} + {1}: {2}".format(feat_a, feat_b, repr(e)))
            report_rows.append({
                "pair_rank": idx,
                "feature_a": feat_a,
                "feature_b": feat_b,
                "auc_before": current_auc,
                "auc_after": np.nan,
                "delta": np.nan,
                "kept": False,
                "n_features_after": len(current_features),
                "seconds": time.time() - start,
                "error": repr(e),
            })

        if len(report_rows) % 5 == 0:
            pd.DataFrame(report_rows).to_csv(output_dir / "pair_selection_report_live.csv", index=False)
            (output_dir / "selected_features_pair_live.txt").write_text("\n".join(current_features), encoding="utf-8")

    log("Pair search complete. Kept pairs: {0}; selected feature count: {1}".format(kept_pairs, len(current_features)))
    return current_features, pd.DataFrame(report_rows)

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def tune_optuna(
    train: pd.DataFrame,
    features: List[str],
    target_col: str,
    categorical_features: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    timeout: Optional[int],
    seed: int,
    quick: bool,
) -> Dict[str, Any]:
    if n_trials <= 0:
        return default_lgbm_params(seed=seed, quick=quick)
    if optuna is None:
        log("Optuna is not installed. Skipping hyperparameter tuning. Install with: py -m pip install optuna")
        return default_lgbm_params(seed=seed, quick=quick)

    def objective(trial: Any) -> float:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "auc",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": -1,
            "n_estimators": 3000 if not quick else 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        res = train_cv_predict(
            train=train,
            test=None,
            features=features,
            target_col=target_col,
            categorical_features=categorical_features,
            params=params,
            folds=folds,
            early_stopping_rounds=120,
            verbose_eval=0,
            return_importance=False,
        )
        return res["auc"]

    log("Starting Optuna tuning: n_trials={0}".format(n_trials))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log("Best Optuna AUC={0:.6f}".format(study.best_value))
    best = default_lgbm_params(seed=seed, quick=quick)
    best.update(study.best_params)
    best["n_estimators"] = 7000 if not quick else 1500
    return best

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_submission(test: pd.DataFrame, sample: Optional[pd.DataFrame], pred: np.ndarray, output_dir: Path) -> Path:
    if sample is not None and ID_COL in sample.columns:
        sub = sample.copy()
        sub[TARGET] = pred
    else:
        if ID_COL not in test.columns:
            raise ValueError("test data has no id column and no sample_submission.csv was provided")
        sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred})
    sub[TARGET] = np.clip(sub[TARGET].astype(float), 0, 1)
    path = output_dir / "submission.csv"
    sub.to_csv(path, index=False)
    log("Saved submission: {0}".format(path))
    return path


def save_plots(output_dir: Path, oof: np.ndarray, y: np.ndarray, feature_importance: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib not installed; skipping plots.")
        return

    try:
        plt.figure(figsize=(8, 5))
        plt.hist(oof[y == 0], bins=50, alpha=0.6, label="target=0")
        plt.hist(oof[y == 1], bins=50, alpha=0.6, label="target=1")
        plt.xlabel("OOF predicted probability")
        plt.ylabel("count")
        plt.title("OOF prediction distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "oof_prediction_distribution.png", dpi=160)
        plt.close()
    except Exception as e:
        log("Could not save OOF plot: {0}".format(repr(e)))

    try:
        if feature_importance is not None and not feature_importance.empty:
            imp = feature_importance.groupby("feature", as_index=False)["importance_gain"].mean().sort_values("importance_gain", ascending=False).head(30)
            plt.figure(figsize=(9, 8))
            plt.barh(imp["feature"][::-1], imp["importance_gain"][::-1])
            plt.xlabel("mean gain importance")
            plt.title("Top 30 LightGBM feature importances")
            plt.tight_layout()
            plt.savefig(output_dir / "feature_importance_top30.png", dpi=160)
            plt.close()
    except Exception as e:
        log("Could not save importance plot: {0}".format(repr(e)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F1 PitNextLap LightGBM + feature selection pipeline")
    p.add_argument("--data-dir", type=str, default=".", help="Folder containing train.csv, test.csv, sample_submission.csv")
    p.add_argument("--output-dir", type=str, default=None, help="Output folder. Default: data-dir/runs/f1_pitnextlap_<timestamp>")
    p.add_argument("--original-train-csv", type=str, default=None, help="Optional external/original dataset CSV with PitNextLap target to append to train")
    p.add_argument("--n-splits", type=int, default=5, help="CV folds for final model")
    p.add_argument("--selection-splits", type=int, default=3, help="CV folds for one-by-one feature selection")
    p.add_argument("--cv-mode", type=str, default="stratified_group", choices=["stratified_group", "group", "stratified"], help="CV split mode")
    p.add_argument("--max-candidates", type=int, default=100, help="Maximum engineered features to test one-by-one after quick ranking")
    p.add_argument("--selection-rounds", type=int, default=1, help="Number of forward-selection passes")
    p.add_argument("--min-delta", type=float, default=0.00002, help="Minimum AUC improvement required to keep a feature")
    p.add_argument("--pair-search", action="store_true", help="After one-by-one selection, test pairs of remaining candidate features to catch interaction-only gains")
    p.add_argument("--pair-top-k", type=int, default=25, help="Top remaining candidates to use for pair search")
    p.add_argument("--pair-max-pairs", type=int, default=200, help="Maximum candidate pairs to test during pair search")
    p.add_argument("--pair-min-delta", type=float, default=None, help="Minimum AUC improvement required to keep a two-feature pair. Default: same as --min-delta")
    p.add_argument("--n-trials", type=int, default=30, help="Optuna trials. Set 0 to skip tuning")
    p.add_argument("--optuna-timeout", type=int, default=None, help="Optuna timeout in seconds")
    p.add_argument("--skip-feature-selection", action="store_true", help="Skip one-by-one feature selection and use baseline only")
    p.add_argument("--use-all-engineered", action="store_true", help="Use all engineered candidate features without selection")
    p.add_argument("--quick", action="store_true", help="Faster settings for testing the script")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if args.output_dir is None:
        output_dir = data_dir / "runs" / ("f1_pitnextlap_" + now_str())
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Output directory: {0}".format(output_dir))

    # Save args for reproducibility.
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    original_path = Path(args.original_train_csv).resolve() if args.original_train_csv else None
    train_raw, test_raw, sample = load_data(data_dir=data_dir, original_train_csv=original_path)
    if TARGET not in train_raw.columns:
        raise ValueError("train.csv must contain target column {0}".format(TARGET))

    train, test, base_features, candidates, categorical_features = add_engineered_features(train_raw, test_raw)

    # Missing target rows should not be used.
    train = train[train[TARGET].notna()].copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)
    y = pd.to_numeric(train[TARGET], errors="coerce").astype(int)

    if args.quick:
        args.max_candidates = min(args.max_candidates, 20)
        args.n_trials = min(args.n_trials, 5)
        args.n_splits = min(args.n_splits, 3)
        args.selection_splits = min(args.selection_splits, 2)
        args.pair_top_k = min(args.pair_top_k, 8)
        args.pair_max_pairs = min(args.pair_max_pairs, 20)
        log("Quick mode enabled.")

    selection_folds = make_folds(train, y, n_splits=args.selection_splits, mode=args.cv_mode)
    final_folds = make_folds(train, y, n_splits=args.n_splits, mode=args.cv_mode)

    selection_params = default_lgbm_params(seed=args.seed, quick=True)
    if args.skip_feature_selection:
        selected_features = list(base_features)
        selection_report = pd.DataFrame()
        log("Feature selection skipped. Using baseline features only.")
    elif args.use_all_engineered:
        selected_features = list(dict.fromkeys(base_features + candidates))
        selection_report = pd.DataFrame()
        log("Using all engineered features: {0}".format(len(selected_features)))
    else:
        selected_features, selection_report = forward_select_features(
            train=train,
            base_features=base_features,
            candidates=candidates,
            target_col=TARGET,
            categorical_features=categorical_features,
            params=selection_params,
            folds=selection_folds,
            min_delta=args.min_delta,
            max_candidates=args.max_candidates,
            selection_rounds=args.selection_rounds,
            output_dir=output_dir,
        )
        selection_report.to_csv(output_dir / "feature_selection_report.csv", index=False)

    if args.pair_search and not args.skip_feature_selection and not args.use_all_engineered:
        pair_min_delta = args.min_delta if args.pair_min_delta is None else args.pair_min_delta
        selected_features, pair_report = pair_select_features(
            train=train,
            selected_features=selected_features,
            candidates=candidates,
            target_col=TARGET,
            categorical_features=categorical_features,
            params=selection_params,
            folds=selection_folds,
            min_delta=pair_min_delta,
            pair_top_k=args.pair_top_k,
            max_pairs=args.pair_max_pairs,
            output_dir=output_dir,
        )
        pair_report.to_csv(output_dir / "pair_selection_report.csv", index=False)

    (output_dir / "selected_features.txt").write_text("\n".join(selected_features), encoding="utf-8")
    log("Selected feature count: {0}".format(len(selected_features)))

    # Tune hyperparameters using selected features.
    best_params = tune_optuna(
        train=train,
        features=selected_features,
        target_col=TARGET,
        categorical_features=categorical_features,
        folds=selection_folds if args.n_trials > 0 else final_folds,
        n_trials=args.n_trials,
        timeout=args.optuna_timeout,
        seed=args.seed,
        quick=args.quick,
    )
    (output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Best/final params saved.")

    log("Training final CV ensemble...")
    final_res = train_cv_predict(
        train=train,
        test=test,
        features=selected_features,
        target_col=TARGET,
        categorical_features=categorical_features,
        params=best_params,
        folds=final_folds,
        early_stopping_rounds=200 if not args.quick else 80,
        verbose_eval=100 if not args.quick else 0,
        return_importance=True,
    )
    final_auc = final_res["auc"]
    log("Final OOF AUC = {0:.6f}".format(final_auc))

    # Save OOF predictions.
    oof_df = pd.DataFrame({
        ID_COL: train[ID_COL].values if ID_COL in train.columns else np.arange(len(train)),
        TARGET: y.values,
        "oof_pred": final_res["oof"],
    })
    for c in ["DataSource", "RaceKey", "RaceYear", "Driver", "Compound", "LapNumber", "TyreLife", "RaceProgressClean"]:
        if c in train.columns:
            oof_df[c] = train[c].astype(str).values if str(train[c].dtype) == "category" else train[c].values
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Save submission.
    save_submission(test, sample, final_res["test_pred"], output_dir)

    # Feature importance.
    imp = final_res["importance"]
    if not imp.empty:
        imp_summary = imp.groupby("feature", as_index=False).agg(
            importance_gain_mean=("importance_gain", "mean"),
            importance_gain_std=("importance_gain", "std"),
            importance_split_mean=("importance_split", "mean"),
        ).sort_values("importance_gain_mean", ascending=False)
        imp_summary.to_csv(output_dir / "feature_importance.csv", index=False)
    else:
        imp_summary = pd.DataFrame()

    # Summary JSON.
    summary = {
        "final_oof_auc": final_auc,
        "fold_scores": final_res["fold_scores"],
        "best_iterations": final_res["best_iterations"],
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_base_features": int(len(base_features)),
        "n_candidate_features": int(len(candidates)),
        "n_selected_features": int(len(selected_features)),
        "output_dir": str(output_dir),
        "selected_features_first_30": selected_features[:30],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    save_plots(output_dir, final_res["oof"], y.values, imp)

    log("Done.")
    log("Important files:")
    log("  submission.csv")
    log("  run_summary.json")
    log("  feature_selection_report.csv")
    log("  selected_features.txt")
    log("  feature_importance.csv")


if __name__ == "__main__":
    main()
