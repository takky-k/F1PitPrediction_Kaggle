#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_pit_lgbm_feature_ab.py

F1 pit prediction 用の LightGBM パイプラインです。

やること:
1. train.csv / test.csv / 任意の original dataset を読み込む
2. original dataset から train/test と重複する行を削除して train に追加する
3. 基本クリーニングを行う
4. ベースライン特徴量で LightGBM + Optuna を実行する
5. 追加特徴量を1つずつABテストし、CV AUCが改善したものだけ採用する
6. 採用後、特徴量重要度が低い順に削除テストし、削除でCV AUCが改善すれば削除する
7. 最終特徴量で再度Optunaを実行し、OOF / submission / レポートを保存する

実行例:
    python f1_pit_lgbm_feature_ab.py --input-dir . --output-dir outputs --optuna-trials 50 --final-optuna-trials 100

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm optuna

任意:
    race_metadata.csv を input-dir に置くと、Race/Yearごとの外部メタデータを上書きできます。
    期待カラム例:
        Race, Year, RaceDistanceKm, RaceCornerCount, RaceRound, RaceMonth,
        Weather, Temperature, Humidity, is_sprint_weekend
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import optuna
except Exception:
    optuna = None


# ============================================================
# 1. 固定メタデータ: ないよりは強いfallback
#    正確なRace/Year別情報を使いたい場合は race_metadata.csv で上書きしてください。
# ============================================================


def _norm_text(x: Any) -> str:
    """Race名のゆらぎを吸収するための正規化。"""
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("grand prix", "")
    s = s.replace("gp", "")
    s = s.replace("formula 1", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# RaceDistanceKm は1周の距離、RaceCornerCount はコーナー数の概算。
# サーキット改修により年ごとに変わる場合があるため、race_metadata.csvで上書き推奨。
RACE_INFO_FALLBACK_RAW: Dict[str, Dict[str, Any]] = {
    "australian": {"RaceDistanceKm": 5.278, "RaceCornerCount": 14, "RaceRound": 3, "RaceMonth": 4},
    "bahrain": {"RaceDistanceKm": 5.412, "RaceCornerCount": 15, "RaceRound": 1, "RaceMonth": 3},
    "saudi arabian": {"RaceDistanceKm": 6.174, "RaceCornerCount": 27, "RaceRound": 2, "RaceMonth": 3},
    "emilia romagna": {"RaceDistanceKm": 4.909, "RaceCornerCount": 19, "RaceRound": 7, "RaceMonth": 5},
    "imola": {"RaceDistanceKm": 4.909, "RaceCornerCount": 19, "RaceRound": 7, "RaceMonth": 5},
    "miami": {"RaceDistanceKm": 5.412, "RaceCornerCount": 19, "RaceRound": 6, "RaceMonth": 5},
    "spanish": {"RaceDistanceKm": 4.657, "RaceCornerCount": 14, "RaceRound": 8, "RaceMonth": 6},
    "monaco": {"RaceDistanceKm": 3.337, "RaceCornerCount": 19, "RaceRound": 8, "RaceMonth": 5},
    "azerbaijan": {"RaceDistanceKm": 6.003, "RaceCornerCount": 20, "RaceRound": 4, "RaceMonth": 4},
    "canadian": {"RaceDistanceKm": 4.361, "RaceCornerCount": 14, "RaceRound": 9, "RaceMonth": 6},
    "british": {"RaceDistanceKm": 5.891, "RaceCornerCount": 18, "RaceRound": 10, "RaceMonth": 7},
    "70th anniversary": {"RaceDistanceKm": 5.891, "RaceCornerCount": 18, "RaceRound": 5, "RaceMonth": 8},
    "austrian": {"RaceDistanceKm": 4.318, "RaceCornerCount": 10, "RaceRound": 11, "RaceMonth": 7},
    "styrian": {"RaceDistanceKm": 4.318, "RaceCornerCount": 10, "RaceRound": 8, "RaceMonth": 6},
    "french": {"RaceDistanceKm": 5.842, "RaceCornerCount": 15, "RaceRound": 12, "RaceMonth": 7},
    "hungarian": {"RaceDistanceKm": 4.381, "RaceCornerCount": 14, "RaceRound": 13, "RaceMonth": 7},
    "belgian": {"RaceDistanceKm": 7.004, "RaceCornerCount": 19, "RaceRound": 14, "RaceMonth": 8},
    "dutch": {"RaceDistanceKm": 4.259, "RaceCornerCount": 14, "RaceRound": 15, "RaceMonth": 8},
    "italian": {"RaceDistanceKm": 5.793, "RaceCornerCount": 11, "RaceRound": 16, "RaceMonth": 9},
    "singapore": {"RaceDistanceKm": 4.940, "RaceCornerCount": 19, "RaceRound": 17, "RaceMonth": 9},
    "japanese": {"RaceDistanceKm": 5.807, "RaceCornerCount": 18, "RaceRound": 18, "RaceMonth": 9},
    "qatar": {"RaceDistanceKm": 5.419, "RaceCornerCount": 16, "RaceRound": 18, "RaceMonth": 10},
    "united states": {"RaceDistanceKm": 5.513, "RaceCornerCount": 20, "RaceRound": 19, "RaceMonth": 10},
    "mexico city": {"RaceDistanceKm": 4.304, "RaceCornerCount": 17, "RaceRound": 20, "RaceMonth": 10},
    "mexican": {"RaceDistanceKm": 4.304, "RaceCornerCount": 17, "RaceRound": 20, "RaceMonth": 10},
    "sao paulo": {"RaceDistanceKm": 4.309, "RaceCornerCount": 15, "RaceRound": 21, "RaceMonth": 11},
    "brazilian": {"RaceDistanceKm": 4.309, "RaceCornerCount": 15, "RaceRound": 21, "RaceMonth": 11},
    "las vegas": {"RaceDistanceKm": 6.201, "RaceCornerCount": 17, "RaceRound": 22, "RaceMonth": 11},
    "abu dhabi": {"RaceDistanceKm": 5.281, "RaceCornerCount": 16, "RaceRound": 23, "RaceMonth": 11},
    "chinese": {"RaceDistanceKm": 5.451, "RaceCornerCount": 16, "RaceRound": 5, "RaceMonth": 4},
    "portuguese": {"RaceDistanceKm": 4.653, "RaceCornerCount": 15, "RaceRound": 3, "RaceMonth": 5},
    "turkish": {"RaceDistanceKm": 5.338, "RaceCornerCount": 14, "RaceRound": 16, "RaceMonth": 10},
    "russian": {"RaceDistanceKm": 5.848, "RaceCornerCount": 18, "RaceRound": 15, "RaceMonth": 9},
    "tuscan": {"RaceDistanceKm": 5.245, "RaceCornerCount": 15, "RaceRound": 9, "RaceMonth": 9},
    "sakhir": {"RaceDistanceKm": 3.543, "RaceCornerCount": 11, "RaceRound": 16, "RaceMonth": 12},
    "eifel": {"RaceDistanceKm": 5.148, "RaceCornerCount": 15, "RaceRound": 11, "RaceMonth": 10},
}
RACE_INFO_FALLBACK = {_norm_text(k): v for k, v in RACE_INFO_FALLBACK_RAW.items()}

SPRINT_RACES_BY_YEAR_RAW: Dict[int, List[str]] = {
    2021: ["british", "italian", "sao paulo", "brazilian"],
    2022: ["emilia romagna", "austrian", "sao paulo", "brazilian"],
    2023: ["azerbaijan", "austrian", "belgian", "qatar", "united states", "sao paulo", "brazilian"],
    2024: ["chinese", "miami", "austrian", "united states", "sao paulo", "brazilian", "qatar"],
    2025: ["chinese", "miami", "belgian", "united states", "sao paulo", "brazilian", "qatar"],
}
SPRINT_RACES_BY_YEAR = {y: {_norm_text(r) for r in races} for y, races in SPRINT_RACES_BY_YEAR_RAW.items()}


# ============================================================
# 2. 汎用ユーティリティ
# ============================================================


def clean_column_name(col: Any) -> str:
    """LightGBMや保存ファイルで扱いやすい列名に変換。"""
    s = str(col).strip()
    s = re.sub(r"[\s\(\)\[\]\{\}/%+\-]+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s.lower() == "id":
        return "id"
    if not s:
        s = "col"
    if re.match(r"^\d", s):
        s = "col_" + s
    return s


def clean_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    used: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in df.columns:
        base = clean_column_name(c)
        name = base
        if name in used:
            used[name] += 1
            name = f"{base}_{used[base]}"
        else:
            used[name] = 0
        mapping[str(c)] = name
        new_cols.append(name)
    out = df.copy()
    out.columns = new_cols
    return out, mapping


def find_csv(input_dir: Path, explicit: Optional[str], candidates: Sequence[str], required: bool) -> Optional[Path]:
    if explicit and explicit.lower() not in {"auto", "none", "null"}:
        p = Path(explicit)
        if not p.is_absolute():
            p = input_dir / p
        if p.exists():
            return p
        if required:
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {p}")
        return None

    if explicit and explicit.lower() in {"none", "null"}:
        return None

    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p

    # ゆるく探索
    all_csvs = list(input_dir.glob("*.csv"))
    lower_map = {p.name.lower(): p for p in all_csvs}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for p in all_csvs:
        low = p.name.lower()
        for name in candidates:
            stem = Path(name).stem.lower()
            if stem in low:
                return p

    if required:
        raise FileNotFoundError(f"CSVが見つかりません。探した候補: {candidates} in {input_dir}")
    return None


def read_csv_clean(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)
    df, mapping = clean_columns(df)
    return df, mapping


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_divide(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    a_num = to_numeric_safe(a)
    b_num = to_numeric_safe(b)
    out = a_num / b_num.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def get_existing_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def row_hash(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df.loc[:, list(cols)].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).fillna("__NA__").astype(str)
    return pd.util.hash_pandas_object(tmp, index=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. original dataset の重複除去と結合
# ============================================================


def remove_original_overlap(
    original: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """originalからtrain/testと同一内容の行を削除する。"""
    report: Dict[str, Any] = {
        "original_rows_before": int(len(original)),
        "removed_overlap_train": 0,
        "removed_overlap_test": 0,
        "original_rows_after": int(len(original)),
        "status": "ok",
    }

    if original is None or len(original) == 0:
        report["status"] = "no_original"
        return original, report

    # targetがないoriginalは教師あり学習に追加できない
    if target_col not in original.columns:
        report["status"] = f"skip_original_no_{target_col}"
        report["original_rows_after"] = 0
        return original.iloc[0:0].copy(), report

    id_like = {id_col, target_col}

    # trainとの重複削除
    common_train = [c for c in original.columns if c in train.columns and c not in id_like]
    if len(common_train) >= 5:
        h_org = row_hash(original, common_train)
        h_train = set(row_hash(train, common_train).tolist())
        mask = ~h_org.isin(h_train)
        report["removed_overlap_train"] = int((~mask).sum())
        original = original.loc[mask].copy()

    # testとの重複削除
    common_test = [c for c in original.columns if c in test.columns and c not in id_like]
    if len(common_test) >= 5 and len(original) > 0:
        h_org = row_hash(original, common_test)
        h_test = set(row_hash(test, common_test).tolist())
        mask = ~h_org.isin(h_test)
        report["removed_overlap_test"] = int((~mask).sum())
        original = original.loc[mask].copy()

    report["original_rows_after"] = int(len(original))
    return original, report


# ============================================================
# 4. クリーニング
# ============================================================


def basic_train_clean(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """trainだけ厳しめに行削除。testは提出ID維持のため後で補完する。"""
    before = len(df)
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    if target_col in out.columns:
        out[target_col] = to_numeric_safe(out[target_col])
        out = out[out[target_col].isin([0, 1])].copy()
        out[target_col] = out[target_col].astype(int)

    filters: List[pd.Series] = []
    if "LapNumber" in out.columns:
        lap = to_numeric_safe(out["LapNumber"])
        filters.append(lap.gt(0))
    if "TyreLife" in out.columns:
        tl = to_numeric_safe(out["TyreLife"])
        filters.append(tl.ge(0))
    if "RaceProgress" in out.columns:
        rp = to_numeric_safe(out["RaceProgress"])
        filters.append(rp.between(0, 1, inclusive="both"))
    if "Position" in out.columns:
        pos = to_numeric_safe(out["Position"])
        filters.append(pos.between(1, 30, inclusive="both"))
    if "LapTime_s" in out.columns:
        lt = to_numeric_safe(out["LapTime_s"])
        filters.append(lt.gt(0))
    if "Year" in out.columns:
        yr = to_numeric_safe(out["Year"])
        filters.append(yr.between(1950, 2035, inclusive="both"))

    if filters:
        mask = pd.Series(True, index=out.index)
        for f in filters:
            mask &= f.fillna(False)
        out = out.loc[mask].copy()

    # raw列のNAを落としすぎるとoriginal追加時に危険なので、主要列だけチェック
    essential_cols = [
        c
        for c in ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "LapTime_s", target_col]
        if c in out.columns
    ]
    if essential_cols:
        out = out.dropna(subset=essential_cols).copy()

    report = {
        "train_rows_before_clean": int(before),
        "train_rows_after_clean": int(len(out)),
        "train_rows_removed_clean": int(before - len(out)),
    }
    return out.reset_index(drop=True), report


def basic_test_clean_keep_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """testは提出行数を維持する。異常値はNaNにして後で補完。"""
    out = df.copy().replace([np.inf, -np.inf], np.nan)
    report: Dict[str, Any] = {"test_rows": int(len(out)), "note": "test rows are kept for submission"}

    numeric_nonnegative = ["LapNumber", "TyreLife", "Position", "LapTime_s"]
    for c in numeric_nonnegative:
        if c in out.columns:
            x = to_numeric_safe(out[c])
            bad = x.lt(0) if c != "LapNumber" else x.le(0)
            out.loc[bad.fillna(False), c] = np.nan
    if "RaceProgress" in out.columns:
        x = to_numeric_safe(out["RaceProgress"])
        bad = ~x.between(0, 1, inclusive="both")
        out.loc[bad.fillna(False), "RaceProgress"] = np.nan
    return out.reset_index(drop=True), report


# ============================================================
# 5. レースメタデータ
# ============================================================


def fuzzy_race_info(race_value: Any) -> Dict[str, Any]:
    key = _norm_text(race_value)
    if key in RACE_INFO_FALLBACK:
        return dict(RACE_INFO_FALLBACK[key])
    for k, v in RACE_INFO_FALLBACK.items():
        if k and (k in key or key in k):
            return dict(v)
    return {
        "RaceDistanceKm": np.nan,
        "RaceCornerCount": np.nan,
        "RaceRound": np.nan,
        "RaceMonth": np.nan,
    }


def is_sprint_weekend_value(year: Any, race: Any) -> int:
    yr = pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0]
    if pd.isna(yr):
        return 0
    yr = int(yr)
    race_key = _norm_text(race)
    sprint_set = SPRINT_RACES_BY_YEAR.get(yr, set())
    for k in sprint_set:
        if k in race_key or race_key in k:
            return 1
    return 0


def load_external_metadata(input_dir: Path, explicit: Optional[str]) -> Optional[pd.DataFrame]:
    candidates = ["race_metadata.csv", "race_info.csv", "f1_race_metadata.csv"]
    p = find_csv(input_dir, explicit, candidates, required=False)
    if p is None:
        return None
    meta, _ = read_csv_clean(p)
    if "Race" not in meta.columns:
        print(f"[WARN] {p.name} に Race カラムがないため無視します。")
        return None
    meta["Race_norm_key"] = meta["Race"].map(_norm_text)
    if "Year" in meta.columns:
        meta["Year_num_key"] = to_numeric_safe(meta["Year"]).astype("Int64")
    else:
        meta["Year_num_key"] = pd.NA
    print(f"[INFO] external metadata loaded: {p} rows={len(meta)}")
    return meta


def apply_external_metadata(all_df: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if meta is None or len(meta) == 0:
        return all_df

    out = all_df.copy()
    out["Race_norm_key"] = out["Race"].map(_norm_text) if "Race" in out.columns else ""
    out["Year_num_key"] = to_numeric_safe(out["Year"]).astype("Int64") if "Year" in out.columns else pd.NA

    update_cols = [
        "RaceDistanceKm", "RaceCornerCount", "RaceRound", "RaceMonth", "Weather", "Temperature", "Humidity", "is_sprint_weekend"
    ]
    update_cols = [c for c in update_cols if c in meta.columns]
    if not update_cols:
        return out.drop(columns=[c for c in ["Race_norm_key", "Year_num_key"] if c in out.columns], errors="ignore")

    # Yearありでmerge
    meta_year = meta.dropna(subset=["Year_num_key"]).copy()
    if len(meta_year) > 0:
        keep = ["Race_norm_key", "Year_num_key"] + update_cols
        merged = out.merge(meta_year[keep].drop_duplicates(["Race_norm_key", "Year_num_key"]), on=["Race_norm_key", "Year_num_key"], how="left", suffixes=("", "_ext"))
        for c in update_cols:
            ext = c + "_ext"
            if ext in merged.columns:
                merged[c] = merged[ext].combine_first(merged[c] if c in merged.columns else pd.Series(np.nan, index=merged.index))
        out = merged.drop(columns=[c for c in merged.columns if c.endswith("_ext")])

    # Yearなしfallbackでmerge
    meta_race = meta[meta["Year_num_key"].isna()].copy()
    if len(meta_race) > 0:
        keep = ["Race_norm_key"] + update_cols
        merged = out.merge(meta_race[keep].drop_duplicates(["Race_norm_key"]), on=["Race_norm_key"], how="left", suffixes=("", "_ext"))
        for c in update_cols:
            ext = c + "_ext"
            if ext in merged.columns:
                merged[c] = merged[ext].combine_first(merged[c] if c in merged.columns else pd.Series(np.nan, index=merged.index))
        out = merged.drop(columns=[c for c in merged.columns if c.endswith("_ext")])

    return out.drop(columns=[c for c in ["Race_norm_key", "Year_num_key"] if c in out.columns], errors="ignore")


# ============================================================
# 6. 特徴量作成
# ============================================================


def add_engineered_features(all_df: pd.DataFrame, external_meta: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """train/testを結合したDataFrameに特徴量を作る。"""
    df = all_df.copy()
    created: List[Dict[str, Any]] = []

    def add_num(name: str, values: Any, desc: str) -> None:
        df[name] = values
        created.append({"feature": name, "type": "numeric", "description": desc})

    def add_cat(name: str, values: Any, desc: str) -> None:
        df[name] = values
        created.append({"feature": name, "type": "categorical", "description": desc})

    # 必須列がなければNaNで作るためのhelper
    def col(name: str, default: float = np.nan) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(default, index=df.index)

    # 数値化した作業用Series
    lap = to_numeric_safe(col("LapNumber"))
    tyre = to_numeric_safe(col("TyreLife"))
    progress = to_numeric_safe(col("RaceProgress"))
    stint = to_numeric_safe(col("Stint"))
    pos = to_numeric_safe(col("Position"))
    lap_time = to_numeric_safe(col("LapTime_s"))
    delta = to_numeric_safe(col("LapTime_Delta"))
    degr = to_numeric_safe(col("Cumulative_Degradation"))
    pos_change = to_numeric_safe(col("Position_Change"))
    year = to_numeric_safe(col("Year"))

    # ユーザー指定・追加希望の特徴量
    add_num("tyre_life_ratio", safe_divide(tyre, lap), "TyreLife / LapNumber: 今のタイヤをどれくらい長く使っているか")
    add_num("progress_x_tyre", progress * tyre, "RaceProgress * TyreLife: レース進行度とタイヤ劣化の組み合わせ")
    if "Compound" in df.columns:
        tyre_bin = pd.cut(
            tyre,
            bins=[-1, 3, 7, 12, 18, 25, 35, 10000],
            labels=["00_03", "04_07", "08_12", "13_18", "19_25", "26_35", "36_plus"],
        ).astype(str)
        add_cat("compound_tyre_life", df["Compound"].astype(str).fillna("__MISSING__") + "__TL_" + tyre_bin, "Compound × TyreLife bin")
    else:
        add_cat("compound_tyre_life", "__MISSING__", "Compound × TyreLife bin")
    add_num("stint_progress", stint * progress, "Stint * RaceProgress: 何回目のスティントでレースのどこか")
    add_num("lap_delta_abs", delta.abs(), "abs(LapTime_Delta): 異常ラップの検出")

    # Raceごとの平均ピット周回: PitStop==1を使う。targetは使わない。
    if "Race" in df.columns and "PitStop" in df.columns and "LapNumber" in df.columns:
        pit_flag = to_numeric_safe(df["PitStop"]).fillna(0)
        tmp = df[["Race"]].copy()
        tmp["LapNumber_num"] = lap
        tmp["PitStop_num"] = pit_flag
        pit_mean_by_race = tmp.loc[tmp["PitStop_num"].eq(1)].groupby("Race")["LapNumber_num"].mean()
        add_num("race_avg_pit_lap", df["Race"].map(pit_mean_by_race), "Raceごとの平均ピット周回: PitStop==1のLapNumber平均")
    else:
        add_num("race_avg_pit_lap", np.nan, "Raceごとの平均ピット周回")

    # メモ由来の追加特徴量
    if "Driver" in df.columns:
        add_num("D_number", df["Driver"].astype(str).str.match(r"^D\d+$", na=False).astype(int), "DriverがD+数値形式かどうか")
    else:
        add_num("D_number", 0, "DriverがD+数値形式かどうか")

    add_num("Is_2023", year.eq(2023).astype(int), "Year == 2023 flag")

    if "Race" in df.columns:
        info_rows = df["Race"].map(fuzzy_race_info)
        info_df = pd.DataFrame(list(info_rows), index=df.index)
        for c in ["RaceDistanceKm", "RaceCornerCount", "RaceRound", "RaceMonth"]:
            if c not in df.columns:
                add_num(c, info_df[c], f"{c}: Race名からfallback辞書で付与")
    else:
        for c in ["RaceDistanceKm", "RaceCornerCount", "RaceRound", "RaceMonth"]:
            if c not in df.columns:
                add_num(c, np.nan, f"{c}: Race名なし")

    if "RaceRound" in df.columns:
        round_num = to_numeric_safe(df["RaceRound"])
        if "Year" in df.columns:
            max_round = round_num.groupby(year).transform("max")
        else:
            max_round = pd.Series(round_num.max(), index=df.index)
        add_num("SeasonProgress", safe_divide(round_num, max_round), "RaceRound / その年の最大Round")
    else:
        add_num("SeasonProgress", np.nan, "RaceRound / その年の最大Round")

    # Weather系は外部CSVがあれば上書きされる。fallbackはUnknown/NaN。
    if "Weather" not in df.columns:
        add_cat("Weather", "Unknown", "天気カテゴリ。race_metadata.csvがあれば上書き")
    if "Temperature" not in df.columns:
        add_num("Temperature", np.nan, "気温。race_metadata.csvがあれば上書き")
    if "Humidity" not in df.columns:
        add_num("Humidity", np.nan, "湿度。race_metadata.csvがあれば上書き")

    if "Race" in df.columns:
        sprint_values = [is_sprint_weekend_value(y, r) for y, r in zip(df.get("Year", pd.Series(np.nan, index=df.index)), df["Race"])]
        add_num("is_sprint_weekend", pd.Series(sprint_values, index=df.index).astype(int), "Sprint weekend flag")
    else:
        add_num("is_sprint_weekend", 0, "Sprint weekend flag")

    if "Race" in df.columns:
        race_norm = df["Race"].map(_norm_text)
    else:
        race_norm = pd.Series("", index=df.index)

    is_qatar = race_norm.str.contains("qatar", na=False)
    is_monaco = race_norm.str.contains("monaco", na=False)
    add_num("is_qatar_2023", (is_qatar & year.eq(2023)).astype(int), "Qatar 2023 flag")
    add_num("qatar_2023_near_limit", (is_qatar & year.eq(2023) & tyre.ge(15)).astype(int), "Qatar 2023 18周制限に近いタイヤ年齢flag")
    add_num("is_monaco_2025", (is_monaco & year.eq(2025)).astype(int), "Monaco 2025 special flag")
    add_num("is_qatar_2025", (is_qatar & year.eq(2025)).astype(int), "Qatar 2025 flag")
    add_num("qatar_2025_near_limit", (is_qatar & year.eq(2025) & tyre.ge(22)).astype(int), "Qatar 2025 25周制限に近いタイヤ年齢flag")

    add_num("LapNumber_minus_TyreLife", lap - tyre, "LapNumber - TyreLife: 前回ピットタイミングに近い情報")

    # AveLapTime: 同じRaceの平均ラップタイム
    if "Race" in df.columns and "LapTime_s" in df.columns:
        ave_lap = lap_time.groupby(df["Race"]).transform("mean")
        add_num("AveLapTime", ave_lap, "同じRace内の平均LapTime_s")
    else:
        add_num("AveLapTime", np.nan, "同じRace内の平均LapTime_s")

    # 人間が意味を説明しやすい追加候補
    add_num("lap_time_minus_race_avg", lap_time - to_numeric_safe(df.get("AveLapTime", pd.Series(np.nan, index=df.index))), "LapTime_s - AveLapTime")
    add_num("lap_time_ratio_race_avg", safe_divide(lap_time, to_numeric_safe(df.get("AveLapTime", pd.Series(np.nan, index=df.index))), 1.0), "LapTime_s / AveLapTime")
    add_num("tyre_x_lap_delta", tyre * delta, "TyreLife * LapTime_Delta")
    add_num("tyre_x_degradation", tyre * degr, "TyreLife * Cumulative_Degradation")
    add_num("progress_x_degradation", progress * degr, "RaceProgress * Cumulative_Degradation")
    add_num("position_x_progress", pos * progress, "Position * RaceProgress")
    add_num("position_change_abs", pos_change.abs(), "abs(Position_Change)")
    add_num("tyre_life_squared", tyre ** 2, "TyreLife^2")
    add_num("race_progress_squared", progress ** 2, "RaceProgress^2")
    add_num("stint_x_tyre", stint * tyre, "Stint * TyreLife")
    add_num("stint_x_lap", stint * lap, "Stint * LapNumber")
    add_num("remaining_progress", 1.0 - progress, "1 - RaceProgress")
    add_num("tyre_per_remaining_progress", safe_divide(tyre, 1.0 - progress, 0.0), "TyreLife / remaining progress")
    add_num("lap_per_progress", safe_divide(lap, progress, 0.0), "LapNumber / RaceProgress: 推定総周回数に近い値")
    if "RaceDistanceKm" in df.columns:
        add_num("estimated_total_distance_km", to_numeric_safe(df["RaceDistanceKm"]) * safe_divide(lap, progress, 0.0), "RaceDistanceKm * 推定総周回数")
        add_num("current_distance_km", to_numeric_safe(df["RaceDistanceKm"]) * lap, "RaceDistanceKm * LapNumber")
    if "RaceCornerCount" in df.columns:
        add_num("corners_completed", to_numeric_safe(df["RaceCornerCount"]) * lap, "RaceCornerCount * LapNumber")
        add_num("tyre_corner_load", to_numeric_safe(df["RaceCornerCount"]) * tyre, "RaceCornerCount * TyreLife")

    # 外部race_metadata.csvで上書き
    df = apply_external_metadata(df, external_meta)

    # SeasonProgress再計算: external metadataでRaceRoundが上書きされた場合に対応
    if "RaceRound" in df.columns:
        round_num = to_numeric_safe(df["RaceRound"])
        if "Year" in df.columns:
            max_round = round_num.groupby(to_numeric_safe(df["Year"])).transform("max")
        else:
            max_round = pd.Series(round_num.max(), index=df.index)
        df["SeasonProgress"] = safe_divide(round_num, max_round)

    return df, created


# ============================================================
# 7. 特徴量リストと型処理
# ============================================================


def base_categorical_candidates() -> List[str]:
    return [
        "Driver",
        "Compound",
        "Race",
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "Weather",
        "compound_tyre_life",
    ]


def feature_groups() -> List[Tuple[str, List[str]]]:
    """ABテストする特徴量グループ。基本は1グループ1特徴量。"""
    groups = [
        ("tyre_life_ratio", ["tyre_life_ratio"]),
        ("progress_x_tyre", ["progress_x_tyre"]),
        ("compound_tyre_life", ["compound_tyre_life"]),
        ("stint_progress", ["stint_progress"]),
        ("lap_delta_abs", ["lap_delta_abs"]),
        ("race_avg_pit_lap", ["race_avg_pit_lap"]),
        ("D_number", ["D_number"]),
        ("Is_2023", ["Is_2023"]),
        ("RaceDistanceKm", ["RaceDistanceKm"]),
        ("RaceCornerCount", ["RaceCornerCount"]),
        ("RaceRound", ["RaceRound"]),
        ("RaceMonth", ["RaceMonth"]),
        ("SeasonProgress", ["SeasonProgress"]),
        ("Weather", ["Weather"]),
        ("Temperature", ["Temperature"]),
        ("Humidity", ["Humidity"]),
        ("is_sprint_weekend", ["is_sprint_weekend"]),
        ("is_qatar_2023", ["is_qatar_2023"]),
        ("qatar_2023_near_limit", ["qatar_2023_near_limit"]),
        ("is_monaco_2025", ["is_monaco_2025"]),
        ("is_qatar_2025", ["is_qatar_2025"]),
        ("qatar_2025_near_limit", ["qatar_2025_near_limit"]),
        ("LapNumber_minus_TyreLife", ["LapNumber_minus_TyreLife"]),
        ("AveLapTime", ["AveLapTime"]),
        ("lap_time_minus_race_avg", ["lap_time_minus_race_avg"]),
        ("lap_time_ratio_race_avg", ["lap_time_ratio_race_avg"]),
        ("tyre_x_lap_delta", ["tyre_x_lap_delta"]),
        ("tyre_x_degradation", ["tyre_x_degradation"]),
        ("progress_x_degradation", ["progress_x_degradation"]),
        ("position_x_progress", ["position_x_progress"]),
        ("position_change_abs", ["position_change_abs"]),
        ("tyre_life_squared", ["tyre_life_squared"]),
        ("race_progress_squared", ["race_progress_squared"]),
        ("stint_x_tyre", ["stint_x_tyre"]),
        ("stint_x_lap", ["stint_x_lap"]),
        ("remaining_progress", ["remaining_progress"]),
        ("tyre_per_remaining_progress", ["tyre_per_remaining_progress"]),
        ("lap_per_progress", ["lap_per_progress"]),
        ("estimated_total_distance_km", ["estimated_total_distance_km"]),
        ("current_distance_km", ["current_distance_km"]),
        ("corners_completed", ["corners_completed"]),
        ("tyre_corner_load", ["tyre_corner_load"]),
    ]
    return groups


def valid_feature_list(df_train: pd.DataFrame, df_test: pd.DataFrame, features: Sequence[str]) -> List[str]:
    out: List[str] = []
    for f in features:
        if f in df_train.columns and f in df_test.columns and f not in out:
            out.append(f)
    return out


def is_non_constant_feature(train_df: pd.DataFrame, feature: str) -> bool:
    if feature not in train_df.columns:
        return False
    s = train_df[feature]
    return s.nunique(dropna=True) > 1


def prepare_X_y(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    categorical_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], Dict[str, Any]]:
    """LightGBM用に欠損補完・カテゴリ型変換をする。"""
    features = list(features)
    cat_features = [c for c in categorical_features if c in features]
    num_features = [c for c in features if c not in cat_features]

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y = train_df[target_col].astype(int).copy()

    impute_report: Dict[str, Any] = {"numeric_medians": {}, "categorical_fill": "__MISSING__"}

    # 数値列補完
    for c in num_features:
        X_train[c] = to_numeric_safe(X_train[c])
        X_test[c] = to_numeric_safe(X_test[c])
        med = X_train[c].median()
        if pd.isna(med):
            med = 0.0
        impute_report["numeric_medians"][c] = float(med)
        X_train[c] = X_train[c].fillna(med).replace([np.inf, -np.inf], med)
        X_test[c] = X_test[c].fillna(med).replace([np.inf, -np.inf], med)

    # カテゴリ列補完・カテゴリ統一
    for c in cat_features:
        tr = X_train[c].astype(str).where(~X_train[c].isna(), "__MISSING__")
        te = X_test[c].astype(str).where(~X_test[c].isna(), "__MISSING__")
        cats = pd.Index(pd.concat([tr, te], axis=0).unique())
        X_train[c] = pd.Categorical(tr, categories=cats)
        X_test[c] = pd.Categorical(te, categories=cats)

    return X_train, y, X_test, cat_features, impute_report


# ============================================================
# 8. LightGBM CV / Optuna
# ============================================================


def default_lgbm_params(seed: int, n_jobs: int) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "n_estimators": 4000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 60,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "min_split_gain": 0.0,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "force_col_wise": True,
    }


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    categorical_features: Sequence[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    return_oof: bool = False,
    return_importance: bool = False,
) -> Dict[str, Any]:
    if lgb is None:
        raise ImportError("lightgbm が見つかりません。pip install lightgbm を実行してください。")

    features = list(features)
    cat_features = [c for c in categorical_features if c in features]
    valid_aucs: List[float] = []
    train_aucs: List[float] = []
    best_iterations: List[int] = []
    oof = np.zeros(len(X), dtype=float) if return_oof else None
    importances: List[pd.DataFrame] = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[tr_idx][features]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx][features]
        y_va = y.iloc[va_idx]

        model_params = dict(params)
        model_params["random_state"] = seed + fold
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features if cat_features else "auto",
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False), lgb.log_evaluation(period=0)],
        )

        p_va = model.predict_proba(X_va)[:, 1]
        p_tr = model.predict_proba(X_tr)[:, 1]
        va_auc = roc_auc_score(y_va, p_va)
        tr_auc = roc_auc_score(y_tr, p_tr)
        valid_aucs.append(float(va_auc))
        train_aucs.append(float(tr_auc))
        best_iterations.append(int(model.best_iteration_ or model_params.get("n_estimators", 0)))
        if return_oof and oof is not None:
            oof[va_idx] = p_va
        if return_importance:
            gain = model.booster_.feature_importance(importance_type="gain")
            split = model.booster_.feature_importance(importance_type="split")
            importances.append(
                pd.DataFrame({
                    "feature": features,
                    f"gain_fold_{fold}": gain,
                    f"split_fold_{fold}": split,
                })
            )

    result: Dict[str, Any] = {
        "valid_auc_mean": float(np.mean(valid_aucs)),
        "valid_auc_std": float(np.std(valid_aucs)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "train_auc_std": float(np.std(train_aucs)),
        "overfit_gap_mean": float(np.mean(np.array(train_aucs) - np.array(valid_aucs))),
        "best_iteration_mean": float(np.mean(best_iterations)),
        "best_iteration_median": float(np.median(best_iterations)),
        "fold_valid_aucs": valid_aucs,
        "fold_train_aucs": train_aucs,
        "features": list(features),
        "n_features": int(len(features)),
    }
    if return_oof and oof is not None:
        result["oof"] = oof
    if return_importance and importances:
        imp = importances[0]
        for x in importances[1:]:
            imp = imp.merge(x, on="feature", how="outer")
        gain_cols = [c for c in imp.columns if c.startswith("gain_fold_")]
        split_cols = [c for c in imp.columns if c.startswith("split_fold_")]
        imp["gain_mean"] = imp[gain_cols].mean(axis=1)
        imp["split_mean"] = imp[split_cols].mean(axis=1)
        imp = imp.sort_values("gain_mean", ascending=False).reset_index(drop=True)
        result["importance"] = imp
    return result


def tune_lgbm_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    categorical_features: Sequence[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    n_trials: int,
    study_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = default_lgbm_params(seed=seed, n_jobs=n_jobs)
    if optuna is None or n_trials <= 0:
        eval_result = evaluate_cv(X, y, features, categorical_features, base, folds, seed)
        return base, {"status": "optuna_skipped", "score": eval_result["valid_auc_mean"], "params": base}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = default_lgbm_params(seed=seed, n_jobs=n_jobs)
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 256),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8, 10, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 240),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.20),
        })
        res = evaluate_cv(X, y, features, categorical_features, params, folds, seed)
        trial.set_user_attr("train_auc_mean", res["train_auc_mean"])
        trial.set_user_attr("overfit_gap_mean", res["overfit_gap_mean"])
        trial.set_user_attr("best_iteration_mean", res["best_iteration_mean"])
        return float(res["valid_auc_mean"])

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = default_lgbm_params(seed=seed, n_jobs=n_jobs)
    best_params.update(study.best_params)
    summary = {
        "status": "ok",
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": int(len(study.trials)),
        "study_name": study_name,
    }
    return best_params, summary


# ============================================================
# 9. ABテストと削除テスト
# ============================================================


def should_accept(
    candidate_auc: float,
    current_auc: float,
    candidate_gap: float,
    current_gap: float,
    min_delta: float,
    max_gap_increase: float,
) -> bool:
    return (candidate_auc > current_auc + min_delta) and (candidate_gap <= current_gap + max_gap_increase)


def run_ab_feature_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    baseline_features: List[str],
    categorical_all: List[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    min_delta: float,
    max_gap_increase: float,
    output_dir: Path,
) -> Tuple[List[str], Dict[str, Any], pd.DataFrame]:
    print("\n[STEP] AB feature selection")
    selected = list(baseline_features)

    X, y, X_test, cats, _ = prepare_X_y(train_df, test_df, selected, target_col, categorical_all)
    current = evaluate_cv(X, y, selected, cats, params, folds, seed)
    current_auc = current["valid_auc_mean"]
    current_gap = current["overfit_gap_mean"]
    print(f"[BASELINE] features={len(selected)} valid_auc={current_auc:.6f} gap={current_gap:.6f}")

    rows: List[Dict[str, Any]] = []
    for group_name, feats in feature_groups():
        add_feats = [f for f in feats if f not in selected and f in train_df.columns and f in test_df.columns and is_non_constant_feature(train_df, f)]
        if not add_feats:
            rows.append({
                "group": group_name,
                "features": ",".join(feats),
                "status": "skipped_missing_or_constant",
                "accepted": False,
                "before_auc": current_auc,
                "after_auc": np.nan,
                "delta_auc": np.nan,
                "after_gap": np.nan,
            })
            continue

        trial_features = selected + add_feats
        try:
            before_auc = float(current_auc)
            before_gap = float(current_gap)
            X_trial, y, _, cats_trial, _ = prepare_X_y(train_df, test_df, trial_features, target_col, categorical_all)
            res = evaluate_cv(X_trial, y, trial_features, cats_trial, params, folds, seed)
            cand_auc = float(res["valid_auc_mean"])
            cand_gap = float(res["overfit_gap_mean"])
            delta_auc = cand_auc - before_auc
            accepted = should_accept(cand_auc, before_auc, cand_gap, before_gap, min_delta, max_gap_increase)
            status = "accepted" if accepted else "rejected"
            if accepted:
                selected = trial_features
                current = res
                current_auc = cand_auc
                current_gap = cand_gap
            print(
                f"[AB] {group_name:30s} {status:8s} "
                f"auc={cand_auc:.6f} delta={delta_auc:+.6f} "
                f"gap={cand_gap:.6f} selected={len(selected)}"
            )
            rows.append({
                "group": group_name,
                "features": ",".join(add_feats),
                "status": status,
                "accepted": bool(accepted),
                "before_auc": before_auc,
                "after_auc": cand_auc,
                "delta_auc_vs_previous": float(delta_auc),
                "before_gap": before_gap,
                "after_gap": cand_gap,
                "valid_auc_std": float(res["valid_auc_std"]),
                "train_auc_mean": float(res["train_auc_mean"]),
                "fold_valid_aucs": json.dumps(res["fold_valid_aucs"]),
                "n_features_after_trial": int(len(trial_features)),
            })
        except Exception as e:
            print(f"[AB][ERROR] {group_name}: {e}")
            rows.append({
                "group": group_name,
                "features": ",".join(add_feats),
                "status": f"error: {e}",
                "accepted": False,
                "before_auc": float(current_auc),
                "after_auc": np.nan,
                "delta_auc_vs_previous": np.nan,
                "after_gap": np.nan,
            })

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "feature_selection_report.csv", index=False)
    return selected, current, report


def run_backward_elimination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    features: List[str],
    categorical_all: List[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    min_delta: float,
    max_gap_increase: float,
    min_features: int,
    output_dir: Path,
) -> Tuple[List[str], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    print("\n[STEP] backward elimination from lowest importance")
    selected = list(features)
    X, y, _, cats, _ = prepare_X_y(train_df, test_df, selected, target_col, categorical_all)
    current = evaluate_cv(X, y, selected, cats, params, folds, seed, return_importance=True)
    current_auc = current["valid_auc_mean"]
    current_gap = current["overfit_gap_mean"]
    imp = current.get("importance", pd.DataFrame({"feature": selected, "gain_mean": 0.0}))
    imp.to_csv(output_dir / "feature_importance_before_elimination.csv", index=False)

    order = imp.sort_values(["gain_mean", "split_mean" if "split_mean" in imp.columns else "gain_mean"], ascending=True)["feature"].tolist()
    rows: List[Dict[str, Any]] = []

    for feat in order:
        if feat not in selected:
            continue
        if len(selected) <= min_features:
            rows.append({"feature": feat, "status": "skipped_min_features", "accepted_removal": False})
            continue

        trial_features = [f for f in selected if f != feat]
        try:
            X_trial, y, _, cats_trial, _ = prepare_X_y(train_df, test_df, trial_features, target_col, categorical_all)
            res = evaluate_cv(X_trial, y, trial_features, cats_trial, params, folds, seed)
            cand_auc = res["valid_auc_mean"]
            cand_gap = res["overfit_gap_mean"]
            remove_it = should_accept(cand_auc, current_auc, cand_gap, current_gap, min_delta, max_gap_increase)
            status = "removed" if remove_it else "kept"
            print(f"[DEL] {feat:30s} {status:8s} auc={cand_auc:.6f} current={current_auc:.6f} n={len(trial_features)}")
            rows.append({
                "feature": feat,
                "status": status,
                "accepted_removal": bool(remove_it),
                "before_auc": float(current_auc),
                "after_auc": float(cand_auc),
                "delta_auc": float(cand_auc - current_auc),
                "before_gap": float(current_gap),
                "after_gap": float(cand_gap),
                "n_features_after_trial": int(len(trial_features)),
            })
            if remove_it:
                selected = trial_features
                current = res
                current_auc = cand_auc
                current_gap = cand_gap
        except Exception as e:
            print(f"[DEL][ERROR] {feat}: {e}")
            rows.append({
                "feature": feat,
                "status": f"error: {e}",
                "accepted_removal": False,
                "before_auc": float(current_auc),
                "after_auc": np.nan,
                "delta_auc": np.nan,
            })

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "backward_elimination_report.csv", index=False)

    X_final, y, _, cats_final, _ = prepare_X_y(train_df, test_df, selected, target_col, categorical_all)
    final_res = evaluate_cv(X_final, y, selected, cats_final, params, folds, seed, return_importance=True)
    final_imp = final_res.get("importance", pd.DataFrame({"feature": selected, "gain_mean": 0.0}))
    final_imp.to_csv(output_dir / "feature_importance_after_elimination.csv", index=False)
    return selected, final_res, report, final_imp


# ============================================================
# 10. 最終学習・予測
# ============================================================


def train_final_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
    params: Dict[str, Any],
    best_iteration_mean: float,
    seed: int,
    output_dir: Path,
) -> np.ndarray:
    if lgb is None:
        raise ImportError("lightgbm が見つかりません。pip install lightgbm を実行してください。")
    final_params = dict(params)
    if best_iteration_mean and best_iteration_mean > 0:
        final_params["n_estimators"] = max(200, int(best_iteration_mean * 1.15))
    final_params["random_state"] = seed
    model = lgb.LGBMClassifier(**final_params)
    cats = [c for c in categorical_features if c in features]
    model.fit(X[features], y, categorical_feature=cats if cats else "auto")
    try:
        model.booster_.save_model(str(output_dir / "final_lgbm_model.txt"))
    except Exception as e:
        print(f"[WARN] model save failed: {e}")
    preds = model.predict_proba(X_test[features])[:, 1]
    return preds


# ============================================================
# 11. main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 PitNextLap LightGBM feature AB testing pipeline")
    parser.add_argument("--input-dir", type=str, default=".", help="train/test/original CSVがあるフォルダ")
    parser.add_argument("--output-dir", type=str, default="pit_model_outputs", help="出力先フォルダ")
    parser.add_argument("--train-file", type=str, default="auto", help="train.csvのパス。auto可")
    parser.add_argument("--test-file", type=str, default="auto", help="test.csvのパス。auto可")
    parser.add_argument("--original-file", type=str, default="auto", help="original dataset CSV。noneで無効化")
    parser.add_argument("--metadata-file", type=str, default="auto", help="race_metadata.csv。noneで無効化")
    parser.add_argument("--target", type=str, default="PitNextLap", help="目的変数名")
    parser.add_argument("--id-col", type=str, default="id", help="提出用ID列名")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--optuna-trials", type=int, default=30, help="最初のOptuna trial数。0でskip")
    parser.add_argument("--final-optuna-trials", type=int, default=60, help="最終Optuna trial数。0でskip")
    parser.add_argument("--min-delta", type=float, default=1e-5, help="特徴量採用/削除の最小AUC改善幅")
    parser.add_argument("--max-gap-increase", type=float, default=0.015, help="train-valid gapの許容悪化幅")
    parser.add_argument("--min-features", type=int, default=8, help="削除テストで最低限残す特徴量数")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if lgb is None:
        raise ImportError("lightgbm が見つかりません。先に pip install lightgbm を実行してください。")
    if optuna is None and (args.optuna_trials > 0 or args.final_optuna_trials > 0):
        print("[WARN] optuna が見つかりません。Optunaをskipしてdefault paramsで実行します。")

    target_col = clean_column_name(args.target)
    id_col = clean_column_name(args.id_col)

    train_path = find_csv(input_dir, args.train_file, ["train.csv"], required=True)
    test_path = find_csv(input_dir, args.test_file, ["test.csv"], required=True)
    original_path = find_csv(
        input_dir,
        args.original_file,
        ["f1_strategy_dataset_v4.csv", "original.csv", "f1_strategy_dataset.csv"],
        required=False,
    )

    print(f"[INFO] train: {train_path}")
    print(f"[INFO] test : {test_path}")
    print(f"[INFO] original: {original_path if original_path else 'not used'}")

    train_raw, train_mapping = read_csv_clean(train_path)
    test_raw, test_mapping = read_csv_clean(test_path)
    write_json({"train": train_mapping, "test": test_mapping}, output_dir / "column_mapping.json")

    if target_col not in train_raw.columns:
        raise ValueError(f"target column '{target_col}' がtrainにありません。列一覧: {train_raw.columns.tolist()}")
    if id_col not in test_raw.columns:
        print(f"[WARN] testにID列 '{id_col}' がありません。indexをidとして使います。")
        test_raw[id_col] = np.arange(len(test_raw))

    # original結合
    original_report: Dict[str, Any] = {"status": "not_used"}
    if original_path is not None:
        original_raw, original_mapping = read_csv_clean(original_path)
        write_json(original_mapping, output_dir / "original_column_mapping.json")
        original_clean, original_report = remove_original_overlap(original_raw, train_raw, test_raw, target_col, id_col)
        # train/testにない列は落とし、trainと列を合わせる
        if len(original_clean) > 0:
            common_for_train = [c for c in train_raw.columns if c in original_clean.columns]
            original_clean = original_clean[common_for_train].copy()
            train_raw = pd.concat([train_raw, original_clean], axis=0, ignore_index=True)
            print(f"[INFO] original added rows: {len(original_clean)}")
        else:
            print(f"[INFO] original skipped or no usable rows. status={original_report.get('status')}")

    train_raw, train_clean_report = basic_train_clean(train_raw, target_col)
    test_raw, test_clean_report = basic_test_clean_keep_ids(test_raw)

    # ベースライン特徴量: train/testの共通raw列のみ。id/targetは使わない。
    exclude = {id_col, target_col}
    baseline_features = [c for c in train_raw.columns if c in test_raw.columns and c not in exclude]
    if not baseline_features:
        raise ValueError("train/testの共通特徴量がありません。列名を確認してください。")

    print(f"[INFO] train rows after clean: {len(train_raw):,}")
    print(f"[INFO] test rows: {len(test_raw):,}")
    print(f"[INFO] baseline raw features: {len(baseline_features)}")

    # train/testを結合して同じ特徴量処理を行う
    train_raw["__is_train"] = 1
    test_raw["__is_train"] = 0
    all_cols = sorted(set(train_raw.columns).union(set(test_raw.columns)))
    train_aligned = train_raw.reindex(columns=all_cols)
    test_aligned = test_raw.reindex(columns=all_cols)
    all_df = pd.concat([train_aligned, test_aligned], axis=0, ignore_index=True)

    external_meta = load_external_metadata(input_dir, args.metadata_file)
    all_feat, created_features = add_engineered_features(all_df, external_meta)

    train_feat = all_feat[all_feat["__is_train"].eq(1)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)
    test_feat = all_feat[all_feat["__is_train"].eq(0)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)

    # 特徴量作成後、baselineの存在確認
    baseline_features = valid_feature_list(train_feat, test_feat, baseline_features)
    baseline_features = [f for f in baseline_features if is_non_constant_feature(train_feat, f)]

    categorical_all = [c for c in base_categorical_candidates() if c in train_feat.columns and c in test_feat.columns]

    pd.DataFrame(created_features).to_csv(output_dir / "engineered_feature_catalog.csv", index=False)

    # CV folds固定: 全ABテストで完全に同じ分割を使う
    y_full = train_feat[target_col].astype(int)
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(train_feat, y_full))

    # baseline Optuna
    print("\n[STEP] initial Optuna on baseline features")
    X_base, y, X_test_base, cats_base, impute_report_base = prepare_X_y(train_feat, test_feat, baseline_features, target_col, categorical_all)
    initial_params, initial_optuna_summary = tune_lgbm_with_optuna(
        X_base,
        y,
        baseline_features,
        cats_base,
        folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        n_trials=args.optuna_trials,
        study_name="initial_baseline_lgbm",
    )
    write_json(initial_optuna_summary, output_dir / "initial_optuna_summary.json")
    write_json(initial_params, output_dir / "initial_best_params.json")
    write_json(impute_report_base, output_dir / "baseline_impute_report.json")

    # AB feature selection
    selected_after_ab, ab_result, ab_report = run_ab_feature_selection(
        train_feat,
        test_feat,
        target_col,
        baseline_features,
        categorical_all,
        initial_params,
        folds,
        args.seed,
        args.min_delta,
        args.max_gap_increase,
        output_dir,
    )

    with open(output_dir / "selected_features_after_ab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_after_ab))

    # 重要度下位から削除テスト
    selected_after_del, del_result, del_report, imp_after_del = run_backward_elimination(
        train_feat,
        test_feat,
        target_col,
        selected_after_ab,
        categorical_all,
        initial_params,
        folds,
        args.seed,
        args.min_delta,
        args.max_gap_increase,
        args.min_features,
        output_dir,
    )

    with open(output_dir / "selected_features_after_elimination.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_after_del))

    # 最終Optuna
    print("\n[STEP] final Optuna on selected features")
    X_final_pre, y, X_test_final_pre, cats_final, impute_report_final_pre = prepare_X_y(
        train_feat, test_feat, selected_after_del, target_col, categorical_all
    )
    final_params, final_optuna_summary = tune_lgbm_with_optuna(
        X_final_pre,
        y,
        selected_after_del,
        cats_final,
        folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        n_trials=args.final_optuna_trials,
        study_name="final_selected_lgbm",
    )
    write_json(final_params, output_dir / "final_best_params.json")
    write_json(final_optuna_summary, output_dir / "final_optuna_summary.json")

    # 最終CV + OOF + importance
    print("\n[STEP] final CV / OOF / importance")
    X_final, y, X_test_final, cats_final, impute_report_final = prepare_X_y(
        train_feat, test_feat, selected_after_del, target_col, categorical_all
    )
    final_cv = evaluate_cv(
        X_final,
        y,
        selected_after_del,
        cats_final,
        final_params,
        folds,
        args.seed,
        return_oof=True,
        return_importance=True,
    )
    final_importance = final_cv.get("importance", pd.DataFrame())
    final_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    write_json(impute_report_final, output_dir / "final_impute_report.json")

    oof_df = pd.DataFrame({
        "oof_pred": final_cv["oof"],
        target_col: y.values,
    })
    if id_col in train_feat.columns:
        oof_df.insert(0, "id", train_feat[id_col].values)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 最終学習・test予測
    print("\n[STEP] train final model and predict test")
    test_preds = train_final_and_predict(
        X_final,
        y,
        X_test_final,
        selected_after_del,
        cats_final,
        final_params,
        final_cv.get("best_iteration_mean", 0),
        args.seed,
        output_dir,
    )

    submission = pd.DataFrame({"id": test_feat[id_col].values if id_col in test_feat.columns else np.arange(len(test_feat)), target_col: test_preds})
    submission.to_csv(output_dir / "submission.csv", index=False)

    with open(output_dir / "selected_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_after_del))

    # サマリー
    accepted_features = ab_report.loc[ab_report["accepted"].eq(True), "features"].tolist() if len(ab_report) else []
    removed_features = del_report.loc[del_report.get("accepted_removal", pd.Series(False, index=del_report.index)).eq(True), "feature"].tolist() if len(del_report) else []
    summary = {
        "runtime_seconds": float(time.time() - start_time),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "original_path": str(original_path) if original_path else None,
        "original_report": original_report,
        "train_clean_report": train_clean_report,
        "test_clean_report": test_clean_report,
        "target": target_col,
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "target_mean": float(y.mean()),
        "baseline_features_count": int(len(baseline_features)),
        "selected_after_ab_count": int(len(selected_after_ab)),
        "final_features_count": int(len(selected_after_del)),
        "accepted_ab_features": accepted_features,
        "removed_features": removed_features,
        "initial_optuna": initial_optuna_summary,
        "ab_best_valid_auc_mean": float(ab_result["valid_auc_mean"]),
        "after_elimination_valid_auc_mean": float(del_result["valid_auc_mean"]),
        "final_cv_valid_auc_mean": float(final_cv["valid_auc_mean"]),
        "final_cv_valid_auc_std": float(final_cv["valid_auc_std"]),
        "final_cv_train_auc_mean": float(final_cv["train_auc_mean"]),
        "final_cv_overfit_gap_mean": float(final_cv["overfit_gap_mean"]),
        "final_cv_fold_valid_aucs": final_cv["fold_valid_aucs"],
        "final_best_iteration_mean": float(final_cv["best_iteration_mean"]),
        "final_params": final_params,
        "top_30_importance_gain": final_importance.head(30).to_dict(orient="records") if len(final_importance) else [],
        "outputs": {
            "submission": str(output_dir / "submission.csv"),
            "oof_predictions": str(output_dir / "oof_predictions.csv"),
            "feature_selection_report": str(output_dir / "feature_selection_report.csv"),
            "backward_elimination_report": str(output_dir / "backward_elimination_report.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "selected_features": str(output_dir / "selected_features.txt"),
            "final_params": str(output_dir / "final_best_params.json"),
        },
    }
    write_json(summary, output_dir / "run_summary.json")

    print("\n[DONE]")
    print(f"final CV AUC: {final_cv['valid_auc_mean']:.6f} ± {final_cv['valid_auc_std']:.6f}")
    print(f"train-valid gap: {final_cv['overfit_gap_mean']:.6f}")
    print(f"final features: {len(selected_after_del)}")
    print(f"submission saved: {output_dir / 'submission.csv'}")
    print(f"report saved: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
