#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_catboost2000_xgb_ensemble_next.py

Purpose
-------
Use the already-created AB-selected LightGBM outputs, then:
  1) train a stronger CatBoost model with more iterations,
  2) create CatBoost-only submission immediately,
  3) search the best LGBM AB-selected + new CatBoost weight and create ensemble submission,
  4) optionally train a lightweight XGBoost model,
  5) create XGBoost-only submission and LGBM + CatBoost + XGBoost ensemble submission.

Expected project layout
-----------------------
C:\\Projects\\F1PitPrediction
├─ src
│  ├─ f1_pit_lgbm_feature_ab.py
│  └─ f1_catboost2000_xgb_ensemble_next.py
├─ data
│  ├─ train.csv
│  ├─ test.csv
│  └─ f1_strategy_dataset_v4.csv   # optional, but recommended if previous runs used it
└─ runs
   └─ f1_stage2_ensemble
      ├─ selected_features_after_ab.txt
      ├─ ab_selected_oof.csv
      └─ ab_selected_test_pred.csv

Example commands
----------------
CatBoost 2000 + LGBM/Cat ensemble only:
python .\src\f1_catboost2000_xgb_ensemble_next.py `
  --input-dir .\data `
  --stage2-run-dir .\runs\f1_stage2_ensemble `
  --output-dir .\runs\f1_cat2000_next `
  --catboost-iterations 2000

CatBoost 2000 + lightweight XGBoost:
python .\src\f1_catboost2000_xgb_ensemble_next.py `
  --input-dir .\data `
  --stage2-run-dir .\runs\f1_stage2_ensemble `
  --output-dir .\runs\f1_cat2000_xgb_next `
  --catboost-iterations 2000 `
  --run-xgboost `
  --xgboost-n-estimators 700
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = e
else:
    CATBOOST_IMPORT_ERROR = None

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    xgb = None
    XGBOOST_IMPORT_ERROR = e
else:
    XGBOOST_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"selected feature file not found: {path}")
    feats = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not feats:
        raise ValueError(f"No features found in {path}")
    return feats


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_prediction_column(df: pd.DataFrame, target: str, preferred: Sequence[str]) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if c.lower() not in {"id", target.lower()}]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_candidates:
        raise ValueError(f"No prediction column found. columns={df.columns.tolist()}")
    return numeric_candidates[0]


def load_helper(helper_path: Path):
    if not helper_path.exists():
        raise FileNotFoundError(f"helper file not found: {helper_path}")
    spec = importlib.util.spec_from_file_location("f1_helper", str(helper_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper from {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["f1_helper"] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Data reconstruction: use the previous helper so features match earlier runs
# -----------------------------------------------------------------------------


def reconstruct_feature_data(args: argparse.Namespace, helper) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    input_dir = Path(args.input_dir)
    target_col = helper.clean_column_name(args.target)
    id_col = helper.clean_column_name(args.id_col)

    train_path = helper.find_csv(input_dir, "auto", ["train.csv"], required=True)
    test_path = helper.find_csv(input_dir, "auto", ["test.csv"], required=True)
    original_path = helper.find_csv(
        input_dir,
        args.original_file,
        ["f1_strategy_dataset_v4.csv", "original.csv", "f1_strategy_dataset.csv"],
        required=False,
    )

    log(f"[DATA] train    = {train_path}")
    log(f"[DATA] test     = {test_path}")
    log(f"[DATA] original = {original_path if original_path else 'not used'}")

    train_raw, train_mapping = helper.read_csv_clean(train_path)
    test_raw, test_mapping = helper.read_csv_clean(test_path)

    if target_col not in train_raw.columns:
        raise ValueError(f"target column '{target_col}' not found in train. columns={train_raw.columns.tolist()}")
    if id_col not in test_raw.columns:
        log(f"[WARN] test id column '{id_col}' not found. using index as id.")
        test_raw[id_col] = np.arange(len(test_raw))

    original_report: Dict[str, Any] = {"status": "not_used"}
    if original_path is not None:
        original_raw, original_mapping = helper.read_csv_clean(original_path)
        original_clean, original_report = helper.remove_original_overlap(original_raw, train_raw, test_raw, target_col, id_col)
        if len(original_clean) > 0:
            common_for_train = [c for c in train_raw.columns if c in original_clean.columns]
            original_clean = original_clean[common_for_train].copy()
            train_raw = pd.concat([train_raw, original_clean], axis=0, ignore_index=True)
            log(f"[DATA] original added rows = {len(original_clean):,}")
        else:
            log(f"[DATA] original skipped/no usable rows. status={original_report.get('status')}")

    train_raw, train_clean_report = helper.basic_train_clean(train_raw, target_col)
    test_raw, test_clean_report = helper.basic_test_clean_keep_ids(test_raw)

    train_raw["__is_train"] = 1
    test_raw["__is_train"] = 0
    all_cols = sorted(set(train_raw.columns).union(set(test_raw.columns)))
    all_df = pd.concat(
        [train_raw.reindex(columns=all_cols), test_raw.reindex(columns=all_cols)],
        axis=0,
        ignore_index=True,
    )

    external_meta = helper.load_external_metadata(input_dir, args.metadata_file)
    all_feat, created_features = helper.add_engineered_features(all_df, external_meta)
    train_feat = all_feat[all_feat["__is_train"].eq(1)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)
    test_feat = all_feat[all_feat["__is_train"].eq(0)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)

    y = train_feat[target_col].astype(int)
    test_ids = test_feat[id_col] if id_col in test_feat.columns else pd.Series(np.arange(len(test_feat)), name="id")

    info = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "original_path": str(original_path) if original_path else None,
        "original_report": original_report,
        "train_clean_report": train_clean_report,
        "test_clean_report": test_clean_report,
        "train_mapping": train_mapping,
        "test_mapping": test_mapping,
        "target": target_col,
        "id_col": id_col,
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "target_mean": float(y.mean()),
        "created_features_count": int(len(created_features)),
    }
    return train_feat, test_feat, y, test_ids, info


# -----------------------------------------------------------------------------
# Prepare matrices for CatBoost/XGBoost
# -----------------------------------------------------------------------------


def prepare_catboost_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    features = list(features)
    missing = [f for f in features if f not in train_df.columns or f not in test_df.columns]
    if missing:
        raise ValueError(f"Selected features missing from data: {missing}")

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    cat_features = [c for c in categorical_features if c in features]
    num_features = [c for c in features if c not in cat_features]

    impute: Dict[str, Any] = {"categorical_fill": "__MISSING__", "numeric_medians": {}, "categorical_features": cat_features}

    for c in cat_features:
        X_train[c] = X_train[c].astype(str).where(~X_train[c].isna(), "__MISSING__")
        X_test[c] = X_test[c].astype(str).where(~X_test[c].isna(), "__MISSING__")

    for c in num_features:
        X_train[c] = safe_to_numeric(X_train[c])
        X_test[c] = safe_to_numeric(X_test[c])
        med = X_train[c].median()
        if pd.isna(med):
            med = 0.0
        impute["numeric_medians"][c] = float(med)
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan).fillna(med)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan).fillna(med)

    return X_train, X_test, cat_features, impute


def prepare_xgboost_data(
    X_train_cat: pd.DataFrame,
    X_test_cat: pd.DataFrame,
    categorical_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # For reliability across XGBoost versions, one-hot encode categorical columns.
    combined = pd.concat([X_train_cat, X_test_cat], axis=0, ignore_index=True)
    cat_features = [c for c in categorical_features if c in combined.columns]
    combined = pd.get_dummies(combined, columns=cat_features, dummy_na=False, dtype=np.float32)
    combined = combined.astype(np.float32)
    X_train = combined.iloc[: len(X_train_cat)].reset_index(drop=True)
    X_test = combined.iloc[len(X_train_cat) :].reset_index(drop=True)
    report = {
        "encoding": "pd.get_dummies",
        "categorical_features": list(cat_features),
        "n_features_after_encoding": int(X_train.shape[1]),
        "encoded_columns": X_train.columns.tolist(),
    }
    return X_train, X_test, report


# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------


def catboost_default_params(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(args.catboost_iterations),
        "learning_rate": float(args.catboost_learning_rate),
        "depth": int(args.catboost_depth),
        "l2_leaf_reg": float(args.catboost_l2_leaf_reg),
        "random_strength": float(args.catboost_random_strength),
        "bootstrap_type": "Bernoulli",
        "subsample": float(args.catboost_subsample),
        "rsm": float(args.catboost_rsm),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": int(args.n_jobs),
    }


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_features: Sequence[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    test_ids: pd.Series,
    output_dir: Path,
    args: argparse.Namespace,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if CatBoostClassifier is None or Pool is None:
        raise ImportError(f"catboost is not installed/importable: {CATBOOST_IMPORT_ERROR}")

    model_dir = output_dir / "models" / tag
    ensure_dir(model_dir)

    cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
    test_pool = Pool(X_test, cat_features=cat_idx)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    fold_rows: List[Dict[str, Any]] = []
    start = time.time()
    log(f"[CAT:{tag}] start | rows={len(X):,} test={len(X_test):,} features={X.shape[1]} cats={len(cat_idx)} folds={len(folds)} iterations={args.catboost_iterations}")

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_start = time.time()
        params = catboost_default_params(args, args.seed + fold)
        model = CatBoostClassifier(**params)

        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)

        log(f"[CAT:{tag}][fold {fold}/{len(folds)}] train={len(tr_idx):,} valid={len(va_idx):,} start")
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=int(args.catboost_early_stopping),
            verbose=False,
        )

        p_va = model.predict_proba(valid_pool)[:, 1]
        p_tr = model.predict_proba(train_pool)[:, 1]
        p_te = model.predict_proba(test_pool)[:, 1]
        oof[va_idx] = p_va
        test_pred += p_te / len(folds)

        valid_auc = float(roc_auc_score(y.iloc[va_idx], p_va))
        train_auc = float(roc_auc_score(y.iloc[tr_idx], p_tr))
        gap = train_auc - valid_auc
        best_iter = int(model.get_best_iteration() or args.catboost_iterations)
        fold_time = time.time() - fold_start
        elapsed = time.time() - start
        eta = elapsed / fold * (len(folds) - fold)
        log(
            f"[CAT:{tag}][fold {fold}/{len(folds)}] done | valid_auc={valid_auc:.6f} "
            f"train_auc={train_auc:.6f} gap={gap:.6f} best_iter={best_iter} "
            f"fold_time={fold_time/60:.1f}m eta={eta/60:.1f}m"
        )

        model_path = model_dir / f"catboost_{tag}_fold{fold}.cbm"
        model.save_model(str(model_path))
        fold_rows.append(
            {
                "fold": fold,
                "valid_auc": valid_auc,
                "train_auc": train_auc,
                "gap": gap,
                "best_iteration": best_iter,
                "fold_time_seconds": fold_time,
                "model_path": str(model_path),
            }
        )

    valid_auc = float(roc_auc_score(y, oof))
    fold_df = pd.DataFrame(fold_rows)
    summary: Dict[str, Any] = {
        "tag": tag,
        "model": "CatBoostClassifier",
        "oof_auc": valid_auc,
        "fold_valid_auc_mean": float(fold_df["valid_auc"].mean()),
        "fold_valid_auc_std": float(fold_df["valid_auc"].std(ddof=0)),
        "fold_train_auc_mean": float(fold_df["train_auc"].mean()),
        "fold_gap_mean": float(fold_df["gap"].mean()),
        "best_iteration_mean": float(fold_df["best_iteration"].mean()),
        "runtime_seconds": float(time.time() - start),
        "params": catboost_default_params(args, args.seed),
        "folds": fold_rows,
    }
    fold_df.to_csv(output_dir / f"{tag}_fold_report.csv", index=False)
    write_json(summary, output_dir / f"{tag}_cv_summary.json")

    pd.DataFrame({"id": test_ids.values, "pred": test_pred}).to_csv(output_dir / f"{tag}_test_pred.csv", index=False)
    pd.DataFrame({"oof_pred": oof, args.target: y.values}).to_csv(output_dir / f"{tag}_oof.csv", index=False)
    pd.DataFrame({"id": test_ids.values, args.target: test_pred}).to_csv(output_dir / f"submission_010_{tag}.csv", index=False)
    log(f"[CAT:{tag}] finished | oof_auc={valid_auc:.6f} | submission=submission_010_{tag}.csv")
    return oof, test_pred, summary


def xgb_default_params(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": float(args.xgboost_learning_rate),
        "max_depth": int(args.xgboost_max_depth),
        "min_child_weight": float(args.xgboost_min_child_weight),
        "subsample": float(args.xgboost_subsample),
        "colsample_bytree": float(args.xgboost_colsample_bytree),
        "reg_alpha": float(args.xgboost_reg_alpha),
        "reg_lambda": float(args.xgboost_reg_lambda),
        "seed": int(seed),
        "nthread": int(args.n_jobs),
    }


def train_xgboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    test_ids: pd.Series,
    output_dir: Path,
    args: argparse.Namespace,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if xgb is None:
        raise ImportError(f"xgboost is not installed/importable: {XGBOOST_IMPORT_ERROR}")

    model_dir = output_dir / "models" / tag
    ensure_dir(model_dir)
    dtest = xgb.DMatrix(X_test)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []
    start = time.time()
    log(f"[XGB:{tag}] start | rows={len(X):,} test={len(X_test):,} encoded_features={X.shape[1]} folds={len(folds)} rounds={args.xgboost_n_estimators}")

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_start = time.time()
        params = xgb_default_params(args, args.seed + 1000 + fold)
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], label=y.iloc[va_idx])
        watchlist = [(dtrain, "train"), (dvalid, "valid")]
        log(f"[XGB:{tag}][fold {fold}/{len(folds)}] train={len(tr_idx):,} valid={len(va_idx):,} start")
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.xgboost_n_estimators),
            evals=watchlist,
            early_stopping_rounds=int(args.xgboost_early_stopping),
            verbose_eval=False,
        )
        p_va = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        p_tr = booster.predict(dtrain, iteration_range=(0, booster.best_iteration + 1))
        p_te = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
        oof[va_idx] = p_va
        test_pred += p_te / len(folds)
        valid_auc = float(roc_auc_score(y.iloc[va_idx], p_va))
        train_auc = float(roc_auc_score(y.iloc[tr_idx], p_tr))
        gap = train_auc - valid_auc
        fold_time = time.time() - fold_start
        elapsed = time.time() - start
        eta = elapsed / fold * (len(folds) - fold)
        model_path = model_dir / f"xgboost_{tag}_fold{fold}.json"
        booster.save_model(str(model_path))
        log(
            f"[XGB:{tag}][fold {fold}/{len(folds)}] done | valid_auc={valid_auc:.6f} "
            f"train_auc={train_auc:.6f} gap={gap:.6f} best_iter={booster.best_iteration} "
            f"fold_time={fold_time/60:.1f}m eta={eta/60:.1f}m"
        )
        fold_rows.append(
            {
                "fold": fold,
                "valid_auc": valid_auc,
                "train_auc": train_auc,
                "gap": gap,
                "best_iteration": int(booster.best_iteration),
                "fold_time_seconds": fold_time,
                "model_path": str(model_path),
            }
        )

    valid_auc = float(roc_auc_score(y, oof))
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "tag": tag,
        "model": "XGBoost train",
        "oof_auc": valid_auc,
        "fold_valid_auc_mean": float(fold_df["valid_auc"].mean()),
        "fold_valid_auc_std": float(fold_df["valid_auc"].std(ddof=0)),
        "fold_train_auc_mean": float(fold_df["train_auc"].mean()),
        "fold_gap_mean": float(fold_df["gap"].mean()),
        "best_iteration_mean": float(fold_df["best_iteration"].mean()),
        "runtime_seconds": float(time.time() - start),
        "params": xgb_default_params(args, args.seed),
        "folds": fold_rows,
    }
    fold_df.to_csv(output_dir / f"{tag}_fold_report.csv", index=False)
    write_json(summary, output_dir / f"{tag}_cv_summary.json")
    pd.DataFrame({"id": test_ids.values, "pred": test_pred}).to_csv(output_dir / f"{tag}_test_pred.csv", index=False)
    pd.DataFrame({"oof_pred": oof, args.target: y.values}).to_csv(output_dir / f"{tag}_oof.csv", index=False)
    pd.DataFrame({"id": test_ids.values, args.target: test_pred}).to_csv(output_dir / f"submission_012_{tag}.csv", index=False)
    log(f"[XGB:{tag}] finished | oof_auc={valid_auc:.6f} | submission=submission_012_{tag}.csv")
    return oof, test_pred, summary


# -----------------------------------------------------------------------------
# Ensemble utilities
# -----------------------------------------------------------------------------


def save_ensemble_submission(
    test_ids: pd.Series,
    y: pd.Series,
    output_dir: Path,
    target: str,
    tag: str,
    oof: np.ndarray,
    test_pred: np.ndarray,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    auc = float(roc_auc_score(y, oof))
    pd.DataFrame({"oof_pred": oof, target: y.values}).to_csv(output_dir / f"{tag}_oof.csv", index=False)
    pd.DataFrame({"id": test_ids.values, "pred": test_pred}).to_csv(output_dir / f"{tag}_test_pred.csv", index=False)
    sub_path = output_dir / f"submission_{tag}.csv"
    pd.DataFrame({"id": test_ids.values, target: test_pred}).to_csv(sub_path, index=False)
    summary = {"tag": tag, "oof_auc": auc, "weights": weights, "submission": str(sub_path)}
    write_json(summary, output_dir / f"{tag}_summary.json")
    log(f"[ENS:{tag}] saved | oof_auc={auc:.6f} | submission={sub_path.name} | weights={weights}")
    return summary


def grid_search_two_model_weights(
    y: pd.Series,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    grid_size: int,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    rows = []
    best_auc = -np.inf
    best_w = 0.5
    grid = np.linspace(0, 1, int(grid_size))
    for i, w_a in enumerate(grid):
        pred = w_a * pred_a + (1 - w_a) * pred_b
        auc = float(roc_auc_score(y, pred))
        rows.append({f"weight_{name_a}": float(w_a), f"weight_{name_b}": float(1 - w_a), "oof_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_w = float(w_a)
        if (i + 1) % max(1, len(grid) // 10) == 0:
            log(f"[WEIGHT2] {i+1}/{len(grid)} | best_auc={best_auc:.6f} | {name_a}={best_w:.4f} {name_b}={1-best_w:.4f}")
    return {name_a: best_w, name_b: 1 - best_w}, best_auc, pd.DataFrame(rows)


def random_search_three_model_weights(
    y: pd.Series,
    preds: Dict[str, np.ndarray],
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    names = list(preds.keys())
    rng = np.random.default_rng(seed)
    rows = []
    best_auc = -np.inf
    best_weights = {n: 1 / len(names) for n in names}

    # include useful deterministic candidates
    candidates: List[np.ndarray] = []
    candidates.append(np.ones(len(names)) / len(names))
    for i in range(len(names)):
        w = np.zeros(len(names)); w[i] = 1.0; candidates.append(w)
    candidates.extend([
        np.array([0.40, 0.50, 0.10]),
        np.array([0.35, 0.55, 0.10]),
        np.array([0.45, 0.45, 0.10]),
        np.array([0.30, 0.60, 0.10]),
        np.array([0.40, 0.40, 0.20]),
        np.array([0.25, 0.55, 0.20]),
    ])
    for _ in range(max(0, n_trials - len(candidates))):
        candidates.append(rng.dirichlet(np.ones(len(names))))

    for i, w in enumerate(candidates, start=1):
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        pred = sum(float(w[j]) * preds[names[j]] for j in range(len(names)))
        auc = float(roc_auc_score(y, pred))
        row = {f"weight_{names[j]}": float(w[j]) for j in range(len(names))}
        row["oof_auc"] = auc
        rows.append(row)
        if auc > best_auc:
            best_auc = auc
            best_weights = {names[j]: float(w[j]) for j in range(len(names))}
        if i % max(1, len(candidates) // 10) == 0:
            log(f"[WEIGHT3] {i}/{len(candidates)} | best_auc={best_auc:.6f} | weights={best_weights}")
    return best_weights, best_auc, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-step ensemble: stronger CatBoost + optional lightweight XGBoost")
    p.add_argument("--input-dir", type=str, default=".\\data")
    p.add_argument("--stage2-run-dir", type=str, default=".\\runs\\f1_stage2_ensemble")
    p.add_argument("--output-dir", type=str, default=".\\runs\\f1_cat2000_next")
    p.add_argument("--helper-file", type=str, default="auto")
    p.add_argument("--original-file", type=str, default="auto")
    p.add_argument("--metadata-file", type=str, default="auto")
    p.add_argument("--target", type=str, default="PitNextLap")
    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)

    # CatBoost stronger version
    p.add_argument("--catboost-iterations", type=int, default=2000)
    p.add_argument("--catboost-learning-rate", type=float, default=0.03)
    p.add_argument("--catboost-depth", type=int, default=8)
    p.add_argument("--catboost-l2-leaf-reg", type=float, default=6.0)
    p.add_argument("--catboost-random-strength", type=float, default=1.0)
    p.add_argument("--catboost-subsample", type=float, default=0.85)
    p.add_argument("--catboost-rsm", type=float, default=0.85)
    p.add_argument("--catboost-early-stopping", type=int, default=250)

    # Ensemble
    p.add_argument("--weight-grid-size", type=int, default=1001)

    # XGBoost optional
    p.add_argument("--run-xgboost", action="store_true")
    p.add_argument("--xgboost-n-estimators", type=int, default=700)
    p.add_argument("--xgboost-learning-rate", type=float, default=0.035)
    p.add_argument("--xgboost-max-depth", type=int, default=6)
    p.add_argument("--xgboost-min-child-weight", type=float, default=20.0)
    p.add_argument("--xgboost-subsample", type=float, default=0.85)
    p.add_argument("--xgboost-colsample-bytree", type=float, default=0.75)
    p.add_argument("--xgboost-reg-alpha", type=float, default=0.10)
    p.add_argument("--xgboost-reg-lambda", type=float, default=5.0)
    p.add_argument("--xgboost-early-stopping", type=int, default=80)
    p.add_argument("--three-model-weight-trials", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    stage2_run_dir = Path(args.stage2_run_dir)
    ensure_dir(output_dir)

    log("=" * 80)
    log("F1 Pit Prediction: CatBoost2000 + optional XGBoost next-step ensemble")
    log("Each major step saves submission CSV immediately.")
    log("=" * 80)
    log(f"[CONFIG] input_dir        = {args.input_dir}")
    log(f"[CONFIG] stage2_run_dir   = {stage2_run_dir}")
    log(f"[CONFIG] output_dir       = {output_dir}")
    log(f"[CONFIG] catboost_iter    = {args.catboost_iterations}")
    log(f"[CONFIG] run_xgboost      = {args.run_xgboost}")

    # Locate helper next to this script if auto
    if args.helper_file.lower() == "auto":
        helper_path = Path(__file__).resolve().parent / "f1_pit_lgbm_feature_ab.py"
    else:
        helper_path = Path(args.helper_file)
    helper = load_helper(helper_path)
    log(f"[INFO] helper loaded = {helper_path}")

    # Load selected features and existing LGBM predictions
    selected_features_path = stage2_run_dir / "selected_features_after_ab.txt"
    selected_features = read_feature_list(selected_features_path)
    log(f"[INFO] selected AB features loaded = {len(selected_features)}")

    lgbm_oof_path = stage2_run_dir / "ab_selected_oof.csv"
    lgbm_test_path = stage2_run_dir / "ab_selected_test_pred.csv"
    if not lgbm_oof_path.exists() or not lgbm_test_path.exists():
        raise FileNotFoundError("Need ab_selected_oof.csv and ab_selected_test_pred.csv in stage2-run-dir")
    lgbm_oof_df = pd.read_csv(lgbm_oof_path)
    lgbm_test_df = pd.read_csv(lgbm_test_path)
    lgbm_oof_col = find_prediction_column(lgbm_oof_df, args.target, ["oof_pred", "pred", args.target])
    lgbm_test_col = find_prediction_column(lgbm_test_df, args.target, ["pred", "test_pred", args.target])
    lgbm_oof = lgbm_oof_df[lgbm_oof_col].to_numpy(dtype=float)
    lgbm_test = lgbm_test_df[lgbm_test_col].to_numpy(dtype=float)
    log(f"[LOAD] LGBM OOF  = {lgbm_oof_path} col={lgbm_oof_col}")
    log(f"[LOAD] LGBM test = {lgbm_test_path} col={lgbm_test_col}")

    # Reconstruct full feature data
    train_feat, test_feat, y, test_ids, data_info = reconstruct_feature_data(args, helper)
    if len(lgbm_oof) != len(train_feat):
        raise ValueError(f"LGBM OOF length mismatch: {len(lgbm_oof)} vs train rows {len(train_feat)}")
    if len(lgbm_test) != len(test_feat):
        raise ValueError(f"LGBM test length mismatch: {len(lgbm_test)} vs test rows {len(test_feat)}")

    lgbm_auc = float(roc_auc_score(y, lgbm_oof))
    log(f"[LGBM] AB-selected OOF AUC = {lgbm_auc:.6f}")

    # CV folds matching previous seed/n_folds
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(train_feat, y))

    categorical_all = [c for c in helper.base_categorical_candidates() if c in selected_features]
    X_cat, X_test_cat, cat_features, impute_report = prepare_catboost_data(train_feat, test_feat, selected_features, categorical_all)
    write_json(data_info, output_dir / "data_reconstruction_info.json")
    write_json(impute_report, output_dir / "catboost2000_impute_report.json")
    with open(output_dir / "selected_features_used.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    # ------------------------------------------------------------------
    # Step 2: stronger CatBoost
    # ------------------------------------------------------------------
    cat_tag = f"catboost_iter{args.catboost_iterations}"
    cat_oof, cat_test, cat_summary = train_catboost_cv(
        X=X_cat,
        y=y,
        X_test=X_test_cat,
        cat_features=cat_features,
        folds=folds,
        test_ids=test_ids,
        output_dir=output_dir,
        args=args,
        tag=cat_tag,
    )

    # ------------------------------------------------------------------
    # Step 3: LGBM + stronger CatBoost ensemble
    # ------------------------------------------------------------------
    log("[STEP] LGBM AB-selected + stronger CatBoost weight search")
    best_w2, best_auc2, report2 = grid_search_two_model_weights(
        y=y,
        pred_a=lgbm_oof,
        pred_b=cat_oof,
        name_a="lgbm_ab",
        name_b=cat_tag,
        grid_size=args.weight_grid_size,
    )
    report2.to_csv(output_dir / "lgbm_catboost2000_weight_search_report.csv", index=False)
    write_json({"best_weights": best_w2, "best_oof_auc": best_auc2}, output_dir / "best_lgbm_catboost2000_weights.json")
    ens2_oof = best_w2["lgbm_ab"] * lgbm_oof + best_w2[cat_tag] * cat_oof
    ens2_test = best_w2["lgbm_ab"] * lgbm_test + best_w2[cat_tag] * cat_test
    ens2_summary = save_ensemble_submission(
        test_ids=test_ids,
        y=y,
        output_dir=output_dir,
        target=args.target,
        tag="011_lgbm_catboost2000_ensemble",
        oof=ens2_oof,
        test_pred=ens2_test,
        weights=best_w2,
    )

    xgb_summary = None
    ens3_summary = None

    # ------------------------------------------------------------------
    # Step 4: optional lightweight XGBoost
    # ------------------------------------------------------------------
    if args.run_xgboost:
        log("[STEP] optional lightweight XGBoost")
        X_xgb, X_test_xgb, xgb_encoding_report = prepare_xgboost_data(X_cat, X_test_cat, cat_features)
        write_json(xgb_encoding_report, output_dir / "xgboost_onehot_encoding_report.json")
        xgb_tag = f"xgboost_light{args.xgboost_n_estimators}"
        xgb_oof, xgb_test, xgb_summary = train_xgboost_cv(
            X=X_xgb,
            y=y,
            X_test=X_test_xgb,
            folds=folds,
            test_ids=test_ids,
            output_dir=output_dir,
            args=args,
            tag=xgb_tag,
        )

        # 3-model ensemble
        log("[STEP] LGBM + CatBoost2000 + XGBoost lightweight weight search")
        pred_dict = {
            "lgbm_ab": lgbm_oof,
            cat_tag: cat_oof,
            xgb_tag: xgb_oof,
        }
        best_w3, best_auc3, report3 = random_search_three_model_weights(
            y=y,
            preds=pred_dict,
            n_trials=args.three_model_weight_trials,
            seed=args.seed,
        )
        report3.to_csv(output_dir / "lgbm_catboost2000_xgboost_weight_search_report.csv", index=False)
        write_json({"best_weights": best_w3, "best_oof_auc": best_auc3}, output_dir / "best_lgbm_catboost2000_xgboost_weights.json")
        ens3_oof = best_w3["lgbm_ab"] * lgbm_oof + best_w3[cat_tag] * cat_oof + best_w3[xgb_tag] * xgb_oof
        ens3_test = best_w3["lgbm_ab"] * lgbm_test + best_w3[cat_tag] * cat_test + best_w3[xgb_tag] * xgb_test
        ens3_summary = save_ensemble_submission(
            test_ids=test_ids,
            y=y,
            output_dir=output_dir,
            target=args.target,
            tag="013_lgbm_catboost2000_xgboost_ensemble",
            oof=ens3_oof,
            test_pred=ens3_test,
            weights=best_w3,
        )
    else:
        log("[SKIP] XGBoost skipped. Add --run-xgboost if time remains.")

    # Score/correlation report
    models_for_corr = {
        "lgbm_ab": lgbm_oof,
        cat_tag: cat_oof,
        "lgbm_catboost2000_ensemble": ens2_oof,
    }
    if args.run_xgboost and xgb_summary is not None:
        xgb_tag = f"xgboost_light{args.xgboost_n_estimators}"
        models_for_corr[xgb_tag] = pd.read_csv(output_dir / f"{xgb_tag}_oof.csv")["oof_pred"].to_numpy(dtype=float)
        if ens3_summary is not None:
            models_for_corr["lgbm_catboost2000_xgboost_ensemble"] = pd.read_csv(output_dir / "013_lgbm_catboost2000_xgboost_ensemble_oof.csv")["oof_pred"].to_numpy(dtype=float)

    corr_df = pd.DataFrame(models_for_corr).corr()
    corr_df.to_csv(output_dir / "prediction_correlation_matrix.csv")
    score_rows = [{"model": name, "oof_auc": float(roc_auc_score(y, pred))} for name, pred in models_for_corr.items()]
    score_df = pd.DataFrame(score_rows).sort_values("oof_auc", ascending=False)
    score_df.to_csv(output_dir / "model_oof_score_report.csv", index=False)

    summary = {
        "runtime_seconds": float(time.time() - start),
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "catboost_available": CatBoostClassifier is not None,
            "xgboost_available": xgb is not None,
        },
        "lgbm_ab_oof_auc": lgbm_auc,
        "catboost_summary": cat_summary,
        "lgbm_catboost2000_ensemble_summary": ens2_summary,
        "xgboost_summary": xgb_summary,
        "three_model_ensemble_summary": ens3_summary,
        "recommended_submission_order": [
            "submission_011_lgbm_catboost2000_ensemble.csv",
            f"submission_010_{cat_tag}.csv",
            "submission_013_lgbm_catboost2000_xgboost_ensemble.csv" if args.run_xgboost else None,
            f"submission_012_xgboost_light{args.xgboost_n_estimators}.csv" if args.run_xgboost else None,
        ],
    }
    write_json(summary, output_dir / "next_step_run_summary.json")

    log("=" * 80)
    log("DONE")
    log(f"[OUT] {output_dir}")
    log("[KEY SUBMISSIONS]")
    log(f"  1) {output_dir / 'submission_011_lgbm_catboost2000_ensemble.csv'}")
    log(f"  2) {output_dir / f'submission_010_{cat_tag}.csv'}")
    if args.run_xgboost:
        log(f"  3) {output_dir / 'submission_013_lgbm_catboost2000_xgboost_ensemble.csv'}")
        log(f"  4) {output_dir / f'submission_012_xgboost_light{args.xgboost_n_estimators}.csv'}")
    log("=" * 80)


if __name__ == "__main__":
    main()
