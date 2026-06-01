#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_xgboost_three_model_ensemble.py

LightGBM AB-selected + CatBoost + lightweight XGBoost の3モデルアンサンブル用スクリプト。

狙い:
- 既存の LGBM AB-selected 予測と CatBoost 予測を再利用する
- 新しく lightweight XGBoost だけ学習する
- XGBoost単体submissionをすぐ保存する
- LGBM + CatBoost + XGBoost のOOF最適weightを探索し、ensemble submissionを保存する

想定フォルダ:
C:\Projects\F1PitPrediction
├─ src
│  ├─ f1_pit_lgbm_feature_ab.py
│  └─ f1_xgboost_three_model_ensemble.py
├─ data
│  ├─ train.csv
│  ├─ test.csv
│  └─ f1_strategy_dataset_v4.csv
└─ runs
   ├─ f1_stage2_ensemble
   │  ├─ selected_features_after_ab.txt
   │  ├─ ab_selected_oof.csv
   │  └─ ab_selected_test_pred.csv
   └─ f1_lgbm_catboost_ensemble
      ├─ catboost_oof.csv
      └─ catboost_test_pred.csv

実行例:
python .\src\f1_xgboost_three_model_ensemble.py `
  --input-dir .\data `
  --stage2-run-dir .\runs\f1_stage2_ensemble `
  --catboost-run-dir .\runs\f1_lgbm_catboost_ensemble `
  --output-dir .\runs\f1_xgb_three_model `
  --xgboost-n-estimators 700

必要:
python -m pip install pandas numpy scikit-learn xgboost
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    xgb = None
    XGBOOST_IMPORT_ERROR = e
else:
    XGBOOST_IMPORT_ERROR = None


# =============================================================================
# Utilities
# =============================================================================


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"selected feature file not found: {path}")
    feats = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not feats:
        raise ValueError(f"No features found in {path}")
    return feats


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


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clip_pred(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


# =============================================================================
# Robust prediction file loading
# =============================================================================


def _is_probability_like(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() < 0.95:
        return False
    mn = float(x.min())
    mx = float(x.max())
    # row_indexなどを間違って読まないため、0〜1範囲を必須にする
    return mn >= -1e-9 and mx <= 1.0 + 1e-9 and x.nunique(dropna=True) > 10


def find_prediction_col(df: pd.DataFrame, target: str, preferred: Sequence[str], label: str) -> str:
    cols = list(df.columns)
    lower_to_col = {c.lower(): c for c in cols}

    for c in preferred:
        if c in df.columns and _is_probability_like(df[c]):
            return c
        if c.lower() in lower_to_col and _is_probability_like(df[lower_to_col[c.lower()]]):
            return lower_to_col[c.lower()]

    # 名前にpred/prob/proba/oof/predictionを含み、確率っぽい列を優先
    name_candidates: List[str] = []
    for c in cols:
        low = c.lower()
        if low in {"id", "row_index", "index", target.lower()}:
            continue
        if any(k in low for k in ["pred", "prob", "proba", "prediction", "oof"]):
            if _is_probability_like(df[c]):
                name_candidates.append(c)
    if len(name_candidates) == 1:
        return name_candidates[0]
    if len(name_candidates) > 1:
        # よくある列名をさらに優先
        for c in ["oof_pred", "pred", "prediction", "test_pred", "prob", "proba"]:
            if c in name_candidates:
                return c
        return name_candidates[0]

    # submission/test predictionの場合だけ、target名の確率列を許可
    if target in cols and _is_probability_like(df[target]):
        return target

    prob_cols = [c for c in cols if c.lower() not in {"id", "row_index", "index", target.lower()} and _is_probability_like(df[c])]
    if len(prob_cols) == 1:
        return prob_cols[0]

    raise ValueError(
        f"Could not infer prediction column for {label}. "
        f"Columns={cols}. Probability-like columns={prob_cols}"
    )


def find_target_col(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    binary_cols: List[str] = []
    for c in df.columns:
        low = c.lower()
        if low in {"id", "row_index", "index"}:
            continue
        if any(k in low for k in ["pred", "prob", "proba", "prediction", "oof"]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        vals = set(x.dropna().unique().tolist())
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(c)
    if len(binary_cols) == 1:
        return binary_cols[0]
    raise ValueError(f"Could not infer target column. Expected '{target}'. Columns={list(df.columns)}")


def load_oof_prediction(path: Path, target: str, label: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if not path.exists():
        raise FileNotFoundError(f"OOF file not found for {label}: {path}")
    df = pd.read_csv(path)
    pred_col = find_prediction_col(
        df,
        target=target,
        preferred=["oof_pred", "pred", "prediction", "prob", "proba", f"{target}_pred"],
        label=f"{label} OOF",
    )
    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    y = None
    try:
        y_col = find_target_col(df, target)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=int)
    except Exception:
        y = None
    log(f"[LOAD] {label} OOF  = {path} | pred_col={pred_col}")
    return pred, y, pred_col


def load_test_prediction(path: Path, target: str, label: str, id_col: str = "id") -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if not path.exists():
        raise FileNotFoundError(f"test prediction file not found for {label}: {path}")
    df = pd.read_csv(path)
    pred_col = find_prediction_col(
        df,
        target=target,
        preferred=["pred", "test_pred", "prediction", "prob", "proba", target, f"{target}_pred"],
        label=f"{label} test",
    )
    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    ids = df[id_col].to_numpy() if id_col in df.columns else None
    log(f"[LOAD] {label} test = {path} | pred_col={pred_col}")
    return pred, ids, pred_col


def validate_existing_model_auc(y: pd.Series, pred: np.ndarray, model_name: str, min_auc: float) -> float:
    auc = float(roc_auc_score(y, pred))
    log(f"[CHECK] {model_name} OOF AUC = {auc:.6f}")
    if auc < min_auc:
        raise ValueError(
            f"{model_name} OOF AUC is suspiciously low ({auc:.6f}). "
            f"This usually means the wrong prediction column was loaded. "
            f"Minimum expected AUC is {min_auc}."
        )
    return auc


# =============================================================================
# Data reconstruction using existing helper
# =============================================================================


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
        original_raw, _ = helper.read_csv_clean(original_path)
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


# =============================================================================
# XGBoost matrix prep
# =============================================================================


def prepare_xgboost_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    features = list(features)
    missing = [f for f in features if f not in train_df.columns or f not in test_df.columns]
    if missing:
        raise ValueError(f"Selected features missing from reconstructed data: {missing}")

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    cat_features = [c for c in categorical_features if c in features]
    num_features = [c for c in features if c not in cat_features]

    impute: Dict[str, Any] = {
        "numeric_medians": {},
        "categorical_fill": "__MISSING__",
        "categorical_features": cat_features,
    }

    for c in num_features:
        X_train[c] = safe_num(X_train[c])
        X_test[c] = safe_num(X_test[c])
        med = X_train[c].median()
        if pd.isna(med):
            med = 0.0
        impute["numeric_medians"][c] = float(med)
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan).fillna(med)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan).fillna(med)

    for c in cat_features:
        X_train[c] = X_train[c].astype(str).where(~X_train[c].isna(), "__MISSING__")
        X_test[c] = X_test[c].astype(str).where(~X_test[c].isna(), "__MISSING__")

    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_features, dummy_na=False, dtype=np.float32)
    combined = combined.astype(np.float32)
    X_train_enc = combined.iloc[: len(X_train)].reset_index(drop=True)
    X_test_enc = combined.iloc[len(X_train) :].reset_index(drop=True)

    report = {
        "encoding": "pd.get_dummies",
        "n_raw_features": len(features),
        "categorical_features": cat_features,
        "numeric_features": num_features,
        "n_encoded_features": int(X_train_enc.shape[1]),
        "encoded_columns": X_train_enc.columns.tolist(),
        "impute": impute,
    }
    return X_train_enc, X_test_enc, report


# =============================================================================
# XGBoost training
# =============================================================================


def xgb_params(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
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
        "max_bin": int(args.xgboost_max_bin),
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
        params = xgb_params(args, args.seed + 1000 + fold)
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], label=y.iloc[va_idx])

        log(f"[XGB:{tag}][fold {fold}/{len(folds)}] train={len(tr_idx):,} valid={len(va_idx):,} start")
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.xgboost_n_estimators),
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=int(args.xgboost_early_stopping),
            verbose_eval=False,
        )

        best_iter = int(getattr(booster, "best_iteration", args.xgboost_n_estimators - 1))
        iteration_range = (0, best_iter + 1)
        p_va = booster.predict(dvalid, iteration_range=iteration_range)
        p_tr = booster.predict(dtrain, iteration_range=iteration_range)
        p_te = booster.predict(dtest, iteration_range=iteration_range)
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
            f"train_auc={train_auc:.6f} gap={gap:.6f} best_iter={best_iter} "
            f"fold_time={fold_time/60:.1f}m eta={eta/60:.1f}m"
        )
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

    auc = float(roc_auc_score(y, oof))
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "tag": tag,
        "model": "XGBoost hist",
        "oof_auc": auc,
        "fold_valid_auc_mean": float(fold_df["valid_auc"].mean()),
        "fold_valid_auc_std": float(fold_df["valid_auc"].std(ddof=0)),
        "fold_train_auc_mean": float(fold_df["train_auc"].mean()),
        "fold_gap_mean": float(fold_df["gap"].mean()),
        "best_iteration_mean": float(fold_df["best_iteration"].mean()),
        "runtime_seconds": float(time.time() - start),
        "params": xgb_params(args, args.seed),
        "folds": fold_rows,
    }
    fold_df.to_csv(output_dir / f"{tag}_fold_report.csv", index=False)
    write_json(summary, output_dir / f"{tag}_cv_summary.json")
    pd.DataFrame({"oof_pred": oof, args.target: y.values}).to_csv(output_dir / f"{tag}_oof.csv", index=False)
    pd.DataFrame({"id": test_ids.values, "pred": test_pred}).to_csv(output_dir / f"{tag}_test_pred.csv", index=False)
    pd.DataFrame({"id": test_ids.values, args.target: clip_pred(test_pred, args.pred_clip_eps)}).to_csv(output_dir / f"submission_020_{tag}.csv", index=False)
    log(f"[XGB:{tag}] finished | oof_auc={auc:.6f} | submission=submission_020_{tag}.csv")
    return oof, test_pred, summary


# =============================================================================
# Ensemble
# =============================================================================


def grid_search_three_weights(
    y: pd.Series,
    preds: Dict[str, np.ndarray],
    step: float,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    names = list(preds.keys())
    if len(names) != 3:
        raise ValueError("This function expects exactly 3 models.")
    n = int(round(1.0 / step))
    rows: List[Dict[str, Any]] = []
    best_auc = -np.inf
    best_weights: Dict[str, float] = {}

    # 0.01 stepなら約5151通り。かなり軽い。
    for i in range(n + 1):
        w0 = i / n
        for j in range(n - i + 1):
            w1 = j / n
            w2 = 1.0 - w0 - w1
            weights = {names[0]: w0, names[1]: w1, names[2]: w2}
            pred = w0 * preds[names[0]] + w1 * preds[names[1]] + w2 * preds[names[2]]
            auc = float(roc_auc_score(y, pred))
            rows.append({"auc": auc, **weights})
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
    report = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return best_weights, best_auc, report


def random_search_three_weights(
    y: pd.Series,
    preds: Dict[str, np.ndarray],
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    names = list(preds.keys())
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    best_auc = -np.inf
    best_weights: Dict[str, float] = {}

    # 既存2モデルベースも必ず含める
    fixed = [
        np.array([0.427, 0.573, 0.0]),
        np.array([0.40, 0.60, 0.0]),
        np.array([0.45, 0.55, 0.0]),
        np.array([0.35, 0.65, 0.0]),
        np.array([0.30, 0.60, 0.10]),
        np.array([0.40, 0.50, 0.10]),
        np.array([0.35, 0.55, 0.10]),
        np.array([0.30, 0.50, 0.20]),
    ]
    for w in fixed:
        if abs(w.sum() - 1.0) < 1e-9 and np.all(w >= 0):
            pred = sum(w[k] * preds[names[k]] for k in range(3))
            auc = float(roc_auc_score(y, pred))
            weights = {names[k]: float(w[k]) for k in range(3)}
            rows.append({"auc": auc, **weights})
            if auc > best_auc:
                best_auc = auc
                best_weights = weights

    for t in range(int(n_trials)):
        # XGBが弱い場合を考え、XGBの重みが大きくなりすぎない候補も多めにする
        if t % 2 == 0:
            w = rng.dirichlet([4.0, 5.0, 1.2])
        else:
            w = rng.dirichlet([2.0, 2.0, 2.0])
        pred = sum(w[k] * preds[names[k]] for k in range(3))
        auc = float(roc_auc_score(y, pred))
        weights = {names[k]: float(w[k]) for k in range(3)}
        rows.append({"auc": auc, **weights})
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    report = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return best_weights, best_auc, report


def save_submission(test_ids: pd.Series, pred: np.ndarray, target: str, path: Path, eps: float) -> None:
    pd.DataFrame({"id": test_ids.values, target: clip_pred(pred, eps)}).to_csv(path, index=False)


# =============================================================================
# Args / main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight XGBoost and create LGBM+CatBoost+XGBoost ensemble submissions.")
    p.add_argument("--input-dir", type=str, default=r".\data")
    p.add_argument("--stage2-run-dir", type=str, default=r".\runs\f1_stage2_ensemble")
    p.add_argument("--catboost-run-dir", type=str, default=r".\runs\f1_lgbm_catboost_ensemble")
    p.add_argument("--output-dir", type=str, default=r".\runs\f1_xgb_three_model")
    p.add_argument("--helper-file", type=str, default="auto")
    p.add_argument("--original-file", type=str, default="auto")
    p.add_argument("--metadata-file", type=str, default="auto")
    p.add_argument("--target", type=str, default="PitNextLap")
    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--min-existing-auc", type=float, default=0.90, help="既存OOFの読み込みミス検出用。通常は0.90以上ならOK。")
    p.add_argument("--pred-clip-eps", type=float, default=1e-7)

    # Lightweight XGBoost defaults: residual time friendly
    p.add_argument("--xgboost-n-estimators", type=int, default=700)
    p.add_argument("--xgboost-learning-rate", type=float, default=0.035)
    p.add_argument("--xgboost-max-depth", type=int, default=4)
    p.add_argument("--xgboost-min-child-weight", type=float, default=80.0)
    p.add_argument("--xgboost-subsample", type=float, default=0.85)
    p.add_argument("--xgboost-colsample-bytree", type=float, default=0.75)
    p.add_argument("--xgboost-reg-alpha", type=float, default=0.20)
    p.add_argument("--xgboost-reg-lambda", type=float, default=8.0)
    p.add_argument("--xgboost-max-bin", type=int, default=256)
    p.add_argument("--xgboost-early-stopping", type=int, default=80)

    p.add_argument("--weight-step", type=float, default=0.01, help="3モデルgrid searchの刻み。0.01なら約5151通り。")
    p.add_argument("--random-weight-trials", type=int, default=8000)
    p.add_argument("--make-rank-ensemble", action="store_true", help="rank average版も追加で作る。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    stage2_run_dir = Path(args.stage2_run_dir)
    catboost_run_dir = Path(args.catboost_run_dir)
    ensure_dir(output_dir)

    log("=" * 88)
    log("F1 Pit Prediction: lightweight XGBoost + 3-model ensemble")
    log("Major outputs: XGB-only submission first, then LGBM+CatBoost+XGB ensemble submission.")
    log("=" * 88)
    log(f"[CONFIG] input_dir        = {args.input_dir}")
    log(f"[CONFIG] stage2_run_dir   = {stage2_run_dir}")
    log(f"[CONFIG] catboost_run_dir = {catboost_run_dir}")
    log(f"[CONFIG] output_dir       = {output_dir}")
    log(f"[CONFIG] xgb_rounds       = {args.xgboost_n_estimators}")

    if xgb is None:
        raise ImportError(f"xgboost is not installed/importable: {XGBOOST_IMPORT_ERROR}")

    if args.helper_file.lower() == "auto":
        helper_path = Path(__file__).resolve().parent / "f1_pit_lgbm_feature_ab.py"
    else:
        helper_path = Path(args.helper_file)
    helper = load_helper(helper_path)
    log(f"[INFO] helper loaded = {helper_path}")

    selected_features = read_feature_list(stage2_run_dir / "selected_features_after_ab.txt")
    log(f"[INFO] selected AB features loaded = {len(selected_features)}")

    # Reconstruct train/test feature data first to get true y and test ids
    train_feat, test_feat, y, test_ids, data_info = reconstruct_feature_data(args, helper)
    write_json(data_info, output_dir / "data_reconstruction_info.json")
    with open(output_dir / "selected_features_used.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    # Load existing LGBM AB-selected and CatBoost1000 predictions
    lgbm_oof, y_lgbm, lgbm_oof_col = load_oof_prediction(stage2_run_dir / "ab_selected_oof.csv", args.target, "LGBM_AB")
    lgbm_test, ids_lgbm, lgbm_test_col = load_test_prediction(stage2_run_dir / "ab_selected_test_pred.csv", args.target, "LGBM_AB", args.id_col)
    cat_oof, y_cat, cat_oof_col = load_oof_prediction(catboost_run_dir / "catboost_oof.csv", args.target, "CatBoost")
    cat_test, ids_cat, cat_test_col = load_test_prediction(catboost_run_dir / "catboost_test_pred.csv", args.target, "CatBoost", args.id_col)

    if len(lgbm_oof) != len(y) or len(cat_oof) != len(y):
        raise ValueError(f"OOF length mismatch: y={len(y)} lgbm={len(lgbm_oof)} cat={len(cat_oof)}")
    if len(lgbm_test) != len(test_ids) or len(cat_test) != len(test_ids):
        raise ValueError(f"test length mismatch: test_ids={len(test_ids)} lgbm={len(lgbm_test)} cat={len(cat_test)}")

    if y_lgbm is not None and not np.array_equal(y_lgbm, y.to_numpy()):
        raise ValueError("LGBM OOF target does not match reconstructed target by row order.")
    if y_cat is not None and not np.array_equal(y_cat, y.to_numpy()):
        raise ValueError("CatBoost OOF target does not match reconstructed target by row order.")

    lgbm_auc = validate_existing_model_auc(y, lgbm_oof, "LGBM_AB", args.min_existing_auc)
    cat_auc = validate_existing_model_auc(y, cat_oof, "CatBoost", args.min_existing_auc)

    # Save loaded prediction reports for reproducibility
    write_json(
        {
            "lgbm_oof_col": lgbm_oof_col,
            "lgbm_test_col": lgbm_test_col,
            "catboost_oof_col": cat_oof_col,
            "catboost_test_col": cat_test_col,
            "lgbm_oof_auc": lgbm_auc,
            "catboost_oof_auc": cat_auc,
        },
        output_dir / "loaded_existing_prediction_info.json",
    )

    # Prepare XGBoost matrices
    categorical_all = [c for c in helper.base_categorical_candidates() if c in selected_features]
    log("[STEP] Prepare XGBoost one-hot matrices")
    X_xgb, X_test_xgb, encoding_report = prepare_xgboost_matrix(train_feat, test_feat, selected_features, categorical_all)
    write_json(encoding_report, output_dir / "xgboost_onehot_encoding_report.json")
    log(f"[XGB] encoded features = {X_xgb.shape[1]} | train rows={len(X_xgb):,} test rows={len(X_test_xgb):,}")

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(train_feat, y))

    # Step 1: train lightweight XGBoost and save XGB-only submission immediately
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

    xgb_auc = float(roc_auc_score(y, xgb_oof))
    log(f"[RESULT] XGBoost OOF AUC = {xgb_auc:.6f}")
    log(f"[SAVE] XGB-only submission = {output_dir / f'submission_020_{xgb_tag}.csv'}")

    # Step 2: 3 model OOF weight search
    log("[STEP] 3-model weight search: LGBM + CatBoost + XGBoost")
    preds = {
        "lgbm_ab": lgbm_oof,
        "catboost": cat_oof,
        xgb_tag: xgb_oof,
    }
    test_preds = {
        "lgbm_ab": lgbm_test,
        "catboost": cat_test,
        xgb_tag: xgb_test,
    }

    best_grid_w, best_grid_auc, grid_report = grid_search_three_weights(y, preds, step=args.weight_step)
    grid_report.to_csv(output_dir / "three_model_weight_grid_report.csv", index=False)
    log(f"[WEIGHT][grid] best_auc={best_grid_auc:.6f} weights={best_grid_w}")

    best_rand_w, best_rand_auc, rand_report = random_search_three_weights(y, preds, n_trials=args.random_weight_trials, seed=args.seed)
    rand_report.to_csv(output_dir / "three_model_weight_random_report.csv", index=False)
    log(f"[WEIGHT][random] best_auc={best_rand_auc:.6f} weights={best_rand_w}")

    if best_rand_auc > best_grid_auc:
        best_w = best_rand_w
        best_auc = best_rand_auc
        best_method = "random"
    else:
        best_w = best_grid_w
        best_auc = best_grid_auc
        best_method = "grid"

    ens_oof = sum(best_w[name] * preds[name] for name in preds)
    ens_test = sum(best_w[name] * test_preds[name] for name in test_preds)
    ens_auc = float(roc_auc_score(y, ens_oof))
    write_json(
        {"best_method": best_method, "best_oof_auc": ens_auc, "best_weights": best_w},
        output_dir / "best_three_model_weights.json",
    )
    pd.DataFrame({"oof_pred": ens_oof, args.target: y.values}).to_csv(output_dir / "three_model_ensemble_oof.csv", index=False)
    pd.DataFrame({"id": test_ids.values, "pred": ens_test}).to_csv(output_dir / "three_model_ensemble_test_pred.csv", index=False)
    save_submission(test_ids, ens_test, args.target, output_dir / "submission_021_lgbm_catboost_xgboost_ensemble.csv", args.pred_clip_eps)
    log(f"[ENSEMBLE] saved | oof_auc={ens_auc:.6f} | weights={best_w}")
    log(f"[SAVE] 3-model submission = {output_dir / 'submission_021_lgbm_catboost_xgboost_ensemble.csv'}")

    rank_summary = None
    if args.make_rank_ensemble:
        log("[STEP] Rank-average ensemble using the same weights")
        rank_oof = sum(best_w[name] * rank01(preds[name]) for name in preds)
        rank_test = sum(best_w[name] * rank01(test_preds[name]) for name in test_preds)
        rank_auc = float(roc_auc_score(y, rank_oof))
        pd.DataFrame({"oof_pred": rank_oof, args.target: y.values}).to_csv(output_dir / "three_model_rank_ensemble_oof.csv", index=False)
        pd.DataFrame({"id": test_ids.values, "pred": rank_test}).to_csv(output_dir / "three_model_rank_ensemble_test_pred.csv", index=False)
        save_submission(test_ids, rank_test, args.target, output_dir / "submission_022_rank_lgbm_catboost_xgboost_ensemble.csv", args.pred_clip_eps)
        rank_summary = {"oof_auc": rank_auc, "weights": best_w}
        log(f"[RANK ENSEMBLE] saved | oof_auc={rank_auc:.6f} | submission=submission_022_rank_lgbm_catboost_xgboost_ensemble.csv")

    # Correlation and scores
    model_oofs = {
        "lgbm_ab": lgbm_oof,
        "catboost": cat_oof,
        xgb_tag: xgb_oof,
        "three_model_ensemble": ens_oof,
    }
    if args.make_rank_ensemble:
        model_oofs["rank_three_model_ensemble"] = rank_oof
    corr_df = pd.DataFrame(model_oofs).corr()
    corr_df.to_csv(output_dir / "prediction_correlation_matrix.csv")
    score_df = pd.DataFrame(
        [{"model": name, "oof_auc": float(roc_auc_score(y, pred))} for name, pred in model_oofs.items()]
    ).sort_values("oof_auc", ascending=False)
    score_df.to_csv(output_dir / "model_oof_score_report.csv", index=False)

    summary = {
        "runtime_seconds": float(time.time() - start),
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost": getattr(xgb, "__version__", None),
        },
        "selected_features_count": len(selected_features),
        "existing_model_scores": {
            "lgbm_ab_oof_auc": lgbm_auc,
            "catboost_oof_auc": cat_auc,
        },
        "xgboost_summary": xgb_summary,
        "three_model_ensemble": {
            "oof_auc": ens_auc,
            "weights": best_w,
            "weight_search_method": best_method,
        },
        "rank_ensemble": rank_summary,
        "recommended_submission_order": [
            "submission_021_lgbm_catboost_xgboost_ensemble.csv",
            f"submission_020_{xgb_tag}.csv",
            "submission_022_rank_lgbm_catboost_xgboost_ensemble.csv" if args.make_rank_ensemble else None,
        ],
    }
    write_json(summary, output_dir / "xgb_three_model_run_summary.json")

    log("=" * 88)
    log("DONE")
    log(f"[OUT] {output_dir}")
    log("[KEY FILES]")
    log(f"  XGB-only       : {output_dir / f'submission_020_{xgb_tag}.csv'}")
    log(f"  3-model blend  : {output_dir / 'submission_021_lgbm_catboost_xgboost_ensemble.csv'}")
    if args.make_rank_ensemble:
        log(f"  rank blend     : {output_dir / 'submission_022_rank_lgbm_catboost_xgboost_ensemble.csv'}")
    log(f"  weights        : {output_dir / 'best_three_model_weights.json'}")
    log("=" * 88)


if __name__ == "__main__":
    main()
