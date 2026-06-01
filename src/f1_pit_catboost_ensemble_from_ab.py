#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_pit_catboost_ensemble_from_ab.py

目的:
  既に保存済みの AB-selected LightGBM 結果を使って、CatBoost を追加学習し、
  LGBM(AB-selected) + CatBoost の最適重み付きアンサンブル submission を作る。

前提:
  - src/f1_pit_lgbm_feature_ab.py が存在すること
  - stage2-run-dir に以下が存在すること
      selected_features_after_ab.txt
      ab_selected_oof.csv
      ab_selected_test_pred.csv
      submission_002_ab_selected.csv  (なくても可)
  - data に train.csv / test.csv / 任意で f1_strategy_dataset_v4.csv があること

実行例:
  python .\src\f1_pit_catboost_ensemble_from_ab.py `
    --input-dir .\data `
    --stage2-run-dir .\runs\f1_stage2_ensemble `
    --output-dir .\runs\f1_lgbm_catboost_ensemble `
    --catboost-iterations 1000 `
    --weight-grid-size 1001

必要:
  pip install pandas numpy scikit-learn catboost

任意:
  lightgbm / optuna はこのスクリプトでは直接使いませんが、helperがimportする場合があります。
  pip install lightgbm optuna
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = e
else:
    CATBOOST_IMPORT_ERROR = None


# ============================================================
# Utility
# ============================================================


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
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {path}")
    feats = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not feats:
        raise ValueError(f"特徴量ファイルが空です: {path}")
    return feats


def detect_prediction_column(df: pd.DataFrame, kind: str, target_col: str = "PitNextLap") -> str:
    """OOF/test_predファイルから予測列を推定する。"""
    priority = [
        "oof_pred",
        "test_pred",
        "pred",
        "prediction",
        "prob",
        "probability",
        "catboost_pred",
        "lgbm_pred",
        target_col,
    ]
    for c in priority:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            if kind == "oof" and c == target_col and len(df.select_dtypes(include=[np.number]).columns) > 1:
                # OOFではPitNextLapは目的変数の可能性があるため、他に候補があるなら避けたい
                continue
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    exclude = {"id", target_col, "target", "y", "label"}
    candidates = [c for c in numeric_cols if c not in exclude]
    if candidates:
        return candidates[0]
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        return target_col
    raise ValueError(f"{kind} の予測列を検出できません。columns={df.columns.tolist()}")


def save_submission(ids: Sequence[Any], preds: np.ndarray, path: Path, target_col: str) -> None:
    preds = np.asarray(preds, dtype=float)
    preds = np.clip(preds, 1e-7, 1 - 1e-7)
    sub = pd.DataFrame({"id": ids, target_col: preds})
    sub.to_csv(path, index=False)


def environment_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for name in ["numpy", "pandas", "sklearn", "catboost", "lightgbm", "optuna"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


# ============================================================
# Load helper from src/f1_pit_lgbm_feature_ab.py
# ============================================================


def load_helper(helper_path: Path):
    if not helper_path.exists():
        raise FileNotFoundError(
            f"helper が見つかりません: {helper_path}\n"
            "src/f1_pit_lgbm_feature_ab.py を同じ src フォルダに置いてください。"
        )
    spec = importlib.util.spec_from_file_location("f1_pit_lgbm_feature_ab_helper", str(helper_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"helper を読み込めません: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# Rebuild train/test features exactly using helper logic
# ============================================================


def rebuild_features(args: argparse.Namespace, helper, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, List[str]]:
    input_dir = Path(args.input_dir)
    target_col = helper.clean_column_name(args.target)
    id_col = helper.clean_column_name(args.id_col)

    train_path = helper.find_csv(input_dir, args.train_file, ["train.csv"], required=True)
    test_path = helper.find_csv(input_dir, args.test_file, ["test.csv"], required=True)
    original_path = helper.find_csv(
        input_dir,
        args.original_file,
        ["f1_strategy_dataset_v4.csv", "original.csv", "f1_strategy_dataset.csv"],
        required=False,
    )

    print(f"[INFO] train: {train_path}")
    print(f"[INFO] test : {test_path}")
    print(f"[INFO] original: {original_path if original_path else 'not used'}")

    train_raw, train_mapping = helper.read_csv_clean(train_path)
    test_raw, test_mapping = helper.read_csv_clean(test_path)
    write_json({"train": train_mapping, "test": test_mapping}, output_dir / "column_mapping_rebuild.json")

    if target_col not in train_raw.columns:
        raise ValueError(f"target column '{target_col}' がtrainにありません。columns={train_raw.columns.tolist()}")
    if id_col not in test_raw.columns:
        print(f"[WARN] testにID列 '{id_col}' がありません。indexをidとして使います。")
        test_raw[id_col] = np.arange(len(test_raw))

    original_report: Dict[str, Any] = {"status": "not_used"}
    if original_path is not None:
        original_raw, original_mapping = helper.read_csv_clean(original_path)
        write_json(original_mapping, output_dir / "original_column_mapping_rebuild.json")
        original_clean, original_report = helper.remove_original_overlap(original_raw, train_raw, test_raw, target_col, id_col)
        if len(original_clean) > 0:
            common_for_train = [c for c in train_raw.columns if c in original_clean.columns]
            original_clean = original_clean[common_for_train].copy()
            train_raw = pd.concat([train_raw, original_clean], axis=0, ignore_index=True)
            print(f"[INFO] original added rows: {len(original_clean):,}")
        else:
            print(f"[INFO] original skipped or no usable rows. status={original_report.get('status')}")

    train_raw, train_clean_report = helper.basic_train_clean(train_raw, target_col)
    test_raw, test_clean_report = helper.basic_test_clean_keep_ids(test_raw)

    print(f"[INFO] train rows after clean: {len(train_raw):,}")
    print(f"[INFO] test rows: {len(test_raw):,}")

    train_raw["__is_train"] = 1
    test_raw["__is_train"] = 0
    all_cols = sorted(set(train_raw.columns).union(set(test_raw.columns)))
    train_aligned = train_raw.reindex(columns=all_cols)
    test_aligned = test_raw.reindex(columns=all_cols)
    all_df = pd.concat([train_aligned, test_aligned], axis=0, ignore_index=True)

    external_meta = helper.load_external_metadata(input_dir, args.metadata_file)
    all_feat, created_features = helper.add_engineered_features(all_df, external_meta)
    pd.DataFrame(created_features).to_csv(output_dir / "engineered_feature_catalog_rebuild.csv", index=False)

    train_feat = all_feat[all_feat["__is_train"].eq(1)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)
    test_feat = all_feat[all_feat["__is_train"].eq(0)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)

    categorical_all = [c for c in helper.base_categorical_candidates() if c in train_feat.columns and c in test_feat.columns]

    rebuild_report = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "original_path": str(original_path) if original_path else None,
        "original_report": original_report,
        "train_clean_report": train_clean_report,
        "test_clean_report": test_clean_report,
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "target_mean": float(train_feat[target_col].astype(int).mean()),
        "categorical_all": categorical_all,
    }
    write_json(rebuild_report, output_dir / "data_rebuild_report.json")
    return train_feat, test_feat, target_col, id_col, categorical_all


# ============================================================
# CatBoost model
# ============================================================


def make_catboost_matrix(
    helper,
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    features: List[str],
    target_col: str,
    categorical_all: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[int], Dict[str, Any]]:
    X, y, X_test, cats, impute_report = helper.prepare_X_y(train_feat, test_feat, features, target_col, categorical_all)

    # CatBoostにはカテゴリ列を文字列で渡すのが安全
    for c in cats:
        X[c] = X[c].astype(str).fillna("__MISSING__")
        X_test[c] = X_test[c].astype(str).fillna("__MISSING__")

    # 数値列はfloat化
    for c in features:
        if c not in cats:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0.0).astype(float)

    cat_indices = [features.index(c) for c in cats if c in features]
    return X, y, X_test, cats, cat_indices, impute_report


def default_catboost_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(args.catboost_iterations),
        "learning_rate": float(args.catboost_learning_rate),
        "depth": int(args.catboost_depth),
        "l2_leaf_reg": float(args.catboost_l2_leaf_reg),
        "random_strength": float(args.catboost_random_strength),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": float(args.catboost_bagging_temperature),
        "od_type": "Iter",
        "od_wait": int(args.early_stopping_rounds),
        "random_seed": int(args.seed),
        "thread_count": int(args.n_jobs),
        "allow_writing_files": False,
        "verbose": False,
    }


def run_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    features: List[str],
    cat_indices: List[int],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if CatBoostClassifier is None or Pool is None:
        raise ImportError(
            "catboost が見つかりません。先に `python -m pip install catboost` を実行してください。\n"
            f"original error: {CATBOOST_IMPORT_ERROR}"
        )

    print("\n[STEP] CatBoost CV / OOF / test prediction")
    print(f"[CAT] rows={len(X):,} test={len(X_test):,} features={len(features)} categorical={len(cat_indices)} folds={len(folds)}")

    oof = np.zeros(len(X), dtype=float)
    test_preds_folds: List[np.ndarray] = []
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []
    start_all = time.time()

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_start = time.time()
        print(f"[CAT][fold {fold}/{len(folds)}] start | train={len(tr_idx):,} valid={len(va_idx):,}")
        X_tr = X.iloc[tr_idx][features]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx][features]
        y_va = y.iloc[va_idx]

        model_params = dict(params)
        model_params["random_seed"] = seed + fold
        model = CatBoostClassifier(**model_params)
        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        valid_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        p_va = model.predict_proba(valid_pool)[:, 1]
        p_tr = model.predict_proba(train_pool)[:, 1]
        test_pool = Pool(X_test[features], cat_features=cat_indices)
        p_test = model.predict_proba(test_pool)[:, 1]

        va_auc = float(roc_auc_score(y_va, p_va))
        tr_auc = float(roc_auc_score(y_tr, p_tr))
        gap = tr_auc - va_auc
        best_iter = int(model.get_best_iteration() or model_params.get("iterations", 0))
        best_iters.append(best_iter)
        oof[va_idx] = p_va
        test_preds_folds.append(p_test)

        try:
            model.save_model(str(output_dir / "models" / f"catboost_fold{fold}.cbm"))
        except Exception as e:
            print(f"[WARN] failed to save fold model: {e}")

        fold_time = time.time() - fold_start
        done = fold
        remain = (time.time() - start_all) / max(done, 1) * (len(folds) - done)
        print(
            f"[CAT][fold {fold}/{len(folds)}] done | valid_auc={va_auc:.6f} "
            f"train_auc={tr_auc:.6f} gap={gap:.6f} best_iter={best_iter} "
            f"fold_time={fold_time/60:.1f}m eta={remain/60:.1f}m"
        )
        fold_rows.append({
            "fold": fold,
            "valid_auc": va_auc,
            "train_auc": tr_auc,
            "gap": gap,
            "best_iteration": best_iter,
            "fold_time_seconds": fold_time,
        })

    test_pred = np.mean(test_preds_folds, axis=0)
    valid_aucs = [r["valid_auc"] for r in fold_rows]
    train_aucs = [r["train_auc"] for r in fold_rows]
    gaps = [r["gap"] for r in fold_rows]
    summary = {
        "model": "CatBoostClassifier",
        "params": params,
        "features": features,
        "n_features": int(len(features)),
        "cat_indices": cat_indices,
        "valid_auc_mean": float(np.mean(valid_aucs)),
        "valid_auc_std": float(np.std(valid_aucs)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "gap_mean": float(np.mean(gaps)),
        "best_iteration_mean": float(np.mean(best_iters)),
        "best_iteration_median": float(np.median(best_iters)),
        "folds": fold_rows,
        "runtime_seconds": float(time.time() - start_all),
    }
    print(
        f"[CAT] finished | valid_auc={summary['valid_auc_mean']:.6f} ± {summary['valid_auc_std']:.6f} "
        f"train_auc={summary['train_auc_mean']:.6f} gap={summary['gap_mean']:.6f} "
        f"time={summary['runtime_seconds']/60:.1f}m"
    )
    pd.DataFrame(fold_rows).to_csv(output_dir / "catboost_fold_report.csv", index=False)
    return oof, test_pred, summary


def fit_final_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    features: List[str],
    cat_indices: List[int],
    params: Dict[str, Any],
    best_iteration_median: float,
    seed: int,
    output_dir: Path,
) -> np.ndarray:
    print("\n[STEP] Train final CatBoost on full train")
    final_params = dict(params)
    # CVで見つかったbest iterationに少し余裕を持たせる
    if best_iteration_median and best_iteration_median > 0:
        final_params["iterations"] = max(200, int(best_iteration_median * 1.15))
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)
    final_params["random_seed"] = seed
    model = CatBoostClassifier(**final_params)
    train_pool = Pool(X[features], y, cat_features=cat_indices)
    model.fit(train_pool)
    try:
        model.save_model(str(output_dir / "models" / "catboost_final.cbm"))
    except Exception as e:
        print(f"[WARN] failed to save final CatBoost model: {e}")
    test_pool = Pool(X_test[features], cat_features=cat_indices)
    pred = model.predict_proba(test_pool)[:, 1]
    write_json(final_params, output_dir / "catboost_final_params.json")
    return pred


# ============================================================
# Ensemble
# ============================================================


def correlation_and_scores(
    y: np.ndarray,
    model_oofs: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    corr_df = pd.DataFrame(model_oofs).corr()
    corr_df.to_csv(output_dir / "prediction_correlation_matrix.csv")

    rows = []
    for name, pred in model_oofs.items():
        rows.append({
            "model": name,
            "oof_auc": float(roc_auc_score(y, pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
        })
    pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_csv(output_dir / "model_oof_score_report.csv", index=False)


def search_two_model_weights(
    y: np.ndarray,
    lgbm_oof: np.ndarray,
    cat_oof: np.ndarray,
    lgbm_test: np.ndarray,
    cat_test: np.ndarray,
    grid_size: int,
    output_dir: Path,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, pd.DataFrame]:
    print("\n[STEP] Weight search: LGBM AB-selected + CatBoost")
    grid_size = max(11, int(grid_size))
    weights = np.linspace(0.0, 1.0, grid_size)  # weight for LGBM
    rows = []
    best_auc = -np.inf
    best_w = 0.5
    start = time.time()

    for i, w_lgbm in enumerate(weights, start=1):
        w_cat = 1.0 - w_lgbm
        pred = w_lgbm * lgbm_oof + w_cat * cat_oof
        auc = float(roc_auc_score(y, pred))
        rows.append({"weight_lgbm_ab_selected": float(w_lgbm), "weight_catboost": float(w_cat), "oof_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_w = float(w_lgbm)
        if i == 1 or i == grid_size or i % max(1, grid_size // 10) == 0:
            print(f"[WEIGHT] {i}/{grid_size} | current_best_auc={best_auc:.6f} | best_lgbm_weight={best_w:.3f}")

    report = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    report.to_csv(output_dir / "ensemble_weight_search_report.csv", index=False)

    best_weights = {
        "lgbm_ab_selected": best_w,
        "catboost": 1.0 - best_w,
        "oof_auc": best_auc,
        "grid_size": grid_size,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(best_weights, output_dir / "best_ensemble_weights.json")

    ens_oof = best_w * lgbm_oof + (1.0 - best_w) * cat_oof
    ens_test = best_w * lgbm_test + (1.0 - best_w) * cat_test
    print(
        f"[ENSEMBLE] best | auc={best_auc:.6f} | "
        f"lgbm={best_w:.4f} catboost={1.0-best_w:.4f}"
    )
    return best_weights, ens_oof, ens_test, report


# ============================================================
# Args / main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost + LGBM AB-selected ensemble for F1 PitNextLap")
    parser.add_argument("--input-dir", type=str, default=".\\data")
    parser.add_argument("--stage2-run-dir", type=str, default=".\\runs\\f1_stage2_ensemble", help="ab_selected_oof等があるフォルダ")
    parser.add_argument("--output-dir", type=str, default=".\\runs\\f1_lgbm_catboost_ensemble")
    parser.add_argument("--helper-file", type=str, default="auto", help="src/f1_pit_lgbm_feature_ab.py。autoならこのファイルと同じフォルダから読む")
    parser.add_argument("--train-file", type=str, default="auto")
    parser.add_argument("--test-file", type=str, default="auto")
    parser.add_argument("--original-file", type=str, default="auto")
    parser.add_argument("--metadata-file", type=str, default="auto")
    parser.add_argument("--target", type=str, default="PitNextLap")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    # CatBoost params: 時間重視なら iterations を 700〜1000 にする
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.045)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--catboost-random-strength", type=float, default=0.5)
    parser.add_argument("--catboost-bagging-temperature", type=float, default=0.25)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)

    # Ensemble
    parser.add_argument("--weight-grid-size", type=int, default=1001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_dir = Path(args.output_dir)
    stage2_dir = Path(args.stage2_run_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "models")

    if args.helper_file.lower() == "auto":
        helper_path = Path(__file__).resolve().parent / "f1_pit_lgbm_feature_ab.py"
    else:
        helper_path = Path(args.helper_file)
    helper = load_helper(helper_path)

    print("=" * 90)
    print("F1 Pit Prediction: LGBM AB-selected + CatBoost Ensemble")
    print("=" * 90)
    print(f"[CONFIG] input_dir      = {args.input_dir}")
    print(f"[CONFIG] stage2_run_dir = {stage2_dir}")
    print(f"[CONFIG] output_dir     = {output_dir}")
    print(f"[CONFIG] helper         = {helper_path}")
    print(f"[CONFIG] catboost_iter  = {args.catboost_iterations}")
    print(f"[CONFIG] weight_grid    = {args.weight_grid_size}")
    write_json(vars(args), output_dir / "args.json")
    write_json(environment_versions(), output_dir / "environment_versions.json")

    # 必須ファイル確認
    selected_features_path = stage2_dir / "selected_features_after_ab.txt"
    lgbm_oof_path = stage2_dir / "ab_selected_oof.csv"
    lgbm_test_path = stage2_dir / "ab_selected_test_pred.csv"
    for p in [selected_features_path, lgbm_oof_path, lgbm_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"必要ファイルが見つかりません: {p}")

    selected_features = read_feature_list(selected_features_path)
    print(f"[INFO] selected AB features loaded: {len(selected_features)}")
    (output_dir / "selected_features_after_ab_used.txt").write_text("\n".join(selected_features), encoding="utf-8")

    # データ再構築
    train_feat, test_feat, target_col, id_col, categorical_all = rebuild_features(args, helper, output_dir)
    missing_features = [f for f in selected_features if f not in train_feat.columns or f not in test_feat.columns]
    if missing_features:
        raise ValueError(f"再構築データに存在しない特徴量があります: {missing_features}")

    y = train_feat[target_col].astype(int).values
    test_ids = test_feat[id_col].values if id_col in test_feat.columns else np.arange(len(test_feat))

    # LGBM AB-selected OOF/test予測を読み込み
    lgbm_oof_df = pd.read_csv(lgbm_oof_path)
    lgbm_test_df = pd.read_csv(lgbm_test_path)
    lgbm_oof_col = detect_prediction_column(lgbm_oof_df, kind="oof", target_col=target_col)
    lgbm_test_col = detect_prediction_column(lgbm_test_df, kind="test", target_col=target_col)
    lgbm_oof = lgbm_oof_df[lgbm_oof_col].astype(float).values
    lgbm_test = lgbm_test_df[lgbm_test_col].astype(float).values

    if len(lgbm_oof) != len(train_feat):
        raise ValueError(f"LGBM OOFの行数がtrainと一致しません: oof={len(lgbm_oof)} train={len(train_feat)}")
    if len(lgbm_test) != len(test_feat):
        raise ValueError(f"LGBM test predの行数がtestと一致しません: pred={len(lgbm_test)} test={len(test_feat)}")

    lgbm_auc = float(roc_auc_score(y, lgbm_oof))
    print(f"[INFO] LGBM AB-selected OOF loaded | col={lgbm_oof_col} | auc={lgbm_auc:.6f}")

    # CatBoost用行列
    X, y_series, X_test, cats, cat_indices, impute_report = make_catboost_matrix(
        helper,
        train_feat,
        test_feat,
        selected_features,
        target_col,
        categorical_all,
    )
    write_json(impute_report, output_dir / "catboost_impute_report.json")
    write_json({"categorical_features": cats, "cat_indices": cat_indices}, output_dir / "catboost_categorical_report.json")

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X, y_series))
    fold_assign = np.zeros(len(X), dtype=int)
    for i, (_, va_idx) in enumerate(folds, start=1):
        fold_assign[va_idx] = i
    pd.DataFrame({"row_index": np.arange(len(X)), "fold": fold_assign, target_col: y_series.values}).to_csv(
        output_dir / "cv_fold_assignment.csv", index=False
    )

    # CatBoost CV
    cat_params = default_catboost_params(args)
    write_json(cat_params, output_dir / "catboost_params.json")
    cat_oof, cat_test_cv, cat_summary = run_catboost_cv(
        X,
        y_series,
        X_test,
        selected_features,
        cat_indices,
        cat_params,
        folds,
        args.seed,
        output_dir,
    )
    write_json(cat_summary, output_dir / "catboost_cv_summary.json")

    # final CatBoost on full train: submissionにはこちらを使う。CV平均版も保存する。
    cat_test_final = fit_final_catboost(
        X,
        y_series,
        X_test,
        selected_features,
        cat_indices,
        cat_params,
        cat_summary.get("best_iteration_median", args.catboost_iterations),
        args.seed,
        output_dir,
    )

    # OOF / test pred / CatBoost単体submission保存
    cat_oof_df = pd.DataFrame({"oof_pred": cat_oof, target_col: y_series.values})
    if id_col in train_feat.columns:
        cat_oof_df.insert(0, "id", train_feat[id_col].values)
    cat_oof_df.to_csv(output_dir / "catboost_oof.csv", index=False)

    pd.DataFrame({"id": test_ids, "test_pred_cv_mean": cat_test_cv, "test_pred_final_full_train": cat_test_final}).to_csv(
        output_dir / "catboost_test_pred.csv", index=False
    )
    save_submission(test_ids, cat_test_final, output_dir / "submission_005_catboost.csv", target_col)
    print(f"[SAVE] {output_dir / 'submission_005_catboost.csv'}")

    # 相関と単体スコア
    model_oofs = {"lgbm_ab_selected": lgbm_oof, "catboost": cat_oof}
    correlation_and_scores(y, model_oofs, output_dir)

    # 重み探索: OOFではCatBoost CV OOFを使う。testはfinal full-train predを使う。
    best_weights, ens_oof, ens_test, weight_report = search_two_model_weights(
        y,
        lgbm_oof,
        cat_oof,
        lgbm_test,
        cat_test_final,
        args.weight_grid_size,
        output_dir,
    )

    ens_oof_df = pd.DataFrame({"oof_pred": ens_oof, target_col: y})
    if id_col in train_feat.columns:
        ens_oof_df.insert(0, "id", train_feat[id_col].values)
    ens_oof_df.to_csv(output_dir / "ensemble_oof.csv", index=False)

    pd.DataFrame({"id": test_ids, "test_pred": ens_test}).to_csv(output_dir / "ensemble_test_pred.csv", index=False)
    save_submission(test_ids, ens_test, output_dir / "submission_007_lgbm_catboost_ensemble.csv", target_col)
    print(f"[SAVE] {output_dir / 'submission_007_lgbm_catboost_ensemble.csv'}")

    summary = {
        "runtime_seconds": float(time.time() - start_time),
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "n_features": int(len(selected_features)),
        "features": selected_features,
        "lgbm_ab_selected_oof_auc": lgbm_auc,
        "catboost_oof_auc": float(roc_auc_score(y, cat_oof)),
        "ensemble_oof_auc": float(roc_auc_score(y, ens_oof)),
        "best_ensemble_weights": best_weights,
        "outputs": {
            "catboost_submission": str(output_dir / "submission_005_catboost.csv"),
            "ensemble_submission": str(output_dir / "submission_007_lgbm_catboost_ensemble.csv"),
            "catboost_oof": str(output_dir / "catboost_oof.csv"),
            "ensemble_oof": str(output_dir / "ensemble_oof.csv"),
            "weights": str(output_dir / "best_ensemble_weights.json"),
            "correlation": str(output_dir / "prediction_correlation_matrix.csv"),
        },
    }
    write_json(summary, output_dir / "lgbm_catboost_ensemble_summary.json")

    print("\n" + "=" * 90)
    print("DONE")
    print(f"LGBM AB-selected OOF AUC : {summary['lgbm_ab_selected_oof_auc']:.6f}")
    print(f"CatBoost OOF AUC         : {summary['catboost_oof_auc']:.6f}")
    print(f"Ensemble OOF AUC         : {summary['ensemble_oof_auc']:.6f}")
    print(f"Best weights             : LGBM={best_weights['lgbm_ab_selected']:.4f}, CatBoost={best_weights['catboost']:.4f}")
    print(f"Submission               : {output_dir / 'submission_007_lgbm_catboost_ensemble.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
