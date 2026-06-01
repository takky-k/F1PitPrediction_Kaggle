#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_pit_stage2_ensemble_pipeline.py

前提:
- baseline Optuna は既に完了していて、initial_best_params.json が保存されている。
- このファイルは src フォルダに置いて実行する想定。
- 既存の src/f1_pit_lgbm_feature_ab.py を helper として読み込み、同じ前処理・特徴量作成を使う。

やること:
1. baseline best params を読み込む
2. baseline LGBM の OOF / test_pred / submission を作成
3. 追加特徴量ABテスト
4. AB-selected LGBM の OOF / test_pred / submission を作成
5. feature importance 下位から削除テスト
6. pruned LGBM の OOF / test_pred / submission を作成
7. Group CV by Race-Year で過学習チェック
8. LGBM seed ensemble
9. CatBoost / XGBoost 単体モデル
10. OOF予測相関チェック
11. OOF AUCでアンサンブル比率探索
12. 効いたモデルだけ final Optuna
13. final ensemble submission を作成

実行例:
    cd C:\Projects\F1PitPrediction
    python .\src\f1_pit_stage2_ensemble_pipeline.py ^
      --input-dir .\data ^
      --baseline-run-dir .\runs\f1_lgbm_ab_001 ^
      --output-dir .\runs\f1_stage2_ensemble

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm optuna

任意:
    pip install catboost xgboost

注意:
- CatBoost / XGBoost が未インストールなら自動でスキップします。
- submission はCSVを作るだけです。Kaggleへの提出は手動で行ってください。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None

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
# 0. helper script loader
# ============================================================


def load_helper(helper_path: Optional[str] = None):
    """既存の f1_pit_lgbm_feature_ab.py を読み込む。"""
    candidates: List[Path] = []
    if helper_path and helper_path.lower() not in {"auto", "none", "null"}:
        candidates.append(Path(helper_path))

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates.extend([
        here / "f1_pit_lgbm_feature_ab.py",
        cwd / "src" / "f1_pit_lgbm_feature_ab.py",
        cwd / "f1_pit_lgbm_feature_ab.py",
    ])

    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("f1_pit_helper", str(p))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            print(f"[INFO] helper loaded: {p}")
            return mod

    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "helper script f1_pit_lgbm_feature_ab.py が見つかりません。\n"
        "この stage2 script と同じ src フォルダに置いてください。\n"
        f"探した場所:\n{msg}"
    )


# ============================================================
# 1. utility
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_params_from_run(run_dir: Path, seed: int, n_jobs: int, hp: Any) -> Dict[str, Any]:
    """baseline Optunaで保存されたbest paramsを読み込む。"""
    candidates = [
        run_dir / "initial_best_params.json",
        run_dir / "baseline_best_params.json",
        run_dir / "best_params.json",
        run_dir / "final_best_params.json",
    ]
    found = None
    for p in candidates:
        if p.exists():
            found = p
            break
    if found is None:
        msg = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "baseline best params が見つかりません。\n"
            "baseline Optuna終了後の initial_best_params.json を指定フォルダに置いてください。\n"
            f"探した場所:\n{msg}"
        )

    obj = read_json(found)
    if isinstance(obj, dict) and "best_params" in obj and isinstance(obj["best_params"], dict):
        params = obj["best_params"]
    else:
        params = obj

    base = hp.default_lgbm_params(seed=seed, n_jobs=n_jobs)
    base.update(params)
    base["random_state"] = seed
    base["n_jobs"] = n_jobs
    base.setdefault("objective", "binary")
    base.setdefault("metric", "auc")
    base.setdefault("boosting_type", "gbdt")
    base.setdefault("verbosity", -1)
    base.setdefault("force_col_wise", True)
    print(f"[INFO] params loaded: {found}")
    return base


def make_submission(test_ids: pd.Series, preds: np.ndarray, target_col: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": test_ids.values, target_col: np.clip(preds, 0.0, 1.0)})
    out.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_oof(row_ids: Optional[pd.Series], y: pd.Series, preds: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "target": y.values,
        "prediction": np.clip(preds, 0.0, 1.0),
    })
    if row_ids is not None:
        df.insert(1, "id", row_ids.values)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_test_pred(test_ids: pd.Series, preds: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids.values, "prediction": np.clip(preds, 0.0, 1.0)}).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def parse_seed_list(s: str) -> List[int]:
    vals: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals or [42]


def get_train_ids(train_df: pd.DataFrame, id_col: str) -> Optional[pd.Series]:
    if id_col in train_df.columns:
        return train_df[id_col].copy()
    return None


def get_test_ids(test_df: pd.DataFrame, id_col: str) -> pd.Series:
    if id_col in test_df.columns:
        return test_df[id_col].copy()
    return pd.Series(np.arange(len(test_df)), name="id")



def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def print_banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def evaluate_lgbm_cv_verbose(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    categorical_features: Sequence[str],
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    name: str,
    return_oof: bool = False,
    return_importance: bool = False,
) -> Dict[str, Any]:
    """LightGBM CV with detailed console progress."""
    if lgb is None:
        raise ImportError("lightgbm が見つかりません。pip install lightgbm を実行してください。")

    features = list(features)
    cat_features = [c for c in categorical_features if c in features]
    valid_aucs: List[float] = []
    train_aucs: List[float] = []
    best_iterations: List[int] = []
    oof = np.zeros(len(X), dtype=float) if return_oof else None
    importances: List[pd.DataFrame] = []

    stage_start = time.time()
    print(
        f"[LGBM][{name}] CV start | folds={len(folds)} | rows={len(X):,} | "
        f"features={len(features)} | categorical={len(cat_features)}",
        flush=True,
    )

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_start = time.time()
        X_tr = X.iloc[tr_idx][features]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx][features]
        y_va = y.iloc[va_idx]

        print(
            f"[LGBM][{name}][fold {fold}/{len(folds)}] start | "
            f"train={len(tr_idx):,} valid={len(va_idx):,}",
            flush=True,
        )

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
        best_iter = int(model.best_iteration_ or model_params.get("n_estimators", 0))
        best_iterations.append(best_iter)
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

        elapsed_fold = time.time() - fold_start
        avg_fold = (time.time() - stage_start) / fold
        remaining = avg_fold * (len(folds) - fold)
        print(
            f"[LGBM][{name}][fold {fold}/{len(folds)}] done | "
            f"valid_auc={va_auc:.6f} train_auc={tr_auc:.6f} "
            f"gap={tr_auc - va_auc:.6f} best_iter={best_iter} | "
            f"fold_time={format_seconds(elapsed_fold)} eta={format_seconds(remaining)}",
            flush=True,
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

    print(
        f"[LGBM][{name}] CV finished | valid_auc={result['valid_auc_mean']:.6f} "
        f"± {result['valid_auc_std']:.6f} | train_auc={result['train_auc_mean']:.6f} | "
        f"gap={result['overfit_gap_mean']:.6f} | time={format_seconds(time.time() - stage_start)}",
        flush=True,
    )
    return result


# ============================================================
# 2. data reconstruction using helper
# ============================================================


def reconstruct_dataset(args: argparse.Namespace, hp: Any):
    """baseline pipeline と同じデータ結合・前処理・特徴量作成を再現する。"""
    input_dir = Path(args.input_dir)
    target_col = hp.clean_column_name(args.target)
    id_col = hp.clean_column_name(args.id_col)

    train_path = hp.find_csv(input_dir, args.train_file, ["train.csv"], required=True)
    test_path = hp.find_csv(input_dir, args.test_file, ["test.csv"], required=True)
    original_path = hp.find_csv(
        input_dir,
        args.original_file,
        ["f1_strategy_dataset_v4.csv", "original.csv", "f1_strategy_dataset.csv"],
        required=False,
    )

    print(f"[INFO] train: {train_path}")
    print(f"[INFO] test : {test_path}")
    print(f"[INFO] original: {original_path if original_path else 'not used'}")

    train_raw, train_mapping = hp.read_csv_clean(train_path)
    test_raw, test_mapping = hp.read_csv_clean(test_path)

    if target_col not in train_raw.columns:
        raise ValueError(f"target column '{target_col}' がtrainにありません。columns={train_raw.columns.tolist()}")
    if id_col not in test_raw.columns:
        print(f"[WARN] testにID列 '{id_col}' がありません。indexをidとして使います。")
        test_raw[id_col] = np.arange(len(test_raw))

    original_report: Dict[str, Any] = {"status": "not_used"}
    if original_path is not None:
        original_raw, _ = hp.read_csv_clean(original_path)
        original_clean, original_report = hp.remove_original_overlap(original_raw, train_raw, test_raw, target_col, id_col)
        if len(original_clean) > 0:
            common_for_train = [c for c in train_raw.columns if c in original_clean.columns]
            original_clean = original_clean[common_for_train].copy()
            train_raw = pd.concat([train_raw, original_clean], axis=0, ignore_index=True)
            print(f"[INFO] original added rows: {len(original_clean):,}")
        else:
            print(f"[INFO] original skipped: {original_report}")

    train_raw, train_clean_report = hp.basic_train_clean(train_raw, target_col)
    test_raw, test_clean_report = hp.basic_test_clean_keep_ids(test_raw)

    exclude = {id_col, target_col}
    baseline_features = [c for c in train_raw.columns if c in test_raw.columns and c not in exclude]
    if not baseline_features:
        raise ValueError("train/testの共通baseline特徴量がありません。")

    train_raw["__is_train"] = 1
    test_raw["__is_train"] = 0
    all_cols = sorted(set(train_raw.columns).union(set(test_raw.columns)))
    all_df = pd.concat([
        train_raw.reindex(columns=all_cols),
        test_raw.reindex(columns=all_cols),
    ], axis=0, ignore_index=True)

    external_meta = hp.load_external_metadata(input_dir, args.metadata_file)
    all_feat, created_features = hp.add_engineered_features(all_df, external_meta)

    train_feat = all_feat[all_feat["__is_train"].eq(1)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)
    test_feat = all_feat[all_feat["__is_train"].eq(0)].drop(columns=["__is_train"], errors="ignore").reset_index(drop=True)

    baseline_features = hp.valid_feature_list(train_feat, test_feat, baseline_features)
    baseline_features = [f for f in baseline_features if hp.is_non_constant_feature(train_feat, f)]
    categorical_all = [c for c in hp.base_categorical_candidates() if c in train_feat.columns and c in test_feat.columns]

    info = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "original_path": str(original_path) if original_path else None,
        "train_mapping": train_mapping,
        "test_mapping": test_mapping,
        "original_report": original_report,
        "train_clean_report": train_clean_report,
        "test_clean_report": test_clean_report,
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "n_baseline_features": int(len(baseline_features)),
        "baseline_features": baseline_features,
        "categorical_all": categorical_all,
        "created_features": created_features,
    }
    return train_feat, test_feat, baseline_features, categorical_all, target_col, id_col, info


# ============================================================
# 3. LightGBM evaluation / prediction
# ============================================================


def lgbm_cv_predict_save(
    hp: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    id_col: str,
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    output_dir: Path,
    name: str,
    submission_name: Optional[str] = None,
) -> Dict[str, Any]:
    """LGBMのCV, OOF保存, full-train prediction, submission作成をまとめて実行。"""
    print(f"\n[STEP] LGBM CV + prediction: {name}")
    model_dir = output_dir / "models" / name
    ensure_dir(model_dir)

    X, y, X_test, cats, impute_report = hp.prepare_X_y(train_df, test_df, features, target_col, categorical_all)
    cv_res = evaluate_lgbm_cv_verbose(
        X, y, features, cats, params, folds, seed, name=name,
        return_oof=True, return_importance=True,
    )
    oof = cv_res["oof"]
    test_preds = hp.train_final_and_predict(
        X, y, X_test, features, cats, params, cv_res.get("best_iteration_mean", 0), seed, model_dir
    )

    train_ids = get_train_ids(train_df, id_col)
    test_ids = get_test_ids(test_df, id_col)
    save_oof(train_ids, y, oof, output_dir / f"{name}_oof.csv")
    save_test_pred(test_ids, test_preds, output_dir / f"{name}_test_pred.csv")
    if submission_name:
        make_submission(test_ids, test_preds, target_col, output_dir / submission_name)

    if "importance" in cv_res:
        cv_res["importance"].to_csv(output_dir / f"{name}_feature_importance.csv", index=False)

    summary = {k: v for k, v in cv_res.items() if k not in {"oof", "importance"}}
    summary.update({
        "name": name,
        "submission": submission_name,
        "test_pred_file": f"{name}_test_pred.csv",
        "oof_file": f"{name}_oof.csv",
        "features_file": f"{name}_features.txt",
    })
    write_json(summary, output_dir / f"{name}_cv_summary.json")
    write_json(impute_report, output_dir / f"{name}_impute_report.json")
    with open(output_dir / f"{name}_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(features))

    print(
        f"[RESULT] {name}: valid_auc={cv_res['valid_auc_mean']:.6f} "
        f"± {cv_res['valid_auc_std']:.6f}, gap={cv_res['overfit_gap_mean']:.6f}, features={len(features)}"
    )
    return {
        "name": name,
        "oof": np.asarray(oof, dtype=float),
        "test": np.asarray(test_preds, dtype=float),
        "auc": float(roc_auc_score(y, oof)),
        "summary": summary,
        "features": list(features),
    }


# ============================================================
# 4. Group CV overfitting check
# ============================================================


def make_group_folds(train_df: pd.DataFrame, y: pd.Series, n_splits: int, seed: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    if "Race" in train_df.columns and "Year" in train_df.columns:
        groups = train_df["Year"].astype(str) + "__" + train_df["Race"].astype(str)
    elif "Race" in train_df.columns:
        groups = train_df["Race"].astype(str)
    else:
        return None

    n_groups = groups.nunique()
    if n_groups < 2:
        return None
    n_splits = min(n_splits, int(n_groups))
    if n_splits < 2:
        return None

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(train_df, y, groups=groups))

    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(train_df, y, groups=groups))


def run_group_cv_check(
    hp: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    params: Dict[str, Any],
    n_splits: int,
    seed: int,
    output_dir: Path,
    normal_cv_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print("\n[STEP] Group CV overfitting check by Race-Year")
    y = train_df[target_col].astype(int)
    group_folds = make_group_folds(train_df, y, n_splits, seed)
    if group_folds is None:
        result = {"status": "skipped", "reason": "Race/Year groups unavailable or too few"}
        write_json(result, output_dir / "overfit_check_report.json")
        print("[WARN] Group CV skipped")
        return result

    X, y, _, cats, _ = hp.prepare_X_y(train_df, test_df, features, target_col, categorical_all)
    res = hp.evaluate_cv(X, y, features, cats, params, group_folds, seed, return_oof=False, return_importance=False)

    rows = []
    for i, (tr_auc, va_auc) in enumerate(zip(res["fold_train_aucs"], res["fold_valid_aucs"]), start=1):
        rows.append({
            "fold": i,
            "train_auc": tr_auc,
            "valid_auc": va_auc,
            "gap": tr_auc - va_auc,
        })
    pd.DataFrame(rows).to_csv(output_dir / "group_cv_report.csv", index=False)

    normal_auc = None
    if normal_cv_summary:
        normal_auc = normal_cv_summary.get("valid_auc_mean")

    result = {
        "status": "ok",
        "group_cv_valid_auc_mean": res["valid_auc_mean"],
        "group_cv_valid_auc_std": res["valid_auc_std"],
        "group_cv_train_auc_mean": res["train_auc_mean"],
        "group_cv_overfit_gap_mean": res["overfit_gap_mean"],
        "normal_cv_valid_auc_mean": normal_auc,
        "normal_minus_group_auc": (float(normal_auc) - res["valid_auc_mean"]) if normal_auc is not None else None,
        "n_group_folds": len(group_folds),
        "fold_valid_aucs": res["fold_valid_aucs"],
        "fold_train_aucs": res["fold_train_aucs"],
    }
    write_json(result, output_dir / "overfit_check_report.json")
    print(
        f"[RESULT] Group CV: valid_auc={res['valid_auc_mean']:.6f} "
        f"gap={res['overfit_gap_mean']:.6f}"
    )
    return result


# ============================================================
# 5. LGBM seed ensemble
# ============================================================


def run_lgbm_seed_ensemble(
    hp: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    id_col: str,
    params: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seeds: List[int],
    output_dir: Path,
    include_individual: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    print("\n[STEP] LGBM seed ensemble")
    y = train_df[target_col].astype(int)
    test_ids = get_test_ids(test_df, id_col)
    train_ids = get_train_ids(train_df, id_col)

    seed_models: Dict[str, Dict[str, Any]] = {}
    oofs = []
    tests = []

    for seed in seeds:
        name = f"lgbm_seed{seed}"
        res = lgbm_cv_predict_save(
            hp, train_df, test_df, features, categorical_all, target_col, id_col,
            params, folds, seed, output_dir, name, submission_name=None,
        )
        seed_models[name] = res
        oofs.append(res["oof"])
        tests.append(res["test"])

    ens_oof = np.mean(np.vstack(oofs), axis=0)
    ens_test = np.mean(np.vstack(tests), axis=0)
    ens_auc = float(roc_auc_score(y, ens_oof))
    save_oof(train_ids, y, ens_oof, output_dir / "lgbm_seed_ensemble_oof.csv")
    save_test_pred(test_ids, ens_test, output_dir / "lgbm_seed_ensemble_test_pred.csv")
    make_submission(test_ids, ens_test, target_col, output_dir / "submission_004_lgbm_seed_ensemble.csv")
    write_json({
        "name": "lgbm_seed_ensemble",
        "auc": ens_auc,
        "seeds": seeds,
        "method": "equal_average",
        "n_models": len(seeds),
    }, output_dir / "lgbm_seed_ensemble_summary.json")
    print(f"[RESULT] lgbm_seed_ensemble AUC={ens_auc:.6f}")

    ensemble_model = {
        "name": "lgbm_seed_ensemble",
        "oof": ens_oof,
        "test": ens_test,
        "auc": ens_auc,
        "summary": {"seeds": seeds, "method": "equal_average"},
        "features": list(features),
    }

    if include_individual:
        return ensemble_model, seed_models
    return ensemble_model, {}


# ============================================================
# 6. CatBoost
# ============================================================


def prepare_catboost_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
):
    cat_features = [c for c in categorical_all if c in features]
    num_features = [c for c in features if c not in cat_features]
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y = train_df[target_col].astype(int).copy()

    for c in num_features:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X_train[c].median()
        if pd.isna(med):
            med = 0.0
        X_train[c] = X_train[c].fillna(med).replace([np.inf, -np.inf], med)
        X_test[c] = X_test[c].fillna(med).replace([np.inf, -np.inf], med)
    for c in cat_features:
        X_train[c] = X_train[c].astype(str).where(~X_train[c].isna(), "__MISSING__")
        X_test[c] = X_test[c].astype(str).where(~X_test[c].isna(), "__MISSING__")
    return X_train, y, X_test, cat_features


def catboost_default_params(seed: int, iterations: int, learning_rate: float, depth: int, n_jobs: int) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": 5.0,
        "random_seed": int(seed),
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": int(n_jobs),
    }


def fit_cat_model(CatBoostClassifier, X_tr, y_tr, X_va=None, y_va=None, cat_features=None, params=None, early_stopping_rounds=150):
    model = CatBoostClassifier(**params)
    if X_va is not None:
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_features,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
    return model


def run_catboost_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    id_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    output_dir: Path,
    name: str = "catboost",
    params_override: Optional[Dict[str, Any]] = None,
    submission_name: Optional[str] = "submission_005_catboost.csv",
) -> Optional[Dict[str, Any]]:
    print(f"\n[STEP] CatBoost model: {name}")
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        print(f"[WARN] CatBoost not installed. skipped. error={e}")
        write_json({"status": "skipped", "reason": str(e)}, output_dir / f"{name}_summary.json")
        return None

    X, y, X_test, cats = prepare_catboost_data(train_df, test_df, features, categorical_all, target_col)
    cat_indices = [X.columns.get_loc(c) for c in cats]
    params = catboost_default_params(seed, iterations, learning_rate, depth, n_jobs)
    if params_override:
        params.update(params_override)
    params["random_seed"] = seed

    oof = np.zeros(len(X), dtype=float)
    train_aucs, valid_aucs, best_iters = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = fit_cat_model(
            CatBoostClassifier,
            X.iloc[tr_idx], y.iloc[tr_idx],
            X.iloc[va_idx], y.iloc[va_idx],
            cat_features=cat_indices,
            params=params,
            early_stopping_rounds=150,
        )
        p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
        oof[va_idx] = p_va
        valid_aucs.append(float(roc_auc_score(y.iloc[va_idx], p_va)))
        train_aucs.append(float(roc_auc_score(y.iloc[tr_idx], p_tr)))
        try:
            best_iters.append(int(model.get_best_iteration() or params["iterations"]))
        except Exception:
            best_iters.append(int(params["iterations"]))
        print(f"[CAT][fold {fold}] valid_auc={valid_aucs[-1]:.6f} train_auc={train_aucs[-1]:.6f}")

    best_iter_mean = float(np.mean(best_iters)) if best_iters else float(iterations)
    final_params = dict(params)
    final_params["iterations"] = max(200, int(best_iter_mean * 1.15))
    final_model = fit_cat_model(
        CatBoostClassifier,
        X, y,
        cat_features=cat_indices,
        params=final_params,
        early_stopping_rounds=150,
    )
    model_dir = output_dir / "models" / name
    ensure_dir(model_dir)
    try:
        final_model.save_model(str(model_dir / f"{name}.cbm"))
    except Exception as e:
        print(f"[WARN] CatBoost model save failed: {e}")
    test_pred = final_model.predict_proba(X_test)[:, 1]

    train_ids = get_train_ids(train_df, id_col)
    test_ids = get_test_ids(test_df, id_col)
    save_oof(train_ids, y, oof, output_dir / f"{name}_oof.csv")
    save_test_pred(test_ids, test_pred, output_dir / f"{name}_test_pred.csv")
    if submission_name:
        make_submission(test_ids, test_pred, target_col, output_dir / submission_name)

    auc = float(roc_auc_score(y, oof))
    summary = {
        "name": name,
        "valid_auc_mean": float(np.mean(valid_aucs)),
        "valid_auc_std": float(np.std(valid_aucs)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "overfit_gap_mean": float(np.mean(np.array(train_aucs) - np.array(valid_aucs))),
        "fold_valid_aucs": valid_aucs,
        "fold_train_aucs": train_aucs,
        "best_iteration_mean": best_iter_mean,
        "final_iterations": final_params["iterations"],
        "params": final_params,
        "features": features,
    }
    write_json(summary, output_dir / f"{name}_summary.json")
    print(f"[RESULT] {name}: AUC={auc:.6f}, gap={summary['overfit_gap_mean']:.6f}")
    return {"name": name, "oof": oof, "test": test_pred, "auc": auc, "summary": summary, "features": features}


# ============================================================
# 7. XGBoost
# ============================================================


def prepare_xgb_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
):
    cat_features = [c for c in categorical_all if c in features]
    num_features = [c for c in features if c not in cat_features]
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y = train_df[target_col].astype(int).copy()

    for c in num_features:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X_train[c].median()
        if pd.isna(med):
            med = 0.0
        X_train[c] = X_train[c].fillna(med).replace([np.inf, -np.inf], med)
        X_test[c] = X_test[c].fillna(med).replace([np.inf, -np.inf], med)
    for c in cat_features:
        combined = pd.concat([
            X_train[c].astype(str).where(~X_train[c].isna(), "__MISSING__"),
            X_test[c].astype(str).where(~X_test[c].isna(), "__MISSING__"),
        ], axis=0)
        codes, uniques = pd.factorize(combined, sort=True)
        X_train[c] = codes[: len(X_train)].astype(np.int32)
        X_test[c] = codes[len(X_train):].astype(np.int32)

    return X_train.astype(np.float32), y, X_test.astype(np.float32)


def xgb_default_params(seed: int, n_jobs: int, n_estimators: int = 2500) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    }


def fit_xgb_model(XGBClassifier, X_tr, y_tr, X_va=None, y_va=None, params=None):
    params = dict(params)
    # newer xgboost accepts early_stopping_rounds in constructor
    params_with_es = dict(params)
    params_with_es["early_stopping_rounds"] = 150
    try:
        model = XGBClassifier(**params_with_es)
        if X_va is not None:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            params_no_es = dict(params)
            model = XGBClassifier(**params_no_es)
            model.fit(X_tr, y_tr, verbose=False)
        return model
    except TypeError:
        model = XGBClassifier(**params)
        if X_va is not None:
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=150, verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)
        return model


def run_xgboost_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    id_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    n_estimators: int,
    output_dir: Path,
    name: str = "xgboost",
    params_override: Optional[Dict[str, Any]] = None,
    submission_name: Optional[str] = "submission_006_xgboost.csv",
) -> Optional[Dict[str, Any]]:
    print(f"\n[STEP] XGBoost model: {name}")
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print(f"[WARN] XGBoost not installed. skipped. error={e}")
        write_json({"status": "skipped", "reason": str(e)}, output_dir / f"{name}_summary.json")
        return None

    X, y, X_test = prepare_xgb_data(train_df, test_df, features, categorical_all, target_col)
    params = xgb_default_params(seed, n_jobs, n_estimators=n_estimators)
    if params_override:
        params.update(params_override)
    params["random_state"] = seed

    oof = np.zeros(len(X), dtype=float)
    train_aucs, valid_aucs, best_iters = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = fit_xgb_model(XGBClassifier, X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx], params=params)
        p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
        oof[va_idx] = p_va
        valid_aucs.append(float(roc_auc_score(y.iloc[va_idx], p_va)))
        train_aucs.append(float(roc_auc_score(y.iloc[tr_idx], p_tr)))
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = params.get("n_estimators", n_estimators)
        best_iters.append(int(best_iter))
        print(f"[XGB][fold {fold}] valid_auc={valid_aucs[-1]:.6f} train_auc={train_aucs[-1]:.6f}")

    best_iter_mean = float(np.mean(best_iters)) if best_iters else float(n_estimators)
    final_params = dict(params)
    final_params["n_estimators"] = max(200, int(best_iter_mean * 1.15))
    final_model = fit_xgb_model(XGBClassifier, X, y, params=final_params)
    model_dir = output_dir / "models" / name
    ensure_dir(model_dir)
    try:
        final_model.save_model(str(model_dir / f"{name}.json"))
    except Exception as e:
        print(f"[WARN] XGBoost model save failed: {e}")
    test_pred = final_model.predict_proba(X_test)[:, 1]

    train_ids = get_train_ids(train_df, id_col)
    test_ids = get_test_ids(test_df, id_col)
    save_oof(train_ids, y, oof, output_dir / f"{name}_oof.csv")
    save_test_pred(test_ids, test_pred, output_dir / f"{name}_test_pred.csv")
    if submission_name:
        make_submission(test_ids, test_pred, target_col, output_dir / submission_name)

    auc = float(roc_auc_score(y, oof))
    summary = {
        "name": name,
        "valid_auc_mean": float(np.mean(valid_aucs)),
        "valid_auc_std": float(np.std(valid_aucs)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "overfit_gap_mean": float(np.mean(np.array(train_aucs) - np.array(valid_aucs))),
        "fold_valid_aucs": valid_aucs,
        "fold_train_aucs": train_aucs,
        "best_iteration_mean": best_iter_mean,
        "final_n_estimators": final_params["n_estimators"],
        "params": final_params,
        "features": features,
    }
    write_json(summary, output_dir / f"{name}_summary.json")
    print(f"[RESULT] {name}: AUC={auc:.6f}, gap={summary['overfit_gap_mean']:.6f}")
    return {"name": name, "oof": oof, "test": test_pred, "auc": auc, "summary": summary, "features": features}


# ============================================================
# 8. correlation and ensemble weights
# ============================================================


def model_registry_to_frames(models: Dict[str, Dict[str, Any]], y: pd.Series):
    valid_models = {}
    for name, obj in models.items():
        if obj is None:
            continue
        oof = np.asarray(obj.get("oof"), dtype=float)
        test = np.asarray(obj.get("test"), dtype=float)
        if len(oof) != len(y) or not np.isfinite(oof).all() or not np.isfinite(test).all():
            continue
        valid_models[name] = obj
    if not valid_models:
        raise ValueError("有効なOOF予測がありません。")
    oof_df = pd.DataFrame({name: np.asarray(obj["oof"], dtype=float) for name, obj in valid_models.items()})
    test_df = pd.DataFrame({name: np.asarray(obj["test"], dtype=float) for name, obj in valid_models.items()})
    score_df = pd.DataFrame([
        {"model": name, "oof_auc": float(roc_auc_score(y, oof_df[name]))}
        for name in oof_df.columns
    ]).sort_values("oof_auc", ascending=False)
    return valid_models, oof_df, test_df, score_df


def save_prediction_correlation(models: Dict[str, Dict[str, Any]], y: pd.Series, output_dir: Path, prefix: str = ""):
    valid_models, oof_df, _, score_df = model_registry_to_frames(models, y)
    corr = oof_df.corr(method="pearson")
    corr_path = output_dir / f"{prefix}prediction_correlation_matrix.csv"
    score_path = output_dir / f"{prefix}model_oof_score_report.csv"
    corr.to_csv(corr_path)
    score_df.to_csv(score_path, index=False)
    print(f"[SAVE] {corr_path}")
    print(f"[SAVE] {score_path}")
    return corr, score_df


def search_ensemble_weights(
    models: Dict[str, Dict[str, Any]],
    y: pd.Series,
    output_dir: Path,
    test_ids: pd.Series,
    target_col: str,
    n_trials: int,
    seed: int,
    auc_tolerance: float,
    prefix: str = "",
    submission_name: str = "submission_007_ensemble.csv",
) -> Dict[str, Any]:
    print(f"\n[STEP] Ensemble weight search: {prefix or 'main'}")
    valid_models, oof_df_all, test_df_all, score_df = model_registry_to_frames(models, y)

    best_auc = float(score_df["oof_auc"].max())
    keep_names = score_df.loc[score_df["oof_auc"] >= best_auc - auc_tolerance, "model"].tolist()
    if not keep_names:
        keep_names = [score_df.iloc[0]["model"]]
    print(f"[INFO] ensemble candidates: {keep_names}")

    oof_mat = oof_df_all[keep_names].values
    test_mat = test_df_all[keep_names].values
    rng = np.random.default_rng(seed)

    candidates: List[Tuple[float, np.ndarray, str]] = []
    n = len(keep_names)

    # single models
    for i, name in enumerate(keep_names):
        w = np.zeros(n)
        w[i] = 1.0
        auc = float(roc_auc_score(y, oof_mat @ w))
        candidates.append((auc, w, f"single_{name}"))

    # equal average
    w_equal = np.ones(n) / n
    candidates.append((float(roc_auc_score(y, oof_mat @ w_equal)), w_equal, "equal"))

    # random Dirichlet search
    if n > 1:
        progress_every = max(1, int(n_trials) // 10)
        current_best = max(c[0] for c in candidates)
        loop_start = time.time()
        for i in range(int(n_trials)):
            alpha = np.ones(n)
            # small bias toward strong models, but still allow diversity
            ranks = score_df.set_index("model").loc[keep_names]["oof_auc"].rank(method="first").values
            alpha = 0.8 + ranks / ranks.max()
            w = rng.dirichlet(alpha)
            auc = float(roc_auc_score(y, oof_mat @ w))
            if auc > current_best:
                current_best = auc
            candidates.append((auc, w, f"random_{i}"))
            if (i + 1) % progress_every == 0 or (i + 1) == int(n_trials):
                pct = 100.0 * (i + 1) / max(1, int(n_trials))
                elapsed = time.time() - loop_start
                eta = elapsed / (i + 1) * (int(n_trials) - i - 1) if i + 1 > 0 else 0.0
                print(
                    f"[ENSEMBLE][{prefix or 'main'}] weight search {i+1}/{int(n_trials)} "
                    f"({pct:.0f}%) | current_best_auc={current_best:.6f} | "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                    flush=True,
                )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_auc2, best_w, best_source = candidates[0]
    best_oof = oof_mat @ best_w
    best_test = test_mat @ best_w

    rows = []
    for rank, (auc, w, source) in enumerate(candidates[:200], start=1):
        row = {"rank": rank, "auc": auc, "source": source}
        for name, weight in zip(keep_names, w):
            row[f"weight_{name}"] = float(weight)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}ensemble_weight_search_report.csv", index=False)

    weights = {name: float(w) for name, w in zip(keep_names, best_w)}
    result = {
        "status": "ok",
        "best_auc": best_auc2,
        "best_source": best_source,
        "weights": weights,
        "candidates": keep_names,
        "auc_tolerance": auc_tolerance,
        "n_trials": n_trials,
    }
    write_json(result, output_dir / f"{prefix}best_ensemble_weights.json")

    save_oof(None, y, best_oof, output_dir / f"{prefix}ensemble_oof.csv")
    save_test_pred(test_ids, best_test, output_dir / f"{prefix}ensemble_test_pred.csv")
    make_submission(test_ids, best_test, target_col, output_dir / submission_name)
    print(f"[RESULT] ensemble AUC={best_auc2:.6f}, weights={weights}")
    return {"name": f"{prefix}ensemble", "oof": best_oof, "test": best_test, "auc": best_auc2, "weights": weights, "summary": result}


# ============================================================
# 9. optional Optuna for CatBoost
# ============================================================


def tune_catboost_with_optuna(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_all: List[str],
    target_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    n_trials: int,
    base_iterations: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    if n_trials <= 0:
        return None
    if optuna is None:
        print("[WARN] optuna not installed. CatBoost tuning skipped.")
        return None
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        print(f"[WARN] CatBoost tuning skipped. error={e}")
        return None

    print("\n[STEP] final Optuna: CatBoost")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y, _, cats = prepare_catboost_data(train_df, test_df, features, categorical_all, target_col)
    cat_indices = [X.columns.get_loc(c) for c in cats]

    def objective(trial):
        params = catboost_default_params(seed, base_iterations, 0.03, 8, n_jobs)
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        })
        valid_aucs = []
        train_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            model = fit_cat_model(
                CatBoostClassifier,
                X.iloc[tr_idx], y.iloc[tr_idx],
                X.iloc[va_idx], y.iloc[va_idx],
                cat_features=cat_indices,
                params=params,
                early_stopping_rounds=150,
            )
            p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
            p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
            valid_aucs.append(float(roc_auc_score(y.iloc[va_idx], p_va)))
            train_aucs.append(float(roc_auc_score(y.iloc[tr_idx], p_tr)))
        trial.set_user_attr("train_auc_mean", float(np.mean(train_aucs)))
        trial.set_user_attr("overfit_gap_mean", float(np.mean(np.array(train_aucs) - np.array(valid_aucs))))
        return float(np.mean(valid_aucs))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="final_catboost")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = catboost_default_params(seed, base_iterations, 0.03, 8, n_jobs)
    best_params.update(study.best_params)
    summary = {"status": "ok", "best_value": float(study.best_value), "best_params": best_params, "n_trials": len(study.trials)}
    write_json(summary, output_dir / "final_catboost_best_params.json")
    return best_params


# ============================================================
# 10. argument parser / main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 PitNextLap stage2 ensemble pipeline")
    parser.add_argument("--input-dir", type=str, default="data", help="train/test/original CSVがあるフォルダ")
    parser.add_argument("--baseline-run-dir", type=str, default="runs/f1_lgbm_ab_001", help="initial_best_params.json があるbaseline出力フォルダ")
    parser.add_argument("--output-dir", type=str, default="runs/f1_stage2_ensemble", help="出力先フォルダ")
    parser.add_argument("--helper-script", type=str, default="auto", help="f1_pit_lgbm_feature_ab.py のパス。auto可")

    parser.add_argument("--train-file", type=str, default="auto")
    parser.add_argument("--test-file", type=str, default="auto")
    parser.add_argument("--original-file", type=str, default="auto", help="noneでoriginal追加なし")
    parser.add_argument("--metadata-file", type=str, default="auto", help="race_metadata.csv。noneで無効")
    parser.add_argument("--target", type=str, default="PitNextLap")
    parser.add_argument("--id-col", type=str, default="id")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--max-gap-increase", type=float, default=0.015)
    parser.add_argument("--min-features", type=int, default=8)

    parser.add_argument("--lgbm-seeds", type=str, default="42,777,2024,3407,1001")
    parser.add_argument("--include-individual-lgbm-seeds-in-ensemble", action="store_true")

    parser.add_argument("--run-catboost", action="store_true", help="CatBoostを実行する。未インストールならskip")
    parser.add_argument("--run-xgboost", action="store_true", help="XGBoostを実行する。未インストールならskip")
    parser.add_argument("--catboost-iterations", type=int, default=2500)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--xgboost-n-estimators", type=int, default=2000)

    parser.add_argument("--ensemble-weight-trials", type=int, default=10000)
    parser.add_argument("--ensemble-auc-tolerance", type=float, default=0.006, help="best AUCからこの範囲内のモデルをweight探索に使う")

    parser.add_argument("--final-lgbm-trials", type=int, default=50)
    parser.add_argument("--final-catboost-trials", type=int, default=20)
    parser.add_argument("--final-xgboost-trials", type=int, default=0, help="未実装に近いので通常0推奨")
    parser.add_argument("--final-weight-threshold", type=float, default=0.10, help="この重み以上ならfinal Optuna候補")

    parser.add_argument("--skip-ab", action="store_true", help="既存selected_features_after_ab.txtを使う")
    parser.add_argument("--skip-pruning", action="store_true", help="既存selected_features_after_pruning.txtを使う")
    parser.add_argument("--skip-final-optuna", action="store_true")
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    output_dir = Path(args.output_dir)
    baseline_run_dir = Path(args.baseline_run_dir)
    ensure_dir(output_dir)

    print_banner("F1 Pit Prediction Stage2 Ensemble Pipeline")
    print(f"[CONFIG] input_dir={args.input_dir}", flush=True)
    print(f"[CONFIG] baseline_run_dir={args.baseline_run_dir}", flush=True)
    print(f"[CONFIG] output_dir={args.output_dir}", flush=True)
    print(f"[CONFIG] n_folds={args.n_folds}, seed={args.seed}, n_jobs={args.n_jobs}", flush=True)
    print(f"[CONFIG] catboost={args.run_catboost}, xgboost={args.run_xgboost}, skip_final_optuna={args.skip_final_optuna}", flush=True)
    print("[PLAN] 1 baseline -> 2 AB -> 3 pruning -> 4 GroupCV -> 5 seed ensemble -> 6 Cat/XGB -> 7 corr -> 8 weights -> 9 final", flush=True)

    if lgb is None:
        raise ImportError("lightgbm が見つかりません。pip install lightgbm を実行してください。")

    hp = load_helper(args.helper_script)
    seed = int(args.seed)

    train_df, test_df, baseline_features, categorical_all, target_col, id_col, data_info = reconstruct_dataset(args, hp)
    write_json(data_info, output_dir / "stage2_data_reconstruction_report.json")
    pd.DataFrame(data_info.get("created_features", [])).to_csv(output_dir / "engineered_feature_catalog.csv", index=False)

    y = train_df[target_col].astype(int)
    test_ids = get_test_ids(test_df, id_col)

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(train_df, y))

    initial_params = load_params_from_run(baseline_run_dir, seed=seed, n_jobs=args.n_jobs, hp=hp)
    write_json(initial_params, output_dir / "baseline_best_params_used.json")

    models: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Any] = {}

    # ① baseline submission
    baseline_model = lgbm_cv_predict_save(
        hp, train_df, test_df, baseline_features, categorical_all, target_col, id_col,
        initial_params, folds, seed, output_dir,
        name="baseline", submission_name="submission_001_baseline.csv",
    )
    models["baseline"] = baseline_model
    summaries["baseline"] = baseline_model["summary"]

    # ② AB feature selection
    if args.skip_ab and (output_dir / "selected_features_after_ab.txt").exists():
        selected_after_ab = [x.strip() for x in (output_dir / "selected_features_after_ab.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"[INFO] AB skipped. loaded selected_after_ab: {len(selected_after_ab)} features")
    else:
        selected_after_ab, ab_result, ab_report = hp.run_ab_feature_selection(
            train_df, test_df, target_col, baseline_features, categorical_all,
            initial_params, folds, seed, args.min_delta, args.max_gap_increase, output_dir,
        )
        with open(output_dir / "selected_features_after_ab.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(selected_after_ab))

    ab_model = lgbm_cv_predict_save(
        hp, train_df, test_df, selected_after_ab, categorical_all, target_col, id_col,
        initial_params, folds, seed, output_dir,
        name="ab_selected", submission_name="submission_002_ab_selected.csv",
    )
    models["ab_selected"] = ab_model
    summaries["ab_selected"] = ab_model["summary"]

    # ③ backward elimination
    pruning_file = output_dir / "selected_features_after_pruning.txt"
    alt_pruning_file = output_dir / "selected_features_after_elimination.txt"
    if args.skip_pruning and (pruning_file.exists() or alt_pruning_file.exists()):
        p = pruning_file if pruning_file.exists() else alt_pruning_file
        selected_after_pruning = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"[INFO] pruning skipped. loaded selected_after_pruning: {len(selected_after_pruning)} features")
        pruning_result = None
    else:
        selected_after_pruning, pruning_result, pruning_report, imp_after_pruning = hp.run_backward_elimination(
            train_df, test_df, target_col, selected_after_ab, categorical_all,
            initial_params, folds, seed, args.min_delta, args.max_gap_increase,
            args.min_features, output_dir,
        )
        with open(pruning_file, "w", encoding="utf-8") as f:
            f.write("\n".join(selected_after_pruning))

    pruned_model = lgbm_cv_predict_save(
        hp, train_df, test_df, selected_after_pruning, categorical_all, target_col, id_col,
        initial_params, folds, seed, output_dir,
        name="pruned", submission_name="submission_003_pruned.csv",
    )
    models["pruned"] = pruned_model
    summaries["pruned"] = pruned_model["summary"]

    # ④ Group CV check
    group_cv_result = run_group_cv_check(
        hp, train_df, test_df, selected_after_pruning, categorical_all, target_col,
        initial_params, args.n_folds, seed, output_dir,
        normal_cv_summary=pruned_model["summary"],
    )
    summaries["group_cv_check"] = group_cv_result

    # ⑤ LGBM seed ensemble
    lgbm_seeds = parse_seed_list(args.lgbm_seeds)
    lgbm_seed_ens, individual_seed_models = run_lgbm_seed_ensemble(
        hp, train_df, test_df, selected_after_pruning, categorical_all,
        target_col, id_col, initial_params, folds, lgbm_seeds, output_dir,
        include_individual=args.include_individual_lgbm_seeds_in_ensemble,
    )
    models["lgbm_seed_ensemble"] = lgbm_seed_ens
    if args.include_individual_lgbm_seeds_in_ensemble:
        models.update(individual_seed_models)
    summaries["lgbm_seed_ensemble"] = lgbm_seed_ens["summary"]

    # ⑥ CatBoost / XGBoost
    cat_model = None
    if args.run_catboost:
        cat_model = run_catboost_model(
            train_df, test_df, selected_after_pruning, categorical_all, target_col, id_col,
            folds, seed, args.n_jobs, args.catboost_iterations, args.catboost_learning_rate,
            args.catboost_depth, output_dir, name="catboost", submission_name="submission_005_catboost.csv",
        )
        if cat_model is not None:
            models["catboost"] = cat_model
            summaries["catboost"] = cat_model["summary"]
    else:
        print("[INFO] CatBoost skipped. use --run-catboost to enable.")

    xgb_model = None
    if args.run_xgboost:
        xgb_model = run_xgboost_model(
            train_df, test_df, selected_after_pruning, categorical_all, target_col, id_col,
            folds, seed, args.n_jobs, args.xgboost_n_estimators, output_dir,
            name="xgboost", submission_name="submission_006_xgboost.csv",
        )
        if xgb_model is not None:
            models["xgboost"] = xgb_model
            summaries["xgboost"] = xgb_model["summary"]
    else:
        print("[INFO] XGBoost skipped. use --run-xgboost to enable.")

    # ⑦ prediction correlation
    corr, score_df = save_prediction_correlation(models, y, output_dir, prefix="")

    # ⑧ ensemble weight search
    ensemble_model = search_ensemble_weights(
        models, y, output_dir, test_ids, target_col,
        n_trials=args.ensemble_weight_trials, seed=seed,
        auc_tolerance=args.ensemble_auc_tolerance,
        prefix="", submission_name="submission_007_ensemble.csv",
    )
    models["ensemble"] = ensemble_model
    summaries["ensemble"] = ensemble_model["summary"]

    # ⑨ final Optuna for models that actually helped
    final_models: Dict[str, Dict[str, Any]] = {}
    if not args.skip_final_optuna:
        weights = ensemble_model.get("weights", {})
        lgbm_weight = max(
            float(weights.get("pruned", 0.0)),
            float(weights.get("ab_selected", 0.0)),
            float(weights.get("baseline", 0.0)),
            float(weights.get("lgbm_seed_ensemble", 0.0)),
        )
        cat_weight = float(weights.get("catboost", 0.0))
        xgb_weight = float(weights.get("xgboost", 0.0))

        if args.final_lgbm_trials > 0 and lgbm_weight >= args.final_weight_threshold:
            print("\n[STEP] final Optuna: LGBM")
            X_pruned, y2, _, cats_pruned, _ = hp.prepare_X_y(train_df, test_df, selected_after_pruning, target_col, categorical_all)
            final_lgbm_params, final_lgbm_summary = hp.tune_lgbm_with_optuna(
                X_pruned, y2, selected_after_pruning, cats_pruned,
                folds, seed=seed, n_jobs=args.n_jobs, n_trials=args.final_lgbm_trials,
                study_name="final_lgbm_pruned",
            )
            write_json(final_lgbm_summary, output_dir / "final_lgbm_optuna_summary.json")
            write_json(final_lgbm_params, output_dir / "final_lgbm_best_params.json")
            final_lgbm_model = lgbm_cv_predict_save(
                hp, train_df, test_df, selected_after_pruning, categorical_all, target_col, id_col,
                final_lgbm_params, folds, seed, output_dir,
                name="final_lgbm", submission_name="submission_008a_final_lgbm.csv",
            )
            final_models["final_lgbm"] = final_lgbm_model
        else:
            print(f"[INFO] final LGBM Optuna skipped. weight={lgbm_weight:.4f}, trials={args.final_lgbm_trials}")

        if args.final_catboost_trials > 0 and cat_weight >= args.final_weight_threshold:
            final_cat_params = tune_catboost_with_optuna(
                train_df, test_df, selected_after_pruning, categorical_all, target_col,
                folds, seed, args.n_jobs, args.final_catboost_trials,
                args.catboost_iterations, output_dir,
            )
            if final_cat_params is not None:
                final_cat_model = run_catboost_model(
                    train_df, test_df, selected_after_pruning, categorical_all, target_col, id_col,
                    folds, seed, args.n_jobs, args.catboost_iterations, args.catboost_learning_rate,
                    args.catboost_depth, output_dir, name="final_catboost",
                    params_override=final_cat_params, submission_name="submission_008b_final_catboost.csv",
                )
                if final_cat_model is not None:
                    final_models["final_catboost"] = final_cat_model
        else:
            print(f"[INFO] final CatBoost Optuna skipped. weight={cat_weight:.4f}, trials={args.final_catboost_trials}")

        if args.final_xgboost_trials > 0 and xgb_weight >= args.final_weight_threshold:
            print("[WARN] final XGBoost Optuna is not fully implemented in this script. skipped by design.")
        else:
            print(f"[INFO] final XGBoost Optuna skipped. weight={xgb_weight:.4f}, trials={args.final_xgboost_trials}")
    else:
        print("[INFO] final Optuna skipped by --skip-final-optuna")

    # final ensemble: original candidates + tuned candidates
    if final_models:
        final_candidate_models = dict(models)
        final_candidate_models.update(final_models)
        save_prediction_correlation(final_candidate_models, y, output_dir, prefix="final_")
        final_ensemble = search_ensemble_weights(
            final_candidate_models, y, output_dir, test_ids, target_col,
            n_trials=args.ensemble_weight_trials, seed=seed + 2027,
            auc_tolerance=args.ensemble_auc_tolerance,
            prefix="final_", submission_name="submission_008_final_ensemble.csv",
        )
        summaries["final_ensemble"] = final_ensemble["summary"]
    else:
        # final tuned modelがない場合は、submission_007をそのままfinal扱いとして複製
        final_test = ensemble_model["test"]
        make_submission(test_ids, final_test, target_col, output_dir / "submission_008_final_ensemble.csv")
        summaries["final_ensemble"] = {
            "status": "copied_from_submission_007",
            "reason": "no final tuned models were created",
            "source_auc": ensemble_model["auc"],
            "weights": ensemble_model.get("weights", {}),
        }

    elapsed = time.time() - start
    final_summary = {
        "elapsed_seconds": elapsed,
        "args": vars(args),
        "data_info": {k: v for k, v in data_info.items() if k != "created_features"},
        "baseline_features": baseline_features,
        "selected_features_after_ab": selected_after_ab,
        "selected_features_after_pruning": selected_after_pruning,
        "model_summaries": summaries,
        "outputs": {
            "submission_001_baseline": str(output_dir / "submission_001_baseline.csv"),
            "submission_002_ab_selected": str(output_dir / "submission_002_ab_selected.csv"),
            "submission_003_pruned": str(output_dir / "submission_003_pruned.csv"),
            "submission_004_lgbm_seed_ensemble": str(output_dir / "submission_004_lgbm_seed_ensemble.csv"),
            "submission_005_catboost": str(output_dir / "submission_005_catboost.csv"),
            "submission_006_xgboost": str(output_dir / "submission_006_xgboost.csv"),
            "submission_007_ensemble": str(output_dir / "submission_007_ensemble.csv"),
            "submission_008_final_ensemble": str(output_dir / "submission_008_final_ensemble.csv"),
            "run_summary": str(output_dir / "stage2_run_summary.json"),
        },
    }
    write_json(final_summary, output_dir / "stage2_run_summary.json")

    print("\n[DONE] stage2 ensemble pipeline finished")
    print(f"elapsed: {elapsed/3600:.2f} hours")
    print(f"output_dir: {output_dir}")
    print(f"final submission: {output_dir / 'submission_008_final_ensemble.csv'}")
    print(f"summary: {output_dir / 'stage2_run_summary.json'}")


if __name__ == "__main__":
    main()
