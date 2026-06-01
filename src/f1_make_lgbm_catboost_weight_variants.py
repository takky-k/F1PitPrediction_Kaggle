#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_make_lgbm_catboost_weight_variants.py

Purpose
-------
Use already-created OOF/test predictions from:
  - LGBM AB-selected model
  - CatBoost model
and create several LGBM+CatBoost weighted-ensemble submission files.

This script does NOT retrain models, so it should finish in seconds to a few minutes.
It is designed for the final few hours of a Kaggle competition when the current best
submission is already an LGBM+CatBoost ensemble and you want a small number of
nearby blend variants without wasting time on expensive training.

Expected project structure
--------------------------
C:\\Projects\\F1PitPrediction
├─ runs
│  ├─ f1_stage2_ensemble
│  │  ├─ ab_selected_oof.csv
│  │  ├─ ab_selected_test_pred.csv
│  │  └─ submission_002_ab_selected.csv
│  └─ f1_lgbm_catboost_ensemble
│     ├─ catboost_oof.csv
│     ├─ catboost_test_pred.csv
│     └─ submission_005_catboost.csv

Example
-------
python .\\src\\f1_make_lgbm_catboost_weight_variants.py `
  --stage2-run-dir .\\runs\\f1_stage2_ensemble `
  --catboost-run-dir .\\runs\\f1_lgbm_catboost_ensemble `
  --output-dir .\\runs\\f1_weight_variants

Outputs
-------
- submission_prob_lgbm_0p350_cat_0p650.csv etc.
- submission_rank_lgbm_0p350_cat_0p650.csv etc.
- ensemble_weight_grid.csv
- suggested_submission_list.csv
- best_weights.json
- prediction_correlation_matrix.csv
- model_oof_score_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# -----------------------------
# Utility
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"weight must be between 0 and 1: {v}")
        vals.append(v)
    return sorted(set(vals))


def fmt_weight(w: float) -> str:
    return f"{w:.3f}".replace(".", "p")


def clip_pred(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def rank01(x: np.ndarray) -> np.ndarray:
    """Convert predictions to percentile ranks in [0, 1]."""
    s = pd.Series(np.asarray(x, dtype=float))
    r = s.rank(method="average", pct=True).to_numpy(dtype=float)
    return np.clip(r, 0.0, 1.0)


def existing_file(paths: Iterable[Path], label: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n  ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n  {tried}")


def find_prediction_col(df: pd.DataFrame, target: str, preferred: List[str], context: str) -> str:
    """Find prediction column robustly across files made by earlier scripts."""
    cols = list(df.columns)
    for c in preferred:
        if c in cols:
            return c
    # test/submission files often use target column name as prediction
    if target in cols and context in {"test", "submission"}:
        return target
    # oof files often have target + one prediction column
    numeric_cols = []
    for c in cols:
        if c.lower() in {"id", target.lower()}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.95:
            numeric_cols.append(c)
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    # fallback: look for names containing pred/proba/oof
    for c in cols:
        low = c.lower()
        if any(k in low for k in ["pred", "proba", "oof", "prob"]):
            return c
    raise ValueError(f"Could not infer prediction column for {context}. Columns={cols}")


def find_target_col(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    # common fallback: the only binary-looking numeric column not id/pred
    candidates: List[str] = []
    for c in df.columns:
        if c.lower() == "id":
            continue
        low = c.lower()
        if any(k in low for k in ["pred", "proba", "oof", "prob"]):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        vals = set(s.dropna().unique().tolist())
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not infer target column. Expected '{target}'. Columns={list(df.columns)}")


def load_oof(path: Path, target: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pred_col = find_prediction_col(
        df,
        target=target,
        preferred=["oof_pred", "pred", "prediction", "prob", "proba", "PitNextLap_pred"],
        context="oof",
    )
    y_col = find_target_col(df, target)
    out = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "y": pd.to_numeric(df[y_col], errors="coerce").astype(int),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    if id_col in df.columns:
        out[id_col] = df[id_col].values
    return out


def load_test_pred(path: Path, target: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pred_col = find_prediction_col(
        df,
        target=target,
        preferred=["test_pred", "pred", "prediction", "prob", "proba", "PitNextLap", "oof_pred"],
        context="test",
    )
    out = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    if id_col in df.columns:
        out[id_col] = df[id_col].values
    else:
        out[id_col] = np.arange(len(df))
    return out


def align_oof(lgbm: pd.DataFrame, cat: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in lgbm.columns and id_col in cat.columns and lgbm[id_col].is_unique and cat[id_col].is_unique:
        merged = lgbm[[id_col, "y", "pred"]].rename(columns={"pred": "lgbm"}).merge(
            cat[[id_col, "y", "pred"]].rename(columns={"pred": "cat", "y": "y_cat"}),
            on=id_col,
            how="inner",
        )
        if len(merged) != len(lgbm):
            print(f"[WARN] OOF merge length differs: lgbm={len(lgbm)} merged={len(merged)}")
        if not np.array_equal(merged["y"].to_numpy(), merged["y_cat"].to_numpy()):
            raise ValueError("OOF target mismatch after ID merge.")
        return merged.drop(columns=["y_cat"])

    if len(lgbm) != len(cat):
        raise ValueError(f"OOF length mismatch and no reliable ID merge: lgbm={len(lgbm)} cat={len(cat)}")
    if not np.array_equal(lgbm["y"].to_numpy(), cat["y"].to_numpy()):
        raise ValueError("OOF target mismatch by row order.")
    return pd.DataFrame({
        id_col: lgbm[id_col].values if id_col in lgbm.columns else np.arange(len(lgbm)),
        "y": lgbm["y"].values,
        "lgbm": lgbm["pred"].values,
        "cat": cat["pred"].values,
    })


def align_test(lgbm: pd.DataFrame, cat: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in lgbm.columns and id_col in cat.columns and lgbm[id_col].is_unique and cat[id_col].is_unique:
        merged = lgbm[[id_col, "pred"]].rename(columns={"pred": "lgbm"}).merge(
            cat[[id_col, "pred"]].rename(columns={"pred": "cat"}),
            on=id_col,
            how="inner",
        )
        if len(merged) != len(lgbm):
            print(f"[WARN] test merge length differs: lgbm={len(lgbm)} merged={len(merged)}")
        return merged

    if len(lgbm) != len(cat):
        raise ValueError(f"test length mismatch and no reliable ID merge: lgbm={len(lgbm)} cat={len(cat)}")
    return pd.DataFrame({
        id_col: lgbm[id_col].values if id_col in lgbm.columns else np.arange(len(lgbm)),
        "lgbm": lgbm["pred"].values,
        "cat": cat["pred"].values,
    })


def make_submission(ids: np.ndarray, pred: np.ndarray, target: str, out_path: Path, eps: float) -> None:
    sub = pd.DataFrame({"id": ids, target: clip_pred(pred, eps)})
    sub.to_csv(out_path, index=False)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LGBM+CatBoost final weight variant submissions without retraining.")
    p.add_argument("--stage2-run-dir", type=str, default=r".\runs\f1_stage2_ensemble",
                   help="Directory containing ab_selected_oof.csv and ab_selected_test_pred.csv")
    p.add_argument("--catboost-run-dir", type=str, default=r".\runs\f1_lgbm_catboost_ensemble",
                   help="Directory containing catboost_oof.csv and catboost_test_pred.csv")
    p.add_argument("--output-dir", type=str, default=r".\runs\f1_weight_variants",
                   help="Output directory for new submissions/reports")
    p.add_argument("--target", type=str, default="PitNextLap")
    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--grid-size", type=int, default=1001,
                   help="OOF weight grid size. 1001 means 0.000, 0.001, ..., 1.000")
    p.add_argument("--manual-lgbm-weights", type=str, default="0.25,0.30,0.35,0.40,0.427,0.45,0.50,0.55,0.60",
                   help="Comma-separated LGBM weights to produce probability-blend submissions")
    p.add_argument("--rank-weights", type=str, default="0.35,0.40,0.427,0.45,0.50",
                   help="Comma-separated LGBM weights to produce rank-blend submissions")
    p.add_argument("--top-n-oof-submissions", type=int, default=5,
                   help="Also create probability submissions for top-N OOF grid weights")
    p.add_argument("--clip", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    stage2_dir = Path(args.stage2_run_dir)
    cat_dir = Path(args.catboost_run_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    print("=" * 80)
    print("F1 Pit Prediction: LGBM + CatBoost final weight variants")
    print("No model training is performed. Expected runtime: seconds to a few minutes.")
    print("=" * 80)
    print(f"[CONFIG] stage2_run_dir   = {stage2_dir}")
    print(f"[CONFIG] catboost_run_dir = {cat_dir}")
    print(f"[CONFIG] output_dir       = {out_dir}")
    print(f"[CONFIG] target           = {args.target}")

    lgbm_oof_path = existing_file([
        stage2_dir / "ab_selected_oof.csv",
        stage2_dir / "lgbm_ab_selected_oof.csv",
    ], "LGBM AB-selected OOF")
    lgbm_test_path = existing_file([
        stage2_dir / "ab_selected_test_pred.csv",
        stage2_dir / "submission_002_ab_selected.csv",
    ], "LGBM AB-selected test predictions")
    cat_oof_path = existing_file([
        cat_dir / "catboost_oof.csv",
        cat_dir / "catboost_oof_predictions.csv",
    ], "CatBoost OOF")
    cat_test_path = existing_file([
        cat_dir / "catboost_test_pred.csv",
        cat_dir / "submission_005_catboost.csv",
    ], "CatBoost test predictions")

    print(f"[LOAD] lgbm_oof  = {lgbm_oof_path}")
    print(f"[LOAD] lgbm_test = {lgbm_test_path}")
    print(f"[LOAD] cat_oof   = {cat_oof_path}")
    print(f"[LOAD] cat_test  = {cat_test_path}")

    lgbm_oof = load_oof(lgbm_oof_path, args.target, args.id_col)
    cat_oof = load_oof(cat_oof_path, args.target, args.id_col)
    lgbm_test = load_test_pred(lgbm_test_path, args.target, args.id_col)
    cat_test = load_test_pred(cat_test_path, args.target, args.id_col)

    oof = align_oof(lgbm_oof, cat_oof, args.id_col)
    test = align_test(lgbm_test, cat_test, args.id_col)

    y = oof["y"].to_numpy(dtype=int)
    lgbm_oof_pred = clip_pred(oof["lgbm"].to_numpy(), args.clip)
    cat_oof_pred = clip_pred(oof["cat"].to_numpy(), args.clip)
    lgbm_test_pred = clip_pred(test["lgbm"].to_numpy(), args.clip)
    cat_test_pred = clip_pred(test["cat"].to_numpy(), args.clip)

    lgbm_auc = roc_auc_score(y, lgbm_oof_pred)
    cat_auc = roc_auc_score(y, cat_oof_pred)
    corr = float(np.corrcoef(lgbm_oof_pred, cat_oof_pred)[0, 1])
    print(f"[OOF] LGBM AB-selected AUC = {lgbm_auc:.6f}")
    print(f"[OOF] CatBoost AUC         = {cat_auc:.6f}")
    print(f"[OOF] prediction corr      = {corr:.6f}")

    # Reports for base models
    score_report = pd.DataFrame([
        {"model": "lgbm_ab_selected", "oof_auc": lgbm_auc},
        {"model": "catboost", "oof_auc": cat_auc},
    ])
    score_report.to_csv(out_dir / "model_oof_score_report.csv", index=False)
    pd.DataFrame(
        [[1.0, corr], [corr, 1.0]],
        index=["lgbm_ab_selected", "catboost"],
        columns=["lgbm_ab_selected", "catboost"],
    ).to_csv(out_dir / "prediction_correlation_matrix.csv")

    # Weight grid search on OOF
    print("[STEP] OOF weight grid search")
    grid = np.linspace(0.0, 1.0, int(args.grid_size))
    rows: List[Dict] = []
    for i, w_lgbm in enumerate(grid):
        pred = w_lgbm * lgbm_oof_pred + (1.0 - w_lgbm) * cat_oof_pred
        auc = roc_auc_score(y, pred)
        rows.append({
            "blend_type": "probability",
            "lgbm_weight": float(w_lgbm),
            "catboost_weight": float(1.0 - w_lgbm),
            "oof_auc": float(auc),
        })
    grid_df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    grid_df.to_csv(out_dir / "ensemble_weight_grid.csv", index=False)
    best_prob = grid_df.iloc[0].to_dict()
    print(
        f"[BEST][prob] auc={best_prob['oof_auc']:.6f} "
        f"lgbm={best_prob['lgbm_weight']:.4f} cat={best_prob['catboost_weight']:.4f}"
    )

    # Rank blend grid for diagnostics. Rank blends can sometimes help when calibration differs.
    print("[STEP] rank-blend diagnostics")
    lgbm_oof_rank = rank01(lgbm_oof_pred)
    cat_oof_rank = rank01(cat_oof_pred)
    lgbm_test_rank = rank01(lgbm_test_pred)
    cat_test_rank = rank01(cat_test_pred)

    rank_rows: List[Dict] = []
    rank_weights = parse_float_list(args.rank_weights)
    for w_lgbm in rank_weights:
        pred = w_lgbm * lgbm_oof_rank + (1.0 - w_lgbm) * cat_oof_rank
        auc = roc_auc_score(y, pred)
        rank_rows.append({
            "blend_type": "rank",
            "lgbm_weight": float(w_lgbm),
            "catboost_weight": float(1.0 - w_lgbm),
            "oof_auc": float(auc),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    rank_df.to_csv(out_dir / "rank_blend_report.csv", index=False)
    if len(rank_df):
        best_rank = rank_df.iloc[0].to_dict()
        print(
            f"[BEST][rank] auc={best_rank['oof_auc']:.6f} "
            f"lgbm={best_rank['lgbm_weight']:.4f} cat={best_rank['catboost_weight']:.4f}"
        )

    # Build list of probability submissions:
    manual_weights = parse_float_list(args.manual_lgbm_weights)
    top_weights = grid_df.head(max(0, int(args.top_n_oof_submissions)))["lgbm_weight"].round(6).tolist()
    all_prob_weights = sorted(set([round(float(w), 6) for w in manual_weights + top_weights]))

    suggestion_rows: List[Dict] = []

    print("[STEP] writing probability-blend submissions")
    for w_lgbm in all_prob_weights:
        w_cat = 1.0 - w_lgbm
        oof_pred = w_lgbm * lgbm_oof_pred + w_cat * cat_oof_pred
        test_pred = w_lgbm * lgbm_test_pred + w_cat * cat_test_pred
        auc = roc_auc_score(y, oof_pred)
        fname = f"submission_prob_lgbm_{fmt_weight(w_lgbm)}_cat_{fmt_weight(w_cat)}.csv"
        make_submission(test[args.id_col].to_numpy(), test_pred, args.target, out_dir / fname, args.clip)
        suggestion_rows.append({
            "file": fname,
            "blend_type": "probability",
            "lgbm_weight": float(w_lgbm),
            "catboost_weight": float(w_cat),
            "oof_auc": float(auc),
            "note": "manual/top-oof probability blend",
        })
        print(f"[SAVE] {fname:55s} oof_auc={auc:.6f}")

    print("[STEP] writing rank-blend submissions")
    for w_lgbm in rank_weights:
        w_cat = 1.0 - w_lgbm
        oof_pred = w_lgbm * lgbm_oof_rank + w_cat * cat_oof_rank
        test_pred = w_lgbm * lgbm_test_rank + w_cat * cat_test_rank
        auc = roc_auc_score(y, oof_pred)
        fname = f"submission_rank_lgbm_{fmt_weight(w_lgbm)}_cat_{fmt_weight(w_cat)}.csv"
        make_submission(test[args.id_col].to_numpy(), test_pred, args.target, out_dir / fname, args.clip)
        suggestion_rows.append({
            "file": fname,
            "blend_type": "rank",
            "lgbm_weight": float(w_lgbm),
            "catboost_weight": float(w_cat),
            "oof_auc": float(auc),
            "note": "rank blend; useful if public LB likes ordering more than calibration",
        })
        print(f"[SAVE] {fname:55s} oof_auc={auc:.6f}")

    suggestions = pd.DataFrame(suggestion_rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    suggestions.to_csv(out_dir / "suggested_submission_list.csv", index=False)

    # Copy exact OOF best into a canonical filename for convenience
    best_w = float(best_prob["lgbm_weight"])
    best_cat = 1.0 - best_w
    best_test_pred = best_w * lgbm_test_pred + best_cat * cat_test_pred
    make_submission(test[args.id_col].to_numpy(), best_test_pred, args.target, out_dir / "submission_best_oof_probability_blend.csv", args.clip)

    # Public-LB-oriented shortlist near the previously good region.
    # These are just copies with clearer names if the corresponding weights exist.
    public_shortlist = [0.35, 0.40, 0.45, 0.50]
    for w_lgbm in public_shortlist:
        w_cat = 1.0 - w_lgbm
        pred = w_lgbm * lgbm_test_pred + w_cat * cat_test_pred
        fname = f"submission_public_probe_lgbm_{fmt_weight(w_lgbm)}_cat_{fmt_weight(w_cat)}.csv"
        make_submission(test[args.id_col].to_numpy(), pred, args.target, out_dir / fname, args.clip)

    summary = {
        "runtime_seconds": float(time.time() - t0),
        "stage2_run_dir": str(stage2_dir),
        "catboost_run_dir": str(cat_dir),
        "lgbm_oof_path": str(lgbm_oof_path),
        "lgbm_test_path": str(lgbm_test_path),
        "catboost_oof_path": str(cat_oof_path),
        "catboost_test_path": str(cat_test_path),
        "lgbm_oof_auc": float(lgbm_auc),
        "catboost_oof_auc": float(cat_auc),
        "prediction_correlation": corr,
        "best_probability_blend": {
            "lgbm_weight": float(best_prob["lgbm_weight"]),
            "catboost_weight": float(best_prob["catboost_weight"]),
            "oof_auc": float(best_prob["oof_auc"]),
            "file": "submission_best_oof_probability_blend.csv",
        },
        "top_10_suggested_submissions": suggestions.head(10).to_dict(orient="records"),
        "recommended_public_probe_files": [
            f"submission_public_probe_lgbm_{fmt_weight(w)}_cat_{fmt_weight(1.0-w)}.csv" for w in public_shortlist
        ],
    }
    write_json(summary, out_dir / "weight_variant_summary.json")
    write_json(summary["best_probability_blend"], out_dir / "best_weights.json")

    print("=" * 80)
    print("DONE")
    print(f"[OUT] {out_dir}")
    print("[KEY FILES]")
    print(f"  {out_dir / 'suggested_submission_list.csv'}")
    print(f"  {out_dir / 'submission_best_oof_probability_blend.csv'}")
    print("[PUBLIC-LB PROBE FILES: submit only a few, not all]")
    for w in public_shortlist:
        print(f"  {out_dir / ('submission_public_probe_lgbm_' + fmt_weight(w) + '_cat_' + fmt_weight(1.0-w) + '.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopped by user.")
        sys.exit(130)
