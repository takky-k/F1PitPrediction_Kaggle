import argparse
import importlib.util
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

BASE_MODEL_FILE = SRC_DIR / "f1_pitnextlap_full_features_model.py"

# ここは前回の full_features_model のrunを使う。
# latestはprunedで上書きされやすいので使わない。
DEFAULT_BASE_RUN_DIR = RUNS_DIR / "full_features_model_20260529_201135"
DEFAULT_IMPORTANCE_PATH = DEFAULT_BASE_RUN_DIR / "feature_importance.csv"
DEFAULT_PARAMS_PATH = DEFAULT_BASE_RUN_DIR / "best_params.json"

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42


# ============================================================
# Logger
# ============================================================

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
        "--base-model-file",
        type=str,
        default=str(BASE_MODEL_FILE),
        help="Path to f1_pitnextlap_full_features_model.py",
    )

    parser.add_argument(
        "--importance-path",
        type=str,
        default=str(DEFAULT_IMPORTANCE_PATH),
        help="Feature importance from the full_features_model run.",
    )

    parser.add_argument(
        "--params-path",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="Best params JSON from the full_features_model run.",
    )

    parser.add_argument(
        "--baseline-cv",
        type=float,
        default=0.944658,
        help="Full features model CV baseline.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=210,
        help="Largest keep-top-n to test.",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=80,
        help="Smallest keep-top-n to test.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Step size. Example: 10 means 210, 200, 190...",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="CV folds. Use 3 to compare with the current full_features_model.",
    )

    parser.add_argument(
        "--stratify-year",
        action="store_true",
        help="Use StratifiedKFold by target x Year. Default is target only.",
    )

    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause console at the end.",
    )

    return parser.parse_args()


# ============================================================
# Import base model
# ============================================================

def import_base_model(base_model_file):
    base_model_file = Path(base_model_file)

    if not base_model_file.exists():
        raise FileNotFoundError(
            f"Base model file not found:\n{base_model_file}\n\n"
            f"Expected location:\n{BASE_MODEL_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "base_full_features_model",
        str(base_model_file),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Load full model params / importance
# ============================================================

def load_best_params(params_path, logger):
    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"best_params.json not found:\n{params_path}"
        )

    with open(params_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "best_params_full" in payload:
        params = payload["best_params_full"]
    else:
        params = payload

    # 念のため固定
    params["random_state"] = SEED
    params["n_jobs"] = -1
    params["verbosity"] = -1

    logger.log("\n========== Loaded fixed LightGBM params ==========")
    logger.log(json.dumps(params, indent=2))

    return params


def load_importance(importance_path, logger):
    importance_path = Path(importance_path)

    if not importance_path.exists():
        raise FileNotFoundError(
            f"feature_importance.csv not found:\n{importance_path}"
        )

    imp = pd.read_csv(importance_path)

    if "feature" not in imp.columns or "gain_importance_mean" not in imp.columns:
        raise ValueError(
            "Importance file must contain columns: feature, gain_importance_mean"
        )

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
        imp = imp.sort_values("gain_importance_mean", ascending=False)
        imp["gain_rank"] = np.arange(1, len(imp) + 1)

    logger.log("\n========== Loaded full feature importance ==========")
    logger.log(f"importance_path: {importance_path}")
    logger.log(f"features in file: {len(imp)}")
    logger.log("\nTop 20 importance features:")
    logger.log(
        imp[["feature", "gain_rank", "gain_importance_mean"]]
        .head(20)
        .to_string(index=False)
    )

    return imp


# ============================================================
# CV for selected top N
# ============================================================

def get_selected_features(importance_df, keep_top_n):
    keep = (
        importance_df
        .sort_values(["gain_rank", "gain_importance_mean"], ascending=[True, False])
        .head(int(keep_top_n))
    )
    return keep["feature"].astype(str).tolist()


def filter_existing_features(df, selected_features):
    exclude = {ID_COL, TARGET}
    return [
        c for c in selected_features
        if c in df.columns and c not in exclude
    ]


def run_one_pruned_cv(
    mod,
    train_fe,
    test_fe,
    categorical_cols,
    selected_features,
    lgbm_params,
    n_splits,
    stratify_year,
    logger,
    label,
):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = train_fe[TARGET].astype(int).values

    if stratify_year:
        strat_key = train_fe[TARGET].astype(str) + "__" + train_fe["Year"].astype(str)
    else:
        strat_key = y

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(len(train_fe), dtype=np.float32)
    fold_aucs = []
    best_iters = []
    feature_counts = []

    logger.log(f"\n========== Running {label} ==========")
    logger.log(f"selected feature count from importance: {len(selected_features)}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, strat_key), start=1):
        fold_start = time.time()

        tr_part = train_fe.iloc[tr_idx].copy()
        va_part = train_fe.iloc[va_idx].copy()

        # fold-safe target encoding from the base model code
        tr_part, va_part, _ = mod.apply_target_encoding(
            tr_part,
            va_part,
            test_fe,
            cols=mod.TARGET_ENCODING_COLS,
        )

        feature_cols = filter_existing_features(
            tr_part,
            selected_features,
        )

        feature_counts.append(len(feature_cols))

        X_tr = tr_part[feature_cols]
        y_tr = tr_part[TARGET].astype(int)

        X_va = va_part[feature_cols]
        y_va = va_part[TARGET].astype(int)

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
                    mod.EARLY_STOPPING_ROUNDS,
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
        fold_aucs.append(float(auc))

        best_iter = int(model.best_iteration_)
        best_iters.append(best_iter)

        elapsed = time.time() - fold_start

        logger.log(
            f"{label} | Fold {fold}: "
            f"AUC={auc:.6f} | best_iter={best_iter} | "
            f"features={len(feature_cols)} | time={elapsed:.1f}s"
        )

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    logger.log(
        f"{label} | Mean CV AUC={mean_auc:.6f} | "
        f"std={std_auc:.6f} | features used={int(np.mean(feature_counts))}"
    )

    return {
        "label": label,
        "keep_top_n": int(label.replace("top", "")),
        "cv_auc": mean_auc,
        "cv_std": std_auc,
        "features_used_mean": float(np.mean(feature_counts)),
        "features_used_min": int(np.min(feature_counts)),
        "features_used_max": int(np.max(feature_counts)),
        "best_iter_mean": float(np.mean(best_iters)),
        "best_iter_min": int(np.min(best_iters)),
        "best_iter_max": int(np.max(best_iters)),
        "fold1_auc": fold_aucs[0] if len(fold_aucs) > 0 else np.nan,
        "fold2_auc": fold_aucs[1] if len(fold_aucs) > 1 else np.nan,
        "fold3_auc": fold_aucs[2] if len(fold_aucs) > 2 else np.nan,
        "fold4_auc": fold_aucs[3] if len(fold_aucs) > 3 else np.nan,
        "fold5_auc": fold_aucs[4] if len(fold_aucs) > 4 else np.nan,
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    run_id = f"pruning_sweep_fixedparams_{timestamp_str()}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_dir / "pruning_sweep_log.txt")

    logger.log("========== PRUNING SWEEP CONFIG ==========")
    logger.log(f"run_id         : {run_id}")
    logger.log(f"project_root   : {PROJECT_ROOT}")
    logger.log(f"base_model_file: {args.base_model_file}")
    logger.log(f"importance_path: {args.importance_path}")
    logger.log(f"params_path    : {args.params_path}")
    logger.log(f"baseline_cv    : {args.baseline_cv}")
    logger.log(f"start/stop/step: {args.start}/{args.stop}/{args.step}")
    logger.log(f"n_splits       : {args.n_splits}")
    logger.log(f"stratify_year  : {args.stratify_year}")

    mod = import_base_model(args.base_model_file)

    # Match the chosen CV folds
    mod.N_SPLITS = int(args.n_splits)

    lgbm_params = load_best_params(args.params_path, logger)
    importance_df = load_importance(args.importance_path, logger)

    # Build features once
    logger.log("\n========== Building full features once ==========")

    train, test, sample_submission = mod.load_data(DATA_DIR, logger)
    mod.print_data_overview(train, test, logger)

    outlier_params = mod.fit_outlier_params(train)

    train_fe, test_fe, categorical_cols = mod.prepare_base_features(
        train,
        test,
        outlier_params,
        logger,
    )

    available_feature_count = len([
        c for c in train_fe.columns
        if c not in {ID_COL, TARGET}
    ])

    logger.log(f"\nAvailable features before target encoding: {available_feature_count}")
    logger.log(f"Importance feature count: {len(importance_df)}")

    max_n = min(int(args.start), len(importance_df))
    min_n = int(args.stop)
    step = abs(int(args.step))

    top_ns = list(range(max_n, min_n - 1, -step))

    if not top_ns:
        raise ValueError("No keep_top_n values to test. Check start/stop/step.")

    logger.log(f"\nWill test keep_top_n values: {top_ns}")

    results = []

    for n in top_ns:
        selected = get_selected_features(
            importance_df=importance_df,
            keep_top_n=n,
        )

        result = run_one_pruned_cv(
            mod=mod,
            train_fe=train_fe,
            test_fe=test_fe,
            categorical_cols=categorical_cols,
            selected_features=selected,
            lgbm_params=lgbm_params,
            n_splits=int(args.n_splits),
            stratify_year=bool(args.stratify_year),
            logger=logger,
            label=f"top{n}",
        )

        result["baseline_cv"] = float(args.baseline_cv)
        result["delta_vs_baseline"] = result["cv_auc"] - float(args.baseline_cv)
        results.append(result)

        result_df_live = pd.DataFrame(results)
        result_df_live.to_csv(run_dir / "pruning_sweep_summary_live.csv", index=False)

    summary = pd.DataFrame(results)
    summary = summary.sort_values("keep_top_n", ascending=False).reset_index(drop=True)

    summary_path = run_dir / "pruning_sweep_summary.csv"
    latest_path = PROJECT_ROOT / "pruning_sweep_summary_latest.csv"

    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_path, index=False)

    logger.log("\n========== PRUNING SWEEP RESULT ==========")
    logger.log(
        summary[
            [
                "keep_top_n",
                "cv_auc",
                "delta_vs_baseline",
                "cv_std",
                "features_used_mean",
                "best_iter_mean",
            ]
        ].to_string(index=False)
    )

    best = summary.sort_values("cv_auc", ascending=False).iloc[0]

    logger.log("\n========== BEST KEEP_TOP_N ==========")
    logger.log(best.to_string())

    # Simple trend interpretation
    logger.log("\n========== INTERPRETATION ==========")
    if best["cv_auc"] > float(args.baseline_cv) + 0.0002:
        logger.log("Pruning seems useful. Best pruned model clearly improves over baseline.")
    elif best["cv_auc"] >= float(args.baseline_cv) - 0.0002:
        logger.log("Pruning is roughly neutral. Use LB or ensemble judgment.")
    else:
        logger.log("Pruning hurts. Keep the full_features_model or try only very light pruning.")

    logger.log(f"\nSaved summary       : {summary_path}")
    logger.log(f"Saved latest summary: {latest_path}")
    logger.log(f"Saved log           : {logger.log_path}")

    if args.pause:
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
