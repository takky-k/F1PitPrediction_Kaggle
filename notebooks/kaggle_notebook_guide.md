# Kaggle Notebook Draft Guide: F1 Pit Stop Prediction

This document is written in both English and Japanese. English appears first, followed by Japanese translation.

Use this file as a blueprint for a Kaggle Notebook. Copy the Markdown explanations into Markdown cells and the Python snippets into Code cells. The goal is not only to show a final model, but to explain the full thinking process: what the competition asks, what the data means, what improved the score, what worsened it, and what should be tried next.

Competition URL: https://www.kaggle.com/competitions/playground-series-s6e5/overview

## English Version

### Notebook Title

```markdown
# F1 Pit Stop Prediction: EDA, Feature Engineering, LightGBM/CatBoost Blend, and Reflection

This notebook predicts whether an F1 car will pit on the next lap (`PitNextLap`).
I explain the data, build features from race strategy logic, train LightGBM and CatBoost models, blend their predictions, and discuss what improved or hurt the score.
```

### 1. Problem Statement

```markdown
## 1. What Are We Predicting?

The target column is `PitNextLap`.

- `PitNextLap = 1`: the car will pit on the next lap.
- `PitNextLap = 0`: the car will not pit on the next lap.

Each row describes one lap-level situation: driver, race, tyre compound, lap number, stint, tyre age, position, lap time, degradation, and race progress.

The evaluation metric is ROC AUC. AUC does not simply ask "how many rows did we classify correctly?" Instead, it asks whether rows that truly pit next lap are ranked above rows that do not. That is why this notebook focuses on producing good risk scores, not just hard 0/1 labels.
```

### 2. Local Project Context

```markdown
## 2. How This Notebook Relates To My Local Work

Most experiments were run on my local PC. The reason was practical: local scripts were easier to rerun, easier to organize into folders, and I did not need to worry as much about Kaggle Notebook session or storage limits while exploring many intermediate files.

For GitHub, I did not commit raw Kaggle data, OOF prediction arrays, submission CSVs, or model binaries. Those files are either large, generated, or should be downloaded from the competition page. The repository keeps code, reports, selected summaries, EDA tables, and generated figures.

My final private score was **0.95152**, around **782nd out of roughly 3,000 participants**. The local OOF experiments showed that the best path in my project was a LightGBM + CatBoost probability blend.
```

### 3. Imports

```python
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
```

Explanation to write under the cell:

```markdown
`pandas` and `numpy` handle data. `StratifiedKFold` creates validation folds while preserving the target ratio. `roc_auc_score` computes the competition-style validation metric. LightGBM and CatBoost are strong tree-based models for tabular data.
```

### 4. Load Data

```python
DATA_DIR = Path("/kaggle/input/playground-series-s6e5")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("train:", train.shape)
print("test:", test.shape)
print("sample_submission:", sample_submission.shape)

display(train.head())
display(test.head())
```

Expected output from the local EDA:

```markdown
- train: 439,140 rows and 16 columns
- test: 188,165 rows and 15 columns
- sample_submission: 188,165 rows and 2 columns
- No missing cells were found in train, test, or sample submission.
```

### 5. Column Meanings

```python
column_meanings = pd.DataFrame(
    [
        ("id", "Unique row identifier."),
        ("Driver", "Driver code or anonymized driver ID."),
        ("Compound", "Tyre compound, such as HARD, MEDIUM, SOFT, INTERMEDIATE, or WET."),
        ("Race", "Grand Prix or session name."),
        ("Year", "Season year."),
        ("PitStop", "Current-row pit stop related flag."),
        ("LapNumber", "Lap number in the race/session."),
        ("Stint", "Current stint number."),
        ("TyreLife", "How many laps the current tyre has been used."),
        ("Position", "Race position."),
        ("LapTime (s)", "Lap time in seconds."),
        ("LapTime_Delta", "Difference from a reference or expected lap time."),
        ("Cumulative_Degradation", "Accumulated degradation or pace-loss signal."),
        ("RaceProgress", "Fraction of race/session completed."),
        ("Position_Change", "Change in position."),
        ("PitNextLap", "Target: whether the car pits on the next lap."),
    ],
    columns=["column", "meaning"],
)

display(column_meanings)
```

Notebook explanation:

```markdown
Before modeling, I explain the columns in plain language. This prevents the notebook from becoming a black-box modeling script. In this problem, the most important ideas are tyre age, race phase, compound, stint, circuit context, and pace/degradation.
```

### 6. Target Distribution

```python
TARGET = "PitNextLap"

target_counts = train[TARGET].value_counts().sort_index()
target_rate = train[TARGET].mean()

display(target_counts.to_frame("count"))
print(f"Positive rate: {target_rate:.4%}")
```

Markdown explanation:

```markdown
`PitNextLap = 1` appears in about **19.90%** of the training rows. This is imbalanced, but not extremely rare. Because the metric is AUC, the model should rank high-risk pit situations above low-risk situations.
```

Optional plot:

```python
import matplotlib.pyplot as plt

ax = target_counts.plot(kind="bar", figsize=(5, 3), title="Target distribution")
ax.set_xlabel("PitNextLap")
ax.set_ylabel("count")
plt.show()
```

### 7. Basic EDA

```python
for col in ["Compound", "Race", "Year", "Stint", "PitStop"]:
    print(f"\n=== {col} ===")
    summary = (
        train.groupby(col)[TARGET]
        .agg(count="count", pit_rate="mean")
        .sort_values("count", ascending=False)
    )
    display(summary.head(30))
```

Markdown explanation:

```markdown
The strongest simple EDA findings were:

- `TyreLife`, `LapNumber`, `Stint`, and `RaceProgress` had the strongest simple numeric relationships with the target.
- `Compound` mattered because tyre type changes pit strategy.
- `Race` mattered because circuits have different pit windows and strategy patterns.
- `Year` had a suspiciously different distribution and needed careful validation.
- `LapTime (s)`, `LapTime_Delta`, and `Cumulative_Degradation` contained large outliers.
```

Numeric EDA:

```python
numeric_cols = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]

display(train[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T)

corr = (
    train[numeric_cols + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda s: s.abs(), ascending=False)
)
display(corr.to_frame("corr_with_target"))
```

Expected local correlation ordering:

```markdown
The strongest simple correlations with the target were:

1. `TyreLife`
2. `LapNumber`
3. `Stint`
4. `RaceProgress`
5. `Cumulative_Degradation`

Correlation is only a first check. Tree models can still use non-linear patterns that correlation does not capture.
```

### 8. Train/Test Distribution Check

```python
def compare_train_test(col, top_n=20):
    tr = train[col].value_counts(normalize=True).head(top_n).rename("train")
    te = test[col].value_counts(normalize=True).head(top_n).rename("test")
    return pd.concat([tr, te], axis=1).fillna(0)

for col in ["Compound", "Race", "Year", "Stint", "PitStop"]:
    print(f"\n=== train/test comparison: {col} ===")
    display(compare_train_test(col))
```

Markdown explanation:

```markdown
This step checks whether train and test look similar. A feature can look powerful in validation but fail on the leaderboard if the test distribution is different. This was one reason I generated distribution comparison figures in the local report.
```

### 9. Feature Engineering

```python
def add_basic_features(df):
    df = df.copy()
    df["LapTime_s"] = df["LapTime (s)"]

    eps = 1e-6
    df["tyre_life_ratio"] = df["TyreLife"] / (df["LapNumber"] + 1)
    df["progress_x_tyre"] = df["RaceProgress"] * df["TyreLife"]
    df["LapNumber_minus_TyreLife"] = df["LapNumber"] - df["TyreLife"]
    df["progress_x_degradation"] = df["RaceProgress"] * df["Cumulative_Degradation"]
    df["lap_per_progress"] = df["LapNumber"] / (df["RaceProgress"] + eps)
    df["estimated_total_distance_km"] = df["lap_per_progress"]

    # Simple bins create strategy "tags".
    df["race_phase"] = pd.cut(
        df["RaceProgress"],
        bins=[-0.01, 0.25, 0.60, 1.01],
        labels=["early", "mid", "late"],
    ).astype(str)
    df["tyre_life_bin"] = pd.cut(
        df["TyreLife"],
        bins=[-1, 5, 15, 30, 100],
        labels=["very_young", "young", "used", "old"],
    ).astype(str)

    # Contextual categorical interactions.
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
    df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["Compound_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)
    df["RacePhase_TyreLifeBin"] = df["race_phase"].astype(str) + "_" + df["tyre_life_bin"].astype(str)
    df["Compound_TyreLifeBin"] = df["Compound"].astype(str) + "_" + df["tyre_life_bin"].astype(str)
    return df


train_fe = add_basic_features(train)
test_fe = add_basic_features(test)
```

Markdown explanation:

```markdown
The goal is to describe the pit decision mechanism.

- Tyre age matters because older tyres are more likely to be changed.
- Race progress matters because pit windows depend on race phase.
- Compound matters because tyre durability differs by compound.
- Race context matters because circuits have different lap lengths and strategies.
- Interaction features make context explicit, such as `Race_Compound` and `Compound_Stint`.
```

### 10. Frequency Encoding

```python
freq_cols = [
    "Driver", "Race", "Compound", "Race_Year", "Driver_Race",
    "Driver_Compound", "Race_Compound", "Compound_Stint",
    "RacePhase_TyreLifeBin", "Compound_TyreLifeBin",
]

all_fe = pd.concat([train_fe[freq_cols], test_fe[freq_cols]], axis=0)

for col in freq_cols:
    freq = all_fe[col].value_counts(dropna=False)
    train_fe[f"{col}_freq"] = train_fe[col].map(freq).fillna(0).astype("int32")
    test_fe[f"{col}_freq"] = test_fe[col].map(freq).fillna(0).astype("int32")
```

Markdown explanation:

```markdown
Frequency encoding tells the model how common or rare a category is. Rare categories can be noisy; frequent categories often have more stable patterns. This was one of the improvement ideas from stronger notebooks.
```

### 11. Group Statistics

```python
def add_group_stats(train_df, test_df, group_cols, value_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for group_col in group_cols:
        for value_col in value_cols:
            stats = train_df.groupby(group_col)[value_col].agg(["mean", "std"]).rename(
                columns={
                    "mean": f"{value_col}_mean_by_{group_col}",
                    "std": f"{value_col}_std_by_{group_col}",
                }
            )

            for df in [train_df, test_df]:
                df[f"{value_col}_mean_by_{group_col}"] = df[group_col].map(stats[f"{value_col}_mean_by_{group_col}"])
                df[f"{value_col}_std_by_{group_col}"] = df[group_col].map(stats[f"{value_col}_std_by_{group_col}"])
                df[f"{value_col}_diff_mean_by_{group_col}"] = (
                    df[value_col] - df[f"{value_col}_mean_by_{group_col}"]
                )

    return train_df, test_df


group_cols = ["Race", "Race_Year", "Compound_Stint", "Driver_Race"]
value_cols = ["LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "TyreLife", "RaceProgress"]

train_fe, test_fe = add_group_stats(train_fe, test_fe, group_cols, value_cols)
```

Markdown explanation:

```markdown
Group statistics compare a row against its context. A `LapTime_Delta` of 3 seconds may be unusual in one race but normal in another. Difference-from-group-mean and group standard deviation features help the model understand whether the current lap is abnormal for that context.
```

### 12. Lag and Rolling Features

```python
def add_sequence_features(df):
    df = df.copy()
    original_index = df.index
    df["_original_index"] = original_index
    df = df.sort_values(["Driver", "Race", "Stint", "LapNumber", "id"])

    group = df.groupby(["Driver", "Race", "Stint"], sort=False)
    seq_cols = ["LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "TyreLife", "Position_Change"]

    for col in seq_cols:
        df[f"{col}_lag1"] = group[col].shift(1)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_roll3_mean"] = group[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    df = df.sort_values("_original_index").drop(columns=["_original_index"])
    return df


train_fe = add_sequence_features(train_fe)
test_fe = add_sequence_features(test_fe)
```

Markdown explanation:

```markdown
A pit stop usually follows a trend: tyres age, pace changes, degradation increases, or position changes. Lag and rolling features capture recent movement instead of looking only at the current row.
```

### 13. Feature List and Missing Values

```python
target = "PitNextLap"
drop_cols = ["id", target]

features = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]

cat_features = [
    c for c in features
    if train_fe[c].dtype == "object" or str(train_fe[c].dtype).startswith("category")
]

num_features = [c for c in features if c not in cat_features]

for c in cat_features:
    train_fe[c] = train_fe[c].astype("category")
    test_fe[c] = test_fe[c].astype("category")

for c in num_features:
    median_value = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(median_value)
    test_fe[c] = test_fe[c].fillna(median_value)

print("features:", len(features))
print("categorical features:", len(cat_features))
print("numeric features:", len(num_features))
```

Markdown explanation:

```markdown
This notebook keeps both numeric and categorical features. LightGBM can use pandas categorical columns. CatBoost can also use categorical feature names. Missing numeric values are filled with the training median to avoid using information from the target.
```

### 14. Cross-Validation Setup

```python
X = train_fe[features]
y = train_fe[target].astype(int)
X_test = test_fe[features]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Markdown explanation:

```markdown
I use 5-fold stratified cross-validation. "Stratified" means each fold keeps a similar positive/negative target ratio. This is important because `PitNextLap = 1` is only about 19.90%.
```

### 15. LightGBM Model

```python
oof_lgb = np.zeros(len(train_fe))
test_lgb = np.zeros(len(test_fe))

lgb_params = dict(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    oof_lgb[va_idx] = model.predict_proba(X_va)[:, 1]
    test_lgb += model.predict_proba(X_test)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_va, oof_lgb[va_idx])
    print(f"Fold {fold} LightGBM AUC: {fold_auc:.6f}")

lgb_auc = roc_auc_score(y, oof_lgb)
print(f"LightGBM OOF AUC: {lgb_auc:.6f}")
```

Markdown explanation:

```markdown
LightGBM is a strong first model for tabular data. In my local selected-feature path, LightGBM reached about **0.958394 OOF AUC** after AB feature selection. The exact number in this notebook may differ because this notebook version adds extra EDA-inspired features and may run in a different environment.
```

### 16. CatBoost Model

```python
oof_cat = np.zeros(len(train_fe))
test_cat = np.zeros(len(test_fe))

cat_feature_indices = [features.index(c) for c in cat_features]

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_feature_indices,
        early_stopping_rounds=200,
    )

    oof_cat[va_idx] = model.predict_proba(X_va)[:, 1]
    test_cat += model.predict_proba(X_test)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_va, oof_cat[va_idx])
    print(f"Fold {fold} CatBoost AUC: {fold_auc:.6f}")

cat_auc = roc_auc_score(y, oof_cat)
print(f"CatBoost OOF AUC: {cat_auc:.6f}")
```

Markdown explanation:

```markdown
CatBoost was useful because it can model categorical/rule-like structure differently from LightGBM. In my local main path, CatBoost reached about **0.959197 OOF AUC**, stronger than the selected-feature LightGBM single model.
```

### 17. OOF Saving and Model Correlation

```python
oof_df = pd.DataFrame(
    {
        "id": train["id"],
        "target": y,
        "lgb": oof_lgb,
        "cat": oof_cat,
    }
)

test_pred_df = pd.DataFrame(
    {
        "id": test["id"],
        "lgb": test_lgb,
        "cat": test_cat,
    }
)

display(oof_df[["lgb", "cat", "target"]].corr())
```

Markdown explanation:

```markdown
Saving OOF predictions is important. OOF files allow honest blending, model correlation checks, and stacking. If two models are strong and not perfectly identical, blending can improve the final ranking.
```

### 18. Blend Weight Search

```python
weights = np.linspace(0, 1, 1001)
records = []

for w in weights:
    pred = w * oof_lgb + (1 - w) * oof_cat
    auc = roc_auc_score(y, pred)
    records.append((w, 1 - w, auc))

blend_report = pd.DataFrame(records, columns=["lgb_weight", "cat_weight", "oof_auc"])
best = blend_report.sort_values("oof_auc", ascending=False).iloc[0]
display(best)

best_lgb_w = float(best["lgb_weight"])
best_cat_w = float(best["cat_weight"])

oof_blend = best_lgb_w * oof_lgb + best_cat_w * oof_cat
test_blend = best_lgb_w * test_lgb + best_cat_w * test_cat

print("Best blend OOF AUC:", roc_auc_score(y, oof_blend))
print("Best weights:", best_lgb_w, best_cat_w)
```

Markdown explanation:

```markdown
In my local project, the best LightGBM/CatBoost probability blend was approximately:

- LightGBM weight: **0.427**
- CatBoost weight: **0.573**
- OOF AUC: **0.960446**

This improved over both single models because the models made slightly different errors.
```

### 19. Optional: Rank Blend

```python
rank_lgb = pd.Series(oof_lgb).rank(pct=True).values
rank_cat = pd.Series(oof_cat).rank(pct=True).values

rank_test_lgb = pd.Series(test_lgb).rank(pct=True).values
rank_test_cat = pd.Series(test_cat).rank(pct=True).values

rank_blend_oof = best_lgb_w * rank_lgb + best_cat_w * rank_cat
rank_blend_test = best_lgb_w * rank_test_lgb + best_cat_w * rank_test_cat

print("Rank blend OOF AUC:", roc_auc_score(y, rank_blend_oof))
```

Markdown explanation:

```markdown
Rank blending can help when probability calibration is unreliable. In my local experiments, probability blending was slightly better, so I used the probability blend as the main path.
```

### 20. Optional Future Model: RealMLP

```markdown
## Optional Future Work: RealMLP

Top notebooks suggested trying RealMLP from PyTabKit. RealMLP is a neural-network model for tabular data. It may help because it can make different mistakes from tree models such as LightGBM and CatBoost.

I would not make RealMLP the first model. I would first build strong LightGBM/CatBoost OOF predictions, then add RealMLP as a third OOF/test prediction file and check:

1. RealMLP OOF AUC
2. Correlation with LightGBM and CatBoost
3. Whether it improves blend or stacking

If it is weak and highly correlated, it should not be added. If it is slightly weaker but different, it may still improve the ensemble.
```

### 21. Submission

```python
submission = sample_submission.copy()
submission[TARGET] = test_blend
submission.to_csv("submission.csv", index=False)
display(submission.head())
```

Markdown explanation:

```markdown
Kaggle expects a CSV with `id` and `PitNextLap`. The values should be probabilities or risk scores. For AUC, the ordering of predictions is especially important.
```

### 22. What Improved and What Hurt

```markdown
## What Improved The Score

| Change | Why it helped |
| --- | --- |
| Safe numeric features | Tyre age, lap number, race progress, and degradation directly describe pit timing. |
| AB feature selection | It kept features that improved validation instead of trusting every generated feature. |
| CatBoost | It captured categorical/rule-like structure differently from LightGBM. |
| LightGBM + CatBoost probability blend | The two models were strong and not perfectly identical, so blending reduced model-specific errors. |
| OOF-based weight search | It chose weights using validation predictions instead of guessing. |

## What Hurt Or Did Not Help

| Attempt | What happened | Likely reason |
| --- | --- | --- |
| 100+ AI-generated features without enough filtering | Stayed around the 0.93 range and was discarded. | Feature quantity is not feature quality. Many features did not match the pit-stop mechanism. |
| Blind pruning by importance | Some pruning experiments worsened the score. | Weak-looking features can still help in combination. |
| Removing pit-stop/timing signals | Worsened strongly. | Pit timing signals were important for this target. |
| XGBoost as third model | OOF AUC was much lower and final weight became 0.0. | A weak model adds noise even if it increases diversity. |
| Rank blending | Slightly worse than probability blending locally. | The probabilities were informative enough that calibration helped. |
```

### 23. Final Reflection

```markdown
## Final Reflection

The biggest lesson was that the best features were not random. Good features described the mechanism of a pit stop: tyre life, stint, race progress, compound, circuit context, recent pace loss, and degradation.

The second lesson was that OOF predictions are essential. They make model comparison and blending much more honest.

The third lesson was that stronger notebooks were stronger because their whole pipeline was better: richer contextual features, group statistics, frequency encoding, lag/rolling features, RealMLP trials, saved OOF predictions, model correlation checks, and careful final blending.
```

## Japanese Version

### Notebook タイトル

```markdown
# F1 Pit Stop Prediction：EDA、特徴量作成、LightGBM/CatBoostブレンド、振り返り

このNotebookでは、F1の車両が次のラップでピットインするかどうかを予測します。
データの意味、EDA、特徴量作成、LightGBMとCatBoost、ブレンド、スコアを改善した要因と悪化した要因を順番に説明します。
```

### 1. 何を予測するコンペか

```markdown
目的変数は `PitNextLap` です。

- `PitNextLap = 1`: 次のラップでピットインする
- `PitNextLap = 0`: 次のラップでピットインしない

各行は、あるラップ時点の状況を表しています。ドライバー、レース、タイヤコンパウンド、ラップ番号、スティント、タイヤ寿命、順位、ラップタイム、劣化、レース進行率などが含まれます。

評価指標は ROC AUC です。AUC は単純な正解率ではなく、ピットインする行をピットインしない行より高く順位付けできているかを見る指標です。そのため、このNotebookでは0/1の分類だけでなく、ピットインリスクの順位付けを重視します。
```

### 2. ローカル作業との関係

```markdown
多くの実験は Kaggle Notebook 上ではなくローカルPCで行いました。理由は、ローカルの方がスクリプトを何度も回しやすく、フォルダで結果を整理しやすく、容量やセッション制限を気にせず中間ファイルを残せたからです。

GitHubには、生データ、OOF予測配列、提出CSV、モデル本体は入れていません。これらは大きい、生成可能、またはKaggleから取得すべきファイルだからです。

最終 private score は **0.95152** で、約3,000人中 **782位前後** でした。ローカルOOFでは、LightGBMとCatBoostの確率ブレンドが最も良い流れでした。
```

### 3. Notebookで説明するべき流れ

| セクション | 説明する内容 |
| --- | --- |
| Problem | `PitNextLap` を予測する二値分類であること。AUCは順位付け指標であること。 |
| Data loading | train/test/sample_submission を読み込む。trainだけに正解がある。 |
| Column meaning | 各列の意味を初心者向けに説明する。 |
| EDA | target比率、Compound/Race/Year/Stint/PitStopごとの傾向、数値列の外れ値を見る。 |
| Feature engineering | タイヤ寿命、レース進行率、劣化、レース文脈、頻度、グループ統計、ラグ/ローリングを作る。 |
| Validation | StratifiedKFoldとOOF AUCで評価する。 |
| LightGBM | 高速で強い表形式データ向けモデルとして使う。 |
| CatBoost | カテゴリ構造を違う形で拾う2つ目の強いモデルとして使う。 |
| Blend | OOF予測を使って重みを探索し、確率を混ぜる。 |
| Reflection | 何が効いたか、何が効かなかったか、次に何をするかを書く。 |

### 4. EDAで書くべきポイント

```markdown
EDAで分かったこと：

1. `PitNextLap = 1` は約19.90%で、やや不均衡。
2. `TyreLife`, `LapNumber`, `Stint`, `RaceProgress` は目的変数と強い関係があった。
3. `Compound` と `Race` は戦略差を表す重要なカテゴリ。
4. `Year` は分布差が大きく、検証時に注意が必要。
5. `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation` には大きな外れ値がある。
6. train/test分布を確認し、ローカル検証だけで過信しない。
```

### 5. 特徴量作成で説明するべきこと

```markdown
特徴量は「数を増やす」ことが目的ではありません。ピットインが起きる理由につながる特徴量を作ることが重要です。

有効だった考え方：

- タイヤが古くなるほどピットインしやすい
- レース終盤か序盤かで戦略が変わる
- HARD/MEDIUM/SOFTなどコンパウンドで耐久性が違う
- レースごとにピット戦略が違う
- 同じラップタイム差でも、レースやドライバー文脈によって意味が違う
- 直近数ラップの変化を見ると、ピット直前の兆候を拾える可能性がある
```

### 6. モデルで説明するべきこと

```markdown
LightGBMを使った理由：

- 表形式データに強い
- 学習が速い
- 特徴量の効果を見やすい
- Kaggleでよく使われる安定したベースライン

CatBoostを使った理由：

- カテゴリ変数の扱いが得意
- LightGBMとは違う間違い方をする可能性がある
- ブレンドすると個別モデルの誤差を減らせる可能性がある

XGBoostを試した理由：

- 3つ目のモデルとして多様性を増やせるか確認したかった

XGBoostを最終ブレンドで使わなかった理由：

- OOF AUCが低く、重み探索で0.0になった
- 多様性があっても、単体性能が低すぎるとノイズになる
```

### 7. スコア改善・悪化のまとめ

| 試行 | 結果 | 推測理由 |
| --- | --- | --- |
| 安全な数値特徴量 | 改善 | タイヤ寿命、ラップ番号、進行率、劣化がピット判断に直結していた。 |
| AB特徴量選択 | 改善 | 効く特徴量だけを残し、ノイズを減らせた。 |
| CatBoost追加 | 改善 | カテゴリ構造をLightGBMと違う形で拾えた。 |
| LightGBM/CatBoostブレンド | 最良 | 強い2モデルの誤差を平均で減らせた。 |
| 100個以上のAI生成特徴量 | 失敗 | 数は多いが、ピット判断の仕組みに合わない特徴量が多かった。 |
| 重要度だけでpruning | 悪化する場合あり | 単体では弱くても組み合わせで効く特徴量を落とした可能性。 |
| `PitStop`系を落とす | 悪化 | タイミング情報が重要だった。 |
| XGBoost追加 | 改善せず | OOF AUCが低く、混ぜるとノイズになった。 |
| rank blend | 確率blendより少し弱い | 今回は確率自体が十分に有用だった。 |

### 8. 上位Notebookから学んだこと

```markdown
上位Notebookとの差は、単にモデルが違っただけではありません。パイプライン全体がより完成していました。

次回やるべきこと：

1. `Race_Year`, `Driver_Race`, `Race_Compound`, `Compound_Stint` など文脈カテゴリを増やす。
2. グループ平均、標準偏差、平均との差分を作る。
3. 高カーディナリティカテゴリには frequency encoding を入れる。
4. driver × race × stint の中で lag/rolling 特徴量を作る。
5. OOF予測とtest予測を毎回保存する。
6. モデル間相関を見る。
7. RealMLPを早めに試す。
8. 確率ブレンド、rank blend、rank-remap、stackingを比較する。
```

### 9. 最後に書く振り返り

```markdown
今回の一番大きな学びは、特徴量は数ではなく質だということです。良い特徴量は、目的変数が発生する理由とつながっています。

2つ目の学びは、OOF予測の保存が重要だということです。OOFがあると、モデル比較、ブレンド、stacking、相関確認ができます。

3つ目の学びは、強い解法はモデル単体ではなく、EDA、特徴量、検証、OOF管理、相関確認、ブレンドまで含めた総合力で決まるということです。
```
