# F1 Pit Stop Prediction Report

This document is written in both English and Japanese. The English report appears first, followed by a Japanese translation with the same structure.

Competition page: https://www.kaggle.com/competitions/playground-series-s6e5/overview

## English Report

### 1. Competition Objective

The task was to predict `PitNextLap`, a binary target that indicates whether a Formula 1 car will pit on the next lap. The training data contains historical lap-level records with driver, race, tyre, stint, lap timing, position, and degradation-related variables. The test data has the same explanatory columns but does not include `PitNextLap`; the submission file must provide one prediction per `id`.

Local evaluation focused on OOF AUC. The project explored feature engineering, LightGBM feature selection, CatBoost, XGBoost, and probability/rank blending.

### 2. Data Overview

| Dataset | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `data/train.csv` | 439,140 | 16 | Includes the target `PitNextLap`. |
| `data/test.csv` | 188,165 | 15 | Same features as train, without the target. |
| `data/sample_submission.csv` | 188,165 | 2 | Submission template with `id` and `PitNextLap`. |

The target was imbalanced but not extremely rare: `PitNextLap = 1` appeared in 87,381 rows, or 19.90% of the training data.

### 3. Column Meanings and EDA Summary

The column meanings below are inferred from the CSV schema and values.

| Column | Meaning | Train EDA Summary |
| --- | --- | --- |
| `id` | Unique row identifier. | 439,140 unique train IDs, no missing values; test has 188,165 unique IDs. |
| `Driver` | Driver code or anonymized driver ID. | 887 unique train values, no missing values. Frequent values include `MAS` 1,682, `RAI` 1,669, `BAR` 1,656, `BUT` 1,655. |
| `Compound` | Tyre compound. | 5 values. `MEDIUM` 211,141 rows, `HARD` 170,518, `SOFT` 38,744, `INTERMEDIATE` 17,382, `WET` 1,355. |
| `Race` | Grand Prix / session name. | 26 races. Most frequent: Dutch GP 24,462, Mexico City GP 23,672, Pre-Season Testing 22,492, Hungarian GP 22,481. |
| `Year` | Season year. | 2022-2025. Counts by year: 2022 has 82,989 rows, 2023 has 136,147, 2024 has 127,110, 2025 has 92,894. |
| `PitStop` | Whether the current row is associated with a pit stop event flag. | Binary. Mean is 0.136, so about 13.61% of rows have `PitStop = 1`. |
| `LapNumber` | Lap number in the race/session. | Median 19, mean 23.11, 95th percentile 54, max 78. |
| `Stint` | Stint number. | Median 2, mean 1.79, max 8. Most rows are stint 1 or 2. |
| `TyreLife` | Age of the current tyre in laps. | Median 12, mean 14.16, 95th percentile 32, max 77. |
| `Position` | Track/race position. | Positions 1-20. Median 10, mean 9.63. |
| `LapTime (s)` | Lap time in seconds. | Median 90.521s, mean 90.949s. 99th percentile 124.900s; max 2507.610s indicates extreme outliers. |
| `LapTime_Delta` | Difference from a reference lap time. | Median -0.295, mean -3.770. Extreme min/max values around -2403.89 and 2423.93 suggest outlier laps or special sessions. |
| `Cumulative_Degradation` | Accumulated tyre or pace degradation signal. | Median -20.994, mean -25.722, 95th percentile 84.401, max 2412.030. Large outliers exist. |
| `RaceProgress` | Fraction of race/session completed. | Range about 0.013 to 1.000, median 0.269, mean 0.338. |
| `Position_Change` | Change in position. | Median 0, mean 0.102, range -18 to 18. |
| `PitNextLap` | Target: whether the car pits next lap. | Binary. 351,759 zeros and 87,381 ones; positive rate 19.90%. |

### 4. Basic Trends

| View | Observation | Interpretation |
| --- | --- | --- |
| Target balance | Positive rate is 19.90%. | AUC is appropriate because probability ranking matters more than raw accuracy. |
| Tyre compound | `HARD` has a high next-lap pit rate of 32.75%; `MEDIUM` is 10.11%; `SOFT` is 19.35%; `WET` is 2.51%. | Compound is highly informative, but it interacts with stint, race, and lap progress. |
| Race | Monaco GP has a high next-lap pit rate of 35.74%, Spanish GP 32.00%, Bahrain GP 28.75%; Mexico City GP is 9.07%. | Race-specific strategy and lap structure matter. Race metadata and categorical handling are important. |
| Year | 2023 has a very low target rate of 0.96%, while 2022, 2024, and 2025 are around 26.65%, 29.53%, and 28.44%. | Year distribution is suspiciously different and should be handled carefully to avoid validation artifacts. |
| Stint | Stint 2 has the highest next-lap pit rate at 39.11%; stint 1 is only 5.98%. | Stint number and tyre age jointly encode strategy timing. |
| PitStop flag | Rows with `PitStop = 1` have a 24.78% next-lap pit rate vs 19.13% for `PitStop = 0`. | Useful signal, but it must be checked for leakage-like behavior. |
| Lap time and degradation | `LapTime (s)`, `LapTime_Delta`, and `Cumulative_Degradation` contain very large outliers. | Robust clipping, outlier flags, and careful validation are needed. |

### 5. What Was Done

1. Built an initial LightGBM pipeline with feature engineering.
2. Added race, tyre, progress, degradation, lag/rolling, frequency, and target-encoding style features in exploratory scripts.
3. Ran AB feature selection to keep a compact feature set.
4. Trained a stage-2 LightGBM model with 21 selected features.
5. Trained CatBoost on the selected feature set.
6. Searched LightGBM/CatBoost blend weights using OOF predictions.
7. Tested rank blending and several manual blend variants.
8. Added a lightweight XGBoost model as a possible third model.
9. Ran a stronger CatBoost-2000 follow-up experiment.
10. Preserved compact run summaries and excluded raw data, model binaries, OOF arrays, and submissions from GitHub.

### 6. Results

| Experiment | OOF AUC | Interpretation |
| --- | ---: | --- |
| Stage-2 LightGBM baseline | 0.955756 | Strong baseline from 14 core features. |
| LightGBM AB-selected | 0.958394 | Feature selection improved the LightGBM model. |
| CatBoost on selected features | 0.959197 | Strongest single model in the main path. |
| LightGBM + CatBoost probability blend | 0.960446 | Best OOF result; weights were LightGBM 0.427 and CatBoost 0.573. |
| XGBoost lightweight model | 0.942677 | Too weak to help the final blend. |
| Three-model ensemble | 0.960446 | XGBoost received weight 0.0, so this matched the two-model blend. |
| CatBoost-2000 follow-up | 0.960022 | Strong standalone run, but below the best LightGBM/CatBoost blend. |

### 7. Interpretation

The most useful lesson is that a compact, well-selected feature set beat several broader feature-heavy experiments. Adding many time-series and outlier features did not automatically improve the final model. The model benefited most from combining two strong but not identical learners: LightGBM and CatBoost.

XGBoost added theoretical diversity, but its OOF score was much lower, so the weight search correctly assigned it zero weight. Probability blending worked better than the tested rank blends in local OOF.

The `Year` distribution and the strong race/compound effects deserve careful explanation in any final notebook. They may represent real strategy patterns, generated-data quirks, or validation sensitivity.

### 8. GitHub File Inventory

The GitHub repository is designed to keep the work understandable and reproducible without committing large generated artifacts.

| Path | What it is | Why it is included |
| --- | --- | --- |
| `README.md` | Project overview and rerun guide. | Gives visitors the entry point. |
| `REPORT.md` | This combined bilingual report. | Keeps the retrospective, EDA, results, and file inventory in one place. |
| `.gitignore` | Ignore rules. | Excludes data, submissions, predictions, models, caches, and local editor files. |
| `.gitattributes` | Text normalization rules. | Keeps line endings stable. |
| `requirements.txt` | Python dependencies. | Documents the expected environment. |
| `src/*.py` | Modeling scripts. | Preserves feature engineering, LightGBM, CatBoost, XGBoost, and ensemble logic. |
| `outlier_analysis.py` | Outlier EDA script. | Documents early outlier investigation. |
| `reports/key_results.csv` | Compact score table. | Summarizes the experiments that mattered. |
| `data/README.md` | Data placement note. | Explains which Kaggle files to place locally. |
| `notebooks/README.md` | Notebook workspace note. | Reserves a place for the later retrospective notebook. |
| `experiment_summary.csv` | Early experiment chronology. | Shows which feature-heavy/pruning paths helped or failed. |
| `feature_importance_latest.csv` | Feature importance artifact. | Supports later explanation and charts. |
| `feature_effect_report_latest.csv` | Feature effect artifact. | Helps interpret feature behavior. |
| `oof_calibration_by_flags_latest.csv` | Calibration by flags. | Helps diagnose brittleness. |
| `outlier_analysis_report.csv` | Outlier report. | Small EDA artifact from `outlier_analysis.py`. |
| `outlier_flag_effects_latest.csv` | Outlier flag effect report. | Records whether outlier flags helped. |
| `pruning_sweep_summary_latest.csv` | Pruning sweep summary. | Documents unsuccessful pruning paths. |
| `selected_features_latest.json` | Later selected-feature artifact. | Historical comparison artifact. |
| `best_params_latest.json` | Later parameter artifact. | Historical comparison artifact. |
| `runs/f1_stage2_ensemble/*summary/selection small files` | Stage-2 compact artifacts. | Keeps baseline and AB-selected evidence. |
| `runs/f1_lgbm_catboost_ensemble/*summary/weights small files` | CatBoost and blend artifacts. | Records the best two-model path. |
| `runs/f1_weight_variants/*summary/grid small files` | Blend variant artifacts. | Documents probability and rank blend trials. |
| `runs/f1_xgb_three_model/*summary/weights small files` | XGBoost artifacts. | Shows why the third model was not useful. |
| `runs/f1_cat2000_next/*summary/weights small files` | CatBoost-2000 follow-up artifacts. | Preserves the stronger CatBoost comparison. |

Excluded from GitHub: `data/*.csv`, root `submission.csv`, `runs/**/submission*.csv`, OOF/test prediction arrays, model binaries such as `*.cbm`, `.vs/`, `__pycache__/`, and the binary Word document. These are either large, generated, local-only, or better represented in Markdown.

## Japanese Translation

### 1. コンペの目的

このコンペの目的は、F1 の車両が次のラップでピットインするかどうかを表す二値目的変数 `PitNextLap` を予測することです。学習データには、ドライバー、レース、タイヤ、スティント、ラップタイム、順位、劣化に関するラップ単位の情報が含まれています。テストデータには同じ説明変数がありますが、`PitNextLap` は含まれていません。提出ファイルでは `id` ごとに予測値を出します。

ローカル評価では主に OOF AUC を見ました。特徴量エンジニアリング、LightGBM の特徴量選択、CatBoost、XGBoost、確率ブレンドと順位ブレンドを試しました。

### 2. データ概要

| データ | 行数 | 列数 | 内容 |
| --- | ---: | ---: | --- |
| `data/train.csv` | 439,140 | 16 | 目的変数 `PitNextLap` を含む学習データ。 |
| `data/test.csv` | 188,165 | 15 | 学習データと同じ説明変数。目的変数はない。 |
| `data/sample_submission.csv` | 188,165 | 2 | `id` と `PitNextLap` を持つ提出テンプレート。 |

目的変数は不均衡ですが、極端にまれではありません。`PitNextLap = 1` は 87,381 行で、学習データ全体の 19.90% でした。

### 3. 各列の意味と EDA 要約

以下の列説明は、CSV の列名と値から推定したものです。

| 列 | 意味 | 学習データでの EDA 要約 |
| --- | --- | --- |
| `id` | 行ごとの一意な識別子。 | 学習データで 439,140 個すべて一意。欠損なし。テストも 188,165 個すべて一意。 |
| `Driver` | ドライバーコード、または匿名化されたドライバーID。 | 887 種類。欠損なし。多い値は `MAS` 1,682、`RAI` 1,669、`BAR` 1,656、`BUT` 1,655。 |
| `Compound` | タイヤコンパウンド。 | 5 種類。`MEDIUM` 211,141 行、`HARD` 170,518 行、`SOFT` 38,744 行、`INTERMEDIATE` 17,382 行、`WET` 1,355 行。 |
| `Race` | グランプリまたはセッション名。 | 26 種類。多いものは Dutch GP 24,462、Mexico City GP 23,672、Pre-Season Testing 22,492、Hungarian GP 22,481。 |
| `Year` | シーズン年。 | 2022-2025。2022 年 82,989 行、2023 年 136,147 行、2024 年 127,110 行、2025 年 92,894 行。 |
| `PitStop` | 現在行がピットストップに関連するかを示すフラグ。 | 二値。平均 0.136 なので、約 13.61% が `PitStop = 1`。 |
| `LapNumber` | レースまたはセッション内のラップ番号。 | 中央値 19、平均 23.11、95 パーセンタイル 54、最大 78。 |
| `Stint` | スティント番号。 | 中央値 2、平均 1.79、最大 8。多くはスティント 1 または 2。 |
| `TyreLife` | 現在のタイヤの使用ラップ数。 | 中央値 12、平均 14.16、95 パーセンタイル 32、最大 77。 |
| `Position` | 走行順位。 | 1-20 位。中央値 10、平均 9.63。 |
| `LapTime (s)` | ラップタイム秒。 | 中央値 90.521 秒、平均 90.949 秒。99 パーセンタイル 124.900 秒、最大 2507.610 秒で大きな外れ値がある。 |
| `LapTime_Delta` | 基準ラップタイムとの差分。 | 中央値 -0.295、平均 -3.770。最小 -2403.89、最大 2423.93 で、特殊なラップやセッション由来の外れ値がある。 |
| `Cumulative_Degradation` | 累積のタイヤまたはペース劣化シグナル。 | 中央値 -20.994、平均 -25.722、95 パーセンタイル 84.401、最大 2412.030。大きな外れ値がある。 |
| `RaceProgress` | レースまたはセッションの進行率。 | 約 0.013 から 1.000。中央値 0.269、平均 0.338。 |
| `Position_Change` | 順位変化。 | 中央値 0、平均 0.102、範囲は -18 から 18。 |
| `PitNextLap` | 目的変数。次ラップでピットインするか。 | 二値。0 が 351,759 行、1 が 87,381 行。陽性率 19.90%。 |

### 4. 基本的な傾向

| 観点 | 観察結果 | 解釈 |
| --- | --- | --- |
| 目的変数の比率 | 陽性率は 19.90%。 | 単純な正解率よりも、確率の順位付けを見る AUC が適している。 |
| タイヤコンパウンド | `HARD` の次ラップピット率は 32.75%、`MEDIUM` は 10.11%、`SOFT` は 19.35%、`WET` は 2.51%。 | コンパウンドは強い特徴量だが、スティント、レース、進行率との相互作用が重要。 |
| レース | Monaco GP は 35.74%、Spanish GP は 32.00%、Bahrain GP は 28.75%。Mexico City GP は 9.07%。 | レースごとの戦略やラップ構造が効いている。レース情報とカテゴリ処理が重要。 |
| 年 | 2023 年は 0.96% と非常に低く、2022/2024/2025 年は 26.65%、29.53%、28.44%。 | 年による分布差が大きい。実際の傾向、生成データの癖、検証設計の影響を慎重に見る必要がある。 |
| スティント | スティント 2 の次ラップピット率が 39.11% と最も高く、スティント 1 は 5.98%。 | スティント番号とタイヤ寿命は、戦略タイミングを強く表している。 |
| `PitStop` フラグ | `PitStop = 1` は 24.78%、`PitStop = 0` は 19.13%。 | 有用なシグナルだが、リークに近い挙動がないか確認が必要。 |
| ラップタイムと劣化 | `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation` には大きな外れ値がある。 | クリッピング、外れ値フラグ、慎重な検証が必要。 |

### 5. 実施したこと

1. LightGBM の初期パイプラインと特徴量エンジニアリングを作成。
2. レース、タイヤ、進行率、劣化、ラグ/ローリング、頻度、ターゲットエンコーディング系の特徴量を探索。
3. AB テスト型の特徴量選択で、コンパクトな特徴量セットを作成。
4. 21 個の選択特徴量で stage-2 LightGBM を学習。
5. 同じ選択特徴量で CatBoost を学習。
6. OOF 予測を使って LightGBM/CatBoost のブレンド重みを探索。
7. 順位ブレンドと複数の手動ブレンド案を検証。
8. 3 つ目のモデル候補として軽量 XGBoost を追加。
9. CatBoost 2000 iteration の追加実験を実施。
10. GitHub には小さな要約ファイルだけを残し、生データ、モデル本体、OOF 配列、提出 CSV は除外。

### 6. 結果

| 実験 | OOF AUC | 解釈 |
| --- | ---: | --- |
| Stage-2 LightGBM baseline | 0.955756 | 14 個の基本特徴量で強いベースライン。 |
| LightGBM AB-selected | 0.958394 | 特徴量選択により LightGBM が改善。 |
| CatBoost on selected features | 0.959197 | メインの流れで最も強い単体モデル。 |
| LightGBM + CatBoost probability blend | 0.960446 | 最良の OOF。重みは LightGBM 0.427、CatBoost 0.573。 |
| XGBoost lightweight model | 0.942677 | 最終ブレンドを改善するには弱かった。 |
| Three-model ensemble | 0.960446 | XGBoost の重みは 0.0 で、実質的に 2 モデルブレンドと同じ。 |
| CatBoost-2000 follow-up | 0.960022 | 強い単体追加実験だが、最良の LightGBM/CatBoost ブレンドには届かなかった。 |

### 7. 解釈

最も重要な学びは、広く大量の特徴量を入れるよりも、よく選ばれたコンパクトな特徴量セットの方が有効だったことです。時系列特徴量や外れ値特徴量を多く足しても、自動的に最終性能が上がるわけではありませんでした。性能向上に最も効いたのは、LightGBM と CatBoost という強く、かつ完全には同じでない 2 つの学習器を組み合わせることでした。

XGBoost は多様性を増やす候補でしたが、OOF が低かったため、重み探索では 0 が割り当てられました。ローカル OOF では、試した範囲では順位ブレンドより確率ブレンドの方が良い結果でした。

`Year` の分布差、レースごとの違い、コンパウンドの強い影響は、最終 notebook で丁寧に説明すべきです。これらは実際の戦略差、生成データの癖、または検証設計への感度を表している可能性があります。

### 8. GitHub に入れるファイル一覧

GitHub には、内容を理解し再実行できる最小構成だけを入れ、大きな生成物は入れない方針です。

| パス | 何のファイルか | なぜ入れるか |
| --- | --- | --- |
| `README.md` | プロジェクト概要と再実行手順。 | リポジトリの入口として必要。 |
| `REPORT.md` | この一体型の二言語レポート。 | 振り返り、EDA、結果、ファイル一覧を一箇所にまとめるため。 |
| `.gitignore` | Git の除外ルール。 | データ、提出物、予測、モデル、キャッシュ、ローカル設定を除外するため。 |
| `.gitattributes` | テキスト正規化ルール。 | 改行コードを安定させるため。 |
| `requirements.txt` | Python 依存ライブラリ。 | 実行環境を再現しやすくするため。 |
| `src/*.py` | モデリング用スクリプト。 | 特徴量作成、LightGBM、CatBoost、XGBoost、アンサンブル処理を残すため。 |
| `outlier_analysis.py` | 外れ値 EDA スクリプト。 | 初期の外れ値調査を残すため。 |
| `reports/key_results.csv` | 主要スコアの小さな表。 | 重要な実験結果をすぐ確認できるようにするため。 |
| `data/README.md` | データ配置メモ。 | Kaggle データをどこに置くか示すため。 |
| `notebooks/README.md` | notebook 用フォルダのメモ。 | 後で振り返り notebook を置く場所を明示するため。 |
| `experiment_summary.csv` | 初期実験の時系列まとめ。 | どの特徴量追加や pruning が効いたか、効かなかったかを見るため。 |
| `feature_importance_latest.csv` | 特徴量重要度。 | 後の説明やグラフ作成に使えるため。 |
| `feature_effect_report_latest.csv` | 特徴量効果レポート。 | 特徴量の挙動を説明するため。 |
| `oof_calibration_by_flags_latest.csv` | フラグ別キャリブレーション。 | モデルの弱い部分を見るため。 |
| `outlier_analysis_report.csv` | 外れ値レポート。 | `outlier_analysis.py` の小さな EDA 成果物として残すため。 |
| `outlier_flag_effects_latest.csv` | 外れ値フラグ効果。 | 外れ値フラグが効いたかを記録するため。 |
| `pruning_sweep_summary_latest.csv` | pruning sweep の要約。 | うまくいかなかった pruning の記録として必要。 |
| `selected_features_latest.json` | 後半実験の選択特徴量。 | 履歴比較のため。 |
| `best_params_latest.json` | 後半実験のパラメータ。 | 履歴比較のため。 |
| `runs/f1_stage2_ensemble/` の小さな要約ファイル | stage-2 のベースラインと AB-selected の記録。 | LightGBM 改善の根拠を残すため。 |
| `runs/f1_lgbm_catboost_ensemble/` の小さな要約・重みファイル | CatBoost と 2 モデルブレンドの記録。 | 最良経路を説明するため。 |
| `runs/f1_weight_variants/` の小さな要約・グリッドファイル | ブレンド候補の記録。 | 確率ブレンドと順位ブレンドの比較のため。 |
| `runs/f1_xgb_three_model/` の小さな要約・重みファイル | XGBoost 追加実験。 | 3 モデル目が効かなかった理由を残すため。 |
| `runs/f1_cat2000_next/` の小さな要約・重みファイル | CatBoost-2000 追加実験。 | 強い追加実験との比較を残すため。 |

GitHub に入れないものは、`data/*.csv`、ルートの `submission.csv`、`runs/**/submission*.csv`、OOF/test 予測配列、`*.cbm` などのモデル本体、`.vs/`、`__pycache__/`、バイナリの Word 文書です。これらは大きい、生成可能、ローカル専用、または Markdown の方がレビューしやすいものです。
