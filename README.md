# F1 Pit Stop Prediction Retrospective

This repository is documented in both English and Japanese. English appears first, followed by Japanese translation.

## English

This is a cleaned-up retrospective repository for Kaggle Playground Series S6E5, "Predicting F1 Pit Stops".

Competition page: https://www.kaggle.com/competitions/playground-series-s6e5/overview

The full combined report is in `REPORT.md`. It includes the competition objective, column-level EDA, modeling work, results, interpretation, and a file inventory explaining what is included in GitHub and why.

Key local result:

```text
LightGBM AB-selected weight: 0.427
CatBoost weight:             0.573
OOF AUC:                     0.9604462705
```

Raw Kaggle data, generated submissions, OOF prediction arrays, and trained model binaries are intentionally excluded from GitHub.

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the Kaggle files in `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

## 日本語

このリポジトリの文書は英語と日本語の両方で書いています。最初に英語、そのあとに日本語訳を置いています。

これは Kaggle Playground Series S6E5 "Predicting F1 Pit Stops" の振り返り用に整理したリポジトリです。

コンペページ: https://www.kaggle.com/competitions/playground-series-s6e5/overview

一体型のレポートは `REPORT.md` にあります。コンペの目的、列ごとの EDA、実施したモデリング、結果、解釈、GitHub に入れるファイル一覧と理由をまとめています。

主なローカル結果:

```text
LightGBM AB-selected weight: 0.427
CatBoost weight:             0.573
OOF AUC:                     0.9604462705
```

Kaggle の生データ、生成された提出ファイル、OOF 予測配列、学習済みモデル本体は GitHub には入れない方針です。

### セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kaggle からダウンロードしたファイルを `data/` に置きます。

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```
