# F1ピットストップ予測：振り返りレポート

このファイルでは、英語版の本文を削除し、日本語レポートだけに整理しています。画像パス、URL、コード上の列名、ファイル名、モデル名は参照が壊れないようにそのまま残しています。

---

## 0. 含めた項目のチェックリスト

この版では、コンペ目的、専門用語、データ概要、元データ、ローカル実行理由、各列の意味、画像付きEDA、実験ログ、スコア改善・悪化要因、良かった特徴量・悪かった特徴量と数値、解釈、最終モデルでやったこと、GitHub構成、上位者との差分、RealMLP、ブレンド/順位ブレンド、今後のKaggleへの一般化した学びをすべて含めています。

---

## 1. コンペの目的

このKaggleコンペの目的は、F1のラップ単位データから、次のラップでピットインするかどうかを予測することでした。目的変数は `PitNextLap` で、0または1の二値です。1なら「次のラップでピットインする」、0なら「次のラップではピットインしない」という意味です。

これは **二値分類** です。提出では、テストデータの各 `id` に対して、`PitNextLap` になる確率のような値を出します。評価指標はROC AUCなので、確率そのものの絶対値よりも、「実際にピットインする行を、ピットインしない行より上に並べられているか」が重要です。

---

## 2. 登場する専門用語の解説

| 用語 | 説明 | 今回の具体例 |
|---|---|---|
| 二値分類 | 結果が2種類の分類問題。 | `PitNextLap` が0か1。 |
| 目的変数 | モデルが予測したい列。 | `PitNextLap`。 |
| 特徴量・説明変数 | モデルに入力する列。 | `TyreLife`, `LapNumber`, `Compound`。 |
| 元特徴量 | CSVに最初から入っている列。 | `RaceProgress`。 |
| 作成特徴量 | 元の列から新しく作った特徴量。 | `Race_Compound_Stint`, ローリング平均。 |
| 学習データ | 目的変数があり、学習に使うデータ。 | `train.csv`。 |
| テストデータ | 目的変数がなく、予測を提出するデータ。 | `test.csv`。 |
| 元データ | Playgroundの学習データ/テストデータとは別の元データ・外部データ。 | 追加学習や分布確認に使えるが、分布差確認が必要。 |
| 合成データ | 直接の生データではなく、生成・加工されたデータ。 | Playground系列のデータ。 |
| ROC AUC | 正例を負例より上に順位付けできるかを見る指標。 | ピットインする行を上に並べるほど高い。 |
| OOF予測 | Out-of-Fold prediction。その行を学習に使っていないモデルで予測した値。 | モデル比較やブレンドに使う。 |
| 交差検証 | 学習/検証を分けて汎化性能を見る方法。 | OOF AUCを計算する。 |
| 層化K分割 | 目的変数の比率を保って分割するCV。 | 不均衡な二値分類で便利。 |
| グループ交差検証 | 同じグループを学習/検証に混ぜないCV。 | `Race`単位で分けるとリーク対策になる可能性。 |
| リーク | 本来予測時に使えない未来情報が混ざること。 | 未来のピットインを直接表す特徴量は危険。 |
| 特徴量エンジニアリング | 予測に役立つ列を新しく作ること。 | `Race_Year`, 頻度エンコーディングなど。 |
| 頻度エンコーディング | カテゴリの出現回数・頻度を特徴量にする。 | `Driver_count`, `Race_freq`。 |
| ターゲットエンコーディング | カテゴリごとの目的変数平均を特徴量化する。 | `Race_Compound`ごとのピット率。ただしCV内で安全に作る必要がある。 |
| ローリング特徴量 | 直近数行の平均などを使う特徴量。 | 直近3周の`LapTime_Delta`平均。 |
| ラグ特徴量 | 1つ前など過去の値を使う特徴量。 | 前ラップの`LapTime_Delta`。 |
| グループ統計量 | グループごとの平均・標準偏差など。 | `Race_Year`ごとの`LapTime_Delta`平均。 |
| Zスコア・平均との差 | グループ平均との差・標準偏差何個分か。 | そのレース内で今のラップがどれくらい遅いか。 |
| アブレーション・ABテスト | 特徴量を足す/抜く比較実験。 | AB選択特徴量。 |
| アンサンブル | 複数モデルを組み合わせること。 | LightGBM + CatBoost。 |
| ブレンド | 予測値を重み付き平均すること。 | `0.427 * LGBM + 0.573 * CatBoost`。 |
| 順位ブレンド | 確率ではなく順位を混ぜること。 | AUC向けに有効なことがある。 |
| スタッキング | OOF予測を入力にして2段目モデルを学習すること。 | ロジスティック回帰のスタッカーなど。 |
| メタデータ / メタ特徴量 | 元特徴量ではなく、1段目モデルのOOF予測を列として並べたデータ。 | `cat_oof`, `lgbm_oof`, `realmlp_oof` など。 |
| メタモデル | メタデータを使って、モデル予測の混ぜ方を学習する2段目モデル。 | `LogisticRegression`, `AutoGluon` など。 |
| Optuna | ハイパーパラメータを自動探索するツール。 | 複数の設定を試し、CV AUCが高いものを探す。 |
| LightGBM | 高速な勾配ブースティング木モデル。 | 強い表形式データベースライン。 |
| CatBoost | カテゴリ変数に強い勾配ブースティング木モデル。 | 今回の強い単体モデル。 |
| XGBoost | 伝統的で強力な勾配ブースティング木モデル。 | 今回は弱く、最終重み0。 |
| RealMLP | 表形式データ向けに改善されたMLP。 | 自分は未使用だが、上位者が使用。 |

---

## 3. データ概要

### 3.1 メインのコンペファイル

| データセット | 行数 | 列数 | 欠損セル数 | 欠損率 | 重複行数 |
|:--|--:|--:|--:|--:|--:|
| train | 439140 | 16 | 0 | 0 | 0 |
| test | 188165 | 15 | 0 | 0 | 0 |
| sample_submission | 188165 | 2 | 0 | 0 | 0 |

`train`は439,140行、16列で、目的変数 `PitNextLap` を含みます。`test`は188,165行、15列で、目的変数はありません。目的変数の陽性率は約19.90%です。

このデータは不均衡ですが、極端にまれなイベントではありません。今回扱うモデルはLightGBMやCatBoostのような勾配ブースティングモデルなので、この程度の不均衡はAUC評価では大きな問題になりにくいです。

### 3.2 元データについて

KaggleのPlayground `train.csv` / `test.csv` とは別に、F1戦略系の **元データ** も参照・追加候補として扱いました。元データは、Playgroundの合成データとは別系統のデータで、より元のF1戦略パターンを含んでいる可能性があります。

ただし、元データはそのまま混ぜればよいわけではありません。列名、目的変数定義、分布、欠損、外れ値、セッションの意味がコンペデータと違う可能性があります。そのため、正しい使い方は以下です。

1. 列名と意味をコンペデータに合わせる。
2. 予測時に使えない列を除外する。
3. 学習データ/テストデータ/元データの分布差を見る。
4. 元データあり/なしでCVを比較する。
5. OOFとリーダーボードの挙動が安定する場合だけ採用する。

---

## 4. ローカルで実行したこととその理由

多くの実験はKaggle NotebookではなくローカルPCで実施しました。理由は、スクリプトを何度も回しやすく、OOF予測、特徴量重要度、選択特徴量、キャリブレーション表、提出ファイル候補などの中間ファイルを保存しやすかったからです。

次回の理想は、ローカルで素早く実験し、OOF/テスト予測をすべて保存し、最後に再現可能なKaggle NotebookやGitHubレポートにまとめる流れです。

---

## 5. 各列の意味とEDA上の注意

| 列 | 意味 | 型 | モデリング上の注意 |
|:--|:--|:--|:--|
| id | 行ごとの一意な識別子。 | ID | 学習には入れない。 |
| Driver | ドライバーコードまたは匿名化ID。ドライバー・チーム・戦略の癖を含む可能性がある。 | 高カーディナリティカテゴリ | 頻度特徴量やCatBoostのカテゴリ処理と相性がよい。 |
| Compound | ハード, ミディアム, ソフト, インターミディエイト, ウェットなどのタイヤ種類。 | カテゴリ | 戦略文脈として強い。 |
| Race | グランプリまたはセッション名。 | カテゴリ | Race × Yearなどの交互作用が効きやすい。 |
| Year | シーズン年。 | 順序・離散数値 | 分布差が大きいため、検証設計と合わせて慎重に扱う。 |
| PitStop | 現在行のピットストップ関連フラグ。 | 二値・離散 | 有用だが、リークに近い挙動がないか確認が必要。 |
| LapNumber | 現在のラップ番号。 | 順序数値 | 戦略的なピットインのタイミング帯を表しやすい。 |
| Stint | 現在のスティント番号。 | 順序数値 | `Stint` 2のピット率が高く、TyreLifeやRaceProgressと相互作用する。 |
| TyreLife | 現在のタイヤを使った周回数。 | 数値 | 目的変数との関係が最も強い数値特徴量。 |
| Position | 現在順位。 | 順序数値 | 単体相関は弱いが、戦略文脈では意味を持つ。 |
| LapTime (s) | ラップタイム秒。 | 連続数値 | 大きな外れ値があり、レース内相対値の方が役立ちやすい。 |
| LapTime_Delta | 基準ラップタイムとの差。 | 連続数値 | ローリングや平均との差分の方が効きやすい可能性。 |
| Cumulative_Degradation | 累積の劣化・ペース低下シグナル。 | 連続数値 | クリッピングや相対特徴量が重要。 |
| RaceProgress | レースまたはセッションの進行率。 | 0〜1の連続数値 | StintやTyreLifeとの交互作用が重要。 |
| Position_Change | 順位変化。 | 離散数値 | 荒れた局面や戦略変更のサインになりうる。 |
| PitNextLap | 目的変数。1なら次ラップでピットインする。 | 二値目的変数 | AUC評価用に確率として予測する。 |

---

## 6. EDA：何を可視化し、何が分かったか

EDAでは、`id`を除く全カラムについて可視化しました。数値カラムならヒストグラム、箱ひげ図、`PitNextLap=0/1`別の分布比較、目的変数との平均値比較、相関係数、必要に応じたbin別ピット率を確認しています。カテゴリカラムなら、棒グラフ、カテゴリ別件数、カテゴリ別ピット率、上位カテゴリ、必要に応じた割合確認を行っています。

### 6.1 目的変数との相関

| 列 | corr_with_目的変数 | abs_corr_with_目的変数 |
|:--|--:|--:|
| TyreLife | 0.2735 | 0.2735 |
| LapNumber | 0.2671 | 0.2671 |
| Stint | 0.1982 | 0.1982 |
| RaceProgress | 0.1855 | 0.1855 |
| Cumulative_Degradation | -0.1674 | 0.1674 |
| Year | 0.1253 | 0.1253 |
| PitStop | 0.0486 | 0.0486 |
| Position_Change | 0.0462 | 0.0462 |
| LapTime (s) | -0.0341 | 0.0341 |
| Position | 0.0213 | 0.0213 |
| LapTime_Delta | -0.0049 | 0.0049 |

### 6.2 代表的な解釈

`TyreLife`, `LapNumber`, `Stint`, `RaceProgress` は、ピットインのタイミングをかなり直接的に表していました。一方で、`LapTime_Delta` は単純な線形相関では弱いですが、レース内の平均との差分やローリング特徴量にすると意味を持つ可能性があります。

画像パスは既存のGitHub構成を壊さないよう、そのまま利用します。

```text
reports/figures/eda/目的変数_distribution.png
reports/figures/eda/correlation_heatmap.png
reports/figures/eda/correlation_with_目的変数.png
reports/figures/eda/列/categorical_Driver.png
reports/figures/eda/列/categorical_Compound.png
reports/figures/eda/列/categorical_Race.png
reports/figures/eda/dataset_comparison/compare_Driver.png
reports/figures/eda/dataset_comparison/compare_Race.png
reports/figures/eda/dataset_comparison/compare_TyreLife.png
```

---

## 7. 実施したこと

| 手順 | 実施内容 | 理由 | 結果 |
|---|---|---|---|
| 1 | LightGBMの初期パイプライン作成 | 表形式データで強く速いベースラインを作るため | 学習・CV・提出の流れを作成 |
| 2 | AIに100個以上の特徴量を作らせてABテスト | 短時間で多くのアイデアを試すため | 0.93台で伸びず、破棄 |
| 3 | 問題構造から作り直し | AI特徴量がF1戦略理解に結びついていなかったため | 安定した第2段階ベースラインへ |
| 4 | 安全な数値・ドメイン特徴量を追加 | タイヤ寿命、ラップ番号、進行率、劣化がピット判断に関係するため | 0.944248まで改善 |
| 5 | 広い特徴量追加 | レース/year/contextの情報を足すため | 初期流れで0.944658 |
| 6 | 第2段階LightGBMベースライン | コンパクトで強い特徴量セットへ | 0.955756 |
| 7 | AB選択済みLightGBM | 検証で効いた特徴量だけ残す | 0.958394 |
| 8 | CatBoost追加 | カテゴリ構造を別の形で拾うため | 0.959197 |
| 9 | LightGBM/CatBoost ブレンド | 異なる誤差を平均化するため | 0.960446 |
| 10 | XGBoost追加 | 3モデル目の多様性を試すため | 弱く、重み0.0 |
| 11 | CatBoost-2000 | 長めのCatBoostを試すため | 0.960022で最良ブレンド未満 |

---

## 8. 何がスコアを改善し、何が悪化させたのか

| 実験 | OOF AUC | スコアへの影響 | 解釈 |
|:--|:--|:--|:--|
| 初期の安全寄りな数値特徴量ベースライン | 約0.937000 | 出発点 | 広めの初期ベースライン。まだ特徴量設計が安定していなかった。 |
| 安全寄りな数値特徴量 | 0.944248 | 約+0.007248 | タイヤ寿命、レース進行率、ラップ関連、劣化関連がピット判断に近かった。 |
| 全体的な特徴量拡張 | 0.944658 | 安全寄りな数値特徴量から+0.000410 | レース・年・文脈特徴量は効いたが、同時にノイズも増えた。 |
| 第2段階のLightGBMベースライン | 0.955756 | 新しい強いベースライン | コンパクトで整理された特徴量セットにより大きく改善した。 |
| LightGBMのAB選択特徴量 | 0.958394 | 第2段階LightGBMから+0.002638 | 検証で効く特徴量だけ残したことでノイズが減った。 |
| 選択特徴量を使ったCatBoost | 0.959197 | AB選択LightGBMから+0.000803 | カテゴリやルール構造の扱いがLightGBMと異なり、単体モデルとして最も強かった。 |
| LightGBM + CatBoostの確率ブレンド | 0.960446 | CatBoost単体から+0.001249 | 2つの強いモデルの誤差が完全には一致しなかったため、平均により順位付けが改善した。 |
| 軽量版XGBoost | 0.942677 | 最終ブレンドには低すぎた | 予測の多様性よりノイズの方が大きく、最終重みは0.0になった。 |
| CatBoost 2000回の追加実験 | 0.960022 | 最良ブレンドより-0.000424 | 強い単体実験だったが、2モデルブレンドは超えなかった。 |

大事な修正として、OOF自体が「高い/低い」のではありません。OOFは予測の作り方です。高い/低いと言うべきなのは、OOF予測から計算した **OOF AUC** です。

---

## 9. 良かった特徴量・悪かった特徴量・数値・理由

| 出所 | 特徴量・手法 | 評価 | 根拠・数値 | 理由 |
|:--|:--|:--|:--|:--|
| 手動・ドメイン理解 | `TyreLife`, `LapNumber`, `RaceProgress`, `Stint`の交互作用 | 良い | `TyreLife`相関0.2735、`LapNumber`相関0.2671、第2段階ベースライン0.955756 | ピット戦略のタイミングに直接関係する。 |
| 手動・ドメイン理解 | レース局面とタイヤ寿命比率 | 良い | 安全寄りな数値特徴量で約0.937から0.944248へ改善 | タイヤがレース局面に対してどれだけ古いかが、ピット判断に近い。 |
| 手動・検証 | AB選択特徴量セット | 良い | LightGBMが0.955756から0.958394へ改善 | 1つずつ特徴量を検証し、ノイズを減らせた。 |
| 手動・アンサンブル | LightGBM + CatBoostブレンド | とても良い | 0.959197から0.960446へ改善 | モデル固有の順位付けミスが平均化された。 |
| AI生成・広範囲 | 100個以上の自動生成特徴量 | 初回は悪い | 0.93台にとどまり、破棄した | F1戦略と結びつかない特徴量が多く、ノイズが増えた。 |
| AI生成・特徴量削減 | 重要度ベースで150/180特徴量に削減 | 悪い | 全特徴量ベースラインから約-0.0010〜-0.0011 | 弱くても補完的な特徴量まで落としてしまった可能性がある。 |
| 上位ノートブックの案 | 頻度エンコーディング | 試すべきだった | 自分の最終ログには未実施。上位ノートブックで確認 | カテゴリがよく出るか、まれでノイズが大きいかをモデルに伝えられる。 |
| 上位ノートブックの案 | グループ平均・標準偏差・平均との差分 | 試すべきだった | 自分の最終ログには未実施。上位ノートブックで重要 | 同じレース、年、タイヤ、ドライバー文脈の中で、そのラップが異常かどうかを表せる。 |
| 上位ノートブックの案 | RealMLP | 試すべきだった | 自分の最終モデルでは未使用。上位ノートブックでは単体またはブレンドで使用 | 木モデルとは違う誤差を出せるため、アンサンブルの多様性が増える。 |

---

## 10. 最終モデルで実際にスコアを上げた処理

最終的に強かった流れは以下です。

```text
第2段階LightGBMベースライン: 0.955756
→ AB選択済みLightGBM: 0.958394（+0.002638）
→ CatBoost単体モデル: 0.959197（AB選択LightGBMから+0.000803）
→ LGBM/CatBoost確率ブレンド: 0.960446（CatBoost単体から+0.001249）
```

スコアが上がった理由は4つです。

1. **特徴量セットをきれいにしたこと**：AI生成の大量特徴量をやめ、F1のピット戦略に結びつく特徴量へ戻した。
2. **AB特徴量選択**：それっぽい特徴量を全部採用せず、OOF AUCで効いたものだけを残した。
3. **CatBoost追加**：LightGBMとは異なるカテゴリ処理・分割で、別のパターンを拾えた。
4. **確率ブレンド**：LightGBMとCatBoostの予測は似ているが完全には同じでなく、平均することで順位が改善した。

---

## 11. モデルのアルゴリズム、強み・弱み、ユースケース、ハイパーパラメータ

### 11.1 Optuna

Optunaは、ハイパーパラメータを自動で探してくれるツールです。ただし、search space、CV設計、評価指標、特徴量のリーク対策、実行時間の上限は人間が決める必要があります。

### 11.2 LightGBM

LightGBMは勾配ブースティング木モデルです。たくさんの小さな決定木を順番に作り、前の木が間違えたところを次の木が修正していきます。

**使いどころ**：大きめの表形式データ、数値特徴量が多い問題、速いベースライン作成、特徴量重要度確認。

**強み**：速い、強い、表形式データのKaggleコンペで定番、特徴量が多くても扱いやすい。

**弱み**：カテゴリ変数の扱いは工夫が必要。複雑にしすぎると過学習する。

### 11.3 CatBoost

CatBoostも勾配ブースティング木モデルですが、カテゴリ変数の扱いが特に得意です。`Driver`、`Race`、`Compound`のような文字カテゴリが多い今回のデータとは相性が良かったです。

**使いどころ**：カテゴリ変数が多い表形式データ、高カーディナリティカテゴリ、表形式データのKaggleコンペ。

**強み**：カテゴリ処理が強い、デフォルトが比較的強い、今回のようなレース/ドライバー/タイヤ種類文脈に強い。

**弱み**：LightGBMより遅い場合がある。CV設計が甘いと大量カテゴリ特徴量で過学習して見える可能性がある。

### 11.4 XGBoost

XGBoostは古くから強い定番の勾配ブースティング木モデルです。今回は他モデルより弱く、最終ブレンドには大きく貢献しませんでした。

### 11.5 RealMLP

RealMLPは、表形式データ向けに改善されたニューラルネットワークです。通常のMLPは表形式データで木モデルに負けやすいですが、RealMLPは前処理、正則化、学習設定、デフォルト値などを工夫して、表形式データでも戦えるようにしたモデルです。

**ユースケース**：すでにLightGBM/CatBoostが強いが、別系統モデルをブレンドしたいとき。中〜大規模の表形式分類・回帰。モデル多様性が欲しいKaggle終盤。

**強み**：木モデルと違う誤差を出せる、滑らかな特徴量の組み合わせを拾いやすい、アンサンブルの多様性を増やせる。

**弱み**：前処理やスケーリングに敏感、乱数シードや学習時間に影響される、木モデルより解釈しづらい、環境構築がやや面倒。

---

## 12. 手動特徴量 vs AI生成特徴量

最初にAIへ100個以上の特徴量を作らせ、それらをABテストして精度が上がったものだけを使う実験を行いました。しかし、結果的に0.93台と低かったため、最終的にすべてを破棄し、1から作り直しました。

この経験から、AIに頼りすぎても良い結果が出るとは限らないと学びました。重要なのは、F1のピット戦略という問題構造を理解し、「なぜ次のラップでピットインするのか」を説明できる特徴量を作ることです。

AIはアイデア出しには便利ですが、採用するかどうかはOOF AUC、分割ごとの安定性、リークの有無、学習データ/テストデータ分布差を見て判断する必要があります。

---

## 13. 上位者との差分分析

### 13.1 自分のモデルの位置づけ

自分のモデルは、LightGBM、CatBoost、XGBoostを中心にした勾配ブースティング系のモデルでした。最終的にはLightGBM/CatBoostブレンドが強く、かなり戦えるスコアまで到達しました。

ただし上位者は、特徴量の文脈化、モデル多様性、最終ブレンドまでさらに踏み込んでいました。

### 13.2 上位者がやっていて自分が足りなかったこと

| 観点 | 自分 | 上位者 | なぜ効くか |
|---|---|---|---|
| カテゴリ交互作用 | 一部のみ | `Race_Year`, `Driver_Race`, `Race_Compound_Stint`など多数 | F1戦略は単体列ではなく文脈で決まるため。 |
| 頻度エンコーディング | ほぼ未使用 | ドライバー/レース/タイヤ種類の組み合わせ頻度を使用 | よく出るドライバーは戦略が安定し、まれな組み合わせはノイズと判断できるため。 |
| グループ統計量 | 限定的 | レース/年/タイヤ種類/スティント/ドライバーごとの平均・標準偏差 | そのレース内で普通より速い/遅いことを表せるため。 |
| 標準偏差特徴量 | 不十分 | グループ内標準偏差を使用 | そのラップが周囲よりどれだけ異常か分かるため。 |
| ラグ・ローリング特徴量 | 最終モデルでは弱い | 前ラップ・直近3ラップ平均など | ピットインは劣化の流れで起きるため。 |
| ターゲットエンコーディング | 中心ではない | CV分割内で安全に使用 | カテゴリごとのピットインしやすさを表せるため。 |
| RealMLP | 未使用 | 単体またはブレンドで使用 | 木モデルと違う予測を出せるため。 |
| ブレンド | 確率の重み付きブレンド中心 | OOF相関、順位ブレンド、安全なブレンド、スタッキング | AUCでは順位調整が重要なため。 |

### 13.3 OOF保存、相関、重み探索

上位者は、各モデルのOOF予測とテスト予測を保存し、モデル間の相関や重みを見ながらアンサンブルしていました。

```text
model_lgbm_oof.csv
model_catboost_oof.csv
model_realmlp_oof.csv
model_xgb_oof.csv
```

相関が高すぎるモデルを混ぜても新しい情報は少ないです。一方で、AUCがそこそこ高く、相関が低めのモデルはブレンドで効く可能性があります。

---

## 14. 最重要の反省：AIに100個以上の特徴量を作らせた実験

最初にAIへ100個以上の特徴量を作らせ、それらをABテストして精度が上がったものだけを残す実験を行いました。しかし、結果は0.93台にとどまり、最終的にその流れをすべて破棄して、問題構造を理解するところから作り直しました。

重要なのは、AIに特徴量をたくさん作らせることではなく、問題構造、リーク、CV設計、実験結果を見て判断することでした。

---

## 15. GitHubに入れるファイルの考え方

GitHubには、再現性と読みやすさに必要なものだけを入れます。

```text
f1Prediction/
  README.md
  REPORT.md
  requirements.txt
  .gitignore
  src/
    make_eda_report.py
    train_lgbm.py
    train_catboost.py
    ensemble.py
  reports/
    figures/eda/
    eda_summary_tables/
    key_results.csv
  data/
    README.md
```

生データ、元データ本体、提出ファイルCSV、大きなOOF/テスト予測配列、モデルバイナリはGitHubには入れない方針です。

---

## 16. 今後のKaggleで使える一般化した学び

1. 各モデルのOOF予測とテスト予測を必ず保存する。
2. 大量特徴量の前に、強いシンプルベースラインを作る。
3. 手動特徴量とAI生成特徴量を分けて管理する。
4. 特徴量は「なぜ目的変数に効くか」という仮説つきで作る。
5. 学習データ/テストデータ/元データの分布差を見る。
6. LightGBM/CatBoostだけでなくRealMLPのような非木モデルも試す。
7. ブレンド前にモデル間相関を見る。
8. AUCコンペでは確率ブレンドだけでなく順位ブレンドも試す。
9. 公開リーダーボードに寄せすぎない。
10. GitHubには再現可能なコード・図・小さな結果だけを入れる。

---

## 17. 参考・メモ

- LightGBM公式ドキュメント: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- CatBoost公式ドキュメント: https://catboost.ai/docs/en/references/training-parameters/
- XGBoost公式ドキュメント: https://xgboost.readthedocs.io/en/stable/parameter.html
- Optuna公式ドキュメント: https://optuna.readthedocs.io/
- PyTabKit / RealMLPリポジトリ: https://github.com/dholzmueller/pytabkit
- RealMLP論文: https://arxiv.org/abs/2407.04491

---

## 18. 1位writeupから学んだ上位者フロー

この章では、1位のwriteupをもとに、上位者がどのような発想でモデルを作っていたのかを整理します。自分の最終モデルは、LightGBMとCatBoostを強くしてブレンドする方向でした。一方で1位の人は、単体モデルを1つだけ強くするのではなく、**特徴量セット・モデル種類・ハイパーパラメータ・seed・original dataの使い方を変えた大量のOOFを作り、それをメタデータとして使って最終アンサンブルを作る**という流れでした。

### 18.1 モデルの種類を大量に増やした

2位の人は、以下のようなモデルを試して結果は表のとおりです。（結果の欄が空白のものは、GPTが考えて候補になりえる学習法を考えて追加したもの）

| rank | model/class | model count | best model | best CV AUC | public LB | private LB | どんなものか | 使いどころ・役割 |
|---:|---|---:|---|---:|---:|---:|---|---|
| 1 | RealMLP | 40 | realmlp2_exp147_five_seed_d... | 0.954426 | 0.95382 | 0.95421 | 表形式データ向けに強化されたMLP。木モデルとは違う形で数値特徴量の組み合わせを学ぶ。 | 今回の2位では最強クラス。LightGBM/CatBoostと違うOOFを作る主力。 |
| 2 | XGBoost | 36 | gpt1020_xgb_orighazard | 0.953553 | 0.95294 | 0.95354 | 定番の勾配ブースティング木モデル。正則化やobjectiveが豊富。 | LightGBM/CatBoostとは違うGBDT枠。自分は改善余地が大きい。 |
| 3 | CatBoost | 37 | gpt1016_cat_ctrte | 0.953404 | 0.95105 | 0.95190 | カテゴリ変数に強い勾配ブースティング木モデル。 | Driver, Race, Compoundなどカテゴリが多い今回の主力。 |
| 4 | TabM | 11 | tabm_exp089_wider_artifact_... | 0.953371 | 0.95304 | 0.95345 | 表形式データ向けのニューラルネット系モデル。 | RealMLPと並ぶNN系主力候補。 |
| 5 | LightGBM | 25 | lgbm_exp091_slow_lean_origi... | 0.953023 | 0.95267 | 0.95290 | 高速な勾配ブースティング木モデル。表形式データの定番。 | ベースライン、特徴量比較、複数特徴量セットのOOF作成に向く。 |
| 6 | TabICL | 8 | pri589_tabicl_v2_original_a... | 0.950827 | 0.95053 | 0.95085 | 表形式データ向けのin-context learning系モデル。 | GBDT/MLPとは違う予測を出す多様性枠。 |
| 7 | FFM | 3 | pri2_pri515_exp072_full_5_f... | 0.949178 | 0.95048 | 0.95070 | Field-aware Factorization Machine。カテゴリ同士のinteractionをfieldごとに学ぶ。 | Driver×Raceなどカテゴリ相互作用を拾う候補。 |
| 8 | Custom NN | 9 | nn_exp022_duplicate_low_card | 0.948923 | 0.95011 | 0.95050 | 自作ニューラルネット。embeddingやarchitectureを自由に設計できる。 | 特徴量やカテゴリembeddingの実験枠。 |
| 9 | RandomForest | 3 | pri2_pri520_exp125_full_5_f... | 0.948845 | 0.94856 | 0.94885 | 複数の決定木をbaggingする古典的アンサンブル。 | GBDTとは違う木モデル枠。 |
| 10 | GNN | 3 | pri2_pri516_exp079_full_5_f... | 0.947632 | 0.94873 | 0.94927 | Graph Neural Network。データ間の関係をグラフとして扱う。 | Driver/Race/Compoundなどの関係性を扱う多様性枠。 |
| 11 | HistGB | 1 | pub007_histboost | 0.947546 | 0.94742 | 0.94827 | Histogram-based Gradient Boosting。bin化して高速に木を学習する。 | LightGBMとは別実装のヒストグラムGBDT枠。 |
| 12 | FM | 3 | pri2_pri518_exp099_full_5_f... | 0.947381 | 0.94837 | 0.94902 | Factorization Machine。特徴量同士の2次相互作用を低次元で学ぶ。 | カテゴリinteractionが強いときの多様性枠。 |
| 13 | KNN | 1 | pri536_knn_7123 | 0.947231 | 0.94704 | 0.94719 | 近いデータ点の傾向から予測する距離ベースモデル。 | 木モデルと全く違う予測を作る多様性枠。 |
| 14 | Other | 8 | pri2_pri511_exp023_full_5_f... | 0.947160 | 0.94981 | 0.95038 | その他のモデル群。 | 個別には弱くてもensemble diversityのために入れる枠。 |
| 15 | TabTransformer | 3 | pri_exp043_tabtran_domain_s... | 0.947101 | 0.95003 | 0.95059 | Transformerを表形式データに使うモデル。 | カテゴリ文脈をattentionで拾う多様性枠。 |
| 16 | ExcelFormer | 1 | tal005_excelformer | 0.946912 | 0.94902 | 0.94984 | 表形式データ向けTransformer系モデル。 | 単体主力よりもTALENT系の多様性枠。 |
| 17 | Cox/survival | 1 | pri522_cox_8007 | 0.946209 | 0.94717 | 0.94747 | イベントがいつ起きるかを扱う生存時間分析モデル。 | ピットをラップ上のイベントとして見る特殊枠。 |
| 18 | DAE | 1 | pri545_dae_8906 | 0.946046 | 0.94655 | 0.94717 | Denoising AutoEncoder。ノイズ付き入力から元情報を復元するNN。 | 表現学習やNN系多様性枠。 |
| 19 | AMFormer | 1 | tal007_amformer | 0.944542 | 0.94570 | 0.94649 | 表形式データ向けTransformer系モデル。 | 単体より多様性目的。 |
| 20 | YDF GBDT | 1 | pri509_ydf_3000 | 0.943788 | 0.94325 | 0.94420 | Yggdrasil Decision ForestsのGBDT。 | 別実装の木モデル枠。 |
| 21 | MLP-PLR | 1 | tal009_mlp_plr | 0.943629 | 0.94428 | 0.94553 | MLPにPiecewise Linear Representationを組み合わせる表形式NN。 | 数値特徴量の非線形変換を拾う候補。 |
| 22 | Trompt | 1 | tal006_trompt | 0.943453 | 0.94479 | 0.94550 | 表形式データ向けprompt/attention系モデル。 | TALENT系の多様性枠。 |
| 23 | TabR | 1 | tal002_tabr | 0.943445 | 0.94579 | 0.94652 | 近傍・retrieval的な発想を使う表形式モデル。 | KNNやNNに近い多様性枠。 |
| 24 | AutoInt | 1 | tal011_autoint | 0.942763 | 0.94485 | 0.94564 | self-attentionで特徴量interactionを学ぶモデル。 | 高次interactionを拾う候補。 |
| 25 | GrowNet | 1 | tal012_grownet_fixed | 0.942110 | 0.94304 | 0.94439 | ニューラルネットをboosting的に積み上げるモデル。 | GBDTとNNの中間的な多様性枠。 |
| 26 | SAINT | 1 | tal010_saint_fixed | 0.941141 | 0.94407 | 0.94500 | 表形式データ向けTransformer。行・列方向のattentionを使う。 | 表形式Transformer枠。 |
| 27 | ModernNCA | 1 | tal001_modernnca | 0.941058 | 0.94179 | 0.94296 | 近傍表現を学ぶNCA系モデル。 | KNNに近い視点の多様性枠。 |
| 28 | Gemini-derived | 2 | pri555_gemini_9702 | 0.940993 | 0.94081 | 0.94178 | GeminiなどのLLMから派生したモデル/特徴量実験。 | AI生成モデルの多様性枠。 |
| 29 | DCN | 1 | tal008_dcn2 | 0.939530 | 0.94204 | 0.94263 | Deep & Cross Network。明示的な特徴量crossを学ぶNN。 | cross featureをNNで拾う候補。 |
| 30 | GPT-derived | 1 | pri539_gpt_3c | 0.939197 | 0.94313 | 0.94369 | GPT由来のコード/特徴量/モデル実験。 | AI補助で作られた多様性枠。 |
| 31 | GANDALF | 1 | pri553_gandalf_2800 | 0.937605 | 0.93919 | 0.93990 | 表形式データ向けNN/attention系モデル。 | TALENT系の多様性枠。 |
| 32 | TabNet | 3 | pri_exp062_tabnet_combo_te | 0.937210 | 0.94094 | 0.94166 | attentionで特徴量を段階的に選ぶ表形式NN。 | 解釈可能性もあるNN枠。 |
| 33 | NODE | 1 | tal003_node | 0.936121 | 0.93559 | 0.93675 | Neural Oblivious Decision Ensembles。木とNNの中間的モデル。 | GBDTとNNの中間枠。 |
| 34 | Logistic regression | 1 | pri512_logreg_7011 | 0.933935 | 0.93381 | 0.93359 | 線形分類モデル。特徴量の線形結合で予測する。 | 単体は弱いがbaselineやメタモデルに有用。 |
| 35 | FTTransformer | 1 | pri517_ftt_6500 | 0.933224 | 0.93579 | 0.93585 | Feature Tokenizer + Transformer。表形式Transformerの代表例。 | Transformer系の多様性枠。 |
| 36 | Snap/artifact | 4 | pri544_snap_3500 | 0.932076 | 0.93169 | 0.93237 | Playground特有のartifact signalを拾うモデル/特徴量群。 | 合成データの癖を拾う枠。 |
| 37 | LNN | 1 | pri554_lnn_4400 | 0.893791 | 0.91096 | 0.91106 | 特殊な線形/論理/NN系モデルと考えられる弱い多様性枠。 | 今回の表ではかなり弱く、採用は慎重。 |
| 追加 | TabPFN / TabPFN v2 | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 表形式データ向けfoundation model。事前学習済みモデルで小〜中規模表データを予測する。 | 全量は重い可能性があるため、サブサンプル・特徴量絞り込み・Race別でOOF候補にする。 |
| 追加 | RTDL ResNet / MLP ResNet | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 表形式データ向けのResNet型MLP。木モデルとは違う滑らかな関係を拾う。 | RealMLPの次に試すNN baseline。 |
| 追加 | ExtraTreesClassifier | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | RandomForestよりランダム性が強い決定木アンサンブル。 | 単体最強ではないが、OOF相関が低めの多様性枠。 |
| 追加 | LGBMRanker | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | LightGBMのlearning-to-rankモデル。グループ内で順位を学習する。 | AUCは順位が重要なので、Race_Yearなどをgroupにして試す。 |
| 追加 | XGBRanker | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | XGBoostのrankingモデル。rank:ndcgやpairwise系objectiveを使える。 | pitする行を上に並べる目的に近い。 |
| 追加 | CatBoostRanker | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | CatBoostのranking用モデル。カテゴリ処理とranking objectiveを合わせられる。 | Race単位・Race_Year単位で順位学習する特殊候補。 |
| 追加 | EBM / Explainable Boosting Machine | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | GAM系の解釈しやすいboostingモデル。自動interaction detectionもある。 | 効果を見やすく、OOF多様性にも使える。 |
| 追加 | H2O AutoML | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | H2OのAutoML。GBM, DRF, XGBoost, GLM, Stacked Ensembleなどを自動で試す。 | AutoGluon/LightAutoML以外のAutoML枠。OOF保存してpoolに追加したい。 |
| 追加 | ElasticNet Logistic Regression | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | L1/L2を混ぜた正則化付きロジスティック回帰。 | 単体baseline、またはメタモデルの安定版。 |
| 追加 | LinearSVM / LinearSVC + calibration | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 線形SVM。確率化にはcalibrationが必要。 | 木モデルと違う線形境界を作る多様性枠。 |
| 追加 | SGDClassifier | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 確率的勾配降下法で学習する線形モデル。大規模データに軽い。 | log_lossやmodified_huberで軽量OOFを作れる。 |
| 追加 | Nystroem + Logistic Regression | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | カーネル近似で非線形特徴空間を作り、Logistic Regressionで学習する。 | RBF SVMの軽量近似として試す候補。 |
| 追加 | Random Survival Forest | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 生存時間分析用のRandomForest。イベント発生時間を扱う。 | ピットをいつ起きるイベントかとして見る特殊枠。 |
| 追加 | Discrete-time hazard model | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 各ラップでイベントが起きるhazardを分類的に学ぶモデル。 | PitNextLapと発想が近く、F1ピット予測に合う候補。 |
| 追加 | DeepSurv / DeepHit | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | ニューラルネットを使うsurvival/event-timeモデル。 | ラップ上のpit timingをNNで扱う特殊枠。実装コストは高い。 |
| 追加 | GaussianNB | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 特徴量がクラスごとに正規分布すると仮定するNaive Bayes。 | 単体は弱そうだが予測相関が低い可能性。 |
| 追加 | BernoulliNB | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 二値特徴量向けNaive Bayes。 | フラグ特徴量やone-hot中心の軽量baseline。 |
| 追加 | LDA | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | Linear Discriminant Analysis。クラスを線形境界で分ける生成モデル。 | シンプルな線形多様性枠。 |
| 追加 | QDA | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | Quadratic Discriminant Analysis。クラスごとに異なる共分散を持てる。 | 高次元では不安定なので特徴量を絞って試す。 |
| 追加 | AdaBoostClassifier | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 弱い決定木を順番に重み付けして学習する古典的boosting。 | GBDTとは違う古典boosting枠。 |
| 追加 | BaggingClassifier | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 同じモデルをbootstrap sampleで複数作って平均する。 | DecisionTreeやLogisticなどをbaggingして多様性OOFを作る。 |
| 追加 | sklearn GradientBoostingClassifier | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | scikit-learn標準の古典的GBDT。 | 別実装のGBDT枠。 |
| 追加 | DeepFM | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | FMとDeep Neural Networkを組み合わせ、低次・高次interactionを学ぶ。 | カテゴリinteractionが多い今回の多様性候補。 |
| 追加 | Wide & Deep | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 線形のwide部分とNNのdeep部分を組み合わせるモデル。 | 覚えたいカテゴリ規則と一般化したNN表現を両方使う。 |
| 追加 | xDeepFM | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 明示的・暗黙的な高次interactionを学ぶDeepFM拡張。 | Driver×Race×Compoundのような高次interaction候補。 |
| 追加 | FiBiNET | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | feature importanceとbilinear interactionを使うCTR系モデル。 | カテゴリinteraction重視の多様性候補。 |
| 追加 | AFM | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | Attention Factorization Machine。重要なinteractionにattentionをかける。 | FM系の中でも解釈しやすいinteraction枠。 |
| 追加 | LGBM DART | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | LightGBMのDART booster。木をdropoutしながらboostingする。 | 通常LGBMと違う正則化・予測を作れる。 |
| 追加 | LightGBM GOSS | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | Gradient-based One-Side Sampling。勾配の大きい行を重視するLightGBM設定。 | 通常のbagging LGBMと違うOOFを作る候補。 |
| 追加 | XGBoost DART booster | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | XGBoostのDART。木をdropoutして過学習を抑えるbooster。 | 通常XGBと違う予測を出す候補。 |
| 追加 | XGBoost survival:aft | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | XGBoostのsurvival objective。イベント時間を扱う。 | ピットタイミングをsurvival問題として見る候補。 |
| 追加 | CatBoost CrossEntropy | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | CatBoostの別loss設定。Loglossとは少し違う学習になる。 | CatBoostの予測バリエーション作成。 |
| 追加 | CatBoost CTR設定違い | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | CatBoostのカテゴリ統計量・CTRの設定を変える実験。 | Driver, Race, Compoundの扱いを変えたOOFを作る。 |
| 追加 | LightAutoML / FLAML variants | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | AutoMLで複数モデルやHPを自動探索する。 | 1位が使っていたAutoML枠。自分のfoldでOOF保存する。 |
| 追加 | AutoGluon meta model | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | AutoGluonを元特徴量ではなくOOF列に対して使うメタアンサンブラー。 | 1位の重要要素。LR with Logitsとは別のメタ予測を作る。 |
| 追加 | Ridge / RidgeClassifier meta | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | L2正則化された線形モデル。 | OOFメタデータの安定した重み付け候補。 |
| 追加 | LightGBM meta model | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | OOF列を入力にしたLightGBMの2段目モデル。 | 非線形なモデル組み合わせを学ぶが、過学習注意。 |
| 追加 | CatBoost meta model | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | OOF列を入力にしたCatBoostの2段目モデル。 | メタモデルの非線形候補。正則化が重要。 |
| 追加 | Simple MLP meta | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | OOF列を入力にした小さなニューラルネット。 | LRでは拾えないモデル間interactionを拾う候補。 |
| 追加 | EBM meta model | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | OOF列に対する解釈可能なメタモデル。 | どのOOFが効いているか見やすい。 |

ここで重要なのは、「モデル名をたくさん知っていたから強い」というより、**違う性格の予測をたくさん作った**ことです。

例えば、LightGBMとCatBoostはどちらも木モデルですが、カテゴリ変数の扱い、分割の仕方、正則化の効き方が違います。RealMLPやTabMのようなニューラルネット系モデルは、木モデルとは違う誤差を出す可能性があります。単体性能が少し低くても、他モデルと違う間違い方をするなら、アンサンブルで価値があります。

自分はLightGBM、CatBoost、XGBoostを中心にしていました。これは悪い選択ではありません。しかし、上位者はその外側にあるRealMLP、TabM、FT-Transformer、MLP-PLRなども加えて、予測の多様性を増やしていました。

---

### 18.2 Heavy FEで50〜400特徴量のセットを複数作った

自分は、特徴量を作ってABテストし、良さそうなものを残す方向で進めました。これは基本としては正しいです。ただし、上位者の発想は少し違います。

上位者は、**1つの最終特徴量セットを厳選する**というより、**テーマごとに偏った特徴量セットを複数作り、それぞれをモデルごとにOOF化する**という発想でした。

例としては、以下のような特徴量セットです。

| Feature Set | 内容 | 目的 |
|---|---|---|
| Feature Set A | 基本特徴量だけ。およそ50特徴量。 | 安定したbaselineを作る。 |
| Feature Set B | 基本特徴量 + タイヤ状態・劣化系を大量追加。およそ200特徴量。 | タイヤ寿命、劣化、stintの限界を拾う。 |
| Feature Set C | 基本特徴量 + Lap Anomaly / Rolling Set。 | ピット直前のラップ遅れ、直近数周の悪化、不安定さを検知する。 |
| Feature Set D | 基本特徴量 + categorical / target encoding / frequency encoding系。 | Driver、Race、Compound、Stintの癖を拾う。 |
| Feature Set E | 作った特徴量をかなり全部入れる。数百特徴量。 | モデルに広く情報を見せ、使えるものを選ばせる。 |
| Feature Set F | Driver関連特徴量を落としたセット。 | Driver分布差に依存しないモデルを作る。 |

これらをモデルと組み合わせます。

```text
A + CatBoost → OOF_001
A + LGBM     → OOF_002
B + CatBoost → OOF_003
B + XGB      → OOF_004
C + RealMLP  → OOF_005
D + CatBoost → OOF_006
E + LGBM     → OOF_007
F + CatBoost → OOF_008
```

ここでの重要な考え方は、**特徴量は単体で良い/悪いが決まるのではなく、モデルとの相性によって有用度が変わる**ということです。

例えば、`LapTime_Delta_rolling_std_5` はLightGBMではあまり効かなくても、CatBoostやRealMLPでは効くかもしれません。逆に、カテゴリ交互作用はCatBoostでは強くても、ニューラルネット系では扱いづらいかもしれません。だから、特徴量を1つの最終セットに絞りすぎず、複数の特徴量セットとして残し、それぞれをOOF化することに意味があります。

---

### 18.3 Original dataとの分布差を真剣に見た

1位の人は、original dataを追加するとAUCは上がる一方で、competition dataとは分布差があると考え直しました。特に `Driver` が大きく違う特徴量として目立ったため、Driverを使わないモデルも作っています。

自分もoriginal dataを使う発想はありましたが、主に「入れる/入れない」で考えていました。上位者はそれだけでなく、以下のように、original dataをどれくらい信用するかまで調整していました。

```text
original data weight = 1.0
original data weight = 0.75
original data weight = 0.5
original data weight = 0.25
original dataなし
```

これは **sample weight** の調整です。

普通に学習すると、competition dataもoriginal dataも1行あたり同じ重みで学習されます。

```text
competition data: weight = 1.0
original data:    weight = 1.0
```

しかし、original dataが少しズレているなら、以下のように重みを下げます。

```text
competition data: weight = 1.0
original data:    weight = 0.5
```

これは、original dataを完全に捨てるのではなく、**使うけど少し信頼度を下げる**という考え方です。元データは情報量がありますが、competition dataとズレているなら、100%信じると悪さをする可能性があります。だから、original data weightを複数パターン作り、それぞれをOOF化して、最後のアンサンブル材料にします。

---

### 18.4 182個のL1 OOFを作った

1位の人は、最終的に182個のL1 OOFを使っていました。OOFは **Out-of-Fold prediction** の略で、「その行を学習に使っていないモデルが出した予測値」です。

例えば、5個のOOFだけで簡単に表すと、以下のようになります。

| row | 正解 PitNextLap | cat_full_oof | lgbm_tyre_oof | xgb_lap_oof | realmlp_oof | cat_no_driver_oof |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.08 | 0.12 | 0.10 | 0.18 | 0.09 |
| 2 | 1 | 0.82 | 0.76 | 0.70 | 0.68 | 0.61 |
| 3 | 0 | 0.44 | 0.30 | 0.35 | 0.51 | 0.22 |
| 4 | 1 | 0.65 | 0.79 | 0.75 | 0.60 | 0.77 |

これは、1行目のtrainデータに対して、元特徴量の代わりに、各モデルのOOF予測を並べている表です。

例えば、1行目では、`cat_full_oof = 0.08` なので、CatBoostのfull feature modelは「この行は次のラップでpitする確率が低い」と見ています。一方で、2行目では `cat_full_oof = 0.82`、`lgbm_tyre_oof = 0.76` なので、複数モデルがpit寄りに予測しています。

このOOF表が **メタデータ / メタ特徴量** です。182個のOOFを使う場合、このメタデータには182列あります。行数は182行ではなく、trainデータの行数と同じです。

```text
meta_train.shape = train行数 × 182列
meta_test.shape  = test行数 × 182列
```

train側はOOFを使い、test側は各モデルのtest predictionを使います。

---

### 18.5 OOFを大量に作る意味

OOFを大量に作る目的は、単に数を増やすことではありません。目的は、**違う見方をする予測列を増やすこと**です。

例えば、以下のような違いを作ります。

```text
CatBoost full features
CatBoost no Driver
CatBoost originalなし
CatBoost original_weight=0.5
CatBoost category heavy
CatBoost tyre heavy
LGBM rolling heavy
RealMLP all heavy
XGB lap anomaly
```

これらは同じデータを見ていますが、見方が違います。だから、予測のズレに意味が出ます。

例えば、ある行で以下のような予測になったとします。

```text
CatBoost full:      0.90
LGBM tyre heavy:    0.55
CatBoost no Driver: 0.35
```

この場合、CatBoost fullだけが高く、Driverなしモデルは低いです。これは、「Driver特徴量に引っ張られて高く出ているだけかもしれない」と判断できます。

一方で、以下のような場合は違います。

```text
CatBoost full:      0.82
LGBM tyre heavy:    0.79
CatBoost no Driver: 0.77
RealMLP:            0.75
```

この場合、Driverあり/なし、木モデル/ニューラルネット系を問わず、複数モデルがpit寄りに判断しています。これは本当にpitする可能性が高いと考えやすいです。

このように、OOFを使うと、単純に「どのモデルが一番強いか」だけではなく、**モデル同士の一致・不一致のパターン**を使えます。

---

### 18.6 メタモデルは何を学習しているのか

メタモデルは、元の特徴量ではなく、OOF予測を入力として使います。

1段目モデルでは、以下を使います。

```text
TyreLife
RaceProgress
Compound
Driver
LapTime_Delta
Cumulative_Degradation
Position
...
```

2段目のメタモデルでは、以下を使います。

```text
catboost_full_oof
lgbm_tyre_oof
xgb_lap_oof
realmlp_oof
catboost_no_driver_oof
...
```

つまり、メタモデルは「元データを見るモデル」ではなく、**モデルたちの意見を見るモデル**です。

メタモデルは、例えば以下のようなことを学習できます。

```text
CatBoostとLGBMが両方高いなら、本当にpitしそう。
CatBoostだけ高くてDriverなしモデルが低いなら、Driver特徴量に引っ張られている可能性がある。
original dataありモデルだけ高いなら、元データの癖を拾っている可能性がある。
RealMLPは単体では少し弱いが、特定のパターンでは役に立つ。
```

自分の最終ブレンドは、基本的に以下のような固定比率でした。

```text
final = 0.6 * LGBM + 0.4 * CatBoost
```

これは良い方法ですが、常に同じ重みです。一方で、OOFを使ったメタモデルは、行ごとに柔軟に判断できます。

```text
この行ではCatBoostを強めに信じる。
この行ではDriverなしモデルを重視する。
この行ではRealMLPの意見も使う。
```

これが、普通の手動ブレンドより柔軟な点です。

---

### 18.7 182個のOOFはどう作られたのか

182個は、182種類のアルゴリズムという意味ではありません。以下の掛け合わせで作られた、182本の予測列です。

```text
モデル種類 × 特徴量セット × ハイパーパラメータ × seed × original data設定
```

ただし、これは総当たりではありません。すべてを掛け算すると数千通りになってしまいます。実際には、良さそうな組み合わせをピックしながら実験し、OOFとして保存したものが182個になったという理解が自然です。

自分用の推定例としては、以下のような内訳が考えられます。

| グループ | だいたいのOOF数 | 内容 |
|---|---:|---|
| CatBoost系 | 40〜50個 | 特徴量セット違い、depth違い、original weight違い、Driver有無 |
| LGBM系 | 40〜50個 | feature set違い、正則化違い、num_leaves違い、seed違い |
| XGB系 | 20〜30個 | max_depth違い、colsample違い、feature set違い |
| RealMLP / TabM系 | 15〜25個 | NN系モデル、特徴量セット違い、seed違い |
| HGB / RF / YDF系 | 10〜20個 | 木系だがLGBM/CatBoostと違う癖を出す |
| FT-Transformer / MLP-PLR系 | 5〜15個 | ニューラル系の多様性枠 |
| AutoML系 | 10〜25個 | AutoGluon / LightAutoML / FLAML / PyTabKit由来 |
| Driver dropped / originalなし専用系 | 20〜30個 | 分布差対策のモデル群 |

この表は1位本人が公開した正確な内訳ではなく、自分が理解するための推定です。重要なのは、数そのものではなく、**似たモデルを増やすのではなく、違う間違い方をするOOFを増やす**ことです。

---

### 18.8 LR with Logitsとは何か

最後のアンサンブルで重要なのが **LR with Logits** です。

各モデルの予測値は0〜1の確率です。

```text
CatBoost = 0.80
LGBM     = 0.70
RealMLP  = 0.60
```

これをそのままLogistic Regressionに入れるのではなく、まず **logit変換** します。

```text
logit(p) = log(p / (1 - p))
```

例えば、

```text
p = 0.80
logit(0.80) = log(0.80 / 0.20) = log(4) ≈ 1.386
```

`p = 0.20` なら、

```text
logit(0.20) = log(0.20 / 0.80) = log(0.25) ≈ -1.386
```

つまり、0.5より大きい予測は正の値、0.5より小さい予測は負の値になります。

その後、Logistic Regressionで以下のように重みを学習します。

```text
final_score =
  w1 * logit(catboost_oof)
+ w2 * logit(lgbm_oof)
+ w3 * logit(realmlp_oof)
+ ...
+ bias
```

最後にsigmoidで0〜1の確率に戻します。

この方法の良いところは、単純平均よりも柔軟に、各モデルの信頼度を学習できることです。強いOOFには大きな重み、似すぎているOOFには小さな重み、ノイズが多いOOFにはほぼ0の重みを与えられます。

---

### 18.9 AutoGluonをメタアンサンブラーとして使った

自分は主に手動ブレンドでした。

```text
0.6 * LGBM + 0.4 * CatBoost
```

これは良い方法です。ただし、1位の人はAutoGluonをメタアンサンブラーとして使い、多数のOOFから最終予測を作っています。

AutoGluonはAutoMLライブラリです。メタデータを渡すと、内部で複数のモデルやスタッキングを試し、どのOOFをどう混ぜるかを自動で学習します。

ここでの入力は、元特徴量ではなく、OOF列です。

```text
meta_train:
  oof_001
  oof_002
  ...
  oof_182

target:
  PitNextLap
```

AutoGluonは、単純な線形重みでは拾えない関係を拾える可能性があります。例えば、「CatBoostが高く、no Driverモデルも高いときは強くpit寄りにする」「originalありモデルだけ高いときは信用しすぎない」など、複雑なパターンを学習できる可能性があります。

一方で、AutoGluonは中身がややブラックボックスで、過学習リスクもあります。そのため、1位の人はAutoGluonだけを信じるのではなく、LR with Logitsのような安定した方法ともブレンドしています。

---

### 18.10 最後はAutoGluonとLR with Logitsをブレンドした

1位の最終段階は、以下のような流れです。

```text
182個のL1 OOF
↓
一部L2 OOFを追加して186 OOFへ
↓
LR with Logitsでensemble
↓
AutoGluonでもensemble
↓
LR with Logits submission と AutoGluon submission を50-50でblend
↓
final submission
```

ここで重要なのは、最終提出が単体CatBoostや単体LightGBMではないことです。最終提出は、**大量のOOFを使ったメタアンサンブル同士のブレンド**でした。

LR with Logitsは安定した線形寄りの混ぜ方です。AutoGluonはより柔軟で、非線形な混ぜ方を試せます。両方を混ぜることで、安定性と柔軟性のバランスを取ったと考えられます。

---

### 18.11 1位から得た一番大事な気づき

#### Driverを落としたモデル

自分は `Driver` をtarget encodingやcategory encodingで使う方向でした。これは普通に強いです。Driverは戦略やチームの癖を含む可能性があるため、重要特徴量です。

ただし、1位の人は「Driverはoriginal dataとcompetition dataで分布差が大きい」と見て、あえてDriverを落としたモデルを作っています。

普通は、

```text
Driverは重要特徴量だから使う。
```

で終わります。

しかし上位者は、

```text
Driverは重要だが、分布差があるなら、一部モデルでは落とした方がアンサンブルに効く。
```

と考えます。

これはかなり大事な発想です。強い特徴量でも、分布差が大きいなら危険です。そのため、DriverありモデルとDriverなしモデルを両方作り、最後にメタモデルへ判断させるのが上位者の発想です。

#### Original dataのsample weight調整

自分はoriginal dataを「入れる/入れない」で考えていた面が強いです。1位の人は、それをさらに細かく、どれくらい信じるかまで調整していました。

```text
original data weight = 1.0
original data weight = 0.75
original data weight = 0.5
original data weight = 0.25
original dataなし
```

これは非常に実践的です。original dataは情報量がありますが、competition dataとズレているなら100%信じると悪さをする可能性があります。だから、使うけど信頼度を下げるモデルを複数作る。この発想は今後のPlayground系コンペでかなり重要です。

#### AutoGluonをメタアンサンブラーとして使う

自分は主に手動ブレンドでした。これは良い第一歩です。

```text
0.6 * LGBM + 0.4 * CatBoost
```

しかし1位は、AutoGluonを使って、多数のOOFから最終予測を作っています。AutoGluonは内部で複数モデルやstackingを試すので、人間が気づきにくい重みづけを拾う可能性があります。

今後は、手動ブレンドだけでなく、OOFを集めたうえでAutoGluonをメタモデルとして使う価値があります。

#### LR with Logitsを使う

自分は単純な予測値平均や重み付き平均が中心でした。1位は、各モデルの予測をlogit変換してLogistic Regressionに入れています。

これは分類コンペのstackingで非常に使える技です。特にAUCでは、確率の絶対値よりも順位が大事なので、logit変換を使った方がモデル間の強弱を扱いやすいことがあります。

#### 同じモデルのハイパーパラメータ違いを複数残す

自分はOptunaで良いハイパーパラメータを1つ探す方向でした。

一方で1位の人は、毎回ちゃんとOptunaを回すというより、過去モデルやpublic notebookのハイパーパラメータから始めて、重要なパラメータを少し触り、同じnotebookを複数HPで試すという方針でした。

つまり、

```text
best hyperparameterを1個探す
```

ではなく、

```text
少し違う性格のモデルを複数OOFとして残す
```

という考え方です。

これは上位者らしい発想です。大量OOFアンサンブルでは、1つのモデルを完璧にするより、強くて少し違うモデルを複数残す方が効くことがあります。

---

### 18.12 自分のやり方と1位のやり方の違い

| 観点 | 自分 | 1位 |
|---|---|---|
| 基本方針 | 強いLightGBM/CatBoostを作り、少数ブレンドする | 大量の多様なOOFを作り、メタモデルで混ぜる |
| 特徴量 | ABテストで良さそうな特徴量を残す | 複数の特徴量セットを作り、モデルごとに相性を見る |
| モデル種類 | LGBM, CatBoost, XGB中心 | XGB, LGBM, CatBoost, RealMLP, TabM, HGB, RF, YDF, FT-Transformer, MLP-PLRなど |
| original data | 入れる/入れないが中心 | sample weightを変え、original dataをどれくらい信じるか調整 |
| Driver | 重要特徴量として使う | 分布差を考え、Driverなしモデルも作る |
| OOF | 少数モデルの評価・ブレンドに使用 | 182個以上をメタデータ化 |
| アンサンブル | 固定重みの確率ブレンド | LR with Logits, AutoGluon, 最終blend |
| HPO | Optunaで良い設定を探す | 最適HPを1つ探すより、HP違いのOOFも残す |
| 最終思想 | 良いモデルを作る | 良い予測パターンを大量に作る |

---

### 18.13 今後の自分用アクション

次に似たTabular / AUCコンペに参加するときは、以下の流れを標準にしたいです。

```text
1. まず強い単体モデルを作る
2. OOFとtest predictionを必ず保存する
3. feature setを複数作る
4. LGBM / CatBoost / XGB / RealMLP / TabM を最低限試す
5. Driverや重要カテゴリのdrop版を作る
6. original dataがある場合、weight 1.0 / 0.75 / 0.5 / なしを試す
7. OOF poolを作る
8. OOF同士の相関を見る
9. LR with Logitsでメタモデルを作る
10. 余裕があればAutoGluonをメタモデルとして使う
11. 最後に複数の強いメタsubmissionをblendする
```

重要なのは、いきなり200 OOFを目指すことではありません。まずは以下のように段階を踏むべきです。

```text
Stage 1: 強い単体モデルを作る
Stage 2: 10〜20個の多様なOOFを作る
Stage 3: LR with Logitsでメタモデルを作る
Stage 4: 効果が出るなら50個以上に増やす
Stage 5: 上位狙いなら100〜200個のOOF poolを作る
```

今回の自分は、Stage 1と少数ブレンドまでは到達できました。次に伸ばすべきなのは、**強い単体モデルを大量のOOF資産に展開するフロー**です。

---

## 19. 2位writeupから見える補足：1位と同じ方向だが、よりL1 logit ensembleに振り切っている

2位の人も、1位と同じく「強い単体モデルを1個作る」よりも、**大量の多様なモデルを作り、OOFとtest predictionを保存し、最後にlogit変換 + Logistic Regressionで混ぜる**方向でした。

ただし、1位と2位には少し違いがあります。

### 19.1 2位の最終解法

2位の人は、最終解を **218 models のweighted average via logistic regression** と説明していました。各モデルの予測をまずlogit変換し、それをNVIDIA cuML Logistic Regressionに入れて、最終提出を作っています。

つまり、流れは以下です。

```text
218個のL1モデル
↓
各モデルのOOFとtest predictionを保存
↓
予測値をlogit変換
↓
cuML Logistic Regressionで重みを学習
↓
test側218列に対して最終予測
↓
submission.csv
```

ここでの218は、218種類のアルゴリズムという意味ではありません。`RealMLP`, `XGBoost`, `CatBoost`, `LightGBM` などのモデル種類に対して、特徴量セット、ハイパーパラメータ、データ設定を変えた218本の予測列という意味です。

### 19.2 2位のモデル数の内訳

2位の人は、以下のようなモデル群を使っていました。

| モデル/class | model count | 位置づけ |
|---|---:|---|
| RealMLP | 40 | 最強クラスの主力モデル |
| XGBoost | 36 | 主力モデル |
| CatBoost | 37 | 主力モデル |
| TabM | 11 | ニューラル系の主力補完 |
| LightGBM | 25 | 主力モデル |
| TabICL | 8 | 多様性と追加性能 |
| FFM | 3 | 多様性枠 |
| Custom NN | 9 | ニューラル系多様性枠 |
| RandomForest | 3 | 木系だがGBDTとは違う枠 |
| GNN | 3 | 多様性枠 |
| HistGB | 1 | 追加の木系枠 |
| FM | 3 | interaction系の多様性枠 |
| KNN | 1 | 距離ベースの多様性枠 |
| TabTransformer | 3 | Transformer系の多様性枠 |
| ExcelFormer | 1 | Transformer系/表形式DL枠 |
| Cox/survival | 1 | ピットタイミングをsurvival的に見る枠 |
| DAE | 1 | deep learning系の多様性枠 |
| AMFormer | 1 | Transformer系の多様性枠 |
| YDF GBDT | 1 | 別実装のGBDT枠 |
| MLP-PLR | 1 | ニューラル系の多様性枠 |
| Trompt | 1 | 表形式DL枠 |
| TabR | 1 | 表形式DL枠 |
| AutoInt | 1 | interaction系DL枠 |
| GrowNet | 1 | ニューラル系boosting枠 |
| SAINT | 1 | 表形式Transformer枠 |
| ModernNCA | 1 | 近傍系/表形式DL枠 |
| その他 | 複数 | 多様性枠 |

これを見ると、上位者は単にLightGBMとCatBoostを強くするだけではなく、RealMLP、TabM、TabICLなどの表形式ニューラルネット系をかなり使っています。

### 19.3 1位との違い

1位と2位は、大枠ではかなり似ています。

共通点は以下です。

```text
大量のモデルを作る
OOFとtest predictionを保存する
予測列を横に並べる
logit変換する
Logistic Regressionで混ぜる
```

一方で、違いもあります。

| 観点 | 1位 | 2位 |
|---|---|---|
| モデル数 | 186 OOF | 218 models |
| 中心手法 | OOF ensemble + LR with Logits + AutoGluon | 218 L1予測 + logit + cuML Logistic Regression |
| L2 OOF | 一部使った | 使っていないと明記 |
| AutoGluon | 最終Ensemblerとして重要 | 主役ではない |
| 最終提出 | LR with LogitsとAutoGluonのblend | cuML Logistic Regression一本寄り |
| 実験方法 | Claudeなどを活用しつつ多様な実験 | Codex Agentで大量実験を自動化 |

2位は、1位よりもさらに **L1モデルのlogit-weighted blend** に振り切っています。2位本人も、今回の最終解では「predictions of other modelsでさらにモデルを学習するstackingは使っていない」と説明しています。

ここは少し言葉がややこしいですが、理解としては以下です。

```text
2位は、L1モデルの予測をLogistic Regressionで重みづけしている。
ただし、L2モデルをさらに作って、それをまたOOF poolに入れるような多段stackingはしていない。
```

### 19.4 2位から特に学ぶべきこと

2位から学ぶべき一番大きいことは、**public notebookにtest predictionしかない場合でも、自分のfoldで再実行してOOFとtest predictionを保存する**ことです。

test predictionだけでは、Logistic Regression ensembleに入れるためのtrain側情報がありません。だから上位者は、public notebookをそのままブレンドするのではなく、以下のように再現します。

```text
public notebookを読む
↓
自分のlocal 5-foldに書き換える
↓
OOF predictionを保存する
↓
test predictionも保存する
↓
自分のOOF poolに追加する
```

これができると、公開ノートブックの良いモデルを、ただの提出ファイルとしてではなく、自分のメタデータの一部として使えます。

### 19.5 2位のAI Agent活用

2位のwriteupで特徴的だったのは、Codex Agentを使って大量実験を自動化していたことです。

具体的には、Codexに以下をやらせていました。

```text
public notebookを自分の5-foldに書き換える
過去コンペのコードを今回用に変換する
特徴量を作る
ハイパーパラメータを調整する
local_leaderboard.mdにCVを記録する
結果を見て次の実験を考える
GPUを使って並列で実験を回す
```

このフローは、自分が最初にAIに100個特徴量を作らせた実験とはかなり違います。

自分のAI活用は、

```text
特徴量をたくさん作らせる
↓
ABテストする
```

に近かったです。

2位のAI活用は、

```text
実験管理・再現・OOF保存・モデル多様性の生成まで自動化する
```

です。

つまり、AIを「特徴量アイデア出し」だけに使うのではなく、**実験を継続的に回す研究助手**として使っています。

### 19.6 1位・2位から一般化できる最重要学び

今回の上位者から見ると、Kaggle上位狙いの流れは以下です。

```text
強い単体モデルを作る
↓
複数の特徴量セットを作る
↓
モデル種類を増やす
↓
ハイパーパラメータ・seed・original data設定を変える
↓
OOFとtest predictionを必ず保存する
↓
OOF poolを作る
↓
logit変換 + Logistic Regressionで混ぜる
↓
余裕があればAutoGluonやL2 OOFも試す
↓
最終blend
```

一言でまとめると、**一個の強いモデルを作ることは土台であり、上位に行くには、その強いモデルを多様なOOF資産に展開する必要がある**ということです。

自分は今回、強いCatBoost/LightGBMブレンドまでは到達できました。次回はそこから、OOFを大量に保存し、メタデータ化し、LR with LogitsやAutoGluonで混ぜる段階まで進めたいです。
