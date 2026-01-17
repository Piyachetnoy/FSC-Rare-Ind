# HITL Workflow Comparison Experiment (Section 4.5)

このスクリプトは、preprint論文のSection 4.5「Human-in-the-Loop Workflow Comparison」実験を実行します。

## 概要

4つの異なるワークフローを評価します:

1. **Fully Automated** - 全ての予測を人間の検証なしで受け入れる
2. **Full Manual Verification** - 全ての予測を人間がレビュー
3. **Random Sampling** (25%, 50%, 75%) - ランダムに選択された画像を人間が検証
4. **Adaptive HITL (提案手法)** - 信頼度スコアに基づく選択的検証 (θ=0.3, 0.5, 0.7)

## 使い方

```bash
# 基本的な実行
python scripts/hitl_experiment.py

# カスタムパラメータでの実行
python scripts/hitl_experiment.py \
    -dp ./data-final/ \
    -a ./annotation_json/annotations409.json \
    -m ./logsSave/INDT-409trained.pth \
    -o ./hitl_results \
    -g 0 \
    --dataset_name INDT-409

# CPUで実行する場合
python scripts/hitl_experiment.py -g -1
```

## コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `-dp, --data_path` | `./data-final/` | データセットのパス |
| `-a, --anno_file` | `./annotation_json/annotations409.json` | アノテーションJSONファイル |
| `-m, --model_path` | `./logsSave/INDT-409trained.pth` | 学習済みモデルのパス |
| `-o, --output_dir` | `./hitl_results` | 結果の出力ディレクトリ |
| `-g, --gpu-id` | `0` | GPU ID (-1でCPU使用) |
| `--dataset_name` | `INDT-409` | レポート用のデータセット名 |
| `--seed` | `42` | 再現性のための乱数シード |
| `--alpha` | `0.5` | 信頼度スコアの分散成分の重み |
| `--beta` | `0.5` | 信頼度スコアの相関成分の重み |
| `--human_time` | `30.0` | 画像1枚あたりの人間の検証時間(秒) |

## 出力ファイル

スクリプトは以下のファイルを生成します:

### 1. `image_results_YYYYMMDD_HHMMSS.csv`

各画像の詳細な予測結果:

| カラム | 説明 |
|--------|------|
| `image_id` | 画像ファイル名 |
| `ground_truth` | 実際のオブジェクト数 |
| `prediction` | モデルの予測数 |
| `density_sum` | 密度マップの合計値 |
| `confidence_score` | 総合信頼度スコア (0-1) |
| `local_variance` | ローカル分散 (低いほど高信頼) |
| `correlation_score` | 相関スコア |
| `high_density_peaks` | 高密度ピーク数 (閾値 0.7) |
| `medium_density_peaks` | 中密度ピーク数 (閾値 0.65) |
| `low_density_peaks` | 低密度ピーク数 (閾値 0.3) |
| `total_unique_peaks` | ユニークなピークの総数 |
| `error` | 絶対誤差 |gt - pred| |
| `squared_error` | 二乗誤差 |

### 2. `workflow_comparison_YYYYMMDD_HHMMSS.csv`

ワークフロー比較結果（論文Table 5に対応）:

| カラム | 説明 |
|--------|------|
| `workflow_name` | ワークフロー名 |
| `mae` | 平均絶対誤差 |
| `rmse` | 二乗平均平方根誤差 |
| `human_intervention_rate` | 人間介入率 (%) |
| `total_images` | テスト画像総数 |
| `images_verified` | 検証された画像数 |
| `estimated_processing_time_min` | 推定処理時間 (分) |
| `threshold` | Adaptive HITLの閾値 (該当する場合) |

### 3. `experiment_report_YYYYMMDD_HHMMSS.json`

完全な実験レポート（JSON形式）:
- 実験設定
- サマリー統計
- 全ワークフロー結果
- 全画像結果

### 4. `summary_YYYYMMDD_HHMMSS.txt`

人間が読みやすい形式のサマリーテーブル。

## 信頼度スコアの計算

信頼度スコア $C$ は以下の式で計算されます:

$$C = \alpha \cdot C_{\text{variance}} + \beta \cdot C_{\text{correlation}}$$

### $C_{\text{variance}}$ (正規化逆分散)

- 検出されたピーク周辺のローカル分散を計算
- **低分散** = 明確で孤立したピーク = **高信頼度**
- **高分散** = 曖昧またはノイズの多い予測 = **低信頼度**

### $C_{\text{correlation}}$ (相関スコア)

- 異なる閾値でのピーク検出の一貫性を測定
- 高閾値と低閾値での検出数の比率
- **高い一貫性** = **高信頼度**

## 例: 結果の解釈

```
================================================================================
HITL Workflow Comparison Experiment Results
Timestamp: 20250117_143052
Dataset: INDT-409
Total Test Images: 82
================================================================================

Workflow Comparison Results:
--------------------------------------------------------------------------------
Workflow                               MAE     RMSE    HIR (%)    Time (min)
--------------------------------------------------------------------------------
Fully Automated                       9.364   16.990       0.0          0.3
Random Sampling (25%)                 7.102   14.210      25.0          4.4
Random Sampling (50%)                 5.418   11.850      50.0          8.5
Random Sampling (75%)                 3.891    9.627      75.0         12.6
Adaptive HITL (θ=0.3)                 8.125   15.420      15.0          3.1
Adaptive HITL (θ=0.5)                 6.234   12.930      32.0          5.6
Adaptive HITL (θ=0.7)                 4.157   10.080      58.0          9.8
Full Manual Verification              2.000    2.000     100.0         16.7
--------------------------------------------------------------------------------
```

この結果から:
- **Adaptive HITL (θ=0.5)** は Random Sampling (50%) と同等の精度を、より低い介入率 (32% vs 50%) で達成
- 信頼度ベースの選択により、人間の労力を効率的に配分可能

## 依存パッケージ

```
torch
torchvision
numpy
pandas
scikit-image
tqdm
Pillow
```

## 注意事項

1. モデルパスとデータパスが正しく設定されていることを確認してください
2. GPU使用時はCUDAが正しくインストールされている必要があります
3. 結果は `--seed` で指定した乱数シードに依存します（再現性のため）
