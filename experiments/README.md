# experiments/

ARKframework 内の実験的・探索的コード置き場。

- 各種シミュレーションのプロトタイプ
- 改善版・厳密化版の実装
- 仮説検証用のスクリプト

このフォルダ内のコードは研究進行中の実験物です。

## 追加モジュール (2026-07)

### ノイズ解析・分類
- `noise_analysis_and_classification.py`
  - ARKv のノイズ解析・分類専用モジュール
  - AdvancedNoiseGenerator (White/Pink/Brown/Structured 対応)
  - NoiseAnalyzer + 多軸ルールベース分類器
  - Multiverse風忠実実験を実施

- `noise_classification_report.md`
  - ノイズの質を多軸分類した解釈重視レポート

- `noise_experiment_detailed_results.md`
  - 288回試走の詳細データテーブル（タイプ別・mod別サマリー、サンプル試走）

### 残渣（Residue）解析・分類
- `residue_analysis_and_classification.py`
  - ノイズ条件を変えながら残渣の挙動を徹底解析・分類
  - ResidueSolver 忠実ロジック使用
  - 多軸分類（蓄積傾向 × ノイズ結合強度 × 変動パターン × 影響度）

- `residue_analysis_report.md`
  - 残渣の分類結果と主な発見をまとめたレポート
  - STRUCTURED_RESIDUE の res_nv_corr が突出して高く、ノイズを積極的に「取り込み・代謝」する挙動を確認

これらの実験はすべて **忠実に sandbox で実際実行** した結果に基づいています。