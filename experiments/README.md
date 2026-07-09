# experiments/

ARKframework 内の実験的・探索的コード置き場。

- 各種シミュレーションのプロトタイプ
- 改善版・厳密化版の実装
- 仮説検証用のスクリプト

このフォルダ内のコードは研究進行中の実験物です。

## 追加モジュール (2026-07)

- `noise_analysis_and_classification.py`
  - ARKv のノイズ解析・分類専用モジュール
  - AdvancedNoiseGenerator (White/Pink/Brown/Structured 対応)
  - NoiseAnalyzer + 多軸ルールベース分類器
  - Multiverse風忠実実験 (55 steps × 5 trials × 6 configs) を実際に実行済み
  - 結果: Pinkノイズは低振幅ながら高い安定性とresidue couplingを示すなど、定量的な知見を得ました。
  - プロットとJSON結果も artifacts/ に生成されます。