# HCE - Hypothesis Consistency Engine

Hypothesis Consistency Engine (HCE) は、観測データに対して複数の仮説を多軸（Geometry, Statistics, Temporal, Information, Dynamical, Robustness, Falsification）で厳密に評価し、整合性を検証するエンジンです。

## 設計原則
- Observation First
- Statistics Before Interpretation
- Pareto Front による多様な有力仮説の保持
- Unknown を正当な結論として認める（原則5）

## 主なファイル
- `hce_evolved.py` : 実計算版 HCE（numpy + scipy 使用、忠実な実験向け）
- `hce_divergent_v4.py` : 制約抽出 + ハルシネーションによる仮説自己進化ループ
- `hce_hexa_from_aether.py` : Aether シミュレーション軌跡 → HexaMetrics 評価
- `hexa_metrics_improved.py` : 改善版 HexaMetrics（飽和しにくい観測器）
- `somatic_void_platform.py` : Somatic Void プラットフォーム（HCE/ARK/ARKv 連携）

## 使い方
```bash
python hce_evolved.py
```

詳細は各ファイルの docstring を参照してください。