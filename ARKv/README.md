# ARKv - Observation, Verification & Statistics

ARKv は観測・検証・統計解析に特化したライブラリです。

## ARKv.py（コア実装）

**モジュラーパイプラインによる多宇宙観測フレームワーク**（2026-07版）

### 主な改善点
- 【改善①】 HomeostasisForce の純粋具象化
- 【改善②】 ModularDynamics によるソルバーの完全パイプライン化
- 【改善③】 ForceVector の明確な型定義
- 【改善④】 MetricResult の動的伸縮対応（Hexa → Octa/Dodeca など柔軟に拡張可能）
- 【改善⑤】 BaseSolver インターフェースによる疎結合な context リレー

### 主要コンポーネント
- `GeometryNetwork` + `Node` / `NodeState` / `NodeDefinition`
- **力場ソルバー群**
  - `TopologyEvolutionSolver`（斥力・引力・恒常性）
  - `NetworkInteractionSolver`（artery/vein/pulse の集計）
  - `PotentialPulseSolver`（脈動・ノイズ・弾性積分・閾値検出）
  - `ResidueSolver`（残渣代謝）
- `MetricsEvaluator` + プラグイン `BaseObserver`
  - GeometryObserver, StatisticsObserver, TemporalObserver, InformationObserver, DynamicalObserver, StabilityObserver
- `HistoryBuffer`（時系列スナップショットの非破壊記録）
- `FastGridAttractorTracker`（アトラクタ容積追跡）
- `MultiverseExperiment`（Normal / Noise(1.15x) / Elastic(1.3x) の3並行世界比較）
  - KL divergence（経験分布5ビン）
  - 局所Lyapunov指数（L2軌道距離の対数成長率）
  - アトラクタ占有容積
- `ConsoleReporter`（結果の整形出力）

### 使い方例

```python
from ARKv.ARKv import MultiverseExperiment, ConsoleReporter

experiment = MultiverseExperiment()
raw_result = experiment.run(steps=30)

reporter = ConsoleReporter()
reporter.report(raw_result)
```

Closed Loop: RFO → FEX → CDE → CME → HGE → HCE → RAE → EVE → RFO

詳細は元リポジトリ https://github.com/kishimoto-void/ARKv を参照（移行中）。