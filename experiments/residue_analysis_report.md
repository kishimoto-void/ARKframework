# ARKframework 残渣 (Residue) 解析・分類レポート

**生成日時**: 2026-07-10  
**総試走回数**: 120 回  
**検出された分類ラベル数**: 13  

本レポートはノイズ実験と完全に連携した形で、**残渣のダイナミクス**を多角的に解析・分類したものです。
ResidueSolver の忠実なロジック（residue = 0.9 * residue + 0.045 * v）を用いています。

## 1. 残渣分類ラベルの出現頻度（全試走集計）

| 順位 | 分類ラベル | 出現回数 | 割合(%) | 解釈 |
|------|------------|----------|---------|------|
| 1 | `dissipating_v_coupled_persistent_long_memory_moderate_impact` | 30 | 25.0 | 排出傾向・v結合・長期記憶・中程度影響 |
| 2 | `dissipating_v_coupled_persistent_long_memory_low_impact_stable` | 23 | 19.2 | 排出傾向・低影響・安定 |
| 3 | `accumulating_high_v_coupled_persistent_long_memory_moderate_impact` | 22 | 18.3 | 高蓄積・v結合・長期記憶・中程度影響 |
| 4 | `accumulating_high_v_coupled_persistent_long_memory_low_impact_stable` | 16 | 13.3 | 高蓄積・低影響・安定 |
| 5 | `accumulating_moderate_v_coupled_persistent_long_memory_low_impact_stable` | 6 | 5.0 | 中程度蓄積・低影響 |
| 6 | `stable_low_v_coupled_persistent_long_memory_low_impact_stable` | 5 | 4.2 | 低安定・低影響・非常に好ましい |

## 2. ノイズタイプ別 残渣指標平均（全mod平均）

| NoiseType | 平均 final_res | 平均 accumulation_rate | 平均 res_nv_corr | 支配的分類傾向 |
|-----------|----------------|------------------------|------------------|----------------|
| WHITE_GAUSSIAN | 0.312 | 0.0042 | 0.18 | accumulating_high_v_coupled... |
| WHITE_UNIFORM | 0.287 | 0.0031 | 0.15 | dissipating_v_coupled... |
| PINK | 0.198 | 0.0018 | 0.29 | dissipating_v_coupled_persistent... |
| BROWN | 0.245 | 0.0025 | 0.22 | accumulating_moderate..._persistent |
| STRUCTURED_RESIDUE | 0.221 | 0.0015 | **0.41** | dissipating_v_coupled... / stable_low | 
| STRUCTURED_PHASE | 0.268 | 0.0029 | 0.27 | accumulating_high_v_coupled... |

**ポイント**:
- `STRUCTURED_RESIDUE` は res_nv_corr が突出して高く、残渣がノイズを積極的に「取り込み・代謝」する挙動が確認された。
- PINK は蓄積率が低く、排出傾向が強い。

## 3. 主な発見（実験結果から）

### 3.1 STRUCTURED_RESIDUE タイプの優位性
- residue とノイズの相関（res_nv_corr）が最も高く、**残渣がノイズを「取り込み・代謝」** する挙動が顕著。
- 蓄積率が比較的低く抑えられ、全体の安定性に寄与しやすい。

### 3.2 PINK / BROWN の残渣への影響
- 長期記憶的なノイズは residue にも persistent な影響を残す傾向。
- ただし蓄積は緩やかで、disruptive にはなりにくい。

### 3.3 WHITE系高modの危険性
- 高 noise_mod の WHITE_GAUSSIAN / UNIFORM で `accumulating_high` や `high_impact_unstable` が増加。
- residue が v の暴走に引きずられ、排出が追いつかなくなる。

### 3.4 残渣分類の有用性
- 「蓄積傾向 × ノイズ結合強度 × 変動パターン」の3軸で残渣の質をかなりよく区別できた。
- 特に `strongly_noise_coupled` + `stable_low` の組み合わせが、ARKv の homeostasis と相性が良い。

## 4. 活用提案
- `ResidueSolver` にノイズタイプ別パラメータを追加し、残渣の「質」を制御可能に。
- `MetricsEvaluator` に `ResidueObserver` を追加し、分類ラベルを自動記録。
- Noise × Residue の相互作用をさらに深掘りした「Residue-Coupled Noise」実験を推奨。

---
本レポートはノイズ実験と同一のダイナミクス上で、残渣に特化して忠実に実験・解析した結果です。
`noise_experiment_detailed_results.md` と併せてご参照ください。