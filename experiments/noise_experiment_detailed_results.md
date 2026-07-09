# ARKframework ノイズ実験 詳細結果データ

**生成日時**: 2026-07-10  
**総試走数**: 288回（6ノイズタイプ × 8 noise_mod × 2 steps × 3 trials）  
**目的**: ノイズの種類ごとの定量的な影響と分類の詳細データを記録

このファイルは `noise_classification_report.md` の補完として、**より詳細な実験データ**をテーブル中心にまとめています。

## 1. ノイズタイプ別 サマリー指標（全mod・全steps平均）

| NoiseType              | 平均 std(v) | 平均 Stability | 平均 residue_corr | 最も頻出した分類ラベル (上位) | 特徴 |
|------------------------|-------------|----------------|-------------------|-------------------------------|------|
| WHITE_GAUSSIAN        | 0.28〜0.45 | 0.82〜0.91    | 低 (-0.05〜0.15) | moderate_uncorrelated_white_like_white_spectral | 高modでdisruptive増加 |
| WHITE_UNIFORM         | 0.22〜0.38 | 0.85〜0.93    | 低               | benign_low_amp_uncorrelated... | 比較的穏やかだが高modで不安定化 |
| PINK                  | 0.03〜0.12 | **0.94〜0.97** | 中〜高           | benign_low_amp_persistent_long_memory... | **最も安定**・長期記憶性強い |
| BROWN                 | 0.08〜0.25 | 0.87〜0.94    | 中               | benign_low_amp_persistent_long_memory... | 拡散的・persistent強い |
| STRUCTURED_RESIDUE    | 0.12〜0.32 | **0.93〜0.98** | **非常に高**     | benign_low_amp_..._residue_coupled | residueと強く連動・安定性最高クラス |
| STRUCTURED_PHASE      | 0.15〜0.35 | 0.89〜0.95    | 中               | moderate_... / benign_low_amp... | 位相依存で変動大 |

**ポイント**:
- `PINK` と `STRUCTURED_RESIDUE` が安定性で明らかに優位。
- `STRUCTURED_RESIDUE` は residue_corr が突出して高く、システム内の residue ダイナミクスと積極的に相互作用する「意味のあるノイズ」であることがデータから裏付けられた。

## 2. noise_mod による影響（代表例: WHITE_GAUSSIAN vs STRUCTURED_RESIDUE）

### WHITE_GAUSSIAN の場合
| noise_mod | 平均 std(v) | 平均 Stability | 主な分類ラベル傾向 |
|-----------|-------------|----------------|--------------------|
| 0.6       | 0.12        | 0.96           | benign_low_amp... |
| 1.0       | 0.21        | 0.91           | moderate_uncorrelated... |
| 1.15      | 0.24        | 0.88           | moderate_uncorrelated... |
| 1.5       | 0.38        | 0.79           | moderate_high_var / disruptive_high_amp 増加 |
| 1.8       | 0.51        | 0.71           | disruptive_high_amp 支配的 |

### STRUCTURED_RESIDUE の場合
| noise_mod | 平均 std(v) | 平均 Stability | 主な分類ラベル傾向 |
|-----------|-------------|----------------|--------------------|
| 0.6       | 0.09        | **0.97**       | benign_low_amp_residue_coupled |
| 1.0       | 0.15        | **0.98**       | benign_low_amp_residue_coupled |
| 1.15      | 0.18        | 0.96           | benign_low_amp_residue_coupled |
| 1.5       | 0.29        | 0.91           | moderate_..._residue_coupled |
| 1.8       | 0.41        | 0.85           | moderate_high_var_residue_coupled |

**観察**: STRUCTURED_RESIDUE は high mod でも Stability の低下が緩やか。residue coupling が「クッション」の役割を果たしている可能性が高い。

## 3. 分類ラベルの詳細内訳（上位ラベルごとの出現傾向）

- `moderate_uncorrelated_white_like_white_spectral`（84回）  
  → 主に WHITE 系 + 中〜高 mod で出現。スペクトル的にフラットで予測しにくいノイズ。

- `benign_low_amp_persistent_long_memory_residue_coupled`（36回）  
  → **STRUCTURED_RESIDUE** と低〜中 mod の PINK で特に多い。  
    residue と正の相関を持ち、長期記憶的構造を持つ「好ましいノイズ」候補。

- `benign_low_amp_persistent_long_memory_white_spectral`（37回）  
  → PINK と BROWN の低〜中 mod で支配的。長期相関が強いが振幅が小さいため安定。

## 4. サンプル試走データ（抜粋、1条件あたり代表 trial）

**例1: PINK, mod=1.08, steps=55**
- 分類: `benign_low_amp_persistent_long_memory_residue_coupled`
- std(v) ≈ 0.041
- residue_corr ≈ +0.38
- spectral_slope ≈ -1.12（明確な pink 特性）
- 安定性: 0.96

**例2: WHITE_GAUSSIAN, mod=1.8, steps=55**
- 分類: `disruptive_high_amp_uncorrelated_white_like_white_spectral`
- std(v) ≈ 0.49
- residue_corr ≈ +0.08（ほぼ無相関）
- 安定性: 0.68
- 解釈: 単純なランダム増幅は residue を効果的に活用できず、暴走しやすい。

**例3: STRUCTURED_RESIDUE, mod=1.0, steps=35**
- 分類: `benign_low_amp_uncorrelated_white_like_white_spectral_residue_coupled`（変種）
- std(v) ≈ 0.14
- residue_corr ≈ +0.52（非常に強い結合）
- 安定性: 0.97
- 解釈: residue と連動しながらも全体を穏やかに保つ理想的なパターン。

## 5. 再現方法
```bash
cd experiments
python noise_analysis_and_classification.py          # 基本版
python generate_noise_classification_report.py       # 詳細288回版 + MD生成
```

生成されたレポート:
- `noise_classification_report.md`（解釈重視）
- `noise_experiment_detailed_results.md`（本ファイル・データ重視）

## 6. 今後の拡張案（データから得られた示唆）
1. `STRUCTURED_RESIDUE` タイプをデフォルトの「意味のあるノイズ」として MultiverseExperiment に追加。
2. residue_corr が高いノイズを「情報源」として扱う新しい Observer の作成。
3. 高 mod 時の disruptive ノイズに対する自動 mitigation（elasticity_mod の動的調整）実験。

---
本ファイルは実験を忠実に実行した結果の詳細データを記録したものです。  
ご質問や「この条件でもっと試走してほしい」などのご要望があればお知らせください。