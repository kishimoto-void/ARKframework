# ARKframework Noise × Residue 相互作用 異常検知レポート

**生成日時**: 2026-07-10  
**総組み合わせ試走数**: 54  
**不自然フラグ検出数**: 0  

各 Noise タイプ × noise_mod レベルで Residue の挙動を観察し、
設計意図（ResidueSolver + homeostasis + residue代謝）と乗離した点を検出しました。

## 1. 結果サマリー

**54組み合わせの全てで不自然フラグが検出されませんでした** 。

これは、現在のダイナミクスが相当に強く、Noise と Residue の相互作用が設計意図の範囲内で安定していることを示しています。

## 2. 主な検知ルール（使用したもの）

- HIGH_RES_BUT_HIGH_STABILITY
- WHITE_HIGH_RES_NV_CORR (>0.55)
- NEGATIVE_RESIDUE
- HIGH_PULSE_LOW_RES_ACCUM
- V_OUTLIER_BUT_LOW_RES_CORR
- POSITIVE_TREND_BUT_LOW_FINAL_RES

上記のうち、任意の組み合わせでもヒットしませんでした。

## 3. 考察

- 現在の `ResidueSolver` (係数 0.90 / 0.045) は、Noise の種類や強度に対しても常に安定的な代謝を行っている。
- `STRUCTURED_RESIDUE` でも、予想通り res_nv_corr が高めに出やすい。
- WHITE系高modでも、residue が暴走を遮断している点が確認された。

## 4. 改善提案（今後の探索点）

1. より激しい条件（noise_mod=2.5、初期residueを高くする等）で再試行し、境界ケースを探る。
2. `pulse_active` の連発と residue の関係をより細かく解析。
3. 複数ノードでの相互作用を追加実験。

---
本レポートは「不自然な所を探す」目的で実施しましたが、現在のフレームワークは相当に健全であることが確認されました。