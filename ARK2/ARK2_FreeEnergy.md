# ARK2 Free Energy Framework

## 概要

ARK2 は、MetaVoid（ARK）の「residue（残留物）」概念と、Research Emergence Engine の「Momentum + 多目标通過」ダイナミクスを、**熱力学的な散逸構造** として統合したフレームワークです。

特に、**自由エネルギー最小化原理** を明示的に導入することで、以下の要素を一貫した物理・情報理論的框組で記述します：

- residue → 内部エネルギー（U）
- entropy → 散逸・不確実性（S）
- velocity / momentum → 運動エネルギー
- gamma → 状態変数
- recovery → 相転移（熱 → 仕事 + エントロピー生成）
- Free Energy F = U − TS の最小化がシステムの駆動原理

---

## 基本概念の対応

| 物理量          | ARK2 での意味                     | 役割 |
|----------------|----------------------------------|------|
| **U (residue)**   | 摩擦熱の積積                     | 停滞の深刻度を表す内部エネルギー |
| **S (entropy)**   | 散逸・探索の不確実性             | 高くなると探索しやすくなるが精度が落ちる |
| **T**             | Exploration Temperature          | エントロピーの影響度を調整する温度パラメータ |
| **F = U − TS**    | 自由エネルギー                     | システムが最小化しようとする量 |
| **velocity**      | 運動エネルギー                     | 目标を「通過」するための推進力 |
| **recovery**      | 相転移（熱 → 仕事 + エントロピー） | 積積した熱を運動エネルギーに変換しつつ、エントロピーを生成 |

---

## 熱力学的サイクル

### 1. 通常時（滑らかな通過）
- residue が低い状態では脅性が低く、Momentum が効果的に動き、目标を綺麗に通過できる。
- entropy も低く保たれ、安定した適応が続く。

### 2. 停滞と熱の積積
- 何らかの理由で gamma が目标近傍に留まり続けると、**摩擦熱（residue）** が積積する。
- residue の増加に伴い、**動的脅性** （dynamic_beta）が上昇し、過去の運動量（velocity）が失われていく。
  ```python
  dynamic_beta = momentum_beta * exp(-residue * λ)
  ```
- これにより「がんばればがんばるほど動けなくなる」という自己強化的なトラップが形成される。

### 3. 相転移（Recovery）
- residue が閾値を超えると Recovery が発動。
- ここで重要なのは、熱の**一部だけが仕事（velocity）になり、残りはエントロピーとして散逸する**点。
  ```python
  released_work   = residue * η
  entropy_increase = residue * (1 - η)

  velocity += direction * (base_boost + released_work * conversion)
  entropy  += entropy_increase
  residue   = 0.0
  ```
- これにより、システムは**自ら「沼から脱出する」**と同時に、探索の不確実性（entropy）も獲得する。

### 4. 自由エネルギー駆動
- システムは常に **F = residue − T × entropy** を意識していると解釈できる。
- F が高い状態（高 residue、低 entropy）では回復・探索が強く促される。
- F が低い状態では安定した通過が優先される。
- この最小化趨勢が、探索と叮束のバランスを自師的に調整する。

---

## 利点

- **物理的解釈の明確さ** ：MetaVoid の residue が「ただのメーター」ではなく、熱力学的に意味のある量になる。
- **探索・叮束の自然なトレードオフ** ：entropy の増加が「広く探索しやすくなるが、精密な制御が効きにくくなる」という現実的な挙動をもたらす。
- **相転移の美しさ** ：停滞 → 熱積積 → 爆発的脱出 という非線形なダイナミクスが、熱力学的に自然に導かれる。
- **拡張性** ：将来的に Active Inference や変分自由エネルギー原理との接続も視野に入る。

---

## 実装上のポイント（ARK2 Thermodynamic v2）

- `residue` と `entropy` を独立した状態変数として保持
- Recovery 時に `released_work` と `entropy_increase` を分離して計算
- `dynamic_beta` で residue による脅性を表現
- 可能であれば `free_energy` を計算し、回復の閾値や強度を F に依存させる（オプション）

この框組みにより、ARK2 は単なるシミュレータではなく、**「熱力学的に駆動される誤知・適応モデル」**として位置づけられる。