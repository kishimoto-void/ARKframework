#!/usr/bin/env python3
"""
ARK2 Theoretical Refactor (v2026-07-10)

【理論的再設計の指針（ユーザー提案を忠実に反映）】
実験は理論を「検証」するものであり、理論を「決める」ものではない。
長期的に発展しやすいよう、以下のように中心概念を整理：

Potential → Fluctuation（ゆらぎ） → Interaction → ResidualState → Free Energy → Recovery（自己組織化） → Emergence → Potential

これにより「ノイズ」「残溜」「構造」「創発」が一つの循環として統一的に表現できる。
「ノイズと残溜は観測階段によるラベルの違いではないか」という発想を採用し、
Fluctuation を基本概念に揚える。

具体的な改善（①～⑦）:
① FluctuationType を White/Pink/Brown/ResidueCoupled/PhaseCoupled として定義。
   デフォルトは固定せず、状態に応じて切り替わる設計（環境適応型）。
② Noise / Residue を Fluctuation → Interaction → ResidualState に統一。
   「ノイズ」は観測者ラベル → Fluctuation（ゆらぎ）がARKの基本。
③ NoiseAnalyzer → FieldAnalyzer に拡張（Potential / Fluctuation / ResidualState / Entropy / Phase を一括解析可能に）。
④ Free Energy を F = Potential + ResidualState - T×Entropy に拡張。
   Potential により「まだ実現していない可能性」をエネルギーとして扱える。
⑤ Recovery を「best_score による最適候補選択」から「複数候補の混合による自己組織化」に変更。
   少しずつ良い方向が残る創発的な回復へ。
⑥ Phase を内部で連続量 (0.0～1.0) として保持。表示ラベルはマッピングのみ。
⑦ ResidualState を thermal / informational / structural / temporal の複数種類に一般化。
   将来的な拡張性を確保（現在は thermal を主軸に）。

実験（sandbox multi-seed 忠実実行）で動作検証済み。
理論が先にあり、実験がそれを裏付ける形を目指す。
"""

from dataclasses import dataclass, field
from collections import deque
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto
from scipy import stats as scipy_stats


# ============================================================
# ① FluctuationModel（環境・状態に応じて切り替わる）
#    実験結果に引っ掛られ過ぎない理論的中立名前
# ============================================================
class FluctuationType(Enum):
    WHITE = auto()
    PINK = auto()
    BROWN = auto()
    RESIDUE_COUPLED = auto()   # 旧 STRUCTURED_RESIDUE（residue と結合しやすい）
    PHASE_COUPLED = auto()


class FluctuationGenerator:
    """
    Fluctuation（ゆらぎ）生成器。
    旧 AdvancedNoiseGenerator を理論名に改名。
    各タイプは「観測ラベル」ではなく、系の相互作用様式として位置付ける。
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed if seed is not None else random.randint(0, 2**32-1))
        self.pink_state = 0.0
        self.brown_v = 0.0

    def generate(self, mod: float = 1.0,
                 ftype: FluctuationType = FluctuationType.WHITE,
                 residual: float = 0.0, phase: float = 0.5) -> Tuple[float, float]:
        """fv: fluctuation velocity, fa: fluctuation acceleration"""
        if ftype == FluctuationType.WHITE:
            fv = self.rng.gauss(0.0, 0.20) * mod
            fa = self.rng.gauss(0.0, 0.10) * mod
        elif ftype == FluctuationType.PINK:
            white = self.rng.gauss(0.0, 0.20)
            self.pink_state = 0.85 * self.pink_state + white * 0.15
            fv = self.pink_state * mod * 0.6
            fa = self.rng.gauss(0.0, 0.06) * mod
        elif ftype == FluctuationType.BROWN:
            dv = self.rng.gauss(0.0, 0.04)
            self.brown_v = self.brown_v * 0.92 + dv
            fv = self.brown_v * mod * 1.2
            fa = self.rng.gauss(0.0, 0.03) * mod
        elif ftype == FluctuationType.RESIDUE_COUPLED:
            base = self.rng.gauss(0.0, 0.15)
            fv = (base + 0.08 * math.tanh(residual * 0.8)) * mod
            fa = (self.rng.gauss(0.0, 0.07) + 0.04 * residual) * mod
        else:  # PHASE_COUPLED
            mod2 = 1.0 + 0.4 * math.sin(phase * 1.3 * 2 * math.pi)
            fv = self.rng.gauss(0.0, 0.18) * mod * mod2
            fa = self.rng.gauss(0.0, 0.09) * mod * (1.0 + 0.2 * math.cos(phase * 2 * math.pi))
        return fv, fa


# ============================================================
# ③ FieldAnalyzer（Field 全体を解析）
#    Potential / Fluctuation / ResidualState / Entropy / Phase を統合
# ============================================================
class FieldAnalyzer:
    """
    Field 全体の状態を解析するアナライザ。
    将来的に Potential や Entropy の時系列も扱えるよう拡張余地を残す。
    """
    def __init__(self):
        self.fluct_v: List[float] = []
        self.fluct_a: List[float] = []
        self.residuals: List[float] = []      # thermal proxy
        self.potentials: List[float] = []
        self.entropies: List[float] = []
        self.phases: List[float] = []         # 連続 0-1
        self.steps: List[int] = []

    def collect(self, fv: float, fa: float,
                residual: float = 0.0,
                potential: float = 0.0,
                entropy: float = 0.0,
                phase: float = 0.5,
                step: int = 0):
        self.fluct_v.append(float(fv))
        self.fluct_a.append(float(fa))
        self.residuals.append(float(residual))
        self.potentials.append(float(potential))
        self.entropies.append(float(entropy))
        self.phases.append(float(phase))
        self.steps.append(step)

    def analyze(self) -> Dict[str, float]:
        if len(self.fluct_v) < 8:
            return {"status": "insufficient_data", "count": len(self.fluct_v)}

        v = np.asarray(self.fluct_v, dtype=float)
        res = np.asarray(self.residuals, dtype=float)
        pot = np.asarray(self.potentials, dtype=float)
        ent = np.asarray(self.entropies, dtype=float)
        ph = np.asarray(self.phases, dtype=float)

        m: Dict[str, float] = {
            "count": len(v),
            "mean_fluct_v": float(np.mean(v)),
            "std_fluct_v": float(np.std(v)),
            "var_fluct_v": float(np.var(v)),
            "residue_corr": self._safe_corr(v, res),
            "potential_corr": self._safe_corr(v, pot),
            "entropy_corr": self._safe_corr(v, ent),
            "phase_corr": self._safe_corr(v, ph),
            "mean_residual": float(np.mean(res)),
            "mean_potential": float(np.mean(pot)),
            "mean_entropy": float(np.mean(ent)),
            "mean_phase": float(np.mean(ph)),
        }
        try:
            f, Pxx = scipy_stats.periodogram(v, fs=1.0, scaling='spectrum')
            if len(f) > 12:
                idx = slice(1, len(f)//3)
                logf = np.log(f[idx] + 1e-12)
                logP = np.log(Pxx[idx] + 1e-12)
                slope, _ = np.polyfit(logf, logP, 1)
                m["spectral_slope"] = float(slope)
            else:
                m["spectral_slope"] = 0.0
        except Exception:
            m["spectral_slope"] = 0.0

        m["fluct_energy"] = float(np.sum(v**2) / len(v))
        return m

    def _safe_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 4 or np.std(y) < 1e-8 or np.std(x) < 1e-8:
            return 0.0
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except Exception:
            return 0.0

    def get_phase_label(self, phase: float) -> str:
        """連続 phase (0.0～1.0) をラベルにマッピング（表示用）"""
        if phase < 0.2:
            return "Trapped"
        elif phase < 0.5:
            return "Prepared"
        elif phase < 0.8:
            return "Emergence"
        else:
            return "Dissipative"


# ============================================================
# ⑦ ResidualState（複数種類の残溜を一般化）
# ============================================================
@dataclass
class ResidualState:
    thermal: float = 0.0          # 旧 residue（摩擦熱）
    informational: float = 0.0    # 将来拡張用
    structural: float = 0.0
    temporal: float = 0.0

    def effective(self) -> float:
        """ 有効残溜量（重み付け和）。現在は thermal を主に使用 """
        return (self.thermal +
                0.25 * self.informational +
                0.15 * self.structural +
                0.10 * self.temporal)

    def metabolize(self, amount: float, efficiency: float = 0.018):
        """ 代謝（ゆらぎとの相互作用で残溜を処理） """
        self.thermal = max(0.0, self.thermal * (1.0 - 0.018) + amount * efficiency)


# ============================================================
# Adaptation / Recovery のデータ構造（phase を連続量対応に）
# ============================================================
@dataclass(frozen=True)
class AdaptationState:
    gamma: float
    velocity: float
    I_target: float
    diversity: float
    entropy: float
    phase_cont: float          # 連続 0.0～1.0
    phase_label: str


@dataclass
class RecoveryEvent:
    event_id: str
    step: int
    residual_before: float
    entropy_before: float
    released_work: float
    entropy_increase: float
    velocity_after: float
    free_energy_before: float
    score: float
    reason: str
    chosen_action: str
    mix_ratio: float = 0.0     # 自己組織化混合の度合い


# ============================================================
# ARK2 本体（理論循環 Potential → Fluctuation → ... → Emergence を体現）
# ============================================================
class ARK2FreeEnergy:
    """
    ARK2 - Free Energy + Thermodynamic + Self-Organizing Emergence

    中心循環:
    Potential → Fluctuation → Interaction → ResidualState
             → Free Energy (Potential + Residual - T*Entropy)
             → Recovery (自己組織化混合) → Emergence → Potential

    実験結果に縛られず、理論的一貫性を優先した設計。
    """

    def __init__(
        self,
        steps: int = 220,
        initial_gamma: float = 0.10,
        momentum_beta: float = 0.78,
        residual_threshold: float = 17.5,
        friction_coefficient: float = 0.023,
        energy_conversion_efficiency: float = 0.42,
        exploration_temperature: float = 1.8,
        fluctuation_type: FluctuationType = FluctuationType.RESIDUE_COUPLED,  # 中立デフォルト
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = steps
        self.gamma = initial_gamma
        self.momentum_beta = momentum_beta
        self.velocity = 0.0
        self.residual = ResidualState(thermal=0.0)   # ⑦ 一般化
        self.entropy = 0.0
        self.potential = 0.3                         # ④ 初期 Potential（未実現可能性の代理）
        self.residual_threshold = residual_threshold
        self.friction_coefficient = friction_coefficient
        self.energy_conversion_efficiency = energy_conversion_efficiency
        self.exploration_temperature = exploration_temperature
        self.fluctuation_type = fluctuation_type     # ① 状態に応じて切り替わる
        self.epsilon = 1e-8

        self.fluct_gen = FluctuationGenerator(seed)
        self.field_analyzer = FieldAnalyzer()        # ③
        self.phase_cont = 0.45                       # ⑥ 連続 phase (0.0～1.0)

        self.checkpoints: List[AdaptationState] = []
        self.structural_reserve: deque[AdaptationState] = deque(maxlen=16)
        self.recovery_events: List[RecoveryEvent] = []
        self.history_metrics: List[Dict[str, Any]] = []
        self.approach_buffer: deque[float] = deque(maxlen=20)

        self.event_counter = 0
        self._stuck_counter = 0
        self.recovery_count = 0

    # 動的 FluctuationType 切り替え（環境適応の簡易版）
    def _adapt_fluctuation_type(self):
        eff = self.residual.effective()
        if eff > 12.0:
            self.fluctuation_type = FluctuationType.RESIDUE_COUPLED
        elif self.phase_cont < 0.25:
            self.fluctuation_type = FluctuationType.BROWN
        elif self.phase_cont > 0.75:
            self.fluctuation_type = FluctuationType.PHASE_COUPLED
        else:
            self.fluctuation_type = FluctuationType.WHITE

    def step(self) -> Dict[str, Any]:
        prev_gamma = self.gamma
        prev_velocity = self.velocity

        # ① 状態に応じた FluctuationType 適応
        self._adapt_fluctuation_type()

        # Phase を少し進める（連続）
        self.phase_cont = float(np.clip(self.phase_cont + 0.008 + 0.003 * np.sin(self.phase_cont * 6.28), 0.0, 1.0))

        # Fluctuation 生成（residue/phase と結合）
        fv, fa = self.fluct_gen.generate(
            mod=max(0.55, 1.12 - self.residual.effective() * 0.01),
            ftype=self.fluctuation_type,
            residual=self.residual.effective(),
            phase=self.phase_cont
        )

        # Fluctuation を dynamics に反映（Interaction）
        influence = 0.08 if self.fluctuation_type == FluctuationType.RESIDUE_COUPLED else 0.055
        self.velocity += fv * influence

        # I_target / critical / variance 計算（内部ゆらぎ + 外部 Fluctuation）
        R = max(0.12, 0.66 - abs(self.gamma - 0.16) * 1.85)
        I_target = 0.87 + 0.33 * R
        current_step = len(self.history_metrics)
        modulation = 0.16 * np.sin(2 * np.pi * current_step / 118)
        I_target = max(0.13, I_target + modulation)

        critical = max(0.0005, 0.72 * np.exp(-abs(self.gamma - 0.13) * 2.1) + abs(fa) * 0.07)
        variance = max(0.0012, 0.72 * (0.125 - abs(self.gamma - 0.125)) + abs(fv) * 0.025)

        tol_I = 0.05 * I_target
        e_I = critical - I_target + fa * 0.4
        near_target = abs(e_I) < tol_I

        if near_target:
            contrib_I = 0.0
            self._stuck_counter = 0
        else:
            contrib_I = e_I / (I_target + self.epsilon)
            self._stuck_counter += 1

        dynamic_beta = self.momentum_beta * np.exp(-self.residual.effective() * self.friction_coefficient)
        d_gamma = 0.0046 * (contrib_I + 0.52 * (variance - 0.05))

        self.velocity = dynamic_beta * self.velocity + (1.0 - dynamic_beta) * d_gamma
        self.gamma = float(np.clip(self.gamma + self.velocity, 0.02, 0.46))
        self.velocity *= 0.935

        delta_gamma = self.gamma - prev_gamma

        if near_target:
            self.approach_buffer.append(delta_gamma)

        approach_diversity = 0.0
        if len(self.approach_buffer) >= 10:
            approach_diversity = float(np.std(self.approach_buffer))

        # ResidualState 更新（thermal を主に）
        if near_target and approach_diversity < 0.0018:
            self.residual.thermal += 1.55 + (self._stuck_counter * 0.07)
        else:
            self.residual.thermal = max(0.0, self.residual.thermal * 0.96)

        # ⑦ ResidualState の代謝（Fluctuation との Interaction）
        if self.fluctuation_type == FluctuationType.RESIDUE_COUPLED:
            self.residual.metabolize(abs(fa), efficiency=0.022)

        # ④ Free Energy 拡張（Potential + Residual - T*Entropy）
        free_energy = (self.potential +
                       self.residual.effective() -
                       self.exploration_temperature * self.entropy)

        # Potential の微小ダイナミクス（未実現可能性のゆらぎ）
        self.potential = max(0.05, self.potential * 0.995 + 0.01 * (0.5 - abs(self.gamma - 0.25)))

        triggered_recovery = False
        released_work = 0.0
        entropy_increase = 0.0
        mix_ratio = 0.0

        if self.residual.effective() >= self.residual_threshold:
            triggered_recovery, released_work, entropy_increase, mix_ratio = self._self_organizing_recovery(
                I_target, approach_diversity, free_energy
            )
            if triggered_recovery:
                self.recovery_count += 1

        phase_label = self.field_analyzer.get_phase_label(self.phase_cont)

        current_state = AdaptationState(
            gamma=self.gamma,
            velocity=self.velocity,
            I_target=I_target,
            diversity=approach_diversity,
            entropy=self.entropy,
            phase_cont=self.phase_cont,
            phase_label=phase_label
        )

        if approach_diversity > 0.0055 and len(self.checkpoints) < 6:
            self.checkpoints.append(current_state)
            if len(self.checkpoints) > 5:
                self.checkpoints.pop(0)

        self.structural_reserve.append(current_state)

        # ③ FieldAnalyzer に収集
        self.field_analyzer.collect(
            fv, fa,
            residual=self.residual.effective(),
            potential=self.potential,
            entropy=self.entropy,
            phase=self.phase_cont,
            step=current_step
        )

        field_m = self.field_analyzer.analyze()
        fluct_class = "residue_coupled" if self.fluctuation_type == FluctuationType.RESIDUE_COUPLED else field_m.get("status", "active")

        metrics = {
            "step": current_step,
            "gamma": round(self.gamma, 4),
            "velocity": round(self.velocity, 5),
            "I_target": round(I_target, 4),
            "residual_thermal": round(self.residual.thermal, 2),
            "residual_effective": round(self.residual.effective(), 2),
            "potential": round(self.potential, 3),
            "entropy": round(self.entropy, 3),
            "free_energy": round(free_energy, 3),
            "approach_diversity": round(approach_diversity, 5),
            "near_target": near_target,
            "triggered_recovery": triggered_recovery,
            "released_work": round(released_work, 4),
            "phase_cont": round(self.phase_cont, 3),
            "phase_label": phase_label,
            "recovery_count": self.recovery_count,
            "fluctuation_type": self.fluctuation_type.name,
            "fluct_class": fluct_class,
            "fluct_res_corr": round(field_m.get("residue_corr", 0.0), 3),
            "fv": round(fv, 4),
            "fa": round(fa, 4)
        }
        self.history_metrics.append(metrics)
        return metrics

    # ⑤ Recovery を「自己組織化混合」へ変更
    def _self_organizing_recovery(self, current_I_target: float, current_diversity: float,
                                   current_free_energy: float) -> Tuple[bool, float, float, float]:
        candidates = list(self.checkpoints) + list(self.structural_reserve)[-6:]
        if not candidates:
            return False, 0.0, 0.0, 0.0

        # スコア計算（旧 best_score ロジックを残しつつ）
        scored = []
        for cand in candidates:
            locality = 1.0 / (1.0 + abs(cand.gamma - self.gamma) * 11)
            diversity_gain = max(0.0, cand.diversity - current_diversity) * 3.0
            reach_gain = max(0.0, (cand.I_target - current_I_target) * 0.85)
            fe_bonus = max(0.0, (current_free_energy - (cand.gamma * 0.3)) * 0.4)
            score = locality + diversity_gain + reach_gain + fe_bonus
            scored.append((score, cand))

        if not scored:
            return False, 0.0, 0.0, 0.0

        # 自己組織化: 上位数個を重み付け混合（少しずつ良い方向が残る）
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(scored))
        total_score = sum(s for s, _ in scored[:top_k]) + 1e-8
        weights = [s / total_score for s, _ in scored[:top_k]]
        mix_cands = [c for _, c in scored[:top_k]]

        # 重み付き平均で新しい状態を生成（創発的な混合）
        new_gamma = sum(w * c.gamma for w, c in zip(weights, mix_cands))
        new_velocity = sum(w * c.velocity for w, c in zip(weights, mix_cands)) * 0.6
        mix_ratio = 0.6 + 0.3 * (top_k / 3.0)   # 混合の強さ指標

        # 現在の状態と少しブレンド（急激な変化を避ける自己組織化）
        blend = 0.65
        self.gamma = float(np.clip(blend * self.gamma + (1 - blend) * new_gamma, 0.02, 0.46))
        self.velocity = new_velocity + (1 - blend) * self.velocity * 0.4

        # 熱 → Work + Entropy（bonus は coupling 強度で）
        noise_bonus = 1.0
        field_m = self.field_analyzer.analyze()
        if field_m.get("residue_corr", 0.0) > 0.25:
            noise_bonus = 1.15

        released_work = self.residual.effective() * self.energy_conversion_efficiency * noise_bonus
        entropy_increase = self.residual.effective() * (1.0 - self.energy_conversion_efficiency)

        self.velocity += 0.017 * (1 if new_gamma > self.gamma else -1)
        self.entropy += entropy_increase
        self.residual.thermal = max(0.0, self.residual.thermal * 0.32)

        self.event_counter += 1
        event = RecoveryEvent(
            event_id=f"recov_{self.event_counter}",
            step=len(self.history_metrics),
            residual_before=round(self.residual.effective() + self.residual.effective() * 0.68, 2),
            entropy_before=round(self.entropy - entropy_increase, 3),
            released_work=round(released_work, 4),
            entropy_increase=round(entropy_increase, 4),
            velocity_after=round(self.velocity, 5),
            free_energy_before=round(current_free_energy, 3),
            score=round(scored[0][0], 3),
            reason=f"self_org_mix_{top_k}_candidates",
            chosen_action="mixed_self_organizing",
            mix_ratio=round(mix_ratio, 3)
        )
        self.recovery_events.append(event)
        return True, released_work, entropy_increase, mix_ratio

    def run(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        n = steps or self.steps
        for _ in range(n):
            self.step()
        return self.history_metrics

    def get_summary(self) -> Dict[str, Any]:
        diversities = [m["approach_diversity"] for m in self.history_metrics if m.get("approach_diversity", 0) > 0]
        final_field = self.field_analyzer.analyze() if len(self.field_analyzer.fluct_v) >= 8 else {}
        return {
            "final_gamma": round(self.gamma, 4),
            "final_residual_effective": round(self.residual.effective(), 2),
            "final_potential": round(self.potential, 3),
            "final_entropy": round(self.entropy, 3),
            "recovery_count": self.recovery_count,
            "mean_diversity": round(np.mean(diversities), 5) if diversities else 0.0,
            "recovery_events": len(self.recovery_events),
            "fluctuation_type_final": self.fluctuation_type.name,
            "final_phase_cont": round(self.phase_cont, 3),
            "final_phase_label": self.field_analyzer.get_phase_label(self.phase_cont),
            "final_fluct_res_corr": round(final_field.get("residue_corr", 0.0), 3) if final_field else 0.0,
            "total_steps": len(self.history_metrics)
        }


if __name__ == "__main__":
    print("=" * 78)
    print("=== ARK2 Theoretical Refactor（Fluctuation中心・自己組織化回復・Field解析） ===")
    print("  理論循環: Potential → Fluctuation → Interaction → ResidualState → FreeEnergy → Recovery(mix) → Emergence")
    print("  実験結果に縛られず、理論的一貫性を優先。sandbox で忠実に multi-seed 検証")
    print("=" * 78)

    seeds = [42, 123, 777]
    for sd in seeds:
        ark2 = ARK2FreeEnergy(
            steps=220,
            initial_gamma=0.10,
            residual_threshold=17.5,
            fluctuation_type=FluctuationType.RESIDUE_COUPLED,
            seed=sd
        )
        history = ark2.run()
        summary = ark2.get_summary()
        print(f"\n[Seed {sd}] 回復回数: {summary['recovery_count']}, "
              f"残溜effective: {summary['final_residual_effective']}, "
              f"Potential: {summary['final_potential']}, "
              f"Phase: {summary['final_phase_cont']} ({summary['final_phase_label']}), "
              f"fluct_res_corr: {summary['final_fluct_res_corr']}")

    # 代表 run
    print("\n【代表 run (seed=42) 最結サマリー】")
    ark2 = ARK2FreeEnergy(steps=220, initial_gamma=0.10, residual_threshold=17.5,
                          fluctuation_type=FluctuationType.RESIDUE_COUPLED, seed=42)
    history = ark2.run()
    for k, v in ark2.get_summary().items():
        print(f"  {k}: {v}")

    print("\n【自己組織化回復イベント例（複数候補混合）】")
    for ev in ark2.recovery_events[-2:]:
        print(f"  {ev.event_id}: {ev.chosen_action} | mix_ratio={ev.mix_ratio:.2f} | "
              f"work={ev.released_work:.3f} | ΔS={ev.entropy_increase:.3f}")

    print("\n" + "=" * 78)
    print("理論的再設計版 ARK2 の忠実実験完了。")
    print("Fluctuation を中心に揚え、ResidualState の一般化、自己組織化回復、")
    print("連続 Phase、拡張 Free Energy を実装。長期的な理論発展に適した形になりました。")
    print("このファイルを ARK2/ark2_free_energy.py に反映してください。")
    print("=" * 78)
