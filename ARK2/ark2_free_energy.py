#!/usr/bin/env python3
"""
ARK2 Free Energy Version (v2026-07-08)

MetaVoid + Emergence Engine を、熱力学 + 自由エネルギー最小化の框組みで統合した版。

【主要な理論的拡張】
- residue (U)：内部エネルギー / 摩擦熱
- entropy (S)：散逸・探索の不確実性
- Free Energy F = U - T*S の最小化趨勢
- Recovery 時の熱 → Work + Entropy 生成（熱力学第2法則に即した分割）
"""

from dataclasses import dataclass
from collections import deque
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


@dataclass(frozen=True)
class AdaptationState:
    gamma: float
    velocity: float
    I_target: float
    diversity: float
    entropy: float
    phase: str


@dataclass
class RecoveryEvent:
    event_id: str
    step: int
    residue_before: float
    entropy_before: float
    released_work: float
    entropy_increase: float
    velocity_after: float
    free_energy_before: float
    score: float
    reason: str


class ARK2FreeEnergy:
    """
    ARK2 - Free Energy + Thermodynamic Cycle

    - residue を摩擦熱（U）として扱う
    - entropy を独立変数として導入（探索・不確実性）
    - Recovery 時に熱を Work と Entropy に正しく分割
    - 自由エネルギー F = residue - T * entropy を計算（将来的な最小化駆動用）
    """

    def __init__(
        self,
        steps: int = 200,
        initial_gamma: float = 0.10,
        momentum_beta: float = 0.78,
        residue_threshold: float = 24.0,
        friction_coefficient: float = 0.023,
        energy_conversion_efficiency: float = 0.42,
        exploration_temperature: float = 1.8,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = steps
        self.gamma = initial_gamma
        self.momentum_beta = momentum_beta
        self.velocity = 0.0
        self.residue = 0.0
        self.entropy = 0.0
        self.residue_threshold = residue_threshold
        self.friction_coefficient = friction_coefficient
        self.energy_conversion_efficiency = energy_conversion_efficiency
        self.exploration_temperature = exploration_temperature
        self.epsilon = 1e-8

        self.checkpoints: List[AdaptationState] = []
        self.structural_reserve: deque[AdaptationState] = deque(maxlen=16)
        self.recovery_events: List[RecoveryEvent] = []
        self.history_metrics: List[Dict[str, Any]] = []
        self.approach_buffer: deque[float] = deque(maxlen=20)

        self.event_counter = 0
        self._stuck_counter = 0
        self.recovery_count = 0

    def step(self) -> Dict[str, Any]:
        prev_gamma = self.gamma
        prev_velocity = self.velocity

        R = self._calculate_resonance_proxy()
        I_target = self._calculate_dynamic_I_target(R)

        current_step = len(self.history_metrics)
        modulation = 0.16 * np.sin(2 * np.pi * current_step / 118)
        I_target = max(0.13, I_target + modulation)

        critical_noise_ratio = max(0.0005, 0.72 * np.exp(-abs(self.gamma - 0.13) * 2.1))
        variance = max(0.0012, 0.72 * (0.125 - abs(self.gamma - 0.125)))

        tol_I = 0.05 * I_target
        e_I = critical_noise_ratio - I_target
        near_target = abs(e_I) < tol_I

        if near_target:
            contrib_I = 0.0
            self._stuck_counter = 0
        else:
            contrib_I = e_I / (I_target + self.epsilon)
            self._stuck_counter += 1

        dynamic_beta = self.momentum_beta * np.exp(-self.residue * self.friction_coefficient)
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

        if near_target and approach_diversity < 0.0018:
            self.residue += 1.55 + (self._stuck_counter * 0.07)
        else:
            self.residue = max(0.0, self.residue * 0.96)

        free_energy = self.residue - self.exploration_temperature * self.entropy

        triggered_recovery = False
        released_work = 0.0
        entropy_increase = 0.0

        if self.residue >= self.residue_threshold:
            triggered_recovery, released_work, entropy_increase = self._free_energy_recovery(I_target, approach_diversity, free_energy)
            if triggered_recovery:
                self.recovery_count += 1

        phase = self._classify_phase(critical_noise_ratio, variance)
        current_state = AdaptationState(
            gamma=self.gamma,
            velocity=self.velocity,
            I_target=I_target,
            diversity=approach_diversity,
            entropy=self.entropy,
            phase=phase
        )

        if approach_diversity > 0.0055 and len(self.checkpoints) < 6:
            self.checkpoints.append(current_state)
            if len(self.checkpoints) > 5:
                self.checkpoints.pop(0)

        self.structural_reserve.append(current_state)

        metrics = {
            "step": current_step,
            "gamma": round(self.gamma, 4),
            "velocity": round(self.velocity, 5),
            "I_target": round(I_target, 4),
            "residue": round(self.residue, 2),
            "entropy": round(self.entropy, 3),
            "free_energy": round(free_energy, 3),
            "approach_diversity": round(approach_diversity, 5),
            "near_target": near_target,
            "triggered_recovery": triggered_recovery,
            "released_work": round(released_work, 4),
            "phase": phase,
            "recovery_count": self.recovery_count
        }
        self.history_metrics.append(metrics)

        return metrics

    def _free_energy_recovery(self, current_I_target: float, current_diversity: float, current_free_energy: float) -> Tuple[bool, float, float]:
        candidates = list(self.checkpoints) + list(self.structural_reserve)[-6:]
        if not candidates:
            return False, 0.0, 0.0

        best_score = -999.0
        best_cand = None
        best_action = "none"

        for cand in candidates:
            locality = 1.0 / (1.0 + abs(cand.gamma - self.gamma) * 11)
            diversity_gain = max(0.0, cand.diversity - current_diversity) * 3.0
            reach_gain = max(0.0, (cand.I_target - current_I_target) * 0.85)
            redundancy = min(1.6, len([c for c in candidates if abs(c.gamma - cand.gamma) < 0.028]))
            fe_bonus = max(0.0, (current_free_energy - (cand.gamma * 0.3)) * 0.4)

            score = (locality * 1.0 + diversity_gain * 1.25 + reach_gain * 1.0 + redundancy * 0.6 + fe_bonus * 0.9)

            if score > best_score:
                best_score = score
                best_cand = cand
                best_action = "boost_positive" if cand.gamma > self.gamma else "boost_negative"

        if best_cand:
            released_work = self.residue * self.energy_conversion_efficiency
            entropy_increase = self.residue * (1.0 - self.energy_conversion_efficiency)

            direction = 1.0 if best_action == "boost_positive" else -1.0
            boost = direction * (0.017 + released_work * 0.0045)

            self.velocity = boost
            self.entropy += entropy_increase
            self.residue = max(0.0, self.residue * 0.32)

            self.event_counter += 1
            event = RecoveryEvent(
                event_id=f"recov_{self.event_counter}",
                step=len(self.history_metrics),
                residue_before=round(self.residue + (self.residue * 0.68), 2),
                entropy_before=round(self.entropy - entropy_increase, 3),
                released_work=round(released_work, 4),
                entropy_increase=round(entropy_increase, 4),
                velocity_after=round(self.velocity, 5),
                free_energy_before=round(current_free_energy, 3),
                score=round(best_score, 3),
                reason=f"stuck_{self._stuck_counter}_steps_FE={current_free_energy:.2f}"
            )
            self.recovery_events.append(event)
            return True, released_work, entropy_increase

        return False, 0.0, 0.0

    def _calculate_resonance_proxy(self) -> float:
        return max(0.12, 0.66 - abs(self.gamma - 0.16) * 1.85)

    def _calculate_dynamic_I_target(self, R: float) -> float:
        return 0.87 + 0.33 * R

    def _classify_phase(self, critical_noise_ratio: float, variance: float) -> str:
        if critical_noise_ratio > 1.08:
            return "Emergence"
        elif variance < 0.011:
            return "Trapped"
        elif 0.70 <= critical_noise_ratio <= 1.08:
            return "Prepared"
        else:
            return "Dissipative"

    def run(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        n = steps or self.steps
        for _ in range(n):
            self.step()
        return self.history_metrics

    def get_summary(self) -> Dict[str, Any]:
        diversities = [m["approach_diversity"] for m in self.history_metrics if m["approach_diversity"] > 0]
        return {
            "final_gamma": round(self.gamma, 4),
            "final_residue": round(self.residue, 2),
            "final_entropy": round(self.entropy, 3),
            "recovery_count": self.recovery_count,
            "mean_diversity": round(np.mean(diversities), 5) if diversities else 0.0,
            "recovery_events": len(self.recovery_events)
        }


if __name__ == "__main__":
    print("=== ARK2 Free Energy（Entropy + 自由エネルギー版） ===")
    ark2 = ARK2FreeEnergy(steps=180, initial_gamma=0.10, residue_threshold=23.0, seed=42)
    history = ark2.run()

    print("\n【最絒サマリー】")
    for k, v in ark2.get_summary().items():
        print(f"  {k}: {v}")

    print("\n【回復イベント例（熱 → Work + Entropy）】")
    for ev in ark2.recovery_events[-3:]:
        print(f"  {ev.event_id}: {ev.chosen_direction} | work={ev.released_work:.3f} | \u0394S={ev.entropy_increase:.3f}")