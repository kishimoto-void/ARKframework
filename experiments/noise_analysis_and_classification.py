#!/usr/bin/env python3
"""
ARKframework Extension: Noise Analysis and Classification
対象: ARKv モジュール内のノイズ生成・影響解析・分類
- 既存の NoiseGenerator / PotentialPulseSolver / MultiverseExperiment の設計思想を尊重
- ノイズタイプ拡張 (White, Pink, Brown, Structured)
- NoiseAnalyzer: 統計特徴量 (var, skew, kurt, acf, spectral_slope, residue/phase相関)
- NoiseClassifier: ルールベース多軸分類 (amplitude x correlation x coupling)
- 忠実な多条件実験 (multiverse風並行世界 + 複数trial平均)
- 可視化と結果保存

This module can be integrated into ARKv/ or run as standalone experiment.
"""

import random
import math
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from enum import Enum, auto
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# 拡張ノイズ生成 (既存 NoiseGenerator を基に多タイプ化)
# ============================================================
class NoiseType(Enum):
    WHITE_UNIFORM = auto()
    WHITE_GAUSSIAN = auto()
    PINK = auto()          # 1/f-like (simple IIR filter)
    BROWN = auto()         # random-walk like (integrated)
    STRUCTURED_RESIDUE = auto()  # residue相関型 (ARKvのresidue couplingを強調)
    STRUCTURED_PHASE = auto()    # phase変調型

class AdvancedNoiseGenerator:
    """既存の NoiseGenerator を拡張した多タイプ対応版"""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed if seed is not None else random.randint(0, 2**32-1))
        self.pink_state = 0.0
        self.brown_v = 0.0

    def generate(self, noise_mod: float = 1.0, 
                 noise_type: NoiseType = NoiseType.WHITE_GAUSSIAN,
                 residue: float = 0.0, phase: float = 0.0) -> Tuple[float, float]:
        """nv: velocity noise, na: acceleration noise を返す"""
        if noise_type == NoiseType.WHITE_GAUSSIAN:
            nv = self.rng.gauss(0.0, 0.20) * noise_mod
            na = self.rng.gauss(0.0, 0.10) * noise_mod
        elif noise_type == NoiseType.WHITE_UNIFORM:
            nv = self.rng.uniform(-0.20, 0.20) * noise_mod
            na = self.rng.uniform(-0.10, 0.10) * noise_mod
        elif noise_type == NoiseType.PINK:
            white = self.rng.gauss(0.0, 0.20)
            self.pink_state = 0.85 * self.pink_state + white * 0.15
            nv = self.pink_state * noise_mod * 0.6
            na = self.rng.gauss(0.0, 0.06) * noise_mod
        elif noise_type == NoiseType.BROWN:
            dv = self.rng.gauss(0.0, 0.04)
            self.brown_v = self.brown_v * 0.92 + dv
            nv = self.brown_v * noise_mod * 1.2
            na = self.rng.gauss(0.0, 0.03) * noise_mod
        elif noise_type == NoiseType.STRUCTURED_RESIDUE:
            base = self.rng.gauss(0.0, 0.15)
            nv = (base + 0.08 * math.tanh(residue * 0.8)) * noise_mod
            na = (self.rng.gauss(0.0, 0.07) + 0.04 * residue) * noise_mod
        else:  # STRUCTURED_PHASE
            mod = 1.0 + 0.4 * math.sin(phase * 1.3)
            nv = self.rng.gauss(0.0, 0.18) * noise_mod * mod
            na = self.rng.gauss(0.0, 0.09) * noise_mod * (1.0 + 0.2 * math.cos(phase))
        return nv, na

# ============================================================
# ノイズ解析器 (ARKvの StatisticsObserver / TemporalObserver 思想を拡張)
# ============================================================
class NoiseAnalyzer:
    """ノイズ時系列の統計的特徴量抽出と影響解析"""
    def __init__(self):
        self.samples_v: List[float] = []
        self.samples_a: List[float] = []
        self.residues: List[float] = []
        self.phases: List[float] = []
        self.steps: List[int] = []

    def collect(self, nv: float, na: float, residue: float = 0.0, phase: float = 0.0, step: int = 0):
        self.samples_v.append(float(nv))
        self.samples_a.append(float(na))
        self.residues.append(float(residue))
        self.phases.append(float(phase))
        self.steps.append(step)

    def analyze(self) -> Dict[str, float]:
        if len(self.samples_v) < 8:
            return {"status": "insufficient_data", "count": len(self.samples_v)}
        v = np.asarray(self.samples_v, dtype=float)
        a = np.asarray(self.samples_a, dtype=float)
        res = np.asarray(self.residues, dtype=float)
        ph = np.asarray(self.phases, dtype=float)

        metrics: Dict[str, float] = {
            "count": len(v),
            "mean_v": float(np.mean(v)),
            "std_v": float(np.std(v)),
            "var_v": float(np.var(v)),
            "skew_v": float(stats.skew(v, bias=False)),
            "kurt_v": float(stats.kurtosis(v, bias=False)),
            "mean_a": float(np.mean(a)),
            "std_a": float(np.std(a)),
            "acf_lag1_v": self._safe_autocorr(v, 1),
            "acf_lag2_v": self._safe_autocorr(v, 2),
            "residue_corr_v": self._safe_corr(v, res),
            "phase_corr_v": self._safe_corr(v, ph),
        }

        # スペクトルスロープ近似 (低周波域 log-log fit) - pink/brown判別に有効
        try:
            f, Pxx = stats.periodogram(v, fs=1.0, scaling='spectrum')
            if len(f) > 12:
                idx = slice(1, len(f)//3)  # 低〜中周波
                logf = np.log(f[idx] + 1e-12)
                logP = np.log(Pxx[idx] + 1e-12)
                slope, _ = np.polyfit(logf, logP, 1)
                metrics["spectral_slope"] = float(slope)
            else:
                metrics["spectral_slope"] = 0.0
        except Exception:
            metrics["spectral_slope"] = 0.0

        # 追加: ノイズエネルギー (ARKvの脈動・ノイズ積分に相当)
        metrics["noise_energy"] = float(np.sum(v**2) / len(v))
        return metrics

    def _safe_autocorr(self, x: np.ndarray, lag: int) -> float:
        if len(x) <= lag + 2:
            return 0.0
        try:
            return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
        except Exception:
            return 0.0

    def _safe_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 4 or np.std(y) < 1e-8 or np.std(x) < 1e-8:
            return 0.0
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except Exception:
            return 0.0

    def classify(self) -> str:
        """多軸ルールベース分類 (amplitude × temporal structure × coupling)"""
        m = self.analyze()
        if m.get("status") == "insufficient_data":
            return "unknown_insufficient_data"

        var = m["var_v"]
        acf1 = m["acf_lag1_v"]
        res_corr = abs(m.get("residue_corr_v", 0.0))
        spec_slope = m.get("spectral_slope", 0.0)
        kurt = m["kurt_v"]
        energy = m["noise_energy"]

        # Amplitude axis
        if var < 0.025:
            amp = "benign_low_amp"
        elif var > 0.65:
            amp = "disruptive_high_amp"
        elif var > 0.28:
            amp = "moderate_high_var"
        else:
            amp = "moderate"

        # Temporal correlation axis
        if acf1 > 0.45:
            temp = "_persistent_long_memory"
        elif acf1 > 0.20:
            temp = "_mildly_persistent"
        elif acf1 < -0.25:
            temp = "_anti_correlated_oscillatory"
        else:
            temp = "_uncorrelated_white_like"

        # Coupling / color axis
        if res_corr > 0.35:
            coup = "_residue_coupled"
        elif spec_slope < -0.95:
            coup = "_pink_colored"
        elif spec_slope > 0.15:
            coup = "_brown_wandering"
        elif -0.6 < spec_slope < -0.15:
            coup = "_pinkish"
        else:
            coup = "_white_spectral"

        # Tail / outlier axis (kurtosis)
        if kurt > 3.5:
            tail = "_heavy_tailed_outliers"
        elif kurt < -0.8:
            tail = "_platykurtic"
        else:
            tail = ""

        # Energy context (ARKvの脈動積分に相当する影響度)
        if energy > 0.12:
            energy_tag = "_high_energy"
        else:
            energy_tag = ""

        return f"{amp}{temp}{coup}{tail}{energy_tag}"

# ============================================================
# 簡易ダイナミクス (ARKv PotentialPulseSolver + ResidueSolver の忠実簡易版)
# ============================================================
@dataclass
class SimpleNodeState:
    v: float = 0.0
    a: float = 0.5
    residue: float = 0.0
    phase: float = 0.0

class SimpleARKvDynamics:
    """
    ARKv の PotentialPulseSolver / ResidueSolver / ThresholdDetector の
    核心ロジックを忠実に再現した簡易1ノード版 (実験用)
    """
    def __init__(self, noise_type: NoiseType = NoiseType.WHITE_GAUSSIAN, seed: Optional[int] = None):
        self.noise_gen = AdvancedNoiseGenerator(seed)
        self.noise_type = noise_type
        self.state = SimpleNodeState(phase=random.uniform(0, 2 * math.pi))
        self.history: Dict[str, List[float]] = {"v": [], "residue": [], "nv": [], "na": []}
        self.analyzer = NoiseAnalyzer()

    def reset(self):
        self.state = SimpleNodeState(phase=random.uniform(0, 2 * math.pi))
        for k in self.history:
            self.history[k].clear()
        self.analyzer = NoiseAnalyzer()

    def step(self, noise_mod: float = 1.0, elasticity_mod: float = 1.0, external: float = 0.0) -> Tuple[float, float]:
        # --- PulseGenerator ---
        pulsation = 0.1 * math.sin(self.state.phase)
        self.state.phase += 0.12  # pulse_speed approx

        # --- NoiseGenerator (拡張版) ---
        nv, na = self.noise_gen.generate(noise_mod, self.noise_type, self.state.residue, self.state.phase)

        # --- ElasticIntegrator (簡易) ---
        smooth_in = 0.12 * external * elasticity_mod

        # --- Potential update (PotentialPulseSolver 核心) ---
        self.state.v += nv + smooth_in + 0.1 * math.tanh(self.state.residue) + pulsation
        self.state.a += na

        # --- ThresholdDetector / clamp ---
        self.state.v = max(-2.5, min(2.5, self.state.v))
        self.state.a = max(0.0, min(1.0, self.state.a))

        # --- ResidueSolver (簡易版) ---
        self.state.residue = self.state.residue * 0.90 + self.state.v * 0.045

        # record
        self.history["v"].append(self.state.v)
        self.history["residue"].append(self.state.residue)
        self.history["nv"].append(nv)
        self.history["na"].append(na)
        self.analyzer.collect(nv, na, self.state.residue, self.state.phase, len(self.history["v"]))

        return nv, na

# ============================================================
# ノイズ実験エンジン (MultiverseExperiment 風の多条件並行比較)
# ============================================================
def run_faithful_noise_experiment(steps: int = 50, trials: int = 5) -> List[Dict[str, Any]]:
    """複数ノイズ設定で忠実に実験実施 → 解析・分類"""
    configs = [
        ("Normal_White", NoiseType.WHITE_GAUSSIAN, 1.00),
        ("Noise_Amplified_White", NoiseType.WHITE_GAUSSIAN, 1.15),
        ("Pink_Moderate", NoiseType.PINK, 1.08),
        ("Brown_Persistent", NoiseType.BROWN, 1.03),
        ("Structured_Residue_Coupled", NoiseType.STRUCTURED_RESIDUE, 1.00),
        ("Structured_Phase_Modulated", NoiseType.STRUCTURED_PHASE, 1.05),
    ]

    all_results = []
    for name, ntype, nmod in configs:
        trial_results = []
        for t in range(trials):
            dyn = SimpleARKvDynamics(noise_type=ntype, seed=1000 + t * 17)
            dyn.reset()
            for _ in range(steps):
                dyn.step(noise_mod=nmod, elasticity_mod=1.0)

            m = dyn.analyzer.analyze()
            cls = dyn.analyzer.classify()

            # ARKv風メトリクス
            v_arr = np.array(dyn.history["v"])
            res_arr = np.array(dyn.history["residue"])
            stability = 1.0 / (1.0 + np.var(v_arr[-min(15, len(v_arr)):]) + 1e-8)
            lyapunov_proxy = float(np.log(np.std(v_arr[-10:]) / (np.std(v_arr[:10]) + 1e-8) + 1e-8)) if len(v_arr) > 20 else 0.0
            max_res = float(np.max(np.abs(res_arr)))
            energy = m.get("noise_energy", 0.0)

            trial_results.append({
                "trial": t + 1,
                "classification": cls,
                "std_v": round(m.get("std_v", 0), 4),
                "acf_lag1": round(m.get("acf_lag1_v", 0), 3),
                "res_corr": round(m.get("residue_corr_v", 0), 3),
                "spectral_slope": round(m.get("spectral_slope", 0), 3),
                "final_residue": round(dyn.state.residue, 4),
                "max_residue": round(max_res, 4),
                "stability": round(stability, 4),
                "lyapunov_proxy": round(lyapunov_proxy, 4),
                "noise_energy": round(energy, 4),
            })

        # 平均・代表値
        dominant_cls = max(set(tr["classification"] for tr in trial_results),
                           key=lambda c: sum(1 for tr in trial_results if tr["classification"] == c))
        avg_std = float(np.mean([tr["std_v"] for tr in trial_results]))
        avg_stab = float(np.mean([tr["stability"] for tr in trial_results]))
        avg_lyap = float(np.mean([tr["lyapunov_proxy"] for tr in trial_results]))

        all_results.append({
            "config_name": name,
            "noise_type": ntype.name,
            "noise_mod": nmod,
            "trials": trial_results,
            "dominant_classification": dominant_cls,
            "avg_std_v": round(avg_std, 4),
            "avg_stability": round(avg_stab, 4),
            "avg_lyapunov_proxy": round(avg_lyap, 4),
        })
    return all_results

# ============================================================
# 可視化 (ARKv ConsoleReporter + 視覚的解析)
# ============================================================
def generate_visualizations(results: List[Dict], output_dir: str = "/home/workdir/artifacts"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. サマリーテーブル風 bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    names = [r["config_name"] for r in results]
    stds = [r["avg_std_v"] for r in results]
    stabs = [r["avg_stability"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, stds, width, label='Avg Std(v) [noise amplitude]', color='#4472C4')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, stabs, width, label='Avg Stability', color='#ED7D31')
    ax.set_ylabel('Noise Std (v)', color='#4472C4')
    ax2.set_ylabel('Stability (higher=better)', color='#ED7D31')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_title('ARKframework Noise Experiment: Amplitude vs Stability Trade-off\n( faithful multiverse-style comparison )')
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ark_noise_amplitude_stability_{timestamp}.png", dpi=140, bbox_inches='tight')
    plt.close()

    # 2. 分類分布 pie
    all_cls = [r["dominant_classification"] for r in results]
    unique_cls, counts = np.unique(all_cls, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cls)))
    ax.pie(counts, labels=unique_cls, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.set_title('Dominant Noise Classifications across Configurations\n(多軸特徴量に基づくルールベース分類)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ark_noise_classification_distribution_{timestamp}.png", dpi=140, bbox_inches='tight')
    plt.close()

    print(f"\n[可視化] プロットを {output_dir} に保存しました。")
    return timestamp

# ============================================================
# メイン実行 (実験忠実実施)
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ARKframework - ノイズの解析と分類 実験モジュール")
    print("  ARKv PotentialPulseSolver / ResidueSolver 設計を尊重した拡張")
    print("  実験は忠実に実際実施 (sandbox上で多seed・多条件シミュレーション)")
    print("=" * 70)

    results = run_faithful_noise_experiment(steps=55, trials=5)

    print("\n【実験結果サマリー表】")
    print(f"{'Config':<26} | {'NoiseType':<22} | {'Mod':>5} | {'AvgStd(v)':>9} | {'AvgStab':>8} | {'Dominant Classification'}")
    print("-" * 115)
    for r in results:
        print(f"{r['config_name']:<26} | {r['noise_type']:<22} | {r['noise_mod']:>5.2f} | "
              f"{r['avg_std_v']:>9.4f} | {r['avg_stability']:>8.4f} | {r['dominant_classification']}")

    print("\n【各条件の代表的分類と解釈】")
    for r in results:
        print(f"  • {r['config_name']}: {r['dominant_classification']}")
        print(f"    └─ 平均ノイズ振幅={r['avg_std_v']:.4f}, 安定性={r['avg_stability']:.4f}, "
              f"リアプノフ近似={r['avg_lyapunov_proxy']:.4f}")

    # 可視化生成
    ts = generate_visualizations(results)

    # 結果JSON保存 (GitHubにpushするスクリプトと一緒に)
    out_json = f"/home/workdir/artifacts/ark_noise_experiment_results_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ARKframework Noise Analysis & Classification",
            "date": datetime.now().isoformat(),
            "steps_per_trial": 55,
            "trials_per_config": 5,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 詳細結果: {out_json}")

    print("\n" + "=" * 70)
    print("実験完了。ノイズの統計的解析と多軸分類が終了しました。")
    print("このスクリプトを ARKframework/experiments/ に配置してご利用ください。")
    print("=" * 70)
