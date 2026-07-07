#!/usr/bin/env python3
"""
AetherDynamics v1.1 - Research OS Framework
設計思想:
1. 自動パラメータスイープ: Conditionの手動定義から、ParameterGridによる全探索へ。
2. 堅牢な統計基盤: Shapiro-Wilk検定による正規性確認と、Welch / Mann-Whitney U の動的切替。
3. 多重比較補正: Benjamini-Hochberg (FDR) 法によるp値の補正を標準化。
4. フルオート・アーティファクト: 一回の実行で生データ(Parquet/CSV)、統計結果、Plot、Markdown、LaTeX表を全自動生成。
"""

import os
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type, Tuple

# ============================================================
# [Layer 1-2: Core & Physics] (安定のため変更を最小化)
# ============================================================
class StateSchema:
    def __init__(self, variables: Dict[str, Type]): self.variables = variables
    def get_dtype(self, var_name: str) -> Type: return self.variables[var_name]

class StateStorage:
    def __init__(self, num_nodes: int, schema: StateSchema):
        self.num_nodes = num_nodes
        self.schema = schema
        self.arrays = {name: np.zeros(num_nodes, dtype=dtype) for name, dtype in schema.variables.items()}

    def copy(self) -> 'StateStorage':
        new_storage = StateStorage(self.num_nodes, self.schema)
        for name, arr in self.arrays.items(): new_storage.arrays[name] = np.copy(arr)
        return new_storage

class NetworkTopology:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    def copy(self) -> 'NetworkTopology':
        new_topo = NetworkTopology(self.num_nodes)
        new_topo.adj_matrix = np.copy(self.adj_matrix)
        return new_topo

class Environment:
    def __init__(self, storage: StateStorage, topology: NetworkTopology, seed: int):
        self.storage, self.topology, self.time = storage, topology, 0.0
        self.rng = np.random.default_rng(seed)

class BasePhysicsEngine:
    def integrate(self, env: Environment, next_storage: StateStorage, dt: float): raise NotImplementedError

class VoidPhysicsEngine(BasePhysicsEngine):
    def __init__(self, interaction_scale: float, noise_scale: float):
        self.interaction_scale, self.noise_scale = interaction_scale, noise_scale
    def integrate(self, env: Environment, next_storage: StateStorage, dt: float):
        v, r, alive = env.storage.arrays["v"], env.storage.arrays["residue"], env.storage.arrays["alive"]
        N = env.storage.num_nodes
        interactions = np.tanh(v[None, :] - v[:, None]) * env.topology.adj_matrix
        interaction_term = np.sum(interactions, axis=1) / np.maximum(1, np.sum(env.topology.adj_matrix > 0, axis=1))
        noise = env.rng.uniform(-1, 1, size=N) * self.noise_scale
        dv = (interaction_term * self.interaction_scale + np.tanh(r) * 0.1 + noise) * alive
        next_storage.arrays["v"][:] = v + dv * dt
        next_storage.arrays["residue"][:] = (r * 0.9 + next_storage.arrays["v"][:] * 0.05) * alive

# ============================================================
# [Layer 3: Observer Plugin] (NEW: 多様な観測器の拡充)
# ============================================================
class BaseObserver:
    def measure(self, env: Environment) -> float: raise NotImplementedError

class OrderParameterObserver(BaseObserver):
    def measure(self, env: Environment) -> float:
        alive = env.storage.arrays["alive"]
        return float(1.0 / (1.0 + np.std(env.storage.arrays["v"][alive]))) if np.any(alive) else 0.0

class VelocityVarianceObserver(BaseObserver):
    def measure(self, env: Environment) -> float:
        alive = env.storage.arrays["alive"]
        return float(np.var(env.storage.arrays["v"][alive])) if np.any(alive) else 0.0

class StateEntropyObserver(BaseObserver):
    """状態分布の簡易シャノンエントロピー"""
    def measure(self, env: Environment) -> float:
        v = env.storage.arrays["v"][env.storage.arrays["alive"]]
        if len(v) < 2: return 0.0
        hist, _ = np.histogram(v, bins=10, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist + 1e-9)))

class ClusterCountObserver(BaseObserver):
    """同期クラスタ数の推定（連結成分数）"""
    def measure(self, env: Environment) -> float:
        alive = env.storage.arrays["alive"]
        if not np.any(alive): return 0.0
        # 生きているノード間のエッジのみ抽出
        active_adj = env.topology.adj_matrix[np.ix_(alive, alive)]
        _, n_components = connected_components(active_adj > 0, directed=False)
        return float(n_components)

# ============================================================
# [Layer 4: Parameter Sweep & Runner] (NEW: 自動探索の導入)
# ============================================================
class ParameterSweep:
    """辞書から全組み合わせの条件（Condition）を自動生成する"""
    @staticmethod
    def generate_conditions(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys, values = zip(*param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

class ExperimentRunner:
    def __init__(self, base_storage: StateStorage, base_topology: NetworkTopology, observers: Dict[str, BaseObserver]):
        self.base_storage = base_storage
        self.base_topology = base_topology
        self.observers = observers

    def run_sweep(self, param_grid: Dict[str, List[Any]], seeds: List[int], max_steps: int, dt: float) -> pd.DataFrame:
        conditions = ParameterSweep.generate_conditions(param_grid)
        all_results = []
        
        for condition in conditions:
            # 識別用の条件名生成 (例: "int0.2_noise0.1")
            cond_name = "_".join([f"{k[:3]}{v}" for k, v in condition.items()])
            
            for seed in seeds:
                env = Environment(self.base_storage.copy(), self.base_topology.copy(), seed)
                env.storage.arrays["v"][:] = env.rng.uniform(-1.0, 1.0, size=env.storage.num_nodes)
                env.storage.arrays["alive"][:] = True

                physics = VoidPhysicsEngine(
                    interaction_scale=condition.get("interaction_scale", 0.2),
                    noise_scale=condition.get("noise_scale", 0.1)
                )

                # Simulation loop inlined for brevity in OS runner
                shadow = env.storage.copy()
                records = []
                for step in range(max_steps):
                    for name in shadow.arrays: shadow.arrays[name][:] = env.storage.arrays[name]
                    physics.integrate(env, shadow, dt)
                    for name in env.storage.arrays: env.storage.arrays[name][:] = shadow.arrays[name]
                    
                    telemetry = {"time": env.time, "condition": cond_name, "seed": seed}
                    for obs_name, obs in self.observers.items():
                        telemetry[obs_name] = obs.measure(env)
                    records.append(telemetry)
                    env.time += dt
                
                all_results.append(pd.DataFrame(records))
                
        return pd.concat(all_results, ignore_index=True)

# ============================================================
# [Layer 5: Advanced Statistical Analyzer] (NEW: 動的検定と多重比較補正)
# ============================================================
class StatisticalAnalyzer:
    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """依存関係なしにFDRを制御する多重比較補正"""
        n = len(p_values)
        if n == 0: return np.array([]), np.array([])
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        q_values = sorted_p * n / np.arange(1, n + 1)
        q_values = np.minimum.accumulate(q_values[::-1])[::-1] # 単調増加の保証
        significant = q_values < alpha
        
        res_q = np.empty(n)
        res_sig = np.empty(n, dtype=bool)
        res_q[sorted_idx], res_sig[sorted_idx] = q_values, significant
        return res_q, res_sig

    @classmethod
    def analyze_conditions(cls, df: pd.DataFrame, control_cond: str, target_cond: str, time_point: float, metrics: List[str]) -> List[Dict[str, Any]]:
        df_time = df[np.isclose(df['time'], time_point)]
        results = []
        raw_p_values = []

        for metric in metrics:
            group_c = df_time[df_time['condition'] == control_cond][metric].values
            group_t = df_time[df_time['condition'] == target_cond][metric].values

            # 1. Shapiro-Wilk 検定による正規性チェック (α=0.05)
            _, p_shap_c = stats.shapiro(group_c)
            _, p_shap_t = stats.shapiro(group_t)
            is_normal = (p_shap_c > 0.05) and (p_shap_t > 0.05)

            # 2. 動的な検定の選択
            if is_normal:
                test_name = "Welch's t-test"
                _, p_val = stats.ttest_ind(group_c, group_t, equal_var=False)
            else:
                test_name = "Mann-Whitney U"
                _, p_val = stats.mannwhitneyu(group_c, group_t, alternative='two-sided')

            # 3. Cohen's d (効果量)
            mean_c, std_c = np.mean(group_c), np.std(group_c, ddof=1)
            mean_t, std_t = np.mean(group_t), np.std(group_t, ddof=1)
            pool_sd = np.sqrt(((len(group_c)-1)*std_c**2 + (len(group_t)-1)*std_t**2) / (len(group_c)+len(group_t)-2))
            cohen_d = (mean_t - mean_c) / pool_sd if pool_sd != 0 else 0.0

            raw_p_values.append(p_val)
            results.append({
                "metric": metric, "test_used": test_name,
                "control_mean": mean_c, "target_mean": mean_t,
                "delta": mean_t - mean_c, "cohens_d": cohen_d, "raw_p": p_val
            })

        # 4. Benjamini-Hochberg (FDR) による多重比較補正
        q_vals, sig_flags = cls._benjamini_hochberg(np.array(raw_p_values), alpha=0.05)
        for i, res in enumerate(results):
            res["fdr_q"] = q_vals[i]
            res["significant"] = sig_flags[i]
            
        return results

# ============================================================
# [Layer 6: Artifact & Report Manager] (NEW: フルオート生成)
# ============================================================
class ArtifactManager:
    def __init__(self, base_dir: str = "./experiment_out"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "plots").mkdir(exist_ok=True)

    def dump_raw(self, df: pd.DataFrame):
        df.to_parquet(self.base_dir / "raw_data.parquet")
        df.to_csv(self.base_dir / "raw_data.csv", index=False)

    def plot_distributions(self, df: pd.DataFrame, time_point: float, metrics: List[str]):
        df_time = df[np.isclose(df['time'], time_point)]
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.violinplot(x="condition", y=metric, data=df_time, inner="quartile")
            plt.title(f"Distribution of {metric} at t={time_point}")
            plt.tight_layout()
            plt.savefig(self.base_dir / "plots" / f"{metric}_violin.png")
            plt.close()

    def generate_report(self, stats_results: List[Dict[str, Any]]):
        df_stats = pd.DataFrame(stats_results)
        df_stats.to_csv(self.base_dir / "statistics_summary.csv", index=False)

        # Markdown Report
        md_lines = ["# Experiment Analysis Report\n\n## Statistical Results\n"]
        md_lines.append("| Metric | Test Used | Delta | Cohen's d | Raw p-value | FDR q-value | Sig (α=0.05) |")
        md_lines.append("|--------|-----------|-------|-----------|-------------|-------------|--------------|")
        for r in stats_results:
            sig_mark = "**Yes**" if r["significant"] else "No"
            md_lines.append(f"| {r['metric']} | {r['test_used']} | {r['delta']:+.4f} | {r['cohens_d']:.4f} | {r['raw_p']:.4e} | {r['fdr_q']:.4e} | {sig_mark} |")
        
        with open(self.base_dir / "report.md", "w") as f:
            f.write("\n".join(md_lines))

        # LaTeX Table for Papers
        tex_table = df_stats[['metric', 'test_used', 'cohens_d', 'fdr_q']].to_latex(index=False, float_format="%.4f")
        with open(self.base_dir / "table.tex", "w") as f:
            f.write(tex_table)

# ============================================================
# [Research OS Interface: The "Holy Grail"]
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("AetherDynamics OS - Automated Pipeline Booting...")
    print("=" * 70)

    # 1. 基盤セットアップ
    schema = StateSchema({"v": np.float64, "residue": np.float64, "alive": bool})
    N = 30
    storage = StateStorage(N, schema)
    topology = NetworkTopology(N)
    for i in range(N): topology.adj_matrix[i, (i + 1) % N] = 1.0  # Ring

    observers = {
        "sync_order": OrderParameterObserver(),
        "vel_variance": VelocityVarianceObserver(),
        "state_entropy": StateEntropyObserver(),
        "cluster_count": ClusterCountObserver()
    }

    # 2. 自動実験・スイープ定義 (Control群 vs Noise増加群)
    sweep_grid = {
        "interaction_scale": [0.3],
        "noise_scale": [0.05, 0.20]  # ここがControl vs Targetになる
    }
    
    # 3. OS起動 (一気通貫実行)
    runner = ExperimentRunner(storage, topology, observers)
    print("[1/4] Running Parameter Sweep Simulation (N=50 seeds)...")
    df_raw = runner.run_sweep(param_grid=sweep_grid, seeds=list(range(50)), max_steps=20, dt=1.0)

    print("[2/4] Executing Statistical Analysis (Dynamic Tests & FDR)...")
    analyzer = StatisticalAnalyzer()
    stats_out = analyzer.analyze_conditions(
        df=df_raw, 
        control_cond="int0.3_noi0.05", 
        target_cond="int0.3_noi0.2", 
        time_point=19.0, 
        metrics=list(observers.keys())
    )

    print("[3/4] Generating Artifacts (Parquet, CSV, Markdown, LaTeX, Plots)...")
    artifact_mgr = ArtifactManager(base_dir="./research_artifacts")
    artifact_mgr.dump_raw(df_raw)
    artifact_mgr.plot_distributions(df_raw, time_point=19.0, metrics=list(observers.keys()))
    artifact_mgr.generate_report(stats_out)

    print("[4/4] Pipeline Complete. Artifacts saved in './research_artifacts'.")