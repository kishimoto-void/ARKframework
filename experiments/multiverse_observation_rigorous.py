import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple

@dataclass
class HexaMetrics:
    geometry_score: float = 0.0
    statistics_score: float = 0.0
    temporal_score: float = 0.0
    information_score: float = 0.0
    dynamical_score: float = 0.0
    stability_score: float = 0.0

    def to_list(self) -> List[float]:
        return [self.geometry_score, self.statistics_score, self.temporal_score,
                self.information_score, self.dynamical_score, self.stability_score]

@dataclass
class CoupledVoid:
    node_id: str
    role_type: str               # "HCE_ARK", "ARKv", "ether"
    orig_x: float
    orig_y: float
    x: float = 0.0
    y: float = 0.0
    seed: int = None
    v: float = 0.0
    a: float = 0.5
    residue: float = 0.0
    phase: float = 0.0
    pulse_active: bool = False
    pulse_threshold: float = 1.8
    artery_buffer: float = 0.0
    elasticity: float = 0.3

    def __post_init__(self):
        self.x = self.orig_x
        self.y = self.orig_y
        self.rng = random.Random(self.seed)
        self.pulse_speed = self.rng.uniform(0.05, 0.15)
        self.phase = self.rng.uniform(0, math.pi * 2)

    def reset_state(self):
        self.x = self.orig_x
        self.y = self.orig_y
        self.v = 0.0
        self.a = 0.5
        self.residue = 0.0
        self.phase = 0.0
        self.pulse_active = False
        self.artery_buffer = 0.0
        self.rng = random.Random(self.seed)
        self.pulse_speed = self.rng.uniform(0.05, 0.15)
        self.phase = self.rng.uniform(0, math.pi * 2)

    def step(self, artery_interaction: float, vein_drainage: float, external_pulse: float = 0.0, noise_modifier: float = 1.0, current_elasticity: float = 0.3):
        self.phase += self.pulse_speed
        pulsation = math.sin(self.phase) * 0.1
        
        noise_v = self.rng.uniform(-1, 1) * 0.2 * noise_modifier
        noise_a = self.rng.uniform(-1, 1) * 0.1 * noise_modifier
        memory = math.tanh(self.residue)
        
        self.artery_buffer += (0.15 * artery_interaction + external_pulse)
        # 修正: 破壊的な乗算を避け、引数として渡された現在のステップの弾性を使用
        smooth_artery_input = self.artery_buffer * current_elasticity
        self.artery_buffer -= smooth_artery_input
        
        self.v += noise_v + smooth_artery_input + 0.1 * memory + pulsation
        self.a += noise_a
        
        if self.v > self.pulse_threshold and not self.pulse_active:
            self.pulse_active = True
        else:
            self.pulse_active = False
            
        self.v = max(-2.5, min(2.5, self.v))
        self.a = max(0.0, min(1.0, self.a))
        
        self.residue = self.residue * 0.9 + self.v * 0.05
        self.residue -= vein_drainage * 0.1 
        
        return {"v": self.v, "a": self.a, "r": self.residue, "fired": self.pulse_active}

class GeometryNetwork:
    def __init__(self):
        self.nodes: Dict[str, CoupledVoid] = {}

    def add_body_node(self, node_id: str, role_type: str, x: float, y: float, seed: int):
        if role_type == "HCE_ARK":
            elasticity = 0.6
            pulse_threshold = 2.2
        elif role_type == "ARKv":
            elasticity = 0.2
            pulse_threshold = 1.4
        else:
            elasticity = 0.05
            pulse_threshold = 2.5
            
        self.nodes[node_id] = CoupledVoid(
            node_id=node_id, role_type=role_type, orig_x=x, orig_y=y, seed=seed,
            elasticity=elasticity, pulse_threshold=pulse_threshold
        )

    def generate_body_topology(self):
        self.nodes.clear()
        self.add_body_node("Brain_HCE_ARK", "HCE_ARK", 0.0, 0.0, seed=777)
        for i in range(6):
            angle = i * (math.pi / 3)
            self.add_body_node(f"ether_Field_{i}", "ether", 1.0 * math.cos(angle), 1.0 * math.sin(angle), seed=500+i)
            
        limbs = {
            0: "Right_Hand_ARKv", 1: "Left_Hand_ARKv", 2: "Left_Foot_ARKv",
            3: "Right_Foot_ARKv", 5: "Tail_Ground_ARKv"
        }
        for idx, name in limbs.items():
            angle = idx * (math.pi / 3)
            self.add_body_node(name, "ARKv", 2.0 * math.cos(angle), 2.0 * math.sin(angle), seed=100+idx)

    def reset_network(self):
        for node in self.nodes.values():
            node.reset_state()

    def _get_distance(self, n1, n2) -> float:
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def update_topology_evolution(self):
        node_list = list(self.nodes.values())
        for i, n1 in enumerate(node_list):
            fx, fy = 0.0, 0.0
            for j, n2 in enumerate(node_list):
                if i == j: continue
                dx = n1.x - n2.x
                dy = n1.y - n2.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 0.1: dist = 0.1
                
                repulsion = (n1.artery_buffer + n2.artery_buffer) * 0.02 / (dist ** 2)
                fx += (dx / dist) * repulsion
                
                attraction = (abs(n1.residue) * abs(n2.residue)) * 0.01 * dist
                fx -= (dx / dist) * attraction
                fy -= (dy / dist) * attraction
            
            dist_origin = math.sqrt(n1.x**2 + n1.y**2)
            if dist_origin > 0:
                fx -= (n1.x / dist_origin) * 0.05 * (dist_origin - math.sqrt(n1.orig_x**2 + n1.orig_y**2))
                fy -= (n1.y / dist_origin) * 0.05 * (dist_origin - math.sqrt(n1.orig_x**2 + n1.orig_y**2))

            n1.x += max(-0.2, min(0.2, fx))
            n1.y += max(-0.2, min(0.2, fy))

    def step(self, noise_modifier: float = 1.0, elasticity_modifier: float = 1.0) -> Dict[str, Any]:
        self.update_topology_evolution()
        
        next_states = {}
        node_keys = list(self.nodes.keys())
        for i_id in node_keys:
            node_i = self.nodes[i_id]
            artery_interaction, vein_drainage, incoming_pulse = 0.0, 0.0, 0.0
            for j_id in node_keys:
                if i_id == j_id: continue
                node_j = self.nodes[j_id]
                dist = self._get_distance(node_i, node_j)
                if dist == 0: continue
                weight = 1.0 / dist
                
                artery_interaction += weight * (node_j.v - node_i.v)
                if node_j.pulse_active: incoming_pulse += weight * 2.0
                vein_drainage += weight * (node_i.residue - node_j.residue)
            
            # 【改善①: 修正】弾性の指数関数暴走を防ぐため、元の初期値に対してのみ一時的にモディファイアを適用
            current_elasticity = node_i.elasticity * elasticity_modifier
            next_states[i_id] = node_i.step(artery_interaction, vein_drainage, incoming_pulse, noise_modifier, current_elasticity)
        return next_states

class FastGridAttractorObserver:
    def __init__(self, grid_size: float = 0.08):
        self.grid_size = grid_size
        self.visited_cells: Set[Tuple[int, ...]] = set()

    def evaluate_and_track(self, metrics: HexaMetrics) -> float:
        vec = metrics.to_list()
        grid_coord = tuple(int(math.floor(x / self.grid_size)) for x in vec)
        
        if grid_coord in self.visited_cells:
            novelty = 0.0
        else:
            self.visited_cells.add(grid_coord)
            novelty = 1.0 / (1.0 + math.log(len(self.visited_cells)))
        return novelty

    def get_attractor_volume(self) -> int:
        return len(self.visited_cells)

class StabilityEvaluator:
    def __init__(self, network: GeometryNetwork):
        self.network = network

    def difference_filter(self, val1: float, val2: float, dynamic_sd: float) -> Dict[str, Any]:
        diff = abs(val1 - val2)
        if dynamic_sd <= 1e-4: dynamic_sd = 0.05
        z_score = diff / dynamic_sd
        accepted = z_score > 1.5
        return {"diff": diff if accepted else 0.0, "accepted": accepted, "raw_diff": diff, "sd_used": dynamic_sd}

    def compute_statistics_score(self, history: List[Dict[str, Any]]) -> float:
        values = [state["v"] for step in history for state in step.values()]
        N = len(values)
        if N < 4: return 0.0
        mean = sum(values) / N
        variance = sum((x - mean)**2 for x in values) / N
        sd = math.sqrt(variance) if variance > 0 else 1e-5
        skewness = (sum(((x - mean)/sd)**3 for x in values) / N)
        kurtosis = (sum(((x - mean)/sd)**4 for x in values) / N) - 3.0
        
        bins = [0] * 5
        for x in values:
            idx = max(0, min(4, int((x + 2.5) / 1.0)))
            bins[idx] += 1
        entropy = 0.0
        for b in bins:
            p = b / N
            if p > 0: entropy -= p * math.log2(p)
        return max(0.0, min(1.0, (entropy / math.log2(5)) * (1.0 / (1.0 + abs(skewness) + abs(kurtosis)))))

    def compute_graph_geometry_score(self) -> float:
        nodes = list(self.network.nodes.values())
        keys = list(self.network.nodes.keys())
        N = len(nodes)
        if N <= 1: return 0.0
        
        adj = {u: set() for u in keys}
        dist_matrix = {u: {v: float('inf') for v in keys} for u in keys}
        for u in keys: dist_matrix[u][u] = 0.0
        
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                d = self.network._get_distance(self.network.nodes[k1], self.network.nodes[k2])
                if d < 2.2:
                    adj[k1].add(k2)
                    adj[k2].add(k1)
                dist_matrix[k1][k2] = d
                dist_matrix[k2][k1] = d

        avg_degree = sum(len(adj[u]) for u in keys) / N
        
        clustering_sum = 0.0
        for u in keys:
            neighbors = adj[u]
            k = len(neighbors)
            if k < 2: continue
            links = 0
            n_list = list(neighbors)
            for i, v1 in enumerate(n_list):
                for v2 in n_list[i+1:]:
                    if v2 in adj[v1]: links += 1
            clustering_sum += (2.0 * links) / (k * (k - 1))
        avg_clustering = clustering_sum / N

        hop_dist = {u: {v: float('inf') for v in keys} for u in keys}
        for u in keys:
            hop_dist[u][u] = 0
            for v in adj[u]: hop_dist[u][v] = 1

        for k in keys:
            for i in keys:
                for j in keys:
                    if hop_dist[i][k] + hop_dist[k][j] < hop_dist[i][j]:
                        hop_dist[i][j] = hop_dist[i][k] + hop_dist[k][j]

        closeness_sum = 0.0
        betweenness = {u: 0.0 for u in keys}
        
        for u in keys:
            sum_d = sum(hop_dist[u][v] for v in keys if hop_dist[u][v] != float('inf'))
            if sum_d > 0: closeness_sum += (N - 1) / sum_d
            
        for s in keys:
            for t in keys:
                if s == t: continue
                for v in keys:
                    if v == s or v == t: continue
                    if hop_dist[s][v] + hop_dist[v][t] == hop_dist[s][t] and hop_dist[s][t] != float('inf'):
                        betweenness[v] += 1.0
        avg_betweenness = sum(betweenness.values()) / (N * (N - 1) + 1e-5)
        avg_closeness = closeness_sum / N

        combined_geom = (avg_degree / 6.0 + avg_clustering + avg_closeness + avg_betweenness * 5) / 4.0
        return max(0.0, min(1.0, combined_geom))

    def compute_stability_score(self, history: List[Dict[str, Any]]) -> float:
        if not history: return 0.0
        brain_v_list = [step["Brain_HCE_ARK"]["v"] for step in history]
        N = len(brain_v_list)
        mean_v = sum(brain_v_list) / N
        var_v = sum((x - mean_v)**2 for x in brain_v_list) / N
        stability_v = 1.0 / (1.0 + var_v)
        
        total_residue = 0.0
        for step in history:
            for state in step.values(): total_residue += abs(state["r"])
        avg_res = total_residue / (N * len(self.network.nodes))
        stability_r = 1.0 / (1.0 + avg_res)
        
        return (stability_v * 0.6) + (stability_r * 0.4)

    def generate_metrics(self, history: List[Dict[str, Any]]) -> HexaMetrics:
        total_steps = len(history)
        total_nodes = len(self.network.nodes)
        sum_v, sum_r, fire_count = 0.0, 0.0, 0
        for step_data in history:
            for state in step_data.values():
                sum_v += abs(state["v"])
                sum_r += abs(state["r"])
                if state["fired"]: fire_count += 1

        return HexaMetrics(
            geometry_score=self.compute_graph_geometry_score(),
            statistics_score=self.compute_statistics_score(history),
            temporal_score=min(1.0, sum_r / (total_steps * total_nodes + 1e-5)),
            information_score=min(1.0, sum_v / (total_steps * total_nodes + 1e-5)),
            dynamical_score=min(1.0, fire_count / (total_steps * total_nodes * 0.2 + 1e-5)),
            stability_score=self.compute_stability_score(history)
        )


class HighDimensionalMultiverseObserver:
    """【数理厳密化多宇宙観測機】演出を排し、限界を認めた上で最善の測定を行うプラットフォーム"""
    def __init__(self, network: GeometryNetwork, evaluator: StabilityEvaluator, observer: FastGridAttractorObserver):
        self.network = network
        self.evaluator = evaluator
        self.observer = observer
        
        self.universe_noise = GeometryNetwork()
        self.universe_noise.generate_body_topology()
        self.universe_elastic = GeometryNetwork()
        self.universe_elastic.generate_body_topology()

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1))
        norm2 = math.sqrt(sum(b*b for a in vec2))
        if norm1 == 0 or norm2 == 0: return 1.0
        return dot / (norm1 * norm2)

    def _compute_kl_divergence(self, states1: Dict[str, Any], states2: Dict[str, Any]) -> float:
        """【改善②: 厳密化】『5ビン離散化経験分布におけるカルバック・ライブラー情報量』と再定義"""
        def get_dist(states):
            bins = [1e-5] * 5
            for s in states.values():
                idx = max(0, min(4, int((s["v"] + 2.5) / 1.0)))
                bins[idx] += 1.0
            total = sum(bins)
            return [b / total for b in bins]
        
        p = get_dist(states1)
        q = get_dist(states2)
        return sum(p[i] * math.log(p[i] / q[i]) for i in range(5))

    def execute_multiverse_analysis(self, total_steps: int = 35):
        print(f"=== [START MATHEMATICALLY RIGOROUS MULTIVERSE OBSERVATION: {total_steps} STEPS] ===")
        # 修正: 演出的な「宇宙間情報距離」を「経験分布KL(5Bins)」に変更。さらに局所リアプノフ指標を追加
        print(f"{'ST':<3}|{'通常脳':<6}|{'ノイズ宇宙':<10}|{'構造硬化宇宙':<10}|{'動的SD':<5}|{'宇宙Aコサイン':<7}|{'経験分布KL(5Bins)':<14}|{'局所リアプノフ'}")
        print("-" * 105)

        self.network.reset_network()
        self.universe_noise.reset_network()
        self.universe_elastic.reset_network()
        
        running_history = []
        brain_v_window = []
        metrics_trajectory: List[HexaMetrics] = []
        
        prev_l2_dist = None

        for step in range(1, total_steps + 1):
            normal_s = self.network.step(noise_modifier=1.0)
            noise_s = self.universe_noise.step(noise_modifier=1.15)
            elastic_s = self.universe_elastic.step(elasticity_modifier=1.3) # 修正: 内部で一時適用されるため暴走しない

            running_history.append(normal_s)

            n_brain_v = normal_s["Brain_HCE_ARK"]["v"]
            noise_brain_v = noise_s["Brain_HCE_ARK"]["v"]
            elastic_brain_v = elastic_s["Brain_HCE_ARK"]["v"]

            normal_vec = [s["v"] for s in normal_s.values()]
            noise_vec = [s["v"] for s in noise_s.values()]
            
            cos_sim = self._compute_cosine_similarity(normal_vec, noise_vec)
            kl_div = self._compute_kl_divergence(normal_s, noise_s)

            # 【改善③: 収縮・拡散の実測】簡易局所リアプノフ指数のインライン・サンプリング
            sq_diff = sum((a - b) ** 2 for a, b in zip(normal_vec, noise_vec))
            current_l2_dist = math.sqrt(sq_diff)
            
            local_lyapunov = "INITIALIZING"
            if prev_l2_dist is not None and prev_l2_dist > 1e-4 and current_l2_dist > 1e-4:
                # 1ステップ間での軌道距離の対数成長率（正なら拡散、負なら縮小）
                local_lyapunov_val = math.log(current_l2_dist / prev_l2_dist)
                local_lyapunov = f"{local_lyapunov_val:>+7.4f}"
            prev_l2_dist = current_l2_dist

            # 動的SD
            brain_v_window.append(n_brain_v)
            if len(brain_v_window) > 20: brain_v_window.pop(0)
            win_mean = sum(brain_v_window) / len(brain_v_window)
            win_var = sum((x - win_mean)**2 for x in brain_v_window) / len(brain_v_window)
            dynamic_sd = math.sqrt(win_var) if win_var > 1e-5 else 0.05

            step_metrics = self.evaluator.generate_metrics(running_history)
            metrics_trajectory.append(step_metrics)
            _ = self.observer.evaluate_and_track(step_metrics)

            print(f"{step:<3}|{n_brain_v:>6.2f}|{noise_brain_v:>10.2f}|{elastic_brain_v:>10.2f}|{dynamic_sd:>5.2f}|{cos_sim:>11.4f}|{kl_div:>16.4f}|{local_lyapunov}")

        final_m = self.evaluator.generate_metrics(running_history)
        print("-" * 105)
        print("\n=== [FINAL SYSTEM TRAJECTORY ANALYSIS REPORT] ===")
        print(f"■ 最終HexaMetrics状態点 : {[round(x, 3) for x in final_m.to_list()]}")
        print(f"■ アトラクタ占有総容積  : {self.observer.get_attractor_volume()} グリッドセル")
        
        # 修正: 「Strange Attractor」という断定的な名称を排し、ヒューリスティックな判定であることを明記
        v_start = metrics_trajectory[-5].to_list() if len(metrics_trajectory) > 5 else metrics_trajectory[0].to_list()
        v_end = final_m.to_list()
        trajectory_velocity = math.sqrt(sum((a-b)**2 for a, b in zip(v_start, v_end)))
        print(f"■ 軌跡収束速度(末期5step) : {trajectory_velocity:.5f}")
        print("■ [ヒューリスティックによる軌道分類 (現時点の暫定予測)]")
        if trajectory_velocity < 0.01:
            print("  ⇒ 状態ベクトルは【固定点(Fixed Point)様軌道】に近接しています。")
        elif trajectory_velocity < 0.08:
            print("  ⇒ 状態ベクトルは【周期軌道(Limit Cycle)様軌道】を周回している可能性があります。")
        else:
            print("  ⇒ 状態ベクトルは【高次元非周期軌道（カオス・アトラクタの可能性あり）】へ遷移中、または発散しています。")
            print("     (※厳密なストレンジアトラクタの証明には、長期間の最大リアプノフ指数の収束性の計算が必要です)")


if __name__ == "__main__":
    net = GeometryNetwork()
    net.generate_body_topology()
    eval = StabilityEvaluator(net)
    grid_obs = FastGridAttractorObserver(grid_size=0.08)
    
    multiverse_engine = HighDimensionalMultiverseObserver(net, eval, grid_obs)
    multiverse_engine.execute_multiverse_analysis(total_steps=30)