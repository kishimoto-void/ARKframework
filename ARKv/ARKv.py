import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple

# =====================================================================
# DATA LAYER: TYPE DEFINITIONS & HISTORY
# =====================================================================
@dataclass(frozen=True)
class ForceVector:
    """[改善③] 責任：物理空間における力ベクトルの明確な型定義（可読性向上）"""
    fx: float = 0.0
    fy: float = 0.0

    def __add__(self, other: 'ForceVector') -> 'ForceVector':
        return ForceVector(self.fx + other.fx, self.fy + other.fy)

@dataclass
class MetricResult:
    """[改善④] 責任：固定次元を排し、Hexa/Octa/Dodecaへ動的に伸縮する観測結果コンテナ"""
    scores: Dict[str, float] = field(default_factory=dict)

    def to_list(self) -> List[float]:
        return list(self.scores.values())

    def get(self, key: str, default: float = 0.0) -> float:
        return self.scores.get(key, default)

@dataclass(frozen=True)
class NodeDefinition:
    """ 静的定義：変化しない固有プロパティ（セーブ／ロードのキー）"""
    node_id: str
    role_type: str
    seed: int
    base_elasticity: float
    pulse_threshold: float

@dataclass
class NodeState:
    """ 動的状態：時間を発展させるための純結な数値パラメータ"""
    orig_x: float
    orig_y: float
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    a: float = 0.5
    residue: float = 0.0
    phase: float = 0.0
    pulse_active: bool = False
    artery_buffer: float = 0.0

    def clone(self) -> 'NodeState':
        return NodeState(
            orig_x=self.orig_x, orig_y=self.orig_y, x=self.x, y=self.y,
            v=self.v, a=self.a, residue=self.residue, phase=self.phase,
            pulse_active=self.pulse_active, artery_buffer=self.artery_buffer
        )

class HistoryBuffer:
    """ 責任：時系列データの非破壊カプセル化データベース"""
    def __init__(self):
        self.records: List[Dict[str, NodeState]] = []

    def record_snapshot(self, nodes: Dict[str, Any]):
        self.records.append({node_id: node.state.clone() for node_id, node in nodes.items()})

    @property
    def total_steps(self) -> int:
        return len(self.records)

    def get_potentials(self, step_idx: int) -> List[float]:
        if step_idx >= len(self.records): return []
        return [state.v for state in self.records[step_idx].values()]

    def get_all_potentials_flat(self) -> List[float]:
        return [state.v for snap in self.records for state in snap.values()]

    def get_node_timeline_v(self, node_id: str) -> List[float]:
        return [snap[node_id].v for snap in self.records if node_id in snap]


# =====================================================================
# LAYER 1: CORE (Pure Structural Container)
# =====================================================================
class Node:
    def __init__(self, definition: NodeDefinition, state: NodeState):
        self.definition = definition
        self.state = state
        self.rng = random.Random(self.definition.seed)
        self.pulse_speed = self.rng.uniform(0.05, 0.15)
        self.state.phase = self.rng.uniform(0, math.pi * 2)

    def reset(self):
        self.state.x = self.state.orig_x
        self.state.y = self.state.orig_y
        self.state.v = 0.0
        self.state.a = 0.5
        self.state.residue = 0.0
        self.state.pulse_active = False
        self.state.artery_buffer = 0.0
        self.rng = random.Random(self.definition.seed)
        self.pulse_speed = self.rng.uniform(0.05, 0.15)
        self.state.phase = self.rng.uniform(0, math.pi * 2)

class GeometryNetwork:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node_id: str, role_type: str, x: float, y: float, seed: int):
        if role_type == "HCE_ARK":
            elasticity, threshold = 0.6, 2.2
        elif role_type == "ARKv":
            elasticity, threshold = 0.2, 1.4
        else:
            elasticity, threshold = 0.05, 2.5
            
        defn = NodeDefinition(node_id, role_type, seed, elasticity, threshold)
        state = NodeState(orig_x=x, orig_y=y, x=x, y=y)
        self.nodes[node_id] = Node(definition=defn, state=state)

    def build_body_topology(self):
        self.nodes.clear()
        self.add_node("Brain_HCE_ARK", "HCE_ARK", 0.0, 0.0, seed=777)
        for i in range(6):
            angle = i * (math.pi / 3)
            self.add_node(f"ether_Field_{i}", "ether", 1.0 * math.cos(angle), 1.0 * math.sin(angle), seed=500+i)
        limbs = {0: "Right_Hand_ARKv", 1: "Left_Hand_ARKv", 2: "Left_Foot_ARKv", 3: "Right_Foot_ARKv"}
        for idx, name in limbs.items():
            angle = idx * (math.pi / 3)
            self.add_node(name, "ARKv", 2.0 * math.cos(angle), 2.0 * math.sin(angle), seed=100+idx)

    def reset(self):
        for node in self.nodes.values():
            node.reset()

    def get_distance(self, id1: str, id2: str) -> float:
        n1, n2 = self.nodes[id1].state, self.nodes[id2].state
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)


# =====================================================================
# LAYER 2: PIPELINE DYNAMICS SOLVERS (Context Relaying)
# =====================================================================
class BaseSolver(ABC):
    """[改善⑤] 共通ソルバーインターフェース。コンテキストを介してデータを疏結合にリレーする"""
    @abstractmethod
    def solve(self, network: GeometryNetwork, context: Dict[str, Any]): pass

class RepulsionForce:
    def calculate(self, target_node: Node, other_node: Node, dist: float) -> ForceVector:
        dx = target_node.state.x - other_node.state.x
        repulsion = (target_node.state.artery_buffer + other_node.state.artery_buffer) * 0.02 / (dist ** 2)
        return ForceVector(fx=(dx / dist) * repulsion)

class AttractionForce:
    def calculate(self, target_node: Node, other_node: Node, dist: float) -> ForceVector:
        dx, dy = target_node.state.x - other_node.state.x, target_node.state.y - other_node.state.y
        attraction = (abs(target_node.state.residue) * abs(other_node.state.residue)) * 0.01 * dist
        return ForceVector(fx=-(dx / dist) * attraction, fy=-(dy / dist) * attraction)

class HomeostasisForce:
    """[改善①] 不要なABC継承を削除し、純結な具象力場コンポーネント化"""
    def calculate_homeostasis(self, node: Node) -> ForceVector:
        dist_origin = math.sqrt(node.state.x**2 + node.state.y**2)
        if dist_origin <= 0: return ForceVector()
        orig_d = math.sqrt(node.state.orig_x**2 + node.state.orig_y**2)
        fx = -(node.state.x / dist_origin) * 0.05 * (dist_origin - orig_d)
        fy = -(node.state.y / dist_origin) * 0.05 * (dist_origin - orig_d)
        return ForceVector(fx=fx, fy=fy)

class TopologyEvolutionSolver(BaseSolver):
    """ 責任：微細力場コンポーネントを合成し、座標変形を解決する"""
    def __init__(self):
        self.repulsion = RepulsionForce()
        self.attraction = AttractionForce()
        self.homeostasis = HomeostasisForce()

    def solve(self, network: GeometryNetwork, context: Dict[str, Any]):
        keys = list(network.nodes.keys())
        for u in keys:
            total_f = ForceVector()
            node_u = network.nodes[u]
            for v in keys:
                if u == v: continue
                dist = network.get_distance(u, v)
                if dist < 0.1: dist = 0.1
                total_f += self.repulsion.calculate(node_u, network.nodes[v], dist)
                total_f += self.attraction.calculate(node_u, network.nodes[v], dist)
            
            total_f += self.homeostasis.calculate_homeostasis(node_u)
            node_u.state.x += max(-0.2, min(0.2, total_f.fx))
            node_u.state.y += max(-0.2, min(0.2, total_f.fy))

class NetworkInteractionSolver(BaseSolver):
    """ 責任：トポロジーから入力を集計し、下流のためにcontextへエクスポートする"""
    def solve(self, network: GeometryNetwork, context: Dict[str, Any]):
        inputs = {}
        keys = list(network.nodes.keys())
        for u in keys:
            node_u = network.nodes[u]
            artery_in, vein_out, pulse_in = 0.0, 0.0, 0.0
            for v in keys:
                if u == v: continue
                dist = network.get_distance(u, v)
                if dist == 0: continue
                w = 1.0 / dist
                artery_in += w * (network.nodes[v].state.v - node_u.state.v)
                if network.nodes[v].state.pulse_active: pulse_in += w * 2.0
                vein_out += w * (node_u.state.residue - network.nodes[v].state.residue)
            inputs[u] = (artery_in, vein_out, pulse_in)
        context["network_inputs"] = inputs

class PulseGenerator:
    def generate_pulsation(self, node: Node) -> float:
        node.state.phase += node.pulse_speed
        return math.sin(node.state.phase) * 0.1

class NoiseGenerator:
    def generate_noise(self, node: Node, noise_mod: float) -> Tuple[float, float]:
        nv = node.rng.uniform(-1, 1) * 0.2 * noise_mod
        na = node.rng.uniform(-1, 1) * 0.1 * noise_mod
        return nv, na

class ElasticIntegrator:
    def integrate_fluid(self, node: Node, art_in: float, pls_in: float, elasticity_mod: float) -> float:
        node.state.artery_buffer += (0.15 * art_in + pls_in)
        current_elasticity = node.definition.base_elasticity * elasticity_mod
        smooth_in = node.state.artery_buffer * current_elasticity
        node.state.artery_buffer -= smooth_in
        return smooth_in

class ThresholdDetector:
    def detect_and_clamp(self, node: Node):
        node.state.pulse_active = node.state.v > node.definition.pulse_threshold and not node.state.pulse_active
        node.state.v = max(-2.5, min(2.5, node.state.v))
        node.state.a = max(0.0, min(1.0, node.state.a))

class PotentialPulseSolver(BaseSolver):
    """ 責任：contextから入力を受け取り、アトム要素を合成して電位を発展させる"""
    def __init__(self):
        self.pulse_gen = PulseGenerator()
        self.noise_gen = NoiseGenerator()
        self.elastic_int = ElasticIntegrator()
        self.detector = ThresholdDetector()

    def solve(self, network: GeometryNetwork, context: Dict[str, Any]):
        inputs = context.get("network_inputs", {})
        noise_mod = context.get("noise_mod", 1.0)
        elasticity_mod = context.get("elasticity_mod", 1.0)

        for u, node in network.nodes.items():
            art_in, _, pls_in = inputs.get(u, (0.0, 0.0, 0.0))
            pulsation = self.pulse_gen.generate_pulsation(node)
            nv, na = self.noise_gen.generate_noise(node, noise_mod)
            smooth_in = self.elastic_int.integrate_fluid(node, art_in, pls_in, elasticity_mod)
            
            node.state.v += nv + smooth_in + 0.1 * math.tanh(node.state.residue) + pulsation
            node.state.a += na
            self.detector.detect_and_clamp(node)

class ResidueSolver(BaseSolver):
    """ 責任：contextから入力を受け取り、排出代謝残槃を発展させる"""
    def solve(self, network: GeometryNetwork, context: Dict[str, Any]):
        inputs = context.get("network_inputs", {})
        for u, node in network.nodes.items():
            _, vein_out, _ = inputs.get(u, (0.0, 0.0, 0.0))
            node.state.residue = node.state.residue * 0.9 + node.state.v * 0.05 - vein_out * 0.1

class ModularDynamics:
    """[改善②] 不要なABCを削除。ソルバーを完全にパイプライン配列化して順序実行を可能にしたコンポーネント"""
    def __init__(self):
        # [改善⑤] ソルバーをリストに集約。これにより将来の順序入れ替えや追加が自由自在になる
        self.pipeline: List[BaseSolver] = [
            TopologyEvolutionSolver(),
            NetworkInteractionSolver(),
            PotentialPulseSolver(),
            ResidueSolver()
        ]

    def evolve_step(self, network: GeometryNetwork, noise_mod: float, elasticity_mod: float):
        # 共有コンテキストの初期化
        context: Dict[str, Any] = {
            "noise_mod": noise_mod,
            "elasticity_mod": elasticity_mod
        }
        # パイプラインストリームを一方向にリレー実行
        for solver in self.pipeline:
            solver.solve(network, context)


# =====================================================================
# LAYER 3: MICRO-OBSERVERS (Pure ReadOnly)
# =====================================================================
class BaseObserver(ABC):
    @abstractmethod
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float: pass

class GeometryObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        keys = list(network.nodes.keys())
        N = len(keys)
        if N <= 1: return 0.0
        adj = {u: set() for u in keys}
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                if network.get_distance(k1, k2) < 2.2:
                    adj[k1].add(k2); adj[k2].add(k1)
        avg_degree = sum(len(adj[u]) for u in keys) / N
        c_sum = 0.0
        for u in keys:
            nb = list(adj[u])
            k = len(nb)
            if k < 2: continue
            links = sum(1 for gi, v1 in enumerate(nb) for v2 in nb[gi+1:] if v2 in adj[v1])
            c_sum += (2.0 * links) / (k * (k - 1))
        avg_clustering = c_sum / N
        hop = {u: {v: float('inf') for v in keys} for u in keys}
        for u in keys:
            hop[u][u] = 0
            for v in adj[u]: hop[u][v] = 1
        for k in keys:
            for i in keys:
                for j in keys:
                    if hop[i][k] + hop[k][j] < hop[i][j]: hop[i][j] = hop[i][k] + hop[k][j]
        closeness_sum = 0.0
        betweenness = {u: 0.0 for u in keys}
        for u in keys:
            sd = sum(hop[u][v] for v in keys if hop[u][v] != float('inf'))
            if sd > 0: closeness_sum += (N - 1) / sd
        for s in keys:
            for t in keys:
                if s == t: continue
                for v in keys:
                    if v == s or v == t: continue
                    if hop[s][v] + hop[v][t] == hop[s][t] and hop[s][t] != float('inf'): betweenness[v] += 1.0
        avg_betweenness = sum(betweenness.values()) / (N * (N - 1) + 1e-5)
        avg_closeness = closeness_sum / N
        return max(0.0, min(1.0, (avg_degree / 6.0 + avg_clustering + avg_closeness + avg_betweenness * 5) / 4.0))

class StatisticsObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        flat = history.get_all_potentials_flat()
        N = len(flat)
        if N < 4: return 0.0
        mean = sum(flat) / N
        var = sum((x - mean)**2 for x in flat) / N
        sd = math.sqrt(var) if var > 0 else 1e-5
        skew = sum(((x - mean)/sd)**3 for x in flat) / N
        kurt = (sum(((x - mean)/sd)**4 for x in flat) / N) - 3.0
        bins = [0] * 5
        for x in flat: bins[max(0, min(4, int((x + 2.5) / 1.0)))] += 1
        entropy = -sum((b/N) * math.log2(b/N) for b in bins if b > 0)
        return max(0.0, min(1.0, (entropy / math.log2(5)) * (1.0 / (1.0 + abs(skew) + abs(kurt)))))

class TemporalObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        return min(1.0, sum(abs(n.state.residue) for n in network.nodes.values()) / (len(network.nodes) + 1e-5))

class InformationObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        return min(1.0, sum(abs(n.state.v) for n in network.nodes.values()) / (len(network.nodes) + 1e-5))

class DynamicalObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        return min(1.0, sum(1 for n in network.nodes.values() if n.state.pulse_active) / (len(network.nodes) * 0.2 + 1e-5))

class StabilityObserver(BaseObserver):
    def observe(self, network: GeometryNetwork, history: HistoryBuffer) -> float:
        brain_timeline = history.get_node_timeline_v("Brain_HCE_ARK")
        if not brain_timeline: return 0.0
        mean_v = sum(brain_timeline) / len(brain_timeline)
        var_v = sum((x - mean_v)**2 for x in brain_timeline) / len(brain_timeline)
        return 1.0 / (1.0 + var_v)


# =====================================================================
# LAYER 3.5: PLUG-IN METRICS EVALUATOR & ATTRACTOR TRACKER
# =====================================================================
class MetricsEvaluator:
    """ 責任：動的に登録されたプラグインObserverから値を集める純結なレジストリビルダー"""
    def __init__(self):
        self._observers: Dict[str, BaseObserver] = {}

    def register_observer(self, name: str, observer: BaseObserver):
        self._observers[name] = observer

    def evaluate(self, network: GeometryNetwork, history: HistoryBuffer) -> MetricResult:
        # [改善④] HexaMetrics固定をやめ、登録された全Observerから自動マッピングする動的伸縮コンテナを生成
        scores = {name: obs.observe(network, history) for name, obs in self._observers.items()}
        return MetricResult(scores=scores)

class FastGridAttractorTracker:
    def __init__(self, grid_size: float = 0.08):
        self.grid_size = grid_size
        self.visited_cells: Set[Tuple[int, ...]] = set()

    def track(self, metrics: MetricResult) -> int:
        grid_coord = tuple(int(math.floor(x / self.grid_size)) for x in metrics.to_list())
        self.visited_cells.add(grid_coord)
        return len(self.visited_cells)


# =====================================================================
# LAYER 4: EXPERIMENT RESULTS & PLUG-IN REPORTERS
# =====================================================================
@dataclass
class MultiverseStepData:
    step: int
    v_normal_brain: float
    v_noise_brain: float
    v_elastic_brain: float
    kl_divergence: float
    lyapunov: str
    attractor_vol: int

@dataclass
class ExperimentResult:
    """ 責任：実験結果の生構造化データコンテナ"""
    steps_log: List[MultiverseStepData] = field(default_factory=list)
    final_metrics: MetricResult = field(default_factory=MetricResult)
    total_volume: int = 0

class BaseReporter(ABC):
    @abstractmethod
    def report(self, result: ExperimentResult): pass

class ConsoleReporter(BaseReporter):
    """ 責任：実験結果を生データから読み込んでテキストとして細かに出力成形する"""
    def report(self, result: ExperimentResult):
        print(f"=== [ARKv RESEARCH OS: COMPOSABLE PIPELINE RUN] ===")
        print(f"{'ST':<3}|{'Normal_V':<8}|{'Noise_V':<8}|{'Elastic_V':<9}|{'\u7d4c験分布KL(5Bins)':<16}|{'\u5c40所リアプノフ\u6307\u6570'}")
        print("-" * 72)
        for data in result.steps_log:
            print(f"{data.step:<3}|{data.v_normal_brain:>8.3f}|{data.v_noise_brain:>8.3f}|{data.v_elastic_brain:>9.3f}|{data.kl_divergence:>16.5f}|{data.lyapunov}")
        print("-" * 72)
        print("\u25a0 \u6700終動的状態ベクトルマップ :")
        for k, v in result.final_metrics.scores.items():
            print(f"  - {k:<12} : {v:.4f}")
        print(f"\u25a0 \u8ecc跡の占有アトラクタ\u7dcf容\u7a4d : {result.total_volume} \u30bbル")


# =====================================================================
# LAYER 5: RUNTIME EXECUTION ENGINE & PROTOCOL
# =====================================================================
class SimulationEngine:
    def __init__(self, network: GeometryNetwork, dynamics: ModularDynamics):
        self.network = network
        self.dynamics = dynamics

    def step(self, noise_mod: float = 1.0, elasticity_mod: float = 1.0):
        self.dynamics.evolve_step(self.network, noise_mod, elasticity_mod)

class MultiverseExperiment:
    """ 責任：並行世界を発展させ、純結な生データオブジェクトを生成する最上位環境"""
    def __init__(self):
        self.net_normal, self.net_noise, self.net_elastic = GeometryNetwork(), GeometryNetwork(), GeometryNetwork()
        self.net_normal.build_body_topology()
        self.net_noise.build_body_topology()
        self.net_elastic.build_body_topology()

        self.dynamics = ModularDynamics()
        self.engine_normal = SimulationEngine(self.net_normal, self.dynamics)
        self.engine_noise = SimulationEngine(self.net_noise, self.dynamics)
        self.engine_elastic = SimulationEngine(self.net_elastic, self.dynamics)

        # [改善④] レジストリ型Evaluatorへプラグインを登録
        self.evaluator = MetricsEvaluator()
        self.evaluator.register_observer("geometry", GeometryObserver())
        self.evaluator.register_observer("statistics", StatisticsObserver())
        self.evaluator.register_observer("temporal", TemporalObserver())
        self.evaluator.register_observer("information", InformationObserver())
        self.evaluator.register_observer("dynamical", DynamicalObserver())
        self.evaluator.register_observer("stability", StabilityObserver())

        self.tracker = FastGridAttractorTracker()
        self.history_normal = HistoryBuffer()

    def _kl_divergence_离散5Bins(self, net1: GeometryNetwork, net2: GeometryNetwork) -> float:
        def get_p(net):
            b = [1e-5] * 5
            for n in net.nodes.values(): b[max(0, min(4, int((n.state.v + 2.5) / 1.0)))] += 1.0
            tot = sum(b)
            return [x / tot for x in b]
        p, q = get_p(net1), get_p(net2)
        return sum(p[i] * math.log(p[i] / q[i]) for i in range(5))

    def run(self, steps: int = 30) -> ExperimentResult:
        self.net_normal.reset(); self.net_noise.reset(); self.net_elastic.reset()
        result = ExperimentResult()
        prev_l2 = None

        for step in range(1, steps + 1):
            self.engine_normal.step(noise_mod=1.0)
            self.engine_noise.step(noise_mod=1.15)
            self.engine_elastic.step(elasticity_mod=1.3)

            self.history_normal.record_snapshot(self.net_normal.nodes)

            v_norm_brain = self.net_normal.nodes['Brain_HCE_ARK'].state.v
            v_noise_brain = self.net_noise.nodes['Brain_HCE_ARK'].state.v
            v_elastic_brain = self.net_elastic.nodes['Brain_HCE_ARK'].state.v

            kl_val = self._kl_divergence_离散5Bins(self.net_normal, self.net_noise)
            
            v_normal_all = [n.state.v for n in self.net_normal.nodes.values()]
            v_noise_all = [n.state.v for n in self.net_noise.nodes.values()]
            l2_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_normal_all, v_noise_all)))
            lyapunov_str = "INIT"
            if prev_l2 is not None and prev_l2 > 1e-4 and l2_dist > 1e-4:
                lyapunov_str = f"{math.log(l2_dist / prev_l2):>+7.4f}"
            prev_l2 = l2_dist

            metrics = self.evaluator.evaluate(self.net_normal, self.history_normal)
            vol = self.tracker.track(metrics)

            result.steps_log.append(MultiverseStepData(
                step=step, v_normal_brain=v_norm_brain, v_noise_brain=v_noise_brain,
                v_elastic_brain=v_elastic_brain, kl_divergence=kl_val, lyapunov=lyapunov_str, attractor_vol=vol
            ))

        result.final_metrics = self.evaluator.evaluate(self.net_normal, self.history_normal)
        result.total_volume = len(self.tracker.visited_cells)
        return result


if __name__ == "__main__":
    # 1. 独立した実験手順の実行（一方向データフローの始点）
    experiment = MultiverseExperiment()
    raw_result = experiment.run(steps=30)

    # 2. 独立した出力プラグイン（Reporter）の結合
    reporter = ConsoleReporter()
    reporter.report(raw_result)
