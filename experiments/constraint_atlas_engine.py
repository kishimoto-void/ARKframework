from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import numpy as np
from typing import Dict, List, Any, Optional

# =============================================================================
# 1. 階層トランスフォーム & 匿名プロバイダー
# =============================================================================

class BaseFeatureTransform(ABC):
    """Base Feature Transform"""
    @abstractmethod
    def transform(self, raw_data: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

class DifferentialTransform(BaseFeatureTransform):
    """Differential Transform"""
    def transform(self, raw_data: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        transformed = raw_data.copy()
        if not history:
            transformed["acceleration"] = 0.0
            return transformed
        prev = history[-1]
        transformed["acceleration"] = raw_data.get("speed", 0.0) - prev.get("speed", 0.0)
        return transformed

class FeatureProvider:
    """Feature Provider"""
    def __init__(self, features: Dict[str, Any], raw_history: List[np.ndarray] = None):
        self._features = features
        self._raw_history = raw_history or []

    def get_scalar(self, key: str, default: float = 0.0) -> float:
        val = self._features.get(key, default)
        return float(val) if isinstance(val, (int, float, np.number)) else default

    def get_vector(self, key: str) -> np.ndarray:
        val = self._features.get(key)
        if isinstance(val, np.ndarray): return val
        if isinstance(val, list): return np.array(val, dtype=float)
        return np.zeros(1)

    @property
    def raw_history(self) -> List[np.ndarray]:
        return self._raw_history

# =============================================================================
# 2. 観測器抽象基盤（不変量階層の新規独立）
# =============================================================================

class BaseObserver(ABC):
    """Base Microscopic Observer"""
    @abstractmethod
    def observe(self, provider: FeatureProvider) -> float:
        pass

class BaseMetaObserver(ABC):
    """Base Macroscopic Axis-Internal Meta Observer"""
    @abstractmethod
    def analyze(self, current_outputs: Dict[str, float], history: List[Dict[str, float]]) -> float:
        pass

class BaseCrossAxisMetaObserver(ABC):
    """Base Macroscopic Cross-Axis Meta Observer"""
    @abstractmethod
    def analyze(self, current_axes: Dict[str, Any]) -> float:
        pass

class BaseTrajectoryObserver(ABC):
    """Base Macroscopic Trajectory Observer"""
    @abstractmethod
    def observe_trajectory(self, timeline: List[Any]) -> Any:
        pass

class BaseInvariantObserver(ABC):
    """Base Infinite-Time Asymptotic Invariant Observer"""
    @abstractmethod
    def compute_invariant(self, timeline: List[Any]) -> float:
        pass

# =============================================================================
# 3. 純粋データ構造群
# =============================================================================

@dataclass
class ConstraintAxis:
    """Constraint Axis Data Container"""
    meta_metrics: Dict[str, float] = field(default_factory=dict)
    observers: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConstraintProfile:
    """Constraint Profile Data Container"""
    dynamics: ConstraintAxis
    geometry: ConstraintAxis
    temporal: ConstraintAxis
    structure: ConstraintAxis
    vector: ConstraintAxis
    entropy: ConstraintAxis

@dataclass
class ConstraintAtlas:
    """Constraint Atlas Container"""
    profile: ConstraintProfile
    cross_axis_metrics: Dict[str, float] = field(default_factory=dict)
    trajectory_metrics: Dict[str, Any] = field(default_factory=dict)
    invariant_metrics: Dict[str, float] = field(default_factory=dict)
    timeline_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =============================================================================
# 4. 時系列・漸近不変量コンポーネント
# =============================================================================

class ConstraintTrajectory:
    """Active Constraint Trajectory Container"""
    def __init__(self, max_memory: int = 200):
        self.timeline: List[ConstraintProfile] = []
        self.max_memory = max_memory
        self.observers: Dict[str, BaseTrajectoryObserver] = {}

    def register_observer(self, name: str, observer: BaseTrajectoryObserver):
        self.observers[name] = observer

    def append(self, profile: ConstraintProfile) -> Dict[str, Any]:
        self.timeline.append(profile)
        if len(self.timeline) > self.max_memory:
            self.timeline.pop(0)
        return {name: obs.observe_trajectory(self.timeline) for name, obs in self.observers.items()}

class InvariantSpace:
    """Infinite-Time Asymptotic Invariant Container"""
    def __init__(self):
        self.observers: Dict[str, BaseInvariantObserver] = {}

    def register_invariant_observer(self, name: str, observer: BaseInvariantObserver):
        self.observers[name] = observer

    def evaluate(self, timeline: List[ConstraintProfile]) -> Dict[str, float]:
        return {name: obs.compute_invariant(timeline) for name, obs in self.observers.items()}

# =============================================================================
# 5. 特化型プラグイン実装（数理的厳密化・完全中立ラベル化）
# =============================================================================

class StrengthObserver(BaseMetaObserver):
    """L2 Norm Strength Meta Observer"""
    def analyze(self, current_outputs: Dict[str, float], history: List[Dict[str, float]]) -> float:
        values = np.array(list(current_outputs.values()))
        if len(values) == 0: return 0.0
        l2_norm = float(np.linalg.norm(values))
        if len(history) > 1:
            hist_norms = [np.linalg.norm(list(h.values())) for h in history]
            max_hist = max(hist_norms) if max(hist_norms) > 0 else 1.0
            return float(np.clip(l2_norm / max_hist, 0.0, 1.0))
        return float(1.0 / (1.0 + np.exp(-l2_norm)))

class CoherenceObserver(BaseMetaObserver):
    """PCA Eigenvalue Coherence Meta Observer"""
    def analyze(self, current_outputs: Dict[str, float], history: List[Dict[str, float]]) -> float:
        if len(history) < 3: return 1.0
        data = np.array([list(h.values()) for h in history])
        if np.allclose(data, data[0, :] ): return 1.0
        try:
            std_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            cov_matrix = np.cov(std_data, rowvar=False)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            return float(np.real(np.max(eigenvalues)) / np.sum(np.real(eigenvalues)))
        except Exception:
            return 0.5

class CrossAxisPhaseCoupling(BaseCrossAxisMetaObserver):
    """Kuramoto-like Cross Axis Phase Coupling Meta Observer"""
    def analyze(self, current_axes: Dict[str, ConstraintAxis]) -> float:
        strengths = np.array([ax.meta_metrics.get("strength", 0.0) for ax in current_axes.values()])
        if len(strengths) < 2: return 0.0
        phases = np.angle(np.exp(1j * np.pi * strengths))
        return float(np.abs(np.mean(np.exp(1j * (phases - phases.mean())))))

class CrossAxisInformationResonance(BaseCrossAxisMetaObserver):
    """Information Dynamics Cross Axis Resonance Meta Observer"""
    def analyze(self, current_axes: Dict[str, ConstraintAxis]) -> float:
        entropy_axis = current_axes.get("entropy")
        ent = entropy_axis.observers.get("shannon_entropy", 0.0) if entropy_axis else 0.0
        strengths = [ax.meta_metrics.get("strength", 0.0) for ax in current_axes.values()]
        coherences = [ax.meta_metrics.get("coherence", 0.0) for ax in current_axes.values()]
        m_str = np.mean(strengths) if strengths else 0.0
        m_coh = np.mean(coherences) if coherences else 0.0
        return float(1.0 - np.exp(-m_str * m_coh / (ent + 1e-8)))

class TrajectoryPeriodicityObserver(BaseTrajectoryObserver):
    """Trajectory Periodicity Observer"""
    def observe_trajectory(self, timeline: List[ConstraintProfile]) -> float:
        if len(timeline) < 8: return 0.0
        strengths = [p.dynamics.meta_metrics.get("strength", 0.0) for p in timeline]
        signal = np.array(strengths) - np.mean(strengths)
        if np.allclose(signal, 0): return 0.0
        fft_vals = np.abs(np.fft.rfft(signal))
        return float(np.max(fft_vals[1:]) / np.sum(fft_vals[1:])) if np.sum(fft_vals[1:]) > 0 else 0.0

class BifurcationDetector(BaseTrajectoryObserver):
    """Bifurcation Detector Trajectory Observer"""
    def observe_trajectory(self, timeline: List[ConstraintProfile]) -> float:
        if len(timeline) < 12: return 0.0
        coh = [p.dynamics.meta_metrics.get("coherence", 0.0) for p in timeline[-12:]]
        grad = np.gradient(coh)
        return float(np.max(np.abs(grad)))

class InformationPotentialProjection(BaseTrajectoryObserver):
    """Information Potential Projection (Formerly ValenceArousalObserver)"""
    def observe_trajectory(self, timeline: List[ConstraintProfile]) -> Dict[str, float]:
        ent = np.array([p.entropy.observers.get("shannon_entropy", 0.0) for p in timeline])
        strg = np.array([p.dynamics.meta_metrics.get("strength", 0.0) for p in timeline])
        potential = float(np.mean(1.0 - ent))
        kinetic = float(np.mean(strg))
        return {"potential_gradient": potential, "kinetic_intensity": kinetic, "density": potential * kinetic}

# --- 独立構造：無限時間漸近不変量観測器群 (Invariant Observers) ---
class LyapunovExponentObserver(BaseInvariantObserver):
    """Lyapunov Exponent Invariant Observer"""
    def compute_invariant(self, timeline: List[ConstraintProfile]) -> float:
        if len(timeline) < 20: return 0.0
        strengths = np.array([p.dynamics.meta_metrics.get("strength", 0.0) for p in timeline])
        diffs = np.diff(strengths)
        return float(np.mean(np.log(np.abs(diffs) + 1e-12)))

class FractalDimensionObserver(BaseInvariantObserver):
    """Fractal Dimension Invariant Observer"""
    def compute_invariant(self, timeline: List[ConstraintProfile]) -> float:
        if len(timeline) < 15: return 1.0
        snapshots = []
        for p in timeline:
            snapshots.extend(list(p.dynamics.observers.values()))
        ts = np.array(snapshots).flatten()
        scales = [2, 4, 8, 16]
        counts = [np.sum(np.abs(np.diff(ts[::s])) > 1e-6) for s in scales]
        log_s = np.log(scales)
        log_c = np.log(counts)
        return float(np.polyfit(log_s, log_c, 1)[0]) if len(log_s) > 1 else 1.5

class AttractorStabilityObserver(BaseInvariantObserver):
    """Attractor Stability Invariant Observer"""
    def compute_invariant(self, timeline: List[ConstraintProfile]) -> float:
        if len(timeline) < 10: return 0.5
        recent_matrices = []
        for p in timeline[-10:]:
            recent_matrices.append(list(p.dynamics.observers.values()))
        recent = np.array(recent_matrices)
        dist = np.linalg.norm(recent[-1] - recent.mean(axis=0))
        return float(np.exp(-dist))

# --- Microscopic Observers ---
class PureMagnitudeObserver(BaseObserver):
    def observe(self, provider: FeatureProvider) -> float: return provider.get_scalar("speed")

class PureForceObserver(BaseObserver):
    def observe(self, provider: FeatureProvider) -> float: return provider.get_scalar("acceleration")

class PureCurvatureObserver(BaseObserver):
    def observe(self, provider: FeatureProvider) -> float: return provider.get_scalar("curvature")

class PureShannonEntropyObserver(BaseObserver):
    def observe(self, provider: FeatureProvider) -> float:
        vec = provider.get_vector("state_vector")
        if np.all(vec == 0) or len(vec) <= 1: return 1.0
        e_x = np.exp(vec - np.max(vec))
        p = e_x / e_x.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(vec)))

# =============================================================================
# 6. レジストリ & コンテナ
# =============================================================================

class ObserverRegistry:
    """Observer Registry"""
    def __init__(self):
        self._observers: Dict[str, BaseObserver] = {}

    def register(self, name: str, observer: BaseObserver):
        self._observers[name] = observer

    def get_all(self) -> Dict[str, BaseObserver]:
        return self._observers

class ConstraintAxisSpace:
    """Constraint Axis Space"""
    def __init__(self):
        self.registry = ObserverRegistry()
        self.meta_observers: Dict[str, BaseMetaObserver] = {}
        self.history: List[Dict[str, float]] = []
        self.max_history = 20

    def register_meta_observer(self, name: str, meta_observer: BaseMetaObserver):
        self.meta_observers[name] = meta_observer

    def evaluate(self, provider: FeatureProvider) -> ConstraintAxis:
        current_outputs = {name: obs.observe(provider) for name, obs in self.registry.get_all().items()}
        self.history.append(current_outputs)
        if len(self.history) > self.max_history: self.history.pop(0)
        calculated_meta = {m_name: m_obs.analyze(current_outputs, self.history) for m_name, m_obs in self.meta_observers.items()}
        return ConstraintAxis(meta_metrics=calculated_meta, observers=current_outputs)

# =============================================================================
# 7. 生成エンジンコア (ConstraintAtlasEngine)
# =============================================================================

class ConstraintAtlasEngine:
    """Constraint Atlas Generator Engine"""
    def __init__(self):
        self.transforms: List[BaseFeatureTransform] = []
        self.transform_history: List[Dict[str, Any]] = []
        self.spaces: Dict[str, ConstraintAxisSpace] = {
            "dynamics": ConstraintAxisSpace(), "geometry": ConstraintAxisSpace(),
            "temporal": ConstraintAxisSpace(), "structure": ConstraintAxisSpace(),
            "vector": ConstraintAxisSpace(), "entropy": ConstraintAxisSpace()
        }
        self.cross_meta_observers: Dict[str, BaseCrossAxisMetaObserver] = {}
        self.trajectory = ConstraintTrajectory()
        self.invariant_space = InvariantSpace()
        self.raw_history: List[np.ndarray] = []
        self._build_native_architecture()

    def _build_native_architecture(self):
        self.transforms.append(DifferentialTransform())
        
        for space in self.spaces.values():
            space.register_meta_observer("strength", StrengthObserver())
            space.register_meta_observer("coherence", CoherenceObserver())

        self.cross_meta_observers["phase_coupling"] = CrossAxisPhaseCoupling()
        self.cross_meta_observers["information_resonance"] = CrossAxisInformationResonance()

        self.trajectory.register_observer("periodicity", TrajectoryPeriodicityObserver())
        self.trajectory.register_observer("bifurcation", BifurcationDetector())
        self.trajectory.register_observer("potential_projection", InformationPotentialProjection())

        self.invariant_space.register_invariant_observer("lyapunov", LyapunovExponentObserver())
        self.invariant_space.register_invariant_observer("fractal_dimension", FractalDimensionObserver())
        self.invariant_space.register_invariant_observer("attractor_stability", AttractorStabilityObserver())

        self.spaces["dynamics"].registry.register("magnitude", PureMagnitudeObserver())
        self.spaces["dynamics"].registry.register("force", PureForceObserver())
        self.spaces["geometry"].registry.register("curvature", PureCurvatureObserver())
        self.spaces["entropy"].registry.register("shannon_entropy", PureShannonEntropyObserver())

    def generate_atlas(self, raw_reality_data: Dict[str, Any]) -> ConstraintAtlas:
        """Reality -> Feature Transform -> Feature Provider -> Observer -> Axis Meta -> Cross Axis Meta -> Trajectory -> Invariant Observer -> Constraint Atlas"""
        transformed_features = raw_reality_data.copy()
        for transform_plugin in self.transforms:
            transformed_features = transform_plugin.transform(transformed_features, self.transform_history)
        
        self.transform_history.append(transformed_features)
        if len(self.transform_history) > 50: self.transform_history.pop(0)

        if "state_vector" in transformed_features:
            self.raw_history.append(transformed_features["state_vector"])
            if len(self.raw_history) > 50: self.raw_history.pop(0)
        provider = FeatureProvider(transformed_features, self.raw_history)

        current_profile = ConstraintProfile(
            dynamics=self.spaces["dynamics"].evaluate(provider),
            geometry=self.spaces["geometry"].evaluate(provider),
            temporal=self.spaces["temporal"].evaluate(provider),
            structure=self.spaces["structure"].evaluate(provider),
            vector=self.spaces["vector"].evaluate(provider),
            entropy=self.spaces["entropy"].evaluate(provider)
        )

        axes_dict = {
            "dynamics": current_profile.dynamics, "geometry": current_profile.geometry,
            "temporal": current_profile.temporal, "structure": current_profile.structure,
            "vector": current_profile.vector, "entropy": current_profile.entropy
        }
        cross_metrics = {name: obs.analyze(axes_dict) for name, obs in self.cross_meta_observers.items()}

        trajectory_metrics = self.trajectory.append(current_profile)
        invariant_metrics = self.invariant_space.evaluate(self.trajectory.timeline)

        return ConstraintAtlas(
            profile=current_profile,
            cross_axis_metrics=cross_metrics,
            trajectory_metrics=trajectory_metrics,
            invariant_metrics=invariant_metrics,
            timeline_snapshots=[asdict(p) for p in self.trajectory.timeline[-5:]]
        )
