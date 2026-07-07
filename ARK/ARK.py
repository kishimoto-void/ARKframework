#!/usr/bin/env python3
"""
MetaVoid core (v2026-06-25) - user's provided implementation
This is the authoritative current version.
"""

from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import random
import json
from typing import Optional, List, Dict, Any

@dataclass(frozen=True)
class Experience:
    pattern: str
    direction: float
    effect: float

@dataclass(frozen=True)
class StateNode:
    pattern: str
    direction: float
    effect: float

@dataclass
class RecoveryEvent:
    event_id: str
    step: int
    formation: Dict[str, Any] = field(default_factory=dict)
    utilization: Dict[str, Any] = field(default_factory=dict)
    damage: Dict[str, Any] = field(default_factory=dict)
    residue_before: float = 0.0
    residue_after: float = 0.0
    residue_gradient: float = 0.0
    chosen_node_pattern: str = ""
    node_role: str = ""
    total_score: float = 0.0
    locality_score: float = 0.0
    residue_window_score: float = 0.0
    reachability_before: int = 0
    reachability_after: int = 0
    reachability_gain: float = 0.0
    redundancy_gain: float = 0.0
    pattern_bonus: float = 0.0

class MetaVoid:
    def __init__(self, metabolism_prob: float = 0.0062):
        self.state: Optional[StateNode] = None
        self.last_good_state: Optional[StateNode] = None
        self.abstractions: List[StateNode] = []
        self.graph: dict[StateNode, dict[StateNode, float]] = defaultdict(dict)
        self.checkpoints: List[StateNode] = []
        self.structural_reserve: dict[str, List] = defaultdict(list)
        self.residue: float = 0.0
        self.metabolism_prob = metabolism_prob
        self.metabolism_count = 0
        self.maintenance_count = 0
        self.recovery_events: List[RecoveryEvent] = []
        self.event_counter = 0
        self.MAINTENANCE_THRESHOLD = 62.0

    def observe(self, experience: Experience):
        features = self.abstract(experience)
        if self.residue >= self.MAINTENANCE_THRESHOLD:
            self._discharge_surplus()
        else:
            if random.random() < self._get_metabolism_prob():
                self._metabolism(features)
                self.metabolism_count += 1
            else:
                self._stable_update(features)
        self.update_abstractions(features)
        self.update_reserve()
        if len(self.abstractions) % 5 == 0 and self.state is not None:
            self.checkpoints.append(self.state)
            if len(self.checkpoints) > 5:
                self.checkpoints.pop(0)
        self.last_good_state = self.state

    def abstract(self, experience: Experience) -> StateNode:
        return StateNode(pattern=experience.pattern, direction=round(experience.direction, 2), effect=round(experience.effect, 2))

    def _get_metabolism_prob(self) -> float:
        if self.residue > 38: return min(0.024, self.metabolism_prob * 3.2)
        if self.residue > 24: return min(0.015, self.metabolism_prob * 2.2)
        if self.residue > 14: return min(0.009, self.metabolism_prob * 1.5)
        return self.metabolism_prob

    def _stable_update(self, features: StateNode):
        if self.state is None: self.state = features; return
        if self.state == features: return
        w = self.graph[self.state].get(features, 0.0) + 1.0
        self.graph[self.state][features] = w
        self.graph[features][self.state] = w
        self.state = features

    def _metabolism(self, new_features: StateNode):
        if self.state is None: self.state = new_features; return
        sources = set(self.graph.get(self.state, {}).keys())
        if 'active_states' in self.structural_reserve:
            sources.update(self.structural_reserve['active_states'][-4:])
        sources.update(self.checkpoints[-2:])
        best = max((s for s in sources if s in self.graph), key=lambda s: sum(self.graph[s].values()), default=None)
        if best and best != self.state:
            self.state = best
            self.residue = max(0.0, self.residue * 0.85)

    def recover(self) -> bool:
        if self.state is not None:
            self.residue = max(0.0, self.residue * 0.87)
            return True
        target = self._bfs_recovery()
        if target:
            self.state = target
            self.residue = max(0.0, self.residue * 0.64)
            return True
        if self.checkpoints:
            self.state = self.checkpoints[-1]
            self.residue = max(0.0, self.residue * 0.58)
            return True
        return False

    def _bfs_recovery(self) -> Optional[StateNode]:
        if not self.graph: return None
        sources = set(self.checkpoints[-3:])
        if 'active_states' in self.structural_reserve:
            sources.update(self.structural_reserve['active_states'][-5:])
        if not sources:
            sources = set(list(self.graph.keys())[:5])
        visited = set()
        queue = deque()
        for s in sources:
            if s in self.graph:
                queue.append((s, 0))
                visited.add(s)
        best = None
        best_score = -999.0
        metrics: Dict[str, float] = {}
        reach_before = self._count_reachable(self.state) if self.state else 0
        while queue:
            current, dist = queue.popleft()
            if dist > 2: continue
            strength = sum(self.graph[current].values())
            locality = strength / (dist + 1.0)
            res_win = self._residue_window(current)
            reach_gain = self._count_reachable(current) - reach_before
            redun = min(2.5, len(self.graph.get(current, {})) / 2.0)
            pat = 0.06 if (self.last_good_state and current.pattern == self.last_good_state.pattern) else 0.0
            score = (locality * 1.0 + res_win * 0.65 + reach_gain * 0.6 + redun * 0.45 + pat * 0.15)
            if score > best_score:
                best_score = score
                best = current
                metrics = {"locality": round(locality, 3), "residue_win": round(res_win, 3), "reach_gain": reach_gain, "redundancy": round(redun, 3), "pattern": round(pat, 3), "total": round(score, 3)}
            for n in self.graph.get(current, {}):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        if best:
            role = self._get_role(best)
            reach_after = self._count_reachable(best)
            self.event_counter += 1
            event = RecoveryEvent(
                event_id=f"recov_{self.event_counter}",
                step=len(self.abstractions),
                residue_before=round(self.residue, 2),
                residue_after=round(self.residue, 2),
                residue_gradient=round(self.residue * 0.32, 2),
                chosen_node_pattern=best.pattern,
                node_role=role,
                locality_score=metrics.get("locality", 0),
                residue_window_score=metrics.get("residue_win", 0),
                reachability_before=reach_before,
                reachability_after=reach_after,
                reachability_gain=metrics.get("reach_gain", 0),
                redundancy_gain=metrics.get("redundancy", 0),
                pattern_bonus=metrics.get("pattern", 0),
                total_score=metrics.get("total", 0),
                formation={"step": len(self.abstractions), "pattern": best.pattern, "role": role, "reachability_gain": metrics.get("reach_gain", 0), "residue_before": round(self.residue, 2)},
                utilization={"used": False, "usage_count": 0, "usage_history": []},
                damage={"survived": None, "recovery_success": None, "recovery_path_length": None}
            )
            self.recovery_events.append(event)
        return best

    def _residue_window(self, node: StateNode) -> float:
        # Simple implementation (original code had it, assuming it's defined)
        # For completeness, a basic version:
        if not self.graph: return 0.5
        deg = len(self.graph.get(node, {}))
        return min(1.0, deg / 8.0) * (1.0 - min(0.8, self.residue / 100.0))

    def _count_reachable(self, start: Optional[StateNode]) -> int:
        if not start or start not in self.graph: return 0
        visited, q = set(), deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            for nei in self.graph.get(cur, {}):
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)
        return len(visited)

    def _get_role(self, node: StateNode) -> str:
        if node in self.checkpoints: return "CORE"
        if sum(self.graph.get(node, {}).values()) >= 6: return "ADAPTIVE"
        return "RESERVOIR_CANDIDATE"

    def _discharge_surplus(self):
        self.maintenance_count += 1
        self.residue = max(0.0, self.residue * 0.47)

    def update_reserve(self):
        if self.state:
            self.structural_reserve['active_states'].append(self.state)
            if len(self.structural_reserve['active_states']) > 26:
                self.structural_reserve['active_states'].pop(0)

    def update_abstractions(self, features: StateNode):
        """抽象化リストを更新（簡易実装）"""
        if features not in self.abstractions:
            self.abstractions.append(features)
            if len(self.abstractions) > 50:
                self.abstractions.pop(0)

    def damage(self, severity: float = 0.11) -> int:
        if not self.graph: return 0
        nodes = list(self.graph.keys())
        num = max(1, int(len(nodes) * min(severity, 0.82)))
        to_remove = random.sample(nodes, num)
        for node in to_remove:
            if node not in self.graph: continue
            for neigh in list(self.graph[node]):
                self.graph[neigh].pop(node, None)
                if not self.graph[neigh]:
                    self.graph.pop(neigh, None)
            self.graph.pop(node, None)
        if self.state in to_remove:
            self.state = None
        self.residue += num * 1.4
        return num

    def get_stats(self) -> dict:
        return {"nodes": len(self.graph), "residue": round(self.residue, 2), "metabolism_count": self.metabolism_count, "recovery_events": len(self.recovery_events), "maintenance_count": self.maintenance_count}

    def get_recovery_summary(self) -> dict:
        if not self.recovery_events: return {"count": 0}
        events = self.recovery_events
        reach_gains = [e.reachability_gain for e in events]
        total_scores = [e.total_score for e in events]
        return {"count": len(events), "avg_reachability_gain": round(sum(reach_gains) / len(reach_gains), 2), "max_reachability_gain": max(reach_gains), "avg_total_score": round(sum(total_scores) / len(total_scores), 2), "core_ratio": round(sum(1 for e in events if e.node_role == "CORE") / len(events), 3), "reservoir_ratio": round(sum(1 for e in events if e.node_role == "RESERVOIR_CANDIDATE") / len(events), 3)}

    def analyze_by_node_role(self) -> dict:
        from collections import defaultdict
        groups = defaultdict(list)
        for e in self.recovery_events:
            groups[e.node_role].append(e)
        result = {}
        for role, events in groups.items():
            gains = [e.reachability_gain for e in events]
            result[role] = {"count": len(events), "avg_reachability_gain": round(sum(gains) / len(gains), 2), "max_reachability_gain": max(gains)}
        return result

    def get_reachability_correlation(self) -> dict:
        if len(self.recovery_events) < 3: return {"message": "データが少なすぎます"}
        events = self.recovery_events
        reach = [e.reachability_gain for e in events]
        residue = [e.residue_before for e in events]
        total = [e.total_score for e in events]
        def corr(x, y):
            n = len(x)
            mx, my = sum(x) / n, sum(y) / n
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            den = (sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for xi in y)) ** 0.5
            return round(num / den, 3) if den != 0 else 0.0
        return {"reachability_vs_residue": corr(reach, residue), "reachability_vs_total_score": corr(reach, total)}

    def export_recovery_events(self, filepath: str = "data1.jsonl"):
        with open(filepath, "w", encoding="utf-8") as f:
            for event in self.recovery_events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("MetaVoid core loaded.")
