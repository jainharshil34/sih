"""
Topological Road Network Map-Matching using Hidden Markov Models (HMM).

Unified re-export module pointing to canonical implementations:
- HMMMapMatcher in navresilient.mapmatching.hmm_matcher
- OSMGraphLoader and RoadEdge in navresilient.mapmatching.graph_loader
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navresilient.mapmatching.graph_loader import OSMGraphLoader, RoadEdge
from navresilient.mapmatching.hmm_matcher import HMMMapMatcher, MatchResult


@dataclass
class RoadSegment:
    """Directed road segment between two 2D coordinates (East, North)."""
    id: int
    p1: np.ndarray  # [x1, y1]
    p2: np.ndarray  # [x2, y2]
    speed_limit_mps: float = 25.0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def bearing(self) -> float:
        """Road segment azimuth in radians (0 = North, pi/2 = East)."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return float(math.atan2(dx, dy))

    def project_point(self, p: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Project query point p onto segment."""
        v = self.p2 - self.p1
        L_sq = float(np.dot(v, v))
        if L_sq < 1e-6:
            return self.p1.copy(), float(np.linalg.norm(p - self.p1)), 0.0

        t = np.clip(np.dot(p - self.p1, v) / L_sq, 0.0, 1.0)
        proj = self.p1 + t * v
        dist = float(np.linalg.norm(p - proj))
        return proj, dist, float(t)


class RoadNetwork:
    """Compatibility RoadNetwork wrapping OSMGraphLoader."""

    def __init__(self, loader: Optional[OSMGraphLoader] = None):
        self.loader = loader or OSMGraphLoader.create_synthetic_grid()

    @property
    def segments(self) -> List[RoadSegment]:
        return [
            RoadSegment(
                id=e.edge_id,
                p1=e.p1,
                p2=e.p2,
                speed_limit_mps=25.0
            )
            for e in self.loader.edges
        ]

    def add_segment(self, x1: float, y1: float, x2: float, y2: float,
                    seg_id: Optional[int] = None) -> RoadSegment:
        sid = seg_id if seg_id is not None else len(self.loader.edges)
        p1 = np.array([x1, y1], dtype=float)
        p2 = np.array([x2, y2], dtype=float)
        self.loader.add_edge_cartesian(f"node_{sid}_a", f"node_{sid}_b", p1, p2)
        return RoadSegment(id=sid, p1=p1, p2=p2)

    def add_polyline(self, points: np.ndarray):
        """Add sequential waypoints as connected road segments."""
        for i in range(len(points) - 1):
            self.add_segment(points[i, 0], points[i, 1], points[i+1, 0], points[i+1, 1])

    @classmethod
    def from_trajectory(cls, x: np.ndarray, y: np.ndarray, step: int = 5) -> RoadNetwork:
        """Build reference road network from a known ground truth polyline."""
        pts = np.column_stack([x, y])
        loader = OSMGraphLoader.from_polyline(pts, ref_lat=12.9716, ref_lon=77.5946, step_m=float(step))
        return cls(loader=loader)


__all__ = [
    "HMMMapMatcher",
    "MatchResult",
    "OSMGraphLoader",
    "RoadEdge",
    "RoadNetwork",
    "RoadSegment",
]

