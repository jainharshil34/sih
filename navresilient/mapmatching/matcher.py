"""
Topological Road Network Map-Matching using Hidden Markov Models (HMM).

Problem Statement SIH26168 Requirement:
During prolonged GNSS outages, small heading bias or road curvature will eventually
lead to slight cross-track drift. Since land vehicles travel strictly along physical
road lanes, this module snaps the drifting inertial dead reckoning state back onto
the road geometry using an online HMM/Viterbi map matcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
        """Project query point p onto segment.

        Returns:
            proj_point: nearest point on segment [x, y]
            perp_dist: perpendicular distance from point to segment
            fraction: fractional distance along segment [0.0, 1.0]
        """
        v = self.p2 - self.p1
        L_sq = float(np.dot(v, v))
        if L_sq < 1e-6:
            return self.p1.copy(), float(np.linalg.norm(p - self.p1)), 0.0

        t = np.clip(np.dot(p - self.p1, v) / L_sq, 0.0, 1.0)
        proj = self.p1 + t * v
        dist = float(np.linalg.norm(p - proj))
        return proj, dist, float(t)


class RoadNetwork:
    """Road network graph containing topological road segments."""

    def __init__(self):
        self.segments: List[RoadSegment] = []

    def add_segment(self, x1: float, y1: float, x2: float, y2: float,
                    seg_id: Optional[int] = None) -> RoadSegment:
        sid = seg_id if seg_id is not None else len(self.segments)
        seg = RoadSegment(id=sid, p1=np.array([x1, y1], dtype=float),
                          p2=np.array([x2, y2], dtype=float))
        self.segments.append(seg)
        return seg

    def add_polyline(self, points: np.ndarray):
        """Add sequential waypoints as connected road segments."""
        for i in range(len(points) - 1):
            self.add_segment(points[i, 0], points[i, 1], points[i+1, 0], points[i+1, 1])

    @classmethod
    def from_trajectory(cls, x: np.ndarray, y: np.ndarray, step: int = 5) -> RoadNetwork:
        """Build reference road network from a known ground truth polyline."""
        net = cls()
        for i in range(0, len(x) - step, step):
            net.add_segment(x[i], y[i], x[i + step], y[i + step])
        return net


class HMMMapMatcher:
    """Online Hidden Markov Model Map Matcher for Real-Time Inertial Snapping."""

    def __init__(self, road_net: RoadNetwork,
                 sigma_z: float = 6.0, beta: float = 4.0, max_search_radius: float = 35.0):
        self.road_net = road_net
        self.sigma_z = sigma_z
        self.beta = beta
        self.max_search_radius = max_search_radius

        # Viterbi state tracking
        self.prev_candidates: List[dict] = []
        self.prev_raw_p: Optional[np.ndarray] = None

    def match(self, raw_p: np.ndarray, heading_rad: float) -> Tuple[np.ndarray, dict]:
        """Snap a 2D inertial position [East, North] to the most likely road segment.

        Returns:
            snapped_p: [x, y] position on the road centerline
            info: metadata containing segment_id, cross_track_error, confidence
        """
        raw_2d = raw_p[:2]
        candidates = []

        for seg in self.road_net.segments:
            proj, dist, frac = seg.project_point(raw_2d)
            if dist > self.max_search_radius:
                continue

            # Heading difference penalty
            d_head = abs((heading_rad - seg.bearing + math.pi) % (2 * math.pi) - math.pi)
            head_score = max(0.05, math.cos(d_head) ** 2 if d_head < math.pi / 2 else 0.01)

            # Spatial emission probability: P(z | road)
            emission = (1.0 / (math.sqrt(2 * math.pi) * self.sigma_z)) * math.exp(-0.5 * (dist / self.sigma_z) ** 2) * head_score

            candidates.append({
                "segment": seg,
                "proj": proj,
                "dist": dist,
                "frac": frac,
                "emission": emission,
                "score": emission
            })

        if not candidates:
            # No nearby road found within search radius, keep raw inertial position
            return raw_2d.copy(), {"segment_id": -1, "snapped": False, "dist": 0.0, "confidence": 0.0}

        # Transition probabilities if previous step exists
        if self.prev_candidates and self.prev_raw_p is not None:
            dist_inertial = float(np.linalg.norm(raw_2d - self.prev_raw_p))
            for cand in candidates:
                best_trans = 0.0
                for prev in self.prev_candidates:
                    dist_road = float(np.linalg.norm(cand["proj"] - prev["proj"]))
                    diff = abs(dist_inertial - dist_road)
                    trans_prob = (1.0 / self.beta) * math.exp(-diff / self.beta)
                    total_prob = prev["score"] * trans_prob * cand["emission"]
                    if total_prob > best_trans:
                        best_trans = total_prob
                cand["score"] = max(best_trans, cand["emission"] * 0.1)

        # Select maximum likelihood candidate
        best_cand = max(candidates, key=lambda c: c["score"])
        self.prev_candidates = candidates
        self.prev_raw_p = raw_2d.copy()

        # Confidence blending (soft snapping for continuous vehicle movement)
        confidence = min(1.0, best_cand["emission"] * self.sigma_z * 2.5)
        snapped = (1.0 - confidence) * raw_2d + confidence * best_cand["proj"]

        return snapped, {
            "segment_id": best_cand["segment"].id,
            "snapped": True,
            "dist": best_cand["dist"],
            "confidence": float(confidence)
        }
