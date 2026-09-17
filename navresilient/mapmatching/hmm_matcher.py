"""
Hidden Markov Model (HMM) Map Matcher with Non-Holonomic Constraints (NHC).

Snaps continuous fused inertial dead-reckoned trajectory onto an offline OSM
road network graph.

Robust Edge Cases Handled:
- "No road nearby" / Underground Parking Garage: detects unmapped or off-road
  regions gracefully and seamlessly outputs raw inertial trajectory without
  crashing, distorting the HMM trellis, or snapping to unrelated distant highways.
- Non-Holonomic Constraints: penalizes unphysical reverse driving on one-way
  links or impossible lateral skips across disconnected flyovers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from navresilient.mapmatching.graph_loader import OSMGraphLoader, RoadEdge


@dataclass
class MatchResult:
    """Map-matching output state at current timestep."""
    snapped_pos: np.ndarray          # [East, North] in meters
    raw_pos: np.ndarray              # [East, North] input inertial
    is_snapped: bool                 # True if matched to a road segment
    edge_id: int                     # ID of matched road edge (-1 if off-road)
    road_name: str                   # Name of matched road
    cross_track_dist_m: float        # Perpendicular distance to road centerline
    heading_error_deg: float         # Discrepancy between road bearing and trajectory
    confidence: float                # Match confidence [0.0, 1.0]
    is_off_road: bool                # True when in underground structure or open lot


class HMMMapMatcher:
    """Online HMM Viterbi Map Matcher with Non-Holonomic Constraints & Off-Road Resilience."""

    def __init__(
        self,
        graph_loader: OSMGraphLoader,
        sigma_z: float = 6.0,
        beta: float = 4.0,
        max_search_radius_m: float = 35.0,
        off_road_dist_threshold_m: float = 40.0
    ):
        self.loader = graph_loader
        self.sigma_z = sigma_z
        self.beta = beta
        self.max_search_radius_m = max_search_radius_m
        self.off_road_dist_threshold_m = off_road_dist_threshold_m

        # Viterbi Trellis State
        self.prev_candidates: List[Dict[str, Any]] = []
        self.prev_raw_p: Optional[np.ndarray] = None
        self.off_road_mode: bool = False

    def reset(self):
        """Reset internal HMM trellis history."""
        self.prev_candidates.clear()
        self.prev_raw_p = None
        self.off_road_mode = False

    def match(
        self,
        raw_pos: np.ndarray,
        heading_rad: float,
        speed_mps: float = 10.0
    ) -> MatchResult:
        """Snap a 2D inertial position [East, North] to the most probable road edge.
        
        Args:
            raw_pos: [East, North] in meters
            heading_rad: vehicle azimuth in radians (0 = North, pi/2 = East)
            speed_mps: forward speed in m/s
        
        Returns:
            MatchResult containing snapped coordinates and diagnostic metadata.
        """
        raw_2d = np.array(raw_pos[:2], dtype=float)

        # 1. Fast Spatial Candidate Query via KD-Tree
        candidate_edges = self.loader.query_candidate_edges(raw_2d, radius_m=self.max_search_radius_m)

        candidates = []
        for edge in candidate_edges:
            proj, dist, frac = edge.project_point(raw_2d)
            if dist > self.max_search_radius_m:
                continue

            # Heading alignment score (NHC: vehicle travel must align with road bearing)
            # If road is two-way, allow bearing in both directions
            d_head1 = abs((heading_rad - edge.bearing_rad + math.pi) % (2.0 * math.pi) - math.pi)
            if edge.one_way:
                head_score = math.cos(d_head1) ** 2 if d_head1 < (math.pi / 2.0) else 0.001
            else:
                d_head2 = abs((heading_rad - (edge.bearing_rad + math.pi) + math.pi) % (2.0 * math.pi) - math.pi)
                min_d = min(d_head1, d_head2)
                head_score = math.cos(min_d) ** 2 if min_d < (math.pi / 2.0) else 0.01

            # Spatial Emission Probability: p(z | edge)
            spatial_prob = (1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_z)) * math.exp(-0.5 * (dist / self.sigma_z) ** 2)
            emission = max(1e-6, spatial_prob * max(0.05, head_score))

            candidates.append({
                "edge": edge,
                "proj": proj,
                "dist": dist,
                "frac": frac,
                "emission": emission,
                "viterbi_prob": emission,
                "head_diff_rad": d_head1
            })

        # ---------------------------------------------------------------------
        # EDGE CASE HANDLING: "No Road Nearby" / Underground Parking Structure
        # ---------------------------------------------------------------------
        if not candidates or all(c["dist"] > self.off_road_dist_threshold_m for c in candidates):
            self.off_road_mode = True
            self.prev_candidates.clear()
            self.prev_raw_p = raw_2d.copy()

            # Return raw dead-reckoned trajectory gracefully without crashing or false-snapping
            return MatchResult(
                snapped_pos=raw_2d.copy(),
                raw_pos=raw_2d.copy(),
                is_snapped=False,
                edge_id=-1,
                road_name="Off-Road / Parking Structure",
                cross_track_dist_m=0.0,
                heading_error_deg=0.0,
                confidence=0.0,
                is_off_road=True
            )

        self.off_road_mode = False

        # 2. Viterbi Transition Matrix with NetworkX Shortest Path (NHC)
        if self.prev_candidates and self.prev_raw_p is not None:
            dist_inertial = float(np.linalg.norm(raw_2d - self.prev_raw_p))

            for cand in candidates:
                best_viterbi = 0.0
                curr_edge = cand["edge"]

                for prev in self.prev_candidates:
                    prev_edge = prev["edge"]

                    # Compute network topological distance
                    if prev_edge.edge_id == curr_edge.edge_id:
                        # Same segment
                        dist_network = float(np.linalg.norm(cand["proj"] - prev["proj"]))
                    else:
                        # Shortest path between endpoints
                        try:
                            dist_network = nx.shortest_path_length(
                                self.loader.graph,
                                source=prev_edge.v,
                                target=curr_edge.u,
                                weight="length"
                            )
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            # Topological disconnect or illegal U-turn / one-way violation
                            dist_network = float(np.linalg.norm(cand["proj"] - prev["proj"])) * 3.0

                    # Transition Probability: p(s_j | s_i)
                    delta_diff = abs(dist_inertial - dist_network)
                    trans_prob = (1.0 / self.beta) * math.exp(-min(20.0, delta_diff / self.beta))
                    total_prob = prev["viterbi_prob"] * trans_prob * cand["emission"]

                    if total_prob > best_viterbi:
                        best_viterbi = total_prob

                cand["viterbi_prob"] = max(best_viterbi, cand["emission"] * 0.05)

        # 3. Maximum A Posteriori Candidate Selection
        best_cand = max(candidates, key=lambda c: c["viterbi_prob"])
        self.prev_candidates = candidates
        self.prev_raw_p = raw_2d.copy()

        # 4. Confidence-Weighted Blending (preserves visual smoothness)
        raw_dist = best_cand["dist"]
        conf = float(np.clip(1.0 - (raw_dist / self.max_search_radius_m), 0.0, 1.0))
        # High confidence snaps closely to centerline; low confidence softly blends
        snapped = (1.0 - conf) * raw_2d + conf * best_cand["proj"]

        return MatchResult(
            snapped_pos=snapped,
            raw_pos=raw_2d,
            is_snapped=True,
            edge_id=best_cand["edge"].edge_id,
            road_name=best_cand["edge"].name,
            cross_track_dist_m=raw_dist,
            heading_error_deg=float(math.degrees(best_cand["head_diff_rad"])),
            confidence=conf,
            is_off_road=False
        )
