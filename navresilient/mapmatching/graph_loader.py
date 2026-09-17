"""
OSM Road Graph Loader & Spatial Indexer for Map Matching.

Tech Stack:
- Offline OpenStreetMap graph parsing via networkx (MultiDiGraph / DiGraph)
- Compatible with OSMnx 2.x GraphML files and custom geospatial formats
- High-speed spatial nearest-edge indexing via scipy.spatial.KDTree
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


@dataclass
class RoadEdge:
    """Directed road edge in local Cartesian ENU coordinates (meters)."""
    u: int | str
    v: int | str
    key: int
    edge_id: int
    p1: np.ndarray  # [East, North]
    p2: np.ndarray  # [East, North]
    length_m: float
    bearing_rad: float
    one_way: bool = False
    name: str = "road"
    highway: str = "primary"

    def project_point(self, p: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Project query point p [East, North] onto edge.
        
        Returns:
            proj_p: closest point on road centerline [East, North]
            dist: perpendicular distance in meters
            fraction: fractional distance along edge [0.0, 1.0]
        """
        vec = self.p2 - self.p1
        L_sq = float(np.dot(vec, vec))
        if L_sq < 1e-8:
            return self.p1.copy(), float(np.linalg.norm(p - self.p1)), 0.0

        t = np.clip(np.dot(p - self.p1, vec) / L_sq, 0.0, 1.0)
        proj = self.p1 + t * vec
        dist = float(np.linalg.norm(p - proj))
        return proj, dist, float(t)


class OSMGraphLoader:
    """Loads, manages, and spatially indexes OSM road network graphs."""

    def __init__(self, ref_lat: float = 12.9716, ref_lon: float = 77.5946):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.graph = nx.MultiDiGraph()
        self.edges: List[RoadEdge] = []
        self._kd_tree: Optional[KDTree] = None
        self._edge_midpoints: np.ndarray = np.empty((0, 2))

    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS-84 (lat, lon) to local East-North-Up Cartesian coordinates (m)."""
        R = 6378137.0
        lat0_rad = math.radians(self.ref_lat)
        lon0_rad = math.radians(self.ref_lon)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        east = (lon_rad - lon0_rad) * math.cos(lat0_rad) * R
        north = (lat_rad - lat0_rad) * R
        return east, north

    def enu_to_latlon(self, east: float, north: float) -> Tuple[float, float]:
        """Convert local Cartesian coordinates (m) back to WGS-84 (lat, lon)."""
        R = 6378137.0
        lat0_rad = math.radians(self.ref_lat)
        lon0_rad = math.radians(self.ref_lon)

        lat_rad = lat0_rad + (north / R)
        lon_rad = lon0_rad + (east / (R * math.cos(lat0_rad)))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    def load_from_graphml(self, filepath: str) -> bool:
        """Load offline GraphML file exported by OSMnx or QGIS."""
        if not os.path.exists(filepath):
            return False
        
        try:
            self.graph = nx.read_graphml(filepath)
            self._rebuild_edges_from_networkx()
            return True
        except Exception as e:
            print(f"[OSMGraphLoader] Error reading {filepath}: {e}")
            return False

    def save_to_graphml(self, filepath: str):
        """Save road network to GraphML format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nx.write_graphml(self.graph, filepath)

    def _rebuild_edges_from_networkx(self):
        """Extract Cartesian road edges and build spatial KD-Tree."""
        self.edges.clear()
        midpoints = []

        edge_idx = 0
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            u_node = self.graph.nodes[u]
            v_node = self.graph.nodes[v]

            # Extract lat/lon from nodes
            u_lat = float(u_node.get("y", u_node.get("lat", self.ref_lat)))
            u_lon = float(u_node.get("x", u_node.get("lon", self.ref_lon)))
            v_lat = float(v_node.get("y", v_node.get("lat", self.ref_lat)))
            v_lon = float(v_node.get("x", v_node.get("lon", self.ref_lon)))

            p1 = np.array(self.latlon_to_enu(u_lat, u_lon), dtype=float)
            p2 = np.array(self.latlon_to_enu(v_lat, v_lon), dtype=float)

            diff = p2 - p1
            length = float(np.linalg.norm(diff))
            if length < 0.1:
                continue

            bearing = float(math.atan2(diff[0], diff[1]))
            one_way = bool(data.get("oneway", False))
            name = str(data.get("name", "road"))
            highway = str(data.get("highway", "primary"))

            edge = RoadEdge(
                u=u, v=v, key=k, edge_id=edge_idx,
                p1=p1, p2=p2, length_m=length, bearing_rad=bearing,
                one_way=one_way, name=name, highway=highway
            )
            self.edges.append(edge)
            midpoints.append(0.5 * (p1 + p2))
            edge_idx += 1

        if midpoints:
            self._edge_midpoints = np.array(midpoints)
            self._kd_tree = KDTree(self._edge_midpoints)
        else:
            self._edge_midpoints = np.empty((0, 2))
            self._kd_tree = None

    def add_edge_cartesian(
        self,
        u: int | str,
        v: int | str,
        p1: np.ndarray,
        p2: np.ndarray,
        one_way: bool = False,
        name: str = "road"
    ) -> RoadEdge:
        """Directly add a Cartesian road segment in meters."""
        diff = p2 - p1
        length = float(np.linalg.norm(diff))
        bearing = float(math.atan2(diff[0], diff[1]))
        edge_id = len(self.edges)

        edge = RoadEdge(
            u=u, v=v, key=0, edge_id=edge_id,
            p1=np.array(p1, dtype=float), p2=np.array(p2, dtype=float),
            length_m=length, bearing_rad=bearing,
            one_way=one_way, name=name
        )
        self.edges.append(edge)
        
        # Add to NetworkX
        lat1, lon1 = self.enu_to_latlon(p1[0], p1[1])
        lat2, lon2 = self.enu_to_latlon(p2[0], p2[1])
        self.graph.add_node(u, x=lon1, y=lat1)
        self.graph.add_node(v, x=lon2, y=lat2)
        self.graph.add_edge(u, v, key=0, length=length, oneway=one_way, name=name)

        if not one_way:
            self.graph.add_edge(v, u, key=0, length=length, oneway=one_way, name=name)

        # Refresh KD-Tree
        midpoints = [0.5 * (e.p1 + e.p2) for e in self.edges]
        self._edge_midpoints = np.array(midpoints)
        self._kd_tree = KDTree(self._edge_midpoints)
        return edge

    def query_candidate_edges(self, query_p: np.ndarray, radius_m: float = 35.0) -> List[RoadEdge]:
        """Fast spatial indexing query returning all candidate edges within radius."""
        if self._kd_tree is None or len(self.edges) == 0:
            return []

        # Find all midpoints within extended radius (adding edge length buffer)
        query_2d = query_p[:2]
        indices = self._kd_tree.query_ball_point(query_2d, r=radius_m + 50.0)
        return [self.edges[idx] for idx in indices]

    @classmethod
    def create_synthetic_grid(
        cls,
        ref_lat: float = 12.9716,
        ref_lon: float = 77.5946,
        grid_size_m: float = 500.0,
        block_spacing_m: float = 100.0
    ) -> OSMGraphLoader:
        """Generates a deterministic synthetic urban road grid for offline benchmarking."""
        loader = cls(ref_lat=ref_lat, ref_lon=ref_lon)
        n_blocks = int(grid_size_m / block_spacing_m)

        # Horizontal streets
        for i in range(n_blocks + 1):
            y = i * block_spacing_m
            for j in range(n_blocks):
                x1 = j * block_spacing_m
                x2 = (j + 1) * block_spacing_m
                u = f"n_{j}_{i}"
                v = f"n_{j+1}_{i}"
                loader.add_edge_cartesian(u, v, np.array([x1, y]), np.array([x2, y]), name=f"Street_{i}")

        # Vertical avenues
        for j in range(n_blocks + 1):
            x = j * block_spacing_m
            for i in range(n_blocks):
                y1 = i * block_spacing_m
                y2 = (i + 1) * block_spacing_m
                u = f"n_{j}_{i}"
                v = f"n_{j}_{i+1}"
                loader.add_edge_cartesian(u, v, np.array([x, y1]), np.array([x, y2]), name=f"Avenue_{j}")

        return loader
