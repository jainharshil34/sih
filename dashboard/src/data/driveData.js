/**
 * Real IO-VNBD Drive Data & Ground Truth Trajectories.
 * Sourced directly from IO-VNBD benchmarks and refined smartphone PDR recordings.
 * All units: timestamps in seconds (s), speed in m/s (km/h), distance and errors in meters (m).
 */

import myWalkPoints from './myWalkPoints.json';
import svw12Points from './svw12Points.json';
import svta10Points from './svta10Points.json';
import svw10Points from './svw10Points.json';

export const REAL_DRIVES = {
  "S-Vw12": {
    id: "S-Vw12",
    name: "IO-VNBD S-Vw12 (Ford Fiesta · M5 Motorway & Worcestershire)",
    vehicle: "Ford Fiesta 1.0 EcoBoost (Tire 195/55 R16, R = 0.303m)",
    duration_s: 91.7,
    sample_rate_hz: 10.0,
    ref_lat: 52.234146,
    ref_lon: -2.138447,
    outage_start_s: 20.0,
    outage_len_s: 40.0,
    distance_m: 2154.2,
    generatePoints: () => svw12Points
  },
  "S-Vta10": {
    id: "S-Vta10",
    name: "IO-VNBD S-Vta10 (Ford Fiesta · A38 Dual Carriageway)",
    vehicle: "Ford Fiesta 1.0 EcoBoost (Windshield Suction Mount)",
    duration_s: 150.1,
    sample_rate_hz: 10.0,
    ref_lat: 52.881442,
    ref_lon: -1.717635,
    outage_start_s: 25.0,
    outage_len_s: 50.0,
    distance_m: 3588.0,
    generatePoints: () => svta10Points
  },
  "my-walk": {
    id: "my-walk",
    name: "📱 Real Phone Walk (Thapar University Campus · Patiala)",
    vehicle: "Real Smartphone (Handheld IMU · 132.8s Walk Log)",
    duration_s: 132.8,
    sample_rate_hz: 10.0,
    ref_lat: 30.351814,
    ref_lon: 76.364515,
    outage_start_s: 20.0,
    outage_len_s: 40.0,
    distance_m: 107.5,
    generatePoints: () => myWalkPoints
  },
  "S-Vw10": {
    id: "S-Vw10",
    name: "IO-VNBD S-Vw10 (Ford Fiesta · Worcester Urban Avenues)",
    vehicle: "Ford Fiesta 1.0 EcoBoost (Dashboard Mount)",
    duration_s: 65.1,
    sample_rate_hz: 10.0,
    ref_lat: 52.201747,
    ref_lon: -2.202120,
    outage_start_s: 15.0,
    outage_len_s: 30.0,
    distance_m: 729.7,
    generatePoints: () => svw10Points
  }
};

