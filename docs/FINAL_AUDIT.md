# NavResilient Final Audit

Audit date: 2026-09-25

## Verified fixes

- Dashboard map camera framing is now performed once for a selected route rather
  than once per replay frame. This prevents repeated `fitBounds` calls from
  forcing rapid zoom changes.
- Split-screen Leaflet synchronization now runs after a completed map movement
  with a re-entrancy guard. Live following has one camera source, cancels stale
  animations, and pauses when a user drags the map.
- OSM tile layers no longer refresh during a zoom transformation, reducing tile
  churn on slow connections.
- GNSS outage detection now works when the first valid GNSS timestamp is `0.0`.
- The map-matching spatial index now finds long road segments near their
  endpoints instead of pruning them using a fixed midpoint buffer.

## Requirements status

| Capability | Status | Evidence / limitation |
|---|---|---|
| GNSS/IMU edge engine | PARTIALLY IMPLEMENTED | Python pipeline, typed contracts, stdio/TCP/UDP code and unit tests exist; runtime tests could not be completed in this environment because the pinned Torch dependency download stalled. |
| Alignment, filtering, ZUPT, UKF and HMM | UNVERIFIED | Implementations and unit-test coverage exist, but full replay has not been executed here. |
| AI speed inference | UNVERIFIED | Model artifacts exist, but IO-VNBD source CSVs and reproducible training metrics are absent from this checkout. |
| Real IO-VNBD evaluation | NOT REPRODUCIBLE | The checkout has no `data/IO-VNBD` CSV files. The official dataset is external. |
| Offline OSM map support | PARTIALLY IMPLEMENTED | GraphML loading is implemented, but no GraphML/OSM map asset is supplied. The automatic fallback builds a corridor from prior GNSS points; it is not an offline OSM map. |
| Mobile application | NOT IMPLEMENTED | There is no Android, iOS, Flutter, or React Native application source. There are edge protocol examples and exported model artifacts only. |
| Dashboard | PROTOTYPE | Vite build succeeds and serves a replay UI, but it imports static point arrays and does not consume the edge REST API. It must be presented as a replay/demo, not live engine telemetry. |

## Benchmark and claim integrity

Do not present the README's SIH-pass figures as independently verified. In
particular, `figures/multi_duration_benchmark.json` records failed real-drive
segments, conflicting with the README's blanket success claims. The stored
benchmark metadata also disagrees about latency. A repeatable dataset download,
fixed train/validation/test drive split, and fresh evaluation report are required
before claiming the <10% drift requirement.

Runtime `drift_pct` is derived from filter uncertainty divided by travelled
distance, not a ground-truth position error. It is useful telemetry but is not
evidence of SIH positional drift compliance.

## Remaining must-fix items before SIH

1. Download and retain documented IO-VNBD training and held-out test drives;
   rerun training and outage evaluation without using the test trajectory to
   construct a map corridor.
2. Replace dashboard placeholder/static replay data with the edge-engine API, or
   label it explicitly as a recorded replay.
3. Ship an actual mobile application and test it on a physical device at 10 Hz.
4. Package a real offline OSM/GraphML map for the demo location and validate
   map matching against it.
5. Reconcile or remove unsupported README/deck metrics and record hardware,
   commands, split methodology, and raw outputs for every retained claim.
