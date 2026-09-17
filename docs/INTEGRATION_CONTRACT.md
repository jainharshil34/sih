# NavResilient Engine Integration Contract (for Engineer B)

This document defines the strict API contract, telemetry data structures, model artifacts, and latency budgets between the **NavResilient Navigation Engine (Engineer A)** and the **Mobile UI / App Layer (Engineer B)** for SIH26168 (ISRO).

---

## 1. Streaming Ingestion Interface

Engineer B feeds incoming smartphone sensor streams into `NavResilientEngine`:

```python
from navresilient import NavResilientEngine
from navresilient.contract import IMUFrame, GNSSFix

engine = NavResilientEngine(fs_imu=10.0)

# Feed GNSS fix when received (1 Hz - 10 Hz)
engine.push_gnss(
    lat=12.971598,
    lon=77.594562,
    alt=920.0,
    speed_mps=16.7,
    heading_deg=45.0,
    accuracy_m=2.5,
    timestamp=100.0
)

# Feed high-rate IMU frame (10 Hz up to 200 Hz)
state = engine.push_imu(
    ax=0.12, ay=-0.02, az=9.81,
    gx=0.001, gy=0.002, gz=0.015,
    timestamp=100.1
)
```

---

## 2. Output Telemetry Schema (`DriftCorrectedState`)

Every IMU push returns a typed, JSON-serializable `DriftCorrectedState`:

| Field | Type | Description |
|---|---|---|
| `timestamp_s` | `float` | Unix / monotonic timestamp in seconds |
| `latitude` | `float` | WGS-84 Latitude (deg) |
| `longitude` | `float` | WGS-84 Longitude (deg) |
| `altitude_m` | `float` | Altitude above ellipsoid (meters) |
| `speed_mps` | `float` | Current vehicle forward speed ($m/s$) |
| `speed_kmh` | `float` | Current vehicle forward speed ($km/h$) |
| `heading_deg` | `float` | Azimuth orientation ($0^\circ-360^\circ$, $0^\circ = \text{True North}$) |
| `status` | `str` | `"GNSS_LOCKED"`, `"GNSS_DENIED_INS"`, or `"REACQUIRED"` |
| `drift_pct` | `float` | Estimated drift percentage ($< 10\%$ benchmark ceiling) |
| `pos_uncertainty_1sigma_m` | `float` | $1\sigma$ position uncertainty radius in meters |
| `is_stationary` | `bool` | `True` when vehicle is stopped at signal/junction (ZUPT active) |
| `pothole_detected` | `bool` | `True` when road shock / pothole spike is isolated & rejected |
| `map_matched_lat` | `float?` | Snapped road centerline latitude (if road map provided) |
| `map_matched_lon` | `float?` | Snapped road centerline longitude (if road map provided) |
| `engine_latency_ms` | `float` | Processing execution time per step ($\sim 0.15\text{ ms}$) |

---

## 3. Exported On-Device Model Artifacts

Artifacts exported in `artifacts/models/`:

- `tcn_velocity.pt`: PyTorch weights.
- `tcn_velocity_torchscript.pt`: Traced TorchScript module for mobile native runtime (C++, Android JNI, iOS LibTorch).
- `tcn_velocity.tflite` / `tcn_velocity.onnx`: LiteRT / ONNX graph for edge NPU/GPU acceleration.
- `model_metadata.json`: Specification of tensor input shape `(1, 6, 20)`, 6-DoF channel ordering, and output heads.

---

## 4. Performance & Latency Budget

- **IMU Ingestion Rate**: $10\text{ Hz}$ on smartphone up to $200\text{ Hz}$ on edge telematics.
- **Engine Processing Latency**: $< 0.5\text{ ms}$ per step on standard ARM mobile CPU (measured throughput: $> 700\text{ Hz}$).
- **Dead Reckoning Drift**: Strictly **$< 10\%$ of distance travelled** (achieved: **$2.11\%$** on $767\text{ m}$ tunnel case).
