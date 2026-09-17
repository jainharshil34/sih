# NavResilient Edge Integration Guide & CLI Protocol
> **Problem Statement SIH26168 (ISRO)** — Edge Deployable Resilient Navigation Engine

This guide details how **Engineer B's Mobile App**, **Telematics Edge Boxes**, **Microcontrollers (ESP32/Arduino)**, or **Robot Autonomy Systems** consume live dead-reckoning state from the standalone NavResilient engine without any external runtime dependencies.

---

## 1. Standalone Execution Modes

The standalone engine can be launched in any terminal, Docker container, or systemd service using `navresilient.cli`:

```bash
# 1. UNIX Stdio Pipe (Subprocess IPC for Flutter/React-Native/Desktop apps)
python -m navresilient.cli --mode stdio

# 2. TCP Socket Server (For remote telematics / LAN telemetry streams)
python -m navresilient.cli --mode tcp --host 0.0.0.0 --port 9090

# 3. High-Rate UDP Datagram Listener (10 Hz - 200 Hz embedded IMUs)
python -m navresilient.cli --mode udp --host 0.0.0.0 --port 9091

# 4. Replay Mode (Replays an existing drive log at 1x real-time)
python -m navresilient.cli --mode replay --replay-csv data/IO-VNBD/S-Vw12.csv --realtime
```

---

## 2. Sensor Input Protocol (What to send to NavResilient)

Feed line-delimited JSON objects over stdin, TCP, or UDP.

### High-Rate IMU Sample (10 Hz to 200 Hz)
```json
{
  "type": "IMU",
  "timestamp_s": 125.40,
  "ax_mps2": 0.12,
  "ay_mps2": -0.04,
  "az_mps2": 9.81,
  "gx_rads": 0.001,
  "gy_rads": -0.002,
  "gz_rads": 0.045
}
```
*(Key aliases supported: `ax`, `ay`, `az`, `gx`/`wx`, `gy`/`wy`, `gz`/`wz`, `t`/`timestamp`)*

### Low-Rate GNSS / NavIC Fix (1 Hz to 10 Hz)
```json
{
  "type": "GNSS",
  "timestamp_s": 125.00,
  "latitude": 12.971598,
  "longitude": 77.594562,
  "altitude_m": 920.0,
  "speed_mps": 14.5,
  "heading_deg": 182.4,
  "accuracy_m": 2.5
}
```

---

## 3. Position Output Stream (What Engineer B receives)

For each IMU frame pushed, the engine immediately responds with a JSON line containing the `DriftCorrectedState`:

```json
{
  "timestamp_s": 125.40,
  "latitude": 12.971612,
  "longitude": 77.594565,
  "altitude_m": 920.0,
  "speed_mps": 14.48,
  "speed_kmh": 52.13,
  "heading_deg": 182.35,
  "status": "GNSS_DENIED_INS",
  "drift_pct": 2.45,
  "pos_uncertainty_1sigma_m": 1.25,
  "is_stationary": false,
  "pothole_detected": false,
  "map_matched_lat": 12.971610,
  "map_matched_lon": 77.594568,
  "engine_latency_ms": 3.82
}
```

### System Status Values
- `"INITIALIZING"`: Calibrating initial tilt and waiting for first GNSS fix.
- `"GNSS_LOCKED"`: High-accuracy GNSS/NavIC lock active; UKF parameters continuously tuned.
- `"GNSS_DENIED_INS"`: Active blackout (tunnel/basement); pure AI-augmented dead reckoning active.
- `"REACQUIRED"`: Smooth handoff phase on satellite recovery (zero step jumps).

---

## 4. Consumer Integration Examples

### A. Python Client (Socket IPC)
```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9090))
sock_file = sock.makefile("r", encoding="utf-8")

# Send IMU Frame
imu_msg = {
    "type": "IMU", "timestamp_s": 1.0,
    "ax_mps2": 0.2, "ay_mps2": 0.0, "az_mps2": 9.81,
    "gx_rads": 0.0, "gy_rads": 0.0, "gz_rads": 0.01
}
sock.sendall((json.dumps(imu_msg) + "\n").encode("utf-8"))

# Read DriftCorrectedState
state_json = sock_file.readline()
state = json.loads(state_json)
print(f"Lat: {state['latitude']}, Lon: {state['longitude']}, Drift: {state['drift_pct']}%")
```

### B. Flutter / Dart Subprocess Pipe
```dart
import 'dart:convert';
import 'dart:io';

void startNavResilient() async {
  final process = await Process.start('python', ['-m', 'navresilient.cli', '--mode', 'stdio']);

  // Listen to output stream from NavResilient
  process.stdout
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) {
    if (line.isNotEmpty) {
      final state = jsonDecode(line);
      print("UI Update: ${state['latitude']}, ${state['longitude']} (Status: ${state['status']})");
    }
  });

  // Feed sensor data
  final imuSample = jsonEncode({
    "type": "IMU",
    "timestamp_s": DateTime.now().millisecondsSinceEpoch / 1000.0,
    "ax_mps2": 0.1, "ay_mps2": 0.0, "az_mps2": 9.81,
    "gx_rads": 0.0, "gy_rads": 0.0, "gz_rads": 0.0
  });
  process.stdin.writeln(imuSample);
}
```

### C. POSIX C / C++ Socket Client (Embedded Telematics)
```c
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(9090),
        .sin_addr.s_addr = inet_addr("127.0.0.1")
    };
    connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    const char *imu_line = "{\"type\":\"IMU\",\"timestamp_s\":10.0,\"ax\":0.1,\"ay\":0.0,\"az\":9.81,\"gx\":0,\"gy\":0,\"gz\":0}\n";
    send(sock, imu_line, strlen(imu_line), 0);

    char buffer[2048] = {0};
    read(sock, buffer, sizeof(buffer) - 1);
    printf("NavResilient State: %s\n", buffer);

    close(sock);
    return 0;
}
```
