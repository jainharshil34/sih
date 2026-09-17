"""
Unit Tests for Standalone Edge Deployment (Pipes, Sockets, and IPC Streaming).
"""

import io
import json
import socket
import threading
import time
import pytest

from navresilient.contract import GNSSFix, IMUFrame, SystemStatus
from navresilient.server import StandaloneEdgeEngine, parse_sensor_json


class TestStandaloneEdge:
    def test_json_parsing(self):
        # Test IMU JSON parsing
        imu_json = '{"type": "IMU", "timestamp_s": 10.5, "ax": 0.5, "ay": 0.1, "az": 9.8, "gx": 0.01, "gy": 0.0, "gz": 0.02}'
        msg_type, imu, gnss = parse_sensor_json(imu_json)
        assert msg_type == "IMU"
        assert imu is not None
        assert imu.timestamp_s == 10.5
        assert imu.ax_mps2 == 0.5

        # Test GNSS JSON parsing
        gnss_json = '{"type": "GNSS", "timestamp_s": 10.0, "latitude": 12.9716, "longitude": 77.5946, "speed_mps": 15.0, "heading_deg": 45.0, "accuracy_m": 2.0}'
        msg_type, imu, gnss = parse_sensor_json(gnss_json)
        assert msg_type == "GNSS"
        assert gnss is not None
        assert gnss.latitude == 12.9716
        assert gnss.speed_mps == 15.0

    def test_stdio_pipe_streaming(self):
        engine = StandaloneEdgeEngine(fs_imu=10.0, ref_lat=12.9716, ref_lon=77.5946)

        input_data = (
            '{"type": "GNSS", "timestamp_s": 0.0, "latitude": 12.9716, "longitude": 77.5946, "speed_mps": 10.0, "heading_deg": 0.0}\n'
            '{"type": "IMU", "timestamp_s": 0.1, "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}\n'
            '{"type": "IMU", "timestamp_s": 0.2, "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}\n'
        )

        in_stream = io.StringIO(input_data)
        out_stream = io.StringIO()

        engine.run_stdio_pipe(in_stream=in_stream, out_stream=out_stream)

        out_lines = out_stream.getvalue().strip().split("\n")
        assert len(out_lines) == 2  # 2 IMU frames produced 2 output states

        res1 = json.loads(out_lines[0])
        assert "latitude" in res1
        assert "longitude" in res1
        assert "status" in res1
        assert "drift_pct" in res1

    def test_tcp_socket_server(self):
        engine = StandaloneEdgeEngine(fs_imu=10.0, ref_lat=12.9716, ref_lon=77.5946)
        stop_event = threading.Event()
        port = 19092

        server_thread = threading.Thread(
            target=engine.run_tcp_server,
            kwargs={"host": "127.0.0.1", "port": port, "stop_event": stop_event},
            daemon=True
        )
        server_thread.start()
        time.sleep(0.3)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))

        # Send GNSS Fix
        gnss_msg = '{"type": "GNSS", "timestamp_s": 0.0, "latitude": 12.9716, "longitude": 77.5946, "speed_mps": 12.0, "heading_deg": 0.0}\n'
        client.sendall(gnss_msg.encode("utf-8"))

        # Send IMU Frame
        imu_msg = '{"type": "IMU", "timestamp_s": 0.1, "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}\n'
        client.sendall(imu_msg.encode("utf-8"))

        # Read back position estimate response
        client.settimeout(2.0)
        resp_data = client.recv(4096).decode("utf-8")
        client.close()
        stop_event.set()

        resp_json = json.loads(resp_data.strip())
        assert resp_json["status"] in ["GNSS_LOCKED", "INITIALIZING"]
        assert resp_json["speed_mps"] >= 0.0

    def test_udp_socket_server(self):
        engine = StandaloneEdgeEngine(fs_imu=10.0, ref_lat=12.9716, ref_lon=77.5946)
        stop_event = threading.Event()
        port = 19093

        server_thread = threading.Thread(
            target=engine.run_udp_server,
            kwargs={"host": "127.0.0.1", "port": port, "stop_event": stop_event},
            daemon=True
        )
        server_thread.start()
        time.sleep(0.3)

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2.0)

        # Send IMU datagram
        imu_msg = '{"type": "IMU", "timestamp_s": 0.1, "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}'
        client.sendto(imu_msg.encode("utf-8"), ("127.0.0.1", port))

        resp_data, _ = client.recvfrom(4096)
        client.close()
        stop_event.set()

        resp_json = json.loads(resp_data.decode("utf-8").strip())
        assert "latitude" in resp_json
        assert "engine_latency_ms" in resp_json
