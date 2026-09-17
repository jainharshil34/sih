"""
Standalone Edge & Telematics Server for NavResilient Engine.

Allows NavResilient to run 100% standalone on laptops, Raspberry Pi, NVIDIA Jetson,
and telematics edge boxes without any mobile application dependency.

Supports:
1. TCP Socket Server: accepts streaming JSON frames over TCP connection.
2. UDP Datagram Listener: high-frequency (10-200Hz) low-overhead telemetry listener.
3. Stdin/Stdout Line Pipe: UNIX-style sub-process stream for local apps and IPC.
4. File/Replay Mode: Replays arbitrary external IMU/GNSS recordings.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import threading
import time
from typing import Callable, Optional, TextIO

from navresilient.contract import DriftCorrectedState, GNSSFix, IMUFrame, SystemStatus
from navresilient.engine import NavResilientEngine


def parse_sensor_json(line: str) -> tuple[Optional[str], Optional[IMUFrame], Optional[GNSSFix]]:
    """Parse incoming JSON frame into typed IMUFrame or GNSSFix."""
    line = line.strip()
    if not line:
        return None, None, None

    try:
        data = json.loads(line)
    except Exception:
        return None, None, None

    msg_type = data.get("type", "").upper()

    # Auto-detect if type not explicitly supplied
    if not msg_type:
        if "ax" in data or "ax_mps2" in data:
            msg_type = "IMU"
        elif "lat" in data or "latitude" in data:
            msg_type = "GNSS"

    if msg_type == "IMU":
        frame = IMUFrame(
            timestamp_s=float(data.get("timestamp_s", data.get("timestamp", data.get("t", time.time())))),
            ax_mps2=float(data.get("ax_mps2", data.get("ax", 0.0))),
            ay_mps2=float(data.get("ay_mps2", data.get("ay", 0.0))),
            az_mps2=float(data.get("az_mps2", data.get("az", 9.81))),
            gx_rads=float(data.get("gx_rads", data.get("gx", data.get("wx", 0.0)))),
            gy_rads=float(data.get("gy_rads", data.get("gy", data.get("wy", 0.0)))),
            gz_rads=float(data.get("gz_rads", data.get("gz", data.get("wz", 0.0)))),
        )
        return "IMU", frame, None

    elif msg_type == "GNSS":
        fix = GNSSFix(
            timestamp_s=float(data.get("timestamp_s", data.get("timestamp", data.get("t", time.time())))),
            latitude=float(data.get("latitude", data.get("lat", 0.0))),
            longitude=float(data.get("longitude", data.get("lon", 0.0))),
            altitude_m=float(data.get("altitude_m", data.get("alt", 0.0))),
            speed_mps=float(data.get("speed_mps", data.get("speed", 0.0))),
            heading_deg=float(data.get("heading_deg", data.get("heading", data.get("course", 0.0)))),
            accuracy_m=float(data.get("accuracy_m", data.get("accuracy", 2.5))),
        )
        return "GNSS", None, fix

    return None, None, None


class StandaloneEdgeEngine:
    """Standalone wrapper around NavResilientEngine for Edge and Telematics hardware."""

    def __init__(
        self,
        fs_imu: float = 10.0,
        ref_lat: float = 12.9716,
        ref_lon: float = 77.5946,
        ref_alt: float = 920.0,
    ):
        self.engine = NavResilientEngine(
            fs_imu=fs_imu,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )

    def process_line(self, line: str) -> Optional[DriftCorrectedState]:
        """Process a single JSON string line."""
        msg_type, imu, gnss = parse_sensor_json(line)
        if msg_type == "GNSS" and gnss is not None:
            self.engine.push_gnss(gnss)
            return None
        elif msg_type == "IMU" and imu is not None:
            return self.engine.push_imu(imu)
        return None

    def run_stdio_pipe(self, in_stream: TextIO = sys.stdin, out_stream: TextIO = sys.stdout):
        """Standard UNIX Stdin/Stdout JSON-lines pipeline for sub-process IPC."""
        for line in in_stream:
            state = self.process_line(line)
            if state is not None:
                out_stream.write(state.to_json() + "\n")
                out_stream.flush()

    def run_tcp_server(self, host: str = "0.0.0.0", port: int = 9090, stop_event: Optional[threading.Event] = None):
        """TCP Socket server for remote/telematics hardware streaming."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)
        server.settimeout(0.5)

        print(f"[NavResilient Edge] TCP Server listening on {host}:{port}...")

        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                client_thread = threading.Thread(
                    target=self._handle_tcp_client,
                    args=(conn, addr, stop_event),
                    daemon=True,
                )
                client_thread.start()
        finally:
            server.close()

    def _handle_tcp_client(self, conn: socket.socket, addr: tuple, stop_event: Optional[threading.Event]):
        conn_file = conn.makefile("r", encoding="utf-8")
        try:
            for line in conn_file:
                if stop_event is not None and stop_event.is_set():
                    break
                state = self.process_line(line)
                if state is not None:
                    resp = (state.to_json() + "\n").encode("utf-8")
                    conn.sendall(resp)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()

    def run_udp_server(self, host: str = "0.0.0.0", port: int = 9091, stop_event: Optional[threading.Event] = None):
        """UDP Datagram listener for high-rate (up to 200Hz) embedded sensors."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(0.5)

        print(f"[NavResilient Edge] UDP Datagram Listener on {host}:{port}...")

        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                text = data.decode("utf-8", errors="ignore")
                state = self.process_line(text)
                if state is not None:
                    resp = (state.to_json() + "\n").encode("utf-8")
                    sock.sendto(resp, addr)
        finally:
            sock.close()
