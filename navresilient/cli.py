"""
Command-Line Interface (CLI) for Standalone Edge Deployment.

Allows running the NavResilient engine in:
  1. Stdio line-streaming mode (for stdin/stdout pipes and subprocess IPC)
  2. TCP socket daemon mode
  3. UDP datagram daemon mode
  4. CSV file replay mode
"""

from __future__ import annotations

import argparse
import sys
import time

from navresilient.contract import GNSSFix, IMUFrame
from navresilient.io_vnbd_loader import load_smartphone_drive
from navresilient.server import StandaloneEdgeEngine
from navresilient.simulate_gnss_outage import GNSSOutageSimulator


def main():
    parser = argparse.ArgumentParser(
        description="NavResilient Edge Engine CLI & Telematics Streaming Daemon"
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "tcp", "udp", "replay"],
        default="stdio",
        help="Streaming mode: 'stdio' (stdin/stdout pipe), 'tcp' (socket server), 'udp' (datagram), or 'replay' (CSV file)"
    )
    parser.add_argument("--port", type=int, default=9090, help="Port for TCP/UDP server (default: 9090)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind server (default: 0.0.0.0)")
    parser.add_argument("--replay-csv", help="CSV path for replay mode")
    parser.add_argument("--realtime", action="store_true", help="Throttle replay to 1x real-time speed")
    parser.add_argument("--ref-lat", type=float, default=12.9716, help="Reference origin latitude")
    parser.add_argument("--ref-lon", type=float, default=77.5946, help="Reference origin longitude")
    parser.add_argument("--fs", type=float, default=10.0, help="IMU sampling frequency in Hz")

    args = parser.parse_args()

    engine = StandaloneEdgeEngine(
        fs_imu=args.fs,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
    )

    if args.mode == "stdio":
        engine.run_stdio_pipe(sys.stdin, sys.stdout)

    elif args.mode == "tcp":
        engine.run_tcp_server(host=args.host, port=args.port)

    elif args.mode == "udp":
        engine.run_udp_server(host=args.host, port=args.port)

    elif args.mode == "replay":
        if not args.replay_csv:
            print("Error: --replay-csv required for replay mode", file=sys.stderr)
            sys.exit(1)

        drive = load_smartphone_drive(args.replay_csv)
        sim = GNSSOutageSimulator(drive, outage_start_s=30.0, outage_duration_s=60.0)
        t_prev = 0.0

        for imu_frame, gnss_fix, is_outage in sim.stream_sensors():
            if args.realtime and t_prev > 0:
                dt = max(0.0, imu_frame.timestamp_s - t_prev)
                time.sleep(dt)
            t_prev = imu_frame.timestamp_s

            if gnss_fix is not None:
                engine.engine.push_gnss(gnss_fix)

            state = engine.engine.push_imu(imu_frame)
            sys.stdout.write(state.to_json() + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
