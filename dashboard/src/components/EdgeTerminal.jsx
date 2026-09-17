import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Terminal, Copy, Check, Code, Radio } from 'lucide-react';

export default function EdgeTerminal({ currentPoint, gnssStatus, latencyMs }) {
  const [activeLang, setActiveLang] = useState('python');
  const [copied, setCopied] = useState(false);

  const snippets = {
    python: `# NavResilient Python Edge Telematics Client (Socket IPC)
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9090))
sock_file = sock.makefile("r", encoding="utf-8")

# Push 10-200 Hz IMU Frame
imu_frame = {
    "type": "IMU", "timestamp_s": 125.40,
    "ax_mps2": 0.12, "ay_mps2": -0.04, "az_mps2": 9.81,
    "gx_rads": 0.001, "gy_rads": -0.002, "gz_rads": 0.045
}
sock.sendall((json.dumps(imu_frame) + "\\n").encode("utf-8"))

# Receive Real-Time DriftCorrectedState
state_json = sock_file.readline()
state = json.loads(state_json)
print(f"Position: {state['latitude']}, {state['longitude']} (Drift: {state['drift_pct']}%)")`,

    flutter: `// Flutter / Dart Subprocess Pipe for Mobile App (Engineer B)
import 'dart:convert';
import 'dart:io';

void startNavResilient() async {
  final process = await Process.start('python', ['-m', 'navresilient.cli', '--mode', 'stdio']);

  // Listen for real-time PositionEstimates
  process.stdout
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) {
    if (line.isNotEmpty) {
      final state = jsonDecode(line);
      print("UI Map Marker -> \${state['latitude']}, \${state['longitude']}");
    }
  });

  // Stream raw phone IMU
  final imuPacket = jsonEncode({
    "type": "IMU", "timestamp_s": DateTime.now().millisecondsSinceEpoch / 1000.0,
    "ax_mps2": 0.12, "ay_mps2": 0.0, "az_mps2": 9.81,
    "gx_rads": 0.0, "gy_rads": 0.0, "gz_rads": 0.0
  });
  process.stdin.writeln(imuPacket);
}`,

    cpp: `// C++ POSIX TCP Telematics Client (Raspberry Pi / Jetson)
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

    const char *imu_json = "{\\"type\\":\\"IMU\\",\\"ax\\":0.12,\\"ay\\":0.0,\\"az\\":9.81,\\"gx\\":0,\\"gy\\":0,\\"gz\\":0}\\n";
    send(sock, imu_json, strlen(imu_json), 0);

    char buffer[2048] = {0};
    read(sock, buffer, sizeof(buffer) - 1);
    printf("Received DriftCorrectedState: %s\\n", buffer);

    close(sock);
    return 0;
}`
  };

  const copyCode = () => {
    navigator.clipboard.writeText(snippets[activeLang]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sampleJsonOutput = JSON.stringify(
    {
      timestamp_s: currentPoint ? currentPoint.t : 125.4,
      latitude: 52.4068 + ((currentPoint ? currentPoint.fused_y : 0) / 111320),
      longitude: -1.5065 + ((currentPoint ? currentPoint.fused_x : 0) / 67800),
      altitude_m: 920.0,
      speed_mps: currentPoint ? Number(currentPoint.speed_mps.toFixed(2)) : 14.48,
      speed_kmh: currentPoint ? Number(currentPoint.speed_kmh.toFixed(1)) : 52.13,
      heading_deg: currentPoint ? Number(currentPoint.heading_deg.toFixed(1)) : 182.35,
      status: gnssStatus,
      drift_pct: currentPoint ? Number(((currentPoint.fused_err / (currentPoint.gt_x + 1)) * 100).toFixed(2)) : 2.11,
      pos_uncertainty_1sigma_m: 1.25,
      is_stationary: currentPoint?.speed_mps < 0.2,
      pothole_detected: !!currentPoint?.pothole,
      engine_latency_ms: latencyMs
    },
    null,
    2
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 rounded-2xl bg-white border border-slate-200 shadow-xl">
      {/* Left: Live Packet Stream Terminal */}
      <div className="flex flex-col rounded-xl bg-slate-900 border border-slate-800 overflow-hidden shadow-lg">
        <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-sky-400" />
            <span className="font-['Space_Grotesk'] text-xs font-bold text-slate-100 uppercase tracking-wider">
              Live JSON Telemetry Stream (stdout / TCP)
            </span>
          </div>
          <span className="flex items-center gap-1 text-[10px] font-mono text-emerald-400 font-bold bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/30">
            <Radio className="w-3 h-3 animate-pulse" /> 10 Hz LIVE
          </span>
        </div>

        <div className="p-4 font-mono text-xs text-sky-300 overflow-y-auto max-h-[380px] leading-relaxed select-all bg-slate-900/90">
          <pre>{sampleJsonOutput}</pre>
        </div>
      </div>

      {/* Right: Copyable Code Snippets */}
      <div className="flex flex-col rounded-xl bg-slate-900 border border-slate-800 overflow-hidden shadow-lg">
        <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Code className="w-4 h-4 text-sky-400" />
            <span className="font-['Space_Grotesk'] text-xs font-bold text-slate-100 uppercase tracking-wider">
              Client Integration Snippets (Engineer B)
            </span>
          </div>

          <button
            onClick={copyCode}
            className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold transition-all cursor-pointer shadow-xs"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy Code'}</span>
          </button>
        </div>

        {/* Code Tabs */}
        <div className="flex items-center gap-2 px-4 pt-3 bg-slate-950 border-b border-slate-800">
          {[
            { id: 'python', label: 'Python (Socket)' },
            { id: 'flutter', label: 'Flutter / Dart (Pipe)' },
            { id: 'cpp', label: 'C++ (POSIX TCP)' }
          ].map((lang) => (
            <button
              key={lang.id}
              onClick={() => setActiveLang(lang.id)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-t-lg transition-colors cursor-pointer ${
                activeLang === lang.id
                  ? 'bg-slate-900 text-sky-400 border-t-2 border-sky-400 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {lang.label}
            </button>
          ))}
        </div>

        <div className="p-4 font-mono text-xs text-slate-200 overflow-y-auto max-h-[330px] leading-relaxed bg-slate-900/90">
          <pre>{snippets[activeLang]}</pre>
        </div>
      </div>
    </div>
  );
}
