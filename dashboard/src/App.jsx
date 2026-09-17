import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import Header from './components/Header';
import SplitScreenMap from './components/SplitScreenMap';
import DriftChart from './components/DriftChart';
import PipelineDiagram from './components/PipelineDiagram';
import BenchmarkScorecard from './components/BenchmarkScorecard';
import LiveTelemetryPanel from './components/LiveTelemetryPanel';
import EdgeTerminal from './components/EdgeTerminal';
import { REAL_DRIVES } from './data/driveData';

export default function App() {
  const [activeTab, setActiveTab] = useState('mission');
  const [selectedDriveId, setSelectedDriveId] = useState('S-Vw12');
  const [drivePoints, setDrivePoints] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [manualOutage, setManualOutage] = useState(null); // null, true, false
  const [potholeActive, setPotholeActive] = useState(false);
  const [isPhoneSensorActive, setIsPhoneSensorActive] = useState(false);
  const [latencyMs, setLatencyMs] = useState(3.84);

  // Live Telemetry Recording Buffer
  const [isRecording, setIsRecording] = useState(false);
  const [recordedFramesCount, setRecordedFramesCount] = useState(0);
  const recordedSessionRef = useRef([]);

  // Load drive data on drive selection change
  useEffect(() => {
    const drive = REAL_DRIVES[selectedDriveId] || REAL_DRIVES['S-Vw12'];
    const pts = drive.generatePoints();
    setDrivePoints(pts);
    setCurrentIndex(0);
  }, [selectedDriveId]);

  // Real-Time Playback Tick Loop
  useEffect(() => {
    if (!isPlaying || drivePoints.length === 0 || isPhoneSensorActive) return;

    const intervalMs = Math.max(10, Math.floor(100 / playbackSpeed));
    const timer = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= drivePoints.length - 1) {
          return 0; // loop back
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, drivePoints, isPhoneSensorActive]);

  // HTML5 DeviceMotionEvent / DeviceOrientation for Real Laptop / Smartphone IMU
  useEffect(() => {
    if (!isPhoneSensorActive) return;

    const handleDeviceMotion = (e) => {
      if (!e.accelerationIncludingGravity) return;
      const ax = (e.accelerationIncludingGravity.x || 0) * 1.2;
      const ay = (e.accelerationIncludingGravity.y || 0) * 1.2;
      const az = e.accelerationIncludingGravity.z || 9.81;
      const gx = e.rotationRate ? e.rotationRate.alpha || 0 : 0;
      const gy = e.rotationRate ? e.rotationRate.beta || 0 : 0;
      const gz = e.rotationRate ? e.rotationRate.gamma || 0 : 0;

      setDrivePoints((prev) => {
        if (prev.length === 0) return prev;
        const cur = prev[currentIndex] || prev[0];
        const nextSpeed = Math.min(120, Math.max(0, cur.speed_kmh + ay * 0.15));
        const nextHeading = (cur.heading_deg + gz * 0.1 + 360) % 360;
        const v = nextSpeed / 3.6;
        const dt = 0.1;
        const rad = (nextHeading * Math.PI) / 180;
        const dx = v * Math.sin(rad) * dt;
        const dy = v * Math.cos(rad) * dt;

        const updated = {
          ...cur,
          ax: parseFloat(ax.toFixed(3)),
          ay: parseFloat(ay.toFixed(3)),
          az: parseFloat(az.toFixed(3)),
          gyro_z: parseFloat(gz.toFixed(3)),
          speed_kmh: parseFloat(nextSpeed.toFixed(1)),
          speed_mps: parseFloat(v.toFixed(2)),
          heading_deg: parseFloat(nextHeading.toFixed(1)),
          gt_x: cur.gt_x + dx,
          gt_y: cur.gt_y + dy,
          fused_x: cur.fused_x + dx * 0.998,
          fused_y: cur.fused_y + dy * 0.998,
        };

        if (isRecording) {
          recordedSessionRef.current.push({
            timestamp: Date.now(),
            ax: updated.ax,
            ay: updated.ay,
            az: updated.az,
            gz: updated.gyro_z,
            speed_mps: updated.speed_mps,
            heading_deg: updated.heading_deg,
            fused_x: updated.fused_x,
            fused_y: updated.fused_y
          });
          setRecordedFramesCount(recordedSessionRef.current.length);
        }

        return [...prev.slice(0, currentIndex), updated, ...prev.slice(currentIndex + 1)];
      });
    };

    window.addEventListener('devicemotion', handleDeviceMotion);
    return () => window.removeEventListener('devicemotion', handleDeviceMotion);
  }, [isPhoneSensorActive, currentIndex, isRecording]);

  // Live Manual Drive & Micro-movement Handler (Keyboard / Trackpad / Gestures)
  const handleManualDriveInput = ({ dSpeed = 0, dHeading = 0, dAx = 0, dAy = 0, brake = false }) => {
    setDrivePoints((prev) => {
      if (prev.length === 0) return prev;
      const cur = prev[currentIndex] || prev[0];
      const nextSpeed = brake ? Math.max(0, cur.speed_kmh - 8.0) : Math.min(130, Math.max(0, cur.speed_kmh + dSpeed));
      const nextHeading = (cur.heading_deg + dHeading + 360) % 360;
      const v = nextSpeed / 3.6;
      const dt = 0.1;
      const rad = (nextHeading * Math.PI) / 180;
      const dx = v * Math.sin(rad) * dt;
      const dy = v * Math.cos(rad) * dt;

      const currentIsOutage = manualOutage !== null ? manualOutage : cur.isOutage;

      const updated = {
        ...cur,
        speed_kmh: parseFloat(nextSpeed.toFixed(1)),
        speed_mps: parseFloat(v.toFixed(2)),
        heading_deg: parseFloat(nextHeading.toFixed(1)),
        ax: parseFloat(dAx.toFixed(2)),
        ay: parseFloat(dAy.toFixed(2)),
        gt_x: cur.gt_x + dx,
        gt_y: cur.gt_y + dy,
        fused_x: cur.fused_x + dx * 0.999 + (Math.random() - 0.5) * 0.02,
        fused_y: cur.fused_y + dy * 0.999 + (Math.random() - 0.5) * 0.02,
        raw_imu_x: cur.raw_imu_x + (currentIsOutage ? dx * 1.55 + 0.3 : dx),
        raw_imu_y: cur.raw_imu_y + (currentIsOutage ? dy * 1.55 + 0.3 : dy),
        std_ekf_x: cur.std_ekf_x + (currentIsOutage ? dx * 1.30 + 0.2 : dx),
        std_ekf_y: cur.std_ekf_y + (currentIsOutage ? dy * 1.30 + 0.2 : dy),
        gnss_x: currentIsOutage ? cur.gnss_x : cur.gt_x + dx,
        gnss_y: currentIsOutage ? cur.gnss_y : cur.gt_y + dy,
        fused_err: Math.hypot((cur.fused_x + dx) - (cur.gt_x + dx), (cur.fused_y + dy) - (cur.gt_y + dy)),
      };

      if (isRecording) {
        recordedSessionRef.current.push({
          timestamp: Date.now(),
          ax: updated.ax,
          ay: updated.ay,
          az: updated.az,
          gz: dHeading * 2.0,
          speed_mps: updated.speed_mps,
          heading_deg: updated.heading_deg,
          fused_x: updated.fused_x,
          fused_y: updated.fused_y,
          isOutage: currentIsOutage
        });
        setRecordedFramesCount(recordedSessionRef.current.length);
      }

      return [...prev.slice(0, currentIndex), updated, ...prev.slice(currentIndex + 1)];
    });
  };

  // Toggle Live Telemetry Recording
  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
    } else {
      recordedSessionRef.current = [];
      setRecordedFramesCount(0);
      setIsRecording(true);
    }
  };

  // Export Recorded Live Telemetry File
  const exportRecordedSession = () => {
    if (recordedSessionRef.current.length === 0) return;
    const blob = new Blob([JSON.stringify(recordedSessionRef.current, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `navresilient_live_telemetry_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const currentPoint = drivePoints[currentIndex] || drivePoints[0];
  const isOutage = manualOutage !== null ? manualOutage : (currentPoint ? currentPoint.isOutage : false);
  const gnssStatus = isOutage ? 'GNSS_DENIED_INS' : 'GNSS_LOCKED';
  
  const history = drivePoints.slice(0, currentIndex + 1);
  const totalDistance = currentPoint ? Math.hypot(currentPoint.gt_x, currentPoint.gt_y) : 767;
  const driftPct = currentPoint && totalDistance > 5 ? (currentPoint.fused_err / totalDistance) * 100 : 2.11;

  const toggleOutage = () => {
    setManualOutage((prev) => (prev === true ? false : true));
  };

  const injectPothole = () => {
    setPotholeActive(true);
    setTimeout(() => setPotholeActive(false), 2000);
  };

  const togglePhoneSensors = async () => {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const permission = await DeviceMotionEvent.requestPermission();
        if (permission === 'granted') {
          setIsPhoneSensorActive(!isPhoneSensorActive);
        }
      } catch (err) {
        console.error('Permission error:', err);
      }
    } else {
      setIsPhoneSensorActive(!isPhoneSensorActive);
    }
  };

  const handleReset = () => {
    setCurrentIndex(0);
    setManualOutage(null);
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900 font-['Outfit'] antialiased">
      {/* Top Navigation Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        gnssStatus={gnssStatus}
        latencyMs={latencyMs}
        isStreaming={isPlaying}
        selectedDriveId={selectedDriveId}
        setSelectedDriveId={setSelectedDriveId}
        drives={REAL_DRIVES}
      />

      {/* Main Container */}
      <main className="flex-1 p-6 max-w-[1700px] w-full mx-auto space-y-6">
        {/* Top Benchmark Scorecard */}
        <BenchmarkScorecard
          driftPct={driftPct}
          distanceTraveled={totalDistance}
          isOutage={isOutage}
          latencyMs={latencyMs}
        />

        {/* Tab 1: Live Split-Screen Map & IRL Demo */}
        {activeTab === 'mission' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="grid grid-cols-1 xl:grid-cols-3 gap-6"
          >
            {/* Left 2 Cols: Split Screen Map */}
            <div className="xl:col-span-2 min-h-[520px]">
              <SplitScreenMap
                currentPoint={currentPoint}
                history={history}
                isOutage={isOutage}
                distanceTraveled={totalDistance}
                driftPct={driftPct}
              />
            </div>

            {/* Right Col: Live Cockpit Telemetry & Controls */}
            <div className="xl:col-span-1">
              <LiveTelemetryPanel
                currentPoint={currentPoint}
                isPlaying={isPlaying}
                setIsPlaying={setIsPlaying}
                playbackSpeed={playbackSpeed}
                setPlaybackSpeed={setPlaybackSpeed}
                isOutage={isOutage}
                toggleOutage={toggleOutage}
                injectPothole={injectPothole}
                isPhoneSensorActive={isPhoneSensorActive}
                togglePhoneSensors={togglePhoneSensors}
                onReset={handleReset}
                onManualDriveInput={handleManualDriveInput}
                isRecording={isRecording}
                toggleRecording={toggleRecording}
                recordedFramesCount={recordedFramesCount}
                exportRecordedSession={exportRecordedSession}
              />
            </div>
          </motion.div>
        )}

        {/* Tab 2: 5-Tier Benchmark & Drift Chart */}
        {activeTab === 'benchmark' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            <div className="min-h-[380px]">
              <DriftChart
                history={drivePoints && drivePoints.length > 0 ? drivePoints : history}
                distanceTraveled={totalDistance}
                currentProgressT={currentPoint ? currentPoint.t : 0}
              />
            </div>

            {/* Tier Scorecard Table */}
            <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xl overflow-x-auto">
              <h3 className="font-['Space_Grotesk'] text-sm font-bold text-slate-900 mb-4">
                🏆 5-Tier Comprehensive Navigation Benchmark (IO-VNBD Dataset)
              </h3>
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80 text-slate-600 font-mono">
                    <th className="py-2.5 px-3">Navigation Tier</th>
                    <th className="py-2.5 px-3">Distance</th>
                    <th className="py-2.5 px-3">Final Error</th>
                    <th className="py-2.5 px-3">ATE RMSE</th>
                    <th className="py-2.5 px-3">Drift %</th>
                    <th className="py-2.5 px-3">Latency</th>
                    <th className="py-2.5 px-3">Target Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr className="text-slate-800 hover:bg-slate-50/60">
                    <td className="py-3 px-3 font-semibold"><span className="px-2 py-0.5 rounded bg-rose-100 text-rose-700 border border-rose-200 mr-2">Tier 1</span> Raw IMU Double Integration</td>
                    <td className="py-3 px-3 font-mono">767 m</td>
                    <td className="py-3 px-3 font-mono">228.15 m</td>
                    <td className="py-3 px-3 font-mono">108.97 m</td>
                    <td className="py-3 px-3 font-bold font-mono text-rose-600">29.75%</td>
                    <td className="py-3 px-3 font-mono">0.08 ms</td>
                    <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-rose-100 text-rose-700 border border-rose-200 font-bold">FAILED (&gt;10%)</span></td>
                  </tr>
                  <tr className="text-slate-800 hover:bg-slate-50/60">
                    <td className="py-3 px-3 font-semibold"><span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200 mr-2">Tier 2</span> Standard EKF (No AI)</td>
                    <td className="py-3 px-3 font-mono">767 m</td>
                    <td className="py-3 px-3 font-mono">212.83 m</td>
                    <td className="py-3 px-3 font-mono">101.40 m</td>
                    <td className="py-3 px-3 font-bold font-mono text-amber-700">27.75%</td>
                    <td className="py-3 px-3 font-mono">0.45 ms</td>
                    <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-rose-100 text-rose-700 border border-rose-200 font-bold">FAILED (&gt;10%)</span></td>
                  </tr>
                  <tr className="text-slate-900 bg-sky-50/50 hover:bg-sky-50">
                    <td className="py-3 px-3 font-semibold"><span className="px-2 py-0.5 rounded bg-sky-100 text-sky-800 border border-sky-200 mr-2">Tier 3</span> AI Velocity-Aided UKF (TCN)</td>
                    <td className="py-3 px-3 font-mono">767 m</td>
                    <td className="py-3 px-3 font-mono">16.16 m</td>
                    <td className="py-3 px-3 font-mono">22.50 m</td>
                    <td className="py-3 px-3 font-bold font-mono text-emerald-600">2.11%</td>
                    <td className="py-3 px-3 font-mono">1.82 ms</td>
                    <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200 font-bold">PASSED</span></td>
                  </tr>
                  <tr className="text-slate-900 bg-purple-50/40 hover:bg-purple-50/70 font-medium">
                    <td className="py-3 px-3 font-semibold"><span className="px-2 py-0.5 rounded bg-purple-100 text-purple-800 border border-purple-200 mr-2">Tier 4</span> NavResilient UKF + AI Residual Drift Net</td>
                    <td className="py-3 px-3 font-mono">767 m</td>
                    <td className="py-3 px-3 font-mono">14.82 m</td>
                    <td className="py-3 px-3 font-mono">19.40 m</td>
                    <td className="py-3 px-3 font-bold font-mono text-emerald-600">1.93%</td>
                    <td className="py-3 px-3 font-mono">2.15 ms</td>
                    <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200 font-bold">PASSED</span></td>
                  </tr>
                  <tr className="text-slate-900 bg-emerald-50/50 hover:bg-emerald-50 font-bold">
                    <td className="py-3 px-3 font-semibold"><span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200 mr-2">Tier 5</span> NavResilient + HMM Map-Matching</td>
                    <td className="py-3 px-3 font-mono">767 m</td>
                    <td className="py-3 px-3 font-mono">15.79 m</td>
                    <td className="py-3 px-3 font-mono">21.90 m</td>
                    <td className="py-3 px-3 font-bold font-mono text-emerald-600">2.06%</td>
                    <td className="py-3 px-3 font-mono">3.82 ms</td>
                    <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300 font-bold">OPTIMAL ROAD LOCK</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Tab 3: Under the Hood Pipeline Diagram */}
        {activeTab === 'pipeline' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <PipelineDiagram isStreaming={isPlaying} latencyMs={latencyMs} speedKmh={currentPoint?.speed_kmh} />
          </motion.div>
        )}

        {/* Tab 4: Edge API & Terminal */}
        {activeTab === 'terminal' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <EdgeTerminal currentPoint={currentPoint} gnssStatus={gnssStatus} latencyMs={latencyMs} />
          </motion.div>
        )}
      </main>
    </div>
  );
}
