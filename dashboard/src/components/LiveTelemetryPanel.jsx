import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import {
  ShieldAlert,
  Zap,
  RotateCcw,
  Smartphone,
  Play,
  Pause,
  Gamepad2,
  Mic,
  MousePointer,
  Radio,
  Download,
  Activity
} from 'lucide-react';

export default function LiveTelemetryPanel({
  currentPoint,
  isPlaying,
  setIsPlaying,
  playbackSpeed,
  setPlaybackSpeed,
  isOutage,
  toggleOutage,
  injectPothole,
  isPhoneSensorActive,
  togglePhoneSensors,
  onReset,
  onManualDriveInput,
  isRecording = false,
  toggleRecording,
  recordedFramesCount = 0,
  exportRecordedSession
}) {
  const [isKeyboardDrive, setIsKeyboardDrive] = useState(false);
  const [isTrackpadActive, setIsTrackpadActive] = useState(false);
  const [isMicVibeActive, setIsMicVibeActive] = useState(false);
  const [micLevel, setMicLevel] = useState(0);

  const speedKmh = currentPoint ? currentPoint.speed_kmh : 0;
  const speedMps = currentPoint ? currentPoint.speed_mps : 0;
  const ax = currentPoint ? currentPoint.ax : 0;
  const ay = currentPoint ? currentPoint.ay : 0;
  const az = currentPoint ? currentPoint.az : 9.81;
  const headingDeg = currentPoint ? currentPoint.heading_deg : 0;

  // Speedometer fill arc calculation (0 to 120 km/h maps to 0 to 210 dashoffset)
  const maxSpeed = 100;
  const speedFraction = Math.min(1, Math.max(0, speedKmh / maxSpeed));
  const dashOffset = 210 - speedFraction * 140;

  // 1. Keyboard Live Drive Controls (Arrow Keys / WASD)
  useEffect(() => {
    if (!isKeyboardDrive) return;

    const handleKeyDown = (e) => {
      if (['ArrowUp', 'KeyW'].includes(e.code)) {
        onManualDriveInput && onManualDriveInput({ dSpeed: 2.5, dHeading: 0, dAx: 1.8 });
      } else if (['ArrowDown', 'KeyS'].includes(e.code)) {
        onManualDriveInput && onManualDriveInput({ dSpeed: -3.0, dHeading: 0, dAx: -2.2 });
      } else if (['ArrowLeft', 'KeyA'].includes(e.code)) {
        onManualDriveInput && onManualDriveInput({ dSpeed: 0, dHeading: -4.5, dAy: -1.5 });
      } else if (['ArrowRight', 'KeyD'].includes(e.code)) {
        onManualDriveInput && onManualDriveInput({ dSpeed: 0, dHeading: 4.5, dAy: 1.5 });
      } else if (e.code === 'Space') {
        onManualDriveInput && onManualDriveInput({ dSpeed: -6.0, dHeading: 0, dAx: -4.5, brake: true });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isKeyboardDrive, onManualDriveInput]);

  // 2. Trackpad / Mouse Micro-Movement Listener (Converts subtle laptop gesture deltas to physical forces)
  useEffect(() => {
    if (!isTrackpadActive) return;

    let lastX = null;
    let lastY = null;

    const handleMouseMove = (e) => {
      if (lastX !== null && lastY !== null) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        const dAy = Math.min(3.0, Math.max(-3.0, dx * 0.15));
        const dAx = Math.min(3.0, Math.max(-3.0, -dy * 0.15));
        const dHeading = Math.min(6.0, Math.max(-6.0, dx * 0.25));
        const dSpeed = Math.min(2.0, Math.max(-2.0, -dy * 0.1));

        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          onManualDriveInput && onManualDriveInput({ dSpeed, dHeading, dAx, dAy });
        }
      }
      lastX = e.clientX;
      lastY = e.clientY;
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [isTrackpadActive, onManualDriveInput]);

  // 3. Microphone Table-Tap Vibration Listener (Web Audio API)
  const audioCtxRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameRef = useRef(null);

  const toggleMicVibrationDetector = async () => {
    if (isMicVibeActive) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close();
      }
      setIsMicVibeActive(false);
      setMicLevel(0);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      await audioCtx.resume();

      const analyser = audioCtx.createAnalyser();
      const microphone = audioCtx.createMediaStreamSource(stream);
      microphone.connect(analyser);
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.2;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      setIsMicVibeActive(true);

      let lastPotholeTime = 0;

      const checkVibration = () => {
        if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') return;
        analyser.getByteFrequencyData(dataArray);

        // Calculate average energy and peak frequency
        let sum = 0;
        let peak = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
          if (dataArray[i] > peak) peak = dataArray[i];
        }
        const avg = sum / bufferLength;
        const currentLevel = Math.min(100, Math.round((peak / 255) * 100));
        setMicLevel(currentLevel);

        const now = Date.now();
        // Trigger pothole if loud acoustic transient knock detected (> 35 level and sharp spike)
        if ((peak > 110 || (peak > 45 && peak > avg * 2.2)) && now - lastPotholeTime > 1200) {
          lastPotholeTime = now;
          injectPothole();
        }

        animFrameRef.current = requestAnimationFrame(checkVibration);
      };

      checkVibration();
    } catch (err) {
      console.warn('Microphone permission or hardware error:', err);
      // Fallback: trigger pothole once to show filter response
      injectPothole();
    }
  };

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close();
      }
    };
  }, []);

  return (
    <div className="flex flex-col gap-4">
      {/* 1. Cockpit Instrument Cluster */}
      <div className="flex flex-col p-5 rounded-2xl bg-white border border-slate-200 shadow-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-['Space_Grotesk'] text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-sky-600" />
            Live Cockpit Instrument Cluster
          </h3>
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${currentPoint?.pothole ? 'bg-rose-100 text-rose-700 border border-rose-300 animate-pulse' : 'bg-slate-100 text-slate-600'}`}>
            {currentPoint?.pothole ? '⚡ POTHOLE SHOCK (>30 m/s²)' : 'ROAD: CLEAR'}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {/* Speedometer */}
          <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-slate-50 border border-slate-200 relative">
            <span className="text-[10px] font-bold text-slate-500 mb-1">FORWARD SPEED</span>
            <div className="relative w-24 h-20 flex items-center justify-center">
              <svg className="w-full h-full" viewBox="0 0 120 120">
                <path
                  d="M 20 100 A 50 50 0 1 1 100 100"
                  fill="none"
                  stroke="#e2e8f0"
                  strokeWidth="10"
                  strokeLinecap="round"
                />
                <path
                  d="M 20 100 A 50 50 0 1 1 100 100"
                  fill="none"
                  stroke="#0284c7"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray="210"
                  strokeDashoffset={dashOffset}
                  className="transition-all duration-150"
                />
              </svg>
              <div className="absolute top-7 flex flex-col items-center">
                <span className="font-['Space_Grotesk'] text-2xl font-extrabold text-sky-700 leading-none">
                  {speedKmh.toFixed(0)}
                </span>
                <span className="text-[10px] font-medium text-slate-500">km/h</span>
              </div>
            </div>
            <span className="text-[10px] font-mono text-slate-600 font-medium">{speedMps.toFixed(1)} m/s</span>
          </div>

          {/* Artificial Horizon / Phone Attitude */}
          <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[10px] font-bold text-slate-500 mb-1">MOUNT ATTITUDE</span>
            <div className="relative w-18 h-18 rounded-full border-2 border-slate-300 overflow-hidden shadow-inner flex items-center justify-center">
              <div
                className="absolute inset-0 transition-transform duration-100"
                style={{
                  transform: `rotate(${-(headingDeg % 45)}deg) translateY(${-ax * 2}px)`
                }}
              >
                <div className="w-full h-1/2 bg-sky-500" />
                <div className="w-full h-1/2 bg-amber-700" />
                <div className="absolute top-1/2 w-full h-[2px] bg-white -translate-y-1/2" />
              </div>
              <div className="relative z-10 w-6 h-6 rounded-full border border-white" />
            </div>
            <span className="text-[10px] font-mono text-slate-600 font-medium mt-1">HDG: {headingDeg.toFixed(0)}°</span>
          </div>

          {/* G-Force Ball */}
          <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-slate-50 border border-slate-200 col-span-2 sm:col-span-1">
            <span className="text-[10px] font-bold text-slate-500 mb-1">G-FORCE ACCEL</span>
            <div className="relative w-18 h-18 rounded-full border border-dashed border-slate-300 flex items-center justify-center">
              <div className="w-8 h-8 rounded-full border border-slate-200" />
              <div
                className="absolute w-3.5 h-3.5 rounded-full bg-sky-600 shadow-md shadow-sky-600/30 transition-transform duration-75"
                style={{
                  transform: `translate(${Math.max(-25, Math.min(25, ay * 6))}px, ${Math.max(-25, Math.min(25, -ax * 6))}px)`
                }}
              />
            </div>
            <span className="text-[10px] font-mono text-slate-600 font-medium mt-1">Ax: {ax.toFixed(1)} m/s²</span>
          </div>
        </div>
      </div>

      {/* 2. Live Laptop Moment & Outage Controls */}
      <div className="flex flex-col p-5 rounded-2xl bg-white border border-slate-200 shadow-md space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-['Space_Grotesk'] text-xs font-bold uppercase tracking-wider text-slate-800">
            ⚡ Live Motion &amp; Blackout Controls
          </h3>
          {isRecording && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-100 text-rose-700 border border-rose-300 text-[10px] font-mono font-bold animate-pulse">
              <span className="w-2 h-2 rounded-full bg-rose-600" />
              REC: {recordedFramesCount} frames
            </span>
          )}
        </div>

        {/* Primary Row: Outage & Pothole Injection */}
        <div className="grid grid-cols-2 gap-2.5">
          <button
            onClick={toggleOutage}
            className={`flex items-center justify-center gap-2 p-3 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isOutage
                ? 'bg-rose-600 text-white shadow-md shadow-rose-600/30 animate-pulse'
                : 'bg-rose-50 text-rose-700 border border-rose-300 hover:bg-rose-100'
            }`}
          >
            <ShieldAlert className="w-4 h-4" />
            <span>{isOutage ? 'Tunnel Outage ACTIVE' : 'Trigger Tunnel Outage'}</span>
          </button>

          <button
            onClick={injectPothole}
            className="flex items-center justify-center gap-2 p-3 rounded-xl bg-slate-100 text-slate-800 border border-slate-200 hover:bg-slate-200 font-bold text-xs transition-all cursor-pointer shadow-xs"
          >
            <Zap className="w-4 h-4 text-amber-600" />
            <span>Inject Pothole</span>
          </button>
        </div>

        {/* Second Row: Arrow Keys & Trackpad Steer */}
        <div className="grid grid-cols-2 gap-2.5">
          <button
            onClick={() => setIsKeyboardDrive(!isKeyboardDrive)}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isKeyboardDrive
                ? 'bg-sky-600 text-white shadow-md shadow-sky-600/30 ring-2 ring-sky-300'
                : 'bg-slate-100 text-slate-700 border border-slate-200 hover:bg-slate-200'
            }`}
          >
            <Gamepad2 className="w-4 h-4" />
            <span>{isKeyboardDrive ? '🎮 Keys Driving ON' : '🎮 Steer with Arrow Keys'}</span>
          </button>

          <button
            onClick={() => setIsTrackpadActive(!isTrackpadActive)}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isTrackpadActive
                ? 'bg-purple-600 text-white shadow-md shadow-purple-600/30 ring-2 ring-purple-300'
                : 'bg-slate-100 text-slate-700 border border-slate-200 hover:bg-slate-200'
            }`}
          >
            <MousePointer className="w-4 h-4" />
            <span>{isTrackpadActive ? '🖱️ Trackpad Motion ON' : '🖱️ Trackpad Micro-Moments'}</span>
          </button>
        </div>

        {/* Third Row: Table Tap Mic Detector & Hardware IMU */}
        <div className="grid grid-cols-2 gap-2.5">
          <button
            onClick={toggleMicVibrationDetector}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isMicVibeActive
                ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30'
                : 'bg-slate-100 text-slate-700 border border-slate-200 hover:bg-slate-200'
            }`}
          >
            <Mic className="w-4 h-4" />
            <span>{isMicVibeActive ? `🎙️ Mic Tap (${micLevel}%)` : '🎙️ Tap Table (Mic Shock)'}</span>
          </button>

          <button
            onClick={togglePhoneSensors}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isPhoneSensorActive
                ? 'bg-sky-600 text-white animate-pulse shadow-md shadow-sky-600/40'
                : 'bg-slate-100 text-slate-700 border border-slate-200 hover:bg-slate-200'
            }`}
          >
            <Smartphone className="w-4 h-4" />
            <span>{isPhoneSensorActive ? '📱 Device IMU ON' : '📱 Device IMU Mode'}</span>
          </button>
        </div>

        {/* Fourth Row: Live Session Recording & Export */}
        <div className="grid grid-cols-2 gap-2.5 pt-1">
          <button
            onClick={toggleRecording}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              isRecording
                ? 'bg-rose-600 text-white shadow-md shadow-rose-600/30 ring-2 ring-rose-300 animate-pulse'
                : 'bg-rose-50 text-rose-700 border border-rose-300 hover:bg-rose-100'
            }`}
          >
            <Radio className="w-4 h-4" />
            <span>{isRecording ? '⏹️ Stop Recording' : '🔴 Record Live Telemetry'}</span>
          </button>

          <button
            onClick={exportRecordedSession}
            disabled={recordedFramesCount === 0}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl font-bold text-xs transition-all cursor-pointer ${
              recordedFramesCount > 0
                ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30 hover:bg-emerald-500'
                : 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
            }`}
          >
            <Download className="w-4 h-4" />
            <span>Download Telemetry JSON</span>
          </button>
        </div>
      </div>

      {/* 3. Playback Bar */}
      <div className="flex items-center justify-between p-3.5 rounded-xl bg-white border border-slate-200 shadow-sm">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center justify-center w-8 h-8 rounded-lg bg-sky-600 text-white hover:bg-sky-500 transition-colors cursor-pointer shadow-xs"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
          <button
            onClick={onReset}
            className="flex items-center justify-center w-8 h-8 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200 transition-colors cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
          {[1, 2, 5].map((speed) => (
            <button
              key={speed}
              onClick={() => setPlaybackSpeed(speed)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-colors cursor-pointer ${
                playbackSpeed === speed ? 'bg-sky-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
