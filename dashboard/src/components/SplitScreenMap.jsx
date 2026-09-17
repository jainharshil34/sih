import React, { useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { ShieldAlert, Zap, Compass, AlertTriangle, CheckCircle2, MapPin, Navigation } from 'lucide-react';

export default function SplitScreenMap({ currentPoint, history, isOutage, distanceTraveled, driftPct }) {
  const canvasLeftRef = useRef(null);
  const canvasRightRef = useRef(null);

  useEffect(() => {
    if (!history || history.length === 0) return;

    // Render Canvas Left (Raw GNSS Only)
    renderTrajectory(canvasLeftRef.current, {
      history,
      currentPoint,
      isRawOnly: true,
      trackKeyX: 'raw_x',
      trackKeyY: 'raw_y',
      trailColor: '#ef4444',
      vehicleColor: '#f87171',
      title: 'Raw GNSS-Only Track (Unaugmented)',
      subtitle: isOutage ? '⚠️ GNSS BLACKOUT ACTIVE · FROZEN FIX' : '🛰️ GNSS 1 Hz FIX ACQUIRED'
    });

    // Render Canvas Right (NavResilient AI-Fused)
    renderTrajectory(canvasRightRef.current, {
      history,
      currentPoint,
      isRawOnly: false,
      trackKeyX: 'fused_x',
      trackKeyY: 'fused_y',
      trailColor: '#0284c7',
      snappedKeyX: 'snapped_x',
      snappedKeyY: 'snapped_y',
      vehicleColor: '#38bdf8',
      title: 'NavResilient AI-Fused Track (TCN + UKF + HMM)',
      subtitle: isOutage ? '🛡️ AI DEAD RECKONING ACTIVE · DRIFT < 2.1%' : '🛰️ CONTINUOUS SENSOR FUSION'
    });
  }, [currentPoint, history, isOutage]);

  const renderTrajectory = (canvas, opts) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Dynamic grid background
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    const gridSize = 25;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!opts.currentPoint || opts.history.length === 0) return;

    // Auto-compute bounding box scale
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    opts.history.forEach((pt) => {
      minX = Math.min(minX, pt.gt_x, pt[opts.trackKeyX] || pt.gt_x);
      maxX = Math.max(maxX, pt.gt_x, pt[opts.trackKeyX] || pt.gt_x);
      minY = Math.min(minY, pt.gt_y, pt[opts.trackKeyY] || pt.gt_y);
      maxY = Math.max(maxY, pt.gt_y, pt[opts.trackKeyY] || pt.gt_y);
    });

    const rangeX = Math.max(20, maxX - minX);
    const rangeY = Math.max(20, maxY - minY);
    const maxRange = Math.max(rangeX, rangeY);
    const scale = Math.min((width - 80) / maxRange, (height - 80) / maxRange, 1.8);

    const centerX = width / 2;
    const centerY = height / 2;
    const curGtX = opts.currentPoint.gt_x;
    const curGtY = opts.currentPoint.gt_y;

    const toScreen = (x, y) => ({
      px: centerX + (x - curGtX) * scale,
      py: centerY - (y - curGtY) * scale
    });

    // 1. Draw Ground Truth Corridor / Road Base
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    opts.history.forEach((pt, idx) => {
      const pos = toScreen(pt.gt_x, pt.gt_y);
      if (idx === 0) ctx.moveTo(pos.px, pos.py);
      else ctx.lineTo(pos.px, pos.py);
    });
    ctx.stroke();

    // Road Centerline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    opts.history.forEach((pt, idx) => {
      const pos = toScreen(pt.gt_x, pt.gt_y);
      if (idx === 0) ctx.moveTo(pos.px, pos.py);
      else ctx.lineTo(pos.px, pos.py);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Mark Outage Tunnel Zone on Road
    const outagePoints = opts.history.filter((pt) => pt.isOutage);
    if (outagePoints.length > 1) {
      ctx.strokeStyle = opts.isRawOnly ? 'rgba(244, 63, 94, 0.35)' : 'rgba(14, 165, 233, 0.35)';
      ctx.lineWidth = 14;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      outagePoints.forEach((pt, idx) => {
        const pos = toScreen(pt.gt_x, pt.gt_y);
        if (idx === 0) ctx.moveTo(pos.px, pos.py);
        else ctx.lineTo(pos.px, pos.py);
      });
      ctx.stroke();

      // Tunnel Ingress Pin
      const ingressPos = toScreen(outagePoints[0].gt_x, outagePoints[0].gt_y);
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(ingressPos.px, ingressPos.py, 4.5, 0, Math.PI * 2);
      ctx.fill();

      // Tunnel Egress Pin
      const egressPos = toScreen(outagePoints[outagePoints.length - 1].gt_x, outagePoints[outagePoints.length - 1].gt_y);
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(egressPos.px, egressPos.py, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // 3. Draw Track Trail Up To Current Time
    const activeHistory = opts.history.filter((pt) => pt.t <= opts.currentPoint.t);
    if (activeHistory.length > 0) {
      ctx.strokeStyle = opts.trailColor;
      ctx.lineWidth = 3.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.shadowColor = opts.trailColor;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      activeHistory.forEach((pt, idx) => {
        const pos = toScreen(pt[opts.trackKeyX], pt[opts.trackKeyY]);
        if (idx === 0) ctx.moveTo(pos.px, pos.py);
        else ctx.lineTo(pos.px, pos.py);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 4. Snapped Map Path (Right canvas only)
    if (!opts.isRawOnly && opts.snappedKeyX && activeHistory.length > 0) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      activeHistory.forEach((pt, idx) => {
        const pos = toScreen(pt[opts.snappedKeyX], pt[opts.snappedKeyY]);
        if (idx === 0) ctx.moveTo(pos.px, pos.py);
        else ctx.lineTo(pos.px, pos.py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 5. Start Pin
    const startPt = opts.history[0];
    const startPos = toScreen(startPt.gt_x, startPt.gt_y);
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(startPos.px, startPos.py, 5, 0, Math.PI * 2);
    ctx.fill();

    // 6. Active Vehicle Pointer & Uncertainty Ellipse
    const curPos = toScreen(opts.currentPoint[opts.trackKeyX], opts.currentPoint[opts.trackKeyY]);
    const headingRad = (opts.currentPoint.heading_deg * Math.PI) / 180;

    ctx.save();
    ctx.translate(curPos.px, curPos.py);

    // Uncertainty Pulse
    if (!opts.isRawOnly) {
      const errRadius = Math.max(10, Math.min(30, opts.currentPoint.fused_err * scale * 2));
      ctx.fillStyle = 'rgba(56, 189, 248, 0.20)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.60)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(0, 0, errRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.rotate(headingRad);

    // Vehicle Chevron
    ctx.fillStyle = opts.vehicleColor;
    ctx.shadowColor = opts.vehicleColor;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(0, -13);
    ctx.lineTo(8, 10);
    ctx.lineTo(0, 5);
    ctx.lineTo(-8, 10);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Compass Indicator on Top-Right Corner
    ctx.save();
    ctx.translate(width - 32, 32);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(0, 0, 16, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = '#f43f5e';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('N', 0, -18);
    ctx.restore();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full min-h-[520px]">
      {/* Left Screen: Raw GNSS-Only Track */}
      <div className="relative flex flex-col rounded-2xl bg-white border border-slate-200 overflow-hidden shadow-md">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-50 border-b border-slate-200 z-10">
          <div className="flex items-center gap-2.5">
            <span className={`w-2.5 h-2.5 rounded-full ${isOutage ? 'bg-rose-500 animate-ping' : 'bg-slate-400'}`} />
            <h3 className="font-['Space_Grotesk'] text-xs font-bold tracking-wide text-slate-800 uppercase">
              Raw GNSS-Only Track (Unaugmented)
            </h3>
          </div>
          <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded ${isOutage ? 'bg-rose-100 text-rose-700 border border-rose-300' : 'bg-slate-100 text-slate-600'}`}>
            {isOutage ? 'BLACKOUT: SIGNAL FROZEN' : 'LOCK ACTIVE'}
          </span>
        </div>

        {/* Viewport Canvas */}
        <div className="relative flex-1 w-full h-[340px] lg:h-full bg-[#0b132b]">
          <canvas ref={canvasLeftRef} width={600} height={420} className="w-full h-full object-cover" />

          {/* Outage Warning Banner */}
          {isOutage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-4 left-4 right-4 flex items-center gap-2.5 p-2.5 rounded-xl bg-rose-950/90 backdrop-blur-md border border-rose-500/40 text-rose-200 shadow-xl"
            >
              <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
              <div className="text-xs">
                <span className="font-bold">GNSS Denied (Tunnel / Outage)</span>
                <p className="text-[11px] text-rose-300/80">Satellite signals lost. Fix is frozen at tunnel entrance.</p>
              </div>
            </motion.div>
          )}

          {/* Bottom Metric Pill */}
          <div className="absolute bottom-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 backdrop-blur-md border border-white/10 text-xs font-mono text-white shadow-md">
            <span className="text-slate-400">POS ERROR:</span>
            <span className={`font-bold ${isOutage ? 'text-rose-400' : 'text-slate-200'}`}>
              {currentPoint ? currentPoint.raw_err.toFixed(1) : '0.0'} m
            </span>
          </div>
        </div>
      </div>

      {/* Right Screen: NavResilient AI-Fused Track */}
      <div className="relative flex flex-col rounded-2xl bg-white border border-sky-300 overflow-hidden shadow-md">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-sky-50/70 border-b border-sky-200 z-10">
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-sky-600 shadow-sm shadow-sky-500/40 animate-pulse" />
            <h3 className="font-['Space_Grotesk'] text-xs font-bold tracking-wide text-sky-950 uppercase">
              NavResilient AI-Fused Track (Our Engine)
            </h3>
          </div>
          <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-emerald-100 text-emerald-800 border border-emerald-300">
            <CheckCircle2 className="w-3 h-3" />
            DRIFT: {driftPct.toFixed(2)}% (&lt; 10% TARGET)
          </span>
        </div>

        {/* Viewport Canvas */}
        <div className="relative flex-1 w-full h-[340px] lg:h-full bg-[#0b132b]">
          <canvas ref={canvasRightRef} width={600} height={420} className="w-full h-full object-cover" />

          {/* AI Resilience Status Banner */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-4 left-4 right-4 flex items-center justify-between p-2.5 rounded-xl bg-slate-900/90 backdrop-blur-md border border-sky-500/30 text-slate-100 shadow-xl"
          >
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <div className="text-xs">
                <span className="font-bold text-cyan-300">TCN + 15-State UKF Active</span>
                <span className="text-[11px] text-slate-300 ml-2">Snapped to Road via HMM</span>
              </div>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 font-bold">100% RESILIENT</span>
          </motion.div>

          {/* Legend Overlay */}
          <div className="absolute bottom-4 left-4 flex flex-col gap-1.5 p-2 rounded-lg bg-slate-900/90 backdrop-blur-md border border-white/10 text-[11px] shadow-md">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400" />
              <span className="text-slate-200 font-medium">NavResilient AI-Fused (Inferred)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400" />
              <span className="text-slate-200 font-medium">OSM Snapped Road Trajectory</span>
            </div>
          </div>

          {/* Live Metric Pill */}
          <div className="absolute bottom-4 right-4 flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900/90 backdrop-blur-md border border-sky-500/30 text-xs font-mono text-white shadow-md">
            <div>
              <span className="text-slate-400">ERROR: </span>
              <span className="font-bold text-cyan-400">
                {currentPoint ? currentPoint.fused_err.toFixed(2) : '0.00'} m
              </span>
            </div>
            <div className="border-l border-white/10 pl-3">
              <span className="text-slate-400">DIST: </span>
              <span className="font-bold text-slate-100">{distanceTraveled.toFixed(0)} m</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
