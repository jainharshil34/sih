import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea
} from 'recharts';

import { REAL_DRIVES } from '../data/driveData';

export default function DriftChart({ history = [], distanceTraveled, currentProgressT }) {
  // Use full history points or fallback to default drive to ensure chart is never empty
  const sourcePoints = (history && history.length > 0)
    ? history
    : (REAL_DRIVES["S-Vw12"]?.generatePoints() || []);

  const maxT = sourcePoints.length > 0 ? (sourcePoints[sourcePoints.length - 1].t || 160) : 160;
  const step = Math.max(1, Math.floor(sourcePoints.length / 140));

  // Find outage bounds for shading
  const outagePts = sourcePoints.filter((pt) => pt.isOutage);
  const outageStartT = outagePts.length > 0 ? outagePts[0].t : 25;
  const outageEndT = outagePts.length > 0 ? outagePts[outagePts.length - 1].t : 65;

  const chartData = sourcePoints.filter((_, idx) => idx % step === 0).map((pt) => {
    const ptDist = Math.hypot(pt.gt_x || 0, pt.gt_y || 0);
    const distAtT = Math.max(1, ptDist > 1 ? ptDist : ((pt.t / maxT) * (distanceTraveled || 526)));
    const budgetMeters = 0.10 * distAtT;

    return {
      time: `${pt.t.toFixed(0)}s`,
      t: pt.t,
      fused_err: Number((pt.fused_err ?? 0).toFixed(2)),
      raw_err: Number((pt.raw_err ?? 0).toFixed(2)),
      budget_line: Number(budgetMeters.toFixed(2)),
      speed_kmh: Number((pt.speed_kmh ?? 0).toFixed(1)),
      isOutage: pt.isOutage,
      drift_pct: Number(((pt.fused_err / distAtT) * 100).toFixed(2))
    };
  });

  return (
    <div className="flex flex-col rounded-2xl bg-white border border-slate-200 p-5 shadow-md">
      <div className="flex flex-wrap items-center justify-between mb-4 gap-2">
        <div>
          <h3 className="font-['Space_Grotesk'] text-sm font-bold tracking-wide text-slate-900">
            📈 Positional Error &amp; Drift Growth vs Budget Target
          </h3>
          <p className="text-xs text-slate-500">
            Real-time tracking of cumulative dead reckoning drift vs the strict &lt; 10% distance threshold
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-sky-500" />
            <span className="text-slate-700 font-medium">NavResilient Error (m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-rose-500" />
            <span className="text-slate-700 font-medium">Raw GNSS Denial Drift</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-amber-500 border-b border-dashed border-amber-500" />
            <span className="text-amber-700 font-bold">10% Budget Ceiling</span>
          </div>
        </div>
      </div>

      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#0284c7" stopOpacity={0.35}/>
                <stop offset="95%" stopColor="#0284c7" stopOpacity={0.0}/>
              </linearGradient>
              <linearGradient id="roseGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.25}/>
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0}/>
              </linearGradient>
              <linearGradient id="budgetGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.12}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.01}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#64748b', fontSize: 11 }} />
            <YAxis stroke="#94a3b8" tick={{ fill: '#64748b', fontSize: 11 }} unit="m" />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="p-3 bg-white border border-slate-200 rounded-xl shadow-xl text-xs font-mono">
                    <div className="font-bold text-slate-800 mb-1 border-b border-slate-100 pb-1 flex items-center justify-between gap-3">
                      <span>Time: {label}</span>
                      <span className={`px-1.5 py-0.2 rounded text-[10px] ${d.isOutage ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'}`}>
                        {d.isOutage ? 'BLACKOUT ACTIVE' : 'GNSS LOCKED'}
                      </span>
                    </div>
                    <div className="space-y-1 text-[11px]">
                      <div className="flex justify-between gap-3">
                        <span className="text-sky-600 font-semibold">NavResilient Error:</span>
                        <span className="font-bold text-slate-900">{d.fused_err} m ({d.drift_pct}%)</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-rose-600 font-semibold">Raw GNSS Drift:</span>
                        <span className="font-bold text-slate-900">{d.raw_err} m</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-amber-600 font-semibold">10% Budget Ceiling:</span>
                        <span className="font-bold text-slate-900">{d.budget_line} m</span>
                      </div>
                      <div className="flex justify-between gap-3 pt-1 border-t border-slate-100">
                        <span className="text-slate-500">Speed:</span>
                        <span className="font-bold text-slate-700">{d.speed_kmh} km/h</span>
                      </div>
                    </div>
                  </div>
                );
              }}
            />
            {/* Shaded Compliant Budget Area */}
            <Area
              type="monotone"
              dataKey="budget_line"
              name="10% Budget Ceiling"
              stroke="#d97706"
              strokeDasharray="4 4"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#budgetGrad)"
            />
            {/* Raw GNSS Denial Error */}
            <Area
              type="monotone"
              dataKey="raw_err"
              name="Raw GNSS Blackout Error"
              stroke="#f43f5e"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#roseGrad)"
            />
            {/* NavResilient Inferred Error */}
            <Area
              type="monotone"
              dataKey="fused_err"
              name="NavResilient Inferred Error"
              stroke="#0284c7"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#cyanGrad)"
            />
            {/* Real-time playback position cursor */}
            {currentProgressT !== undefined && (
              <ReferenceLine
                x={`${Math.round(currentProgressT)}s`}
                stroke="#0284c7"
                strokeWidth={2}
                strokeDasharray="3 3"
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
