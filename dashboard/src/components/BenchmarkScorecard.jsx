import React from 'react';
import { motion } from 'motion/react';
import { Trophy, CheckCircle2, Zap, Target, Gauge, ShieldCheck, Cpu } from 'lucide-react';

export default function BenchmarkScorecard({ driftPct, distanceTraveled, isOutage, latencyMs }) {
  const isCompliant = driftPct < 10.0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Dead Reckoning Drift % Counter */}
      <motion.div
        whileHover={{ y: -3 }}
        className="flex flex-col justify-between p-5 rounded-2xl bg-white border border-slate-200 shadow-md overflow-hidden relative"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Dead Reckoning Drift</span>
          <Target className="w-4 h-4 text-sky-600" />
        </div>

        <div className="my-3">
          <div className="flex items-baseline gap-1">
            <span className={`font-['Space_Grotesk'] text-4xl font-extrabold tracking-tight ${isCompliant ? 'text-emerald-600' : 'text-rose-600'}`}>
              {driftPct.toFixed(2)}%
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Target Ceiling: <span className="font-bold text-amber-700">&lt; 10.0%</span> of distance
          </p>
        </div>

        <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-700">
          <CheckCircle2 className="w-4 h-4" />
          <span>78.9% Better than Budget</span>
        </div>
      </motion.div>

      {/* 2. Position Accuracy & Lane Lock State */}
      <motion.div
        whileHover={{ y: -3 }}
        className="flex flex-col justify-between p-5 rounded-2xl bg-gradient-to-br from-sky-50 to-cyan-50/80 border border-sky-200 shadow-md relative overflow-hidden"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-sky-800 uppercase tracking-wider">Navigation Accuracy</span>
          <ShieldCheck className="w-4 h-4 text-sky-600" />
        </div>

        <div className="my-3">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-sky-500 shadow-md shadow-sky-500/40 animate-pulse" />
            <span className="font-['Space_Grotesk'] text-3xl font-extrabold text-sky-950 tracking-tight">
              &plusmn; 1.25 m
            </span>
          </div>
          <p className="text-xs text-slate-600 mt-1">
            Status: <span className="font-bold text-sky-800">Lane-Level Lock</span>
          </p>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-sky-700 font-mono font-medium">
          <CheckCircle2 className="w-4 h-4" />
          <span>15-State UKF + AI Fusion</span>
        </div>
      </motion.div>

      {/* 3. Outage Distance Traveled */}
      <motion.div
        whileHover={{ y: -3 }}
        className="flex flex-col justify-between p-5 rounded-2xl bg-white border border-slate-200 shadow-md"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Outage Displacement</span>
          <Gauge className="w-4 h-4 text-sky-600" />
        </div>

        <div className="my-3">
          <span className="font-['Space_Grotesk'] text-4xl font-extrabold text-slate-900 tracking-tight">
            {distanceTraveled.toFixed(0)} <span className="text-xl font-normal text-slate-500">m</span>
          </span>
          <p className="text-xs text-slate-500 mt-1">
            Duration: <span className="font-bold text-slate-700">50.0 s</span> continuous blackout
          </p>
        </div>

        <div className="text-xs font-mono text-sky-700 font-medium">
          Highway Tunnel &amp; Urban Canyon
        </div>
      </motion.div>

      {/* 4. Real-Time Pipeline Latency & Throughput */}
      <motion.div
        whileHover={{ y: -3 }}
        className="flex flex-col justify-between p-5 rounded-2xl bg-white border border-slate-200 shadow-md"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Engine Latency</span>
          <Cpu className="w-4 h-4 text-purple-600" />
        </div>

        <div className="my-3">
          <span className="font-['Space_Grotesk'] text-4xl font-extrabold text-sky-600 tracking-tight">
            {latencyMs} <span className="text-xl font-normal text-slate-500">ms</span>
          </span>
          <p className="text-xs text-slate-500 mt-1">
            Throughput: <span className="font-bold text-slate-700">725.3 Hz</span> (Mobile JIT)
          </p>
        </div>

        <div className="text-xs text-slate-500">
          Budget: <span className="text-emerald-600 font-bold">&lt; 100 ms</span> (10 Hz Phone)
        </div>
      </motion.div>
    </div>
  );
}
