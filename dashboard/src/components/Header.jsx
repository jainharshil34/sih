import React from 'react';
import { motion } from 'motion/react';
import { Navigation, Radio, Activity, Cpu, Terminal, ShieldCheck } from 'lucide-react';

export default function Header({
  activeTab,
  setActiveTab,
  gnssStatus,
  latencyMs,
  isStreaming,
  selectedDriveId,
  setSelectedDriveId,
  drives = {}
}) {
  const tabs = [
    { id: 'mission', label: 'Live Split-Screen & IRL Demo', icon: Navigation },
    { id: 'benchmark', label: '5-Tier Benchmark & Drift', icon: Activity },
    { id: 'pipeline', label: 'Under the Hood Engine', icon: Cpu },
    { id: 'terminal', label: 'Edge API & Integration', icon: Terminal },
  ];

  const getStatusColor = () => {
    switch (gnssStatus) {
      case 'GNSS_LOCKED':
        return 'bg-emerald-50 text-emerald-700 border-emerald-300';
      case 'GNSS_DENIED_INS':
        return 'bg-rose-50 text-rose-700 border-rose-300 animate-pulse';
      case 'REACQUIRED':
        return 'bg-sky-50 text-sky-700 border-sky-300';
      default:
        return 'bg-amber-50 text-amber-800 border-amber-300';
    }
  };

  return (
    <header className="flex flex-wrap items-center justify-between px-6 py-3.5 bg-white/95 backdrop-blur-xl border-b border-slate-200 shadow-xs z-50 sticky top-0 gap-4">
      {/* Brand & ISRO Badge */}
      <div className="flex items-center gap-3.5">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-cyan-600 shadow-md shadow-sky-600/20">
          <motion.div
            animate={{ scale: [1, 1.35, 1], opacity: [0.8, 0, 0.8] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute inset-0 rounded-xl border border-cyan-300"
          />
          <Navigation className="w-5 h-5 text-white fill-white/20 transform -rotate-45" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-['Space_Grotesk'] text-lg font-bold tracking-tight text-slate-900">
              NAVRESILIENT
            </h1>
            <span className="px-2 py-0.5 text-[10px] font-mono font-bold tracking-wider rounded-full bg-sky-100 text-sky-800 border border-sky-300">
              ISRO SIH26168
            </span>
          </div>
          <p className="text-xs text-slate-500 font-medium">
            AI-Aided Resilient Smartphone/Edge Inertial Navigation Engine
          </p>
        </div>
      </div>

      {/* Dataset / Drive Selector */}
      <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-xl shadow-xs">
        <span className="text-[11px] font-mono text-sky-700 font-bold uppercase tracking-wider">Dataset:</span>
        <select
          value={selectedDriveId}
          onChange={(e) => setSelectedDriveId(e.target.value)}
          className="bg-white text-xs font-semibold text-slate-800 border border-slate-200 rounded-lg px-2.5 py-1 outline-none focus:border-sky-500 cursor-pointer shadow-xs"
        >
          {Object.values(drives).map((d) => (
            <option key={d.id} value={d.id} className="bg-white text-slate-800">
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {/* Tabs Navigation with Motion */}
      <nav className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100/90 border border-slate-200 shadow-inner">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer ${
                isActive ? 'text-white' : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="activeTabBadge"
                  className="absolute inset-0 rounded-lg bg-gradient-to-r from-sky-600 to-cyan-600 shadow-md shadow-sky-600/25"
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
              <Icon className="w-4 h-4 relative z-10" />
              <span className="relative z-10">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Live Engine Status & Latency Pills */}
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono font-bold transition-all shadow-xs ${getStatusColor()}`}>
          <span className={`w-2 h-2 rounded-full ${gnssStatus === 'GNSS_DENIED_INS' ? 'bg-rose-500 animate-ping' : 'bg-emerald-500'}`} />
          <span>{gnssStatus.replace(/_/g, ' ')}</span>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs text-slate-700 font-mono shadow-xs">
          <Radio className="w-3.5 h-3.5 text-sky-600 animate-pulse" />
          <span className="text-slate-500">LATENCY:</span>
          <span className="font-bold text-sky-700">{latencyMs} ms</span>
        </div>
      </div>
    </header>
  );
}
