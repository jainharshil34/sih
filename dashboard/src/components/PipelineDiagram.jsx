import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Layers,
  Compass,
  Filter,
  BrainCircuit,
  Workflow,
  MapPin,
  CheckCircle,
  Info,
  Clock,
  Sparkles
} from 'lucide-react';

export default function PipelineDiagram({ isStreaming, latencyMs, speedKmh }) {
  const [hoveredStage, setHoveredStage] = useState(null);

  const stages = [
    {
      id: 'stage-imu',
      name: '1. Raw IMU Ingest',
      subtitle: '10 Hz – 200 Hz Streams',
      icon: Layers,
      color: 'from-blue-500 to-indigo-600',
      borderColor: 'border-blue-200 hover:border-blue-400',
      bgGradient: 'bg-blue-50/50',
      activeColor: '#2563eb',
      latency: '0.05 ms',
      math: 'a_{raw} = [a_x, a_y, a_z]^T, \\; \\omega_{raw} = [\\omega_x, \\omega_y, \\omega_z]^T',
      details: 'Accepts raw 6-DoF specific force and angular velocity from smartphone MEMS sensors or edge telematics hardware over TCP/UDP/Stdio.'
    },
    {
      id: 'stage-calib',
      name: '2. Auto-Calibration',
      subtitle: 'Dynamic Phone Mount R_mount',
      icon: Compass,
      color: 'from-cyan-500 to-teal-600',
      borderColor: 'border-cyan-200 hover:border-cyan-400',
      bgGradient: 'bg-cyan-50/50',
      activeColor: '#0891b2',
      latency: '0.12 ms',
      math: 'R_{mount} = R_z(\\psi_{align}) \\cdot R_y(\\theta) \\cdot R_x(\\phi)',
      details: 'Estimates arbitrary phone tilt (pitch/roll from gravity vector) and yaw heading offset from initial acceleration correlation without manual alignment.'
    },
    {
      id: 'stage-denoise',
      name: '3. Butterworth Denoise',
      subtitle: '< 2.5 Hz Kinematic Filter',
      icon: Filter,
      color: 'from-teal-500 to-emerald-600',
      borderColor: 'border-teal-200 hover:border-teal-400',
      bgGradient: 'bg-teal-50/50',
      activeColor: '#0d9488',
      latency: '0.18 ms',
      math: 'H(s) = \\frac{1}{1 + \\sqrt{2}(s/\\omega_c) + (s/\\omega_c)^2}, \\; f_c = 2.5\\text{ Hz}',
      details: 'Separates low-frequency vehicle kinematics from high-frequency (15-40 Hz) engine vibration, pothole shocks (>30 m/s²), and handling noise.'
    },
    {
      id: 'stage-tcn',
      name: '4. AI TCN Velocity',
      subtitle: '1D Dilated ConvNet (8.4k params)',
      icon: BrainCircuit,
      color: 'from-violet-500 to-purple-600',
      borderColor: 'border-purple-200 hover:border-purple-400',
      bgGradient: 'bg-purple-50/50',
      activeColor: '#7c3aed',
      latency: '0.68 ms',
      math: '\\hat{v}_x, \\; \\log \\sigma_v^2 = f_{TCN}(W_{imu}), \\; W \\in \\mathbb{R}^{6 \\times 20}',
      details: 'Regresses forward speed and aleatoric uncertainty from 2.0s IMU windows, trained on IO-VNBD dataset ECU wheel speeds (MAE: 0.709 m/s).'
    },
    {
      id: 'stage-ukf',
      name: '5. UKF + AI Residuals',
      subtitle: '15-State Merwe Fusion',
      icon: Workflow,
      color: 'from-purple-500 to-fuchsia-600',
      borderColor: 'border-fuchsia-200 hover:border-fuchsia-400',
      bgGradient: 'bg-fuchsia-50/50',
      activeColor: '#9333ea',
      latency: '1.45 ms',
      math: '\\mathbf{x}_{k} = f(\\mathbf{x}_{k-1}, \\mathbf{u}_k) - \\Delta \\mathbf{x}_{residual}^{AI}',
      details: 'FilterPy Unscented Kalman Filter fusing IMU kinematics, AI velocity, and dynamic covariance propagation with 1D-CNN residual drift cancellation.'
    },
    {
      id: 'stage-hmm',
      name: '6. HMM Map Matcher',
      subtitle: 'Offline OSM Road Graph',
      icon: MapPin,
      color: 'from-emerald-500 to-green-600',
      borderColor: 'border-emerald-200 hover:border-emerald-400',
      bgGradient: 'bg-emerald-50/50',
      activeColor: '#059669',
      latency: '1.34 ms',
      math: 'P(r_t | z_t) \\propto \\exp\\left(-\\frac{d(z_t, r_t)^2}{2\\sigma_z^2}\\right) \\cdot \\cos(\\Delta \\theta)',
      details: 'Non-Holonomic Viterbi map matching on KD-Tree indexed OSM graph with graceful off-road / underground parking structure fallback.'
    },
    {
      id: 'stage-output',
      name: '7. DriftCorrectedState',
      subtitle: '10 Hz Clean State Stream',
      icon: CheckCircle,
      color: 'from-amber-500 to-orange-600',
      borderColor: 'border-amber-200 hover:border-amber-400',
      bgGradient: 'bg-amber-50/50',
      activeColor: '#d97706',
      latency: '0.02 ms',
      math: '\\text{Emits: } (\\text{lat}, \\text{lon}, v, \\theta, \\text{drift\\%}, \\sigma_{pos})',
      details: 'Delivered to Engineer B mobile app or edge telematics dashboard at 10-200 Hz with zero step jumps across satellite outages.'
    }
  ];

  return (
    <div className="flex flex-col gap-6 p-6 rounded-2xl bg-white border border-slate-200 shadow-xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-sky-600" />
            <h2 className="font-['Space_Grotesk'] text-lg font-bold text-slate-900 tracking-wide">
              ⚡ Under the Hood: Real-Time Algorithmic Fusion Pipeline
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Data flows end-to-end through 7 high-performance stages in &lt; 3.8 ms (P95: 4.1 ms)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-50 border border-sky-200 text-xs font-mono text-sky-700">
            <Clock className="w-3.5 h-3.5 text-sky-600" />
            <span>TOTAL LATENCY: </span>
            <span className="font-bold text-sky-800">3.84 ms / step</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs font-mono text-emerald-700">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>725 Hz THROUGHPUT</span>
          </div>
        </div>
      </div>

      {/* Pipeline Stages Flow */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-7 gap-3 relative py-4">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          const isHovered = hoveredStage === stage.id;

          return (
            <motion.div
              key={stage.id}
              onMouseEnter={() => setHoveredStage(stage.id)}
              onMouseLeave={() => setHoveredStage(null)}
              whileHover={{ y: -4, scale: 1.02 }}
              className={`relative flex flex-col p-4 rounded-xl ${stage.bgGradient} bg-white border ${stage.borderColor} shadow-xs hover:shadow-md transition-all duration-200 cursor-pointer overflow-hidden ${
                isHovered ? 'ring-2 ring-sky-500 shadow-lg' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-3 relative z-10">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${stage.color} text-white shadow-xs`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-[10px] font-mono font-bold text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                  {stage.latency}
                </span>
              </div>

              <h4 className="text-xs font-bold text-slate-900 relative z-10 leading-tight">
                {stage.name}
              </h4>
              <p className="text-[11px] text-slate-500 relative z-10 mt-1 line-clamp-2">
                {stage.subtitle}
              </p>

              {/* Connected Arrow Indicator for Desktop */}
              {idx < stages.length - 1 && (
                <div className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-20 text-slate-300 font-bold text-xs pointer-events-none">
                  ▶
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Hover Deep-Dive Detail Drawer */}
      <AnimatePresence mode="wait">
        {hoveredStage && (
          <motion.div
            key={hoveredStage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="p-4 rounded-xl bg-slate-50 border border-sky-200 shadow-md flex flex-col md:flex-row items-start justify-between gap-4"
          >
            {(() => {
              const stg = stages.find((s) => s.id === hoveredStage);
              return (
                <>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-mono font-bold text-sky-700">STAGE SPECIFICATION:</span>
                      <h4 className="text-sm font-bold text-slate-900">{stg.name} — {stg.subtitle}</h4>
                    </div>
                    <p className="text-xs text-slate-600 leading-relaxed">{stg.details}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-white border border-slate-200 font-mono text-xs text-sky-800 shrink-0 shadow-xs">
                    <div className="text-[10px] text-slate-500 mb-1 font-sans font-bold">MATHEMATICAL FORMULATION:</div>
                    <code>{stg.math}</code>
                  </div>
                </>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
