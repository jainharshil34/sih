import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'motion/react';
import { ShieldAlert, Zap, CheckCircle2, LocateFixed, Layers } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Earth radius for local ENU Cartesian (meters) to WGS-84 Geodesic (lat, lon)
const R_EARTH = 6378137.0;

function enuToLatLon(east, north, refLat, refLon) {
  const latRad = (refLat * Math.PI) / 180.0;
  const dLat = (north / R_EARTH) * (180.0 / Math.PI);
  const dLon = (east / (R_EARTH * Math.cos(latRad))) * (180.0 / Math.PI);
  return [refLat + dLat, refLon + dLon];
}

// Custom vehicle chevron icon factory
function createVehicleIcon(color = '#0284c7', headingDeg = 0) {
  return L.divIcon({
    className: 'custom-vehicle-marker',
    html: `
      <div style="transform: rotate(${headingDeg}deg); transform-origin: center center; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style="filter: drop-shadow(0 0 6px ${color});">
          <path d="M12 2L3 21L12 17L21 21L12 2Z" fill="${color}" stroke="#ffffff" stroke-width="1.8" stroke-linejoin="round"/>
        </svg>
      </div>
    `,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

// Pin icon factory for Ingress / Egress
function createPinIcon(color = '#f59e0b', label = '') {
  return L.divIcon({
    className: 'custom-pin-marker',
    html: `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div style="width: 12px; height: 12px; border-radius: 50%; background: ${color}; border: 2px solid #ffffff; box-shadow: 0 0 8px ${color};"></div>
        ${label ? `<span style="font-size: 9px; font-weight: 700; font-family: monospace; color: #ffffff; background: rgba(15,23,42,0.85); padding: 1px 4px; border-radius: 4px; margin-top: 2px; border: 1px solid rgba(255,255,255,0.2);">${label}</span>` : ''}
      </div>
    `,
    iconSize: [20, 24],
    iconAnchor: [10, 6],
  });
}

export const OSM_THEMES = [
  {
    id: 'voyager',
    name: 'OSM Voyager (Crisp HD)',
    url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    subdomains: ['a', 'b', 'c', 'd'],
    maxZoom: 20,
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  },
  {
    id: 'dark',
    name: 'OSM Dark Cockpit',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    subdomains: ['a', 'b', 'c', 'd'],
    maxZoom: 20,
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  },
  {
    id: 'hot',
    name: 'OSM Detailed (HOT)',
    url: 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  },
  {
    id: 'standard',
    name: 'OSM Standard',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  },
  {
    id: 'satellite',
    name: 'Satellite Hybrid',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    subdomains: ['a', 'b', 'c'],
    maxZoom: 19,
    attribution: '&copy; Esri &mdash; High-Res Satellite',
  },
];

export default function SplitScreenMap({
  currentPoint,
  history,
  isOutage,
  distanceTraveled,
  driftPct,
  selectedDrive
}) {
  const leftContainerRef = useRef(null);
  const rightContainerRef = useRef(null);

  const leftMapRef = useRef(null);
  const rightMapRef = useRef(null);
  const leftTileRef = useRef(null);
  const rightTileRef = useRef(null);

  const [activeTheme, setActiveTheme] = useState('voyager');

  // Layer references for high-performance direct Leaflet updates (60fps)
  const leftLayersRef = useRef({
    gtCorridor: null,
    outagePolyline: null,
    rawPolyline: null,
    vehicleMarker: null,
    ingressMarker: null,
    egressMarker: null,
  });

  const rightLayersRef = useRef({
    gtCorridor: null,
    outagePolyline: null,
    fusedPolyline: null,
    snappedPolyline: null,
    vehicleMarker: null,
    uncertaintyCircle: null,
    ingressMarker: null,
    egressMarker: null,
  });

  const [followVehicle, setFollowVehicle] = useState(true);
  const refLat = selectedDrive?.ref_lat || 52.4068;
  const refLon = selectedDrive?.ref_lon || -1.5065;

  // Function to create a tile layer for a given theme
  const createTileLayer = (themeId) => {
    const theme = OSM_THEMES.find(t => t.id === themeId) || OSM_THEMES[0];
    return L.tileLayer(theme.url, {
      subdomains: theme.subdomains,
      maxZoom: theme.maxZoom,
      detectRetina: true,
      crossOrigin: true,
      keepBuffer: 6,
      updateWhenZooming: true,
      attribution: theme.attribution,
    });
  };

  // Change base tile layer dynamically when theme changes
  useEffect(() => {
    if (leftMapRef.current && leftTileRef.current) {
      leftMapRef.current.removeLayer(leftTileRef.current);
      const newLeftTile = createTileLayer(activeTheme).addTo(leftMapRef.current);
      newLeftTile.bringToBack();
      leftTileRef.current = newLeftTile;
    }
    if (rightMapRef.current && rightTileRef.current) {
      rightMapRef.current.removeLayer(rightTileRef.current);
      const newRightTile = createTileLayer(activeTheme).addTo(rightMapRef.current);
      newRightTile.bringToBack();
      rightTileRef.current = newRightTile;
    }
  }, [activeTheme]);

  // Initialize both Leaflet Maps once
  useEffect(() => {
    if (!leftContainerRef.current || !rightContainerRef.current) return;

    // Destroy previous instances if any
    if (leftMapRef.current) leftMapRef.current.remove();
    if (rightMapRef.current) rightMapRef.current.remove();

    const startPos = [refLat, refLon];

    // Map Options
    const mapOptions = {
      center: startPos,
      zoom: 17,
      zoomControl: false,
      attributionControl: false,
      preferCanvas: true,
      maxZoom: 20,
      minZoom: 12,
    };

    const leftMap = L.map(leftContainerRef.current, mapOptions);
    const rightMap = L.map(rightContainerRef.current, mapOptions);

    leftMapRef.current = leftMap;
    rightMapRef.current = rightMap;

    // Attach Initial Tile Layers
    const leftTile = createTileLayer(activeTheme).addTo(leftMap);
    leftTile.bringToBack();
    leftTileRef.current = leftTile;

    const rightTile = createTileLayer(activeTheme).addTo(rightMap);
    rightTile.bringToBack();
    rightTileRef.current = rightTile;

    // Synchronize pan & zoom between both maps on user interactions (drag & zoom)
    let isUserInteracting = false;
    const bindMapSync = (sourceMap, targetMap) => {
      sourceMap.on('drag', () => {
        targetMap.setView(sourceMap.getCenter(), sourceMap.getZoom(), { animate: false });
      });
      sourceMap.on('zoomend', () => {
        targetMap.setView(sourceMap.getCenter(), sourceMap.getZoom(), { animate: false });
      });
    };

    bindMapSync(leftMap, rightMap);
    bindMapSync(rightMap, leftMap);

    // Initialize Left Map Layers
    const lLayers = leftLayersRef.current;
    lLayers.gtCorridor = L.polyline([], { color: '#64748b', weight: 8, opacity: 0.45, lineCap: 'round', lineJoin: 'round' }).addTo(leftMap);
    lLayers.outagePolyline = L.polyline([], { color: '#ef4444', weight: 10, opacity: 0.35, lineCap: 'round', lineJoin: 'round' }).addTo(leftMap);
    lLayers.rawPolyline = L.polyline([], { color: '#dc2626', weight: 3.5, opacity: 0.9, lineCap: 'round', lineJoin: 'round' }).addTo(leftMap);
    lLayers.vehicleMarker = L.marker(startPos, { icon: createVehicleIcon('#dc2626', 0), zIndexOffset: 1000 }).addTo(leftMap);
    lLayers.ingressMarker = L.marker([0, 0], { icon: createPinIcon('#f59e0b', 'OUTAGE START') });
    lLayers.egressMarker = L.marker([0, 0], { icon: createPinIcon('#dc2626', 'RECOVERY') });

    // Initialize Right Map Layers
    const rLayers = rightLayersRef.current;
    rLayers.gtCorridor = L.polyline([], { color: '#64748b', weight: 8, opacity: 0.45, lineCap: 'round', lineJoin: 'round' }).addTo(rightMap);
    rLayers.outagePolyline = L.polyline([], { color: '#0284c7', weight: 10, opacity: 0.30, lineCap: 'round', lineJoin: 'round' }).addTo(rightMap);
    rLayers.fusedPolyline = L.polyline([], { color: '#0284c7', weight: 3, opacity: 0.85, dashArray: '6, 6', lineCap: 'round', lineJoin: 'round' }).addTo(rightMap);
    rLayers.snappedPolyline = L.polyline([], { color: '#059669', weight: 4, opacity: 0.95, lineCap: 'round', lineJoin: 'round' }).addTo(rightMap);
    rLayers.uncertaintyCircle = L.circle(startPos, { radius: 3, color: '#0284c7', fillColor: '#38bdf8', fillOpacity: 0.25, weight: 1.2 }).addTo(rightMap);
    rLayers.vehicleMarker = L.marker(startPos, { icon: createVehicleIcon('#059669', 0), zIndexOffset: 1000 }).addTo(rightMap);
    rLayers.ingressMarker = L.marker([0, 0], { icon: createPinIcon('#f59e0b', 'TUNNEL IN') });
    rLayers.egressMarker = L.marker([0, 0], { icon: createPinIcon('#059669', 'TUNNEL OUT') });

    return () => {
      leftMap.remove();
      rightMap.remove();
      leftMapRef.current = null;
      rightMapRef.current = null;
    };
  }, [refLat, refLon, selectedDrive?.id]);

  // Fit bounds when new drive loads
  useEffect(() => {
    if (!history || history.length === 0 || !leftMapRef.current || !rightMapRef.current) return;

    const gtCoords = history.map(pt => enuToLatLon(pt.gt_x, pt.gt_y, refLat, refLon));
    if (gtCoords.length > 1) {
      const bounds = L.latLngBounds(gtCoords);
      leftMapRef.current.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
      rightMapRef.current.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });

      // Set static Ground Truth Corridor and Outage bounds
      leftLayersRef.current.gtCorridor.setLatLngs(gtCoords);
      rightLayersRef.current.gtCorridor.setLatLngs(gtCoords);

      const outagePts = history.filter(pt => pt.isOutage);
      if (outagePts.length > 1) {
        const outCoords = outagePts.map(pt => enuToLatLon(pt.gt_x, pt.gt_y, refLat, refLon));
        leftLayersRef.current.outagePolyline.setLatLngs(outCoords);
        rightLayersRef.current.outagePolyline.setLatLngs(outCoords);

        // Position Ingress and Egress Pins
        const ingCoord = outCoords[0];
        const egCoord = outCoords[outCoords.length - 1];

        leftLayersRef.current.ingressMarker.setLatLng(ingCoord).addTo(leftMapRef.current);
        leftLayersRef.current.egressMarker.setLatLng(egCoord).addTo(leftMapRef.current);
        rightLayersRef.current.ingressMarker.setLatLng(ingCoord).addTo(rightMapRef.current);
        rightLayersRef.current.egressMarker.setLatLng(egCoord).addTo(rightMapRef.current);
      }
    }
  }, [history, refLat, refLon]);

  // Real-Time Frame Update Loop (High-Performance 60fps, Zero-Jitter)
  useEffect(() => {
    if (!currentPoint || !history || history.length === 0) return;

    const curIndex = history.findIndex(pt => pt.t === currentPoint.t);
    const activeSlice = curIndex >= 0 ? history.slice(0, curIndex + 1) : history.filter(pt => pt.t <= currentPoint.t);
    if (activeSlice.length === 0) return;

    const curHeading = currentPoint.heading_deg || 0;

    // 1. Update Left Map (Raw GNSS Only)
    const rawCoords = activeSlice.map(pt => enuToLatLon(pt.raw_x !== undefined ? pt.raw_x : pt.gt_x, pt.raw_y !== undefined ? pt.raw_y : pt.gt_y, refLat, refLon));
    const curRawPos = rawCoords[rawCoords.length - 1];

    if (leftLayersRef.current.rawPolyline) {
      leftLayersRef.current.rawPolyline.setLatLngs(rawCoords);
    }
    if (leftLayersRef.current.vehicleMarker) {
      leftLayersRef.current.vehicleMarker.setLatLng(curRawPos);
      leftLayersRef.current.vehicleMarker.setIcon(createVehicleIcon('#dc2626', curHeading));
    }

    // 2. Update Right Map (Pre-MM Fused + Post-MM Snapped)
    const fusedCoords = activeSlice.map(pt => enuToLatLon(pt.fused_x !== undefined ? pt.fused_x : pt.gt_x, pt.fused_y !== undefined ? pt.fused_y : pt.gt_y, refLat, refLon));
    const snappedCoords = activeSlice.map(pt => enuToLatLon(pt.snapped_x !== undefined ? pt.snapped_x : (pt.fused_x || pt.gt_x), pt.snapped_y !== undefined ? pt.snapped_y : (pt.fused_y || pt.gt_y), refLat, refLon));
    const curSnappedPos = snappedCoords[snappedCoords.length - 1];

    if (rightLayersRef.current.fusedPolyline) {
      rightLayersRef.current.fusedPolyline.setLatLngs(fusedCoords);
    }
    if (rightLayersRef.current.snappedPolyline) {
      rightLayersRef.current.snappedPolyline.setLatLngs(snappedCoords);
    }
    if (rightLayersRef.current.vehicleMarker) {
      rightLayersRef.current.vehicleMarker.setLatLng(curSnappedPos);
      rightLayersRef.current.vehicleMarker.setIcon(createVehicleIcon('#059669', curHeading));
    }
    if (rightLayersRef.current.uncertaintyCircle) {
      const errMeters = Math.max(1.8, currentPoint.fused_err || 2.5);
      rightLayersRef.current.uncertaintyCircle.setLatLng(curSnappedPos);
      rightLayersRef.current.uncertaintyCircle.setRadius(errMeters);
    }

    // Smooth Deadband Camera Follow (Only re-center if vehicle moves near viewport edge)
    if (followVehicle && leftMapRef.current && rightMapRef.current) {
      const map = rightMapRef.current;
      const bounds = map.getBounds();
      // If vehicle approaches within 15% of viewport edge, smoothly re-center
      const padLat = (bounds.getNorth() - bounds.getSouth()) * 0.15;
      const padLng = (bounds.getEast() - bounds.getWest()) * 0.15;
      const innerBounds = L.latLngBounds(
        [bounds.getSouth() + padLat, bounds.getWest() + padLng],
        [bounds.getNorth() - padLat, bounds.getEast() - padLng]
      );
      if (!innerBounds.contains(curSnappedPos)) {
        map.panTo(curSnappedPos, { animate: true, duration: 0.6, easeLinearity: 0.25 });
        leftMapRef.current.panTo(curSnappedPos, { animate: true, duration: 0.6, easeLinearity: 0.25 });
      }
    }
  }, [currentPoint, history, followVehicle, refLat, refLon]);

  const handleRecenter = useCallback(() => {
    if (!currentPoint || !leftMapRef.current || !rightMapRef.current) return;
    const curPos = enuToLatLon(currentPoint.snapped_x || currentPoint.gt_x, currentPoint.snapped_y || currentPoint.gt_y, refLat, refLon);
    leftMapRef.current.setView(curPos, 18, { animate: true });
    rightMapRef.current.setView(curPos, 18, { animate: true });
    setFollowVehicle(true);
  }, [currentPoint, refLat, refLon]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full min-h-[520px]">
      {/* LEFT PANEL: Raw GNSS-Only Track */}
      <div className="relative flex flex-col rounded-2xl bg-white border border-slate-200 overflow-hidden shadow-sm">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-2 bg-slate-50 border-b border-slate-200 z-10">
          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isOutage ? 'bg-rose-500 animate-ping' : 'bg-slate-400'}`} />
            <h3 className="font-['Space_Grotesk'] text-xs font-bold tracking-wide text-slate-800 uppercase">
              Raw GNSS-Only Track (Unaugmented)
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded ${isOutage ? 'bg-rose-100 text-rose-700 border border-rose-300' : 'bg-slate-100 text-slate-600'}`}>
              {isOutage ? 'BLACKOUT: SIGNAL FROZEN' : 'LOCK ACTIVE'}
            </span>
          </div>
        </div>

        {/* Viewport Map Container */}
        <div className="relative flex-1 w-full h-[360px] lg:h-full bg-slate-100">
          <div ref={leftContainerRef} className="w-full h-full" style={{ minHeight: '380px' }} />

          {/* Outage Warning Banner */}
          {isOutage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-3 left-3 right-3 flex items-center gap-2.5 p-2.5 rounded-xl bg-rose-950/90 backdrop-blur-md border border-rose-500/40 text-rose-200 shadow-lg z-[1000]"
            >
              <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
              <div className="text-xs">
                <span className="font-bold">GNSS Denied (Tunnel Blackout)</span>
                <p className="text-[11px] text-rose-300/80">Satellite signals lost. Fix is frozen at tunnel entrance.</p>
              </div>
            </motion.div>
          )}

          {/* Bottom Metric Pill */}
          <div className="absolute bottom-3 left-3 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 backdrop-blur-md border border-white/10 text-xs font-mono text-white shadow-md z-[1000]">
            <span className="text-slate-400">POS ERROR:</span>
            <span className={`font-bold ${isOutage ? 'text-rose-400' : 'text-slate-200'}`}>
              {currentPoint ? currentPoint.raw_err.toFixed(1) : '0.0'} m
            </span>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: NavResilient AI-Fused + Map-Matched Track */}
      <div className="relative flex flex-col rounded-2xl bg-white border border-sky-300 overflow-hidden shadow-sm">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-2 bg-sky-50/70 border-b border-sky-200 z-10">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 shadow-sm shadow-emerald-500/40 animate-pulse" />
            <h3 className="font-['Space_Grotesk'] text-xs font-bold tracking-wide text-sky-950 uppercase">
              NavResilient AI-Fused Track (TCN + UKF + HMM)
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {/* OSM Layer Selector Dropdown */}
            <div className="flex items-center gap-1 bg-white/90 backdrop-blur-sm border border-sky-200 rounded-lg p-0.5 shadow-xs">
              <Layers className="w-3 h-3 text-sky-600 ml-1.5" />
              <select
                value={activeTheme}
                onChange={(e) => setActiveTheme(e.target.value)}
                className="text-[10px] font-medium text-slate-700 bg-transparent border-0 py-0.5 pl-1 pr-5 focus:ring-0 focus:outline-hidden cursor-pointer"
                title="Select OpenStreetMap Map Layer Style"
              >
                {OSM_THEMES.map(theme => (
                  <option key={theme.id} value={theme.id}>
                    {theme.name}
                  </option>
                ))}
              </select>
            </div>

            <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-emerald-100 text-emerald-800 border border-emerald-300">
              <CheckCircle2 className="w-3 h-3" />
              DRIFT: {driftPct.toFixed(2)}% (&lt; 10%)
            </span>
          </div>
        </div>

        {/* Viewport Map Container */}
        <div className="relative flex-1 w-full h-[360px] lg:h-full bg-slate-100">
          <div ref={rightContainerRef} className="w-full h-full" style={{ minHeight: '380px' }} />

          {/* AI Resilience Status Banner */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-3 left-3 right-3 flex items-center justify-between p-2.5 rounded-xl bg-slate-900/90 backdrop-blur-md border border-sky-500/30 text-slate-100 shadow-lg z-[1000]"
          >
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              <div className="text-xs">
                <span className="font-bold text-emerald-300">Topological Map-Matching Active</span>
                <span className="text-[11px] text-slate-300 ml-2">Snapped to OSM Centerline</span>
              </div>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 font-bold">LOCKED</span>
          </motion.div>

          {/* Legend Overlay */}
          <div className="absolute bottom-3 left-3 flex flex-col gap-1.5 p-2 rounded-lg bg-slate-900/90 backdrop-blur-md border border-white/10 text-[11px] shadow-md z-[1000]">
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 border-t-2 border-dashed border-sky-400" />
              <span className="text-slate-200 font-medium">Pre-MM Inferred UKF Track</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-1 bg-emerald-500 rounded-sm" />
              <span className="text-emerald-300 font-bold">OSM Snapped Road Centerline</span>
            </div>
          </div>

          {/* Controls & Metrics Pills */}
          <div className="absolute bottom-3 right-3 flex items-center gap-2 z-[1000]">
            <button
              onClick={handleRecenter}
              title="Recenter Camera on Vehicle"
              className="p-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 text-slate-200 border border-white/10 shadow-md transition-colors cursor-pointer"
            >
              <LocateFixed className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900/90 backdrop-blur-md border border-sky-500/30 text-xs font-mono text-white shadow-md">
              <div>
                <span className="text-slate-400">ERROR: </span>
                <span className="font-bold text-emerald-400">
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
    </div>
  );
}
