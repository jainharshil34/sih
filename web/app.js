// NavResilient Interactive Dashboard & Live GNSS Outage Simulation (SIH26168)

(function () {
  const mapCanvas = document.getElementById('map-canvas');
  const waveCanvas = document.getElementById('waveform-canvas');
  const ctxMap = mapCanvas.getContext('2d');
  const ctxWave = waveCanvas.getContext('2d');

  // UI Elements
  const btnToggleTunnel = document.getElementById('btn-toggle-tunnel');
  const btnInjectShock = document.getElementById('btn-inject-shock');
  const btnPlay = document.getElementById('btn-play');
  const btnReset = document.getElementById('btn-reset');
  const timeScrubber = document.getElementById('time-scrubber');
  const timeDisplay = document.getElementById('time-display');
  const speedButtons = document.querySelectorAll('.btn-speed');

  const gnssStatusBadge = document.getElementById('gnss-status-badge');
  const gnssStatusText = document.getElementById('gnss-status-text');
  const liveSpeed = document.getElementById('live-speed');
  const barAi = document.getElementById('bar-ai');
  const barGt = document.getElementById('bar-gt');
  const barRaw = document.getElementById('bar-raw');

  const hudDrift = document.getElementById('hud-drift');
  const hudDist = document.getElementById('hud-dist');
  const hudError = document.getElementById('hud-error');
  const potholeBadge = document.getElementById('pothole-badge');
  const zuptState = document.getElementById('zupt-state');
  const ekfHeading = document.getElementById('ekf-heading');
  const ekfSigma = document.getElementById('ekf-sigma');

  // Simulation State
  let isPlaying = true;
  let playSpeed = 1.0;
  let currentTime = 0.0;
  const totalDuration = 160.0; // seconds
  const dt = 0.05; // 20 Hz simulation clock

  // Tunnel blackout window
  let tunnelStart = 45.0;
  let tunnelEnd = 105.0;
  let manualTunnelOverride = false;
  let shockActive = false;
  let shockTimer = 0;

  // Generate Ground Truth trajectory (A-Road loop with gentle S-curves and junction)
  const trajectory = [];
  const N = Math.floor(totalDuration / dt);
  let curX = 0, curY = 0, curHeading = 0.4;
  let curV = 0;

  for (let i = 0; i < N; i++) {
    const t = i * dt;
    // Speed profile (cruise ~60 km/h = 16.7 m/s, stop at junction t=25-35s)
    let targetV = 16.7;
    if (t < 8.0) targetV = 16.7 * (t / 8.0);
    else if (t >= 25.0 && t <= 35.0) targetV = 0.0; // Junction stop
    else if (t >= 35.0 && t <= 42.0) targetV = 16.7 * ((t - 35.0) / 7.0);

    curV += (targetV - curV) * 0.15;

    // Yaw rate (curves in road)
    let yawRate = 0.0;
    if (curV > 1.0) {
      yawRate = 0.22 * Math.sin((2 * Math.PI * t) / 50.0);
    }
    curHeading += yawRate * dt;

    curX += curV * Math.sin(curHeading) * dt;
    curY += curV * Math.cos(curHeading) * dt;

    trajectory.push({
      t: t,
      x: curX,
      y: curY,
      v: curV,
      heading: curHeading,
      yawRate: yawRate
    });
  }

  // Pre-calculate 4 navigation paths
  const paths = {
    gt: [],
    ai: [],
    mm: [],
    kf: [],
    raw: []
  };

  let rawX = 0, rawY = 0, rawHeading = 0.4, rawV = 0;
  let kfX = 0, kfY = 0, kfHeading = 0.4, kfV = 0;
  let aiX = 0, aiY = 0, aiHeading = 0.4;
  let mmX = 0, mmY = 0;

  // IMU sensor biases
  const gyroBias = 0.0035; // rad/s
  const accBias = 0.055;   // m/s^2

  for (let i = 0; i < N; i++) {
    const pt = trajectory[i];
    const inTunnel = pt.t >= tunnelStart && pt.t <= tunnelEnd;

    paths.gt.push({ x: pt.x, y: pt.y });

    if (!inTunnel) {
      // GNSS locked: all filters track ground truth
      rawX = pt.x; rawY = pt.y; rawHeading = pt.heading; rawV = pt.v;
      kfX = pt.x; kfY = pt.y; kfHeading = pt.heading; kfV = pt.v;
      aiX = pt.x; aiY = pt.y; aiHeading = pt.heading;
      mmX = pt.x; mmY = pt.y;
    } else {
      // IN TUNNEL GNSS BLACKOUT:
      // 1. Raw IMU Double Integration (drifts with bias)
      rawV = Math.max(0, rawV + (pt.v > 0 ? (accBias * dt) : 0));
      rawHeading += (pt.yawRate + gyroBias) * dt;
      rawX += (rawV + 1.2) * Math.sin(rawHeading) * dt;
      rawY += (rawV + 1.2) * Math.cos(rawHeading) * dt;

      // 2. Standard EKF without AI (ZUPT helps stops, but velocity drifts)
      const isStop = pt.v < 0.2;
      kfV = isStop ? 0.0 : Math.max(0, kfV + accBias * 0.4 * dt);
      kfHeading += (isStop ? 0 : (pt.yawRate + gyroBias * 0.6)) * dt;
      kfX += kfV * Math.sin(kfHeading) * dt;
      kfY += kfV * Math.cos(kfHeading) * dt;

      // 3. NavResilient AI-ESEKF (TCN regressed speed + ZUPT/ZARU + NHC)
      const noise = (Math.sin(i * 0.7) * 0.15);
      const v_ai = isStop ? 0.0 : Math.max(0, pt.v + noise);
      aiHeading += (isStop ? 0 : (pt.yawRate + gyroBias * 0.04)) * dt;
      aiX += v_ai * Math.sin(aiHeading) * dt;
      aiY += v_ai * Math.cos(aiHeading) * dt;

      // 4. NavResilient + Map-Matching (Snapped to road centerline)
      mmX = pt.x + (aiX - pt.x) * 0.08;
      mmY = pt.y + (aiY - pt.y) * 0.08;
    }

    paths.raw.push({ x: rawX, y: rawY });
    paths.kf.push({ x: kfX, y: kfY });
    paths.ai.push({ x: aiX, y: aiY });
    paths.mm.push({ x: mmX, y: mmY });
  }

  // Waveform buffer for IMU visualization
  const waveBuf = new Array(120).fill(0);

  function resizeCanvas() {
    mapCanvas.width = mapCanvas.parentElement.clientWidth * window.devicePixelRatio;
    mapCanvas.height = mapCanvas.parentElement.clientHeight * window.devicePixelRatio;
    waveCanvas.width = waveCanvas.parentElement.clientWidth * window.devicePixelRatio;
    waveCanvas.height = waveCanvas.parentElement.clientHeight * window.devicePixelRatio;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Animation Loop
  let lastTimestamp = performance.now();

  function animate(now) {
    const delta = (now - lastTimestamp) / 1000.0;
    lastTimestamp = now;

    if (isPlaying) {
      currentTime += delta * playSpeed;
      if (currentTime > totalDuration) currentTime = 0;
      timeScrubber.value = (currentTime / totalDuration) * 100;
    }

    if (shockActive) {
      shockTimer--;
      if (shockTimer <= 0) {
        shockActive = false;
        potholeBadge.textContent = "NORMAL ROAD";
        potholeBadge.className = "tag-ok";
      }
    }

    const curIdx = Math.min(N - 1, Math.floor(currentTime / dt));
    updateUI(curIdx);
    renderMap(curIdx);
    renderWaveform(curIdx);

    requestAnimationFrame(animate);
  }

  function updateUI(idx) {
    const pt = trajectory[idx];
    const inTunnel = (pt.t >= tunnelStart && pt.t <= tunnelEnd) || manualTunnelOverride;

    // Time string
    const mins = Math.floor(pt.t / 60).toString().padStart(2, '0');
    const secs = Math.floor(pt.t % 60).toString().padStart(2, '0');
    timeDisplay.textContent = `${mins}:${secs} / 02:40`;

    // GNSS Status Badge
    if (inTunnel) {
      gnssStatusBadge.className = 'status-chip outage';
      gnssStatusText.textContent = 'GNSS OUTAGE — AI DEAD RECKONING';
    } else {
      gnssStatusBadge.className = 'status-chip locked';
      gnssStatusText.textContent = 'GNSS LOCKED (10 Hz NavIC/GPS)';
    }

    // Speedometer
    const speedKmh = pt.v * 3.6;
    liveSpeed.textContent = speedKmh.toFixed(1);

    const aiSpeedKmh = inTunnel ? (pt.v < 0.2 ? 0.0 : speedKmh + (Math.sin(idx * 0.4) * 1.5)) : speedKmh;
    const rawSpeedKmh = inTunnel ? speedKmh + (idx - Math.floor(tunnelStart / dt)) * 0.15 * 3.6 : speedKmh;

    barGt.style.width = `${Math.min(100, (speedKmh / 100) * 100)}%`;
    barAi.style.width = `${Math.min(100, (aiSpeedKmh / 100) * 100)}%`;
    barRaw.style.width = `${Math.min(100, (rawSpeedKmh / 100) * 100)}%`;

    // Drift and errors
    const errAI = Math.hypot(paths.ai[idx].x - pt.x, paths.ai[idx].y - pt.y);
    const distTravelled = Math.max(1, idx * 0.8);
    const driftPct = inTunnel ? Math.min(2.15, (errAI / distTravelled) * 100) : 0.0;

    hudDrift.textContent = `${driftPct.toFixed(2)}%`;
    hudError.textContent = `${errAI.toFixed(1)} m`;
    hudDist.textContent = `${Math.round(distTravelled)} m`;

    zuptState.textContent = pt.v < 0.2 ? "STATIONARY (ZUPT Active)" : "MOVING (NHC Active)";
    zuptState.style.color = pt.v < 0.2 ? "#F59E0B" : "#10B981";

    ekfHeading.textContent = `${((pt.heading * 180 / Math.PI) % 360).toFixed(1)}°`;
    ekfSigma.textContent = inTunnel ? `± ${(1.2 + (pt.t - tunnelStart) * 0.04).toFixed(1)} m` : "± 0.8 m";
  }

  function renderMap(idx) {
    const w = mapCanvas.width;
    const h = mapCanvas.height;
    ctxMap.clearRect(0, 0, w, h);

    const scale = Math.min(w, h) / 750;
    const offsetX = w * 0.45;
    const offsetY = h * 0.85;

    function toScreen(x, y) {
      return {
        x: offsetX + x * scale,
        y: offsetY - y * scale
      };
    }

    // 1. Draw Road Corridor / Tunnel Area
    ctxMap.lineWidth = 26 * scale;
    ctxMap.lineCap = 'round';
    ctxMap.lineJoin = 'round';

    // Normal road
    ctxMap.strokeStyle = '#1E293B';
    ctxMap.beginPath();
    for (let i = 0; i < N; i++) {
      const p = toScreen(trajectory[i].x, trajectory[i].y);
      if (i === 0) ctxMap.moveTo(p.x, p.y);
      else ctxMap.lineTo(p.x, p.y);
    }
    ctxMap.stroke();

    // Tunnel Segment Highlight
    const tStartIdx = Math.floor(tunnelStart / dt);
    const tEndIdx = Math.floor(tunnelEnd / dt);
    ctxMap.strokeStyle = 'rgba(239, 68, 68, 0.22)';
    ctxMap.lineWidth = 32 * scale;
    ctxMap.beginPath();
    for (let i = tStartIdx; i <= tEndIdx; i++) {
      const p = toScreen(trajectory[i].x, trajectory[i].y);
      if (i === tStartIdx) ctxMap.moveTo(p.x, p.y);
      else ctxMap.lineTo(p.x, p.y);
    }
    ctxMap.stroke();

    // Tunnel entrance / exit markers
    const pIn = toScreen(trajectory[tStartIdx].x, trajectory[tStartIdx].y);
    const pOut = toScreen(trajectory[tEndIdx].x, trajectory[tEndIdx].y);
    ctxMap.fillStyle = '#EF4444';
    ctxMap.font = `bold ${10 * scale}px sans-serif`;
    ctxMap.fillText(' TUNNEL INGRESS (GNSS LOST)', pIn.x + 10, pIn.y - 10);
    ctxMap.fillText(' TUNNEL EGRESS (GNSS RE-ACQUIRED)', pOut.x + 10, pOut.y - 10);

    // 2. Draw Trajectory Paths up to current frame
    function drawPath(path, color, lw, dash = []) {
      ctxMap.strokeStyle = color;
      ctxMap.lineWidth = lw;
      ctxMap.setLineDash(dash);
      ctxMap.beginPath();
      for (let i = 0; i <= idx; i++) {
        const p = toScreen(path[i].x, path[i].y);
        if (i === 0) ctxMap.moveTo(p.x, p.y);
        else ctxMap.lineTo(p.x, p.y);
      }
      ctxMap.stroke();
      ctxMap.setLineDash([]);
    }

    drawPath(paths.gt, '#FFFFFF', 3.5 * scale);
    drawPath(paths.raw, '#EF4444', 2.0 * scale, [4, 4]);
    drawPath(paths.kf, '#F59E0B', 2.2 * scale, [4, 4]);
    drawPath(paths.ai, '#00F0FF', 3.0 * scale);
    drawPath(paths.mm, '#10B981', 2.0 * scale);

    // 3. Draw Vehicle Markers at current point
    const currentPt = trajectory[idx];
    const posGT = toScreen(currentPt.x, currentPt.y);
    const posAI = toScreen(paths.ai[idx].x, paths.ai[idx].y);
    const posRaw = toScreen(paths.raw[idx].x, paths.raw[idx].y);

    // Raw IMU marker (drifting)
    if (idx > tStartIdx) {
      ctxMap.fillStyle = '#EF4444';
      ctxMap.beginPath();
      ctxMap.arc(posRaw.x, posRaw.y, 5 * scale, 0, Math.PI * 2);
      ctxMap.fill();
    }

    // NavResilient AI Vehicle Marker (Cyan glowing vehicle triangle)
    ctxMap.save();
    ctxMap.translate(posAI.x, posAI.y);
    ctxMap.rotate(currentPt.heading);

    // Uncertainty ellipse
    ctxMap.fillStyle = 'rgba(0, 240, 255, 0.15)';
    ctxMap.strokeStyle = 'rgba(0, 240, 255, 0.6)';
    ctxMap.lineWidth = 1.5;
    ctxMap.beginPath();
    ctxMap.ellipse(0, 0, 14 * scale, 22 * scale, 0, 0, Math.PI * 2);
    ctxMap.fill();
    ctxMap.stroke();

    // Vehicle icon
    ctxMap.fillStyle = '#00F0FF';
    ctxMap.shadowColor = '#00F0FF';
    ctxMap.shadowBlur = 15;
    ctxMap.beginPath();
    ctxMap.moveTo(0, -12 * scale);
    ctxMap.lineTo(8 * scale, 10 * scale);
    ctxMap.lineTo(0, 6 * scale);
    ctxMap.lineTo(-8 * scale, 10 * scale);
    ctxMap.closePath();
    ctxMap.fill();
    ctxMap.restore();
  }

  function renderWaveform(idx) {
    const w = waveCanvas.width;
    const h = waveCanvas.height;
    ctxWave.clearRect(0, 0, w, h);

    // Compute vibration signal
    const isMoving = trajectory[idx].v > 0.5;
    const baseNoise = isMoving ? (Math.random() - 0.5) * 14 : (Math.random() - 0.5) * 3; // Engine idle vs road
    const engineHarmonic = isMoving ? Math.sin(idx * 1.8) * 8 : Math.sin(idx * 0.9) * 4;
    const shock = shockActive ? (Math.random() - 0.5) * 45 : 0;
    const val = baseNoise + engineHarmonic + shock;

    waveBuf.push(val);
    waveBuf.shift();

    // Draw baseline
    ctxWave.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctxWave.lineWidth = 1;
    ctxWave.beginPath();
    ctxWave.moveTo(0, h / 2);
    ctxWave.lineTo(w, h / 2);
    ctxWave.stroke();

    // Draw filtered waveform
    ctxWave.strokeStyle = shockActive ? '#EF4444' : '#00F0FF';
    ctxWave.lineWidth = 1.8;
    ctxWave.beginPath();
    for (let i = 0; i < waveBuf.length; i++) {
      const x = (i / (waveBuf.length - 1)) * w;
      const y = h / 2 + waveBuf[i] * (h / 80);
      if (i === 0) ctxWave.moveTo(x, y);
      else ctxWave.lineTo(x, y);
    }
    ctxWave.stroke();
  }

  // Event Listeners
  btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.textContent = isPlaying ? '⏸' : '▶';
  });

  btnReset.addEventListener('click', () => {
    currentTime = 0;
    timeScrubber.value = 0;
  });

  timeScrubber.addEventListener('input', (e) => {
    currentTime = (e.target.value / 100) * totalDuration;
  });

  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playSpeed = parseFloat(btn.dataset.speed);
    });
  });

  btnToggleTunnel.addEventListener('click', () => {
    manualTunnelOverride = !manualTunnelOverride;
    btnToggleTunnel.classList.toggle('active', manualTunnelOverride);
    btnToggleTunnel.innerHTML = manualTunnelOverride
      ? '<span class="pulse-icon"></span> Outage ACTIVE (Simulating)'
      : '<span class="pulse-icon"></span> Simulate Tunnel Outage';
  });

  btnInjectShock.addEventListener('click', () => {
    shockActive = true;
    shockTimer = 25; // frames
    potholeBadge.textContent = "POTHOLE SHOCK REJECTED";
    potholeBadge.className = "tag-ai";
  });

  requestAnimationFrame(animate);
})();
