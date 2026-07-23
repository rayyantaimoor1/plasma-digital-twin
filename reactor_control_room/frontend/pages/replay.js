/* Session Replay - flight-recorder view of saved experiment runs, read from
 * /api/sessions. Select a run (or press play to sweep) and the per-run gauges +
 * cross-run sparklines update. Every value is a stored session field from the
 * backend; the page never simulates. If the store is empty it says how to seed. */
import { createGauge, createSparkline, fmtSci } from '../components.js';

const MODE = {
  'Stable Plasma': { label: 'Stable', color: 'var(--green)' },
  'Exploratory Sweep': { label: 'Explore', color: 'var(--cyan)' },
  'Stress Test': { label: 'Stress', color: 'var(--orange)' },
};
const modeOf = (m) => MODE[m] || { label: m || '—', color: 'var(--text-dim)' };

const RG = [
  { key: 'ion_energy_ev', label: 'Ion Energy', unit: 'eV', min: 0, max: 100, decimals: 0, color: 'var(--orange)', maxWidth: 118 },
  { key: 'defect_probability', label: 'Defect Prob.', unit: '0–1', min: 0, max: 1, decimals: 3, color: 'var(--red)', maxWidth: 118 },
  { key: 'process_quality', label: 'Process Quality', unit: '0–1', min: 0, max: 1, decimals: 3, color: 'var(--purple)', maxWidth: 118 },
  { key: 'plasma_density_m3', label: 'Plasma Density', unit: 'm⁻³', min: 1e15, max: 1e18, log: true, sci: true, color: 'var(--cyan)', maxWidth: 118 },
];
const SP = [
  { key: 'ion_energy_ev', label: 'Ion Energy', unit: 'eV', color: 'var(--orange)', decimals: 0 },
  { key: 'defect_probability', label: 'Defect Probability', unit: '0–1', color: 'var(--red)', decimals: 3 },
  { key: 'process_quality', label: 'Process Quality', unit: '0–1', color: 'var(--purple)', decimals: 3 },
];

const el = (tag, css, html) => { const n = document.createElement(tag); if (css) n.style.cssText = css; if (html !== undefined) n.innerHTML = html; return n; };

function downloadReport(sessions) {
  const rows = sessions.map((s) => {
    const m = modeOf(s.mode);
    return `<tr><td class="m">${s.session_id.slice(0, 8)}</td><td><span class="mode" style="color:${m.color === 'var(--green)' ? '#0a7d54' : m.color === 'var(--cyan)' ? '#0369a1' : '#c2410c'}">${m.label}</span></td><td class="m r">${s.rf_power_w}</td><td class="m r">${s.pressure_mtorr}</td><td class="m r">${s.electron_temperature_ev.toFixed(2)}</td><td class="m r">${s.ion_energy_ev.toFixed(0)}</td><td class="m r">${s.etch_rate_nm_min.toFixed(1)}</td><td class="m r">${s.process_quality.toFixed(3)}</td><td class="m r" style="color:${s.defect_probability > 0.55 ? '#dc2626' : '#111'}">${s.defect_probability.toFixed(3)}</td></tr>`;
  }).join('');
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Session Replay Export</title><style>@page{size:A4 landscape;margin:14mm}body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#111;padding:26px}.m{font-family:ui-monospace,Menlo,monospace}h1{font-size:20px;margin:0 0 3px}.sub{color:#555;font-size:12px;margin:0 0 14px}.badge{display:inline-block;font-family:ui-monospace,monospace;font-size:10px;color:#7c3aed;border:1px solid #e0d6f5;background:#f7f3ff;border-radius:5px;padding:3px 9px;margin-bottom:16px}table{width:100%;border-collapse:collapse;font-size:11px}th,td{padding:7px 9px;border-bottom:1px solid #e5e7eb;text-align:left}th{font-size:9px;letter-spacing:.5px;text-transform:uppercase;color:#6b7280;border-bottom:2px solid #111}.r{text-align:right}.mode{font-size:9px;text-transform:uppercase;letter-spacing:.5px;border:1px solid;border-radius:4px;padding:1px 6px}footer{margin-top:22px;font-size:10px;color:#888;border-top:1px solid #e5e7eb;padding-top:10px}</style></head><body><h1>Reactor Control Room — Session Replay</h1><p class="sub">Flight-recorder export · ${sessions.length} saved runs · generated ${new Date().toLocaleString()}</p><span class="badge">SIMULATION · argon CCP · 13.56 MHz · 0D global model</span><table><thead><tr><th>Session</th><th>Mode</th><th class="r">RF (W)</th><th class="r">Press</th><th class="r">Te (eV)</th><th class="r">Ion E</th><th class="r">Etch</th><th class="r">Quality</th><th class="r">Defect</th></tr></thead><tbody>${rows}</tbody></table><footer>Every field maps 1:1 to a named twin output, read from the saved session store. Values are physics-derived, not measured hardware data.</footer></body></html>`;
  const w = window.open('', '_blank');
  if (w) { w.document.write(html); w.document.close(); w.focus(); setTimeout(() => { try { w.print(); } catch (_) {} }, 400); }
}

export async function renderReplay(main, { api }) {
  main.innerHTML = '';

  const head = el('div', 'display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px');
  head.innerHTML = `<div><h1 style="margin:0;font-size:25px;font-weight:600;color:var(--text)">Session Replay</h1>
    <p style="margin:5px 0 0;font-size:13px;color:var(--text-dim)">Flight-recorder view of saved experiment runs. Select a run, or press play to sweep the record.</p></div>`;
  const btnWrap = el('div', 'display:flex;gap:10px');
  head.appendChild(btnWrap);
  main.appendChild(head);

  // sessions come newest-first; reverse to chronological for a natural sweep.
  const sessions = (await api.sessions()).slice().reverse();

  if (sessions.length === 0) {
    main.appendChild(el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:28px;color:var(--text-dim);font-size:13px;line-height:1.6',
      `No saved sessions yet. Seed a demo record with:<br><span class="mono" style="color:var(--cyan);font-size:12px">python reactor_control_room/seed_demo_sessions.py</span><br>or save runs from the Streamlit dashboard's Session History page.`));
    return {};
  }

  const playBtn = el('button', '', '');
  const pdfBtn = el('button', 'font-family:var(--mono);font-size:11px;letter-spacing:0.5px;padding:9px 16px;border-radius:9px;cursor:pointer;border:1px solid #26456a;background:rgba(56,189,248,0.08);color:var(--cyan-2)', '⭳  Download PDF');
  pdfBtn.onclick = () => downloadReport(sessions);
  btnWrap.append(playBtn, pdfBtn);

  const grid = el('div', 'display:grid;grid-template-columns:300px 1fr;gap:20px;align-items:start');
  main.appendChild(grid);

  // ---- run log ----
  const logCard = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:8px;display:flex;flex-direction:column;gap:2px');
  logCard.appendChild(el('div', 'padding:10px 12px 8px;display:flex;justify-content:space-between;align-items:baseline',
    `<span style="font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--text-dim);font-weight:600">Run Log</span><span style="font-family:var(--mono);font-size:9px;color:var(--text-mute)">${sessions.length} runs</span>`));
  const rows = sessions.map((s, i) => {
    const m = modeOf(s.mode);
    const btn = el('button', '', '');
    btn.innerHTML = `<div style="display:flex;align-items:center;gap:8px">
        <span class="dot" style="width:7px;height:7px;border-radius:50%;flex:none;background:${m.color}"></span>
        <span class="mono" style="font-size:11px;color:var(--text)">${s.session_id.slice(0, 6)}</span>
        <span style="margin-left:auto;font-size:8.5px;letter-spacing:0.5px;text-transform:uppercase;color:${m.color}">${m.label}</span>
      </div>
      <div style="display:flex;gap:10px;margin-top:5px;font-family:var(--mono);font-size:9.5px;color:var(--text-dim)">
        <span style="color:var(--purple)">${s.rf_power_w}W</span><span style="color:var(--cyan)">${s.pressure_mtorr}mT</span><span>Eᵢ ${s.ion_energy_ev.toFixed(0)}</span><span style="color:${s.defect_probability > 0.55 ? 'var(--red)' : 'var(--text-dim)'}">def ${s.defect_probability.toFixed(2)}</span>
      </div>`;
    btn.onclick = () => select(i);
    logCard.appendChild(btn);
    return btn;
  });
  grid.appendChild(logCard);

  // ---- right: gauges + sparklines ----
  const rightCol = el('div', 'display:flex;flex-direction:column;gap:16px');
  const gaugeGrid = el('div', 'display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px');
  const gauges = RG.map((cfg) => { const g = createGauge(cfg); gaugeGrid.appendChild(g.el); return { cfg, g }; });
  rightCol.appendChild(gaugeGrid);
  const sparkStack = el('div', 'display:flex;flex-direction:column;gap:12px');
  const sparks = SP.map((cfg) => { const s = createSparkline(cfg); sparkStack.appendChild(s.el); return { cfg, s }; });
  rightCol.appendChild(sparkStack);
  grid.appendChild(rightCol);

  // ---- selection state ----
  let sel = 0;
  let playing = false;
  let timer = null;

  function paintRows() {
    rows.forEach((btn, i) => {
      const on = i === sel;
      const m = modeOf(sessions[i].mode);
      btn.style.cssText = `display:flex;flex-direction:column;align-items:flex-start;width:100%;text-align:left;cursor:pointer;padding:9px 11px;border-radius:9px;border:1px solid ${on ? '#2b3a4d' : 'transparent'};background:${on ? '#131c28' : 'transparent'}`;
      btn.querySelector('.dot').style.boxShadow = on ? `0 0 8px ${m.color}` : 'none';
    });
  }
  function select(i) {
    sel = i;
    const s = sessions[sel];
    for (const { cfg, g } of gauges) g.update(s[cfg.key]);
    for (const { cfg, s: sp } of sparks) sp.update({ values: sessions.map((x) => x[cfg.key]), currentIndex: sel });
    paintRows();
  }
  function paintPlay() {
    playBtn.textContent = playing ? '❚❚  Pause' : '▶  Play record';
    playBtn.style.cssText = `font-family:var(--mono);font-size:11px;letter-spacing:0.5px;padding:9px 16px;border-radius:9px;cursor:pointer;border:1px solid ${playing ? 'var(--cyan)' : 'var(--border)'};background:${playing ? 'rgba(56,189,248,0.12)' : 'var(--panel)'};color:${playing ? 'var(--cyan-2)' : 'var(--text-dim)'}`;
  }
  playBtn.onclick = () => {
    playing = !playing;
    if (playing) timer = setInterval(() => select((sel + 1) % sessions.length), 1400);
    else { clearInterval(timer); timer = null; }
    paintPlay();
  };

  paintPlay();
  select(0);

  return { cleanup: () => { if (timer) clearInterval(timer); } };
}
