/* AI Verdict - the three AI outputs for the current operating point as dramatic
 * dials: classification (RF/XGBoost), relationship-anomaly status (Isolation
 * Forest, with fault injection), and best-fit application (compliance windows).
 *
 * Every value is fetched: /api/classify, /api/anomaly, /api/suitability-scorecard.
 * Colours, arc fills and bar widths are the only things computed here (visual). */
import { createArcGauge, createShapBars, clamp } from '../components.js';

const CLASS_COLOR = { Optimal: 'var(--green)', Acceptable: 'var(--cyan)', Marginal: 'var(--amber)', Unsuitable: 'var(--red)' };
// Severity -> arc fill fraction + colour (a VISUAL encoding of the backend's
// severity label; the label and the numeric score both come from the backend).
const SEVERITY = { Normal: { color: 'var(--green)', frac: 0.2 }, Warning: { color: 'var(--amber)', frac: 0.62 }, Critical: { color: 'var(--red)', frac: 0.95 } };
const FAULTS = [
  { value: 'none', label: 'None — healthy run' },
  { value: 'pressure_gauge_fault', label: 'Pressure-gauge fault' },
  { value: 'electrode_coupling_fault', label: 'Electrode-coupling fault' },
  { value: 'te_sensor_drift', label: 'Tₑ-sensor drift' },
];

const el = (tag, css, html) => { const n = document.createElement(tag); if (css) n.style.cssText = css; if (html !== undefined) n.innerHTML = html; return n; };
const pct0 = (f) => Math.round(f * 100) + '%';

function card(accent, title, rightLabel) {
  const c = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:22px 20px 20px;position:relative;overflow:hidden;display:flex;flex-direction:column;gap:14px');
  c.appendChild(el('div', `position:absolute;top:0;left:0;right:0;height:3px;background:${accent};opacity:.85`));
  c.appendChild(el('div', 'display:flex;align-items:center;justify-content:space-between',
    `<span style="font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--text-dim);font-weight:600">${title}</span>`
    + `<span style="font-family:var(--mono);font-size:9px;color:var(--text-mute)">${rightLabel}</span>`));
  const donutBox = el('div', 'position:relative;width:186px;height:186px;margin:2px auto 0');
  c.appendChild(donutBox);
  return { el: c, donutBox };
}

function probRow(name, frac, color) {
  const r = el('div', 'display:flex;align-items:center;gap:8px');
  r.innerHTML = `<span style="width:96px;flex:none;font-size:10px;color:${color.name}">${name}</span>`
    + `<div style="flex:1;height:6px;border-radius:3px;background:var(--border-2);overflow:hidden"><div style="height:100%;border-radius:3px;background:${color.bar};width:${clamp(frac, 0, 1) * 100}%;transition:width 450ms cubic-bezier(0.22,1,0.36,1)"></div></div>`
    + `<span style="width:38px;text-align:right;font-family:var(--mono);font-size:10px;color:var(--text)">${pct0(frac)}</span>`;
  return r;
}

export async function renderVerdict(main, { state, api }) {
  const rf = state.rf_power_w, p = state.pressure_mtorr;
  main.innerHTML = '';

  // ---- header ----
  main.appendChild(el('div', 'display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:6px',
    `<div><h1 style="margin:0;font-size:25px;font-weight:600;color:var(--text)">AI Verdict</h1>
       <p style="margin:5px 0 0;font-size:13px;color:var(--text-dim)">Reading the current operating point — <span class="mono" style="color:var(--purple)">${Math.round(rf)} W</span> · <span class="mono" style="color:var(--cyan)">${+p.toFixed(1)} mTorr</span></p></div>
     <span style="font-family:var(--mono);font-size:10px;letter-spacing:0.5px;color:var(--orange);border:1px solid rgba(251,146,60,0.35);background:rgba(251,146,60,0.06);border-radius:6px;padding:5px 10px">All three verdicts are live from the engine &amp; trained models</span>`));

  const pillsRow = el('div', 'display:flex;flex-wrap:wrap;gap:12px;margin-top:16px');
  main.appendChild(pillsRow);

  const cardsGrid = el('div', 'display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:16px');
  main.appendChild(cardsGrid);

  // ---- fetch the verdicts in parallel (SHAP is fetched separately below, since
  // its first call is slow and must not hold up the rest of the page) ----
  const [cls, clsXgb, anom0, sc] = await Promise.all([
    api.classify(rf, p),
    api.classify(rf, p, 'xgboost'),
    api.anomaly(rf, p, 'none'),
    api.suitabilityScorecard(rf, p),
  ]);
  const agree = cls.predicted_class === clsXgb.predicted_class;

  // ---- summary pills ----
  const clsColor = CLASS_COLOR[cls.predicted_class] || 'var(--text)';
  function pill(label, value, color) {
    return el('div', `flex:1 1 200px;display:flex;flex-direction:column;gap:4px;padding:13px 16px;border-radius:12px;background:var(--panel);border:1px solid var(--border);border-left:3px solid ${color}`,
      `<span style="font-size:9px;letter-spacing:1.4px;text-transform:uppercase;color:var(--text-dim)">${label}</span>`
      + `<span style="font-size:16px;font-weight:700;line-height:1.15;color:${color}">${value}</span>`);
  }
  const sevPill = pill('Anomaly Status', anom0.severity, SEVERITY[anom0.severity].color);
  pillsRow.append(
    pill('Suitability Class', cls.predicted_class, clsColor),
    sevPill,
    pill('Best-Fit Application', sc.best_application, 'var(--purple)'),
    pill('Model Agreement', agree ? 'RF = XGBoost' : 'RF ≠ XGBoost',
      agree ? 'var(--green)' : 'var(--amber)'),
  );

  // ---- Card 1: Suitability Class ----
  const c1 = card('var(--purple)', 'Suitability Class', 'RF / XGBoost');
  const g1 = createArcGauge({ strokeWidth: 8 });
  c1.donutBox.appendChild(g1.el);
  g1.update({ frac: cls.confidence, color: clsColor,
    centerHTML: `<div style="font-size:17px;font-weight:700;line-height:1.05;color:${clsColor}">${cls.predicted_class}</div><div class="mono" style="font-size:12px;color:var(--text-dim)">${pct0(cls.confidence)} conf.</div>` });
  const probsWrap = el('div', 'display:flex;flex-direction:column;gap:7px;margin-top:4px',
    '<span style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--text-mute);margin-bottom:2px">Class probabilities</span>');
  Object.entries(cls.class_probabilities).sort((a, b) => b[1] - a[1]).forEach(([name, pr]) => {
    probsWrap.appendChild(probRow(name, pr, { name: 'var(--text-dim)', bar: CLASS_COLOR[name] || 'var(--cyan)' }));
  });
  c1.el.appendChild(probsWrap);

  // Second-opinion row: XGBoost's independent verdict on the same operating point.
  const xgbColor = CLASS_COLOR[clsXgb.predicted_class] || 'var(--text)';
  const secondOpinion = el('div',
    `display:flex;align-items:center;gap:9px;margin-top:8px;padding:9px 11px;border-radius:9px;background:var(--panel-2);border:1px solid ${agree ? 'rgba(52,211,153,0.3)' : 'rgba(245,158,11,0.35)'}`);
  secondOpinion.innerHTML =
    `<span style="width:8px;height:8px;border-radius:50%;flex:none;background:${agree ? 'var(--green)' : 'var(--amber)'};box-shadow:0 0 7px ${agree ? 'var(--green)' : 'var(--amber)'}"></span>`
    + `<span style="font-size:10.5px;color:var(--text-mute);letter-spacing:0.4px;text-transform:uppercase;flex:none">2nd model</span>`
    + `<span style="font-size:12px;font-weight:600;color:${xgbColor}">XGBoost: ${clsXgb.predicted_class}</span>`
    + `<span class="mono" style="font-size:10.5px;color:var(--text-dim)">${pct0(clsXgb.confidence)}</span>`
    + `<span style="margin-left:auto;font-size:10px;font-weight:600;color:${agree ? 'var(--green)' : 'var(--amber)'}">${agree ? 'AGREES' : 'DIFFERS'}</span>`;
  c1.el.appendChild(secondOpinion);
  cardsGrid.appendChild(c1.el);

  // ---- Card 2: Anomaly Status (with fault injection) ----
  const c2 = card('var(--cyan)', 'Anomaly Status', 'Isolation Forest');
  const g2 = createArcGauge({ strokeWidth: 8 });
  c2.donutBox.appendChild(g2.el);
  const causeBox = el('div', '', '');
  const select = el('select', 'flex:1;background:var(--panel-2);color:var(--text);border:1px solid var(--border);border-radius:7px;padding:6px 8px;font-size:11px;cursor:pointer',
    FAULTS.map((f) => `<option value="${f.value}">${f.label}</option>`).join(''));
  const body2 = el('div', 'display:flex;flex-direction:column;gap:8px;margin-top:2px');
  body2.appendChild(el('span', 'font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--text-mute)', 'Root cause'));
  causeBox.style.cssText = 'min-height:34px;font-size:11.5px;line-height:1.45;background:var(--panel-2);border:1px solid var(--border-2);border-radius:8px;padding:8px 10px';
  body2.appendChild(causeBox);
  const faultRow = el('div', 'display:flex;align-items:center;gap:8px',
    '<span style="font-size:10px;color:var(--text-mute);letter-spacing:0.5px;text-transform:uppercase;flex:none">Inject fault</span>');
  faultRow.appendChild(select);
  body2.appendChild(faultRow);
  c2.el.appendChild(body2);
  cardsGrid.appendChild(c2.el);

  function paintAnomaly(anom) {
    const sev = SEVERITY[anom.severity];
    g2.update({ frac: sev.frac, color: sev.color,
      centerHTML: `<div style="font-size:19px;font-weight:700;color:${sev.color}">${anom.severity}</div><div class="mono" style="font-size:11px;color:var(--text-dim)">IF score ${anom.score.toFixed(2)}</div>` });
    const healthy = anom.fault === 'none';
    causeBox.textContent = healthy ? 'No fault injected — outputs consistent with logged inputs.' : anom.root_cause;
    causeBox.style.color = healthy ? 'var(--text-dim)' : sev.color;
    sevPill.querySelector('span:last-child').textContent = anom.severity;
    sevPill.querySelector('span:last-child').style.color = sev.color;
    sevPill.style.borderLeftColor = sev.color;
  }
  paintAnomaly(anom0);
  select.addEventListener('change', async () => {
    const anom = await api.anomaly(rf, p, select.value);
    paintAnomaly(anom);
  });

  // ---- Card 3: Best-Fit Application ----
  const c3 = card('var(--orange)', 'Best-Fit Application', 'windows');
  const g3 = createArcGauge({ strokeWidth: 8 });
  c3.donutBox.appendChild(g3.el);
  g3.update({ frac: sc.best_compliance_pct / 100, color: 'var(--purple)',
    centerHTML: `<div style="font-size:15px;font-weight:700;line-height:1.1;color:var(--text)">${sc.best_application}</div><div class="mono" style="font-size:12px;color:var(--purple)">${sc.best_compliance_pct.toFixed(0)}% window</div>` });
  const appWrap = el('div', 'display:flex;flex-direction:column;gap:7px;margin-top:4px',
    '<span style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--text-mute);margin-bottom:2px">Application compliance</span>');
  sc.ratings.slice().sort((a, b) => b.overall_compliance_pct - a.overall_compliance_pct).forEach((r, i) => {
    appWrap.appendChild(probRow(r.application, r.overall_compliance_pct / 100,
      { name: i === 0 ? 'var(--purple-2)' : 'var(--text-dim)', bar: i === 0 ? 'var(--purple)' : '#3a4656' }));
  });
  appWrap.appendChild(el('span', 'font-size:9.5px;color:var(--text-mute);margin-top:2px',
    `Ion energy ${sc.ion_energy_ev.toFixed(0)} eV · defect prob ${sc.defect_probability.toFixed(2)} · etching window (100+ eV) needs an applied RF-voltage drive.`));
  c3.el.appendChild(appWrap);
  cardsGrid.appendChild(c3.el);

  // ---- SHAP explanation panel (full width, loaded separately) ----
  const shapCard = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:22px 24px 20px;margin-top:18px;position:relative;overflow:hidden;display:flex;flex-direction:column;gap:12px');
  shapCard.appendChild(el('div', `position:absolute;top:0;left:0;right:0;height:3px;background:${clsColor};opacity:.85`));
  shapCard.appendChild(el('div', 'display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap',
    `<span style="font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--text-dim);font-weight:600">Why this verdict — SHAP feature contributions</span>`
    + `<span style="font-family:var(--mono);font-size:9px;color:var(--text-mute)">Random Forest · TreeExplainer</span>`));
  shapCard.appendChild(el('p', 'margin:0;font-size:12.5px;color:var(--text-dim);line-height:1.5',
    `How much each observable the classifier can see pushed this operating point toward or away from <strong style="color:${clsColor}">${cls.predicted_class}</strong>. The confounders (wall temperature, electrode aging, gas purity) are deliberately withheld from the model, so they cannot appear here.`));
  const shapBody = el('div', '');
  shapBody.innerHTML = `<div class="loading" style="padding:18px 2px"><span class="dot"></span> Fitting the SHAP explainer (first run only)…</div>`;
  shapCard.appendChild(shapBody);
  main.appendChild(shapCard);

  // Fetched after the page has painted: the first /api/explain call lazily fits
  // the SHAP-only estimators server-side and is slow; every later call is cached.
  api.explain(rf, p)
    .then((expl) => {
      shapBody.innerHTML = '';
      const bars = createShapBars();
      shapBody.appendChild(bars.el);
      bars.update(expl.feature_contributions, { predictedClass: expl.predicted_class });
    })
    .catch((e) => {
      shapBody.innerHTML = `<div class="err">SHAP explanation unavailable: ${e.message}</div>`;
    });

  return {};
}
