/* Reactor Control Room - API layer.
 *
 * The SINGLE place the frontend talks to the FastAPI backend. Every number the
 * UI ever shows comes through one of these functions; NOTHING in this app
 * computes physics or AI results in JavaScript. Each function maps 1:1 to a
 * backend endpoint that itself only wraps an existing project function, so the
 * companion app and the Streamlit dashboard are guaranteed to agree.
 *
 * Same-origin: the backend serves this frontend at `/`, so `/api/...` needs no
 * host or CORS. */

async function getJSON(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (_) { /* non-JSON error body */ }
    throw new Error(`${path} -> HTTP ${res.status}${detail ? ' · ' + detail : ''}`);
  }
  return res.json();
}

const q = (params) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');

export const api = {
  // Reactor View: the full 0D global-model output vector.
  simulate: (rf_power_w, pressure_mtorr, rf_voltage_v) =>
    getJSON(`/api/simulate?${q({ rf_power_w, pressure_mtorr, rf_voltage_v })}`),

  // AI Verdict: classifier verdict + confidence + class probabilities.
  classify: (rf_power_w, pressure_mtorr) =>
    getJSON(`/api/classify?${q({ rf_power_w, pressure_mtorr })}`),

  // AI Verdict: window-compliance scorecard (best-fit + compliance %).
  suitabilityScorecard: (rf_power_w, pressure_mtorr, rf_voltage_v) =>
    getJSON(`/api/suitability-scorecard?${q({ rf_power_w, pressure_mtorr, rf_voltage_v })}`),

  // AI Verdict: relationship-anomaly severity + score + root-cause (+ fault injection).
  anomaly: (rf_power_w, pressure_mtorr, fault) =>
    getJSON(`/api/anomaly?${q({ rf_power_w, pressure_mtorr, fault })}`),

  // Physics Validation: the literature benchmark checks (one row per check).
  physicsValidation: () => getJSON('/api/physics-validation'),

  // Session Replay: every stored experiment, newest first.
  sessions: () => getJSON('/api/sessions'),

  // Persistent strip: live host CPU% / RAM% (GPU honestly idle).
  systemStats: () => getJSON('/api/system/stats'),
};
