/* Physics Validation - the literature-benchmark checks (Sub-Module 1.6) rendered
 * as a pass-rate hero gauge, per-source reference cards, and one comparison bar
 * per check. Everything comes from /api/physics-validation (which serialises
 * run_literature_benchmarks, including each check's source citation). Nothing is
 * recomputed here - pass counts and source grouping are just tallies of fetched
 * fields. */
import { createArcGauge, createComparisonBar } from '../components.js';

const el = (tag, css, html) => { const n = document.createElement(tag); if (css) n.style.cssText = css; if (html !== undefined) n.innerHTML = html; return n; };

/** Split a long citation into a short "Author — 'Title'" and keep the rest as detail. */
function parseSource(src) {
  const qm = src.match(/'([^']+)'/);
  const title = qm ? qm[1] : src.slice(0, 46);
  const author = (qm ? src.slice(0, qm.index) : src).replace(/[,\s]+$/, '').trim();
  return { title, author };
}

export async function renderValidation(main, { api }) {
  main.innerHTML = '';

  main.appendChild(el('div', 'display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px',
    `<div><h1 style="margin:0;font-size:25px;font-weight:600;color:var(--text)">Physics Validation</h1>
       <p style="margin:5px 0 0;font-size:13px;color:var(--text-dim);max-width:560px">Model output at fixed reference conditions vs. independently published values, each against a tolerance band stated before the check was run.</p></div>`));

  const checks = await api.physicsValidation();
  const nPass = checks.filter((c) => c.passed).length;
  const nTotal = checks.length;

  const grid = el('div', 'display:grid;grid-template-columns:264px 1fr;gap:20px;align-items:start');
  main.appendChild(grid);

  // ---- left column: hero pass-rate gauge + reference sources ----
  const left = el('div', 'display:flex;flex-direction:column;gap:16px;position:sticky;top:0');
  grid.appendChild(left);

  const heroCard = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px;display:flex;flex-direction:column;align-items:center;gap:10px');
  heroCard.appendChild(el('span', 'font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--text-dim);font-weight:600;align-self:flex-start', 'Checks Passing'));
  const heroBox = el('div', 'position:relative;width:170px;height:170px;margin-top:4px');
  heroCard.appendChild(heroBox);
  const heroColor = nPass === nTotal ? 'var(--green)' : 'var(--amber)';
  const hero = createArcGauge({ strokeWidth: 8 });
  heroBox.appendChild(hero.el);
  hero.update({ frac: nPass / nTotal, color: heroColor,
    centerHTML: `<div class="mono" style="font-size:30px;font-weight:700;color:var(--text);line-height:1">${nPass}<span style="color:var(--text-mute);font-size:18px">/${nTotal}</span></div><div style="font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim)">within tolerance</div>` });
  heroCard.appendChild(el('span', 'font-size:10.5px;line-height:1.5;color:var(--text-dim);text-align:center', 'Model evaluated at fixed reference conditions; each check carries its own tolerance band, fixed before the result was seen.'));
  left.appendChild(heroCard);

  // reference sources, grouped from the checks' citations
  const groups = [];
  const bySrc = new Map();
  for (const c of checks) {
    if (!bySrc.has(c.source)) { const g = { source: c.source, total: 0, pass: 0 }; bySrc.set(c.source, g); groups.push(g); }
    const g = bySrc.get(c.source); g.total += 1; if (c.passed) g.pass += 1;
  }
  const srcCard = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px 18px;display:flex;flex-direction:column;gap:11px');
  srcCard.appendChild(el('div', 'display:flex;align-items:center;justify-content:space-between',
    `<span style="font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--text-dim);font-weight:600">Reference Sources</span><span style="font-family:var(--mono);font-size:9px;color:var(--text-mute)">${groups.length} active</span>`));
  groups.forEach((g, i) => {
    const { title, author } = parseSource(g.source);
    const item = el('div', 'display:flex;flex-direction:column;gap:5px;padding:10px 11px;border-radius:9px;border:1px solid #26456a;background:rgba(56,189,248,0.05)');
    item.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-family:var(--mono);font-size:8.5px;font-weight:700;letter-spacing:0.5px;padding:2px 6px;border-radius:4px;flex:none;color:var(--cyan);background:rgba(56,189,248,0.12)">S${i + 1}</span>
        <span style="font-size:11.5px;font-weight:600;color:var(--text)">${title}</span>
      </div>
      <span style="font-size:10px;line-height:1.5;color:var(--text-dim)">${author}</span>
      <span style="font-family:var(--mono);font-size:9px;color:var(--cyan)">${g.total} check${g.total > 1 ? 's' : ''} · ${g.pass}/${g.total} pass</span>`;
    srcCard.appendChild(item);
  });
  left.appendChild(srcCard);

  // ---- right column: one comparison bar per check ----
  const right = el('div', 'display:flex;flex-direction:column;gap:12px');
  grid.appendChild(right);
  for (const check of checks) {
    const bar = createComparisonBar();
    right.appendChild(bar.el);
    bar.update(check);
  }

  return {};
}
