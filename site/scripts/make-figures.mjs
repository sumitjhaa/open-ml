// Draws the figures for the notes written without lecture PDFs. Every number comes from lib/models.mjs, so figures match the interactives.
// Usage: node scripts/make-figures.mjs   (writes SVGs into ../notes/images/<note>/figures)
import { mkdirSync, writeFileSync } from 'node:fs';
import { gauss, linfit, rng, solve as solveG } from '../lib/math.mjs';
import { EIGHT_RUNS, LATENCIES, accuracy, agglomerate, buildTree, covariance2, dataset, dbscan, eig2, em1d, kmeans, linspace, logisticStep, mahalanobis, meanSilhouette, moons, normalPdf, predictTree, pruneTree, sigmoid, spreads, stats1, svd, medianOf, madOf } from '../lib/models.mjs';

import * as X from '../lib/extended.mjs';
import { normalCdf as cdfN } from '../lib/models.mjs';

const C = { blue: '#2f6fdf', orange: '#e07b24', green: '#2a9d6a', red: '#d0384b', ink: '#1d1d1f', muted: '#6e6e73', grid: '#e6e6ea', purple: '#7a4fd0' };
const TONES = [C.blue, C.orange, C.green, C.red, C.purple];
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

function panel({ x: [x0, x1], y: [y0, y1], title, xl, yl, ticks = true, equal }, draw) {
  return (ox, oy, W, H) => {
    const l = ox + (ticks ? 46 : 14), r = ox + W - 12, t = oy + 30, b = oy + H - (ticks ? 44 : 14);
    let bw = r - l, bh = b - t, L = l, T = t;
    if (equal) { const u = Math.min(bw / (x1 - x0), bh / (y1 - y0)); L = l + (bw - u * (x1 - x0)) / 2; T = t + (bh - u * (y1 - y0)) / 2; bw = u * (x1 - x0); bh = u * (y1 - y0); }
    const sx = v => +(L + ((v - x0) / (x1 - x0)) * bw).toFixed(2), sy = v => +(T + bh - ((v - y0) / (y1 - y0)) * bh).toFixed(2);
    const out = [`<text x="${ox + W / 2}" y="${oy + 18}" text-anchor="middle" font-size="14" font-weight="600" fill="${C.ink}">${esc(title ?? '')}</text>`];
    if (ticks) {
      for (const f of [0, 0.5, 1]) {
        const vx = x0 + f * (x1 - x0), vy = y0 + f * (y1 - y0);
        out.push(`<line x1="${L}" x2="${L + bw}" y1="${sy(vy)}" y2="${sy(vy)}" stroke="${C.grid}"/>`);
        out.push(`<text x="${sx(vx)}" y="${T + bh + 16}" text-anchor="middle" font-size="11" fill="${C.muted}">${+vx.toFixed(2)}</text>`, `<text x="${L - 6}" y="${sy(vy) + 4}" text-anchor="end" font-size="11" fill="${C.muted}">${+vy.toFixed(2)}</text>`);
      }
      if (xl) out.push(`<text x="${L + bw / 2}" y="${T + bh + 34}" text-anchor="middle" font-size="12" fill="${C.muted}">${esc(xl)}</text>`);
      if (yl) out.push(`<text transform="translate(${ox + 12} ${T + bh / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${C.muted}">${esc(yl)}</text>`);
    }
    const g = {
      line: (p, o = {}) => out.push(`<path d="M${p.map(q => `${sx(q[0])},${sy(q[1])}`).join('L')}" fill="${o.fill ?? 'none'}" fill-opacity="${o.fo ?? 1}" stroke="${o.c ?? C.blue}" stroke-width="${o.w ?? 2.4}" stroke-linejoin="round" stroke-linecap="round"${o.dash ? ' stroke-dasharray="6 5"' : ''}/>`),
      dots: (p, o = {}) => p.forEach(q => out.push(`<circle cx="${sx(q[0])}" cy="${sy(q[1])}" r="${o.r ?? 4.5}" fill="${q[2] ?? o.c ?? C.ink}" stroke="#fff" stroke-width="1"/>`)),
      ring: (p, o = {}) => p.forEach(q => out.push(`<circle cx="${sx(q[0])}" cy="${sy(q[1])}" r="${o.r ?? 11}" fill="none" stroke="${o.c ?? C.ink}" stroke-width="2"/>`)),
      arrow: (a, c2, o = {}) => { const X = sx(c2[0]), Y = sy(c2[1]), th = Math.atan2(Y - sy(a[1]), X - sx(a[0])), at = d => `${(X - 11 * Math.cos(th + d)).toFixed(1)},${(Y - 11 * Math.sin(th + d)).toFixed(1)}`; out.push(`<line x1="${sx(a[0])}" y1="${sy(a[1])}" x2="${X}" y2="${Y}" stroke="${o.c ?? C.ink}" stroke-width="2.4"/><polygon points="${X},${Y} ${at(0.42)} ${at(-0.42)}" fill="${o.c ?? C.ink}"/>`); },
      rect: (a, b2, c2, d, o = {}) => out.push(`<rect x="${Math.min(sx(a), sx(c2))}" y="${Math.min(sy(b2), sy(d))}" width="${Math.abs(sx(c2) - sx(a))}" height="${Math.abs(sy(d) - sy(b2))}" fill="${o.c ?? C.blue}" fill-opacity="${o.fo ?? 1}"/>`),
      text: (a, b2, s, o = {}) => out.push(`<text x="${sx(a)}" y="${sy(b2)}" text-anchor="${o.a ?? 'middle'}" font-size="${o.s ?? 12}" font-weight="${o.b ? 600 : 400}" fill="${o.c ?? C.ink}">${esc(s)}</text>`),
    };
    draw(g);
    return out.join('');
  };
}

function save(dir, name, panels, { w = 340, h = 300, cols = panels.length } = {}) {
  const rows = Math.ceil(panels.length / cols), W = cols * w, H = rows * h;
  const body = panels.map((p, i) => p((i % cols) * w, Math.floor(i / cols) * h, w, h)).join('');
  const path = `../notes/images/${dir}/figures`;
  mkdirSync(path, { recursive: true });
  writeFileSync(`${path}/${name}.svg`, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="-apple-system, Helvetica, Arial, sans-serif"><rect width="${W}" height="${H}" fill="#fff"/>${body}</svg>\n`);
}
const circle = linspace(0, 2 * Math.PI, 121).map(t => [Math.cos(t), Math.sin(t)]);
const mapp = (M, p) => [M[0][0] * p[0] + M[0][1] * p[1], M[1][0] * p[0] + M[1][1] * p[1]];
const blobs = (seed, centres, n, s) => { const r = rng(seed), out = []; centres.forEach((c, k) => { for (let i = 0; i < n; i++) out.push([c[0] + s * gauss(r), c[1] + s * gauss(r), k]); }); return out; };

/* ---------- SVD ---------- */
{
  const A = [[3, 0], [4, 5]], S = svd(A), V = S.V, U = S.U, s = S.s, Vt = [[V[0][0], V[1][0]], [V[0][1], V[1][1]]];
  const R0 = 7.2, box = { x: [-R0, R0], y: [-R0, R0], ticks: false, equal: true };
  const shape = (g, fn, a1, a2) => { g.line(circle.map(fn), { c: C.green, fill: C.green, fo: 0.1 }); g.arrow([0, 0], a1, { c: C.blue }); g.arrow([0, 0], a2, { c: C.orange }); };
  const v1 = [V[0][0], V[1][0]], v2 = [V[0][1], V[1][1]];
  save('L13', 'svd_rotate_stretch_rotate', [
    panel({ ...box, title: 'Unit circle, with v₁ and v₂' }, g => shape(g, p => p, v1, v2)),
    panel({ ...box, title: 'Rotate: Vᵀ' }, g => shape(g, p => mapp(Vt, p), [1, 0], [0, 1])),
    panel({ ...box, title: 'Stretch: Σ' }, g => shape(g, p => { const q = mapp(Vt, p); return [s[0] * q[0], s[1] * q[1]]; }, [s[0], 0], [0, s[1]])),
    panel({ ...box, title: 'Rotate: U' }, g => shape(g, p => mapp(A, p), [s[0] * U[0][0], s[0] * U[1][0]], [s[1] * U[0][1], s[1] * U[1][1]])),
  ], { w: 250, h: 270 });

  const N = 24, M = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { const u = (j - 11.5) / 12, v = (i - 11.5) / 12; return Math.abs(Math.hypot(u, v) - 0.6) < 0.16 ? 0.95 : Math.abs(v) < 0.09 ? -0.8 : u > 0.68 ? -0.5 : 0; })), D = svd(M), tot = D.s.reduce((a, b) => a + b * b, 0);
  const img = (X, title) => panel({ x: [0, N], y: [0, N], ticks: false, equal: true, title }, g => X.forEach((row, i) => row.forEach((v, j) => g.rect(j, N - i, j + 1.03, N - i - 1.03, { c: v > 0 ? C.blue : C.orange, fo: Math.min(Math.abs(v), 1) }))));
  const kept = k => `${Math.round((100 * D.s.slice(0, k).reduce((a, b) => a + b * b, 0)) / tot)}%`;
  save('L13', 'low_rank_layers', [...[1, 2, 4, 8].map(k => img(D.rank(k), `Rank ${k}: ${kept(k)} of the energy`)), img(M, 'Original matrix')], { w: 210, h: 240 });

  const clean = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => 0.7 * Math.sin((i / N) * 6.3) * Math.cos((j / N) * 3.1) + 0.5 * ((i + j) / N - 1))), r = rng(7), noisy = clean.map(row => row.map(v => v + 0.35 * gauss(r))), Sn = svd(noisy);
  const rm = (a, b) => Math.sqrt(a.flatMap((row, i) => row.map((v, j) => (v - b[i][j]) ** 2)).reduce((p, q) => p + q, 0) / (N * N)), ks = Array.from({ length: 16 }, (_, i) => i + 1), curve = ks.map(k => [k, rm(Sn.rank(k), clean), rm(Sn.rank(k), noisy)]), best = curve.reduce((a, b) => (b[1] < a[1] ? b : a));
  save('L13', 'denoising_u_curve', [panel({ x: [1, 16], y: [0, 0.5], title: 'Error against the rank kept', xl: 'rank k', yl: 'root mean squared error' }, g => { g.line(curve.map(c => [c[0], c[2]]), { c: C.green }); g.line(curve.map(c => [c[0], c[1]]), { c: C.red }); g.dots([[best[0], best[1]]], { c: C.red, r: 6 }); g.text(best[0] + 0.5, best[1] - 0.035, `lowest true error, rank ${best[0]}`, { a: 'start', c: C.red }); g.text(15.5, curve[15][2] - 0.03, 'against the noisy data', { a: 'end', c: C.green }); g.text(15.5, curve[15][1] + 0.025, 'against the truth', { a: 'end', c: C.red }); })], { w: 620, h: 340 });

  const sig = linspace(0.02, 3, 200);
  save('L13', 'ridge_filter', [panel({ x: [0, 3], y: [0, 6], title: 'How much each direction is amplified', xl: 'singular value σ', yl: 'amplification' }, g => { g.line(sig.map(q => [q, Math.min(1 / q, 7)]), { c: C.red }); g.line(sig.map(q => [q, q / (q * q + 0.1)]), { c: C.blue }); g.text(0.42, 4.7, 'least squares: 1/σ', { a: 'start', c: C.red }); g.text(1.25, 1.75, 'ridge: σ/(σ² + λ), λ = 0.1', { a: 'start', c: C.blue }); })], { w: 620, h: 320 });
}

/* ---------- clustering ---------- */
{
  const X = blobs(4, [[-2.2, -1.2], [2, -1.5], [0.2, 2.2]], 18, 0.55), P = X.map(p => [p[0], p[1]]), F = kmeans(P, 3, { seed: 22 });
  const frame = (f, t) => panel({ x: [-5, 5], y: [-5, 5], ticks: false, equal: true, title: t }, g => { P.forEach((p, i) => g.line([p, f.centres[f.assign[i]]], { c: '#d8d8de', w: 0.8 })); g.dots(P.map((p, i) => [p[0], p[1], TONES[f.assign[i]]])); g.ring(f.centres); g.dots(f.centres, { c: C.ink, r: 5 }); });
  save('L14', 'kmeans_steps', [frame(F[0], `Start: inertia ${F[0].inertia.toFixed(0)}`), frame(F[1], `After one update: ${F[1].inertia.toFixed(0)}`), frame(F.at(-1), `Converged: ${F.at(-1).inertia.toFixed(0)}`)], { w: 280, h: 290 });

  const rows = Array.from({ length: 8 }, (_, i) => { const k = i + 1, b = [1, 2, 3, 4, 5].map(s => kmeans(P, k, { seed: s * 13, init: 'plus' }).at(-1)).reduce((u, v) => (v.inertia < u.inertia ? v : u)); return [k, b.inertia, k > 1 ? meanSilhouette(P, b.assign) : null]; });
  save('L14', 'choosing_k', [
    panel({ x: [1, 8], y: [0, rows[0][1] * 1.05], title: 'Inertia always falls', xl: 'k', yl: 'inertia' }, g => { g.line(rows.map(r => [r[0], r[1]])); g.dots(rows.map(r => [r[0], r[1]]), { c: C.blue }); g.dots([[3, rows[2][1]]], { c: C.red, r: 7 }); g.text(3.3, rows[2][1] + rows[0][1] * 0.14, 'the bend, at k = 3', { a: 'start', c: C.red }); }),
    panel({ x: [2, 8], y: [0, 1], title: 'Silhouette has a peak', xl: 'k', yl: 'average silhouette' }, g => { const q = rows.slice(1); g.line(q.map(r => [r[0], r[2]]), { c: C.orange }); g.dots(q.map(r => [r[0], r[2]]), { c: C.orange }); g.dots([[3, rows[2][2]]], { c: C.red, r: 7 }); }),
  ], { w: 340, h: 300 });

  const r = rng(29), n = () => gauss(r), long = [], rings = [], un = [];
  for (let i = 0; i < 90; i++) { const y = i % 2, t = 2 * Math.PI * r(), R0 = y ? 4 : 1.5; long.push([3.2 * n(), (y ? 1.3 : -1.3) + 0.25 * n()]); rings.push([R0 * Math.cos(t) + 0.15 * n(), R0 * Math.sin(t) + 0.15 * n()]); }
  for (let i = 0; i < 90; i++) un.push([-1.5 + 1.4 * n(), 1.4 * n()]);
  for (let i = 0; i < 12; i++) un.push([3.6 + 0.25 * n(), 3 + 0.25 * n()]);
  const fm = (D, t) => panel({ x: [-6, 6], y: [-6, 6], ticks: false, equal: true, title: t }, g => { const f = [1, 2, 3, 4, 5, 6].map(s => kmeans(D, 2, { seed: s * 7, init: 'plus' }).at(-1)).reduce((u, v) => (v.inertia < u.inertia ? v : u)); g.dots(D.map((p, i) => [p[0], p[1], TONES[f.assign[i]]]), { r: 3.4 }); g.ring(f.centres); });
  save('L14', 'kmeans_failures', [fm(long, 'Elongated: cut across'), fm(rings, 'Rings: split in half'), fm(un, 'Unequal: large group split')], { w: 280, h: 290 });

  const pts = [[1], [2], [4], [8]], tree = (link, t) => panel({ x: [0.5, 4.5], y: [0, 8], title: t, yl: 'merge height' }, g => { const m = agglomerate(pts, link), pos = new Map([[0, [1, 0]], [1, [2, 0]], [2, [3, 0]], [3, [4, 0]]]); m.forEach(q => { const A = pos.get(q.a), B = pos.get(q.b); g.line([[A[0], A[1]], [A[0], q.h], [B[0], q.h], [B[0], B[1]]]); g.text((A[0] + B[0]) / 2, q.h + 0.25, q.h.toFixed(0), { c: C.red, b: true }); pos.set(q.id, [(A[0] + B[0]) / 2, q.h]); }); pts.forEach((p, i) => g.text(i + 1, -0.0 + 0.28, '', {})); });
  save('L14', 'dendrograms', [tree('single', 'Single linkage on 1, 2, 4, 8'), tree('complete', 'Complete linkage on 1, 2, 4, 8')], { w: 340, h: 300 });

  const q = rng(41), U = Array.from({ length: 70 }, () => [-4.5 + 9 * q(), -4.5 + 9 * q()]), fu = kmeans(U, 3, { seed: 3, init: 'plus' }).at(-1), fr = kmeans(P, 3, { seed: 3, init: 'plus' }).at(-1);
  const ph = (D, f, t) => panel({ x: [-5, 5], y: [-5, 5], ticks: false, equal: true, title: `${t}: silhouette ${meanSilhouette(D, f.assign).toFixed(2)}` }, g => { g.dots(D.map((p, i) => [p[0], p[1], TONES[f.assign[i]]])); g.ring(f.centres); });
  save('L14', 'phantom_clusters', [ph(P, fr, 'Real groups'), ph(U, fu, 'Uniform noise')], { w: 330, h: 320 });
}

/* ---------- features, diagnostics, validation ---------- */
{
  const XOR = dataset('xor', 41, 60), fit = phi => { const X = XOR.map(d => phi(d[0], d[1])), y = XOR.map(d => d[2]); let w = Array(X[0].length).fill(0); for (let i = 0; i < 40; i++) w = logisticStep(w, X, y, 'gd', 0.8); return (a, b) => sigmoid(phi(a, b).reduce((s, v, i) => s + v * w[i], 0)); };
  const fld = (p, t) => panel({ x: [-3, 3], y: [-3, 3], ticks: false, equal: true, title: t }, g => { const n = 46, d = 6 / n; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const a = -3 + i * d, b = -3 + j * d, v = 2 * p(a + d / 2, b + d / 2) - 1; g.rect(a, b, a + d * 1.04, b + d * 1.04, { c: v > 0 ? C.orange : C.blue, fo: Math.abs(v) * 0.55 }); } g.dots(XOR.map(s => [s[0], s[1], s[2] ? C.orange : C.blue]), { r: 4 }); });
  save('L12g', 'interaction_xor', [fld(fit((a, b) => [1, a, b]), 'Features x₁, x₂: no line works'), fld(fit((a, b) => [1, a, b, a * b]), 'Adding x₁x₂: solved')], { w: 330, h: 320 });

  const bx = [1, 2, 3, 4, 5, 6, 7], by = [1.4, 2.2, 2.9, 4.3, 4.8, 6.2, 6.9], lev = (ex, ey, t) => panel({ x: [0, 15], y: [0, 15], title: t, xl: 'x', yl: 'y' }, g => { const a = linfit(bx, by), b = linfit([...bx, ex], [...by, ey]); g.line([[0, a.b], [15, a.w * 15 + a.b]], { c: C.green, dash: true }); g.line([[0, b.b], [15, b.w * 15 + b.b]], { c: C.blue }); g.dots(bx.map((q, i) => [q, by[i]])); g.dots([[ex, ey]], { c: C.red, r: 7 }); });
  save('L12g', 'leverage', [lev(4, 11, 'Outlier in y, low leverage'), lev(14, 14, 'High leverage, on the trend'), lev(14, 3, 'High leverage, off the trend')], { w: 280, h: 290 });

  const k = 20, eta = 1.2 / k, run = beta => { let p = [2.8, 1.6], v = [0, 0]; const o = [p]; for (let i = 0; i < 40; i++) { const g = [p[0], k * p[1]]; v = v.map((u, j) => beta * u - eta * g[j]); p = p.map((u, j) => u + v[j]); o.push(p); } return o; };
  save('L12g', 'momentum_paths', [panel({ x: [-3, 3], y: [-2, 2], ticks: false, equal: true, title: 'Plain descent and momentum in a long valley' }, g => { for (const c of [0.5, 1.5, 3.5, 6, 10]) g.line(linspace(0, 6.3, 121).map(t => [Math.sqrt(2 * c) * Math.cos(t), Math.sqrt((2 * c) / k) * Math.sin(t)]), { c: C.grid, w: 1.2 }); g.line(run(0), { c: C.green }); g.line(run(0.7), { c: C.red }); g.text(-2.9, -1.75, 'green: plain descent, red: momentum 0.7', { a: 'start', c: C.muted }); })], { w: 620, h: 400 });

  const ball = (n, t, c) => panel({ x: [-1.5, 1.5], y: [-1.5, 1.5], ticks: false, equal: true, title: t }, g => { g.line([[-1.5, 0], [1.5, 0]], { c: C.grid, w: 1 }); g.line([[0, -1.5], [0, 1.5]], { c: C.grid, w: 1 }); g.line(linspace(0, 2 * Math.PI, 241).map(a => { const co = Math.cos(a), si = Math.sin(a), d = n(co, si); return [co / d, si / d]; }), { c, fill: c, fo: 0.12 }); });
  save('L12g', 'unit_balls', [ball((a, b) => Math.abs(a) + Math.abs(b), 'Manhattan', C.blue), ball(() => 1, 'Euclidean', C.green), ball((a, b) => Math.max(Math.abs(a), Math.abs(b)), 'Chebyshev', C.orange)], { w: 230, h: 240 });

  const N = 24, cols = { 'k-fold': (i, f) => (i * 7) % 4 === f, group: (i, f) => Math.floor(i / 3) % 4 === f, 'forward chaining': (i, f) => i >= 8 + 4 * f && i < 12 + 4 * f };
  save('L12g', 'validation_designs', Object.entries(cols).map(([name, fn]) => panel({ x: [0, N], y: [0, 4], ticks: false, title: name }, g => { for (let f = 0; f < 4; f++) for (let i = 0; i < N; i++) { const unused = name === 'forward chaining' && i >= 12 + 4 * f; g.rect(i + 0.08, 3 - f + 0.12, i + 0.92, 4 - f - 0.12, { c: fn(i, f) ? C.red : unused ? '#d8d8de' : C.blue }); } })), { w: 620, h: 130, cols: 1 });

  const D = dataset('overlap', 31, 90), T = dataset('overlap', 32, 400), X = D.map(d => [d[0], d[1]]), y = D.map(d => d[2]), deep = buildTree(X, y, { maxDepth: 12 }), A = linspace(-3, -0.7, 24).map(l => { const p = pruneTree(deep, 10 ** l, X, y); return [l, accuracy(x => predictTree(p.node, x), D), accuracy(x => predictTree(p.node, x), T), p.leaves]; });
  const bestP = A.reduce((a, b) => (b[2] > a[2] ? b : a));
  save('L12g', 'pruning_curve', [panel({ x: [-3, -0.7], y: [0.5, 1], title: 'Accuracy as the price per leaf rises', xl: 'log₁₀ α', yl: 'accuracy' }, g => { g.line(A.map(a => [a[0], a[1]]), { c: C.green }); g.line(A.map(a => [a[0], a[2]]), { c: C.red }); g.dots([[bestP[0], bestP[2]]], { c: C.red, r: 6 }); g.text(-2.9, 0.54, 'green: training, red: test', { a: 'start', c: C.muted }); })], { w: 620, h: 320 });

  const ns = linspace(20, 2000, 200), iv = p => ns.map(n => [n, 1.96 * Math.sqrt((p * (1 - p)) / n)]);
  save('L12g', 'test_score_interval', [panel({ x: [20, 2000], y: [0, 0.2], title: 'Half-width of a 95% interval', xl: 'test cases', yl: 'half-width' }, g => { g.line(iv(0.5), { c: C.blue }); g.line(iv(0.9), { c: C.orange }); g.dots([[400, 1.96 * Math.sqrt(0.09 / 400)]], { c: C.red, r: 6 }); g.text(440, 0.052, 'accuracy 0.9 on 400 cases: ±0.029', { a: 'start', c: C.red }); g.text(1950, 0.038, 'accuracy 0.5', { a: 'end', c: C.blue }); g.text(1950, 0.004, 'accuracy 0.9', { a: 'end', c: C.orange }); })], { w: 620, h: 300 });
}
/* ---------- unsupervised handbook and K-means slides ---------- */
{
  const R8 = EIGHT_RUNS, F = kmeans(R8, 2, { init: [[2, 3], [1, 3]] }), all = spreads(R8, F.at(-1).assign), box8 = { x: [0, 10], y: [0, 11], ticks: false, equal: true };
  save('L14', 'eight_runs_lloyd', F.slice(0, 4).map((f, i) => panel({ ...box8, title: `Iteration ${i + 1}: J = ${f.inertia.toFixed(2)}` }, g => { R8.forEach((p, q) => g.line([p, f.centres[f.assign[q]]], { c: '#d8d8de', w: 1 })); g.dots(R8.map((p, q) => [p[0], p[1], TONES[f.assign[q]]]), { r: 6 }); g.ring(f.centres); g.dots(f.centres, { c: C.ink, r: 4 }); })), { w: 230, h: 260 });
  const mean8 = [5.125, 5.5], final = F.at(-1);
  save('L14', 'within_between', [
    panel({ ...box8, title: `T = ${all.T.toFixed(3)}: every run to the overall mean` }, g => { R8.forEach(p => g.line([p, mean8], { c: C.purple, w: 1.4 })); g.dots(R8, { r: 6, c: C.ink }); g.dots([mean8], { c: C.purple, r: 8 }); }),
    panel({ ...box8, title: `W = ${all.W.toFixed(2)} (solid) + B = ${all.B.toFixed(3)} (dashed)` }, g => { R8.forEach((p, q) => g.line([p, final.centres[final.assign[q]]], { c: TONES[final.assign[q]], w: 1.4 })); final.centres.forEach(c => g.line([c, mean8], { c: C.purple, dash: true })); g.dots(R8.map((p, q) => [p[0], p[1], TONES[final.assign[q]]]), { r: 6 }); g.dots(final.centres, { c: C.ink, r: 6 }); g.dots([mean8], { c: C.purple, r: 8 }); }),
  ], { w: 360, h: 360 });
  const D4 = [0, 1, 2, 4], J4 = (a, b) => D4.reduce((s, x) => s + Math.min((x - a) ** 2, (x - b) ** 2), 0);
  save('L14', 'local_minima_valley', [panel({ x: [-1, 5], y: [-1, 5], equal: true, title: 'J over the two centroids, for data 0, 1, 2, 4', xl: 'centroid a', yl: 'centroid b' }, g => { const n = 72, d = 6 / n; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const a = -1 + i * d, b = -1 + j * d, v = Math.min(J4(a + d / 2, b + d / 2), 30); g.rect(a, b, a + d * 1.05, b + d * 1.05, { c: C.blue, fo: +(1 - Math.log(1 + v) / Math.log(31)).toFixed(3) }); } for (const [a, b, s] of [[0.5, 3, 'J = 2.5'], [1, 4, 'J = 2'], [3, 0.5, 'J = 2.5'], [4, 1, 'J = 2']]) { g.dots([[a, b]], { c: C.orange, r: 6 }); g.text(a + 0.18, b + 0.1, s, { a: 'start', c: C.ink, b: true }); } })], { w: 520, h: 520 });

  const lat = LATENCIES, E = em1d(lat, [15, 25], 4, 40), dens = mu => linspace(0, 48, 241).map(x => [x, 0.5 * normalPdf(x, mu[0], 4) + 0.5 * normalPdf(x, mu[1], 4)]);
  save('L15', 'em_density', [panel({ x: [0, 48], y: [0, 0.07], title: 'The mixture density before and after EM', xl: 'latency (ms)', yl: 'density' }, g => { g.line(dens(E[0].mu), { c: C.muted, dash: true }); g.line(dens(E[40].mu), { c: C.blue }); g.dots(lat.map(x => [x, 0.004]), { c: C.orange, r: 6 }); g.text(46, 0.063, 'solid: means 14.2 and 32.8', { a: 'end', c: C.blue }); g.text(46, 0.056, 'dashed: start, means 15 and 25', { a: 'end', c: C.muted }); })], { w: 640, h: 330 });
  const rs = s => linspace(5, 35, 241).map(x => [x, 1 / (1 + Math.exp((((x - 15) ** 2 - (x - 25) ** 2) / (2 * s * s))))]);
  save('L15', 'responsibility_sigma', [panel({ x: [5, 35], y: [0, 1], title: 'Responsibility of the fast component, means 15 and 25', xl: 'latency (ms)', yl: 'responsibility' }, g => { [[8, C.green, 'σ = 8', 28, 0.26], [4, C.blue, 'σ = 4', 25.5, 0.13], [2, C.orange, 'σ = 2', 22, 0.6], [0.6, C.red, 'σ = 0.6', 19.5, 0.86]].forEach(([s, c, t, a, b]) => { g.line(rs(s), { c }); g.text(a, b, t, { a: 'start', c, b: true }); }); })], { w: 640, h: 320 });
  const M8 = agglomerate(R8, 'average'), order = [], kids = new Map(M8.map(m => [m.id, m])), vis = id => { if (id < 8) order.push(id); else { vis(kids.get(id).a); vis(kids.get(id).b); } };
  vis(M8.at(-1).id);
  const pos = new Map(order.map((id, i) => [id, { x: i, h: 0 }]));
  save('L15', 'dendrogram_eight', [panel({ x: [-0.6, 7.6], y: [-0.6, 7.2], title: 'Average linkage on the eight runs', yl: 'merge height' }, g => { for (const m of M8) { const A = pos.get(m.a), B = pos.get(m.b); g.line([[A.x, A.h], [A.x, m.h], [B.x, m.h], [B.x, B.h]], { c: C.blue }); pos.set(m.id, { x: (A.x + B.x) / 2, h: m.h }); } order.forEach((id, i) => g.text(i, -0.45, `run ${id + 1}`, { c: C.muted, s: 11 })); g.line([[-0.6, 4.4], [7.6, 4.4]], { c: C.red, dash: true }); g.text(7.5, 4.6, 'cut: two clusters', { a: 'end', c: C.red }); })], { w: 640, h: 360 });
  const MN = moons(11, 200, 0.08), XY = MN.map(p => [p[0], p[1]]), KM = [1, 2, 3, 4].map(s => kmeans(XY, 2, { seed: s, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a)), DB = dbscan(XY, 0.3, 5), mb = { x: [-1.6, 2.6], y: [-1.2, 1.6], ticks: false, equal: true };
  save('L15', 'moons_kmeans_dbscan', [
    panel({ ...mb, title: 'K-means: a straight boundary' }, g => g.dots(XY.map((p, i) => [p[0], p[1], TONES[KM.assign[i]]]), { r: 4 })),
    panel({ ...mb, title: `DBSCAN: ${DB.clusters} clusters follow the moons` }, g => { g.dots(XY.map((p, i) => [p[0], p[1], DB.labels[i] < 0 ? C.muted : TONES[DB.labels[i]]]).filter((_, i) => DB.labels[i] >= 0), { r: 4 }); XY.forEach((p, i) => { if (DB.labels[i] < 0) { g.line([[p[0] - 0.04, p[1] - 0.04], [p[0] + 0.04, p[1] + 0.04]], { c: C.red, w: 1.6 }); g.line([[p[0] - 0.04, p[1] + 0.04], [p[0] + 0.04, p[1] - 0.04]], { c: C.red, w: 1.6 }); } }); }),
  ], { w: 340, h: 300 });

  const S = covariance2(R8), e = eig2(S.a, S.b, S.d), th = Math.atan2(e.l1 - S.a, S.b), v1 = [Math.cos(th), Math.sin(th)], v2 = [-v1[1], v1[0]], varAt = t => S.a * Math.cos(t) ** 2 + 2 * S.b * Math.sin(t) * Math.cos(t) + S.d * Math.sin(t) ** 2, deg = (th * 180) / Math.PI;
  save('L16', 'variance_by_angle', [panel({ x: [0, 180], y: [0, 13], title: 'Variance kept along each direction', xl: 'angle with the first feature (degrees)', yl: 'variance' }, g => { g.line(linspace(0, 180, 181).map(d => [d, varAt((d * Math.PI) / 180)])); g.dots([[deg, e.l1]], { c: C.red, r: 6 }); g.dots([[deg + 90, e.l2]], { c: C.orange, r: 6 }); g.text(deg + 4, e.l1 + 0.6, `maximum ${e.l1.toFixed(3)} at ${deg.toFixed(1)}°`, { a: 'start', c: C.red }); g.text(deg + 86, e.l2 + 0.9, `minimum ${e.l2.toFixed(3)}`, { a: 'end', c: C.orange }); })], { w: 640, h: 320 });
  const m = S.m;
  save('L16', 'projection_line', [panel({ x: [-1, 11], y: [-1, 12], equal: true, title: 'Projection onto the first principal direction', xl: 'training time', yl: 'peak memory' }, g => { g.line([[m[0] - 8 * v1[0], m[1] - 8 * v1[1]], [m[0] + 8 * v1[0], m[1] + 8 * v1[1]]], { c: C.blue }); R8.forEach(p => { const z = (p[0] - m[0]) * v1[0] + (p[1] - m[1]) * v1[1], f = [m[0] + z * v1[0], m[1] + z * v1[1]]; g.line([p, f], { c: C.red, dash: true, w: 1.6 }); g.ring([f], { r: 5, c: C.blue }); }); g.dots(R8, { r: 6, c: C.ink }); g.dots([m], { r: 7, c: C.orange }); })], { w: 520, h: 520 });
  const L = [5.452, 3.341, 0.202, 0.18, 0.167, 0.153, 0.151, 0.128, 0.119, 0.107]; let cum = 0;
  const C10 = L.map((v, i) => [i + 1, (cum += v) / 10]);
  save('L16', 'scree_cumulative', [
    panel({ x: [0.4, 10.6], y: [0, 6], title: 'Scree plot', xl: 'component', yl: 'eigenvalue' }, g => { L.forEach((v, i) => g.rect(i + 0.7, 0, i + 1.3, v, { c: i < 2 ? C.blue : '#b9bcc6' })); g.line([[0.4, 1], [10.6, 1]], { c: C.red, dash: true }); g.text(10.5, 1.2, 'eigenvalue = 1', { a: 'end', c: C.red }); }),
    panel({ x: [1, 10], y: [0.4, 1], title: 'Cumulative proportion explained', xl: 'components', yl: 'proportion' }, g => { g.line(C10); g.dots(C10, { c: C.blue }); g.line([[1, 0.9], [10, 0.9]], { c: C.red, dash: true }); g.text(9.9, 0.86, '90%', { a: 'end', c: C.red }); }),
  ], { w: 340, h: 300 });
  const dist = linspace(0, 6, 121), gs = d => Math.exp(-d * d / 2), ts = d => 1 / (1 + d * d);
  save('L16', 'gaussian_vs_t', [
    panel({ x: [0, 6], y: [0, 1], title: 'Similarity against distance', xl: 'distance', yl: 'similarity' }, g => { g.line(dist.map(d => [d, gs(d)]), { c: C.blue }); g.line(dist.map(d => [d, ts(d)]), { c: C.orange }); g.text(1.8, 0.1, 'Gaussian', { a: 'end', c: C.blue, b: true }); g.text(2.2, 0.32, 'Student-t', { a: 'start', c: C.orange, b: true }); }),
    panel({ x: [0, 6], y: [-8, 0], title: 'Same curves, log₁₀ scale', xl: 'distance', yl: 'log₁₀ similarity' }, g => { g.line(dist.map(d => [d, Math.max(-8, Math.log10(gs(d)))]), { c: C.blue }); g.line(dist.map(d => [d, Math.log10(ts(d))]), { c: C.orange }); g.dots([[4, Math.log10(gs(4))], [4, Math.log10(ts(4))]], { c: C.red, r: 6 }); }),
  ], { w: 340, h: 300 });

  const A8 = [...lat, 95], a = stats1(A8), b = stats1(lat), med = medianOf(A8), rs3 = (3 * madOf(A8)) / 0.6745, rows = [['mean ± 3 sd, all eight', a.m - 3 * a.s, a.m + 3 * a.s, A8, C.orange], ['mean ± 3 sd, without 95', b.m - 3 * b.s, b.m + 3 * b.s, lat, C.blue], ['median ± 3 robust sd, all eight', med - rs3, med + rs3, A8, C.green]];
  save('L17', 'masking_bands', [panel({ x: [-60, 130], y: [0, 3.6], title: 'Which values fall inside the band', xl: 'latency (ms)' }, g => rows.forEach(([t, lo, hi, d, c], i) => { const y = 2.75 - i * 1.15; g.rect(lo, y - 0.28, hi, y + 0.28, { c, fo: 0.2 }); g.dots(lat.map(x => [x, y]), { c: C.ink, r: 4.5 }); g.dots([[95, y]], { c: hi >= 95 ? C.muted : C.red, r: 7 }); if (!d.includes(95)) g.ring([[95, y]], { c: C.red, r: 10 }); g.text(-58, y + 0.42, t, { a: 'start', c, b: true }); }))], { w: 680, h: 340 });
  const P = [2, 8], Q = [11, 11], Cv = S, ell = d => linspace(0, 2 * Math.PI, 121).map(t => [m[0] + d * Math.sqrt(e.l1) * Math.cos(t) * v1[0] + d * Math.sqrt(e.l2) * Math.sin(t) * v2[0], m[1] + d * Math.sqrt(e.l1) * Math.cos(t) * v1[1] + d * Math.sqrt(e.l2) * Math.sin(t) * v2[1]]), circ = d => linspace(0, 2 * Math.PI, 121).map(t => [m[0] + d * Math.cos(t), m[1] + d * Math.sin(t)]), bx = { x: [-3, 14], y: [-3, 14], ticks: false, equal: true };
  const pts = g => { g.dots(R8, { r: 5, c: C.ink }); g.dots([P], { r: 7, c: C.red }); g.dots([Q], { r: 7, c: C.orange }); g.text(P[0] - 0.4, P[1] + 0.5, 'P', { c: C.red, b: true, s: 15 }); g.text(Q[0] + 0.1, Q[1] + 0.6, 'Q', { c: C.orange, b: true, s: 15 }); };
  save('L17', 'euclid_mahalanobis', [
    panel({ ...bx, title: `Euclidean: P ${Math.hypot(P[0] - m[0], P[1] - m[1]).toFixed(3)}, Q ${Math.hypot(Q[0] - m[0], Q[1] - m[1]).toFixed(3)}` }, g => { [2, 4, 6, 8].forEach(d => g.line(circ(d), { c: C.blue, w: 1.4 })); pts(g); }),
    panel({ ...bx, title: `Mahalanobis: P ${mahalanobis(P, Cv).toFixed(3)}, Q ${mahalanobis(Q, Cv).toFixed(3)}` }, g => { [1, 2, 3, 4].forEach(d => g.line(ell(d), { c: C.green, w: 1.4 })); pts(g); }),
  ], { w: 360, h: 360 });
  const iso = blobs(6, [[0, 0]], 60, 1).map(p => [Math.max(-3.4, Math.min(3.4, p[0])), Math.max(-3.4, Math.min(3.4, p[1]))]), inner = iso.reduce((u, v) => (Math.hypot(v[0], v[1]) < Math.hypot(u[0], u[1]) ? v : u)), out3 = [4.2, 3.8], data3 = [...iso, out3];
  const cutsFor = (target, seed) => { const r = rng(seed), cuts = []; let S3 = data3, lo = [-4.8, -4.8], hi = [4.8, 4.8]; while (S3.length > 1 && cuts.length < 40) { const f = r() < 0.5 ? 0 : 1, v = S3.map(p => p[f]), a0 = Math.min(...v), a1 = Math.max(...v); if (a0 === a1) continue; const c = a0 + r() * (a1 - a0); cuts.push([f, c, [...lo], [...hi]]); if (target[f] < c) { hi[f] = c; S3 = S3.filter(p => p[f] < c); } else { lo[f] = c; S3 = S3.filter(p => p[f] >= c); } } return cuts; };
  const cutPanel = (target, seed, t) => { const cuts = cutsFor(target, seed); return panel({ x: [-4.8, 4.8], y: [-4.8, 4.8], ticks: false, equal: true, title: `${t}: alone after ${cuts.length} cuts` }, g => { cuts.forEach(([f, c, a0, a1]) => g.line(f === 0 ? [[c, a0[1]], [c, a1[1]]] : [[a0[0], c], [a1[0], c]], { c: C.orange, w: 1.4 })); g.dots(data3, { r: 3.5, c: C.ink }); g.dots([target], { r: 6, c: C.red }); g.ring([target], { c: C.red, r: 11 }); }); };
  save('L17', 'isolation_cuts', [cutPanel(out3, 3, 'Outlier'), cutPanel(inner, 5, 'Typical point')], { w: 340, h: 320 });
}
/* ---------- additions from ES 335 (2025) ---------- */
{
  const fmtN = v => (Number.isInteger(v) ? String(v) : v.toFixed(2));
  // matrix factorisation
  const f1 = X.als(X.RATINGS, 1, { sweeps: 60 }).at(-1), P1 = X.mfPredict(f1.W, f1.H), mw = 5, mh = 3;
  const table = (t, get, tint) => panel({ x: [0, mw], y: [0, mh], ticks: false, equal: true, title: t }, g => { for (let i = 0; i < mh; i++) for (let j = 0; j < mw; j++) { const v = get(i, j), y = mh - i - 1; g.rect(j + 0.04, y + 0.04, j + 0.96, y + 0.96, { c: tint(i, j), fo: 0.25 }); g.text(j + 0.5, y + 0.4, v === null ? '?' : fmtN(v), { b: v === null, s: 13 }); } });
  const known = (i, j) => X.RATINGS[i][j] !== null;
  save('L18', 'mf_fill', [table('Known ratings', (i, j) => X.RATINGS[i][j], (i, j) => (known(i, j) ? C.blue : C.muted)), table('Rank 1 fit', (i, j) => +P1[i][j].toFixed(2), () => C.green), table('Filled table', (i, j) => (known(i, j) ? X.RATINGS[i][j] : +P1[i][j].toFixed(2)), (i, j) => (known(i, j) ? C.blue : C.orange))], { w: 270, h: 200 });
  const runs = [1, 2, 3, 4].map(s => X.als(X.RATINGS, 1, { sweeps: 12, seed: s * 5 }).map(f => f.loss));
  save('L18', 'als_loss', [panel({ x: [0, 12], y: [0, Math.min(40, Math.max(...runs.map(r => r[0])))], title: 'Loss on the known cells', xl: 'sweep', yl: 'squared error' }, g => { runs.forEach((r, i) => g.line(r.map((v, t) => [t, Math.min(40, v)]), { c: TONES[i] })); g.text(11.8, 6, 'every start ends at 3.36', { a: 'end', c: C.muted }); })], { w: 620, h: 300 });
  // neighbours
  const K0 = [...blobs(3, [[-2.2, -1], [1.5, 2]], 11, 0.8), ...blobs(5, [[1.8, -1.4], [-1.5, 2.2]], 11, 0.8)], Pts = K0.map(p => [p[0], p[1]]), Lab = K0.map((_, i) => +(i >= 22));
  const knnP = K => panel({ x: [-5, 5], y: [-4, 4], ticks: false, equal: true, title: `K = ${K}` }, g => { const n = 56, dx = 10 / n, dy = 8 / 45; for (let i = 0; i < n; i++) for (let j = 0; j < 45; j++) { const a = -5 + i * dx, b = -4 + j * dy, v = X.knnVote(Pts, Lab, [a + dx / 2, b + dy / 2], K); g.rect(a, b, a + dx * 1.04, b + dy * 1.04, { c: v ? C.orange : C.blue, fo: 0.28 }); } g.dots(Pts.map((p, i) => [p[0], p[1], Lab[i] ? C.orange : C.blue]), { r: 4 }); });
  save('L19', 'knn_k', [knnP(1), knnP(5), knnP(25)], { w: 280, h: 240 });
  // naive Bayes
  const m1 = 6.1, s1 = 0.6, m2 = 5.3, s2 = 0.9, xr = linspace(2, 10, 321);
  save('L20', 'gaussian_nb', [panel({ x: [2, 10], y: [0, 0.8], title: 'Densities and the regions each class wins', xl: 'height', yl: 'density' }, g => { let start = xr[0], cur = normalPdf(xr[0], m1, s1) > normalPdf(xr[0], m2, s2); xr.forEach(x => { const a = normalPdf(x, m1, s1) > normalPdf(x, m2, s2); if (a !== cur || x === xr.at(-1)) { g.rect(start, 0, x, 0.8, { c: cur ? C.blue : C.orange, fo: 0.12 }); start = x; cur = a; } }); g.line(xr.map(x => [x, normalPdf(x, m1, s1)]), { c: C.blue }); g.line(xr.map(x => [x, normalPdf(x, m2, s2)]), { c: C.orange }); g.text(7.1, 0.6, 'class A: mean 6.1, spread 0.6', { a: 'start', c: C.blue }); g.text(7.1, 0.52, 'class B: mean 5.3, spread 0.9', { a: 'start', c: C.orange }); })], { w: 640, h: 320 });
  // networks
  const XOR2 = [[0, 0], [0, 1], [1, 0], [1, 1]], Y2 = [0, 1, 1, 0], hand = { W1: [[1, 1], [1, 1]], b1: [0, -1], w2: [1, -2], b2: 0 };
  save('L21', 'xor_hidden', [
    panel({ x: [-0.5, 1.5], y: [-0.5, 1.5], equal: true, title: 'Original inputs', xl: 'x₁', yl: 'x₂' }, g => g.dots(XOR2.map((p, i) => [p[0], p[1], Y2[i] ? C.orange : C.blue]), { r: 10 })),
    panel({ x: [-0.5, 2.5], y: [-0.5, 1.5], equal: true, title: 'After the hidden layer', xl: 'h₁', yl: 'h₂' }, g => { g.line([[-0.5, -0.5], [2.5, 1]], { c: C.red }); g.dots(XOR2.map((p, i) => { const h = X.mlpForward(hand, p).h; return [h[0], h[1], Y2[i] ? C.orange : C.blue]; }), { r: 10 }); }),
  ], { w: 340, h: 320 });
  const MN2 = moons(5, 120, 0.1), XM2 = MN2.map(p => [p[0], p[1]]), YM2 = MN2.map(p => p[2]), nnP = (act, t) => { const f = X.trainMlp(XM2, YM2, { hidden: 8, act, steps: 2000, every: 2000, seed: 3 }).at(-1); return panel({ x: [-1.7, 2.7], y: [-1.3, 1.7], ticks: false, equal: true, title: t }, g => { const n = 60, dx = 4.4 / n, dy = 3 / 41; for (let i = 0; i < n; i++) for (let j = 0; j < 41; j++) { const a = -1.7 + i * dx, b = -1.3 + j * dy, v = 2 * X.mlpForward(f.p, [a + dx / 2, b + dy / 2], act).p - 1; g.rect(a, b, a + dx * 1.04, b + dy * 1.04, { c: v > 0 ? C.orange : C.blue, fo: +(Math.abs(v) * 0.4).toFixed(3) }); } g.dots(MN2.map(p => [p[0], p[1], p[2] ? C.orange : C.blue]), { r: 4 }); }); };
  save('L21', 'mlp_moons', [nnP('none', 'No activation: a straight line'), nnP('relu', 'ReLU: a curved boundary')], { w: 340, h: 260 });
  // autodiff graph
  const t0 = new X.Value(1), t1 = new X.Value(1), t2 = new X.Value(2), f1n = t1.mul(1), f2n = t2.mul(2), f3 = f1n.add(f2n), f4 = f3.add(t0), f5 = f4.neg(), f6 = f5.exp(), f7 = f6.add(1), f8 = f7.inv(), f9 = f8.log(), L = f9.neg(); L.backward();
  const nodes = [['θ₁x₁', f1n], ['θ₂x₂', f2n], ['+', f3], ['+θ₀', f4], ['×−1', f5], ['exp', f6], ['+1', f7], ['1/x', f8], ['log', f9], ['×−1', L]];
  save('L22', 'graph_logistic', [panel({ x: [0, 10.4], y: [0, 3], ticks: false, title: 'Values forward (blue), gradients backward (red)' }, g => nodes.forEach(([n, v], i) => { const x = i + 0.5; g.rect(x - 0.42, 1, x + 0.42, 2, { c: C.grid, fo: 1 }); g.text(x, 1.42, n, { b: true, s: 12 }); g.text(x, 2.22, v.data.toFixed(3), { c: C.blue, s: 11 }); g.text(x, 0.68, v.grad.toFixed(3), { c: C.red, s: 11 }); if (i < 9) g.arrow([x + 0.44, 1.5], [x + 0.56, 1.5], { c: C.muted }); }))], { w: 900, h: 220 });
  // convolution
  const edge = Array.from({ length: 6 }, () => [1, 1, 1, 0, 0, 0]), kern = [[1, 0, -1], [1, 0, -1], [1, 0, -1]], out = X.conv2d(edge, kern), grid = (M, t, c) => panel({ x: [0, M[0].length], y: [0, M.length], ticks: false, equal: true, title: t }, g => M.forEach((row, i) => row.forEach((v, j) => { const y = M.length - i - 1; g.rect(j + 0.03, y + 0.03, j + 0.97, y + 0.97, { c: v < 0 ? C.red : c, fo: v === 0 ? 0.06 : 0.45 }); g.text(j + 0.5, y + 0.36, String(v), { s: 13 }); })));
  save('L23', 'edge_filter', [grid(edge, 'Image', C.blue), grid(kern, 'Filter', C.orange), grid(out, 'Response', C.green)], { w: 270, h: 280 });
  // equivariance and IoU
  const base = Array.from({ length: 8 }, () => Array(8).fill(0)); [[2, 2], [3, 2], [4, 2], [4, 3], [4, 4]].forEach(([a, b]) => { base[a][b] = 1; });
  const sh = (M, k) => M.map(r => r.map((_, j) => r[j - k] ?? 0)), hm = (M, t) => panel({ x: [0, M[0].length], y: [0, M.length], ticks: false, equal: true, title: t }, g => { const pk = Math.max(1, ...M.flat().map(Math.abs)); M.forEach((row, i) => row.forEach((v, j) => { const y = M.length - i - 1; if (v) g.rect(j, y, j + 1, y + 1, { c: v < 0 ? C.red : C.blue, fo: +(Math.abs(v) / pk).toFixed(3) }); })); });
  save('L24', 'equivariance', [hm(X.conv2d(sh(base, 2), kern, 1, 1), 'Shift, then convolve'), hm(sh(X.conv2d(base, kern, 1, 1), 2), 'Convolve, then shift')], { w: 300, h: 300 });
  const boxes = panel({ x: [0, 120], y: [0, 120], ticks: false, equal: true, title: 'IoU = 2500 / 7300 = 0.342' }, g => { g.rect(10, 40, 80, 110, { c: C.blue, fo: 0.25 }); g.rect(30, 20, 100, 90, { c: C.orange, fo: 0.25 }); g.rect(30, 40, 80, 90, { c: C.red, fo: 0.45 }); g.text(55, 62, 'overlap', { b: true }); });
  const curveI = panel({ x: [0, 70], y: [0, 1], title: 'IoU as one box slides', xl: 'shift in each direction', yl: 'IoU' }, g => { g.line(linspace(0, 70, 141).map(s => [s, X.iou([0, 0, 70, 70], [s, s, 70 + s, 70 + s])]), { c: C.blue }); g.line([[0, 0.5], [70, 0.5]], { c: C.red, dash: true }); g.text(68, 0.53, 'common threshold', { a: 'end', c: C.red }); });
  save('L24', 'iou', [boxes, curveI], { w: 340, h: 320 });
  // next token
  const Tr = X.NAMES.filter((_, i) => i % 5), He = X.NAMES.filter((_, i) => !(i % 5)), sc = [1, 2, 3, 4].map(o => { const m = X.ngram(Tr, o); return [o, m.nll(Tr), m.nll(He)]; });
  save('L25', 'ngram_loss', [panel({ x: [1, 4], y: [0, 3.6], title: 'Loss against context length', xl: 'context length', yl: 'nats' }, g => { g.line(sc.map(s => [s[0], s[1]]), { c: C.green }); g.line(sc.map(s => [s[0], s[2]]), { c: C.red }); g.line([[1, Math.log(27)], [4, Math.log(27)]], { c: C.muted, dash: true }); g.text(3.95, Math.log(27) - 0.16, 'uniform guess', { a: 'end', c: C.muted }); g.text(3.95, sc[3][1] + 0.22, 'training', { a: 'end', c: C.green }); g.text(3.95, sc[3][2] + 0.16, 'held-out', { a: 'end', c: C.red }); })], { w: 620, h: 320 });
  // time series
  const Y0 = X.monthly(), Tn = Y0.slice(0, 96), Te = Y0.slice(96), col = { mean: C.muted, naive: C.purple, seasonal: C.orange, drift: C.green };
  save('L26', 'baselines', [panel({ x: [60, 120], y: [70, 160], title: 'Four baselines on the held-out two years', xl: 'month' }, g => { g.line(Tn.slice(60).map((v, i) => [60 + i, v]), { c: C.ink, w: 1.6 }); g.line(Te.map((v, i) => [96 + i, v]), { c: C.ink, w: 1.6, dash: true }); Object.keys(col).forEach(k => g.line(X.BASELINES[k](Tn, 24).map((v, i) => [96 + i, v]), { c: col[k] })); ['mean', 'naive', 'seasonal', 'drift'].forEach((k, i) => g.text(62, 155 - 8 * i, k, { a: 'start', c: col[k], b: true })); })], { w: 680, h: 320 });
  save('L26', 'rnn_gradient', [panel({ x: [1, 60], y: [-10, 8], title: 'Gradient size against steps back', xl: 'steps back', yl: 'log₁₀ size' }, g => { [[0.7, C.blue], [0.9, C.green], [1, C.muted], [1.1, C.orange], [1.3, C.red]].forEach(([r, c], i) => { g.line(linspace(1, 60, 60).map(k => [k, Math.max(-10, Math.min(8, k * Math.log10(r)))]), { c }); g.text(58, 7 - i * 1.3, `ρ = ${r}`, { a: 'end', c, b: true }); }); })], { w: 620, h: 320 });
  // reinforcement learning
  const stuck = X.mountainCar(() => 2), back = X.mountainCar(t => (t <= 40 ? 0 : 2));
  save('L27', 'mountain_car', [panel({ x: [0, 200], y: [-1.2, 0.7], title: 'Position over time', xl: 'step', yl: 'position' }, g => { g.line(stuck.path, { c: C.red }); g.line(back.path, { c: C.blue }); g.line([[0, 0.5], [200, 0.5]], { c: C.green, dash: true }); g.text(195, 0.56, 'flag', { a: 'end', c: C.green }); g.text(195, stuck.path.at(-1)[1] + 0.12, 'always push right', { a: 'end', c: C.red }); g.text(back.reached + 4, 0.2, 'back up, then push right', { a: 'start', c: C.blue }); })], { w: 640, h: 320 });
  const V = X.valueIteration(X.GRID, 0.9, 20).at(-1), G = X.GRID, H5 = G.length;
  save('L27', 'value_iteration', [panel({ x: [0, 5], y: [0, 5], ticks: false, equal: true, title: 'Values and best moves' }, g => G.forEach((row, a) => [...row].forEach((c, b) => { const x = b, y = H5 - a - 1; if (c === '#') { g.rect(x, y, x + 1, y + 1, { c: C.ink, fo: 0.8 }); return; } if (c === 'G' || c === 'P') { g.rect(x, y, x + 1, y + 1, { c: c === 'G' ? C.green : C.red, fo: 0.5 }); g.text(x + 0.5, y + 0.42, c === 'G' ? '+1' : '−1', { b: true, s: 15 }); return; } g.rect(x + 0.03, y + 0.03, x + 0.97, y + 0.97, { c: C.blue, fo: +(V[a][b] * 0.55).toFixed(3) }); g.text(x + 0.5, y + 0.62, V[a][b].toFixed(2), { s: 12 }); const Q = X.MOVES.map((_, m) => { const o = X.gridStep(G, [a, b], m); return o.r + (o.done ? 0 : 0.9 * V[o.s[0]][o.s[1]]); }), best = Q.indexOf(Math.max(...Q)); g.arrow([x + 0.5, y + 0.3], [x + 0.5 + X.MOVES[best][1] * 0.3, y + 0.3 - X.MOVES[best][0] * 0.22], { c: C.ink }); })))], { w: 460, h: 480 });
}
/* ---------- training, attention, generative, probabilistic, systems ---------- */
{
  const lr = 0.035, g = ([x, y]) => [x, 25 * y], run = k => { let p = [-4.5, 0.9], v = [0, 0], s = [0, 0]; const out = [p]; for (let t = 1; t <= 80; t++) { const d = g(p); if (k === 'sgd') p = p.map((a, i) => a - lr * d[i]); if (k === 'mom') { v = v.map((a, i) => 0.9 * a + d[i]); p = p.map((a, i) => a - lr * v[i]); } if (k === 'adam') { v = v.map((a, i) => 0.9 * a + 0.1 * d[i]); s = s.map((a, i) => 0.999 * a + 0.001 * d[i] ** 2); p = p.map((a, i) => a - (lr * 10 * (v[i] / (1 - 0.9 ** t))) / (Math.sqrt(s[i] / (1 - 0.999 ** t)) + 1e-8)); } out.push(p); } return out; };
  save('L29', 'optimizer_paths', [panel({ x: [-5.5, 2], y: [-1.5, 1.5], title: 'Three optimisers on one valley', ticks: false }, c => { for (const l of [0.5, 2, 5, 10, 18]) c.line(linspace(0, 6.3, 121).map(t => [Math.sqrt(2 * l) * Math.cos(t), Math.sqrt((2 * l) / 25) * Math.sin(t)]), { c: C.grid, w: 1.2 }); [['sgd', C.red, 'plain'], ['mom', C.green, 'momentum'], ['adam', C.blue, 'Adam']].forEach(([k, col, n], i) => { c.line(run(k), { c: col, w: 1.8 }); c.text(1.9, 1.3 - i * 0.22, n, { a: 'end', c: col, b: true }); }); })], { w: 680, h: 300 });
  const W = ['the', 'animal', 'was', 'tired'], A = [[0.55, 0.15, 0.15, 0.15], [0.1, 0.6, 0.15, 0.15], [0.1, 0.5, 0.3, 0.1], [0.05, 0.7, 0.1, 0.15]];
  save('L30', 'attention_heat', [panel({ x: [-1.2, 4], y: [0, 5], ticks: false, equal: true, title: 'Attention weights (illustrative)' }, c => { A.forEach((row, i) => { c.text(-0.1, 3 - i + 0.4, W[i], { a: 'end' }); row.forEach((v, j) => { c.rect(j + 0.04, 3 - i + 0.04, j + 0.96, 3 - i + 0.96, { c: C.blue, fo: +v.toFixed(2) }); c.text(j + 0.5, 3 - i + 0.4, v.toFixed(2), { s: 12 }); }); }); W.forEach((w, j) => c.text(j + 0.5, 4.25, w, { s: 12 })); })], { w: 520, h: 420 });
  const mix = (x, a) => 0.5 * normalPdf(x, -2 * Math.sqrt(a), Math.sqrt(1 - a + 0.09 * a)) + 0.5 * normalPdf(x, 2 * Math.sqrt(a), Math.sqrt(1 - a + 0.09 * a));
  save('L31', 'diffusion_forward', [1, 0.6, 0.2, 0.01].map(a => panel({ x: [-4, 4], y: [0, 0.8], title: `ᾱ = ${a}`, ticks: false }, c => { c.line(linspace(-4, 4, 201).map(x => [x, mix(x, a)]), { c: C.blue, fill: C.blue, fo: 0.15 }); })), { w: 210, h: 180 });
  const xs = [-3.2, -1.9, -0.4, 1.1, 2.6], ys = [-0.3, 0.9, 0.2, -0.8, 0.5], rbf = (a, b) => Math.exp(-((a - b) ** 2) / 2), K = xs.map((a, i) => xs.map((b, j) => rbf(a, b) + (i === j ? 0.01 : 0))), al = solveG(K, ys), gx = linspace(-5, 5, 201), P = gx.map(x => { const k = xs.map(a => rbf(x, a)), v = solveG(K, k); return [x, k.reduce((s, c, i) => s + c * al[i], 0), Math.sqrt(Math.max(0, 1 - k.reduce((s, c, i) => s + c * v[i], 0)))]; });
  save('L32', 'gp_posterior', [panel({ x: [-5, 5], y: [-3, 3], title: 'A Gaussian process posterior', xl: 'x' }, c => { c.line([...P.map(p => [p[0], p[1] + 2 * p[2]]), ...[...P].reverse().map(p => [p[0], p[1] - 2 * p[2]])], { c: C.blue, fill: C.blue, fo: 0.15, w: 0.5 }); c.line(P.map(p => [p[0], p[1]]), { c: C.blue }); c.dots(xs.map((x, i) => [x, ys[i]]), { c: C.red, r: 6 }); })], { w: 640, h: 300 });
  const ts = linspace(0, 1, 101), tpr = (t, m) => 1 - cdfN((t - m) / 0.18), fp = (t, m) => 1 - cdfN((t - m) / 0.18);
  save('L33', 'fairness_rates', [panel({ x: [0, 1], y: [0, 1], title: 'Error rates against one threshold (illustrative)', xl: 'threshold', yl: 'rate' }, c => { c.line(ts.map(t => [t, tpr(t, 0.6)]), { c: C.blue }); c.line(ts.map(t => [t, tpr(t, 0.55)]), { c: C.orange }); c.line(ts.map(t => [t, fp(t, 0.35)]), { c: C.blue, dash: true }); c.line(ts.map(t => [t, fp(t, 0.3)]), { c: C.orange, dash: true }); c.text(0.98, 0.93, 'solid: true positive rate, dashed: false positive rate', { a: 'end', c: C.muted }); c.text(0.98, 0.85, 'blue: group A, orange: group B', { a: 'end', c: C.muted }); })], { w: 640, h: 320 });
}
{
  const es = linspace(-4, 4, 161);
  save('D01', 'noise_models', [panel({ x: [-4, 4], y: [0, 8], title: 'Loss against error', xl: 'error', yl: 'loss' }, c => { c.line(es.map(e => [e, e * e / 2]), { c: C.blue }); c.line(es.map(e => [e, Math.abs(e)]), { c: C.orange }); c.text(3.9, 7.4, 'Gaussian: squared', { a: 'end', c: C.blue, b: true }); c.text(3.9, 2.9, 'Laplace: absolute', { a: 'end', c: C.orange, b: true }); }), panel({ x: [-4, 4], y: [0, 0.55], title: 'Implied noise', xl: 'error', yl: 'density' }, c => { c.line(es.map(e => [e, normalPdf(e, 0, 1)]), { c: C.blue }); c.line(es.map(e => [e, 0.5 * Math.exp(-Math.abs(e))]), { c: C.orange }); })], { w: 340, h: 280 });
  const sm = v => Math.exp(v) / (Math.exp(v) + Math.exp(-v * 0.6));
  save('D03', 'cbm_intervention', [panel({ x: [0, 1], y: [0, 1], title: 'Probability of the class as one concept is corrected', xl: 'concept value', yl: 'class probability' }, c => { const L = linspace(0, 1, 101); c.line(L.map(v => [v, sm(4 * v - 2)]), { c: C.blue }); c.dots([[0.2, sm(0.8 - 2)]], { c: C.red, r: 7 }); c.dots([[0.9, sm(3.6 - 2)]], { c: C.green, r: 7 }); c.text(0.25, sm(0.8 - 2) + 0.03, 'predicted concept', { a: 'start', c: C.red }); c.text(0.86, sm(3.6 - 2) - 0.06, 'after correction', { a: 'end', c: C.green }); })], { w: 620, h: 300 });
  const f = 0.02;
  save('D04', 'dice_vs_ce', [panel({ x: [0, 1], y: [0, 1], title: 'A 2% foreground mask', xl: 'share of foreground found', yl: 'score' }, c => { const R0 = linspace(0, 1, 101); c.line(R0.map(r => [r, (2 * f * r) / (f * r + f)]), { c: C.blue }); c.line(R0.map(r => [r, 1 - f * (1 - r)]), { c: C.orange }); c.text(0.98, 0.9, 'pixel accuracy', { a: 'end', c: C.orange, b: true }); c.text(0.5, 0.48, 'Dice', { a: 'start', c: C.blue, b: true }); })], { w: 620, h: 300 });
}
/* ---------- pytorch, architectures, language models, diffusion, graphs, careers ---------- */
{
  const L0 = linspace(0, 30, 31), lr = linspace(0, 1000, 401), w = 0.1;
  save('D02', 'init_variance', [panel({ x: [0, 30], y: [-8, 8], title: 'Activation variance with depth (ReLU)', xl: 'layer', yl: 'log₁₀ variance' }, c => { [[0.5, C.red, 'too small'], [1, C.blue, 'He'], [1.4, C.orange, 'too large']].forEach(([s, col, t]) => { c.line(L0.map(k => [k, Math.max(-8, Math.min(8, k * Math.log10(s)))]), { c: col }); c.text(28, Math.max(-7.4, Math.min(7.4, 28 * Math.log10(s))) + 0.7, t, { a: 'end', c: col, b: true }); }); })], { w: 620, h: 300 });
  const warm = s => { const t = s / 1000; return t < w ? t / w : 0.5 * (1 + Math.cos((Math.PI * (t - w)) / (1 - w))); };
  save('D05', 'lr_schedules', [panel({ x: [0, 1000], y: [0, 1.05], title: 'Learning-rate schedules', xl: 'step', yl: 'share of base rate' }, c => { c.line(lr.map(s => [s, 0.1 ** Math.floor((s / 1000) * 3)]), { c: C.orange }); c.line(lr.map(s => [s, 0.5 * (1 + Math.cos((Math.PI * s) / 1000))]), { c: C.muted }); c.line(lr.map(s => [s, warm(s)]), { c: C.blue, w: 2 }); c.text(170, 0.3, 'warmup, then cosine', { c: C.blue, b: true }); c.text(980, 0.16, 'step', { a: 'end', c: C.orange, b: true }); })], { w: 620, h: 300 });
  const Z = [3.2, 2.7, 2.1, 1.6, 0.9, 0.2, -0.6, -1.5], Ts = linspace(0.1, 3, 200);
  save('D06', 'temperature', [panel({ x: [0.1, 3], y: [0, 1], title: 'Temperature against token probability', xl: 'temperature', yl: 'probability' }, c => { [[0, C.blue, 'most likely token'], [1, C.orange, 'second'], [7, C.red, 'least likely']].forEach(([i, col, t]) => { c.line(Ts.map(T => [T, X.softmaxT(Z, T)[i]]), { c: col }); c.text(2.95, X.softmaxT(Z, 2.8)[i] + 0.05, t, { a: 'end', c: col, b: true }); }); })], { w: 620, h: 300 });
  const resid = (() => { let p = 1, r = 1, gp = 1, gr = 1; const P = [[0, 0]], Q = [[0, 0]]; for (let i = 1; i <= 60; i++) { gp *= 0.8 * (1 - Math.tanh(0.8 * p) ** 2); p = Math.tanh(0.8 * p); gr *= 1 + 0.08 * (1 - Math.tanh(0.8 * r) ** 2); r += 0.1 * Math.tanh(0.8 * r); P.push([i, Math.max(-12, Math.log10(gp))]); Q.push([i, Math.log10(gr)]); } return { P, Q }; })();
  save('D07', 'residual_gradient', [panel({ x: [0, 60], y: [-12, 2], title: 'Gradient reaching each layer', xl: 'layer', yl: 'log₁₀ gradient' }, c => { c.line(resid.P, { c: C.red }); c.line(resid.Q, { c: C.blue }); c.text(42, -7, 'plain', { c: C.red, b: true }); c.text(42, 1, 'residual', { c: C.blue, b: true }); })], { w: 620, h: 300 });
  const A = []; { let a = 1; for (let t = 0; t < 1000; t++) { a *= 1 - (1e-4 + ((0.02 - 1e-4) * t) / 999); A.push(a); } }
  save('D08', 'signal_noise', [panel({ x: [0, 1000], y: [0, 1], title: 'Signal and noise in the forward process', xl: 'step t', yl: 'coefficient' }, c => { c.line(A.map((a, t) => [t, Math.sqrt(a)]), { c: C.blue }); c.line(A.map((a, t) => [t, Math.sqrt(1 - a)]), { c: C.orange }); c.text(120, 0.72, 'signal', { c: C.blue, b: true }); c.text(800, 0.82, 'noise', { c: C.orange, b: true }); })], { w: 620, h: 300 });
  const smooth = (() => { const n = 9, adj = Array.from({ length: n }, (_, i) => [i - 1, i + 1].filter(j => j >= 0 && j < n)); let h = adj.map((_, i) => (i === 1 ? 1 : 0)); const O = []; for (let k = 0; k <= 40; k++) { const m = h.reduce((a, b) => a + b) / n; O.push([k, Math.sqrt(h.reduce((a, b) => a + (b - m) ** 2, 0) / n)]); h = h.map((x, i) => (x + adj[i].reduce((s, u) => s + h[u], 0)) / (1 + adj[i].length)); } return O; })();
  save('D09', 'oversmoothing', [panel({ x: [0, 40], y: [0, 0.32], title: 'Differences between nodes shrink with rounds', xl: 'rounds', yl: 'spread of node values' }, c => { c.line(smooth, { c: C.blue }); })], { w: 620, h: 300 });
  const ps = linspace(0.001, 0.5, 300);
  save('C01', 'base_rate', [panel({ x: [0, 0.5], y: [0, 1], title: 'A positive result from a 95% accurate test', xl: 'prevalence', yl: 'P(condition | positive)' }, c => { c.line(ps.map(p => [p, (0.95 * p) / (0.95 * p + 0.05 * (1 - p))]), { c: C.blue }); })], { w: 620, h: 300 });
  const us = linspace(0, 0.97, 200);
  save('C02', 'latency_utilisation', [panel({ x: [0, 1], y: [0, 34], title: 'Latency as a server fills up', xl: 'utilisation', yl: 'latency, in service times' }, c => { c.line(us.map(u => [u, 1 / (1 - u)]), { c: C.blue }); })], { w: 620, h: 300 });
}
/* ---------- topics beyond the source courses ---------- */
{
  const line = (fn, a, b, n = 301) => linspace(a, b, n).map(x => [x, fn(x)]);
  const W = { w: 620, h: 300 };
  save('E01', 'pinball_loss', [panel({ x: [-3, 3], y: [0, 3], title: 'Pinball loss for two quantiles', xl: 'error', yl: 'loss' }, c => { c.line(line(e => (e >= 0 ? 0.9 * e : -0.1 * e), -3, 3), { c: C.blue }); c.line(line(e => (e >= 0 ? 0.5 * e : -0.5 * e), -3, 3), { c: C.orange }); c.text(2.9, 2.4, 'τ = 0.9', { a: 'end', c: C.blue, b: true }); c.text(2.9, 1.1, 'τ = 0.5', { a: 'end', c: C.orange, b: true }); })], W);
  save('E02', 'ucb_bonus', [panel({ x: [1, 200], y: [0, 3.5], title: 'The UCB bonus shrinks with pulls', xl: 'pulls of this arm', yl: 'bonus at t = 1000' }, c => { c.line(line(n => Math.sqrt(2 * Math.log(1000) / n), 1.5, 200), { c: C.blue }); })], W);
  save('E03', 'soft_threshold', [panel({ x: [-3, 3], y: [-3, 3], title: 'Soft thresholding with λη = 1', xl: 'input', yl: 'output' }, c => { c.line(line(v => v, -3, 3, 3), { c: C.muted }); c.line(line(v => Math.sign(v) * Math.max(Math.abs(v) - 1, 0), -3, 3), { c: C.blue, w: 2 }); })], W);
  const h = p => (p <= 0 || p >= 1 ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p)));
  save('E04', 'epistemic', [panel({ x: [0, 1], y: [0, 0.75], title: 'Two members, one predicting 0.1 and the other p', xl: 'prediction p', yl: 'nats' }, c => { c.line(line(p => h((0.1 + p) / 2), 0, 1), { c: C.blue }); c.line(line(p => (h(0.1) + h(p)) / 2, 0, 1), { c: C.orange }); c.text(0.98, 0.68, 'total', { a: 'end', c: C.blue, b: true }); c.text(0.5, 0.25, 'aleatoric', { c: C.orange, b: true }); })], W);
  save('E05', 'langevin_target', [panel({ x: [-3, 3], y: [0, 2.2], title: 'Double-well energy and its density (shape only)', xl: 'x' }, c => { c.line(line(x => Math.min(2.2, (x * x - 1) ** 2), -3, 3), { c: C.orange }); c.line(line(x => Math.exp(-((x * x - 1) ** 2)), -3, 3), { c: C.blue }); c.text(0, 0.7, 'density', { a: 'middle', c: C.blue, b: true }); c.text(2.7, 2, 'energy', { a: 'end', c: C.orange, b: true }); })], W);
  save('E06', 'kernel_decay', [panel({ x: [0, 10], y: [0, 1], title: 'Error along two kernel directions', xl: 'time', yl: 'remaining error' }, c => { c.line(line(t => Math.exp(-3 * t), 0, 10), { c: C.blue }); c.line(line(t => Math.exp(-0.2 * t), 0, 10), { c: C.orange }); c.text(1, 0.15, 'λ = 3', { c: C.blue, b: true }); c.text(7, 0.4, 'λ = 0.2', { c: C.orange, b: true }); })], W);
  save('E07', 'transmittance', [panel({ x: [0, 4], y: [0, 1], title: 'Transmittance through uniform density', xl: 'distance', yl: 'T' }, c => { [[0.5, C.blue], [1, C.green], [3, C.red]].forEach(([s, col]) => { c.line(line(t => Math.exp(-s * t), 0, 4), { c: col }); c.text(3.9, Math.exp(-s * 3) + 0.05, 'σ = ' + s, { a: 'end', c: col, b: true }); }); })], W);
}
{
  const L = (fn, a, b, n = 301) => linspace(a, b, n).map(x => [x, fn(x)]);
  save('F01', 'holt_components', [panel({ x: [0, 60], y: [30, 110], title: 'Trend and season', xl: 'month', yl: 'value' }, c => { c.line(L(t => 50 + 0.7 * t + 12 * Math.sin((2 * Math.PI * t) / 12), 0, 60), { c: C.blue }); c.line(L(t => 50 + 0.7 * t, 0, 60, 3), { c: C.orange }); c.text(58, 66, 'trend', { a: 'end', c: C.orange, b: true }); })], { w: 620, h: 300 });
  save('F02', 'dilated_field', [panel({ x: [0, 8], y: [0, 520], title: 'Receptive field against layers', xl: 'layers', yl: 'field' }, c => { c.line(L(k => 1 + 2 * k, 0, 8), { c: C.orange }); c.line(L(k => 1 + 2 * (2 ** k - 1), 0, 8), { c: C.blue }); c.text(7.8, 430, 'dilation doubling', { a: 'end', c: C.blue, b: true }); c.text(7.8, 60, 'plain', { a: 'end', c: C.orange, b: true }); })], { w: 620, h: 300 });
  save('G01', 'rosenbrock', [panel({ x: [-1.5, 1.5], y: [0, 40], title: 'Along the valley and across it', xl: 'distance moved', yl: 'loss' }, c => { c.line(L(t => (1 - (1 + t)) ** 2, -1.5, 1.5), { c: C.blue }); c.line(L(t => Math.min(40, 100 * t * t), -0.63, 0.63), { c: C.red }); })]);
  save('G02', 'causal_reach', [panel({ x: [1, 10], y: [0, 1100], title: 'Receptive field against layers', xl: 'layers', yl: 'field' }, c => { c.line(L(k => 1 + k, 1, 10), { c: C.orange }); c.line(L(k => 2 ** k, 1, 10), { c: C.blue }); })]);
  save('G03', 'smoothing_radius', [panel({ x: [0.5, 1], y: [0, 3], title: 'Certified radius by noise level', xl: 'top-class probability', yl: 'radius' }, c => { const ppf = p => { let a = -8, b = 8; const cdf = z => { let s = 0; for (let i = 0; i < 4000; i++) { const x = -8 + ((z + 8) * (i + 0.5)) / 4000; s += Math.exp(-x * x / 2); } return (s * (z + 8)) / 4000 / Math.sqrt(2 * Math.PI); }; for (let i = 0; i < 30; i++) { const m = (a + b) / 2; if (cdf(m) < p) a = m; else b = m; } return a; }; [[0.25, C.orange], [0.5, C.blue], [1, C.green]].forEach(([s, col]) => c.line(L(p => Math.min(3, s * ppf(p)), 0.5, 0.995, 40), { c: col })); })]);
  save('G04', 'taylor_orders', [panel({ x: [-3, 3], y: [-2, 2], title: 'Sine and its Taylor polynomials', xl: 'x', yl: 'value' }, c => { c.line(L(Math.sin, -3, 3), { c: C.ink }); c.line(L(x => x, -2, 2), { c: C.orange }); c.line(L(x => x - x ** 3 / 6, -3, 3).filter(p => Math.abs(p[1]) <= 2), { c: C.blue }); c.line(L(x => x - x ** 3 / 6 + x ** 5 / 120, -3, 3).filter(p => Math.abs(p[1]) <= 2), { c: C.green }); })]);
  save('G05', 'density_families', [panel({ x: [0, 6], y: [0, 1], title: 'Exponential, gamma and log-normal', xl: 'x', yl: 'density' }, c => { c.line(L(x => Math.exp(-x), 0, 6), { c: C.blue }); c.line(L(x => x * Math.exp(-x), 0, 6), { c: C.orange }); c.line(L(x => Math.exp(-((Math.log(x) / 0.6) ** 2) / 2) / (x * 0.6 * Math.sqrt(2 * Math.PI)), 0.02, 6), { c: C.green }); })]);
  save('G06', 'distance_separation', [panel({ x: [0, 6], y: [0, 6], title: 'Distance against separation', xl: 'separation', yl: 'distance' }, c => { c.line(L(d => d, 0, 6), { c: C.green }); c.line(L(d => Math.LN2 * (1 - Math.exp(-(d * d) * 3)), 0, 6), { c: C.blue }); })]);
}
console.log('figures written');
