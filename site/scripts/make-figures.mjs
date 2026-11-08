// Draws the figures for the notes written without lecture PDFs. Every number comes from lib/models.mjs, so figures match the interactives.
// Usage: node scripts/make-figures.mjs   (writes SVGs into ../markdown-notes/images/<note>/figures)
import { mkdirSync, writeFileSync } from 'node:fs';
import { gauss, linfit, rng } from '../lib/math.mjs';
import { EIGHT_RUNS, LATENCIES, accuracy, agglomerate, buildTree, covariance2, dataset, dbscan, eig2, em1d, kmeans, linspace, logisticStep, mahalanobis, meanSilhouette, moons, normalPdf, predictTree, pruneTree, sigmoid, spreads, stats1, svd, medianOf, madOf } from '../lib/models.mjs';

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
  const path = `../markdown-notes/images/${dir}/figures`;
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
console.log('figures written');
