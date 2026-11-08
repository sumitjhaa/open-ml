// Statistics and small from-scratch models behind the explorers. Pure, seeded, and tested in scripts/check.mjs.
import { rng, gauss, solve } from './math.mjs';

export const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
export const pct = (v, d = 1) => (Number.isFinite(v) ? (100 * v).toFixed(d) + '%' : '—');
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const linspace = (a, b, n = 201) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

/* ---------- distributions ---------- */
export const normalPdf = (x, m = 0, s = 1) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
export function normalCdf(x, m = 0, s = 1) { // Abramowitz & Stegun 26.2.17
  const z = (x - m) / s, t = 1 / (1 + 0.2316419 * Math.abs(z));
  const p = normalPdf(z) * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return z > 0 ? 1 - p : p;
}
export function lgamma(x) { // Lanczos
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
  let y = x, tmp = x + 5.5, ser = 1.000000000190015;
  tmp -= (x + 0.5) * Math.log(tmp);
  for (const v of c) ser += v / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
export const betaPdf = (x, a, b) => (x <= 0 || x >= 1 ? 0 : Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)));
export const binomPmf = (k, n, p) => Math.exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * Math.log(Math.max(p, 1e-12)) + (n - k) * Math.log(Math.max(1 - p, 1e-12)));
export const poissonPmf = (k, l) => Math.exp(k * Math.log(l) - l - lgamma(k + 1));
export const sigmoid = z => 1 / (1 + Math.exp(-z));

/* ---------- linear algebra ---------- */
// Symmetric [[a,b],[b,d]]: eigenvalues l1 >= l2 and the unit eigenvector of l1.
export function eig2(a, b, d) {
  const m = (a + d) / 2, r = Math.hypot((a - d) / 2, b), l1 = m + r, l2 = m - r;
  let v = Math.abs(b) > 1e-12 ? [l1 - d, b] : a >= d ? [1, 0] : [0, 1];
  const n = Math.hypot(...v);
  v = [v[0] / n, v[1] / n];
  return { l1, l2, v, u: [-v[1], v[0]] };
}
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

export function ridge(X, y, lam) {
  const n = X.length, p = X[0].length;
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) / n + (i === j ? lam : 0)));
  return solve(A, Array.from({ length: p }, (_, i) => X.reduce((s, r, k) => s + r[i] * y[k], 0) / n));
}
export const soft = (z, l) => Math.sign(z) * Math.max(Math.abs(z) - l, 0);
// Coordinate descent on (1/2n)|y - Xb|^2 + lam|b|_1.
export function lasso(X, y, lam, iters = 300) {
  const n = X.length, p = X[0].length, b = Array(p).fill(0), r = [...y];
  const sq = Array.from({ length: p }, (_, j) => X.reduce((s, row) => s + row[j] ** 2, 0) / n);
  for (let t = 0; t < iters; t++) for (let j = 0; j < p; j++) {
    let rho = 0;
    for (let i = 0; i < n; i++) rho += X[i][j] * (r[i] + X[i][j] * b[j]);
    const nb = soft(rho / n, lam) / sq[j], d = nb - b[j];
    if (d) for (let i = 0; i < n; i++) r[i] -= X[i][j] * d;
    b[j] = nb;
  }
  return b;
}
// Least absolute deviations line, by iteratively reweighted least squares.
export function ladFit(xs, ys, iters = 60) {
  let w = 0, b = 0;
  for (let t = 0; t < iters; t++) {
    const u = xs.map((x, i) => 1 / Math.max(Math.abs(ys[i] - w * x - b), 1e-4));
    let a = 0, c = 0, d = 0, e = 0, g = 0;
    xs.forEach((x, i) => { a += u[i] * x * x; c += u[i] * x; d += u[i]; e += u[i] * x * ys[i]; g += u[i] * ys[i]; });
    [w, b] = solve([[a, c], [c, d]], [e, g]);
  }
  return { w, b };
}

/* ---------- logistic regression ---------- */
export const logLoss = (w, X, y) => X.reduce((s, r, i) => { const p = clamp(sigmoid(dot(w, r)), 1e-12, 1 - 1e-12); return s - (y[i] ? Math.log(p) : Math.log(1 - p)); }, 0) / X.length;
export function logisticStep(w, X, y, mode, eta = 0.5) {
  const d = w.length, g = Array(d).fill(0), H = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1e-6 : 0)));
  X.forEach((r, k) => {
    const p = sigmoid(dot(w, r)), s = p * (1 - p);
    for (let i = 0; i < d; i++) { g[i] += ((p - y[k]) * r[i]) / X.length; for (let j = 0; j < d; j++) H[i][j] += (s * r[i] * r[j]) / X.length; }
  });
  const dir = mode === 'newton' ? solve(H, g) : g.map(v => v * eta);
  return w.map((v, i) => v - dir[i]);
}

/* ---------- ranking metrics ---------- */
// Ties in score move together, so the curve takes one diagonal step and AUC counts a tie as half.
export function rocCurve(scores, labels) {
  const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = labels.filter(Boolean).length, N = labels.length - P;
  let tp = 0, fp = 0, auc = 0;
  const pts = [[0, 0]];
  for (let i = 0; i < idx.length;) {
    const s = scores[idx[i]], ptp = tp, pfp = fp;
    for (; i < idx.length && scores[idx[i]] === s; i++) labels[idx[i]] ? tp++ : fp++;
    auc += ((fp - pfp) * (tp + ptp)) / 2;
    pts.push([fp / N, tp / P]);
  }
  return { pts, auc: auc / (P * N) };
}

/* ---------- SVM (simplified SMO) ---------- */
export const kernels = {
  linear: () => (a, b) => dot(a, b),
  poly: ({ degree = 3 }) => (a, b) => (1 + dot(a, b)) ** degree,
  rbf: ({ gamma = 1 }) => (a, b) => Math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)),
};
export function smo(X, y, { C = 1, k, tol = 1e-3, passes = 12, maxIter = 400, seed = 1 }) {
  const n = X.length, a = Array(n).fill(0), r = rng(seed);
  // Rounded: Node and the browser can differ in the last digit of Math.exp, and the solver would amplify that into a hydration mismatch.
  const K = X.map(p => X.map(q => +k(p, q).toFixed(10)));
  let b = 0, pass = 0, it = 0;
  const F = i => a.reduce((s, v, j) => s + v * y[j] * K[j][i], 0) + b;
  while (pass < passes && it++ < maxIter) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = F(i) - y[i];
      if (!((y[i] * Ei < -tol && a[i] < C) || (y[i] * Ei > tol && a[i] > 0))) continue;
      let j = Math.floor(r() * (n - 1));
      if (j >= i) j++;
      const Ej = F(j) - y[j], ai = a[i], aj = a[j];
      const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C);
      const H = y[i] !== y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
      const eta = 2 * K[i][j] - K[i][i] - K[j][j];
      if (L === H || eta >= 0) continue;
      a[j] = clamp(aj - (y[j] * (Ei - Ej)) / eta, L, H);
      if (Math.abs(a[j] - aj) < 1e-5) continue;
      a[i] = ai + y[i] * y[j] * (aj - a[j]);
      const b1 = b - Ei - y[i] * (a[i] - ai) * K[i][i] - y[j] * (a[j] - aj) * K[i][j];
      const b2 = b - Ej - y[i] * (a[i] - ai) * K[i][j] - y[j] * (a[j] - aj) * K[j][j];
      b = a[i] > 0 && a[i] < C ? b1 : a[j] > 0 && a[j] < C ? b2 : (b1 + b2) / 2;
      changed++;
    }
    pass = changed ? 0 : pass + 1;
  }
  return { alpha: a, b, f: x => a.reduce((s, v, j) => (v > 1e-7 ? s + v * y[j] * k(X[j], x) : s), 0) + b };
}

/* ---------- trees ---------- */
const IMPURITY = {
  gini: p => 2 * p * (1 - p),
  entropy: p => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p)),
  error: p => Math.min(p, 1 - p),
};
export const impurity = (name, p) => IMPURITY[name](p);
// criterion: gini | entropy | error for y in {0,1}, mse for real y. Leaf `value` is P(y=1) or the mean.
export function buildTree(X, y, { maxDepth = 3, minLeaf = 1, criterion = 'gini', w, m, seed = 1 } = {}) {
  const W = w || X.map(() => 1), r = rng(seed), d = X[0].length;
  const score = (ws, s1, s2) => (criterion === 'mse' ? s2 / ws - (s1 / ws) ** 2 : IMPURITY[criterion](s1 / ws));
  const grow = (idx, depth) => {
    let ws = 0, s1 = 0, s2 = 0;
    for (const i of idx) { ws += W[i]; s1 += W[i] * y[i]; s2 += W[i] * y[i] * y[i]; }
    const here = score(ws, s1, s2), node = { value: s1 / ws, n: idx.length };
    if (depth >= maxDepth || idx.length < 2 * minLeaf || here < 1e-12) return node;
    let feats = Array.from({ length: d }, (_, i) => i);
    if (m && m < d) feats = feats.map(f => [r(), f]).sort((a, b) => a[0] - b[0]).slice(0, m).map(p => p[1]);
    let best = { gain: 1e-12 };
    for (const f of feats) {
      const o = [...idx].sort((a, b) => X[a][f] - X[b][f]);
      let lw = 0, l1 = 0, l2 = 0;
      for (let k = 0; k < o.length - 1; k++) {
        const i = o[k];
        lw += W[i]; l1 += W[i] * y[i]; l2 += W[i] * y[i] * y[i];
        if (X[o[k + 1]][f] === X[i][f] || k + 1 < minLeaf || o.length - k - 1 < minLeaf) continue;
        const rw = ws - lw, gain = here - (lw * score(lw, l1, l2) + rw * score(rw, s1 - l1, s2 - l2)) / ws;
        if (gain > best.gain) best = { gain, f, t: (X[i][f] + X[o[k + 1]][f]) / 2 };
      }
    }
    if (best.f === undefined) return node;
    return { ...node, ...best, l: grow(idx.filter(i => X[i][best.f] <= best.t), depth + 1), r: grow(idx.filter(i => X[i][best.f] > best.t), depth + 1) };
  };
  return grow(X.map((_, i) => i), 0);
}
export const predictTree = (t, x) => (t.l ? predictTree(x[t.f] <= t.t ? t.l : t.r, x) : t.value);
const mapLeaves = (t, fn) => (t.l ? { ...t, l: mapLeaves(t.l, fn), r: mapLeaves(t.r, fn) } : fn(t));

export function bagging(X, y, { B = 25, maxDepth = 6, m, criterion = 'gini', seed = 1 } = {}) {
  const r = rng(seed), n = X.length, trees = [], oobSum = Array(n).fill(0), oobN = Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    const idx = Array.from({ length: n }, () => Math.floor(r() * n)), inBag = new Set(idx);
    const t = buildTree(idx.map(i => X[i]), idx.map(i => y[i]), { maxDepth, m, criterion, seed: seed + b + 1 });
    trees.push(t);
    for (let i = 0; i < n; i++) if (!inBag.has(i)) { oobSum[i] += predictTree(t, X[i]); oobN[i]++; }
  }
  const predict = (x, k = B) => trees.slice(0, k).reduce((s, t) => s + predictTree(t, x), 0) / k;
  return { trees, predict, oob: oobSum.map((s, i) => (oobN[i] ? s / oobN[i] : NaN)) };
}

export function gbRegress(X, y, { rounds = 50, lr = 0.1, maxDepth = 1 } = {}) {
  const f0 = y.reduce((s, v) => s + v, 0) / y.length, trees = [], F = y.map(() => f0);
  for (let m = 0; m < rounds; m++) {
    const t = buildTree(X, y.map((v, i) => v - F[i]), { maxDepth, criterion: 'mse' });
    trees.push(t);
    X.forEach((x, i) => { F[i] += lr * predictTree(t, x); });
  }
  return { predict: (x, k = rounds) => trees.slice(0, k).reduce((s, t) => s + lr * predictTree(t, x), f0) };
}

const logOdds = y => { const p = clamp(y.reduce((s, v) => s + v, 0) / y.length, 1e-6, 1 - 1e-6); return Math.log(p / (1 - p)); };
// Regression tree on the residual y - p, then one Newton step in each leaf.
export function gbClass(X, y, { rounds = 50, lr = 0.2, maxDepth = 2 } = {}) {
  const f0 = logOdds(y), trees = [], F = y.map(() => f0);
  for (let m = 0; m < rounds; m++) {
    const p = F.map(sigmoid), res = y.map((v, i) => v - p[i]);
    const t = buildTree(X, res, { maxDepth, criterion: 'mse' });
    const leaf = new Map();
    X.forEach((x, i) => { const v = predictTree(t, x), c = leaf.get(v) || [0, 0]; leaf.set(v, [c[0] + res[i], c[1] + p[i] * (1 - p[i])]); });
    const tt = mapLeaves(t, n => { const c = leaf.get(n.value); return { ...n, value: c ? c[0] / (c[1] + 1e-9) : 0 }; });
    trees.push(tt);
    X.forEach((x, i) => { F[i] += lr * predictTree(tt, x); });
  }
  return { predict: (x, k = rounds) => sigmoid(trees.slice(0, k).reduce((s, t) => s + lr * predictTree(t, x), f0)) };
}

// XGBoost's split score and leaf weight: gain = (SL + SR - S)/2 - gamma with S = G^2/(H + lambda), w = -G/(H + lambda).
export const xgbScore = (G, H, lambda) => (G * G) / (H + lambda);
export function xgbTree(X, g, h, { maxDepth = 2, lambda = 1, gamma = 0 } = {}) {
  const d = X[0].length;
  const grow = (idx, depth) => {
    let G = 0, H = 0;
    for (const i of idx) { G += g[i]; H += h[i]; }
    const node = { value: -G / (H + lambda), n: idx.length };
    if (depth >= maxDepth || idx.length < 2) return node;
    const S = xgbScore(G, H, lambda);
    let best = { gain: 1e-9 };
    for (let f = 0; f < d; f++) {
      const o = [...idx].sort((a, b) => X[a][f] - X[b][f]);
      let GL = 0, HL = 0;
      for (let k = 0; k < o.length - 1; k++) {
        GL += g[o[k]]; HL += h[o[k]];
        if (X[o[k + 1]][f] === X[o[k]][f]) continue;
        const gain = (xgbScore(GL, HL, lambda) + xgbScore(G - GL, H - HL, lambda) - S) / 2 - gamma;
        if (gain > best.gain) best = { gain, f, t: (X[o[k]][f] + X[o[k + 1]][f]) / 2 };
      }
    }
    if (best.f === undefined) return node;
    return { ...node, ...best, l: grow(idx.filter(i => X[i][best.f] <= best.t), depth + 1), r: grow(idx.filter(i => X[i][best.f] > best.t), depth + 1) };
  };
  return grow(X.map((_, i) => i), 0);
}
export function xgbClass(X, y, { rounds = 50, lr = 0.2, maxDepth = 2, lambda = 1, gamma = 0 } = {}) {
  const f0 = logOdds(y), trees = [], F = y.map(() => f0);
  for (let m = 0; m < rounds; m++) {
    const p = F.map(sigmoid);
    const t = xgbTree(X, p.map((v, i) => v - y[i]), p.map(v => v * (1 - v)), { maxDepth, lambda, gamma });
    trees.push(t);
    X.forEach((x, i) => { F[i] += lr * predictTree(t, x); });
  }
  return { predict: (x, k = rounds) => sigmoid(trees.slice(0, k).reduce((s, t) => s + lr * predictTree(t, x), f0)) };
}

// y in {0,1}. Each round records the stump, weighted error, alpha and the weights it was trained on.
export function adaboost(X, y, { rounds = 30 } = {}) {
  const s = y.map(v => (v ? 1 : -1)), n = X.length, out = [];
  let w = Array(n).fill(1 / n);
  for (let m = 0; m < rounds; m++) {
    const tree = buildTree(X, y, { maxDepth: 1, criterion: 'error', w });
    const h = X.map(x => (predictTree(tree, x) > 0.5 ? 1 : -1));
    const eps = clamp(w.reduce((a, v, i) => a + (h[i] !== s[i] ? v : 0), 0), 1e-9, 1 - 1e-9);
    const alpha = 0.5 * Math.log((1 - eps) / eps), weights = w;
    const nw = w.map((v, i) => v * Math.exp(-alpha * s[i] * h[i]));
    const Z = nw.reduce((a, v) => a + v, 0);
    out.push({ tree, eps, alpha, weights, Z });
    w = nw.map(v => v / Z);
  }
  const score = (x, k = rounds) => out.slice(0, k).reduce((a, r) => a + r.alpha * (predictTree(r.tree, x) > 0.5 ? 1 : -1), 0);
  return { rounds: out, score, predict: (x, k) => sigmoid(2 * score(x, k)) };
}

/* ---------- shared toy datasets on [-3, 3]^2: [x1, x2, label] ---------- */
export function dataset(kind, seed = 1, n = 40, noise = 0.35) {
  const r = rng(seed), g = () => noise * gauss(r), out = [];
  for (let i = 0; i < n; i++) {
    const y = i % 2;
    if (kind === 'blobs') out.push([(y ? 1.2 : -1.2) + g() * 2, (y ? 1 : -1) + g() * 2, y]);
    if (kind === 'overlap') out.push([(y ? 0.6 : -0.6) + g() * 3, (y ? 0.5 : -0.5) + g() * 3, y]);
    if (kind === 'xor') { const a = r() < 0.5 ? 1 : -1, b = (y ? -a : a); out.push([a * 1.5 + g() * 1.4, b * 1.5 + g() * 1.4, y]); }
    if (kind === 'circles') { const t = 2 * Math.PI * r(), R = y ? 2.3 : 0.9; out.push([R * Math.cos(t) + g(), R * Math.sin(t) + g(), y]); }
  }
  // Rounded: Node and the browser can differ in the last float digit of Math.log/cos, which breaks hydration.
  return out.map(([a, b, c]) => [+clamp(a, -2.9, 2.9).toFixed(3), +clamp(b, -2.9, 2.9).toFixed(3), c]);
}
export const accuracy = (predict, D) => D.filter(d => (predict([d[0], d[1]]) > 0.5) === !!d[2]).length / D.length;

/* ---------- double descent: minimum-norm random-feature regression ---------- */
// Random inputs and high-frequency features: on easier setups the peak at p = n does not appear.
export function doubleDescent({ n = 14, pmax = 240, seed = 1 } = {}) {
  const r = rng(seed), truth = x => Math.sin(Math.PI * x);
  const xs = Array.from({ length: n }, () => 2 * r() - 1).sort((a, b) => a - b), ys = xs.map(x => truth(x) + 0.2 * gauss(r));
  const om = Array.from({ length: pmax }, () => 12 * gauss(r)), ph = Array.from({ length: pmax }, () => 2 * Math.PI * r());
  const feat = (x, p) => Array.from({ length: p }, (_, j) => Math.sqrt(2 / p) * Math.cos(om[j] * x + ph[j]));
  const grid = linspace(-1, 1, 161), eps = 1e-10;
  const fit = p => {
    const F = xs.map(x => feat(x, p));
    let b;
    if (p <= n) {
      const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => F.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? eps : 0)));
      b = solve(A, Array.from({ length: p }, (_, i) => F.reduce((s, row, k) => s + row[i] * ys[k], 0)));
    } else {
      const K = F.map((u, i) => F.map((v, j) => dot(u, v) + (i === j ? eps : 0))), a = solve(K, ys);
      b = Array.from({ length: p }, (_, j) => F.reduce((s, row, k) => s + row[j] * a[k], 0));
    }
    const g = x => dot(feat(x, p), b);
    return { p, g, train: xs.reduce((s, x, i) => s + (g(x) - ys[i]) ** 2, 0) / n, test: grid.reduce((s, x) => s + (g(x) - truth(x)) ** 2, 0) / grid.length, norm: Math.hypot(...b) };
  };
  return { n, xs, ys, truth, grid, fit };
}

/* ---------- SVD, clustering, pruning ---------- */
// One-sided Jacobi: rotate column pairs until all are orthogonal. Needs rows >= columns, which every caller here has.
export function svd(A, sweeps = 30) {
  const m = A.length, n = A[0].length, U = A.map(r => [...r]), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let a = 0, b = 0, c = 0;
      for (let i = 0; i < m; i++) { a += U[i][p] ** 2; b += U[i][q] ** 2; c += U[i][p] * U[i][q]; }
      if (Math.abs(c) < 1e-13 * Math.sqrt(a * b) || !c) continue;
      off++;
      const z = (b - a) / (2 * c), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z)), co = 1 / Math.sqrt(1 + t * t), si = co * t;
      for (const M of [U, V]) for (const row of M) { const x = row[p], y = row[q]; row[p] = co * x - si * y; row[q] = si * x + co * y; }
    }
    if (!off) break;
  }
  const norms = Array.from({ length: n }, (_, j) => Math.sqrt(U.reduce((s, r) => s + r[j] ** 2, 0))), order = norms.map((_, i) => i).sort((a, b) => norms[b] - norms[a]);
  const s = order.map(j => norms[j]);
  return {
    s,
    U: U.map(r => order.map(j => (norms[j] > 1e-12 ? r[j] / norms[j] : 0))),
    V: V.map(r => order.map(j => r[j])),
    rank: k => { const Uo = U.map(r => order.map(j => r[j])), Vo = V.map(r => order.map(j => r[j])); return Uo.map(u => Vo.map(v => { let t = 0; for (let i = 0; i < k; i++) t += u[i] * v[i]; return t; })); },
  };
}

const dist2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
// Returns one frame per assignment step: { centres, assign, inertia }.
export function kmeans(X, k, { seed = 1, init = 'random', iters = 40 } = {}) {
  const r = rng(seed), n = X.length;
  let centres;
  if (Array.isArray(init)) centres = init.map(c => [...c]);
  else if (init === 'plus') {
    centres = [X[Math.floor(r() * n)]];
    while (centres.length < k) {
      const d = X.map(x => Math.min(...centres.map(c => dist2(x, c)))), t = d.reduce((a, b) => a + b, 0);
      let u = r() * t, i = 0;
      while (i < n - 1 && (u -= d[i]) > 0) i++;
      centres.push(X[i]);
    }
  } else centres = X.map((x, i) => [r(), x, i]).sort((a, b) => a[0] - b[0]).slice(0, k).map(p => p[1]);
  const frames = [];
  let prev = '';
  for (let t = 0; t < iters; t++) {
    const assign = X.map(x => centres.reduce((b, c, j) => (dist2(x, c) < dist2(x, centres[b]) ? j : b), 0));
    frames.push({ centres, assign, inertia: X.reduce((s, x, i) => s + dist2(x, centres[assign[i]]), 0) });
    if (assign.join() === prev) break;
    prev = assign.join();
    centres = centres.map((c, j) => { const pts = X.filter((_, i) => assign[i] === j); return pts.length ? c.map((_, d) => pts.reduce((s, p) => s + p[d], 0) / pts.length) : c; });
  }
  return frames;
}
export const silhouette = (a, b) => (Math.max(a, b) ? (b - a) / Math.max(a, b) : 0);
export function meanSilhouette(X, assign) {
  const k = Math.max(...assign) + 1, d = (a, b) => Math.sqrt(dist2(a, b));
  const vals = X.map((x, i) => {
    const avg = j => { const m = X.filter((_, q) => assign[q] === j && q !== i); return m.length ? m.reduce((s, p) => s + d(x, p), 0) / m.length : NaN; };
    const others = Array.from({ length: k }, (_, j) => j).filter(j => j !== assign[i]).map(avg).filter(Number.isFinite);
    return others.length && Number.isFinite(avg(assign[i])) ? silhouette(avg(assign[i]), Math.min(...others)) : 0;
  });
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
// Agglomerative clustering. Returns merges { a, b, h, id } where a and b are cluster ids; leaves are 0..n-1.
export function agglomerate(X, linkage = 'single') {
  let cl = X.map((_, i) => ({ id: i, m: [i] }));
  const pair = (A, B) => { const ds = A.m.flatMap(i => B.m.map(j => Math.sqrt(dist2(X[i], X[j])))); if (linkage === 'ward') { const c = M => X[0].map((_, d) => M.reduce((s, i) => s + X[i][d], 0) / M.length); return ((A.m.length * B.m.length) / (A.m.length + B.m.length)) * dist2(c(A.m), c(B.m)); } return linkage === 'single' ? Math.min(...ds) : linkage === 'complete' ? Math.max(...ds) : ds.reduce((s, v) => s + v, 0) / ds.length; };
  const merges = [];
  let next = X.length;
  while (cl.length > 1) {
    let best = [0, 1, Infinity];
    for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const h = pair(cl[i], cl[j]); if (h < best[2]) best = [i, j, h]; }
    const [i, j, h] = best, A = cl[i], B = cl[j];
    merges.push({ a: A.id, b: B.id, h, id: next });
    cl = cl.filter((_, q) => q !== i && q !== j).concat({ id: next++, m: [...A.m, ...B.m] });
  }
  return merges;
}
// Cost-complexity pruning of a classification tree, with training error as the risk (as a fraction of all rows).
export function pruneTree(tree, alpha, X, y) {
  const N = X.length;
  const walk = (t, idx) => {
    const pos = idx.reduce((s, i) => s + y[i], 0), leafRisk = Math.min(pos, idx.length - pos) / N;
    if (!t.l) return { node: t, R: leafRisk, leaves: 1 };
    const L = walk(t.l, idx.filter(i => X[i][t.f] <= t.t)), Rt = walk(t.r, idx.filter(i => X[i][t.f] > t.t)), R = L.R + Rt.R, leaves = L.leaves + Rt.leaves;
    if (leafRisk - R <= alpha * (leaves - 1)) return { node: { value: t.value, n: t.n }, R: leafRisk, leaves: 1 };
    return { node: { ...t, l: L.node, r: Rt.node }, R, leaves };
  };
  return walk(tree, X.map((_, i) => i));
}

/* ---------- mixtures, density clustering, maps and anomaly scores ---------- */
export const EIGHT_RUNS = [[4, 4], [4, 3], [2, 3], [1, 3], [8, 8], [8, 6], [8, 9], [6, 8]];
export const LATENCIES = [10, 12, 14, 22, 31, 33, 36];

// EM for two equally likely components with a fixed common sigma. One frame per step: { mu, r, ll }.
export function em1d(xs, mu0, sigma, steps = 20) {
  const frames = [];
  let mu = [...mu0];
  for (let t = 0; t <= steps; t++) {
    const dens = xs.map(x => mu.map(m => normalPdf(x, m, sigma))), r = dens.map(d => { const s = d[0] + d[1]; return [d[0] / s, d[1] / s]; });
    frames.push({ mu: [...mu], r, ll: dens.reduce((a, d) => a + Math.log(0.5 * d[0] + 0.5 * d[1]), 0) });
    mu = [0, 1].map(k => r.reduce((a, q, i) => a + q[k] * xs[i], 0) / r.reduce((a, q) => a + q[k], 0));
  }
  return frames;
}

const gauss2 = (x, m, S) => { const a = x[0] - m[0], b = x[1] - m[1], det = S[0] * S[2] - S[1] * S[1]; return Math.exp(-0.5 * (S[2] * a * a - 2 * S[1] * a * b + S[0] * b * b) / det) / (2 * Math.PI * Math.sqrt(det)); };
// EM with a full covariance per component. S is [s11, s12, s22]. Frames: { pi, mu, S, r, ll }.
export function em2d(X, mu0, steps = 30, tau = 1e-3) {
  const k = mu0.length, n = X.length, frames = [];
  let mu = mu0.map(m => [...m]), S = mu0.map(() => [1.5, 0, 1.5]), pi = mu0.map(() => 1 / k);
  for (let t = 0; t <= steps; t++) {
    let ll = 0;
    const r = X.map(x => { const p = mu.map((m, j) => pi[j] * gauss2(x, m, S[j])), s = p.reduce((a, b) => a + b, 0); ll += Math.log(s); return p.map(v => +(v / s).toFixed(9)); });
    frames.push({ pi, mu, S, r, ll });
    const nk = mu.map((_, j) => r.reduce((a, q) => a + q[j], 0));
    pi = nk.map(v => v / n);
    mu = mu.map((_, j) => [0, 1].map(d => r.reduce((a, q, i) => a + q[j] * X[i][d], 0) / nk[j]));
    S = mu.map((m, j) => { let a = 0, b = 0, c = 0; X.forEach((x, i) => { const u = x[0] - m[0], v = x[1] - m[1], w = r[i][j]; a += w * u * u; b += w * u * v; c += w * v * v; }); return [a / nk[j] + tau, b / nk[j], c / nk[j] + tau]; });
  }
  return frames;
}

// Labels: 0.. for clusters, -1 for noise. kind: 'core' | 'border' | 'noise'.
export function dbscan(X, eps, minPts) {
  const n = X.length, N = X.map(a => X.map((b, j) => [dist2(a, b), j]).filter(p => p[0] <= eps * eps + 1e-12).map(p => p[1])), core = N.map(v => v.length >= minPts), labels = Array(n).fill(-1);
  let c = -1;
  for (let i = 0; i < n; i++) {
    if (!core[i] || labels[i] !== -1) continue;
    labels[i] = ++c;
    const stack = [i];
    while (stack.length) { const j = stack.pop(); if (core[j]) for (const p of N[j]) if (labels[p] === -1) { labels[p] = c; stack.push(p); } }
  }
  return { labels, core, kinds: labels.map((l, i) => (core[i] ? 'core' : l >= 0 ? 'border' : 'noise')), clusters: c + 1 };
}
export const kDistance = (X, k) => X.map(a => X.map(b => Math.sqrt(dist2(a, b))).sort((p, q) => p - q)[k]);

export function silhouettes(X, assign) {
  const k = Math.max(...assign) + 1, d = (a, b) => Math.sqrt(dist2(a, b));
  return X.map((x, i) => {
    const avg = j => { const m = X.filter((_, q) => assign[q] === j && q !== i); return m.length ? m.reduce((s, p) => s + d(x, p), 0) / m.length : NaN; };
    const own = avg(assign[i]), others = Array.from({ length: k }, (_, j) => j).filter(j => j !== assign[i]).map(avg).filter(Number.isFinite);
    return Number.isFinite(own) && others.length ? silhouette(own, Math.min(...others)) : 0;
  });
}
export function spreads(X, assign) {
  const dim = X[0].length, mean = Array.from({ length: dim }, (_, d) => X.reduce((s, x) => s + x[d], 0) / X.length), k = Math.max(...assign) + 1;
  let W = 0, B = 0;
  for (let j = 0; j < k; j++) {
    const pts = X.filter((_, i) => assign[i] === j);
    if (!pts.length) continue;
    const c = Array.from({ length: dim }, (_, d) => pts.reduce((s, x) => s + x[d], 0) / pts.length);
    W += pts.reduce((s, p) => s + dist2(p, c), 0); B += pts.length * dist2(c, mean);
  }
  return { W, B, T: W + B };
}
export const calinski = (B, W, n, k) => (B / (k - 1)) / (W / (n - k));

// Exact t-SNE for small n. Inputs to the iteration are rounded so Node and the browser produce the same map.
export function tsne(X, { perplexity = 15, iters = 400, seed = 1, lr = 60 } = {}) {
  const n = X.length, D = X.map(a => X.map(b => dist2(a, b))), target = Math.log2(perplexity), P = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    let lo = 1e-6, hi = 1e6, row = [];
    for (let it = 0; it < 60; it++) {
      const beta = Math.sqrt(lo * hi);
      row = D[i].map((d, j) => (j === i ? 0 : Math.exp(-beta * d)));
      const s = row.reduce((a, b) => a + b, 0) || 1e-300;
      row = row.map(v => v / s);
      const H = -row.reduce((a, p) => a + (p > 1e-300 ? p * Math.log2(p) : 0), 0);
      if (H > target) lo = beta; else hi = beta;
    }
    row.forEach((v, j) => { P[i][j] = v; });
  }
  const sym = P.map((row, i) => row.map((v, j) => +(Math.max((v + P[j][i]) / (2 * n), 1e-12)).toFixed(10)));
  const r = rng(seed);
  let Y = Array.from({ length: n }, () => [1e-2 * gauss(r), 1e-2 * gauss(r)]), V = Y.map(() => [0, 0]);
  for (let t = 0; t < iters; t++) {
    const w = Y.map(a => Y.map(b => 1 / (1 + dist2(a, b)))), sum = w.reduce((s, row, i) => s + row.reduce((u, v, j) => u + (i === j ? 0 : v), 0), 0), ex = t < 80 ? 12 : 1, m = t < 150 ? 0.5 : 0.8;
    const G = Y.map((a, i) => { let gx = 0, gy = 0; Y.forEach((b, j) => { if (i !== j) { const c = 4 * (ex * sym[i][j] - w[i][j] / sum) * w[i][j]; gx += c * (a[0] - b[0]); gy += c * (a[1] - b[1]); } }); return [gx, gy]; });
    V = V.map((v, i) => [m * v[0] - lr * G[i][0], m * v[1] - lr * G[i][1]]);
    Y = Y.map((a, i) => [+(a[0] + V[i][0]).toFixed(7), +(a[1] + V[i][1]).toFixed(7)]);
  }
  return Y;
}

export const medianOf = a => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
export const madOf = a => { const m = medianOf(a); return medianOf(a.map(x => Math.abs(x - m))); };
export const robustScore = (x, a) => (0.6745 * (x - medianOf(a))) / madOf(a);
export const stats1 = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return { m, s: Math.sqrt(a.reduce((u, v) => u + (v - m) ** 2, 0) / a.length) }; };
export function covariance2(X) {
  const m = [0, 1].map(d => X.reduce((s, x) => s + x[d], 0) / X.length), n = X.length, f = (i, j) => X.reduce((s, x) => s + (x[i] - m[i]) * (x[j] - m[j]), 0) / n;
  return { m, a: f(0, 0), b: f(0, 1), d: f(1, 1) };
}
export const mahalanobis = (x, { m, a, b, d }) => { const u = x[0] - m[0], v = x[1] - m[1], det = a * d - b * b; return Math.sqrt((d * u * u - 2 * b * u * v + a * v * v) / det); };

const harmonic = n => { let s = 0; for (let i = 1; i <= n; i++) s += 1 / i; return s; };
export const cFactor = m => (m <= 1 ? 0 : 2 * harmonic(m - 1) - (2 * (m - 1)) / m);
function iTree(X, r, depth, limit) {
  if (X.length <= 1 || depth >= limit) return { size: X.length };
  const f = Math.floor(r() * X[0].length), v = X.map(p => p[f]), lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) return { size: X.length };
  const cut = lo + r() * (hi - lo);
  return { f, cut, l: iTree(X.filter(p => p[f] < cut), r, depth + 1, limit), r: iTree(X.filter(p => p[f] >= cut), r, depth + 1, limit) };
}
const pathLen = (t, x, d = 0) => (t.l ? pathLen(x[t.f] < t.cut ? t.l : t.r, x, d + 1) : d + cFactor(t.size));
export function isolationForest(X, { trees = 100, psi = 256, seed = 1 } = {}) {
  const r = rng(seed), m = Math.min(psi, X.length), limit = Math.ceil(Math.log2(Math.max(m, 2))), T = [];
  for (let t = 0; t < trees; t++) T.push(iTree(X.map(x => [r(), x]).sort((a, b) => a[0] - b[0]).slice(0, m).map(p => p[1]), r, 0, limit));
  const depth = x => T.reduce((s, t) => s + pathLen(t, x), 0) / T.length;
  return { depth, score: x => 2 ** (-depth(x) / cFactor(m)) };
}
export const averagePrecision = ranks => [...ranks].sort((a, b) => a - b).reduce((s, r, j) => s + (j + 1) / r, 0) / ranks.length;

// Two interleaved half-moons in [-1.6, 2.6] by [-1, 1.5]: [x1, x2, label]. Rounded so Node and the browser agree.
export function moons(seed = 1, n = 120, noise = 0.08) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) { const y = i % 2, t = Math.PI * r(), a = y ? 1 - Math.cos(t) : Math.cos(t), b = y ? 0.5 - Math.sin(t) : Math.sin(t); out.push([+(a + noise * gauss(r)).toFixed(3), +(b + noise * gauss(r)).toFixed(3), y]); }
  return out;
}

// Round gaussian groups, n per centre, listed centre by centre. Rounded so Node and the browser agree.
export const blobs = (seed, centres, n = 18, sd = 0.55) => { const r = rng(seed), out = []; centres.forEach(c => { for (let i = 0; i < n; i++) out.push([+(c[0] + sd * gauss(r)).toFixed(3), +(c[1] + sd * gauss(r)).toFixed(3)]); }); return out; };
// Caches an expensive explorer computation by its arguments, so dragging a slider does not redo it.
export const memo = (f, m = new Map()) => (...a) => { const k = a.join(); if (!m.has(k)) m.set(k, f(...a)); return m.get(k); };
