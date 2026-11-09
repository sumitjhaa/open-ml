// Algorithms for matrix factorisation, neighbours, Bayes, networks, convolution, sequences and reinforcement learning.
import { gauss, mean, rng, solve } from './math.mjs';

const r6 = v => +v.toFixed(6);
const r8 = v => +v.toFixed(8); // training loops round each step, so Node and the browser stay in step

/* ---------- matrix factorisation ---------- */

export const RATINGS = [[5, 4, 2, 3, 2], [null, 5, 1, 4, null], [4, null, 1, 5, null]];
const dotv = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const mfLoss = (A, W, H) => { let s = 0; A.forEach((row, i) => row.forEach((v, j) => { if (v !== null) s += (v - dotv(W[i], H[j])) ** 2; })); return s; };
export const mfPredict = (W, H) => W.map(u => H.map(m => dotv(u, m)));

function ridgeSolve(rows, ys, k, lam) {
  const G = Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) => rows.reduce((s, x) => s + x[a] * x[b], 0) + (a === b ? lam + 1e-9 : 0)));
  return solve(G, Array.from({ length: k }, (_, a) => rows.reduce((s, x, i) => s + x[a] * ys[i], 0))).map(r8);
}
// Alternating least squares on the observed cells. H[j] is movie j's features. Frames: { W, H, loss }.
export function als(A, k, { lam = 0, sweeps = 30, seed = 1 } = {}) {
  const r = rng(seed), n = A.length, m = A[0].length, rand = () => Array.from({ length: k }, () => 0.2 + r());
  let W = Array.from({ length: n }, rand), H = Array.from({ length: m }, rand);
  const frames = [{ W, H, loss: mfLoss(A, W, H) }];
  for (let t = 0; t < sweeps; t++) {
    H = H.map((old, j) => { const o = A.map((_, i) => i).filter(i => A[i][j] !== null); return o.length ? ridgeSolve(o.map(i => W[i]), o.map(i => A[i][j]), k, lam) : old; });
    W = W.map((old, i) => { const o = A[i].map((_, j) => j).filter(j => A[i][j] !== null); return o.length ? ridgeSolve(o.map(j => H[j]), o.map(j => A[i][j]), k, lam) : old; });
    frames.push({ W, H, loss: mfLoss(A, W, H) });
  }
  return frames;
}

/* ---------- nearest neighbours ---------- */

export const lpDistance = (a, b, p) => (p === Infinity ? Math.max(...a.map((v, i) => Math.abs(v - b[i]))) : a.reduce((s, v, i) => s + Math.abs(v - b[i]) ** p, 0) ** (1 / p));
export const nearest = (X, q, k, p = 2) => X.map((x, i) => [lpDistance(x, q, p), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k);
export function knnVote(X, y, q, k, { p = 2, weighted = false } = {}) {
  const score = {};
  for (const [d, i] of nearest(X, q, k, p)) score[y[i]] = (score[y[i]] ?? 0) + (weighted ? 1 / (d + 1e-9) : 1);
  return +Object.entries(score).sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
}
export const TEN_POINTS = [[2, 3], [4, 1], [1, 9], [3, 7], [5, 4], [7, 2], [9, 6], [6, 8], [7, 9], [8, 8]];
// Splits at the median, alternating axes. Returns the cuts as { axis, value, box } and a search that reports the points it looked at.
export function kdTree(X, leaf = 3) {
  const cuts = [];
  const build = (idx, depth, box) => {
    if (idx.length <= leaf) return { idx, box };
    const axis = depth % 2, sorted = [...idx].sort((a, b) => X[a][axis] - X[b][axis]), mid = sorted.length >> 1, value = X[sorted[mid]][axis];
    cuts.push({ axis, value, box });
    const lo = box.map(c => [...c]), hi = box.map(c => [...c]); lo[axis][1] = value; hi[axis][0] = value;
    return { axis, value, l: build(sorted.slice(0, mid), depth + 1, lo), r: build(sorted.slice(mid), depth + 1, hi) };
  };
  const root = build(X.map((_, i) => i), 0, [[0, 10], [0, 10]]);
  const leafOf = q => { let n = root; while (n.l) n = q[n.axis] < n.value ? n.l : n.r; return n; };
  return { root, cuts, leafOf };
}
export function lshBuckets(X, bits, seed = 1) {
  const r = rng(seed), planes = Array.from({ length: bits }, () => { const t = 2 * Math.PI * r(), w = [+Math.cos(t).toFixed(6), +Math.sin(t).toFixed(6)], a = [10 * r(), 10 * r()]; return { w, b: -(w[0] * a[0] + w[1] * a[1]) }; });
  const code = x => planes.map(p => +(p.w[0] * x[0] + p.w[1] * x[1] + p.b > 0)).join('');
  return { planes, code };
}

/* ---------- naive Bayes ---------- */

export const SPAM = { rows: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]], labels: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] };
export function bernoulliNb(rows, labels, alpha = 0) {
  const classes = [0, 1], d = rows[0].length, n = classes.map(c => labels.filter(l => l === c).length);
  const theta = classes.map((c, ci) => Array.from({ length: d }, (_, j) => (rows.filter((r, i) => labels[i] === c && r[j] === 1).length + alpha) / (n[ci] + 2 * alpha)));
  const prior = n.map(v => v / rows.length);
  const score = x => classes.map((_, c) => x.reduce((p, v, j) => p * (v ? theta[c][j] : 1 - theta[c][j]), prior[c]));
  return { theta, prior, score, posterior: x => { const s = score(x); return s[1] / (s[0] + s[1]); } };
}
export const HEIGHTS = [[6, 180, 12, 1], [5.92, 190, 11, 1], [5.58, 170, 12, 1], [5.92, 165, 10, 1], [5, 100, 6, 0], [5.5, 150, 8, 0], [5.42, 130, 7, 0], [5.75, 150, 9, 0]];
export const sampleVar = a => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); };

/* ---------- networks ---------- */

export const ACT = {
  relu: [z => Math.max(0, z), z => +(z > 0)],
  tanh: [z => Math.tanh(z), z => 1 - Math.tanh(z) ** 2],
  sigmoid: [z => 1 / (1 + Math.exp(-z)), z => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }],
  none: [z => z, () => 1],
};
const sig = z => 1 / (1 + Math.exp(-z));
export function mlpForward(p, x, act = 'relu') {
  const z1 = p.W1.map((row, i) => dotv(row, x) + p.b1[i]), h = z1.map(ACT[act][0]), z2 = dotv(p.w2, h) + p.b2;
  return { z1, h, z2, p: sig(z2) };
}
// One hidden layer, sigmoid output, cross-entropy, full-batch descent. Frames every `every` steps.
export function trainMlp(X, y, { hidden = 8, act = 'relu', lr = 0.5, steps = 1500, every = 50, seed = 1 } = {}) {
  const r = rng(seed), d = X[0].length, g = () => r8(gauss(r));
  const p = { W1: Array.from({ length: hidden }, () => Array.from({ length: d }, g)), b1: Array.from({ length: hidden }, g), w2: Array.from({ length: hidden }, () => r8(0.5 * gauss(r))), b2: 0 };
  const copy = () => ({ W1: p.W1.map(a => [...a]), b1: [...p.b1], w2: [...p.w2], b2: p.b2 }), frames = [];
  const dact = ACT[act][1];
  for (let t = 0; t <= steps; t++) {
    const gW = p.W1.map(a => a.map(() => 0)), gb = p.b1.map(() => 0), gw2 = p.w2.map(() => 0); let gb2 = 0, loss = 0;
    X.forEach((x, n) => {
      const f = mlpForward(p, x, act), q = Math.min(1 - 1e-9, Math.max(1e-9, f.p)), e = f.p - y[n];
      loss -= y[n] ? Math.log(q) : Math.log(1 - q);
      gb2 += e;
      f.h.forEach((hv, i) => { gw2[i] += e * hv; const delta = e * p.w2[i] * dact(f.z1[i]); gb[i] += delta; x.forEach((v, j) => { gW[i][j] += delta * v; }); });
    });
    if (t % every === 0) frames.push({ t, loss: loss / X.length, p: copy() });
    const N = X.length;
    p.W1 = p.W1.map((a, i) => a.map((v, j) => r8(v - (lr * gW[i][j]) / N))); p.b1 = p.b1.map((v, i) => r8(v - (lr * gb[i]) / N)); p.w2 = p.w2.map((v, i) => r8(v - (lr * gw2[i]) / N)); p.b2 = r8(p.b2 - (lr * gb2) / N);
  }
  return frames;
}
export const mlpParams = layers => layers.slice(1).reduce((s, n, i) => s + layers[i] * n + n, 0);

// A minimal reverse-mode engine: each Value remembers its inputs and how to pass a gradient back.
export class Value {
  constructor(data, prev = [], op = '', label = '') { this.data = data; this.prev = prev; this.op = op; this.label = label; this.grad = 0; this._back = () => {}; }
  static of(v) { return v instanceof Value ? v : new Value(v); }
  add(o) { o = Value.of(o); const out = new Value(this.data + o.data, [this, o], '+'); out._back = () => { this.grad += out.grad; o.grad += out.grad; }; return out; }
  mul(o) { o = Value.of(o); const out = new Value(this.data * o.data, [this, o], '×'); out._back = () => { this.grad += o.data * out.grad; o.grad += this.data * out.grad; }; return out; }
  exp() { const out = new Value(Math.exp(this.data), [this], 'exp'); out._back = () => { this.grad += out.data * out.grad; }; return out; }
  log() { const out = new Value(Math.log(this.data), [this], 'log'); out._back = () => { this.grad += out.grad / this.data; }; return out; }
  inv() { const out = new Value(1 / this.data, [this], '1/x'); out._back = () => { this.grad += (-1 / this.data ** 2) * out.grad; }; return out; }
  neg() { return this.mul(-1); }
  order() { const seen = new Set(), list = []; const go = v => { if (!seen.has(v)) { seen.add(v); v.prev.forEach(go); list.push(v); } }; go(this); return list; }
  backward() { const list = this.order(); this.grad = 1; [...list].reverse().forEach(v => v._back()); return list; }
}

/* ---------- convolution ---------- */

export const IMG5 = [[3, 3, 2, 1, 0], [0, 0, 1, 3, 1], [3, 1, 2, 2, 3], [2, 0, 0, 2, 2], [2, 0, 0, 0, 1]];
export const KERNEL3 = [[0, 1, 2], [2, 2, 0], [0, 1, 2]];
export const outSide = (n, f, p = 0, s = 1) => Math.floor((n + 2 * p - f) / s) + 1;
export const convParams = (cout, f, cin) => cout * (f * f * cin + 1);
const padded = (x, p) => { const n = x[0].length, zero = Array(n + 2 * p).fill(0), rows = x.map(r => [...Array(p).fill(0), ...r, ...Array(p).fill(0)]); return [...Array.from({ length: p }, () => [...zero]), ...rows, ...Array.from({ length: p }, () => [...zero])]; };
export function conv2d(img, k, s = 1, p = 0) {
  const x = padded(img, p), f = k.length, nr = Math.floor((x.length - f) / s) + 1, nc = Math.floor((x[0].length - k[0].length) / s) + 1;
  return Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => { let t = 0; for (let a = 0; a < f; a++) for (let b = 0; b < k[0].length; b++) t += x[i * s + a][j * s + b] * k[a][b]; return t; }));
}
export function pool2d(x, f, s, kind = 'max') {
  const nr = Math.floor((x.length - f) / s) + 1, nc = Math.floor((x[0].length - f) / s) + 1;
  return Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => { const w = []; for (let a = 0; a < f; a++) for (let b = 0; b < f; b++) w.push(x[i * s + a][j * s + b]); return kind === 'max' ? Math.max(...w) : mean(w); }));
}
export const conv1d = (x, k) => x.slice(0, x.length - k.length + 1).map((_, i) => k.reduce((s, w, j) => s + w * x[i + j], 0));
export const LENET = [['Input', '32×32×1', 0], ['Convolution, 5×5, six filters', '28×28×6', 156], ['Pooling', '14×14×6', 0], ['Convolution, 5×5, sixteen filters', '10×10×16', 2416], ['Pooling', '5×5×16', 0], ['Flatten', '400', 0], ['Dense', '120', 48120], ['Dense', '84', 10164], ['Output', '10', 850]];
export function iou(a, b) { // boxes are [x1, y1, x2, y2]
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), inter = w * h, area = q => (q[2] - q[0]) * (q[3] - q[1]);
  return inter / (area(a) + area(b) - inter || 1);
}
export function nms(boxes, thr) { // boxes are [x1, y1, x2, y2, score]
  const order = boxes.map((b, i) => i).sort((a, c) => boxes[c][4] - boxes[a][4]), keep = [];
  for (const i of order) if (keep.every(j => iou(boxes[i], boxes[j]) <= thr)) keep.push(i);
  return keep;
}

/* ---------- next-character models ---------- */

export const NAMES = 'aarav aditi aditya akash amit amita ananya anil anita anjali ankit anu anuj arjun aruna asha ashok bharat bhavna chandan chitra deepa deepak dev devi dhruv diya divya farhan gauri gita gopal govind harish hari heena hema indira isha ishaan jaya jyoti kabir kajal kamal kavita kiran krishna kumar lakshmi lalita madhu mahesh manish manoj maya meena mohan mohit mukesh nadia naveen neha nikhil nisha nitin padma pallavi pankaj parth pooja prakash pranav prem priya rahul raj rajesh rakesh ramesh rani ravi reena rekha ritu rohan rohit roshni sachin sanjay sanjana sara sarita seema shankar shanti sharda shiv shreya shweta simran sita sneha sonia sunil sunita suresh swati tanvi tara tarun uma usha varun vidya vijay vikram vinod vishal yash yogesh zara'.split(' ');
export const VOCAB = '.abcdefghijklmnopqrstuvwxyz';
export function ngram(names, order, alpha = 0.05) {
  const counts = new Map(), bump = (c, k) => { if (!counts.has(c)) counts.set(c, Array(27).fill(alpha)); counts.get(c)[k]++; };
  for (const w of names) { const s = '.'.repeat(order) + w + '.'; for (let i = order; i < s.length; i++) bump(s.slice(i - order, i), VOCAB.indexOf(s[i])); }
  const uniform = Array(27).fill(1 / 27);
  const probs = (ctx, T = 1) => { const c = counts.get(ctx); if (!c) return uniform; const w = c.map(v => v ** (1 / T)), t = w.reduce((a, b) => a + b, 0); return w.map(v => v / t); };
  const nll = list => { let s = 0, n = 0; for (const w of list) { const t = '.'.repeat(order) + w + '.'; for (let i = order; i < t.length; i++) { s -= Math.log(probs(t.slice(i - order, i))[VOCAB.indexOf(t[i])]); n++; } } return s / n; };
  const sample = (r, T = 1, cap = 10) => { let ctx = '.'.repeat(order), out = ''; while (out.length < cap) { const p = probs(ctx, T); let u = r(), k = 0; while (k < 26 && (u -= p[k]) > 0) k++; if (k === 0) break; out += VOCAB[k]; ctx = (ctx + VOCAB[k]).slice(-order); } return out; };
  return { probs, nll, sample, counts };
}
export const softmaxT = (z, T = 1) => { const m = Math.max(...z), w = z.map(v => Math.exp((v - m) / T)), t = w.reduce((a, b) => a + b, 0); return w.map(v => v / t); };
export const entropyNats = p => -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);

/* ---------- time series ---------- */

export function monthly(seed = 3, n = 120, slope = 0.5, sd = 2.4) {
  const r = rng(seed);
  return Array.from({ length: n }, (_, t) => r6(80 + slope * t + 10 * Math.sin((2 * Math.PI * t) / 12) + sd * gauss(r)));
}
export const BASELINES = {
  mean: (y, h) => Array.from({ length: h }, () => mean(y)),
  naive: (y, h) => Array.from({ length: h }, () => y.at(-1)),
  seasonal: (y, h, m = 12) => Array.from({ length: h }, (_, i) => y[y.length - m + (i % m)]),
  drift: (y, h) => Array.from({ length: h }, (_, i) => y.at(-1) + ((i + 1) * (y.at(-1) - y[0])) / (y.length - 1)),
};
export const forecastScores = (truth, pred) => { const e = truth.map((v, i) => v - pred[i]); return { mae: mean(e.map(Math.abs)), rmse: Math.sqrt(mean(e.map(v => v * v))), mape: 100 * mean(e.map((v, i) => Math.abs(v / truth[i]))) }; };
export const acf = (y, k) => { const m = mean(y), d = y.reduce((s, v) => s + (v - m) ** 2, 0); let t = 0; for (let i = 0; i + k < y.length; i++) t += (y[i] - m) * (y[i + k] - m); return t / d; };
export const difference = (y, m = 1) => y.slice(m).map((v, i) => v - y[i]);
export function arFit(y, p) {
  const rows = [], ys = [];
  for (let t = p; t < y.length; t++) { rows.push([1, ...Array.from({ length: p }, (_, k) => y[t - 1 - k])]); ys.push(y[t]); }
  const d = p + 1, G = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => rows.reduce((s, x) => s + x[a] * x[b], 0) + (a === b ? 1e-9 : 0)));
  return solve(G, Array.from({ length: d }, (_, a) => rows.reduce((s, x, i) => s + x[a] * ys[i], 0)));
}
export function arForecast(hist, c, h) { // recursive: each forecast becomes a lag
  const y = [...hist], out = [], p = c.length - 1;
  for (let i = 0; i < h; i++) { const v = c[0] + c.slice(1).reduce((s, w, k) => s + w * y[y.length - 1 - k], 0); out.push(v); y.push(v); }
  return out.map(v => (p ? v : v));
}
export const arSimulate = (c, phi, n, seed = 1, sd = 1) => { const r = rng(seed), y = [c / (1 - phi.reduce((a, b) => a + b, 0)), c / (1 - phi.reduce((a, b) => a + b, 0))]; for (let i = 0; i < n + 50; i++) y.push(r6(c + phi.reduce((s, w, k) => s + w * y[y.length - 1 - k], 0) + sd * gauss(r))); return y.slice(-n); };
export const rnnRun = (W, U, xs, b = 0) => { let h = 0; return xs.map(x => (h = Math.tanh(W * h + U * x + b))); };

/* ---------- reinforcement learning ---------- */

// '.' open, '#' wall, 'G' goal (+1, ends), 'P' pit (-1, ends), 'S' start.
export const GRID = ['....G', '.##.P', '.....', '.#...', 'S....'];
export const MOVES = [[-1, 0, 'up'], [0, 1, 'right'], [1, 0, 'down'], [0, -1, 'left']];
const cell = (g, a, b) => (a >= 0 && b >= 0 && a < g.length && b < g[0].length ? g[a][b] : '#');
export const startOf = g => { for (let a = 0; a < g.length; a++) { const b = g[a].indexOf('S'); if (b >= 0) return [a, b]; } return [0, 0]; };
export function gridStep(g, [a, b], m) {
  const na = a + MOVES[m][0], nb = b + MOVES[m][1], c = cell(g, na, nb), s = c === '#' ? [a, b] : [na, nb];
  return { s, r: c === 'G' ? 1 : c === 'P' ? -1 : 0, done: c === 'G' || c === 'P' };
}
export const isTerminal = c => c === 'G' || c === 'P' || c === '#';
export function valueIteration(g, gamma = 0.9, sweeps = 20) {
  const frames = [g.map(r => [...r].map(() => 0))];
  for (let t = 0; t < sweeps; t++) {
    const V = frames.at(-1);
    frames.push(g.map((row, a) => [...row].map((c, b) => (isTerminal(c) ? 0 : Math.max(...MOVES.map((_, m) => { const o = gridStep(g, [a, b], m); return r6(o.r + (o.done ? 0 : gamma * V[o.s[0]][o.s[1]])); }))))));
  }
  return frames;
}
export function qLearn(g, { alpha = 0.5, gamma = 0.9, eps = 0.1, episodes = 300, seed = 1, cap = 50 } = {}) {
  const r = rng(seed), Q = g.map(row => [...row].map(() => [0, 0, 0, 0])), returns = [];
  for (let ep = 0; ep < episodes; ep++) {
    let s = startOf(g), G = 0, disc = 1;
    for (let t = 0; t < cap; t++) {
      const q = Q[s[0]][s[1]], a = r() < eps ? Math.floor(r() * 4) : q.indexOf(Math.max(...q)), o = gridStep(g, s, a), future = o.done ? 0 : Math.max(...Q[o.s[0]][o.s[1]]);
      q[a] = r6(q[a] + alpha * (o.r + gamma * future - q[a])); G += disc * o.r; disc *= gamma; s = o.s;
      if (o.done) break;
    }
    returns.push(G);
  }
  return { Q, returns };
}
export const qUpdate = (q, r, gamma, best, alpha) => q + alpha * (r + gamma * best - q);
export function mountainCar(policy, { x0 = -0.45, cap = 200 } = {}) {
  let x = x0, v = 0; const path = [[0, x]];
  for (let t = 1; t <= cap; t++) { v = Math.max(-0.07, Math.min(0.07, +(v + (policy(t, x, v) - 1) * 0.001 - 0.0025 * Math.cos(3 * x)).toFixed(6))); x = +(x + v).toFixed(6); if (x < -1.2) { x = -1.2; v = 0; } path.push([t, x]); if (x >= 0.5) return { path, reached: t }; }
  return { path, reached: null };
}
export const discounted = (rewards, gamma) => rewards.reduceRight((G, r) => r + gamma * G, 0);
