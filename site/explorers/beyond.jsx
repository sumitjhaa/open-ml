'use client';
// Advanced explorers on topics neither source course teaches.
import { R, curve, drill, int, lab, pick } from '../components/labs';
import { gauss, mean, rng, solve } from '../lib/math.mjs';
import { blobs, fmt, kmeans, linspace, memo, pct, svd } from '../lib/models.mjs';
import { softmaxT } from '../lib/extended.mjs';

export const CramerRaoDrill = drill({
  intro: 'No unbiased estimator can have variance below the inverse of the total Fisher information.',
  tex: R`\text{Var}(\hat p)\ge\frac{p(1-p)}{n}`,
  gen: r => ({ p: pick(r, [0.5, 0.2, 0.1, 0.7]), n: pick(r, [10, 50, 100, 400]) }),
  q: v => `A Bernoulli with $p=${v.p}$ and $n=${v.n}$ observations. Find the Cramér-Rao lower bound on the variance.`,
  answer: v => (v.p * (1 - v.p)) / v.n, dp: 5,
  steps: v => [`$I(p)=1/(p(1-p))=${fmt(1 / (v.p * (1 - v.p)), 3)}$`],
});
const mleVar = memo((n, seed) => { const r = rng(seed * 7 + n), trial = () => { const x = Array.from({ length: n }, () => gauss(r)), m = mean(x), s = x.reduce((t, v) => t + (v - m) ** 2, 0); return [s / n, s / (n - 1)]; }, T = Array.from({ length: 3000 }, trial); return [mean(T.map(t => t[0])), mean(T.map(t => t[1]))]; });
export const BiasedMle = lab({
  intro: 'The maximum likelihood estimate of a variance divides by $n$, and comes out too small on average. True variance here is 1.',
  params: [{ k: 'n', label: 'sample size', min: 2, max: 30, step: 1, init: 4 }, { k: 'seed', seed: 'New trials', init: 1 }],
  derive: ({ n, seed }) => ({ e: mleVar(n, seed) }),
  plots: v => [{ x: [2, 30], y: [0, 1.2], w: 480, h: 260, xl: 'n', label: 'expected variance estimate', marks: [{ tone: 1, data: curve(n => (n - 1) / n, 2, 30, 57) }, { as: 'rule', dash: true, tone: 3, y: 1 }, { as: 'points', tone: 4, r: 6, data: [[v.n, v.e[0]]] }] }],
  stats: v => [`dividing by $n$: mean ${fmt(v.e[0], 3)}`, `dividing by $n-1$: mean ${fmt(v.e[1], 3)}`, `theory ${fmt((v.n - 1) / v.n, 3)}`],
  takeaway: 'The sample mean is fitted to the same data, which uses up one degree of freedom.',
});
const huber = (e, d) => (Math.abs(e) <= d ? 0.5 * e * e : d * (Math.abs(e) - 0.5 * d));
const HX = linspace(0, 10, 25), HY = (() => { const r = rng(4); return HX.map(x => +(1 + 0.8 * x + 0.5 * gauss(r)).toFixed(3)); })();
const huberFit = (X, Y, d) => { let a = 0, b = 0; for (let t = 0; t < 40; t++) { const w = X.map((x, i) => { const e = Math.abs(Y[i] - a - b * x); return e <= d ? 1 : d / e; }), G = [[w.reduce((s, v) => s + v, 0), w.reduce((s, v, i) => s + v * X[i], 0)], [0, w.reduce((s, v, i) => s + v * X[i] * X[i], 0)]]; G[1][0] = G[0][1]; [a, b] = solve(G, [w.reduce((s, v, i) => s + v * Y[i], 0), w.reduce((s, v, i) => s + v * X[i] * Y[i], 0)]); } return [a, b]; };
export const HuberLab = lab({
  intro: 'Drag a point far away. Squared loss follows it. Huber loss stops caring once an error passes $\\delta$.',
  tex: R`L_\delta(e)=\tfrac12e^2\ \text{if}\ |e|\le\delta,\ \text{else}\ \delta(|e|-\tfrac12\delta)`,
  params: [{ k: 'o', label: 'value of the last point', min: 0, max: 40, step: 0.5, init: 30 }, { k: 'd', label: 'threshold $\\delta$', min: 0.2, max: 10, step: 0.1, init: 1 }],
  derive: ({ o, d }) => { const Y = [...HY.slice(0, -1), o]; return { Y, ls: huberFit(HX, Y, 1e6), hub: huberFit(HX, Y, d) }; },
  plots: v => [{ caption: 'Fits', x: [0, 10], y: [0, 40], w: 320, h: 260, xl: 'x', label: 'least squares and Huber fits', marks: [{ as: 'points', tone: 'fg', r: 3.5, data: HX.map((x, i) => [x, v.Y[i]]) }, { tone: 4, data: curve(x => v.ls[0] + v.ls[1] * x, 0, 10, 3) }, { tone: 1, data: curve(x => v.hub[0] + v.hub[1] * x, 0, 10, 3) }, { drag: { x: 10, y: 'o', tone: 2 } }] }, { caption: 'Loss', x: [-6, 6], y: [0, 10], w: 300, h: 260, xl: 'error', label: 'loss shapes', marks: [{ tone: 4, data: curve(e => Math.min(10, e * e / 2), -6, 6) }, { tone: 1, data: curve(e => huber(e, v.d), -6, 6) }] }],
  stats: v => [`least squares slope ${fmt(v.ls[1], 3)}`, `Huber slope ${fmt(v.hub[1], 3)}`, 'true slope 0.8'],
  legend: [['--c4', 'squared loss'], ['--c1', 'Huber']],
  takeaway: 'Huber keeps the smoothness of squared loss near zero and the robustness of absolute loss far from it.',
});
export const MulticlassField = lab({
  intro: 'Three classes with a softmax. The regions are always convex, and meet in straight lines.',
  tex: R`p_k=\frac{e^{w_k^\top x+b_k}}{\sum_je^{w_j^\top x+b_j}}`,
  params: [{ k: 'a', label: 'direction of class B (degrees)', min: 60, max: 180, step: 5, init: 120 }, { k: 'b', label: 'bias of class C', min: -2, max: 2, step: 0.1, init: 0 }, { k: 's', label: 'sharpness', min: 0.5, max: 6, step: 0.5, init: 2 }],
  plots: v => { const W = [[1, 0], [Math.cos((v.a * Math.PI) / 180), Math.sin((v.a * Math.PI) / 180)], [-0.5, -0.87]], B = [0, 0, v.b], f = (x, y) => softmaxT(W.map((w, k) => v.s * (w[0] * x + w[1] * y + B[k]))); return [{ x: [-3, 3], y: [-3, 3], w: 360, h: 340, label: 'softmax regions', marks: [{ as: 'field', bands: 3, tones: [1, 2], fn: (x, y) => { const p = f(x, y); return p[0] - p[1]; } }, { as: 'contour', tone: 'fg', fn: (x, y) => { const p = f(x, y); return Math.max(...p) - 0.5; }, levels: [0.001] }] }]; },
  stats: () => ['blue and orange shading show classes A and B, and the pale area is class C'],
  takeaway: 'Softmax compares linear scores, so its boundaries are straight, even with many classes.',
});
export const EceDrill = drill({
  intro: 'Expected calibration error is the weighted gap between confidence and accuracy.',
  tex: R`\text{ECE}=\sum_b\frac{n_b}{N}\,|\text{acc}_b-\text{conf}_b|`,
  gen: r => ({ n1: int(r, 10, 60), n2: int(r, 10, 60), c1: pick(r, [0.6, 0.7, 0.8]), a1: pick(r, [0.5, 0.65, 0.75]), c2: pick(r, [0.85, 0.9, 0.95]), a2: pick(r, [0.7, 0.8, 0.9]) }),
  q: v => `Bin one has $${v.n1}$ points, confidence $${v.c1}$, accuracy $${v.a1}$. Bin two has $${v.n2}$ points, confidence $${v.c2}$, accuracy $${v.a2}$. Find the ECE.`,
  answer: v => (v.n1 * Math.abs(v.a1 - v.c1) + v.n2 * Math.abs(v.a2 - v.c2)) / (v.n1 + v.n2), dp: 4,
  steps: v => [`Gaps: $${fmt(Math.abs(v.a1 - v.c1), 2)}$ and $${fmt(Math.abs(v.a2 - v.c2), 2)}$`, `Total points: $${v.n1 + v.n2}$`],
});
const CAL = (() => { const r = rng(14); return Array.from({ length: 600 }, () => { const x = 6 * r(); return [+x.toFixed(3), +(Math.sin(x) + 0.3 * gauss(r)).toFixed(3)]; }); })();
export const ConformalLab = lab({
  intro: 'Whatever the model, calibration errors give a band that covers new points at the promised rate.',
  tex: R`q=\text{the }\lceil(n+1)(1-\alpha)\rceil\text{-th smallest error}`,
  params: [{ k: 'a', label: 'allowed miss rate $\\alpha$', min: 0.02, max: 0.4, step: 0.01, init: 0.1 }, { k: 'm', label: 'Model', options: [['good', 'sine curve'], ['bad', 'straight line']], init: 'good' }],
  derive: ({ a, m }) => { const f = x => (m === 'good' ? Math.sin(x) : 0.1 - 0.05 * x), cal = CAL.slice(0, 300), test = CAL.slice(300), S = cal.map(([x, y]) => Math.abs(y - f(x))).sort((p, q) => p - q), q = S[Math.min(299, Math.ceil(301 * (1 - a)) - 1)], cov = mean(test.map(([x, y]) => +(Math.abs(y - f(x)) <= q))); return { f, q, cov, test }; },
  plots: v => [{ x: [0, 6], y: [-3, 3], w: 480, h: 280, xl: 'x', label: 'prediction band', marks: [{ as: 'area', tone: 1, soft: true, data: linspace(0, 6, 61).map(x => [x, v.f(x) - v.q, v.f(x) + v.q]) }, { as: 'points', tone: 'muted', r: 2, data: v.test }, { tone: 1, data: curve(v.f, 0, 6) }] }],
  stats: v => [`coverage on new points ${pct(v.cov, 1)}`, `target ${pct(1 - v.a, 0)}`, `band half-width ${fmt(v.q, 3)}`],
  takeaway: 'A worse model gets a wider band, not a broken guarantee.',
});
const SX = linspace(0, 6, 40), SY = (() => { const r = rng(26); return SX.map(x => +(Math.sin(x) + 0.35 * gauss(r)).toFixed(3)); })(), ST = (() => { const r = rng(27); return Array.from({ length: 300 }, () => { const x = 6 * r(); return [x, Math.sin(x) + 0.35 * gauss(r)]; }); })();
const nn1 = (X, Y, q) => { let b = 0; X.forEach((x, i) => { if (Math.abs(x - q) < Math.abs(X[b] - q)) b = i; }); return Y[b]; };
const line2 = (X, Y) => { const mx = mean(X), my = mean(Y), s = X.reduce((t, x, i) => t + (x - mx) * (Y[i] - my), 0) / X.reduce((t, x) => t + (x - mx) ** 2, 0); return x => my + s * (x - mx); };
const stack = memo(() => { const ls = line2(SX, SY), P = (inA, inB, y) => { const G = [[inA.reduce((s, v) => s + v * v, 1e-9), inA.reduce((s, v, i) => s + v * inB[i], 0)], [0, inB.reduce((s, v) => s + v * v, 1e-9)]]; G[1][0] = G[0][1]; return solve(G, [inA.reduce((s, v, i) => s + v * y[i], 0), inB.reduce((s, v, i) => s + v * y[i], 0)]); }, inA = SX.map(ls), inB = SY, oofA = [], oofB = []; SX.forEach((x, i) => { const Xr = SX.filter((_, j) => j !== i), Yr = SY.filter((_, j) => j !== i); oofA.push(line2(Xr, Yr)(x)); oofB.push(nn1(Xr, Yr, x)); }); const score = w => mean(ST.map(([x, y]) => (w[0] * ls(x) + w[1] * nn1(SX, SY, x) - y) ** 2)); return { inS: P(inA, inB, SY), oof: P(oofA, oofB, SY), score, ls }; });
export const StackingLab = lab({
  intro: 'Blend a straight line with a one-neighbour model. Fit the blend on in-sample predictions, or on out-of-fold ones.',
  params: [{ k: 'm', label: 'Blend fitted on', options: [['in', 'in-sample predictions'], ['oof', 'out-of-fold predictions']], init: 'in' }],
  derive: ({ m }) => { const S = stack(), w = m === 'in' ? S.inS : S.oof; return { w, mse: S.score(w), f: x => w[0] * S.ls(x) + w[1] * nn1(SX, SY, x) }; },
  plots: v => [{ x: [0, 6], y: [-2.5, 2.5], w: 480, h: 280, xl: 'x', label: 'stacked prediction', marks: [{ as: 'points', tone: 'fg', r: 3, data: SX.map((x, i) => [x, SY[i]]) }, { tone: 1, data: curve(v.f, 0, 6, 301) }] }],
  stats: v => [`test error ${fmt(v.mse, 3)}`, `weights: line ${fmt(v.w[0], 2)}, neighbour ${fmt(v.w[1], 2)}`],
  takeaway: 'In-sample, a one-neighbour model looks perfect, so the blend trusts it blindly. Out-of-fold predictions tell the truth.',
});
const RINGS = (() => { const r = rng(33), o = []; for (let i = 0; i < 60; i++) { const y = +(i >= 20), t = 2 * Math.PI * ((y ? i - 20 : i) / (y ? 40 : 20)) + 0.06 * gauss(r), R0 = y ? 3 : 1; o.push([+(R0 * Math.cos(t) + 0.12 * gauss(r)).toFixed(3), +(R0 * Math.sin(t) + 0.12 * gauss(r)).toFixed(3), y]); } return o; })(), XR = RINGS.map(p => [p[0], p[1]]);
const specRun = memo(sig => { const n = XR.length, W = XR.map(a => XR.map(b => Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * sig * sig)))), D = W.map(r0 => r0.reduce((s, v) => s + v, 0)), L = W.map((r0, i) => r0.map((v, j) => +((i === j ? D[i] : 0) - v).toFixed(8))), { V } = svd(L), E = XR.map((_, i) => [+V[i][n - 1].toFixed(6), +V[i][n - 2].toFixed(6)]), m = Math.max(...E.flat().map(Math.abs)) || 1, nm = E.map(p => [p[0] / m, p[1] / m]), f = [1, 2, 3, 4].map(s => kmeans(nm, 2, { seed: s, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a)); return { assign: f.assign }; });
const kmRings = memo(() => [1, 2, 3].map(s => kmeans(XR, 2, { seed: s, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a)).assign);
const agree = a => { const c = mean(a.map((v, i) => +(v === RINGS[i][2]))); return Math.max(c, 1 - c); };
export const SpectralLab = lab({
  intro: 'K-means cannot separate rings. Build a similarity graph first, and cluster its eigenvectors instead.',
  tex: R`L=D-W`,
  params: [{ k: 's', label: 'similarity width $\\sigma$', min: 0.2, max: 2, step: 0.1, init: 0.4 }],
  derive: ({ s }) => ({ A: specRun(s).assign, B: kmRings() }),
  plots: v => { const f = (a, t) => ({ caption: t, x: [-4, 4], y: [-4, 4], w: 300, h: 280, label: t, marks: [{ as: 'points', r: 3.5, data: XR.map((x, i) => [x[0], x[1], a[i] + 1]) }] }); return [f(v.B, 'K-means on the points'), f(v.A, 'K-means on the eigenvectors')]; },
  stats: v => [`spectral agreement ${pct(agree(v.A), 0)}`, `plain K-means agreement ${pct(agree(v.B), 0)}`],
  takeaway: 'Connectivity, not distance to a centre, decides the groups. The width must suit the data.',
});
const kpcaRun = memo(g => { const K = XR.map(a => XR.map(b => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)))), row = K.map(r0 => mean(r0)), all = mean(row), C = K.map((r0, i) => r0.map((v, j) => +(v - row[i] - row[j] + all).toFixed(8))), S = svd(C); return XR.map((_, i) => [S.U[i][0] * S.s[0], S.U[i][1] * S.s[1]]).map(p => [+p[0].toFixed(5), +p[1].toFixed(5)]); });
export const KernelPcaLab = lab({
  intro: 'Run PCA on a kernel matrix. Rings that no straight direction separates can come apart.',
  tex: R`\tilde K=K-\mathbf 1K-K\mathbf 1+\mathbf 1K\mathbf 1`,
  params: [{ k: 'g', label: 'kernel width $\\gamma$', min: 0.02, max: 1, step: 0.02, init: 0.1 }],
  derive: ({ g }) => { const Z = kpcaRun(g), m = Math.ceil(Math.max(...Z.flat().map(Math.abs)) * 1.1 * 10) / 10 || 1; return { Z, m }; },
  plots: v => [{ caption: 'Original', x: [-4, 4], y: [-4, 4], w: 300, h: 280, label: 'original rings', marks: [{ as: 'points', r: 3.5, data: RINGS.map(p => [p[0], p[1], p[2] + 1]) }] }, { caption: 'First two kernel components', x: [-v.m, v.m], y: [-v.m, v.m], w: 300, h: 280, label: 'kernel PCA embedding', marks: [{ as: 'points', r: 3.5, data: v.Z.map((p, i) => [p[0], p[1], RINGS[i][2] + 1]) }] }],
  stats: () => ['colours are the true rings'],
  takeaway: 'Kernel PCA finds curved directions by working in an implicit feature space.',
});
const LOFP = [...blobs(11, [[-2.5, -1]], 40, 0.18), ...blobs(12, [[2, 1.5]], 40, 1), [-1.4, -1]];
const knnD = (i, k) => LOFP.map((p, j) => [Math.hypot(p[0] - LOFP[i][0], p[1] - LOFP[i][1]), j]).filter(p => p[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, k);
const lofRun = memo(k => { const N = LOFP.map((_, i) => knnD(i, k)), kd = N.map(n => n.at(-1)[0]), lrd = N.map((n, i) => n.length / n.reduce((s, [d, j]) => s + Math.max(kd[j], d), 0)), lof = N.map((n, i) => mean(n.map(([, j]) => lrd[j])) / lrd[i]); return { lof, kd }; });
export const LofLab = lab({
  intro: 'One point sits just beside a tight cluster. A global distance misses it, and a local density ratio catches it. Disc size is the score.',
  params: [{ k: 'k', label: 'neighbours', min: 3, max: 20, step: 1, init: 8 }, { k: 'm', label: 'Score', options: [['lof', 'local outlier factor'], ['knn', 'distance to neighbour']], init: 'lof' }],
  derive: ({ k, m }) => { const { lof, kd } = lofRun(k), s = m === 'lof' ? lof : kd, rank = [...s].sort((a, b) => b - a).indexOf(s.at(-1)) + 1; return { s, rank, top: Math.max(...s) }; },
  plots: v => [{ x: [-4.5, 6], y: [-4, 5], w: 420, h: 340, label: 'outlier scores', marks: [{ as: 'points', data: LOFP.map((p, i) => [p[0], p[1], i === LOFP.length - 1 ? 4 : 1, 2 + (10 * v.s[i]) / v.top]) }] }],
  stats: v => [`the red point ranks number ${v.rank} of ${LOFP.length}`],
  takeaway: 'A ratio of local densities adapts to clusters of different spreads.',
});
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 4]], NODES = [[0, 1], [1, 2], [2, 2], [3, 1], [2, 0], [1, 0]];
export const GnnLab = lab({
  intro: 'One node starts with a signal. Each round, every node averages with its neighbours, so information travels one hop.',
  tex: R`h_v'=\frac{1}{|N(v)|+1}\sum_{u\in N(v)\cup\{v\}}h_u`,
  params: [{ k: 't', label: 'rounds', min: 0, max: 8, step: 1, init: 0, play: true }, { k: 's', label: 'source node', min: 0, max: 5, step: 1, init: 0 }],
  derive: ({ t, s }) => { const adj = NODES.map((_, i) => EDGES.flatMap(([a, b]) => (a === i ? [b] : b === i ? [a] : []))); let h = NODES.map((_, i) => +(i === s)); for (let r = 0; r < t; r++) h = h.map((v, i) => (v + adj[i].reduce((c, j) => c + h[j], 0)) / (adj[i].length + 1)); return { h }; },
  plots: v => [{ x: [-0.5, 3.5], y: [-0.5, 2.5], w: 440, h: 320, label: 'graph with node values', marks: [...EDGES.map(([a, b]) => ({ tone: 'muted', data: [NODES[a], NODES[b]] })), { as: 'points', data: NODES.map((p, i) => [p[0], p[1], 1, 4 + 26 * Math.sqrt(v.h[i])]) }, ...NODES.map((p, i) => ({ as: 'label', x: p[0], y: p[1] - 0.05, text: v.h[i].toFixed(2), anchor: 'middle' }))] }],
  stats: v => [`nodes reached ${v.h.filter(x => x > 1e-9).length} of ${NODES.length}`],
  takeaway: 'After $k$ rounds a node has heard from $k$ hops away, and too many rounds make all the nodes alike.',
});
const CF = memo(s => { const r = rng(s * 3 + 1); return Array.from({ length: 400 }, () => { const z = gauss(r), t = +(0.9 * z + 0.5 * gauss(r)).toFixed(4), y = +(0.5 * t + 2 * z + 0.4 * gauss(r)).toFixed(4); return [z, t, y]; }); });
export const ConfounderLab = lab({
  intro: 'A hidden factor drives both the treatment and the outcome. The true effect of treatment is 0.5. Compare the two estimates.',
  params: [{ k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ seed }) => { const D = CF(seed), line = (X, Y) => { const a = mean(X), b = mean(Y); return X.reduce((s, x, i) => s + (x - a) * (Y[i] - b), 0) / X.reduce((s, x) => s + (x - a) ** 2, 0); }, naive = line(D.map(d => d[1]), D.map(d => d[2])), G = [[0, 0], [0, 0]], g = [0, 0]; D.forEach(([z, t, y]) => { G[0][0] += t * t; G[0][1] += t * z; G[1][1] += z * z; g[0] += t * y; g[1] += z * y; }); G[1][0] = G[0][1]; return { D, naive, adj: solve(G, g)[0] }; },
  plots: v => [{ x: [-4, 4], y: [-8, 8], w: 480, h: 300, xl: 'treatment', yl: 'outcome', label: 'treatment against outcome', marks: [{ as: 'points', tone: 'muted', r: 2, data: v.D.map(d => [d[1], d[2]]) }, { tone: 4, data: curve(x => v.naive * x, -4, 4, 3) }, { tone: 1, data: curve(x => v.adj * x, -4, 4, 3) }] }],
  stats: v => [`without adjusting ${fmt(v.naive, 3)}`, `adjusting for the confounder ${fmt(v.adj, 3)}`, 'true effect 0.5'],
  legend: [['--c4', 'unadjusted'], ['--c1', 'adjusted']],
  takeaway: 'A strong prediction is not an effect. Randomisation, or adjusting for the confounders, is what makes the difference.',
});
const ARMS = [0.2, 0.45, 0.5, 0.7];
const banditRun = memo((mode, seed) => { const r = rng(seed * 17), n = [0, 0, 0, 0], q = [0, 0, 0, 0], regret = []; let reg = 0; for (let t = 1; t <= 1500; t++) { const score = q.map((v, a) => (n[a] ? v + (mode === 'ucb' ? Math.sqrt((2 * Math.log(t)) / n[a]) : 0) : 1e9)), a = mode === 'eps' && r() < 0.1 ? Math.floor(r() * 4) : score.indexOf(Math.max(...score)), w = +(r() < ARMS[a]); n[a]++; q[a] += (w - q[a]) / n[a]; reg += 0.7 - ARMS[a]; if (t % 10 === 0) regret.push([t, reg]); } return { regret, n }; });
export const BanditLab = lab({
  intro: 'Four slot machines with unknown payouts. Compare strategies by how much reward they give up.',
  tex: R`a_t=\arg\max_a\ \bar r_a+\sqrt{2\ln t/n_a}`,
  params: [{ k: 'm', label: 'Strategy', options: [['ucb', 'upper confidence bound'], ['eps', 'ε-greedy'], ['greedy', 'greedy']], init: 'ucb' }, { k: 'seed', seed: 'New run', init: 1 }],
  derive: ({ m, seed }) => banditRun(m, seed),
  plots: v => [{ x: [0, 1500], y: [0, 300], w: 480, h: 280, xl: 'pulls', yl: 'regret', label: 'cumulative regret', marks: [{ tone: 1, data: v.regret }] }],
  stats: v => [`total regret ${fmt(v.regret.at(-1)[1], 1)}`, `pulls per arm ${v.n.join(', ')}`],
  takeaway: 'Exploring in proportion to uncertainty makes regret grow slowly, like $\\log t$.',
});
