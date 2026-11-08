'use client';
// Explorers for PCA and neighbour-embedding maps.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { EIGHT_RUNS as RUNS, covariance2, eig2, fmt, linspace, memo, pct, svd, tsne } from '../lib/models.mjs';

const S8 = covariance2(RUNS), E8 = eig2(S8.a, S8.b, S8.d), TOTAL = S8.a + S8.d;

export const ProjectionShadow = lab({
  intro: 'The score is the signed length of a point’s shadow on a unit direction. Drag the point and turn the direction.',
  tex: R`z=u^\top x,\qquad\|x\|^2=z^2+\|x-zu\|^2`,
  params: [{ k: 'ax', label: 'first coordinate', min: -1, max: 5, step: 0.1, init: 2 }, { k: 'ay', label: 'second coordinate', min: -1, max: 5, step: 0.1, init: 3 }, { k: 'deg', label: 'direction (degrees)', min: 0, max: 180, step: 1, init: 14 }],
  derive: ({ ax, ay, deg }) => { const t = (deg * Math.PI) / 180, u = [Math.cos(t), Math.sin(t)], z = ax * u[0] + ay * u[1], foot = [z * u[0], z * u[1]]; return { u, z, foot, res: Math.hypot(ax - foot[0], ay - foot[1]), len: Math.hypot(ax, ay) }; },
  plots: v => [{ x: [-2, 6], y: [-2, 6], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'projection onto a direction', marks: [{ tone: 'muted', data: [[-9 * v.u[0], -9 * v.u[1]], [9 * v.u[0], 9 * v.u[1]]] }, { as: 'arrow', tone: 1, data: [[0, 0, v.u[0], v.u[1]]] }, { tone: 4, dash: true, data: [[v.ax, v.ay], v.foot] }, { tone: 2, data: [[0, 0], v.foot] }, { as: 'points', tone: 2, r: 5, data: [v.foot] }, { drag: { x: 'ax', y: 'ay', tone: 3 } }] }],
  stats: v => [`score $z=${fmt(v.z, 3)}$`, `residual length ${fmt(v.res, 3)}`, `$${fmt(v.len ** 2, 3)}=${fmt(v.z ** 2, 3)}+${fmt(v.res ** 2, 3)}$`],
  takeaway: 'A projection splits a vector into a kept part and a perpendicular lost part, and their squared lengths add up exactly.',
});

const TRIPLES = [[3, 4, 5], [5, 12, 13], [8, 15, 17], [4, 3, 5], [12, 5, 13]];
export const ProjectionScoreDrill = drill({
  intro: 'Multiply matching coordinates, add, and divide by the length of the direction.',
  tex: R`z=\frac{a^\top x}{\|a\|}`,
  gen: r => ({ t: pick(r, TRIPLES), x: [int(r, -5, 9), int(r, -5, 9)] }),
  q: v => `Find the score of $(${v.x})$ along the direction $(${v.t[0]},${v.t[1]})$.`,
  answer: v => (v.t[0] * v.x[0] + v.t[1] * v.x[1]) / v.t[2], dp: 3,
  steps: v => [`Dot product: $${v.t[0] * v.x[0] + v.t[1] * v.x[1]}$`, `Length of the direction: $${v.t[2]}$`],
});

export const DirectionVariance = lab({
  intro: 'Turn the line through the eight runs. Variance kept rises and falls, and at every angle kept plus lost is the same total.',
  tex: R`s^2(u)=u^\top Su`,
  params: [{ k: 'deg', label: 'direction (degrees)', min: 0, max: 180, step: 1, init: 0, show: v => `${v}°` }],
  derive: ({ deg }) => { const t = (deg * Math.PI) / 180, u = [Math.cos(t), Math.sin(t)], kept = S8.a * u[0] ** 2 + 2 * S8.b * u[0] * u[1] + S8.d * u[1] ** 2, feet = RUNS.map(x => { const z = (x[0] - S8.m[0]) * u[0] + (x[1] - S8.m[1]) * u[1]; return [S8.m[0] + z * u[0], S8.m[1] + z * u[1]]; }); return { u, kept, feet, lost: TOTAL - kept }; },
  plots: v => [{ caption: 'The runs and their shadows', x: [-1, 11], y: [-1, 12], w: 320, h: 300, xl: 'training time', yl: 'peak memory', label: 'projections on a line', marks: [{ tone: 1, data: [[S8.m[0] - 12 * v.u[0], S8.m[1] - 12 * v.u[1]], [S8.m[0] + 12 * v.u[0], S8.m[1] + 12 * v.u[1]]] }, ...RUNS.map((x, i) => ({ tone: 4, dash: true, data: [x, v.feet[i]] })), { as: 'rings', r: 5, data: v.feet }, { as: 'points', tone: 'fg', r: 5, data: RUNS }] }, { caption: 'Variance kept, by angle', x: [0, 180], y: [0, 13], w: 300, h: 300, xl: 'degrees', label: 'variance against angle', marks: [{ tone: 1, data: curve(d => { const t = (d * Math.PI) / 180; return S8.a * Math.cos(t) ** 2 + 2 * S8.b * Math.sin(t) * Math.cos(t) + S8.d * Math.sin(t) ** 2; }, 0, 180) }, { as: 'rule', dash: true, tone: 4, x: v.deg }, { as: 'points', tone: 2, r: 6, data: [[v.deg, v.kept]] }] }],
  stats: v => [`variance kept ${fmt(v.kept, 3)}`, `lost ${fmt(v.lost, 3)}`, `total ${fmt(TOTAL, 3)}`],
  status: v => (Math.abs(v.kept - E8.l1) < 0.01 ? ['ok', 'This is the first principal direction: the variance kept equals the largest eigenvalue.'] : undefined),
  takeaway: 'Maximising the variance kept and minimising the reconstruction error are the same problem.',
});

export const CovarianceEntryDrill = drill({
  intro: 'Centre each feature, then average the products of matching deviations.',
  tex: R`S_{jk}=\frac1n\sum_i(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)`,
  gen: r => { const n = pick(r, [3, 4]); let a; do { a = Array.from({ length: n }, () => [int(r, 0, 9), int(r, 0, 9)]); } while (a.reduce((s, p) => s + p[0], 0) % n || a.reduce((s, p) => s + p[1], 0) % n); return { a }; },
  q: v => `Points ${v.a.map(p => `$(${p})$`).join(', ')}. Find the covariance $S_{12}$ of the two features.`,
  answer: v => covariance2(v.a).b, dp: 3,
  steps: v => { const c = covariance2(v.a); return [`Means: $${c.m[0]}$ and $${c.m[1]}$`, `Products of deviations: $${v.a.map(p => (p[0] - c.m[0]) * (p[1] - c.m[1])).join(',\\ ')}$`, `Divide their sum by $${v.a.length}$.`]; },
});

export const PcaEigenDrill = drill({
  intro: 'For a $2\\times2$ symmetric matrix, the eigenvalues come from the trace and the determinant.',
  tex: R`\lambda=\frac{\operatorname{tr}S\pm\sqrt{(\operatorname{tr}S)^2-4\det S}}{2}`,
  gen: r => ({ a: int(r, 2, 12), b: int(r, 0, 6), d: int(r, 2, 12) }),
  q: v => `$S=\\begin{bmatrix}${v.a}&${v.b}\\\\${v.b}&${v.d}\\end{bmatrix}$. What fraction of the variance does the first component explain?`,
  answer: v => eig2(v.a, v.b, v.d).l1 / (v.a + v.d), dp: 3,
  steps: v => [`$\\operatorname{tr}S=${v.a + v.d}$ and $\\det S=${v.a * v.d - v.b * v.b}$`, `$\\lambda_1=${fmt(eig2(v.a, v.b, v.d).l1, 3)}$`],
});

export const ExplainedDrill = drill({
  intro: 'Add the eigenvalues you keep, and divide by the sum of them all.',
  gen: r => { const l = Array.from({ length: 5 }, () => int(r, 1, 20)).sort((a, b) => b - a); return { l, k: int(r, 1, 3) }; },
  q: v => `Eigenvalues are $${v.l.join(',\\ ')}$. What proportion of the variance do the first $${v.k}$ explain?`,
  answer: v => v.l.slice(0, v.k).reduce((a, b) => a + b, 0) / v.l.reduce((a, b) => a + b, 0), dp: 3,
  steps: v => [`Kept: $${v.l.slice(0, v.k).reduce((a, b) => a + b, 0)}$`, `Total: $${v.l.reduce((a, b) => a + b, 0)}$`],
});

export const ReconstructionErrorDrill = drill({
  intro: 'The squared error left after keeping $k$ components is $n$ times the sum of the dropped eigenvalues.',
  tex: R`\sum_i\|x_i-\hat x_i\|^2=n\sum_{j>k}\lambda_j`,
  gen: r => { const l = Array.from({ length: 4 }, () => int(r, 1, 12)).sort((a, b) => b - a); return { l, k: int(r, 1, 3), n: int(r, 5, 20) }; },
  q: v => `$n=${v.n}$ with eigenvalues $${v.l.join(',\\ ')}$. Find the total squared error when keeping $${v.k}$ component${v.k > 1 ? 's' : ''}.`,
  answer: v => v.n * v.l.slice(v.k).reduce((a, b) => a + b, 0), dp: 0,
  steps: v => [`Dropped eigenvalues: $${v.l.slice(v.k).join(',\\ ')}$`],
});

const DIMS = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
const curse = memo(d => { const r = rng(d + 8), q = Array.from({ length: d }, () => r()), D = Array.from({ length: 500 }, () => { let s = 0; for (let j = 0; j < d; j++) s += (r() - q[j]) ** 2; return Math.sqrt(s); }), lo = Math.min(...D), hi = Math.max(...D); return { D, lo, hi, ratio: (hi - lo) / lo }; });
export const CurseDistance = lab({
  intro: 'Five hundred random points and one query. As features are added, the nearest and farthest points become almost equally far away.',
  params: [{ k: 'i', label: 'number of features', min: 0, max: 9, step: 1, init: 0, show: v => DIMS[v] }],
  derive: ({ i }) => { const c = curse(DIMS[i]), bins = Array(24).fill(0); c.D.forEach(x => { bins[Math.min(23, Math.floor((x / c.hi) * 24))]++; }); return { ...c, bins }; },
  plots: v => [{ caption: 'Distances, as a share of the largest', x: [0, 1], y: [0, 500], w: 300, h: 260, xl: 'distance / largest distance', label: 'histogram of distances', marks: [{ as: 'rects', tone: 1, data: v.bins.map((c, i) => [i / 24, 0, (i + 0.85) / 24, c]) }] }, { caption: 'Spread of distances', x: [0, 9], y: [-1.5, 2.5], w: 300, h: 260, xl: 'features (2 to 1000, log spaced)', yl: 'log₁₀ ratio', label: 'ratio against features', marks: [{ tone: 2, data: DIMS.map((d, i) => [i, Math.log10(curse(d).ratio)]) }, { as: 'rule', dash: true, tone: 4, x: v.i }] }],
  stats: v => [`(farthest − nearest) / nearest $=${fmt(v.ratio, 2)}$`],
  takeaway: 'When every point is about equally far, "nearest" means little. That is one reason to reduce the number of features.',
});

const EV = [5.452, 3.341, 0.202, 0.18, 0.167, 0.153, 0.151, 0.128, 0.119, 0.107], cumE = EV.map((_, i) => EV.slice(0, i + 1).reduce((a, b) => a + b, 0) / 10);
export const ScreeRules = lab({
  intro: 'Ten standardised metrics give ten eigenvalues adding up to 10. Three common rules give two different answers.',
  params: [{ k: 'k', label: 'components kept', min: 1, max: 10, step: 1, init: 2 }],
  plots: v => [{ caption: 'Eigenvalues', x: [0.4, 10.6], y: [0, 6], w: 300, h: 260, xl: 'component', label: 'scree plot', marks: [{ as: 'rects', tone: 1, data: EV.slice(0, v.k).map((l, i) => [i + 0.7, 0, i + 1.3, l]) }, { as: 'rects', tone: 'muted', soft: true, data: EV.slice(v.k).map((l, i) => [i + v.k + 0.7, 0, i + v.k + 1.3, l]) }, { as: 'rule', dash: true, tone: 4, y: 1 }] }, { caption: 'Cumulative share of variance', x: [1, 10], y: [0.4, 1], w: 300, h: 260, xl: 'components', label: 'cumulative proportion', marks: [{ tone: 2, data: cumE.map((c, i) => [i + 1, c]) }, { as: 'rule', dash: true, tone: 4, y: 0.9 }, { as: 'points', tone: 2, r: 6, data: [[v.k, cumE[v.k - 1]]] }] }],
  stats: v => [`${pct(cumE[v.k - 1], 2)} explained`],
  table: () => ({ head: ['rule', 'components'], rows: [['Elbow of the scree plot', 2], ['Eigenvalue above 1', EV.filter(l => l > 1).length], ['At least 90% explained', cumE.findIndex(c => c >= 0.9) + 1]] }),
  takeaway: 'These are conventions. Decide from the purpose: at most three for a plot, or whatever error your task can tolerate.',
});

const UNITS = (() => { const r = rng(14); return Array.from({ length: 80 }, (_, i) => [+((i % 2 ? 6.6 : 3.8) + 0.5 * gauss(r)).toFixed(3), +(20000 + 3634 * gauss(r)).toFixed(0), i % 2]); })();
export const ScalingPca = lab({
  intro: 'The two kinds of run differ in training time. Memory is in megabytes, so its numbers are far larger. See what PCA does before and after standardising.',
  params: [{ k: 'std', label: 'Features', options: [['raw', 'raw units'], ['std', 'standardised']], init: 'raw' }],
  derive: ({ std }) => { const m = [0, 1].map(j => mean(UNITS.map(p => p[j]))), s = [0, 1].map(j => Math.sqrt(mean(UNITS.map(p => (p[j] - m[j]) ** 2)))), X = UNITS.map(p => (std === 'std' ? [(p[0] - m[0]) / s[0], (p[1] - m[1]) / s[1]] : [p[0], p[1]])), c = covariance2(X), e = eig2(c.a, c.b, c.d), z = X.map(x => (x[0] - c.m[0]) * e.v[0] + (x[1] - c.m[1]) * e.v[1]), g = k => z.filter((_, i) => UNITS[i][2] === k), gap = Math.abs(mean(g(0)) - mean(g(1))) / Math.sqrt((mean(g(0).map(u => (u - mean(g(0))) ** 2)) + mean(g(1).map(u => (u - mean(g(1))) ** 2))) / 2), xr = X.map(p => p[0]), yr = X.map(p => p[1]); return { X, c, e, z, gap, box: [Math.min(...xr), Math.max(...xr), Math.min(...yr), Math.max(...yr)], zmax: Math.max(...z.map(Math.abs)) * 1.1 }; },
  plots: v => { const [a, b, c, d] = v.box, sc = v.std === 'std' ? 1 : 1000, pad = (p, q) => [+(p - 0.08 * (q - p)).toFixed(1), +(q + 0.08 * (q - p)).toFixed(1)], L = Math.hypot(b - a, d - c), top = Math.ceil(v.zmax / sc); return [{ caption: 'Data and first direction', x: pad(a, b), y: pad(c / sc, d / sc), w: 300, h: 260, xl: 'training time', yl: sc > 1 ? 'memory (thousand MB)' : 'memory', label: 'runs in feature space', marks: [{ as: 'points', r: 3.5, data: v.X.map((x, i) => [x[0], x[1] / sc, UNITS[i][2] + 1]) }, { tone: 4, data: [[v.c.m[0] - L * v.e.v[0], (v.c.m[1] - L * v.e.v[1]) / sc], [v.c.m[0] + L * v.e.v[0], (v.c.m[1] + L * v.e.v[1]) / sc]] }] }, { caption: 'Score on the first direction', x: [-top, top], y: [0, 3], w: 300, h: 260, xl: sc > 1 ? 'score (thousands)' : 'score', label: 'scores by group', marks: [{ as: 'points', r: 3.5, data: v.z.map((u, i) => [u / sc, 1 + UNITS[i][2], UNITS[i][2] + 1]) }] }]; },
  stats: v => [`first component explains ${pct(v.e.l1 / (v.e.l1 + v.e.l2), 2)}`, `direction $(${fmt(v.e.v[0], 4)},\\ ${fmt(v.e.v[1], 4)})$`, `group separation on the score: ${fmt(v.gap, 2)}`],
  takeaway: 'A large variance is not the same as an interesting one. PCA never sees which variation you care about, and units alone can decide its answer.',
});

const gs = d => Math.exp(-d * d / 2), ts = d => 1 / (1 + d * d);
export const KernelTails = lab({
  intro: 'Similarity in the data uses a Gaussian. Similarity on the map uses a Student-$t$. Compare them at a moderate distance.',
  params: [{ k: 'd', label: 'distance', min: 0, max: 8, step: 0.1, init: 4 }],
  plots: v => [{ caption: 'Similarity', x: [0, 8], y: [0, 1], w: 300, h: 260, xl: 'distance', label: 'similarity against distance', marks: [{ tone: 1, data: curve(gs, 0, 8) }, { tone: 2, data: curve(ts, 0, 8) }, { as: 'rule', dash: true, tone: 4, x: v.d }] }, { caption: 'On a log₁₀ scale', x: [0, 8], y: [-8, 0], w: 300, h: 260, xl: 'distance', label: 'log similarity against distance', marks: [{ tone: 1, data: curve(d => Math.max(-8, Math.log10(gs(d))), 0, 8) }, { tone: 2, data: curve(d => Math.log10(ts(d)), 0, 8) }, { as: 'rule', dash: true, tone: 4, x: v.d }] }],
  stats: v => [`Gaussian ${gs(v.d).toExponential(2)}`, `Student-$t$ ${fmt(ts(v.d), 4)}`, `ratio ${fmt(ts(v.d) / gs(v.d), 1)}`],
  legend: [['--c1', 'Gaussian'], ['--c2', 'Student-t']],
  takeaway: 'The heavy tail lets moderately distant pairs sit far apart on the map, which relieves crowding.',
});

const DISTS = [[[0.5, 0.5], 'two equal'], [[0.25, 0.25, 0.25, 0.25], 'four equal'], [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], 'eight equal'], [[0.5, 0.25, 0.25], 'one half, then quarters'], [[0.5, 0.25, 0.125, 0.125], 'halving']];
const ent = p => -p.reduce((s, q) => s + q * Math.log2(q), 0);
export const PerplexityDrill = drill({
  intro: 'Perplexity is an effective number of neighbours. It is two to the power of the entropy in bits.',
  tex: R`H=-\sum_jp_j\log_2p_j,\qquad\text{Perp}=2^H`,
  gen: r => ({ d: pick(r, DISTS)[0] }),
  q: v => `A point puts probabilities $${v.d.join(',\\ ')}$ on its neighbours. Find its perplexity.`,
  answer: v => 2 ** ent(v.d), dp: 3,
  steps: v => [`Each term $-p\\log_2p$: $${v.d.map(p => fmt(-p * Math.log2(p), 3)).join(',\\ ')}$`, `$H=${fmt(ent(v.d), 3)}$ bits`],
});

export const KlDrill = drill({
  intro: 'KL divergence is zero only when the two distributions match, and t-SNE minimises it.',
  tex: R`\text{KL}(P\|Q)=\sum_ip_i\ln\frac{p_i}{q_i}`,
  gen: r => { const p = int(r, 2, 8) / 10; let q = int(r, 1, 9) / 10; if (q === p) q = q >= 0.5 ? q - 0.1 : q + 0.1; return { p, q: +q.toFixed(1) }; },
  q: v => `$P=(${v.p},\\ ${fmt(1 - v.p, 1)})$ and $Q=(${v.q},\\ ${fmt(1 - v.q, 1)})$. Find $\\text{KL}(P\\|Q)$ in nats.`,
  answer: v => v.p * Math.log(v.p / v.q) + (1 - v.p) * Math.log((1 - v.p) / (1 - v.q)), dp: 3,
  steps: v => [`First term: $${v.p}\\ln(${v.p}/${v.q})=${fmt(v.p * Math.log(v.p / v.q), 4)}$`, `Second term: $${fmt(1 - v.p, 1)}\\ln(${fmt(1 - v.p, 1)}/${fmt(1 - v.q, 1)})=${fmt((1 - v.p) * Math.log((1 - v.p) / (1 - v.q)), 4)}$`],
});

const ROLL = (() => { const r = rng(30), o = []; for (let i = 0; i < 100; i++) { const t = 4.7 + 9.4 * r(), h = 12 * r(); o.push({ x: [+(t * Math.cos(t)).toFixed(3), +h.toFixed(3), +(t * Math.sin(t)).toFixed(3)], c: 1 + Math.min(3, Math.floor(((t - 4.7) / 9.4) * 4)) }); } return o; })();
const CLUST = (() => { const r = rng(31), o = []; for (let i = 0; i < 90; i++) { const g = i % 3; o.push({ x: Array.from({ length: 5 }, (_, j) => +((j === g ? 8 : 0) + gauss(r)).toFixed(3)), c: g + 1 }); } return o; })();
const SETS = { roll: ROLL, clust: CLUST };
const pcaMap = X => { const m = X[0].map((_, j) => mean(X.map(x => x[j]))), A = X.map(x => x.map((v, j) => v - m[j])), { V } = svd(A); return A.map(a => [a.reduce((s, v, j) => s + v * V[j][0], 0), a.reduce((s, v, j) => s + v * V[j][1], 0)]); };
const knn = (X, i, k) => X.map((x, j) => [x.reduce((s, v, d) => s + (v - X[i][d]) ** 2, 0), j]).filter(p => p[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, k).map(p => p[1]);
const kept = (X, Y, k = 8) => mean(X.map((_, i) => { const a = new Set(knn(X, i, k)); return knn(Y, i, k).filter(j => a.has(j)).length / k; }));
const tMap = memo((kind, perp, seed) => { const X = SETS[kind].map(p => p.x), Y = tsne(X, { perplexity: perp, iters: 300, seed }), Z = pcaMap(X); return { Y, Z, kt: kept(X, Y), kp: kept(X, Z) }; });
const box = Y => { const m = Math.ceil(Math.max(...Y.flatMap(p => p.map(Math.abs))) * 1.1) || 1; return [-m, m]; };
export const TsneLab = lab({
  intro: 'Colours follow position in the original data. PCA can only rotate and flatten, while t-SNE tries to keep neighbours together. Vary the perplexity and compare.',
  params: [{ k: 'kind', label: 'Data', options: [['roll', 'rolled sheet'], ['clust', 'groups in five features']], init: 'roll' }, { k: 'perp', label: 'perplexity', min: 5, max: 40, step: 5, init: 15 }, { k: 'seed', seed: 'New start', init: 1 }],
  derive: ({ kind, perp, seed }) => ({ ...tMap(kind, perp, seed), C: SETS[kind].map(p => p.c) }),
  plots: v => { const f = (Y, t) => ({ caption: t, x: box(Y), y: box(Y), w: 300, h: 300, label: t, marks: [{ as: 'points', r: 3.5, data: Y.map((p, i) => [p[0], p[1], v.C[i]]) }] }); return [f(v.Z, 'PCA'), f(v.Y, 't-SNE')]; },
  stats: v => [`neighbours kept: t-SNE ${pct(v.kt, 0)}`, `neighbours kept: PCA ${pct(v.kp, 0)}`],
  note: 'Neighbours kept is the share of each point’s eight nearest points in the data that are among its eight nearest on the map.',
  takeaway: 'Perplexity changes the picture a great deal. Always compare several perplexities and several seeds before believing a shape.',
});

const LIES = (() => { const r = rng(44), c = [Array(10).fill(0), [12, ...Array(9).fill(0)], [51, ...Array(9).fill(0)]], sd = [0.3, 2.1, 0.3], o = []; for (let i = 0; i < 75; i++) { const g = i % 3; o.push({ x: c[g].map(v => +(v + sd[g] * gauss(r)).toFixed(3)), g }); } return o; })();
const spread = (X, ids) => { const m = X[0].map((_, j) => mean(ids.map(i => X[i][j]))); return mean(ids.map(i => Math.sqrt(X[i].reduce((s, v, j) => s + (v - m[j]) ** 2, 0)))); };
const cent = (X, ids) => X[0].map((_, j) => mean(ids.map(i => X[i][j])));
const gap = (X, a, b) => Math.sqrt(cent(X, a).reduce((s, v, j) => s + (v - cent(X, b)[j]) ** 2, 0));
const ID = [0, 1, 2].map(g => LIES.flatMap((p, i) => (p.g === g ? [i] : [])));
const lieMap = memo((perp, seed) => { const X = LIES.map(p => p.x), Y = tsne(X, { perplexity: perp, iters: 300, seed }); return { X, Y }; });
export const MapLies = lab({
  intro: 'Three clusters in ten features. Group B is seven times as spread as A, and C is four times as far from A as B is. See how much of that survives on the map.',
  params: [{ k: 'perp', label: 'perplexity', min: 5, max: 30, step: 5, init: 15 }, { k: 'seed', seed: 'New start', init: 1 }],
  derive: ({ perp, seed }) => lieMap(perp, seed),
  plots: v => [{ x: box(v.Y), y: box(v.Y), w: 380, h: 360, label: 't-SNE map of three clusters', marks: [{ as: 'points', r: 4, data: v.Y.map((p, i) => [p[0], p[1], LIES[i].g + 1]) }] }],
  stats: () => ['Read the table, then change the perplexity.'],
  table: v => ({ head: ['ratio', 'data', 'map'], rows: [['spread of B / spread of A', fmt(spread(v.X, ID[1]) / spread(v.X, ID[0]), 2), fmt(spread(v.Y, ID[1]) / spread(v.Y, ID[0]), 2)], ['A to C / A to B', fmt(gap(v.X, ID[0], ID[2]) / gap(v.X, ID[0], ID[1]), 2), fmt(gap(v.Y, ID[0], ID[2]) / gap(v.Y, ID[0], ID[1]), 2)]] }),
  takeaway: 'Cluster sizes and the gaps between clusters on such a map are not reliable. Only local neighbourhoods are.',
});

export const ReadingMaps = truths({
  intro: 'How much can you conclude from a t-SNE or UMAP picture?',
  items: [
    ['Points that are neighbours in the data are usually close on the map.', true, 'That is exactly what the objective rewards.'],
    ['Two points close on the map must be neighbours in the data.', false, 'The map has to squeeze many dimensions into two, so some false neighbours appear.'],
    ['The size of a cluster on the map measures its spread in the data.', false, 'The map equalises densities, so sizes carry little information.'],
    ['A larger gap between clusters on the map means they are farther apart in the data.', false, 'Global distances are largely not preserved.'],
    ['A cluster on the map proves a cluster exists in the data.', false, 'Structureless data can form apparent clusters. Compare other seeds and perplexities.'],
    ['Different seeds and perplexities should be tried before trusting a shape.', true, 'Structure that persists across runs deserves more belief.'],
  ],
  takeaway: 'Use such maps to form hypotheses, and test those in the original features.',
});

export const PcaCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Every matrix has an SVD.', true, 'Rectangular or singular, it always exists.'],
    ['PCA and the SVD of the centred data give the same directions.', true, 'The right singular vectors are the eigenvectors of the covariance.'],
    ['The principal directions are perpendicular to each other.', true, 'They are eigenvectors of a symmetric matrix.'],
    ['Maximising variance and minimising reconstruction error give different directions.', false, 'Kept plus lost is a constant, so they are the same problem.'],
    ['Each eigenvalue of the covariance is the variance along its eigenvector.', true, '$v^\\top Sv=\\lambda v^\\top v=\\lambda$.'],
    ['Standardising before PCA never changes the result.', false, 'It changes the covariance matrix, so it can change every direction.'],
    ['PCA finds the direction that best separates the classes.', false, 'PCA never sees labels. High variance need not be the variation you care about.'],
    ['Principal scores on different components are uncorrelated.', true, '$v_j^\\top Sv_l=0$ whenever $j\\ne l$.'],
    ['With curved data, a nonlinear map can preserve neighbours that PCA cannot.', true, 'PCA can only apply a linear projection.'],
  ],
  takeaway: 'PCA is a precise, linear, label-blind summary of variance.',
});
