'use client';
// Explorers for anomaly detection and monitoring.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { EIGHT_RUNS as RUNS, LATENCIES as LAT, averagePrecision, blobs, cFactor, covariance2, eig2, fmt, isolationForest, madOf, mahalanobis, medianOf, memo, normalPdf, pct, robustScore, stats1 } from '../lib/models.mjs';

const C8 = covariance2(RUNS), E8 = eig2(C8.a, C8.b, C8.d), V1 = E8.v, V2 = E8.u;

export const AnomalyTypes = truths({
  intro: 'What kind of unusual is each case?',
  items: [
    ['A single training run that takes 40 hours when all others take about 2 is a point anomaly.', true, 'One example, unusual by itself.'],
    ['A latency of 300 ms that is normal at peak load but unusual at 3 a.m. is a contextual anomaly.', true, 'It depends on the context in which it occurred.'],
    ['A sequence of small failures, each harmless by itself, is a collective anomaly.', true, 'Only the group together is unusual.'],
    ['Anomalies are usually plentiful, so labelled examples are easy to get.', false, 'They are rare, which is why the problem is usually posed without labels.'],
    ['An anomaly can make the estimated spread larger and so hide itself.', true, 'This is called masking.'],
    ['Shifting the centre so normal points look unusual is called swamping.', true, 'The anomaly distorts the statistics used to judge everything else.'],
  ],
  takeaway: 'Every method describes what usual looks like. Decide that first.',
});

export const StandardScoreDrill = drill({
  intro: 'Subtract the mean and divide by the standard deviation.',
  tex: R`z=\frac{x-\bar x}{s}`,
  gen: r => ({ m: int(r, 10, 50), s: int(r, 2, 12), x: int(r, 0, 90) }),
  q: v => `Mean $${v.m}$, standard deviation $${v.s}$. Find the standard score of $${v.x}$.`,
  answer: v => (v.x - v.m) / v.s, dp: 3,
  steps: v => [`$x-\\bar x=${v.x - v.m}$`, `Divide by $${v.s}$.`],
});

export const MaskingLab = lab({
  intro: 'Seven normal latencies and one slow request. Each band holds the values that would not be flagged. Make the request slower and watch which bands stretch.',
  params: [{ k: 'x', label: 'slow request (ms)', min: 40, max: 200, step: 1, init: 95 }],
  derive: ({ x }) => { const all = [...LAT, x], a = stats1(all), b = stats1(LAT), med = medianOf(all), w = (3 * madOf(all)) / 0.6745; return { all, a, b, med, w, zAll: (x - a.m) / a.s, zOut: (x - b.m) / b.s, rob: robustScore(x, all) }; },
  plots: v => { const rows = [[3, v.a.m - 3 * v.a.s, v.a.m + 3 * v.a.s, 2], [2, v.b.m - 3 * v.b.s, v.b.m + 3 * v.b.s, 1], [1, v.med - v.w, v.med + v.w, 3]]; return [{ x: [-160, 240], y: [0.4, 3.6], w: 480, h: 280, xl: 'latency (ms)', label: 'bands of allowed values', marks: [...rows.map(([y, a, b, t]) => ({ as: 'rects', tone: t, soft: true, data: [[a, y - 0.28, b, y + 0.28]] })), ...rows.map(([y]) => ({ as: 'points', tone: 'fg', r: 3.5, data: LAT.map(u => [u, y]) })), ...rows.map(([y, , b]) => ({ as: 'points', r: 6, data: [[v.x, y, v.x > b ? 4 : 1]] })), { as: 'label', x: -155, y: 3.42, text: 'mean ± 3 sd, all eight' }, { as: 'label', x: -155, y: 2.42, text: 'mean ± 3 sd, without the request' }, { as: 'label', x: -155, y: 1.42, text: 'median ± 3 robust sd' }] }]; },
  stats: v => [`standard score with it included: ${fmt(v.zAll, 3)}`, `standard score without it: ${fmt(v.zOut, 2)}`, `robust score: ${fmt(v.rob, 2)}`],
  status: v => (v.zAll < 3 ? ['alert', 'Masked: the request inflated $s$ and stayed under the $|z|>3$ rule.'] : ['ok', 'Caught by the standard score too.']),
  takeaway: 'With eight values, $|z|$ can never exceed $\\sqrt7\\approx2.65$, so no amount of slowness will ever trigger this rule here.',
});

export const SamuelsonDrill = drill({
  intro: 'With $n$ values and $s$ computed using $1/n$, no standard score can exceed $\\sqrt{n-1}$.',
  tex: R`|x_i-\bar x|\le s\sqrt{n-1}`,
  gen: r => ({ n: int(r, 4, 40) }),
  q: v => `You have $n=${v.n}$ values. What is the largest possible $|z|$?`,
  answer: v => Math.sqrt(v.n - 1), dp: 3,
  steps: v => [`$\\sqrt{n-1}=\\sqrt{${v.n - 1}}$`, v.n <= 10 ? 'At $n\\le10$, the $|z|>3$ rule can never fire.' : 'Here the rule can fire.'],
});

export const RobustScoreDrill = drill({
  intro: 'Median and median absolute deviation are barely affected by one bad value.',
  tex: R`\text{score}=\frac{0.6745\,(x-\text{median})}{\text{MAD}}`,
  gen: r => { const a = Array.from({ length: 4 }, () => int(r, 1, 12)); a.push(int(r, 30, 80)); return { a: a.sort((p, q) => p - q) }; },
  q: v => `For $${v.a.join(',\\ ')}$, find the robust score of $${v.a[4]}$.`,
  answer: v => (madOf(v.a) === 0 ? Infinity : robustScore(v.a[4], v.a)), dp: 2,
  steps: v => [`Median: $${medianOf(v.a)}$`, `Absolute deviations: $${v.a.map(x => Math.abs(x - medianOf(v.a))).join(',\\ ')}$`, `MAD: $${madOf(v.a)}$`],
});

const FIT = stats1(LAT), mix = x => 0.5 * normalPdf(x, 14.2, 4) + 0.5 * normalPdf(x, 32.8, 4), one = x => normalPdf(x, FIT.m, FIT.s);
export const DensityScore = lab({
  intro: 'Fit a model of usual and score by how improbable a value is. Compare a two-component mixture with a single Gaussian at a value in the empty middle.',
  tex: R`\text{score}(x)=-\log p(x)`,
  params: [{ k: 'model', label: 'Model', options: [['mix', 'two-component mixture'], ['one', 'single Gaussian']], init: 'mix' }, { k: 'x', label: 'latency $x$', min: 0, max: 100, step: 0.5, init: 23.5 }],
  derive: ({ model, x }) => { const p = model === 'mix' ? mix : one, lo = Math.min(...LAT.map(a => Math.log(p(a)))); return { p, lo, lp: Math.log(p(x)), bad: Math.log(p(x)) < lo }; },
  plots: v => [{ x: [0, 100], y: [0, 0.055], w: 440, h: 280, xl: 'latency (ms)', yl: 'density', label: 'fitted density', marks: [{ tone: 1, data: curve(v.p, 0, 100, 301) }, { as: 'rule', dash: true, tone: 4, x: v.x }, { as: 'points', tone: 'fg', r: 5, data: LAT.map(a => [a, 0.002]) }] }],
  stats: v => [`$\\log p(x)=${fmt(v.lp, 2)}$`, `threshold (lowest training value) ${fmt(v.lo, 2)}`],
  status: v => (v.bad ? ['alert', 'Flagged: less probable than every training value.'] : ['ok', 'Not flagged.']),
  takeaway: 'A density score is only as good as its model. A single Gaussian says the empty gap at 23 ms is the most usual place of all.',
});

const dNear = (p, k) => mean(RUNS.map(x => Math.hypot(x[0] - p[0], x[1] - p[1])).sort((a, b) => a - b).slice(0, k));
export const MahalanobisLab = lab({
  intro: 'Drag the new run. Euclidean distance grows equally in all directions. Mahalanobis distance grows slowly along the trend of the data and quickly across it.',
  tex: R`d_M(x)^2=\sum_j\frac{z_j^2}{\lambda_j}`,
  params: [{ k: 'px', label: 'training time', min: -2, max: 14, step: 0.05, init: 2 }, { k: 'py', label: 'peak memory', min: -2, max: 14, step: 0.05, init: 8 }],
  derive: ({ px, py }) => { const p = [px, py], u = [px - C8.m[0], py - C8.m[1]], z1 = u[0] * V1[0] + u[1] * V1[1], z2 = u[0] * V2[0] + u[1] * V2[1]; return { p, z1, z2, euclid: Math.hypot(u[0], u[1]), mah: mahalanobis(p, C8), knn: dNear(p, 2) }; },
  plots: v => { const eu = (x, y) => Math.hypot(x - C8.m[0], y - C8.m[1]), ma = (x, y) => mahalanobis([x, y], C8), one = (cap, fn, lv, t) => ({ caption: cap, x: [-3, 14], y: [-3, 14], w: 300, h: 300, label: cap, marks: [{ as: 'contour', tone: t, fn, levels: lv }, { as: 'points', tone: 'fg', r: 4.5, data: RUNS }, { drag: { x: 'px', y: 'py', tone: 4 } }] }); return [one('Euclidean contours', eu, [2, 4, 6, 8], 1), one('Mahalanobis contours', ma, [1, 2, 3, 4], 3)]; },
  stats: v => [`Mahalanobis ${fmt(v.mah, 3)}`, `Euclidean ${fmt(v.euclid, 3)}`, `from the principal line ${fmt(Math.abs(v.z2), 3)}`, `mean distance to two nearest runs ${fmt(v.knn, 3)}`],
  note: 'Try $(2,8)$ and $(11,11)$: the two distances rank them in opposite orders.',
  takeaway: 'Mahalanobis distance judges a point by the shape of the data and not only by how far it lies.',
});

export const MahalanobisDrill = drill({
  intro: 'In principal coordinates, each squared score is divided by its eigenvalue.',
  tex: R`d_M=\sqrt{\sum_jz_j^2/\lambda_j}`,
  gen: r => ({ z: [int(r, -8, 8), int(r, -5, 5)], l: [pick(r, [4, 9, 16, 25]), pick(r, [1, 0.25, 0.5, 4])] }),
  q: v => `Scores $z=(${v.z})$ and eigenvalues $\\lambda=(${v.l})$. Find the Mahalanobis distance.`,
  answer: v => Math.sqrt(v.z[0] ** 2 / v.l[0] + v.z[1] ** 2 / v.l[1]), dp: 3,
  steps: v => [`$z_1^2/\\lambda_1=${fmt(v.z[0] ** 2 / v.l[0], 3)}$`, `$z_2^2/\\lambda_2=${fmt(v.z[1] ** 2 / v.l[1], 3)}$`],
});

const NEW = [[2, 8], [11, 11]];
export const WhiteningSteps = lab({
  intro: 'Mahalanobis distance is ordinary distance after two moves. Centre, rotate onto the principal axes, then divide each axis by its standard deviation.',
  params: [{ k: 'step', label: 'step', min: 0, max: 3, step: 1, init: 0, play: true }],
  derive: ({ step }) => { const f = p => { let q = [...p]; if (step >= 1) q = [q[0] - C8.m[0], q[1] - C8.m[1]]; if (step >= 2) q = [q[0] * V1[0] + q[1] * V1[1], q[0] * V2[0] + q[1] * V2[1]]; if (step >= 3) q = [q[0] / Math.sqrt(E8.l1), q[1] / Math.sqrt(E8.l2)]; return q; }, o = step === 0 ? C8.m : [0, 0]; return { X: RUNS.map(f), N: NEW.map(f), o, name: ['The original runs', 'Centred at the mean', 'Rotated onto the principal axes', 'Each axis scaled to unit variance'][step] }; },
  plots: v => [{ caption: v.name, x: [-10, 14], y: [-10, 14], w: 400, h: 360, label: 'whitening transformation', marks: [{ as: 'points', tone: 'fg', r: 5, data: v.X }, { as: 'points', r: 7, data: [[v.N[0][0], v.N[0][1], 4], [v.N[1][0], v.N[1][1], 2]] }, { as: 'rings', r: 9, data: [v.o] }, { as: 'label', x: v.N[0][0] + 0.5, y: v.N[0][1] + 0.5, text: 'P' }, { as: 'label', x: v.N[1][0] + 0.5, y: v.N[1][1] + 0.5, text: 'Q' }] }],
  stats: v => [`distance of P from the ring: ${fmt(Math.hypot(v.N[0][0] - v.o[0], v.N[0][1] - v.o[1]), 3)}`, `distance of Q from the ring: ${fmt(Math.hypot(v.N[1][0] - v.o[0], v.N[1][1] - v.o[1]), 3)}`],
  status: v => (v.step === 3 ? ['ok', 'These ordinary distances are now the Mahalanobis distances: 4.316 and 2.351.'] : undefined),
  takeaway: 'Rotation changes no distances. Only the final scaling reverses the ranking of the two runs.',
});

export const KnnScoreDrill = drill({
  intro: 'Average the distance to the $k$ nearest training examples. No centre or shape is assumed.',
  gen: r => { const a = [...new Set(Array.from({ length: 6 }, () => int(r, 0, 30)))].sort((p, q) => p - q); return { a, x: int(r, 0, 35), k: pick(r, [2, 3]) }; },
  q: v => `Training values $${v.a.join(',\\ ')}$. Find the score of $${v.x}$ with $k=${v.k}$.`,
  answer: v => mean(v.a.map(u => Math.abs(u - v.x)).sort((p, q) => p - q).slice(0, v.k)), dp: 3,
  steps: v => [`Sorted distances: $${v.a.map(u => Math.abs(u - v.x)).sort((p, q) => p - q).join(',\\ ')}$`, `Average the first $${v.k}$.`],
});

const IsoPts = blobs(6, [[0, 0]], 60, 1).map(p => [Math.max(-3.4, Math.min(3.4, p[0])), Math.max(-3.4, Math.min(3.4, p[1]))]), OUT = [4.2, 3.8], ISO = [...IsoPts, OUT], INNER = IsoPts.reduce((u, v) => (Math.hypot(v[0], v[1]) < Math.hypot(u[0], u[1]) ? v : u));
const cutsOf = memo((who, seed) => { const t = who === 'out' ? OUT : INNER, r = rng(seed * 17), cuts = []; let S = ISO, lo = [-4.8, -4.8], hi = [4.8, 4.8]; while (S.length > 1 && cuts.length < 60) { const f = r() < 0.5 ? 0 : 1, v = S.map(p => p[f]), a = Math.min(...v), b = Math.max(...v); if (a === b) continue; const c = a + r() * (b - a); cuts.push(f === 0 ? [[c, lo[1]], [c, hi[1]]] : [[lo[0], c], [hi[0], c]]); if (t[f] < c) { hi = [...hi]; hi[f] = c; S = S.filter(p => p[f] < c); } else { lo = [...lo]; lo[f] = c; S = S.filter(p => p[f] >= c); } } return cuts; });
const avgCuts = memo(who => mean(Array.from({ length: 200 }, (_, i) => cutsOf(who, i + 1).length)));
export const IsolationCuts = lab({
  intro: 'Random cuts, each on a random feature between the smallest and largest remaining values, repeated until the circled point stands alone.',
  params: [{ k: 'who', label: 'Point', options: [['out', 'the outlier'], ['in', 'a typical point']], init: 'out' }, { k: 'seed', seed: 'New tree', init: 1 }, { k: 'step', label: 'cuts shown', min: 0, max: 16, step: 1, init: 0, play: true }],
  derive: ({ who, seed, step }) => { const cuts = cutsOf(who, seed); return { cuts, shown: cuts.slice(0, step), t: who === 'out' ? OUT : INNER, done: step >= cuts.length, av: avgCuts(who) }; },
  plots: v => [{ x: [-4.8, 4.8], y: [-4.8, 4.8], w: 380, h: 360, label: 'random isolation cuts', marks: [...v.shown.map(c => ({ tone: 2, data: c })), { as: 'points', tone: 'muted', r: 3, data: ISO }, { as: 'points', tone: 4, r: 5, data: [v.t] }, { as: 'rings', tone: 4, r: 11, data: [v.t] }] }],
  stats: v => [`alone after ${v.cuts.length} cuts in this tree`, `average over 200 trees ${fmt(v.av, 2)}`],
  status: v => (v.done ? ['ok', 'Isolated.'] : undefined),
  takeaway: 'An unusual point is cut off early because it lies alone in a region. Depth itself is the anomaly score.',
});

export const IsolationScoreDrill = drill({
  intro: '$c(m)$ is the average path length in a random tree on $m$ points, and it normalises the depth.',
  tex: R`c(m)=2H_{m-1}-\frac{2(m-1)}{m},\quad s=2^{-\bar h/c(\psi)}`,
  gen: r => ({ m: pick(r, [2, 4, 8, 16]), h: int(r, 1, 10) / 2 }),
  q: v => `$\\psi=${v.m}$ and the mean depth is $\\bar h=${v.h}$. Find the score $s$.`,
  answer: v => 2 ** (-v.h / cFactor(v.m)), dp: 3,
  steps: v => [`$c(${v.m})=${fmt(cFactor(v.m), 3)}$`, `Exponent: $-${v.h}/${fmt(cFactor(v.m), 3)}=${fmt(-v.h / cFactor(v.m), 3)}$`],
});

const NX = 54, NY = 36;
const forestGrid = memo((trees, psi) => { const F = isolationForest(ISO, { trees, psi, seed: 7 }), G = []; for (let i = 0; i < NX; i++) { G.push([]); for (let j = 0; j < NY; j++) G[i].push(+F.score([-5 + ((i + 0.5) * 10) / NX, -5 + ((j + 0.5) * 10) / NY]).toFixed(3)); } return { F, G }; });
export const IsolationLab = lab({
  intro: 'Colour shows the anomaly score at each location. Drag the query point, and change how many trees vote and how many points each one sees.',
  tex: R`s(x)=2^{-\bar h(x)/c(\psi)}`,
  params: [{ k: 'psi', label: 'Subsample', options: [[16, 'ψ = 16'], [64, 'ψ = 64'], [256, 'ψ = all']], init: 64 }, { k: 'trees', label: 'trees', min: 10, max: 200, step: 10, init: 100 }, { k: 'qx', label: 'query, first feature', min: -5, max: 5, step: 0.1, init: 4.2 }, { k: 'qy', label: 'query, second feature', min: -5, max: 5, step: 0.1, init: 3.8 }],
  derive: ({ trees, psi, qx, qy }) => { const { F, G } = forestGrid(trees, psi); return { G, score: F.score([qx, qy]), depth: F.depth([qx, qy]) }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 400, h: 360, label: 'isolation forest score field', marks: [{ as: 'field', bands: 6, nx: NX, ny: NY, tones: [4, 1], fn: (x, y) => 2 * (v.G[Math.min(NX - 1, Math.floor(((x + 5) / 10) * NX))][Math.min(NY - 1, Math.floor(((y + 5) / 10) * NY))] - 0.5) }, { as: 'points', tone: 'fg', r: 3, data: ISO }, { drag: { x: 'qx', y: 'qy', tone: 2 } }] }],
  stats: v => [`score ${fmt(v.score, 3)}`, `mean depth ${fmt(v.depth, 2)}`],
  status: v => (v.score > 0.6 ? ['alert', 'Isolated quickly: likely an anomaly.'] : ['ok', 'Takes many cuts: looks usual.']),
  takeaway: 'Isolation forests need no distances and no scaling, so they are a good first method on tables of features.',
});

export const PrecisionDrill = drill({
  intro: 'Average precision averages, over the known anomalies, the precision at the rank of each one.',
  tex: R`\text{AP}=\frac1a\sum_{j=1}^a\frac{j}{r_j}`,
  gen: r => { const s = new Set(); while (s.size < 3) s.add(int(r, 1, 12)); return { ranks: [...s].sort((a, b) => a - b), mode: pick(r, ['ap', 'pk']), k: int(r, 3, 8) }; },
  q: v => `Anomalies are ranked $${v.ranks.join(',\\ ')}$. Find ${v.mode === 'ap' ? 'the average precision' : `the precision at $${v.k}$`}.`,
  answer: v => (v.mode === 'ap' ? averagePrecision(v.ranks) : v.ranks.filter(x => x <= v.k).length / v.k), dp: 3,
  steps: v => (v.mode === 'ap' ? v.ranks.map((x, j) => `Anomaly ${j + 1} of 3 at rank $${x}$: precision $${j + 1}/${x}=${fmt((j + 1) / x, 3)}$`) : [`Anomalies within the top $${v.k}$: $${v.ranks.filter(x => x <= v.k).length}$`]),
});

const ranked = memo((sep, seed) => { const r = rng(seed * 53), a = [...Array.from({ length: 52 }, () => [gauss(r), 0]), ...Array.from({ length: 8 }, () => [sep + gauss(r), 1])].sort((p, q) => q[0] - p[0]); return { a, ranks: a.flatMap((p, i) => (p[1] ? [i + 1] : [])) }; });
export const RankingLab = lab({
  intro: 'Sixty cases, eight of them truly anomalies. A better score separates them more. Choose how many top cases an analyst can look at.',
  params: [{ k: 'sep', label: 'how well the score separates', min: 0, max: 5, step: 0.25, init: 2 }, { k: 'k', label: 'cases reviewed', min: 1, max: 60, step: 1, init: 8 }, { k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ sep, k, seed }) => { const { a, ranks } = ranked(sep, seed), hit = ranks.filter(x => x <= k).length; return { a, ranks, hit, p: hit / k, rc: hit / 8, ap: averagePrecision(ranks) }; },
  plots: v => [{ caption: 'Cases, highest score first', x: [0, 61], y: [0, 2], w: 360, h: 200, xl: 'rank', label: 'cases by rank', marks: [{ as: 'stems', tone: 'muted', data: v.a.map((p, i) => (p[1] ? null : [i + 1, 0.6, 1.4])).filter(Boolean) }, { as: 'stems', tone: 4, data: v.a.map((p, i) => (p[1] ? [i + 1, 0.3, 1.7] : null)).filter(Boolean) }, { as: 'rule', dash: true, tone: 1, x: v.k + 0.5 }] }, { caption: 'Precision at each depth', x: [1, 60], y: [0, 1], w: 300, h: 200, xl: 'k', label: 'precision at k', marks: [{ tone: 2, data: v.a.map((_, i) => [i + 1, v.ranks.filter(x => x <= i + 1).length / (i + 1)]) }, { as: 'rule', dash: true, tone: 1, x: v.k }, { as: 'rule', dash: true, tone: 'muted', y: 8 / 60 }] }],
  stats: v => [`average precision ${fmt(v.ap, 3)}`, `precision at ${v.k}: ${fmt(v.p, 3)}`, `found ${v.hit} of 8 (recall ${fmt(v.rc, 3)})`, `random scoring averages about ${fmt(8 / 60, 3)}`],
  takeaway: 'Accuracy is useless here, because answering "normal" every time is 87% accurate. Judge the ranking at the depth someone can actually review.',
});

const showdown = memo(kind => {
  const r = rng(kind === 'A' ? 91 : 92), g = () => gauss(r);
  const normal = kind === 'A' ? Array.from({ length: 200 }, () => { const t = 3 * g(), n = 0.35 * g(); return [0.8 * t - 0.6 * n, 0.6 * t + 0.8 * n]; }) : [...blobs(93, [[-3.2, -2.6]], 100, 0.65), ...blobs(94, [[3.2, 2.6]], 100, 0.65)];
  const bad = kind === 'A' ? Array.from({ length: 20 }, (_, i) => { const t = -3 + (6 * i) / 19, n = (i % 2 ? 1 : -1) * (1.5 + 0.4 * r()); return [0.8 * t - 0.6 * n, 0.6 * t + 0.8 * n]; }) : Array.from({ length: 10 }, () => [1.2 * g(), 1 * g()]);
  const X = [...normal, ...bad].map(p => [+p[0].toFixed(3), +p[1].toFixed(3)]), lab2 = X.map((_, i) => +(i >= normal.length)), a = bad.length, cov = covariance2(X), F = isolationForest(X, { trees: 150, psi: 128, seed: 5 });
  const knn5 = x => X.map(y => Math.hypot(x[0] - y[0], x[1] - y[1])).sort((p, q) => p - q)[5];
  const scores = { euclid: x => Math.hypot(x[0] - cov.m[0], x[1] - cov.m[1]), mahal: x => mahalanobis(x, cov), knn: knn5, iso: x => F.score(x) };
  const out = {};
  for (const k in scores) { const s = X.map(scores[k]), ord = X.map((_, i) => i).sort((p, q) => s[q] - s[p]), ranks = ord.flatMap((id, i) => (lab2[id] ? [i + 1] : [])); out[k] = { top: new Set(ord.slice(0, a)), ap: averagePrecision(ranks), pk: ranks.filter(u => u <= a).length / a }; }
  return { X, lab2, a, out };
});
const NAMES = { euclid: 'Euclidean from mean', mahal: 'Mahalanobis', knn: 'Nearest-neighbour', iso: 'Isolation forest' };
export const ScoreShowdown = lab({
  intro: 'Two datasets with known anomalies, four scores. Large outlined dots are what each score flags as most unusual, as many as there are true anomalies.',
  params: [{ k: 'kind', label: 'Data', options: [['A', 'elongated cloud'], ['B', 'two clusters']], init: 'A' }, { k: 'score', label: 'Score', options: Object.entries(NAMES), init: 'mahal' }],
  derive: ({ kind, score }) => { const d = showdown(kind); return { ...d, cur: d.out[score] }; },
  plots: v => [{ x: [-7, 7], y: [-6, 6], w: 400, h: 340, label: 'data and flagged points', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: v.X.filter((_, i) => !v.lab2[i]) }, { as: 'points', tone: 4, r: 4.5, data: v.X.filter((_, i) => v.lab2[i]) }, { as: 'rings', tone: 1, r: 7, data: v.X.filter((_, i) => v.cur.top.has(i)) }] }],
  stats: v => [`average precision ${fmt(v.cur.ap, 3)}`, `precision at ${v.a}: ${fmt(v.cur.pk, 2)}`],
  table: v => ({ head: ['score', 'AP'], rows: Object.keys(NAMES).map(k => [NAMES[k], fmt(v.out[k].ap, 3)]) }),
  legend: [['--c4', 'true anomaly'], ['--fg', 'flagged (ring)']],
  takeaway: 'No score is best in general. A Mahalanobis score suits one elongated cloud, and a nearest-neighbour score suits several separated clusters.',
});

const REF = blobs(71, [[0, 0]], 200, 1), RECENT = blobs(72, [[0, 0]], 400, 1);
const nnd = (p, A, skip = -1) => { let b = Infinity; for (let i = 0; i < A.length; i++) if (i !== skip) { const d = (A[i][0] - p[0]) ** 2 + (A[i][1] - p[1]) ** 2; if (d < b) b = d; } return Math.sqrt(b); };
const LIMIT = (() => { const s = REF.map((p, i) => nnd(p, REF, i)).sort((a, b) => a - b); return s[Math.floor(0.95 * s.length)]; })();
const alertRate = memo(sh => mean(RECENT.map(p => +(nnd([p[0] + sh, p[1] + sh * 0.6], REF) > LIMIT))));
const SHIFTS = Array.from({ length: 31 }, (_, i) => +(i * 0.2).toFixed(1));
export const DriftMonitor = lab({
  intro: 'A deployed model’s recent inputs drift away from the reference set. An input raises an alert when its nearest reference point is farther than 95% of the reference’s own neighbours.',
  params: [{ k: 'sh', label: 'drift', min: 0, max: 6, step: 0.2, init: 0 }],
  derive: ({ sh }) => ({ cur: RECENT.map(p => [+(p[0] + sh).toFixed(3), +(p[1] + sh * 0.6).toFixed(3)]), rate: alertRate(sh), curveR: SHIFTS.map(s => [s, alertRate(s)]) }),
  plots: v => [{ caption: 'Reference and recent inputs', x: [-4, 10], y: [-4, 8], w: 320, h: 280, label: 'reference and recent inputs', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: REF }, { as: 'points', tone: 4, r: 3, data: v.cur.slice(0, 120) }] }, { caption: 'Alert rate against drift', x: [0, 6], y: [0, 1], w: 300, h: 280, xl: 'drift', label: 'alert rate', marks: [{ tone: 1, data: v.curveR }, { as: 'rule', dash: true, tone: 4, x: v.sh }, { as: 'rule', dash: true, tone: 'muted', y: 0.05 }] }],
  stats: v => [`alert rate ${pct(v.rate, 1)}`, 'about 5% is expected when nothing has changed'],
  status: v => (v.rate > 0.15 ? ['alert', 'Recent inputs no longer resemble the reference. Investigate before trusting the model.'] : ['ok', 'Close to the expected false-alarm rate.']),
  takeaway: 'A model can fail silently when its inputs drift. Monitoring treats each batch as a possible anomaly, and thresholds must balance missed drift against false alarms.',
});

export const AnomalyCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Euclidean distance from the mean is a sound score for elongated data.', false, 'It ignores the shape, so a point along the trend looks worse than one off it.'],
    ['Mahalanobis distance is unchanged by converting hours to minutes.', true, 'It is invariant under any invertible linear change of units.'],
    ['With ten values the $|z|>3$ rule can never fire.', true, 'Samuelson’s inequality: $|z|\\le\\sqrt9=3$.'],
    ['A single Gaussian density is safe for data with two clusters.', false, 'It places high density in the gap, where no normal data lie.'],
    ['An isolation forest requires features to be scaled.', false, 'Cuts are per feature and depend only on order within that feature’s range.'],
    ['A random scorer has an average precision near the base rate.', true, 'Anomalies are scattered uniformly through the ranking.'],
    ['One method is best on every dataset.', false, 'Which score wins depends on what kind of unusual the data contain.'],
    ['Accuracy is a good measure when anomalies are 1% of the data.', false, 'Predicting normal every time is already 99% accurate.'],
    ['The threshold should reflect the cost of a false alarm.', true, 'Frequent spurious alerts are soon ignored.'],
  ],
  takeaway: 'Name the kind of unusual you expect, then choose the score that can detect it.',
});
