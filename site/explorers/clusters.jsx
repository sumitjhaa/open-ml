'use client';
// Explorers for K-means in depth, Gaussian mixtures, hierarchical and density-based clustering.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { EIGHT_RUNS as RUNS, LATENCIES as LAT, agglomerate, blobs, calinski, dbscan, eig2, em1d, em2d, fmt, kDistance, kmeans, linspace, medianOf, memo, moons, normalPdf, pct, silhouettes, spreads, meanSilhouette } from '../lib/models.mjs';

const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
const tone = a => (a % 4) + 1;
const CLOUD = blobs(4, [[-2.2, -1.2], [2, -1.5], [0.2, 2.2]], 18, 0.55);
const uniform = (seed, n = 54, lo = -4.5, hi = 4.5) => { const r = rng(seed); return Array.from({ length: n }, () => [+(lo + (hi - lo) * r()).toFixed(3), +(lo + (hi - lo) * r()).toFixed(3)]); };
const bestOf = (X, k, tries = 5) => [1, 2, 3, 4, 5].slice(0, tries).map(s => kmeans(X, k, { seed: s * 13, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a));

/* ---------- K-means, by hand ---------- */

export const EightRunsLloyd = lab({
  intro: 'The course’s running example: eight training runs, $K=2$. Step through and check the squared distances in the table yourself.',
  tex: R`J=\sum_i\|x_i-\mu_{g_i}\|^2`,
  params: [{ k: 'start', label: 'Starting centroids', options: [['poor', 'both among the small runs'], ['split', 'one in each group']], init: 'poor' }, { k: 'step', label: 'iteration', min: 0, max: 5, step: 1, init: 0, play: true }],
  derive: ({ start, step }) => { const frames = kmeans(RUNS, 2, { init: start === 'poor' ? [[2, 3], [1, 3]] : [[4, 4], [8, 8]] }), i = Math.min(step, frames.length - 1); return { frames, i, f: frames[i] }; },
  plots: v => [{ x: [0, 10], y: [0, 11], w: 380, h: 360, xl: 'training time (hours)', yl: 'peak memory (GB)', label: 'eight runs and two centroids', marks: [...RUNS.map((x, i) => ({ tone: 'muted', soft: true, data: [x, v.f.centres[v.f.assign[i]]] })), { as: 'points', r: 6, data: RUNS.map((x, i) => [x[0], x[1], v.f.assign[i] + 1]) }, { as: 'rings', r: 11, data: v.f.centres }, { as: 'points', tone: 'fg', r: 4, data: v.f.centres }] }],
  stats: v => [`$J=${fmt(v.f.inertia)}$`, `iteration ${v.i} of ${v.frames.length - 1}`, `centroids $(${fmt(v.f.centres[0][0], 3)},\\ ${fmt(v.f.centres[0][1], 3)})$ and $(${fmt(v.f.centres[1][0], 3)},\\ ${fmt(v.f.centres[1][1], 3)})$`],
  status: v => (v.i === v.frames.length - 1 ? ['ok', 'No run changed cluster, so the algorithm has stopped.'] : undefined),
  table: v => ({ head: ['run', 'to A', 'to B', 'joins'], rows: RUNS.map((x, i) => [i + 1, fmt(d2(x, v.f.centres[0]), 2), fmt(d2(x, v.f.centres[1]), 2), 'AB'[v.f.assign[i]]]) }),
  takeaway: 'From either start the partition ends the same here: small runs against large runs, with $J=15.25$.',
});

export const SquaredDistanceDrill = drill({
  intro: 'Distance adds one squared term per feature. Work with the squared value, as K-means does.',
  tex: R`\|x-y\|^2=\sum_j(x_j-y_j)^2`,
  gen: r => ({ a: [int(r, 0, 9), int(r, 0, 9)], b: [int(r, 0, 9), int(r, 0, 9)] }),
  q: v => `Find the squared distance between $(${v.a})$ and $(${v.b})$.`,
  answer: v => d2(v.a, v.b), dp: 0,
  steps: v => [`Differences: $${v.a[0] - v.b[0]}$ and $${v.a[1] - v.b[1]}$`, `Squares: $${(v.a[0] - v.b[0]) ** 2}$ and $${(v.a[1] - v.b[1]) ** 2}$`],
});

const costOf = (a, c, mode) => a.reduce((s, x) => s + (mode === 'abs' ? Math.abs(x - c) : (x - c) ** 2), 0);
export const MeanIsBest = lab({
  intro: 'Move the single representative $c$ and find the cost’s lowest point. Then add an outlier, and see which rule it moves.',
  tex: R`f(c)=\sum_j(x_j-c)^2`,
  params: [{ k: 'mode', label: 'Cost', options: [['sq', 'squared distance'], ['abs', 'absolute distance']], init: 'sq' }, { k: 'set', label: 'Values', options: [['a', '4, 4, 2, 1'], ['b', 'with an outlier at 20']], init: 'a' }, { k: 'c', label: 'representative $c$', min: 0, max: 22, step: 0.05, init: 2 }],
  derive: ({ mode, set, c }) => { const a = set === 'a' ? [4, 4, 2, 1] : [4, 4, 2, 1, 20], best = mode === 'sq' ? mean(a) : medianOf(a), top = Math.max(costOf(a, 0, mode), costOf(a, 22, mode)); return { a, best, top, f: costOf(a, c, mode) }; },
  plots: v => [{ x: [0, 22], y: [0, v.top], w: 400, h: 300, xl: 'c', yl: 'cost', label: 'cost against c', marks: [{ tone: 1, data: curve(c => costOf(v.a, c, v.mode), 0, 22, 221) }, { as: 'points', tone: 2, r: 6, data: [[v.c, v.f]] }, { as: 'rule', tone: 3, dash: true, x: v.best }, { as: 'points', tone: 'fg', data: v.a.map(x => [x, v.top * 0.02]) }] }],
  stats: v => [`$f(${fmt(v.c)})=${fmt(v.f)}$`, `lowest at $c=${fmt(v.best)}$, cost $${fmt(costOf(v.a, v.best, v.mode))}$`],
  takeaway: 'Squared distance is minimised by the mean, and absolute distance by the median. Only the mean follows the outlier far from the rest.',
});

export const BisectorBoundary = lab({
  intro: 'Drag either centroid. The set of points equally far from both is always a straight line, because the $\\|x\\|^2$ terms cancel.',
  tex: R`2(\mu_b-\mu_a)^\top x=\|\mu_b\|^2-\|\mu_a\|^2`,
  params: [{ k: 'ax', label: 'A, first coordinate', min: 0, max: 10, step: 0.25, init: 2.75 }, { k: 'ay', label: 'A, second coordinate', min: 0, max: 10, step: 0.25, init: 3.25 }, { k: 'bx', label: 'B, first coordinate', min: 0, max: 10, step: 0.25, init: 7.5 }, { k: 'by', label: 'B, second coordinate', min: 0, max: 10, step: 0.25, init: 7.75 }],
  plots: v => { const f = (x, y) => (x - v.ax) ** 2 + (y - v.ay) ** 2 - (x - v.bx) ** 2 - (y - v.by) ** 2; return [{ x: [0, 10], y: [0, 10], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'boundary between two centroids', marks: [{ as: 'field', bands: 1, fn: (x, y) => (f(x, y) < 0 ? 0.3 : -0.3) }, { as: 'contour', tone: 'fg', fn: f, levels: [0] }, { as: 'points', r: 5, data: RUNS.map(x => [x[0], x[1], f(x[0], x[1]) < 0 ? 1 : 2]) }, { drag: { x: 'ax', y: 'ay', tone: 1 } }, { drag: { x: 'bx', y: 'by', tone: 2 } }] }]; },
  stats: v => [`$${fmt(2 * (v.bx - v.ax))}\\,x_1+${fmt(2 * (v.by - v.ay))}\\,x_2=${fmt(v.bx ** 2 + v.by ** 2 - v.ax ** 2 - v.ay ** 2, 4)}$`],
  takeaway: 'K-means regions are bounded by straight lines, which is why it cannot wrap a cluster around another.',
});

const PARTS = { one: [0, 0, 0, 0, 0, 0, 0, 0], mixed: [0, 0, 1, 1, 0, 1, 1, 0], split: [0, 0, 0, 0, 1, 1, 1, 1] };
export const TotalSpreadSplit = lab({
  intro: 'For every partition the total spread divides exactly into within-cluster and between-cluster parts. Try each partition and compare the bars.',
  tex: R`T=W+B,\qquad B=\sum_kn_k\|\mu_k-\bar x\|^2`,
  params: [{ k: 'p', label: 'Partition', options: [['one', 'all together'], ['mixed', 'mixed'], ['split', 'small and large']], init: 'mixed' }],
  derive: ({ p }) => { const a = PARTS[p], cs = [0, 1].map(k => { const m = RUNS.filter((_, i) => a[i] === k); return m.length ? [mean(m.map(x => x[0])), mean(m.map(x => x[1]))] : null; }); return { a, cs, ...spreads(RUNS, a) }; },
  plots: v => [{ caption: 'Each run to its cluster mean', x: [0, 10], y: [0, 11], w: 300, h: 280, label: 'runs and cluster means', marks: [...RUNS.map((x, i) => ({ tone: v.a[i] + 1, data: [x, v.cs[v.a[i]]] })), { as: 'points', r: 5, data: RUNS.map((x, i) => [x[0], x[1], v.a[i] + 1]) }, { as: 'rings', r: 9, data: v.cs.filter(Boolean) }] }, { caption: 'The spread, divided', x: [0, 4], y: [0, 110], w: 300, h: 280, label: 'bars for T, W and B', marks: [{ as: 'rects', tone: 1, data: [[0.3, 0, 1.2, v.T]] }, { as: 'rects', tone: 2, data: [[1.5, 0, 2.4, v.W]] }, { as: 'rects', tone: 3, data: [[2.7, 0, 3.6, v.B]] }, { as: 'label', x: 0.75, y: v.T + 3, text: 'T', anchor: 'middle' }, { as: 'label', x: 1.95, y: v.W + 3, text: 'W', anchor: 'middle' }, { as: 'label', x: 3.15, y: v.B + 3, text: 'B', anchor: 'middle' }] }],
  stats: v => [`$${fmt(v.T, 3)}=${fmt(v.W, 3)}+${fmt(v.B, 3)}$`],
  takeaway: '$T$ is fixed by the data, so reducing $W$ and increasing $B$ are the same goal.',
});

const VALLEY = [[0], [1], [2], [4]];
const valleyJ = (a, b) => VALLEY.reduce((s, x) => s + Math.min((x[0] - a) ** 2, (x[0] - b) ** 2), 0);
export const LocalValley = lab({
  intro: 'Data $0,1,2,4$ with $K=2$. Choose where the two centroids start and watch which valley Lloyd’s algorithm falls into.',
  params: [{ k: 'a', label: 'centroid A starts at', min: -1, max: 5, step: 0.25, init: 0 }, { k: 'b', label: 'centroid B starts at', min: -1, max: 5, step: 0.25, init: 2.5 }],
  derive: ({ a, b }) => { const frames = kmeans(VALLEY, 2, { init: a === b ? [[a], [b + 0.01]] : [[a], [b]] }); return { frames, f: frames.at(-1), path: frames.map(q => [q.centres[0][0], q.centres[1][0]]) }; },
  plots: v => [{ caption: 'J over the two centroids (darker is lower)', x: [-1, 5], y: [-1, 5], w: 320, h: 300, xl: 'centroid A', yl: 'centroid B', label: 'objective landscape', marks: [{ as: 'field', bands: 8, fn: (a, b) => 1 - Math.log(1 + Math.min(valleyJ(a, b), 30)) / Math.log(31) }, { tone: 4, data: v.path }, { as: 'points', tone: 4, r: 4, data: v.path }, { drag: { x: 'a', y: 'b', tone: 2 } }] }, { caption: 'The final clusters', x: [-1, 5], y: [0, 1], w: 320, h: 300, xl: 'value', label: 'final assignment', marks: [{ as: 'points', r: 8, data: VALLEY.map((x, i) => [x[0], 0.5, v.f.assign[i] + 1]) }, { as: 'rings', r: 13, data: v.f.centres.map(c => [c[0], 0.5]) }] }],
  stats: v => [`final $J=${fmt(v.f.inertia, 3)}$`, `started at $J=${fmt(v.frames[0].inertia, 3)}$`],
  status: v => (Math.abs(v.f.inertia - 2) < 1e-6 ? ['ok', 'This is the better valley, $J=2$.'] : Math.abs(v.f.inertia - 2.5) < 1e-6 ? ['alert', 'Stopped at a local minimum, $J=2.5$. No single step can improve it.'] : undefined),
  takeaway: 'Lloyd’s algorithm always stops, but where depends on the start. Run it several times and keep the lowest $J$.',
});

export const PlusPlusDrill = drill({
  intro: 'After choosing some centroids, each point’s chance is proportional to its squared distance to the nearest chosen centroid.',
  tex: R`P(i)=\frac{D_i}{\sum_lD_l}`,
  gen: r => { const pts = [0, int(r, 1, 3), int(r, 4, 6), int(r, 8, 12)], j = int(r, 1, 3); return { pts, j, D: pts.map(x => x * x) }; },
  q: v => `Data $${v.pts.join(',\\ ')}$, and the only centroid so far is $0$. What is the probability that $${v.pts[v.j]}$ is chosen next?`,
  answer: v => v.D[v.j] / v.D.reduce((a, b) => a + b, 0), dp: 3,
  steps: v => [`Values of $D$: $${v.D.join(',\\ ')}$`, `Their sum is $${v.D.reduce((a, b) => a + b, 0)}$`],
});

const NEAR = blobs(9, [[0, 0]], 30, 0.7), FARG = blobs(10, [[10, 0]], 60, 1);
export const PlusPlusLab = lab({
  intro: 'One centroid is at the origin. A far group is still empty, and one lone point is farther still. Bigger discs are likelier to be chosen next by K-means++.',
  params: [{ k: 'g', label: 'points in the far group', min: 5, max: 60, step: 5, init: 30 }, { k: 'd', label: 'how far the lone point is', min: 15, max: 40, step: 1, init: 25 }],
  derive: ({ g, d }) => { const far = FARG.slice(0, g), lone = [d, 6], D = p => p[0] ** 2 + p[1] ** 2, tot = [...NEAR, ...far, lone].reduce((s, p) => s + D(p), 0), pg = far.reduce((s, p) => s + D(p), 0) / tot, pl = D(lone) / tot; return { far, lone, D, tot, pg, pl, pn: 1 - pg - pl }; },
  plots: v => { const r = p => Math.min(16, 1.5 + 24 * Math.sqrt(v.D(p) / v.tot)); return [{ x: [-4, 42], y: [-14, 14], w: 440, h: 300, label: 'chance of each point', marks: [{ as: 'points', data: v.far.map(p => [p[0], p[1], 2, r(p)]) }, { as: 'points', data: NEAR.map(p => [p[0], p[1], 1, r(p)]) }, { as: 'points', data: [[v.lone[0], v.lone[1], 4, r(v.lone)]] }, { as: 'rings', r: 11, data: [[0, 0]] }] }]; },
  stats: v => [`far group ${pct(v.pg, 1)}`, `lone point ${pct(v.pl, 1)}`, `near group ${pct(v.pn, 1)}`],
  status: v => (v.pl > v.pg ? ['alert', 'The lone point is now the likelier pick. "Take the farthest" would always pick it.'] : ['ok', 'The group outweighs the lone point, even though each of its members is nearer.']),
  takeaway: 'A random draw gives a lone outlier only its own share, while many moderately far points add up. Taking the farthest point ignores that.',
});

export const CalinskiDrill = drill({
  intro: 'Between-spread per degree of freedom over within-spread per degree of freedom. Pick the $K$ with the largest value.',
  tex: R`\text{CH}(K)=\frac{B/(K-1)}{W/(n-K)}`,
  gen: r => ({ B: int(r, 20, 90) * 5, W: int(r, 4, 30) * 2, n: int(r, 3, 9) * 10, K: int(r, 2, 6) }),
  q: v => `$B=${v.B}$, $W=${v.W}$, $n=${v.n}$, $K=${v.K}$. Find the score.`,
  answer: v => calinski(v.B, v.W, v.n, v.K), dp: 1,
  steps: v => [`$B/(K-1)=${v.B}/${v.K - 1}=${fmt(v.B / (v.K - 1))}$`, `$W/(n-K)=${v.W}/${v.n - v.K}=${fmt(v.W / (v.n - v.K), 3)}$`],
});

const scoresFor = memo(kind => {
  const X = kind === 'three' ? CLOUD : uniform(61);
  return [1, 2, 3, 4, 5, 6, 7, 8].map(k => {
    const f = bestOf(X, k), s = spreads(X, f.assign), refs = [1, 2, 3, 4, 5, 6].map(i => Math.log(bestOf(uniform(300 + i), k, 2).inertia));
    return { k, W: f.inertia, ch: k > 1 ? calinski(s.B, f.inertia, X.length, k) : 0, sil: k > 1 ? meanSilhouette(X, f.assign) : 0, gap: mean(refs) - Math.log(f.inertia) };
  });
});
const argTop = (rows, key) => rows.reduce((a, b) => (b[key] > a[key] ? b : a)).k;
export const KScores = lab({
  intro: 'Four ways of choosing $K$ on the same data. Compare three real groups with uniform noise, where no answer is correct.',
  params: [{ k: 'kind', label: 'Data', options: [['three', 'three real groups'], ['noise', 'uniform noise']], init: 'three' }, { k: 'K', label: 'clusters $K$', min: 1, max: 8, step: 1, init: 3 }],
  derive: ({ kind }) => { const rows = scoresFor(kind); return { rows }; },
  plots: v => { const mk = (key, t, y) => ({ caption: t, x: [1, 8], y, w: 220, h: 200, xl: 'K', label: t, marks: [{ tone: 1, data: v.rows.filter(r => key === 'W' || key === 'gap' || r.k > 1).map(r => [r.k, r[key]]) }, { as: 'rule', dash: true, tone: 4, x: v.K }] }); return [mk('W', 'Within spread', [0, v.rows[0].W * 1.05]), mk('ch', 'Calinski-Harabasz', [0, Math.max(...v.rows.map(r => r.ch)) * 1.1]), mk('sil', 'Silhouette', [0, 1]), mk('gap', 'Gap', [Math.min(...v.rows.map(r => r.gap)) - 0.1, Math.max(...v.rows.map(r => r.gap)) + 0.1])]; },
  stats: v => { const r = v.rows[v.K - 1]; return [`at $K=${v.K}$: gap ${fmt(r.gap, 3)}, silhouette ${fmt(r.sil, 3)}`, `largest Calinski-Harabasz at $K=${argTop(v.rows, 'ch')}$`, `largest silhouette at $K=${argTop(v.rows, 'sil')}$`, `largest gap at $K=${argTop(v.rows, 'gap')}$`]; },
  takeaway: 'Every score still names some $K$ on structureless data. Treat the choice as a hypothesis, and check it against a baseline and against what you need the clusters for.',
});

export const SilhouettePlot = lab({
  intro: 'One bar per point, sorted within each cluster. Bars near 1 are well placed, and bars below 0 are probably in the wrong cluster.',
  tex: R`s_i=\frac{b_i-a_i}{\max(a_i,b_i)}`,
  params: [{ k: 'K', label: 'clusters $K$', min: 2, max: 6, step: 1, init: 3 }],
  derive: ({ K }) => { const f = bestOf(CLOUD, K), s = silhouettes(CLOUD, f.assign); let y = 0; const groups = Array.from({ length: K }, (_, k) => { const vals = s.filter((_, i) => f.assign[i] === k).sort((a, b) => a - b), bars = vals.map(u => { const b = [0, y, u, y + 0.8]; y += 1; return b; }); y += 3; return { k, bars, m: vals.length ? mean(vals) : 0 }; }); return { f, s, groups, top: y, avg: mean(s) }; },
  plots: v => [{ caption: 'Silhouette of each point', x: [-0.4, 1], y: [0, v.top], w: 340, h: 340, xl: 'silhouette', label: 'silhouette bars', marks: [...v.groups.map(g => ({ as: 'rects', tone: tone(g.k), data: g.bars })), { as: 'rule', dash: true, tone: 'fg', x: v.avg }] }, { caption: 'The clusters', x: [-5, 5], y: [-5, 5], w: 300, h: 340, label: 'clusters', marks: [{ as: 'points', r: 4, data: CLOUD.map((x, i) => [x[0], x[1], tone(v.f.assign[i])]) }, { as: 'rings', r: 10, data: v.f.centres }] }],
  stats: v => [`mean silhouette ${fmt(v.avg, 3)}`, `weakest cluster averages ${fmt(Math.min(...v.groups.map(g => g.m)), 3)}`, `${v.s.filter(s => s < 0).length} points below zero`],
  takeaway: 'The mean hides which cluster is responsible. The plot shows it.',
});

/* ---------- mixtures ---------- */

const rFast = (x, m1, m2, s) => 1 / (1 + Math.exp(((x - m1) ** 2 - (x - m2) ** 2) / (2 * s * s)));
export const SoftAssignment = lab({
  intro: 'Responsibility is Bayes’ rule applied to one request. Shrink $\\sigma$ and watch the soft curve become K-means’ hard rule.',
  tex: R`r_{i1}=\frac{e^{-(x-\mu_1)^2/2\sigma^2}}{e^{-(x-\mu_1)^2/2\sigma^2}+e^{-(x-\mu_2)^2/2\sigma^2}}`,
  params: [{ k: 'm1', label: '$\\mu_1$', min: 5, max: 25, step: 0.5, init: 15 }, { k: 'm2', label: '$\\mu_2$', min: 20, max: 40, step: 0.5, init: 25 }, { k: 's', label: '$\\sigma$', min: 0.3, max: 10, step: 0.1, init: 4 }, { k: 'x', label: 'latency $x$', min: 0, max: 45, step: 0.5, init: 22 }],
  plots: v => { const mid = (v.m1 + v.m2) / 2; return [{ x: [0, 45], y: [0, 1], w: 420, h: 300, xl: 'latency (ms)', yl: 'responsibility of the fast component', label: 'responsibility curve', marks: [{ tone: 3, dash: true, data: [[0, 1], [mid, 1], [mid, 0], [45, 0]] }, { tone: 1, data: curve(x => rFast(x, v.m1, v.m2, v.s), 0, 45, 271) }, { as: 'points', tone: 2, r: 6, data: [[v.x, rFast(v.x, v.m1, v.m2, v.s)]] }, { as: 'points', tone: 'fg', data: LAT.map(x => [x, 0.02]) }] }]; },
  stats: v => [`$r=${fmt(rFast(v.x, v.m1, v.m2, v.s), 3)}$ for the fast component`, `$1-r=${fmt(1 - rFast(v.x, v.m1, v.m2, v.s), 3)}$ for the slow component`],
  legend: [['--c1', 'responsibility'], ['--c3', 'K-means rule']],
  takeaway: 'K-means is the limit of EM as $\\sigma\\to0$. A point near the midpoint is uncertain, and only the mixture says so.',
});

export const ResponsibilityDrill = drill({
  intro: 'With equal weights and equal $\\sigma$, only the two exponentials remain.',
  tex: R`r=\frac{e_1}{e_1+e_2},\qquad e_k=e^{-(x-\mu_k)^2/(2\sigma^2)}`,
  gen: (r, n) => (n === 0 ? { x: 22, a: 15, b: 25, s: 4 } : { x: int(r, 12, 28), a: int(r, 10, 16), b: int(r, 22, 30), s: pick(r, [3, 4, 5]) }),
  q: v => `$x=${v.x}$, $\\mu_1=${v.a}$, $\\mu_2=${v.b}$, $\\sigma=${v.s}$. Find the responsibility of component one.`,
  answer: v => rFast(v.x, v.a, v.b, v.s), dp: 3,
  steps: v => [`$2\\sigma^2=${2 * v.s * v.s}$`, `$e_1=e^{-${(v.x - v.a) ** 2}/${2 * v.s * v.s}}=${fmt(Math.exp(-((v.x - v.a) ** 2) / (2 * v.s * v.s)), 4)}$`, `$e_2=e^{-${(v.x - v.b) ** 2}/${2 * v.s * v.s}}=${fmt(Math.exp(-((v.x - v.b) ** 2) / (2 * v.s * v.s)), 4)}$`],
});

export const WeightedMeanDrill = drill({
  intro: 'The M-step: each point counts towards a component by its responsibility.',
  tex: R`\mu_k=\frac{\sum_ir_{ik}x_i}{\sum_ir_{ik}}`,
  gen: r => ({ xs: [int(r, 5, 12), int(r, 13, 22), int(r, 23, 35)], rs: [int(r, 6, 10) / 10, int(r, 2, 6) / 10, int(r, 0, 2) / 10] }),
  q: v => `Points $${v.xs.join(',\\ ')}$ have responsibilities $${v.rs.join(',\\ ')}$. Find the new mean.`,
  answer: v => v.xs.reduce((s, x, i) => s + x * v.rs[i], 0) / v.rs.reduce((a, b) => a + b, 0), dp: 3,
  steps: v => [`Numerator: $${v.xs.map((x, i) => `${v.rs[i]}\\times${x}`).join('+')}=${fmt(v.xs.reduce((s, x, i) => s + x * v.rs[i], 0))}$`, `Denominator: $${fmt(v.rs.reduce((a, b) => a + b, 0))}$`],
});

const emFrames = memo(s => em1d(LAT, [15, 25], s, 40));
export const EmLatencies = lab({
  intro: 'Seven request latencies from $15$ and $25$. Each step recomputes the responsibilities, then the weighted means. The log-likelihood never falls.',
  params: [{ k: 's', label: 'Spread', options: [[4, 'σ = 4'], [6, 'σ = 6'], [2.5, 'σ = 2.5']], init: 4 }, { k: 'step', label: 'EM step', min: 0, max: 40, step: 1, init: 0, play: true }],
  derive: ({ s, step }) => { const F = emFrames(s), f = F[step]; return { F, f, sig: +s }; },
  plots: v => { const pdf = (x, k) => 0.5 * normalPdf(x, v.f.mu[k], v.sig), lows = v.F.map(f => f.ll); return [{ caption: 'Density and requests', x: [0, 48], y: [0, 0.1], w: 340, h: 280, xl: 'latency (ms)', label: 'mixture density', marks: [{ tone: 1, dash: true, data: curve(x => pdf(x, 0), 0, 48) }, { tone: 2, dash: true, data: curve(x => pdf(x, 1), 0, 48) }, { tone: 'fg', data: curve(x => pdf(x, 0) + pdf(x, 1), 0, 48) }, { as: 'points', r: 6, data: LAT.map((x, i) => [x, 0.005, v.f.r[i][0] > 0.5 ? 1 : 2]) }] }, { caption: 'Log-likelihood', x: [0, 40], y: [Math.min(...lows) - 0.3, Math.max(...lows) + 0.3], w: 300, h: 280, xl: 'step', label: 'log-likelihood by step', marks: [{ tone: 1, data: lows.map((l, i) => [i, l]) }, { as: 'rule', dash: true, tone: 4, x: v.step }] }]; },
  stats: v => [`$\\mu_1=${fmt(v.f.mu[0])}$, $\\mu_2=${fmt(v.f.mu[1])}$`, `log-likelihood ${fmt(v.f.ll, 3)}`],
  status: () => ['ok', 'K-means from the same start stops at 12.0 and 30.5, with 22 ms wholly in the slow group.'],
  table: v => ({ head: ['latency', '$r_{i1}$', '$r_{i2}$'], rows: LAT.map((x, i) => [x, fmt(v.f.r[i][0], 3), fmt(v.f.r[i][1], 3)]) }),
  takeaway: 'Soft membership lets the 22 ms request share its weight, so neither mean is pulled as if it wholly belonged.',
});

const TILT = (() => { const r = rng(12), o = []; for (let i = 0; i < 70; i++) { const a = gauss(r), b = gauss(r); o.push(i % 2 ? [+(-2 + 0.95 * a + 0.2 * b).toFixed(3), +(1 + 0.75 * a + 0.3 * b).toFixed(3)] : [+(2 + 0.3 * a).toFixed(3), +(-1.2 + 0.9 * b).toFixed(3)]); } return o; })();
const ellipse = (m, S, c = 2) => { const e = eig2(S[0], S[1], S[2]), th = Math.abs(S[1]) < 1e-9 ? (S[0] >= S[2] ? 0 : Math.PI / 2) : Math.atan2(e.l1 - S[0], S[1]), u = [Math.cos(th), Math.sin(th)], w = [-u[1], u[0]]; return linspace(0, 2 * Math.PI, 81).map(t => { const p = c * Math.sqrt(e.l1) * Math.cos(t), q = c * Math.sqrt(Math.max(e.l2, 0)) * Math.sin(t); return [m[0] + p * u[0] + q * w[0], m[1] + p * u[1] + q * w[1]]; }); };
const em2Frames = memo(seed => em2d(TILT, [TILT[(seed * 7) % 70], TILT[(seed * 13 + 3) % 70]], 30));
export const EmEllipses = lab({
  intro: 'A full Gaussian mixture gives each component its own weight, centre and covariance, so a cluster can be a tilted ellipse. Dot size shows certainty.',
  params: [{ k: 'seed', seed: 'New start', init: 1 }, { k: 'step', label: 'EM step', min: 0, max: 30, step: 1, init: 0, play: true }],
  derive: ({ seed, step }) => { const F = em2Frames(seed); return { F, f: F[step] }; },
  plots: v => [{ caption: 'Two-standard-deviation ellipses', x: [-6, 6], y: [-5, 5], w: 340, h: 300, label: 'gaussian mixture ellipses', marks: [{ as: 'points', data: TILT.map((x, i) => [x[0], x[1], v.f.r[i][0] > 0.5 ? 1 : 2, 2 + 3.5 * Math.abs(2 * v.f.r[i][0] - 1)]) }, { tone: 1, data: ellipse(v.f.mu[0], v.f.S[0]) }, { tone: 2, data: ellipse(v.f.mu[1], v.f.S[1]) }] }, { caption: 'Log-likelihood per point', x: [0, 30], y: [+(Math.min(...v.F.map(f => f.ll / 70)) - 0.1).toFixed(1), +(Math.max(...v.F.map(f => f.ll / 70)) + 0.1).toFixed(1)], w: 300, h: 300, xl: 'step', label: 'log-likelihood by step', marks: [{ tone: 1, data: v.F.map((f, i) => [i, f.ll / 70]) }, { as: 'rule', dash: true, tone: 4, x: v.step }] }],
  stats: v => [`weights ${fmt(v.f.pi[0], 3)} and ${fmt(v.f.pi[1], 3)}`, `log-likelihood ${fmt(v.f.ll, 2)}`],
  takeaway: 'K-means would force round clusters here. Different starts can still end at different solutions.',
});

const llPoint = m => Math.log(0.5 * normalPdf(1, m, 1) + 0.5 * normalPdf(1, 4, 1));
const bound = (m, q) => q * Math.log((0.5 * normalPdf(1, m, 1)) / q) + (1 - q) * Math.log((0.5 * normalPdf(1, 4, 1)) / (1 - q));
export const JensenBound = lab({
  intro: 'One point at $x=1$. The orange curve is a lower bound on the log-likelihood for any weight $q$. Find the $q$ that makes it touch at the current $\\mu$.',
  tex: R`\log\sum_kp_k\ge\sum_kq_k\log\frac{p_k}{q_k}`,
  params: [{ k: 'm', label: 'current mean $\\mu$', min: -3, max: 6, step: 0.05, init: -1 }, { k: 'q', label: 'weight $q$', min: 0.01, max: 0.99, step: 0.01, init: 0.5 }],
  derive: ({ m, q }) => { const a = normalPdf(1, m, 1), b = normalPdf(1, 4, 1); return { r: a / (a + b), gap: llPoint(m) - bound(m, q) }; },
  plots: v => [{ x: [-3, 6], y: [-14, 0], w: 420, h: 300, xl: 'μ', yl: 'value', label: 'log-likelihood and lower bound', marks: [{ tone: 1, data: curve(llPoint, -3, 6) }, { tone: 2, data: curve(m => Math.max(-14, bound(m, v.q)), -3, 6) }, { as: 'rule', dash: true, tone: 'muted', x: v.m }, { as: 'points', tone: 1, r: 5, data: [[v.m, llPoint(v.m)]] }, { as: 'points', tone: 2, r: 5, data: [[v.m, Math.max(-14, bound(v.m, v.q))]] }] }],
  stats: v => [`gap ${fmt(v.gap, 4)}`, `the E-step would choose $q=${fmt(v.r, 3)}$`],
  status: v => (v.gap < 0.003 ? ['ok', 'Tight: the bound touches the log-likelihood here. This is what the E-step does.'] : undefined),
  legend: [['--c1', 'log-likelihood'], ['--c2', 'lower bound']],
  takeaway: 'The E-step makes the bound touch. The M-step climbs the bound. So the log-likelihood cannot go down.',
});

/* ---------- hierarchical ---------- */

export const WardDrill = drill({
  intro: 'Ward’s linkage is the exact rise in within-cluster sum of squares caused by the merge.',
  tex: R`\Delta=\frac{n_1n_2}{n_1+n_2}\|\mu_1-\mu_2\|^2`,
  gen: r => ({ n1: int(r, 1, 6), n2: int(r, 1, 6), dx: int(r, 1, 6), dy: int(r, 0, 6) }),
  q: v => `Clusters of sizes $${v.n1}$ and $${v.n2}$ have means that differ by $(${v.dx},${v.dy})$. Find the cost of merging them.`,
  answer: v => ((v.n1 * v.n2) / (v.n1 + v.n2)) * (v.dx ** 2 + v.dy ** 2), dp: 3,
  steps: v => [`$\\|\\Delta\\mu\\|^2=${v.dx ** 2 + v.dy ** 2}$`, `$n_1n_2/(n_1+n_2)=${v.n1 * v.n2}/${v.n1 + v.n2}$`],
});

const LW = { single: (a, b) => 0.5 * a + 0.5 * b - 0.5 * Math.abs(a - b), complete: (a, b) => 0.5 * a + 0.5 * b + 0.5 * Math.abs(a - b) };
export const LanceWilliamsDrill = drill({
  intro: 'After merging $i$ and $j$, the new distance to $k$ comes from the old table.',
  tex: R`\delta(i\cup j,k)=\alpha_i\delta_{ik}+\alpha_j\delta_{jk}+\beta\delta_{ij}+\gamma|\delta_{ik}-\delta_{jk}|`,
  gen: r => ({ a: int(r, 1, 12), b: int(r, 1, 12), link: pick(r, ['single', 'complete', 'average']), ni: int(r, 1, 5), nj: int(r, 1, 5) }),
  q: v => `$\\delta_{ik}=${v.a}$, $\\delta_{jk}=${v.b}$, $n_i=${v.ni}$, $n_j=${v.nj}$. Find the updated distance for **${v.link}** linkage.`,
  answer: v => (v.link === 'average' ? (v.ni * v.a + v.nj * v.b) / (v.ni + v.nj) : LW[v.link](v.a, v.b)), dp: 3,
  steps: v => (v.link === 'average' ? [`$\\alpha_i=${v.ni}/${v.ni + v.nj}$ and $\\alpha_j=${v.nj}/${v.ni + v.nj}$, with $\\beta=\\gamma=0$.`] : [`$\\alpha_i=\\alpha_j=\\tfrac12$ and $\\gamma=${v.link === 'single' ? '-' : ''}\\tfrac12$, which gives the ${v.link === 'single' ? 'minimum' : 'maximum'} of the two.`]),
});

const BRIDGE = (() => { const b = blobs(21, [[-3.2, 0], [3.2, 0]], 24, 0.6), r = rng(22), o = [...b]; for (let x = -2.2; x <= 2.21; x += 0.34) o.push([+x.toFixed(2), +(0.08 * gauss(r)).toFixed(3)]); return o; })();
const cutTop = (merges, n, k) => { const p = Array.from({ length: 2 * n }, (_, i) => i), f = x => (p[x] === x ? x : (p[x] = f(p[x]))); merges.slice(0, n - k).forEach(m => { p[f(m.a)] = m.id; p[f(m.b)] = m.id; }); const roots = [...new Set(Array.from({ length: n }, (_, i) => f(i)))]; return Array.from({ length: n }, (_, i) => roots.indexOf(f(i))); };
const bridgeCut = memo(link => { const m = agglomerate(BRIDGE, link); return { a: cutTop(m, BRIDGE.length, 2), heights: m.map(q => q.h) }; });
export const LinkageBridge = lab({
  intro: 'Two groups joined by a thin bridge, cut into two clusters. The definition of cluster distance decides what happens.',
  params: [{ k: 'link', label: 'Linkage', options: [['single', 'single'], ['complete', 'complete'], ['average', 'average'], ['ward', 'Ward']], init: 'single' }],
  derive: ({ link }) => { const { a, heights } = bridgeCut(link), sizes = [0, 1].map(k => a.filter(x => x === k).length).sort((p, q) => p - q); return { a, heights, sizes }; },
  plots: v => [{ caption: 'Two clusters', x: [-6, 6], y: [-3, 3], w: 360, h: 260, label: 'clusters after cutting', marks: [{ as: 'points', r: 4, data: BRIDGE.map((x, i) => [x[0], x[1], v.a[i] + 1]) }] }, { caption: 'Merge heights', x: [0, BRIDGE.length], y: [0, v.heights.at(-1) * 1.05], w: 260, h: 260, xl: 'merge', label: 'merge heights', marks: [{ tone: 1, data: v.heights.map((h, i) => [i + 1, h]) }] }],
  stats: v => [`cluster sizes ${v.sizes[0]} and ${v.sizes[1]}`, `last merge at height ${fmt(v.heights.at(-1))}`],
  status: v => (v.sizes[0] < 8 ? ['alert', 'Chaining: the bridge keeps the groups joined, and only a few stray points are split off.'] : ['ok', 'The bridge was cut, giving two substantial groups.']),
  takeaway: 'Single linkage follows chains, and that is a strength on curved shapes and a weakness when a few points bridge two groups.',
});

/* ---------- density ---------- */

const MOONS = moons(11, 160, 0.08).map(p => [p[0], p[1]]);
const MIXED = [...blobs(5, [[-0.8, 0.8]], 45, 0.07), ...blobs(6, [[1.3, -0.3]], 45, 0.33), ...Array.from({ length: 14 }, (_, i) => [+(-1.5 + 0.3 * i).toFixed(3), +(1.45 - (i % 5) * 0.55).toFixed(3)])];
const dbData = { moons: MOONS, mixed: MIXED };
export const DbscanLab = lab({
  intro: 'A core point has at least minPts points within $\\varepsilon$. Connected core points form a cluster, with border points attached and everything else noise.',
  params: [{ k: 'kind', label: 'Data', options: [['moons', 'half-moons'], ['mixed', 'unequal densities']], init: 'moons' }, { k: 'eps', label: '$\\varepsilon$', min: 0.03, max: 0.7, step: 0.01, init: 0.3 }, { k: 'minPts', label: 'minPts', min: 2, max: 12, step: 1, init: 5 }],
  derive: ({ kind, eps, minPts }) => { const X = dbData[kind], d = dbscan(X, eps, minPts); return { X, d, core: d.kinds.filter(k => k === 'core').length, border: d.kinds.filter(k => k === 'border').length, noise: d.kinds.filter(k => k === 'noise').length }; },
  plots: v => [{ x: [-1.7, 2.7], y: [-1.3, 1.7], w: 400, h: 320, label: 'DBSCAN clusters', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: v.X.filter((_, i) => v.d.labels[i] < 0) }, { as: 'points', data: v.X.flatMap((x, i) => (v.d.labels[i] < 0 ? [] : [[x[0], x[1], tone(v.d.labels[i]), v.d.core[i] ? 4.5 : 2.5]])) }] }],
  stats: v => [`${v.d.clusters} cluster${v.d.clusters === 1 ? '' : 's'}`, `${v.core} core, ${v.border} border, ${v.noise} noise`],
  note: 'Large dots are core points, small coloured dots are border points, and grey dots are noise.',
  takeaway: 'DBSCAN finds curved clusters and refuses to label noise, but one $\\varepsilon$ cannot suit very different densities. On the second dataset, try to recover both groups at once.',
});

export const CoreBorderDrill = drill({
  intro: 'Count the neighbourhood, including the point itself. Only core points pass the cluster along.',
  gen: r => { const pts = [...new Set(Array.from({ length: 9 }, () => int(r, 0, 24)))].sort((a, b) => a - b), eps = pick(r, [2, 3]), minPts = 3, t = int(r, 0, pts.length - 1), d = dbscan(pts.map(x => [x]), eps, minPts); return { pts, eps, minPts, t, kinds: d.kinds, kind: d.kinds[t] }; },
  q: v => `Points $${v.pts.join(',\\ ')}$, $\\varepsilon=${v.eps}$, minPts $=${v.minPts}$. What is $${v.pts[v.t]}$?`,
  choices: () => ['Core', 'Border', 'Noise'],
  answer: v => ['core', 'border', 'noise'].indexOf(v.kind),
  steps: v => { const N = v.pts.filter(x => Math.abs(x - v.pts[v.t]) <= v.eps); return [`Within $${v.eps}$ of $${v.pts[v.t]}$: $${N.join(',\\ ')}$, which is ${N.length} points.`, `Core points are: $${v.pts.filter((_, i) => v.kinds[i] === 'core').join(',\\ ') || '\\text{none}'}$`]; },
});

export const KDistancePlot = lab({
  intro: 'To choose $\\varepsilon$, sort each point’s distance to its (minPts − 1)-th nearest other point. The bend separates dense points from sparse ones.',
  params: [{ k: 'minPts', label: 'minPts', min: 3, max: 12, step: 1, init: 5 }, { k: 'eps', label: '$\\varepsilon$', min: 0.03, max: 0.6, step: 0.01, init: 0.3 }],
  derive: ({ minPts, eps }) => { const s = kDistance(MOONS, minPts - 1).sort((a, b) => b - a), d = dbscan(MOONS, eps, minPts); return { s, d, top: Math.max(0.35, s[0] * 1.05) }; },
  plots: v => [{ caption: 'Sorted distances', x: [0, 160], y: [0, v.top], w: 320, h: 280, xl: 'points, most isolated first', yl: 'distance', label: 'sorted k-distance', marks: [{ tone: 1, data: v.s.map((d, i) => [i, d]) }, { as: 'rule', tone: 4, y: Math.min(v.eps, v.top) }, { drag: { x: 150, y: 'eps', tone: 4 } }] }, { caption: 'Result at this ε', x: [-1.7, 2.7], y: [-1.3, 1.7], w: 320, h: 280, label: 'DBSCAN result', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: MOONS.filter((_, i) => v.d.labels[i] < 0) }, { as: 'points', r: 3.5, data: MOONS.flatMap((x, i) => (v.d.labels[i] < 0 ? [] : [[x[0], x[1], tone(v.d.labels[i])]])) }] }],
  stats: v => [`${v.d.clusters} cluster${v.d.clusters === 1 ? '' : 's'}`, `${v.d.labels.filter(l => l < 0).length} noise points`],
  takeaway: 'A value near the bend keeps most points in a cluster, without letting sparse points join separate groups together.',
});

export const MutualReachDrill = drill({
  intro: 'HDBSCAN inflates distances in sparse regions, so a sparse point cannot act as a bridge.',
  tex: R`d_{\text{mreach}}(a,b)=\max\{\text{core}_k(a),\ \text{core}_k(b),\ d(a,b)\}`,
  gen: r => ({ a: int(r, 1, 9) / 2, b: int(r, 1, 9) / 2, d: int(r, 1, 12) / 2 }),
  q: v => `$\\text{core}_k(a)=${v.a}$, $\\text{core}_k(b)=${v.b}$, $d(a,b)=${v.d}$. Find the mutual reachability distance.`,
  answer: v => Math.max(v.a, v.b, v.d), dp: 1,
  steps: () => ['Take the largest of the three numbers.'],
});

/* ---------- is it real? ---------- */

const randIndex = (a, b) => { let s = 0, c = 0; for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) { s += (a[i] === a[j]) === (b[i] === b[j]); c++; } return s / c; };
const stability = memo((kind, K, seed) => {
  const X = kind === 'three' ? CLOUD : uniform(77);
  let last; const agree = Array.from({ length: 12 }, (_, t) => { const r = rng(seed * 101 + t), o = X.map(x => [r(), x]).sort((a, b) => a[0] - b[0]).map(p => p[1]), A = o.slice(0, 27), B = o.slice(27), fa = bestOf(A, K, 3), own = bestOf(B, K, 3), via = B.map(x => { const ds = fa.centres.map(c => d2(x, c)); return ds.indexOf(Math.min(...ds)); }); last = { B, via, own: own.assign }; return randIndex(via, own.assign); });
  return { agree: mean(agree), last };
});
export const StabilitySplits = lab({
  intro: 'Cluster one half of the data, then ask whether the other half is grouped the same way. Agreement averages twelve random splits.',
  params: [{ k: 'kind', label: 'Data', options: [['three', 'three real groups'], ['noise', 'uniform noise']], init: 'three' }, { k: 'K', label: 'clusters $K$', min: 2, max: 6, step: 1, init: 3 }, { k: 'seed', seed: 'New splits', init: 1 }],
  derive: ({ kind, K, seed }) => stability(kind, K, seed),
  plots: v => { const f = (t, lab2) => ({ caption: t, x: [-5, 5], y: [-5, 5], w: 300, h: 280, label: t, marks: [{ as: 'points', r: 4, data: v.last.B.map((x, i) => [x[0], x[1], tone(lab2[i])]) }] }); return [f('Labels carried over from the other half', v.last.via), f('This half clustered alone', v.last.own)]; },
  stats: v => [`agreement ${fmt(v.agree, 3)}`],
  takeaway: 'High agreement alone proves little. Run the same check on uniform data, since a score is only meaningful against that baseline.',
});

export const MethodChoice = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A Gaussian mixture can say a point belongs 30% to one cluster and 70% to another.', true, 'Responsibilities are probabilities, and they are used as weights in the M-step.'],
    ['EM always reaches the global maximum of the likelihood.', false, 'It never decreases the likelihood, but it can stop at a local optimum. Several starts are normal practice.'],
    ['K-means is the $\\sigma\\to0$ limit of EM with equal weights and equal round covariances.', true, 'The nearer mean takes responsibility one, which is the hard assignment.'],
    ['A dendrogram requires choosing $K$ before running.', false, 'The whole tree is built first, and $K$ comes from where you cut it.'],
    ['DBSCAN needs the number of clusters in advance.', false, 'It needs $\\varepsilon$ and minPts. The number of clusters is an output.'],
    ['Single linkage can chain two groups through a thin bridge.', true, 'One short link is enough to join clusters, whatever the rest looks like.'],
    ['DBSCAN handles clusters of very different densities well with a single $\\varepsilon$.', false, 'One radius cannot suit both. HDBSCAN was designed for this.'],
    ['A silhouette of 0.41 proves three real clusters exist.', false, 'K-means on uniform data can reach about that value. Compare with a baseline.'],
    ['With labels unavailable, usefulness for the next task is a legitimate way to judge a clustering.', true, 'There is no single correct answer, so the purpose is part of the criterion.'],
  ],
  takeaway: 'Every clustering method answers a different question about what "together" means.',
});
