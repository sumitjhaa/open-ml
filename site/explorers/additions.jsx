'use client';
// Smaller topics from ES 335 (2025) that fill gaps in the earlier lectures.
import { R, curve, drill, int, lab, pick } from '../components/labs';
import { gauss, linfit, mean, rng, solve } from '../lib/math.mjs';
import { fmt, memo } from '../lib/models.mjs';

const AX = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], AX4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8];
const QUARTET = { i: ['first', AX, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]], ii: ['second', AX, [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74]], iii: ['third', AX, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]], iv: ['fourth', AX4, [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89]] };
const corr = (x, y) => { const s = (u, v) => u.reduce((t, p, i) => t + (p - mean(u)) * (v[i] - mean(v)), 0); return s(x, y) / Math.sqrt(s(x, x) * s(y, y)); };
export const AnscombeLab = lab({
  intro: 'Four datasets with almost identical statistics. Look at the picture before you believe a number.',
  params: [{ k: 'd', label: 'Dataset', options: Object.keys(QUARTET).map(k => [k, QUARTET[k][0]]), init: 'i' }],
  derive: ({ d }) => { const [, x, y] = QUARTET[d], f = linfit(x, y); return { x, y, f, r: corr(x, y) }; },
  plots: v => [{ x: [2, 20], y: [2, 14], w: 420, h: 300, xl: 'x', yl: 'y', label: 'dataset and best-fit line', marks: [{ tone: 2, data: [[2, v.f.b + 2 * v.f.w], [20, v.f.b + 20 * v.f.w]] }, { as: 'points', tone: 1, r: 5, data: v.x.map((a, i) => [a, v.y[i]]) }] }],
  stats: v => [`line $y=${fmt(v.f.b, 2)}+${fmt(v.f.w, 3)}x$`, `correlation ${fmt(v.r, 3)}`, `means ${fmt(mean(v.x), 2)} and ${fmt(mean(v.y), 2)}`],
  takeaway: 'Identical summaries can hide a curve, an outlier, or one point that decides everything.',
});
export const DummyColumnsDrill = drill({
  intro: 'With an intercept in the model, one category must be dropped.',
  gen: r => ({ k: int(r, 2, 9), n: int(r, 1, 4) }),
  q: v => `A model has an intercept and $${v.n}$ categorical feature${v.n > 1 ? 's' : ''}, each with $${v.k}$ categories. How many dummy columns does it use?`,
  answer: v => v.n * (v.k - 1), dp: 0,
  steps: v => [`Each feature uses $${v.k}-1$ columns.`],
});
const collinear = memo((rho, seed) => { const r = rng(seed * 41), runs = []; for (let t = 0; t < 40; t++) { const X = [], y = []; for (let i = 0; i < 30; i++) { const a = gauss(r), b = rho * a + Math.sqrt(1 - rho * rho) * gauss(r); X.push([a, b]); y.push(a + b + 0.8 * gauss(r)); } const G = [[0, 0], [0, 0]], g = [0, 0]; X.forEach((x, i) => { G[0][0] += x[0] * x[0]; G[0][1] += x[0] * x[1]; G[1][1] += x[1] * x[1]; g[0] += x[0] * y[i]; g[1] += x[1] * y[i]; }); G[1][0] = G[0][1]; G[0][0] += 1e-9; G[1][1] += 1e-9; runs.push(solve(G, g)); } return runs; });
export const CollinearityLab = lab({
  intro: 'Forty resampled fits of $y=x_1+x_2+$ noise. As the features become alike, each coefficient swings wildly, while their sum stays put.',
  params: [{ k: 'rho', label: 'correlation of the features', min: 0, max: 0.995, step: 0.005, init: 0.5 }, { k: 'seed', seed: 'New samples', init: 1 }],
  derive: ({ rho, seed }) => { const R0 = collinear(rho, seed), sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); }; return { R0, sdA: sd(R0.map(p => p[0])), sdS: sd(R0.map(p => p[0] + p[1])) }; },
  plots: v => [{ x: [-4, 6], y: [-4, 6], w: 360, h: 340, xl: 'θ₁', yl: 'θ₂', label: 'fitted coefficients over resamples', marks: [{ as: 'points', tone: 1, r: 4, data: v.R0 }, { as: 'points', tone: 4, r: 6, data: [[1, 1]] }] }],
  stats: v => [`spread of $\\theta_1$: ${fmt(v.sdA, 3)}`, `spread of $\\theta_1+\\theta_2$: ${fmt(v.sdS, 3)}`],
  takeaway: 'Predictions can stay good while single coefficients become meaningless.',
});
const ERR = [1, -2, 1.5, -1, 2, -1.5, 0.5];
export const OutlierMetrics = lab({
  intro: 'Seven ordinary errors and one that you control. See how differently the two metrics react.',
  params: [{ k: 'o', label: 'size of the largest error', min: 0, max: 40, step: 0.5, init: 2 }],
  derive: ({ o }) => { const e = [...ERR, o]; return { mae: mean(e.map(Math.abs)), rmse: Math.sqrt(mean(e.map(v => v * v))) }; },
  plots: v => [{ x: [0, 40], y: [0, 16], w: 420, h: 280, xl: 'largest error', yl: 'metric', label: 'MAE and RMSE against the largest error', marks: [{ tone: 1, data: curve(o => mean([...ERR, o].map(Math.abs)), 0, 40) }, { tone: 4, data: curve(o => Math.sqrt(mean([...ERR, o].map(a => a * a))), 0, 40) }, { as: 'rule', dash: true, tone: 'muted', x: v.o }] }],
  stats: v => [`MAE ${fmt(v.mae, 2)}`, `RMSE ${fmt(v.rmse, 2)}`],
  legend: [['--c1', 'MAE'], ['--c4', 'RMSE']],
  takeaway: 'Choose RMSE when large errors are especially costly, and MAE when outliers should not dominate.',
});
export const BalancedMccDrill = drill({
  intro: 'Both metrics refuse to reward a classifier that ignores the rare class.',
  tex: R`\text{BA}=\tfrac12\Big(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\Big)`,
  gen: r => ({ tp: int(r, 2, 20), fn: int(r, 1, 10), tn: int(r, 20, 90), fp: int(r, 1, 15), k: pick(r, ['BA', 'MCC']) }),
  q: v => `$TP=${v.tp}$, $FN=${v.fn}$, $TN=${v.tn}$, $FP=${v.fp}$. Find the ${v.k === 'BA' ? 'balanced accuracy' : 'Matthews correlation'}.`,
  answer: v => (v.k === 'BA' ? 0.5 * (v.tp / (v.tp + v.fn) + v.tn / (v.tn + v.fp)) : (v.tp * v.tn - v.fp * v.fn) / Math.sqrt((v.tp + v.fp) * (v.tp + v.fn) * (v.tn + v.fp) * (v.tn + v.fn))), dp: 3,
  steps: v => (v.k === 'BA' ? [`Recall of positives: $${fmt(v.tp / (v.tp + v.fn), 3)}$`, `Recall of negatives: $${fmt(v.tn / (v.tn + v.fp), 3)}$`] : [`Numerator: $${v.tp * v.tn - v.fp * v.fn}$`]),
});
const score = x => Math.exp(-((x - 0.62) ** 2) / 0.012);
export const SearchStrategies = lab({
  intro: 'Only the horizontal hyperparameter matters here. Compare how many of its values each strategy actually tries.',
  params: [{ k: 'm', label: 'Strategy', options: [['grid', 'grid search'], ['random', 'random search']], init: 'grid' }, { k: 'n', label: 'values per axis (grid)', min: 2, max: 6, step: 1, init: 3 }, { k: 'seed', seed: 'New draw', init: 1 }],
  derive: ({ m, n, seed }) => { const r = rng(seed * 11), N = n * n, pts = m === 'grid' ? Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => [(i + 0.5) / n, (j + 0.5) / n])).flat() : Array.from({ length: N }, () => [+r().toFixed(4), +r().toFixed(4)]); return { pts, N, best: Math.max(...pts.map(p => score(p[0]))), distinct: new Set(pts.map(p => p[0])).size }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 360, h: 340, xl: 'important', yl: 'unimportant', label: 'search trials', marks: [{ as: 'field', bands: 5, tones: [3, 1], fn: x => score(x) }, { as: 'points', tone: 4, r: 5, data: v.pts }] }],
  stats: v => [`best score found ${fmt(v.best, 3)}`, `${v.distinct} distinct values tried, with ${v.N} trials`],
  takeaway: 'With the same budget, random search explores more values of the hyperparameters that matter.',
});
export const GridCountDrill = drill({
  intro: 'Every combination is one fit, and cross-validation repeats each.',
  gen: r => ({ a: int(r, 2, 5), b: int(r, 2, 5), c: int(r, 2, 5), k: pick(r, [3, 5, 10]) }),
  q: v => `A grid has $${v.a}$, $${v.b}$ and $${v.c}$ values for three hyperparameters, with ${v.k}-fold cross-validation. How many model fits is that?`,
  answer: v => v.a * v.b * v.c * v.k, dp: 0,
  steps: v => [`Combinations: $${v.a * v.b * v.c}$`],
});
const SURF = { bowl: ['round bowl', (x, y) => x * x + y * y, (x, y) => [2 * x, 2 * y]], valley: ['narrow valley', (x, y) => x * x + 6 * y * y, (x, y) => [2 * x, 12 * y]], tilted: ['tilted', (x, y) => x * x + y * y + 1.4 * x * y, (x, y) => [2 * x + 1.4 * y, 2 * y + 1.4 * x]] };
export const ContourGradient = lab({
  intro: 'Each curve joins points of equal height. Drag the point: the gradient is perpendicular to the curve through it.',
  params: [{ k: 's', label: 'Surface', options: Object.keys(SURF).map(k => [k, SURF[k][0]]), init: 'valley' }, { k: 'x', label: '$x$', min: -3, max: 3, step: 0.05, init: 1.5 }, { k: 'y', label: '$y$', min: -3, max: 3, step: 0.05, init: 1 }],
  derive: ({ s, x, y }) => ({ f: SURF[s][1], g: SURF[s][2](x, y) }),
  plots: v => { const n = Math.hypot(...v.g) || 1, L = 1.1; return [{ x: [-3, 3], y: [-3, 3], w: 360, h: 340, xl: 'x', yl: 'y', label: 'contours and gradient', marks: [{ as: 'contour', tone: 1, fn: v.f, levels: [0.5, 1.5, 3, 5, 8, 12, 18, 26, 36] }, { as: 'arrow', tone: 4, data: [[v.x, v.y, v.x + (L * v.g[0]) / n, v.y + (L * v.g[1]) / n]] }, { drag: { x: 'x', y: 'y', tone: 2 } }] }]; },
  stats: v => [`$f=${fmt(v.f(v.x, v.y), 2)}$`, `$\\nabla f=(${fmt(v.g[0], 2)},\\ ${fmt(v.g[1], 2)})$`],
  takeaway: 'Gradient descent steps opposite the arrow, straight across the contours.',
});
export const WeightedGiniDrill = drill({
  intro: 'With sample weights, a class’s fraction is its share of the weight, not of the count.',
  tex: R`p_c=\frac{\sum_{i:y_i=c}w_i}{\sum_iw_i}`,
  gen: r => ({ w: Array.from({ length: 4 }, () => int(r, 1, 6)), y: Array.from({ length: 4 }, () => int(r, 0, 1)) }),
  q: v => `Labels $${v.y.join(',\\ ')}$ with weights $${v.w.join(',\\ ')}$. Find the weighted Gini impurity.`,
  answer: v => { const t = v.w.reduce((a, b) => a + b, 0), p = v.w.reduce((s, w, i) => s + (v.y[i] ? w : 0), 0) / t; return 2 * p * (1 - p); }, dp: 3,
  steps: v => { const t = v.w.reduce((a, b) => a + b, 0); return [`Total weight: $${t}$`, `Weight on class 1: $${v.w.reduce((s, w, i) => s + (v.y[i] ? w : 0), 0)}$`]; },
});
export const TreeSplitsDrill = drill({
  intro: 'With $n$ points sorted on one real feature, there are at most $n-1$ places to cut between them.',
  gen: r => ({ n: pick(r, [10, 50, 100, 1000]), d: int(r, 2, 20) }),
  q: v => `$n=${v.n}$ points with $${v.d}$ real features, all values distinct. How many candidate splits are there at the root?`,
  answer: v => (v.n - 1) * v.d, dp: 0,
  steps: v => [`Per feature: $${v.n - 1}$`],
});
export const SubsetCountDrill = drill({
  intro: 'Every feature is either in or out.',
  gen: r => ({ d: int(r, 3, 12) }),
  q: v => `How many non-empty subsets of $${v.d}$ features are there?`,
  answer: v => 2 ** v.d - 1, dp: 0,
  steps: v => [`$2^{${v.d}}-1$`],
});
