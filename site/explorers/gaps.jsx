'use client';
// Explorers filling note sections that had none, across the existing course sections.
import { R, curve, drill, int, lab, mat, pick } from '../components/labs';
import { gauss, linfit, mean, rng, sum } from '../lib/math.mjs';
import { accuracy, bagging, buildTree, clamp, dataset, fmt, linspace, logisticStep, pct, predictTree, pruneTree, ridge, sigmoid } from '../lib/models.mjs';

const ints = (r, n, a, b) => Array.from({ length: n }, () => int(r, a, b));

/* ---------- foundations ---------- */

export const ObjectTypes = drill({
  intro: 'Before computing anything, name the object. That is the first habit of this course.',
  gen: r => pick(r, [[R`\hat y_i`, 0], [R`x_i`, 1], [R`X`, 2], [R`\theta`, 1], [R`y-X\theta`, 1], [R`X^\top X`, 2], [R`\|y-X\theta\|^2`, 0], [R`\hat R(\theta)`, 0], [R`\nabla L`, 1], [R`x_i^\top\theta`, 0]]),
  q: v => `For a dataset with several features, what kind of object is $${v[0]}$?`,
  choices: () => ['Scalar', 'Vector', 'Matrix'],
  answer: v => v[1],
  steps: () => ['Write the shape of each symbol and multiply through.'],
});

export const IndependenceGrid = lab({
  intro: 'Independent events multiply, because knowing one tells you nothing about the other. Push the coupling away from zero and the product stops matching.',
  tex: R`\text{independent}\iff P(A,B)=P(A)\,P(B)`,
  params: [{ k: 'a', label: '$P(A)$', min: 0.1, max: 0.9, step: 0.01, init: 0.4 }, { k: 'b', label: '$P(B)$', min: 0.1, max: 0.9, step: 0.01, init: 0.5 }, { k: 'c', label: 'coupling', min: -1, max: 1, step: 0.01, init: 0 }],
  derive: ({ a, b, c }) => { const hi = Math.min(a, b) - a * b, lo = Math.max(0, a + b - 1) - a * b, ab = a * b + (c > 0 ? c * hi : -c * lo); return { ab }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 340, h: 320, xl: 'share in A', yl: 'share in B', label: 'independence as area', marks: [{ as: 'rects', tone: 1, data: [[0, 0, v.a, v.b]] }, { as: 'rule', dash: true, x: v.a }, { as: 'rule', dash: true, y: v.b }] }],
  stats: v => [`$P(A,B)$ = ${fmt(v.ab, 3)}`, `$P(A)P(B)$ = ${fmt(v.a * v.b, 3)}`, `$P(A\\mid B)$ = ${fmt(v.ab / v.b, 3)}`],
  status: v => (Math.abs(v.ab - v.a * v.b) < 0.002 ? ['ok', 'Independent: the likelihood of independent data is a product.'] : ['alert', 'Dependent: multiplying would be wrong.']),
});

export const ProbabilityChainDrill = drill({
  intro: 'Any joint probability factors one variable at a time.',
  tex: R`P(A,B,C)=P(A)\,P(B\mid A)\,P(C\mid A,B)`,
  gen: r => ({ a: pick(r, [0.2, 0.5, 0.8]), b: pick(r, [0.25, 0.5, 0.75]), c: pick(r, [0.1, 0.4, 0.9]) }),
  q: v => `$P(A)=${v.a}$, $P(B\\mid A)=${v.b}$ and $P(C\\mid A,B)=${v.c}$. Find $P(A,B,C)$.`,
  answer: v => v.a * v.b * v.c, dp: 3,
  steps: v => [`$${v.a}\\times${v.b}\\times${v.c}$`],
});

/* ---------- linear regression ---------- */

export const SpanRank = lab({
  intro: 'A model can only reach the combinations of its columns. Drag the second column onto the line of the first and the reachable set collapses.',
  tex: R`\hat y\in\operatorname{col}(X)=\{X\theta:\theta\in\mathbb R^d\}`,
  params: [{ k: 'a', label: '$c_{1}$ first entry', min: -3, max: 3, step: 0.05, init: 2 }, { k: 'b', label: '$c_{1}$ second entry', min: -3, max: 3, step: 0.05, init: 1 }, { k: 'c', label: '$c_{2}$ first entry', min: -3, max: 3, step: 0.05, init: 0.5 }, { k: 'd', label: '$c_{2}$ second entry', min: -3, max: 3, step: 0.05, init: 1.5 }],
  plots: v => { const det = v.a * v.d - v.b * v.c, grid = Math.abs(det) > 0.05 ? [-3, -2, -1, 0, 1, 2, 3].flatMap(t => [[[t * v.a - 4 * v.c, t * v.b - 4 * v.d], [t * v.a + 4 * v.c, t * v.b + 4 * v.d]], [[-4 * v.a + t * v.c, -4 * v.b + t * v.d], [4 * v.a + t * v.c, 4 * v.b + t * v.d]]]) : [[[-8 * v.a, -8 * v.b], [8 * v.a, 8 * v.b]]]; return [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, label: 'span of two columns', marks: [...grid.map(d => ({ tone: 'muted', soft: true, data: d })), { as: 'arrow', tone: 1, data: [[0, 0, v.a, v.b]] }, { as: 'arrow', tone: 2, data: [[0, 0, v.c, v.d]] }, { drag: { x: 'a', y: 'b', tone: 1 } }, { drag: { x: 'c', y: 'd', tone: 2 } }] }]; },
  stats: v => [`determinant ${fmt(v.a * v.d - v.b * v.c)}`],
  status: v => (Math.abs(v.a * v.d - v.b * v.c) < 0.05 ? ['alert', 'Rank one: the columns are parallel, so many coefficient vectors give the same fit.'] : ['ok', 'Rank two: every target in the plane is reachable.']),
});

export const ProjectionMatrixDrill = drill({
  intro: 'Projecting on to a line uses one matrix. Projecting twice changes nothing, so $P^2=P$.',
  tex: R`P=\frac{aa^\top}{a^\top a}`,
  gen: r => ({ a: pick(r, [[1, 1], [1, 2], [3, 4], [2, 1], [1, 3]]), i: int(r, 0, 1), j: int(r, 0, 1) }),
  q: v => `For $a=(${v.a.join(',')})$, find entry row ${v.i ? 'two' : 'one'}, column ${v.j ? 'two' : 'one'} of $P$.`,
  answer: v => (v.a[v.i] * v.a[v.j]) / (v.a[0] ** 2 + v.a[1] ** 2), dp: 3,
  steps: v => [`$a^\\top a=${v.a[0] ** 2 + v.a[1] ** 2}$`, `Numerator: $${v.a[v.i]}\\times${v.a[v.j]}$`],
});

export const MatrixGradientDrill = drill({
  intro: 'For a symmetric $A$, the gradient of the quadratic form is twice $A\\theta$.',
  tex: R`\nabla_\theta\big(\theta^\top A\theta\big)=2A\theta`,
  gen: r => ({ a: int(r, 1, 4), b: int(r, -2, 3), d: int(r, 1, 4), t: ints(r, 2, -2, 3), k: int(r, 0, 1) }),
  q: v => `For $A=${mat([[v.a, v.b], [v.b, v.d]])}$ and $\\theta=(${v.t.join(',')})$, find entry ${v.k ? 'two' : 'one'} of the gradient.`,
  answer: v => 2 * (v.k ? v.b * v.t[0] + v.d * v.t[1] : v.a * v.t[0] + v.b * v.t[1]), dp: 0,
  steps: v => [`$A\\theta=(${v.a * v.t[0] + v.b * v.t[1]},\\ ${v.b * v.t[0] + v.d * v.t[1]})$`, 'Double it.'],
});

export const CovarianceDrill = drill({
  intro: 'Centre each feature first. Covariance is the average product of the deviations.',
  tex: R`\operatorname{cov}(x,y)=\frac1n\sum_i(x_i-\bar x)(y_i-\bar y)`,
  gen: r => ({ x: ints(r, 4, 0, 8), y: ints(r, 4, 0, 8) }),
  q: v => `For $x=(${v.x.join(',')})$ and $y=(${v.y.join(',')})$, find the covariance.`,
  answer: v => { const a = mean(v.x), b = mean(v.y); return mean(v.x.map((p, i) => (p - a) * (v.y[i] - b))); },
  steps: v => [`Means: $${fmt(mean(v.x))}$ and $${fmt(mean(v.y))}$`, `Products of deviations: $${v.x.map((p, i) => fmt((p - mean(v.x)) * (v.y[i] - mean(v.y)))).join(',\\ ')}$`],
});

export const ChangeOfBasis = lab({
  intro: 'Rotate the axes. The arrow does not move, its coordinates change, and its length stays the same. That is what an orthonormal basis gives you.',
  tex: R`a_j=q_j^\top x,\qquad \|a\|=\|x\|`,
  params: [{ k: 'th', label: 'axis angle', min: -90, max: 90, step: 1, init: 30 }, { k: 'x', label: '$x_1$', min: -4, max: 4, step: 0.05, init: 3 }, { k: 'y', label: '$x_2$', min: -4, max: 4, step: 0.05, init: 1 }],
  derive: ({ th, x, y }) => { const t = (th * Math.PI) / 180, q1 = [Math.cos(t), Math.sin(t)], q2 = [-Math.sin(t), Math.cos(t)]; return { q1, q2, a1: q1[0] * x + q1[1] * y, a2: q2[0] * x + q2[1] * y }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, label: 'a vector in a rotated basis', marks: [{ tone: 1, dash: true, data: [[-6 * v.q1[0], -6 * v.q1[1]], [6 * v.q1[0], 6 * v.q1[1]]] }, { tone: 2, dash: true, data: [[-6 * v.q2[0], -6 * v.q2[1]], [6 * v.q2[0], 6 * v.q2[1]]] }, { as: 'arrow', tone: 1, data: [[0, 0, v.a1 * v.q1[0], v.a1 * v.q1[1]]] }, { as: 'arrow', tone: 2, data: [[v.a1 * v.q1[0], v.a1 * v.q1[1], v.x, v.y]] }, { as: 'arrow', tone: 'fg', data: [[0, 0, v.x, v.y]] }, { drag: { x: 'x', y: 'y' } }] }],
  stats: v => [`coordinates (${fmt(v.a1)}, ${fmt(v.a2)})`, `length ${fmt(Math.hypot(v.x, v.y), 3)} = ${fmt(Math.hypot(v.a1, v.a2), 3)}`],
});

export const LagrangeCircle = lab({
  intro: 'Maximise a linear function on the unit circle. At the best point, the objective’s arrow and the constraint’s arrow line up exactly.',
  tex: R`\nabla f=\lambda\nabla g`,
  params: [{ k: 'th', label: 'position on the circle', min: -180, max: 180, step: 1, init: -40 }, { k: 'cx', label: '$c_1$', min: -2, max: 2, step: 0.05, init: 1.2 }, { k: 'cy', label: '$c_2$', min: -2, max: 2, step: 0.05, init: 1 }],
  derive: ({ th, cx, cy }) => { const t = (th * Math.PI) / 180, p = [Math.cos(t), Math.sin(t)], n = Math.hypot(cx, cy) || 1; return { p, val: cx * p[0] + cy * p[1], best: n, cross: Math.abs(p[0] * cy - p[1] * cx) / n, dotv: (p[0] * cx + p[1] * cy) / n }; },
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, label: 'constraint circle and objective', marks: [{ as: 'contour', tone: 'muted', levels: [-2, -1, 0, 1, 2].map(k => k * 0.8), fn: (a, b) => v.cx * a + v.cy * b }, { tone: 1, data: linspace(0, 6.3, 91).map(t => [Math.cos(t), Math.sin(t)]) }, { as: 'arrow', tone: 2, data: [[...v.p, v.p[0] + v.cx * 0.7, v.p[1] + v.cy * 0.7]] }, { as: 'arrow', tone: 3, data: [[...v.p, v.p[0] * 2, v.p[1] * 2]] }, { as: 'points', tone: 4, r: 6, data: [v.p] }] }],
  stats: v => [`objective ${fmt(v.val, 3)}`, `best possible ${fmt(v.best, 3)}`],
  status: v => (v.cross < 0.02 && v.dotv > 0 ? ['ok', 'Arrows aligned: this is the constrained maximum.'] : undefined),
  legend: [['--c2', 'gradient of the objective'], ['--c3', 'normal of the constraint']],
  takeaway: 'This condition is the whole derivation of PCA: maximise variance on the unit sphere and an eigenvector equation falls out.',
});

/* ---------- logistic and evaluation ---------- */

export const OuterProductDrill = drill({
  intro: 'Each example adds a weighted outer product to the Hessian.',
  tex: R`H=\tfrac1n\sum_i p_i(1-p_i)\,x_ix_i^\top`,
  gen: r => ({ p: pick(r, [0.1, 0.2, 0.5, 0.8]), x: ints(r, 2, -2, 3), i: int(r, 0, 1), j: int(r, 0, 1) }),
  q: v => `For one example with $p=${v.p}$ and $x=(${v.x.join(',')})$, find entry row ${v.i ? 'two' : 'one'}, column ${v.j ? 'two' : 'one'} of $p(1-p)\\,xx^\\top$.`,
  answer: v => v.p * (1 - v.p) * v.x[v.i] * v.x[v.j], dp: 3,
  steps: v => [`$p(1-p)=${fmt(v.p * (1 - v.p))}$`, `$x_ix_j=${v.x[v.i] * v.x[v.j]}$`],
});

export const PopulationCostDrill = drill({
  intro: 'Turn rates into money for a whole population.',
  tex: R`\text{total}=N\big[\pi\,\text{FNR}\,C_{FN}+(1-\pi)\,\text{FPR}\,C_{FP}\big]`,
  gen: r => ({ N: pick(r, [100, 1000]), pi: pick(r, [0.1, 0.2, 0.5]), fnr: pick(r, [0.1, 0.2, 0.3]), fpr: pick(r, [0.05, 0.1, 0.2]), cn: int(r, 2, 10), cf: int(r, 1, 3) }),
  q: v => `$N=${v.N}$, prevalence $${v.pi}$, FNR $${v.fnr}$, FPR $${v.fpr}$, $C_{FN}=${v.cn}$ and $C_{FP}=${v.cf}$. Find the total cost.`,
  answer: v => v.N * (v.pi * v.fnr * v.cn + (1 - v.pi) * v.fpr * v.cf), dp: 1,
  steps: v => [`Missed cases: $${fmt(v.N * v.pi * v.fnr, 1)}$`, `False alarms: $${fmt(v.N * (1 - v.pi) * v.fpr, 1)}$`, 'Multiply each by its price and add.'],
});

const CAL = (() => { const r = rng(61), make = n => Array.from({ length: n }, () => { const p = clamp(0.02 + 0.96 * r(), 0.001, 0.999), z = Math.log(p / (1 - p)); return { m: +(3 * z + 0.6).toFixed(4), y: +(r() < p) }; }); return { fit: make(1500), test: make(2500) }; })();
const FITTED = (() => {
  const nll = (g, D) => mean(D.map(d => { const p = clamp(g(d.m), 1e-9, 1 - 1e-9); return d.y ? -Math.log(p) : -Math.log(1 - p); }));
  const T = linspace(0.2, 8, 157).reduce((b, t) => (nll(m => sigmoid(m / t), CAL.fit) < nll(m => sigmoid(m / b), CAL.fit) ? t : b), 1);
  const X = CAL.fit.map(d => [1, d.m]), y = CAL.fit.map(d => d.y);
  let w = [0, 0.1];
  for (let i = 0; i < 40; i++) w = logisticStep(w, X, y, 'newton');
  const bins = Array.from({ length: 10 }, (_, b) => { const s = CAL.fit.filter(d => Math.min(9, Math.floor(sigmoid(d.m) * 10)) === b); return s.length ? mean(s.map(d => d.y)) : (b + 0.5) / 10; });
  return { T, w, bins };
})();
const CAL_FN = { none: m => sigmoid(m), temp: m => sigmoid(m / FITTED.T), platt: m => sigmoid(FITTED.w[0] + FITTED.w[1] * m), bin: m => FITTED.bins[Math.min(9, Math.floor(sigmoid(m) * 10))] };

export const CalibrationFix = lab({
  intro: 'This model is overconfident and biased. Fit a repair on a held-out set, then judge it on fresh data.',
  params: [{ k: 'm', label: 'Repair', options: [['none', 'none'], ['temp', 'temperature'], ['platt', 'Platt'], ['bin', 'binning']], init: 'none' }],
  derive: ({ m }) => { const g = CAL_FN[m], P = CAL.test.map(d => ({ p: g(d.m), y: d.y })), used = Array.from({ length: 10 }, (_, b) => { const s = P.filter(x => Math.min(9, Math.floor(x.p * 10)) === b); return s.length ? { n: s.length, c: mean(s.map(x => x.p)), a: mean(s.map(x => x.y)) } : null; }).filter(Boolean); return { used, ece: sum(used.map(b => (b.n / P.length) * Math.abs(b.c - b.a))), brier: mean(P.map(x => (x.p - x.y) ** 2)) }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 360, h: 340, xl: 'predicted', yl: 'observed', label: 'reliability diagram', marks: [{ tone: 'muted', dash: true, data: [[0, 0], [1, 1]] }, { tone: 1, data: v.used.map(b => [b.c, b.a]) }, { as: 'points', tone: 1, r: 5, data: v.used.map(b => [b.c, b.a]) }] }],
  stats: v => [`ECE ${fmt(v.ece, 3)}`, `Brier ${fmt(v.brier, 4)}`],
  note: 'Temperature can only soften, so it cannot remove the bias. Platt has a shift as well.',
  takeaway: 'Fit calibration on data the model did not train on, or it learns nothing true.',
});

export const BrierDrill = drill({
  intro: 'The Brier score is the mean squared error of the probabilities.',
  tex: R`\text{Brier}=\frac1n\sum_i(p_i-y_i)^2`,
  gen: r => ({ p: Array.from({ length: 3 }, () => pick(r, [0.1, 0.3, 0.5, 0.7, 0.9])), y: ints(r, 3, 0, 1) }),
  q: v => `Predictions $${v.p.join(',\\ ')}$ for labels $${v.y.join(',\\ ')}$. Find the Brier score.`,
  answer: v => mean(v.p.map((p, i) => (p - v.y[i]) ** 2)), dp: 3,
  steps: v => [`Squared gaps: $${v.p.map((p, i) => fmt((p - v.y[i]) ** 2)).join(',\\ ')}$`],
});

/* ---------- SVM ---------- */

const PEG = dataset('overlap', 12, 40).map(d => [d[0], d[1], d[2] ? 1 : -1]);

export const Pegasos = lab({
  intro: 'SVMs can also be trained one point at a time. Step through stochastic subgradient descent on the hinge loss.',
  tex: R`w\leftarrow(1-\eta_t\lambda)\,w+\eta_t\,y_ix_i\,\mathbb 1\{y_iw^\top x_i<1\},\qquad \eta_t=\frac{1}{\lambda t}`,
  params: [{ k: 't', label: 'updates', min: 0, max: 400, step: 5, init: 40, play: true }, { k: 'l', label: '$\\lambda$', min: 0.01, max: 1, step: 0.01, init: 0.1 }, { k: 'seed', seed: 'Reshuffle', init: 1 }],
  derive: ({ t, l, seed }) => { const r = rng(seed * 47), obj = w => (l / 2) * (w[0] ** 2 + w[1] ** 2) + mean(PEG.map(p => Math.max(0, 1 - p[2] * (w[0] * p[0] + w[1] * p[1])))); let w = [0, 0]; const hist = [[0, obj(w)]]; for (let s = 1; s <= t; s++) { const p = PEG[Math.floor(r() * PEG.length)], e = 1 / (l * s), hit = p[2] * (w[0] * p[0] + w[1] * p[1]) < 1; w = w.map((v, j) => (1 - e * l) * v + (hit ? e * p[2] * p[j] : 0)); if (s % 5 === 0) hist.push([s, obj(w)]); } return { w, hist, loss: obj(w) }; },
  plots: v => [{ caption: 'Boundary', x: [-3, 3], y: [-3, 3], w: 300, h: 280, label: 'linear boundary from subgradient steps', marks: [{ as: 'field', fn: (a, b) => clamp((v.w[0] * a + v.w[1] * b) / 2, -1, 1) }, { as: 'contour', tone: 'fg', fn: (a, b) => v.w[0] * a + v.w[1] * b + 1e-9 }, { as: 'points', r: 4, data: PEG.map(p => [p[0], p[1], p[2] > 0 ? 1 : 2]) }] }, { caption: 'Objective by update', x: [0, 400], y: [0, 2], w: 300, h: 280, xl: 'updates', label: 'objective against updates', marks: [{ tone: 4, data: v.hist.map(h => [h[0], Math.min(h[1], 2)]) }] }],
  stats: v => [`objective ${fmt(v.loss, 3)}`, `‖w‖ = ${fmt(Math.hypot(...v.w), 3)}`],
  takeaway: 'Each update touches one point, so the cost does not grow with the dataset. That is why this works when SMO would be too slow.',
});

/* ---------- trees ---------- */

const six = (seed, n) => { const r = rng(seed); return Array.from({ length: n }, () => { const x = Array.from({ length: 6 }, () => +gauss(r).toFixed(3)); return { x, y: +(x[0] + x[1] + 0.6 * x[2] * x[3] + 0.4 * gauss(r) > 0) }; }); };
let forestCache;
const forestTable = () => (forestCache ??= (() => {
  const tr = six(3, 120), te = six(4, 300), X = tr.map(d => d.x), y = tr.map(d => d.y);
  return [1, 2, 3, 4, 5, 6].map(m => {
    const b = bagging(X, y, { B: 30, maxDepth: 8, m, seed: m * 5 }), acc = mean(te.map(d => +((b.predict(d.x) > 0.5) === !!d.y))), P = b.trees.slice(0, 12).map(t => te.map(d => predictTree(t, d.x)));
    const corr = (a, c) => { const ma = mean(a), mc = mean(c), n = sum(a.map((v, i) => (v - ma) * (c[i] - mc))), d = Math.sqrt(sum(a.map(v => (v - ma) ** 2)) * sum(c.map(v => (v - mc) ** 2))); return d ? n / d : 0; };
    const rs = []; for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) rs.push(corr(P[i], P[j]));
    return { m, acc, rho: mean(rs) };
  });
})());

export const ForestFeatures = lab({
  intro: 'Bagged trees all reach for the strongest feature first, so they look alike. Hiding features at each split makes them differ. Compare the correlation and the accuracy.',
  tex: R`\operatorname{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2`,
  params: [{ k: 'm', label: 'features tried at each split', min: 1, max: 6, step: 1, init: 2 }],
  derive: () => ({ T: forestTable() }),
  plots: v => [{ caption: 'Correlation between trees', x: [1, 6], y: [0, 1], w: 300, h: 240, xl: 'features tried', label: 'tree correlation', marks: [{ tone: 2, data: v.T.map(r => [r.m, r.rho]) }, { as: 'points', tone: 4, r: 6, data: [[v.m, v.T[v.m - 1].rho]] }] }, { caption: 'Test accuracy', x: [1, 6], y: [0.6, 1], w: 300, h: 240, xl: 'features tried', label: 'forest accuracy', marks: [{ tone: 1, data: v.T.map(r => [r.m, r.acc]) }, { as: 'points', tone: 4, r: 6, data: [[v.m, v.T[v.m - 1].acc]] }] }],
  stats: v => [`accuracy ${pct(v.T[v.m - 1].acc)}`, `correlation ${fmt(v.T[v.m - 1].rho, 3)}`],
  takeaway: 'Too few features weaken each tree, and too many make them identical. All features tried is plain bagging.',
});

let oobCache;
const oobTable = () => (oobCache ??= (() => {
  const D = dataset('overlap', 21, 120), T = dataset('overlap', 22, 400), X = D.map(d => [d[0], d[1]]), y = D.map(d => d[2]);
  return [1, 2, 3, 5, 8, 12, 20, 35, 60, 100].map(B => {
    const b = bagging(X, y, { B, maxDepth: 8, seed: 3 }), seen = b.oob.map((p, i) => [p, y[i]]).filter(q => Number.isFinite(q[0]));
    return { B, oob: 1 - mean(seen.map(q => +((q[0] > 0.5) === !!q[1]))), test: 1 - accuracy(b.predict, T), cover: seen.length / y.length };
  });
})());

export const OobCurve = lab({
  intro: 'Out-of-bag error is free: no separate validation set is needed. See how closely it follows the true test error as trees are added.',
  params: [{ k: 'i', label: 'position along the forest size', min: 0, max: 9, step: 1, init: 4, show: i => [1, 2, 3, 5, 8, 12, 20, 35, 60, 100][i] }],
  derive: () => ({ T: oobTable() }),
  plots: v => [{ x: [0, 2], y: [0, 0.6], xl: 'log₁₀ trees', yl: 'error', label: 'out-of-bag and test error', marks: [{ tone: 2, data: v.T.map(r => [Math.log10(r.B), r.oob]) }, { tone: 4, data: v.T.map(r => [Math.log10(r.B), r.test]) }, { as: 'rule', dash: true, x: Math.log10(v.T[v.i].B) }] }],
  stats: v => [`out-of-bag ${pct(v.T[v.i].oob)}`, `test ${pct(v.T[v.i].test)}`, `rows with a prediction ${pct(v.T[v.i].cover, 0)}`],
  legend: [['--c2', 'out-of-bag'], ['--c4', 'test']],
});

export const SplitScoreDrill = drill({
  intro: 'For squared error, the better split has the larger score. Leaf means do the work.',
  tex: R`\text{score}=\frac{\big(\sum_L r\big)^2}{n_L}+\frac{\big(\sum_R r\big)^2}{n_R}`,
  gen: (r, n) => (n === 0 ? { L: [-5, -4, -3], R: [2, 4, 6] } : { L: ints(r, 3, -6, 6), R: ints(r, 3, -6, 6) }),
  q: v => `Left leaf residuals: $${v.L.join(',\\ ')}$. Right: $${v.R.join(',\\ ')}$. Find the split score.`,
  answer: v => sum(v.L) ** 2 / v.L.length + sum(v.R) ** 2 / v.R.length,
  steps: v => [`Left sum: $${sum(v.L)}$`, `Right sum: $${sum(v.R)}$`],
});

const CC = dataset('overlap', 31, 90), CT = dataset('overlap', 32, 400), CX = CC.map(d => [d[0], d[1]]), CY = CC.map(d => d[2]), DEEP = buildTree(CX, CY, { maxDepth: 12 });
const ALPHAS = linspace(-3, -0.7, 24).map(l => { const p = pruneTree(DEEP, 10 ** l, CX, CY); return { l, leaves: p.leaves, tr: accuracy(x => predictTree(p.node, x), CC), te: accuracy(x => predictTree(p.node, x), CT) }; });

export const CostComplexity = lab({
  intro: 'A full tree memorises. Charge a price per leaf and branches that do not earn their keep are cut.',
  tex: R`R_\alpha(T)=R(T)+\alpha\,|T|`,
  params: [{ k: 'lg', label: '$\\log_{10}\\alpha$', min: -3, max: -0.7, step: 0.02, init: -2 }],
  derive: ({ lg }) => { const p = pruneTree(DEEP, 10 ** lg, CX, CY), f = x => predictTree(p.node, x); return { p, f, tr: accuracy(f, CC), te: accuracy(f, CT) }; },
  plots: v => [{ caption: 'Regions', x: [-3, 3], y: [-3, 3], w: 300, h: 280, label: 'pruned tree regions', marks: [{ as: 'field', fn: (a, b) => 2 * v.f([a, b]) - 1 }, { as: 'points', r: 3, data: CC.map(d => [d[0], d[1], d[2] + 1]) }] }, { caption: 'Accuracy against the price', x: [-3, -0.7], y: [0.5, 1], w: 300, h: 280, xl: 'log₁₀ α', label: 'accuracy against alpha', marks: [{ tone: 3, data: ALPHAS.map(a => [a.l, a.tr]) }, { tone: 4, data: ALPHAS.map(a => [a.l, a.te]) }, { as: 'rule', dash: true, x: v.lg }] }],
  stats: v => [`${v.p.leaves} leaves`, `train ${pct(v.tr, 0)}, test ${pct(v.te, 0)}`],
  legend: [['--c3', 'train'], ['--c4', 'test']],
  takeaway: 'Training accuracy only falls as you prune, while test accuracy can rise first. Choose the price by cross-validation.',
});

export const PruneDrill = drill({
  intro: 'The price at which a subtree is no longer worth keeping.',
  tex: R`\alpha=\frac{R(t)-R(T_t)}{|T_t|-1}`,
  gen: (r, n) => (n === 0 ? { c: 0.2, s: 0.1, l: 3 } : { c: pick(r, [0.18, 0.2, 0.3]), s: pick(r, [0.06, 0.1, 0.12]), l: int(r, 2, 6) }),
  q: v => `Collapsed, the node has error ${v.c}. Its subtree has error ${v.s} over ${v.l} leaves. At what $\\alpha$ is it pruned?`,
  answer: v => (v.c - v.s) / (v.l - 1), dp: 3,
  steps: v => [`Error saved: $${fmt(v.c - v.s)}$`, `Extra leaves: $${v.l - 1}$`],
});

/* ---------- practical ---------- */

export const EncodingDrill = drill({
  intro: 'Each category adds its levels as binary columns. With an intercept, each one must drop a level or the columns become collinear.',
  gen: r => ({ L: ints(r, int(r, 1, 3), 2, 6), num: int(r, 0, 4) }),
  q: v => `A model has an intercept, ${v.num} numeric features, and categories with ${v.L.join(', ')} levels. How many columns does the design matrix have?`,
  answer: v => 1 + v.num + sum(v.L.map(l => l - 1)), dp: 0,
  steps: v => [`Category columns: $${v.L.map(l => l - 1).join('+')}$`, `Add the ${v.num} numeric columns, and the intercept.`],
});

const BAS = (() => { const r = rng(55), xs = Array.from({ length: 36 }, () => +(-3 + 6 * r()).toFixed(3)); return { xs, ys: xs.map(x => 0.5 * x * x + Math.sin(2 * x) - 1 + 0.35 * gauss(r)) }; })();
const BASIS = { line: x => [1, x], quad: x => [1, x, x * x], full: x => [1, x, x * x, Math.sin(2 * x), Math.cos(2 * x)] };

export const BasisFeatures = lab({
  intro: 'The model stays linear in its coefficients, and only the columns change. Better features are often worth more than a better algorithm.',
  params: [{ k: 'b', label: 'Features', options: [['line', 'x'], ['quad', 'x and x²'], ['full', 'x, x², sin 2x, cos 2x']], init: 'line' }],
  derive: ({ b }) => { const X = BAS.xs.map(BASIS[b]), c = ridge(X, BAS.ys, 1e-8), f = x => sum(BASIS[b](x).map((v, i) => v * c[i])); return { f, r2: 1 - sum(BAS.xs.map((x, i) => (BAS.ys[i] - f(x)) ** 2)) / sum(BAS.ys.map(y => (y - mean(BAS.ys)) ** 2)) }; },
  plots: v => [{ caption: 'Fit', x: [-3, 3], y: [-4, 6], w: 300, h: 250, label: 'least squares fit on chosen features', marks: [{ as: 'points', tone: 'fg', r: 3, data: BAS.xs.map((x, i) => [x, BAS.ys[i]]) }, { tone: 1, data: curve(v.f, -3, 3) }] }, { caption: 'Residuals', x: [-3, 3], y: [-4, 4], w: 300, h: 250, label: 'residuals', marks: [{ as: 'rule', dash: true, y: 0 }, { as: 'points', tone: 2, r: 3, data: BAS.xs.map((x, i) => [x, BAS.ys[i] - v.f(x)]) }] }],
  stats: v => [`$R^2$ = ${fmt(v.r2, 3)}`],
});

const XOR = dataset('xor', 41, 60);

export const InteractionTerm = lab({
  intro: 'A linear classifier cannot solve XOR. Add one new column, the product of the two features, and it can.',
  params: [{ k: 'f', label: 'Features', options: [['base', 'x₁, x₂'], ['inter', 'x₁, x₂, x₁x₂']], init: 'base' }],
  derive: ({ f }) => { const phi = (a, b) => (f === 'inter' ? [1, a, b, a * b] : [1, a, b]), X = XOR.map(d => phi(d[0], d[1])), y = XOR.map(d => d[2]); let w = Array(X[0].length).fill(0); for (let i = 0; i < 40; i++) w = logisticStep(w, X, y, 'gd', 0.8); const p = (a, b) => sigmoid(phi(a, b).reduce((s, v, i) => s + v * w[i], 0)); return { p, w, acc: mean(XOR.map(d => +((p(d[0], d[1]) > 0.5) === !!d[2]))) }; },
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, label: 'logistic regression with and without an interaction', marks: [{ as: 'field', fn: (a, b) => 2 * v.p(a, b) - 1 }, { as: 'points', r: 4, data: XOR.map(d => [d[0], d[1], d[2] + 1]) }] }],
  stats: v => [`accuracy ${pct(v.acc, 0)}`],
  takeaway: 'Still linear in the coefficients, so the optimization stays convex. Trees and kernels find such terms by themselves.',
});

export const LeveragePoint = lab({
  intro: 'A point far out along $x$ has a long lever. Drag it, and compare the fit with and without it.',
  tex: R`h_{ii}=\frac1n+\frac{(x_i-\bar x)^2}{\sum_j(x_j-\bar x)^2}`,
  params: [{ k: 'x', label: 'position of the last point', min: 0, max: 15, step: 0.1, init: 13 }, { k: 'y', label: 'height of the last point', min: 0, max: 15, step: 0.1, init: 3 }],
  derive: ({ x, y }) => { const bx = [1, 2, 3, 4, 5, 6, 7], by = [1.4, 2.2, 2.9, 4.3, 4.8, 6.2, 6.9], xs = [...bx, x], ys = [...by, y], a = linfit(xs, ys), b = linfit(bx, by), mx = mean(xs), sxx = sum(xs.map(v => (v - mx) ** 2)), h = 1 / xs.length + (x - mx) ** 2 / sxx, res = xs.map((v, i) => ys[i] - a.w * v - a.b), s2 = sum(res.map(r => r * r)) / (xs.length - 2), cook = ((res.at(-1) ** 2 / (2 * s2)) * h) / (1 - h) ** 2; return { bx, by, a, b, h, cook }; },
  plots: v => [{ x: [0, 15], y: [0, 15], w: 380, h: 340, xl: 'x', yl: 'y', label: 'fit with and without a high leverage point', marks: [{ tone: 3, dash: true, data: [[0, v.b.b], [15, v.b.w * 15 + v.b.b]] }, { tone: 1, data: [[0, v.a.b], [15, v.a.w * 15 + v.a.b]] }, { as: 'points', tone: 'fg', r: 4, data: v.bx.map((x, i) => [x, v.by[i]]) }, { drag: { x: 'x', y: 'y', tone: 4 } }] }],
  stats: v => [`leverage ${fmt(v.h, 3)}`, `Cook’s distance ${fmt(v.cook, 2)}`, `slope ${fmt(v.a.w, 3)}, was ${fmt(v.b.w, 3)}`],
  status: v => (v.cook > 1 ? ['alert', 'Influential: this single point is steering the fit.'] : undefined),
  legend: [['--c1', 'with the point'], ['--c3', 'without it']],
  takeaway: 'High leverage alone is not a problem. It takes leverage and a large residual together to turn the line.',
});

export const BaselineDrill = drill({
  intro: 'Always score the simplest possible model first.',
  gen: r => ({ y: ints(r, 4, 0, 9), k: pick(r, ['mse']) }),
  q: v => `Targets: $${v.y.join(',\\ ')}$. What mean squared error does always predicting the mean give?`,
  answer: v => { const m = mean(v.y); return mean(v.y.map(x => (x - m) ** 2)); },
  steps: v => [`Mean: $${fmt(mean(v.y))}$`, 'That error is simply the variance of the targets.'],
});

export const MomentumLab = lab({
  intro: 'In a long valley, plain descent zig-zags. Momentum averages the zig-zag away and keeps the forward motion.',
  tex: R`v_{t+1}=\beta v_t-\eta\nabla L,\qquad \theta_{t+1}=\theta_t+v_{t+1}`,
  params: [{ k: 'k', label: 'steepness $\\kappa$', min: 2, max: 40, step: 1, init: 20 }, { k: 'e', label: '$\\eta\\kappa$', min: 0.1, max: 1.9, step: 0.05, init: 1.2 }, { k: 'b', label: 'momentum $\\beta$', min: 0, max: 0.95, step: 0.01, init: 0.7 }],
  derive: ({ k, e, b }) => { const run = beta => { let p = [2.8, 1.6], v = [0, 0]; const eta = e / k, out = [p]; for (let i = 0; i < 40; i++) { const g = [p[0], k * p[1]]; v = v.map((u, j) => beta * u - eta * g[j]); p = p.map((u, j) => u + v[j]); if (Math.hypot(...p) > 1e3) break; out.push(p); } return out; }; const L = p => 0.5 * (p[0] ** 2 + k * p[1] ** 2); return { plain: run(0), mom: run(b), L }; },
  plots: v => [{ x: [-3, 3], y: [-2, 2], w: 400, h: 320, label: 'paths with and without momentum', marks: [{ as: 'field', bands: 9, tones: [1, 1], fn: (a, b) => Math.sqrt(clamp(v.L([a, b]) / v.L([3, 2]), 0, 1)) }, { tone: 3, data: v.plain }, { tone: 4, data: v.mom }] }],
  stats: v => [`after 40 steps: plain ${v.L(v.plain.at(-1)).toExponential(1)}`, `with momentum ${v.L(v.mom.at(-1)).toExponential(1)}`],
  legend: [['--c3', 'plain descent'], ['--c4', 'momentum']],
});

export const FeatureScaling = lab({
  intro: 'A feature in larger units makes the bowl steeper in that direction, and the safe step smaller. Standardizing makes it round.',
  params: [{ k: 'std', label: 'Scaling', options: [['raw', 'raw'], ['std', 'standardized']], init: 'raw' }, { k: 'lg', label: '$\\log_{10}$ scale of the feature', min: 0, max: 2, step: 0.02, init: 1 }],
  derive: ({ std, lg }) => { const s = std === 'std' ? 1 : 10 ** lg, mu = s * s, eta = 2 / (mu + 1), rate = Math.abs(1 - eta), steps = rate < 1e-9 ? 1 : Math.ceil(Math.log(1e-3) / Math.log(rate)); return { s, mu, eta, steps, kappa: mu }; },
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 360, h: 340, xl: 'scaled feature direction', yl: 'other feature', label: 'loss contours under scaling', marks: [{ as: 'contour', tone: 1, levels: [0.1, 0.5, 1, 2, 4], fn: (a, b) => 0.5 * (Math.min(v.mu, 400) * a * a + b * b) }] }],
  stats: v => [`condition number ${fmt(v.kappa, 0)}`, `best step ${v.eta.toExponential(1)}`, `about ${v.steps} steps to converge`],
});

export const LossDiagnosis = lab({
  intro: 'Most training problems can be read off the loss curve. Change the step size and diagnose.',
  params: [{ k: 'x', label: 'step size times curvature', min: 0.02, max: 2.2, step: 0.01, init: 0.1 }],
  derive: ({ x }) => { const pts = [[0, 1]]; let e = 1; for (let i = 1; i <= 30; i++) { e *= 1 - x; pts.push([i, Math.min(e * e, 3)]); } return { pts }; },
  plots: v => [{ x: [0, 30], y: [0, 2], xl: 'step', yl: 'loss', label: 'loss curve', marks: [{ tone: 4, data: v.pts }] }],
  status: ({ x }) => (x > 2 ? ['alert', 'Diverging: the step is too large.'] : x > 1.3 ? ['alert', 'Oscillating: each step overshoots the valley.'] : x > 0.6 ? ['ok', 'Healthy: fast and stable.'] : [undefined, 'Slow: raise the step, or fix the scaling.']),
});

export const GradientCheckDrill = drill({
  intro: 'Check a derivative numerically. The central difference is accurate to order $h^2$.',
  tex: R`f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}`,
  gen: r => ({ x: int(r, 1, 4), h: pick(r, [0.1, 0.01]) }),
  q: v => `For $f(x)=x^3$ at $x=${v.x}$ with $h=${v.h}$, compute the central difference.`,
  answer: v => ((v.x + v.h) ** 3 - (v.x - v.h) ** 3) / (2 * v.h), dp: 3,
  steps: v => [`$f(x+h)=${fmt((v.x + v.h) ** 3, 4)}$`, `$f(x-h)=${fmt((v.x - v.h) ** 3, 4)}$`, `Exact derivative: $${3 * v.x * v.x}$`],
});

const DM = dataset('blobs', 18, 40).map(d => [d[0], d[1], d[2] + 1]);

export const DistanceMetrics = lab({
  intro: 'Who counts as near depends on the metric. Drag the query and watch the set of neighbours and the shape of the ball.',
  tex: R`d_p(x,z)=\Big(\sum_j|x_j-z_j|^p\Big)^{1/p}`,
  params: [{ k: 'm', label: 'Metric', options: [['l1', 'Manhattan'], ['l2', 'Euclidean'], ['linf', 'Chebyshev']], init: 'l2' }, { k: 'x', label: 'query $x_1$', min: -3, max: 3, step: 0.05, init: 0.4 }, { k: 'y', label: 'query $x_2$', min: -3, max: 3, step: 0.05, init: 0.2 }, { k: 'k', label: 'neighbours', min: 1, max: 15, step: 1, init: 7 }],
  derive: ({ m, x, y, k }) => { const d = p => { const a = Math.abs(p[0] - x), b = Math.abs(p[1] - y); return m === 'l1' ? a + b : m === 'l2' ? Math.hypot(a, b) : Math.max(a, b); }, near = DM.map((p, i) => [d(p), i]).sort((a, b) => a[0] - b[0]).slice(0, k), R0 = near.at(-1)[0], ball = linspace(0, 2 * Math.PI, 241).map(t => { const c = Math.cos(t), s = Math.sin(t), n = m === 'l1' ? Math.abs(c) + Math.abs(s) : m === 'l2' ? 1 : Math.max(Math.abs(c), Math.abs(s)); return [x + (R0 * c) / n, y + (R0 * s) / n]; }); return { near, R0, ball }; },
  plots: v => [{ x: [-3.5, 3.5], y: [-3.5, 3.5], w: 380, h: 360, label: 'neighbours under a chosen metric', marks: [{ tone: 3, data: v.ball }, { as: 'points', r: 4, data: DM }, { as: 'rings', r: 8, data: v.near.map(n => DM[n[1]]) }, { drag: { x: 'x', y: 'y' } }] }],
  stats: v => [`radius ${fmt(v.R0)}`, `votes for class one: ${v.near.filter(n => DM[n[1]][2] === 2).length} of ${v.near.length}`],
});

export const MinkowskiDrill = drill({
  intro: 'Three metrics on the same pair.',
  gen: r => ({ a: ints(r, 2, -4, 6), b: ints(r, 2, -4, 6), m: pick(r, ['Manhattan', 'Euclidean', 'Chebyshev']) }),
  q: v => `Find the ${v.m} distance between $(${v.a.join(',')})$ and $(${v.b.join(',')})$.`,
  answer: v => { const d = [Math.abs(v.a[0] - v.b[0]), Math.abs(v.a[1] - v.b[1])]; return v.m === 'Manhattan' ? d[0] + d[1] : v.m === 'Euclidean' ? Math.hypot(...d) : Math.max(...d); },
  steps: v => [`Coordinate gaps: $${Math.abs(v.a[0] - v.b[0])}$ and $${Math.abs(v.a[1] - v.b[1])}$`],
});

export const CosineDrill = drill({
  intro: 'Cosine similarity ignores length and compares only direction.',
  tex: R`\cos(a,b)=\frac{a\cdot b}{\|a\|\,\|b\|}`,
  gen: r => ({ a: pick(r, [[3, 4], [1, 0], [0, 2], [6, 8]]), b: pick(r, [[4, 3], [0, 5], [1, 0], [3, 4], [5, 12]]) }),
  q: v => `Find the cosine similarity of $(${v.a.join(',')})$ and $(${v.b.join(',')})$.`,
  answer: v => (v.a[0] * v.b[0] + v.a[1] * v.b[1]) / (Math.hypot(...v.a) * Math.hypot(...v.b)), dp: 3,
  steps: v => [`Dot product: $${v.a[0] * v.b[0] + v.a[1] * v.b[1]}$`, `Lengths: $${Math.hypot(...v.a)}$ and $${Math.hypot(...v.b)}$`],
});

const CV_N = 24, GROUP = Array.from({ length: CV_N }, (_, i) => Math.floor(i / 3));
const CV = { random: (i, k) => (i * 7) % 4 === k, stratified: (i, k) => (i % 2 === 0 ? i / 2 : (i - 1) / 2) % 4 === k, group: (i, k) => GROUP[i] % 4 === k, time: (i, k) => i >= 8 + 4 * k && i < 12 + 4 * k };
const TEXT = { random: 'Rows are independent, so any split is fair.', stratified: 'Each fold keeps the same class balance, which matters for rare classes.', group: 'All rows of one subject stay together, so nobody is in both train and test.', time: 'Training always comes from the past, and the future is never used.' };

export const CvSchemes = lab({
  intro: 'Rows run left to right, and folds run top to bottom. Blue trains, red validates, and grey is not used. Each design protects against a different leak.',
  params: [{ k: 's', label: 'Design', options: [['random', 'k-fold'], ['stratified', 'stratified'], ['group', 'group'], ['time', 'forward chaining']], init: 'time' }],
  plots: v => [{ x: [0, CV_N], y: [0, 4], w: 440, h: 180, xl: 'rows', yl: 'fold', label: 'fold assignment grid', marks: [0, 1, 2, 3].flatMap(k => Array.from({ length: CV_N }, (_, i) => ({ as: 'rects', tone: CV[v.s](i, k) ? 4 : v.s === 'time' && i >= 12 + 4 * k ? 'muted' : 1, data: [[i + 0.1, 3 - k + 0.15, i + 0.9, 4 - k - 0.15]] }))) }],
  note: 'For the group design, every three rows belong to one subject.',
  stats: v => [TEXT[v.s]],
});

export const TestScoreInterval = lab({
  intro: 'A test score is an estimate. Move the size of the test set and see how much of a difference you could actually detect.',
  tex: R`\hat p\pm1.96\sqrt{\frac{\hat p(1-\hat p)}{n}}`,
  params: [{ k: 'p', label: 'measured accuracy', min: 0.5, max: 0.99, step: 0.01, init: 0.9 }, { k: 'n', label: 'test cases', min: 20, max: 2000, step: 10, init: 400 }],
  derive: ({ p, n }) => ({ hw: 1.96 * Math.sqrt((p * (1 - p)) / n) }),
  plots: v => [{ x: [20, 2000], y: [0, 0.15], xl: 'test cases', yl: 'half-width of the interval', label: 'interval width against test size', marks: [{ tone: 1, data: curve(n => 1.96 * Math.sqrt((v.p * (1 - v.p)) / n), 20, 2000) }, { drag: { x: 'n', y: v.hw } }] }],
  stats: v => [`${pct(v.p, 0)} plus or minus ${pct(v.hw)}`],
  takeaway: 'To halve the interval you need four times the test cases, so a one-point gain on a small test set shows nothing.',
});

export const IntervalDrill = drill({
  intro: 'The half-width of a 95% interval for an accuracy.',
  gen: (r, n) => (n === 0 ? { p: 0.9, N: 400 } : { p: pick(r, [0.5, 0.8, 0.9]), N: pick(r, [100, 400, 900, 2500]) }),
  q: v => `Accuracy ${v.p} on ${v.N} cases. What is the half-width of the 95% interval?`,
  answer: v => 1.96 * Math.sqrt((v.p * (1 - v.p)) / v.N), dp: 3,
  steps: v => [`Standard error: $\\sqrt{${fmt(v.p * (1 - v.p))}/${v.N}}=${fmt(Math.sqrt((v.p * (1 - v.p)) / v.N), 4)}$`, 'Multiply by 1.96.'],
});

const IMPORT = (() => { const r = rng(71), make = n => Array.from({ length: n }, () => { const z = gauss(r), x = [z + 0.2 * gauss(r), z + 0.2 * gauss(r), gauss(r), gauss(r)].map(v => +v.toFixed(3)); return { x, y: +(2 * z + x[2] + 0.5 * gauss(r) > 0) }; }); return { tr: make(120), te: make(300) }; })();
let importCache = {};
const importance = twin => (importCache[twin] ??= (() => {
  const keep = twin === 'both' ? [0, 1, 2, 3] : [0, 2, 3], sel = x => keep.map(j => x[j]), X = IMPORT.tr.map(d => sel(d.x)), y = IMPORT.tr.map(d => d.y), f = bagging(X, y, { B: 25, maxDepth: 6, m: 2, seed: 9 }), Te = IMPORT.te.map(d => ({ x: sel(d.x), y: d.y })), r = rng(5), base = mean(Te.map(d => +((f.predict(d.x) > 0.5) === !!d.y)));
  return keep.map((name, c) => { const col = Te.map(d => d.x[c]).map(v => [r(), v]).sort((a, b) => a[0] - b[0]).map(p => p[1]); return { name, drop: base - mean(Te.map((d, i) => +((f.predict(d.x.map((v, j) => (j === c ? col[i] : v))) > 0.5) === !!d.y))) }; });
})());

export const PermutationImportance = lab({
  intro: 'Shuffle one column in the test set and measure the damage. Then add a near-copy of the top feature, and watch its importance appear to shrink.',
  params: [{ k: 'twin', label: 'Columns', options: [['one', 'without the twin'], ['both', 'with a near-copy']], init: 'one' }],
  derive: ({ twin }) => ({ I: importance(twin) }),
  plots: v => [{ x: [-0.6, v.I.length - 0.4], y: [0, 0.4], yl: 'accuracy lost when shuffled', label: 'permutation importance', marks: [{ as: 'rects', tone: 1, data: v.I.map((b, i) => [i - 0.35, 0, i + 0.35, Math.max(b.drop, 0)]) }, ...v.I.map((b, i) => ({ as: 'label', x: i, y: 0.37, text: ['signal', 'near-copy', 'weak', 'noise'][b.name], anchor: 'middle' }))] }],
  stats: v => [`largest drop ${pct(Math.max(...v.I.map(b => b.drop)))}`],
  takeaway: 'When two features carry the same information, shuffling one leaves the other to cover for it. Low importance does not mean a feature is useless.',
});
