'use client';
// Explorers added by the final subtopic audit: note sections that still had nothing to try.
import { R, drill, int, lab, mat, pick } from '../components/labs';
import { gauss, mean, rng, sum } from '../lib/math.mjs';
import { accuracy, buildTree, clamp, dataset, fmt, kernels, linspace, pct, predictTree, smo, svd } from '../lib/models.mjs';

/* ---------- introduction ---------- */

export const NotationDrill = drill({
  intro: 'Machine learning notes reuse a small set of marks. Each one tells you what kind of quantity you are looking at.',
  gen: r => pick(r, [[R`\hat\theta`, 0], [R`\hat y_i`, 0], [R`\hat R`, 0], [R`\theta^{*}`, 1], [R`f^{*}`, 1], [R`x_i`, 2], [R`y_i`, 2], [R`x_{ij}`, 3], [R`\bar x`, 4]]),
  q: v => `What does $${v[0]}$ mean?`,
  choices: () => ['An estimate computed from data', 'The ideal or best possible value', 'The value for example $i$', 'Feature $j$ of example $i$', 'An average'],
  answer: v => v[1],
  steps: () => ['A hat means estimated. A star means ideal. A subscript picks an item. A bar means an average.'],
});

export const SlopeBetweenPoints = drill({
  intro: 'Slope is the change in output for each unit change in input.',
  tex: R`m=\frac{y_2-y_1}{x_2-x_1}`,
  gen: r => { const a = int(r, -4, 3), b = a + int(r, 1, 5); return { a, b, c: int(r, -5, 6), d: int(r, -5, 6) }; },
  q: v => `Find the slope of the line through $(${v.a},${v.c})$ and $(${v.b},${v.d})$.`,
  answer: v => (v.d - v.c) / (v.b - v.a),
  steps: v => [`Rise: $${v.d}-(${v.c})=${v.d - v.c}$`, `Run: $${v.b}-(${v.a})=${v.b - v.a}$`],
});

export const ExpectationDrill = drill({
  intro: 'An expectation is a weighted average, with the probabilities as weights.',
  tex: R`\mathbb E[X]=\sum_x x\,p(x)`,
  gen: r => { const P = pick(r, [[0.5, 0.25, 0.25], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]]); return { P, X: [int(r, -2, 1), int(r, 2, 4), int(r, 5, 9)] }; },
  q: v => `$X$ takes values $${v.X.join(',\\ ')}$ with probabilities $${v.P.join(',\\ ')}$. Find $\\mathbb E[X]$.`,
  answer: v => sum(v.X.map((x, i) => x * v.P[i])),
  steps: v => [`Terms: $${v.X.map((x, i) => `(${x})(${v.P[i]})`).join('+')}$`],
});

/* ---------- linear regression ---------- */

export const QuadraticFormDrill = drill({
  intro: 'A quadratic form turns a vector and a matrix into one number. Positive for every nonzero $v$ means positive definite.',
  tex: R`v^\top Av=\sum_{i,j}v_iA_{ij}v_j`,
  gen: r => ({ a: int(r, -2, 4), b: int(r, -3, 3), d: int(r, -2, 4), v: [int(r, -2, 3), int(r, -2, 3)] }),
  q: v => `For $A=${mat([[v.a, v.b], [v.b, v.d]])}$ and $v=(${v.v.join(',')})$, compute $v^\\top Av$.`,
  answer: v => v.a * v.v[0] ** 2 + 2 * v.b * v.v[0] * v.v[1] + v.d * v.v[1] ** 2, dp: 0,
  steps: v => [`$Av=(${v.a * v.v[0] + v.b * v.v[1]},\\ ${v.b * v.v[0] + v.d * v.v[1]})$`, 'Dot that with $v$.'],
});

export const TransposeDrill = drill({
  intro: 'A transpose swaps rows and columns, and reverses the order of a product. Answer like 3x4.',
  tex: R`(AB)^\top=B^\top A^\top`,
  gen: r => { const m = int(r, 1, 5), n = int(r, 1, 5), q = int(r, 1, 5); return { m, n, q }; },
  q: v => `$A$ is $${v.m}\\times${v.n}$ and $B$ is $${v.n}\\times${v.q}$. What is the shape of $(AB)^\\top$?`,
  answer: v => `${v.q}x${v.m}`,
  steps: v => [`$AB$ is $${v.m}\\times${v.q}$.`, 'The transpose swaps those two numbers.'],
});

/* ---------- SVM ---------- */

const SC = dataset('circles', 13, 40), SCT = dataset('circles', 14, 200);

export const SvmScaling = lab({
  intro: 'A Gaussian kernel measures distance, so a feature in large units decides everything. Stretch the second feature, then standardize.',
  tex: R`k(x,z)=\exp(-\gamma\|x-z\|^2)`,
  params: [{ k: 'std', label: 'Preprocessing', options: [['raw', 'raw features'], ['std', 'standardized']], init: 'raw' }, { k: 'lg', label: '$\\log_{10}$ scale of the second feature', min: 0, max: 2, step: 0.05, init: 1 }],
  derive: ({ std, lg }) => { const s = std === 'std' ? 1 : 10 ** lg, y = SC.map(d => (d[2] ? 1 : -1)), m = smo(SC.map(d => [d[0], d[1] * s]), y, { C: 10, k: kernels.rbf({ gamma: 0.6 }) }), f = (a, b) => m.f([a, b * s]); return { f, acc: mean(SCT.map(d => +((f(d[0], d[1]) > 0) === !!d[2]))) }; },
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, xl: 'x₁', yl: 'x₂ (shown in standard units)', label: 'SVM boundary under feature scaling', marks: [{ as: 'field', nx: 44, ny: 34, fn: (a, b) => clamp(v.f(a, b), -1, 1) }, { as: 'points', r: 4, data: SC.map(d => [d[0], d[1], d[2] + 1]) }] }],
  stats: v => [`test accuracy ${pct(v.acc, 0)}`],
  takeaway: 'Scale features using the training data, and apply the same transformation to new data.',
});

/* ---------- trees ---------- */

const TI = dataset('overlap', 37, 50), TIT = dataset('overlap', 38, 300);
const rule = t => (t.l ? `x${t.f === 0 ? '₁' : '₂'} ≤ ${fmt(t.t)}` : 'no split');

export const TreeInstability = lab({
  intro: 'Train the same tree on two slightly different samples. The first question can change, and everything below it changes as well.',
  params: [{ k: 'seed', seed: 'Draw another sample', init: 1 }, { k: 'd', label: 'max depth', min: 1, max: 8, step: 1, init: 5 }],
  derive: ({ seed, d }) => { const r = rng(seed * 89), pick2 = TI.map(() => TI[Math.floor(r() * TI.length)]), fit = D => buildTree(D.map(p => [p[0], p[1]]), D.map(p => p[2]), { maxDepth: d }), A = fit(TI), B = fit(pick2), g = linspace(-2.9, 2.9, 40), agree = mean(g.flatMap(x => g.map(y => +((predictTree(A, [x, y]) > 0.5) === (predictTree(B, [x, y]) > 0.5))))); return { A, B, agree }; },
  plots: v => [v.A, v.B].map((t, i) => ({ caption: `${i ? 'Resampled data' : 'Original data'}: first question ${rule(t)}`, x: [-3, 3], y: [-3, 3], w: 300, h: 280, label: 'tree regions', marks: [{ as: 'field', nx: 44, ny: 34, fn: (a, b) => 2 * predictTree(t, [a, b]) - 1 }] })),
  stats: v => [`the two trees agree on ${pct(v.agree, 0)} of the plane`, `test accuracy ${pct(accuracy(x => predictTree(v.A, x), TIT), 0)} and ${pct(accuracy(x => predictTree(v.B, x), TIT), 0)}`],
  takeaway: 'Similar accuracy, very different trees. This variance is exactly what averaging removes.',
});

export const ResidualIsGradient = drill({
  intro: 'For squared loss, stepping against the gradient means adding the residual.',
  tex: R`L=\tfrac12(y-F)^2\quad\Rightarrow\quad-\frac{\partial L}{\partial F}=y-F`,
  gen: r => ({ y: int(r, -5, 9), F: int(r, -5, 9), k: pick(r, ['g', 'n']) }),
  q: v => `$y=${v.y}$ and $F=${v.F}$. Find the ${v.k === 'g' ? 'gradient $\\partial L/\\partial F$' : 'negative gradient'}.`,
  answer: v => (v.k === 'g' ? v.F - v.y : v.y - v.F), dp: 0,
  steps: v => ['The derivative of $\\tfrac12(y-F)^2$ with respect to $F$ is $F-y$.', v.k === 'n' ? 'Change its sign: that is the residual.' : 'Substitute the two values.'],
});

/* ---------- unsupervised ---------- */

export const MinimumNorm = lab({
  intro: 'One equation, two unknowns: a whole line of perfect solutions. Slide along it. The pseudoinverse picks the point closest to the origin.',
  tex: R`\hat\beta=X^{+}y,\qquad \hat\beta\perp\{\text{all directions that leave }X\beta\text{ unchanged}\}`,
  params: [{ k: 'a', label: '$a_1$', min: 0.2, max: 3, step: 0.05, init: 2 }, { k: 'b', label: '$a_2$', min: 0.2, max: 3, step: 0.05, init: 1 }, { k: 'y', label: 'target $y$', min: 1, max: 6, step: 0.1, init: 4 }, { k: 't', label: 'position along the line', min: -5, max: 5, step: 0.05, init: 2.5 }],
  derive: ({ a, b, y, t }) => { const n = a * a + b * b, p = [(y * a) / n, (y * b) / n], d = [-b / Math.sqrt(n), a / Math.sqrt(n)], q = [p[0] + t * d[0], p[1] + t * d[1]]; return { p, d, q, best: Math.hypot(...p) }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, xl: 'β₁', yl: 'β₂', label: 'line of solutions with minimum norm point', marks: [{ tone: 1, data: [[v.p[0] - 12 * v.d[0], v.p[1] - 12 * v.d[1]], [v.p[0] + 12 * v.d[0], v.p[1] + 12 * v.d[1]]] }, { tone: 'muted', dash: true, data: linspace(0, 6.3, 91).map(s => [v.best * Math.cos(s), v.best * Math.sin(s)]) }, { as: 'arrow', tone: 3, data: [[0, 0, ...v.p]] }, { as: 'points', tone: 4, r: 6, data: [v.q] }] }],
  stats: v => [`length here ${fmt(Math.hypot(...v.q), 3)}`, `smallest possible ${fmt(v.best, 3)}`],
  status: v => (Math.abs(v.t) < 0.06 ? ['ok', 'This is the minimum-norm solution: perpendicular to the line.'] : undefined),
  takeaway: 'Every point on the line fits perfectly. The rule for choosing among them is what decides how the model generalizes.',
});

export const CurvedData = lab({
  intro: 'These points really have one degree of freedom, their position along the curve. A straight line captures them only when the curve is nearly flat.',
  params: [{ k: 'arc', label: 'how far the curve bends (degrees)', min: 10, max: 330, step: 5, init: 200 }],
  derive: ({ arc }) => { const r = rng(5), th = Array.from({ length: 60 }, (_, i) => ((i / 59 - 0.5) * arc * Math.PI) / 180), rad = 180 / Math.max(arc, 40), X = th.map(t => [rad * Math.sin(t) + 0.03 * gauss(r), -rad * Math.cos(t) + 0.03 * gauss(r)]), m = [mean(X.map(p => p[0])), mean(X.map(p => p[1]))], Z = X.map(p => [p[0] - m[0], p[1] - m[1]]), S = svd(Z), v = [S.V[0][0], S.V[1][0]], rec = Z.map(p => { const c = p[0] * v[0] + p[1] * v[1]; return [c * v[0], c * v[1]]; }), lost = sum(Z.map((p, i) => (p[0] - rec[i][0]) ** 2 + (p[1] - rec[i][1]) ** 2)) / sum(Z.map(p => p[0] ** 2 + p[1] ** 2)); return { Z, rec, lost }; },
  plots: v => { const R0 = Math.max(...v.Z.flatMap(p => p.map(Math.abs))) * 1.15; return [{ x: [-R0, R0], y: [-R0, R0], w: 380, h: 360, label: 'curved data and its best straight-line approximation', marks: [...v.Z.map((p, i) => ({ tone: 'muted', soft: true, data: [p, v.rec[i]] })), { as: 'points', tone: 1, r: 3, data: v.Z }, { as: 'points', tone: 2, r: 3, data: v.rec }] }]; },
  stats: v => [`variance lost by one component ${pct(v.lost, 0)}`],
  legend: [['--c1', 'data'], ['--c2', 'best straight-line version']],
  takeaway: 'PCA counts straight directions, so the number of components needed can exceed the true dimension of curved data.',
});

/* ---------- practical ---------- */

const SPL = (() => { const r = rng(26), xs = Array.from({ length: 40 }, (_, i) => +(0.15 * i).toFixed(2)); return xs.map(x => [x, (x < 2.2 ? 1 : x < 4.3 ? 4 : 2.2) + 0.5 * gauss(r)]); })();
const sse = a => { const m = mean(a); return sum(a.map(v => (v - m) ** 2)); };
const splitLoss = t => { const L = SPL.filter(p => p[0] <= t).map(p => p[1]), Rr = SPL.filter(p => p[0] > t).map(p => p[1]); return !L.length || !Rr.length ? sse(SPL.map(p => p[1])) : sse(L) + sse(Rr); };

export const RegressionTreeSplit = lab({
  intro: 'A regression tree predicts the mean in each leaf, and chooses the cut that most reduces the squared error.',
  tex: R`\min_t\ \sum_{x_i\le t}(y_i-\bar y_L)^2+\sum_{x_i>t}(y_i-\bar y_R)^2`,
  params: [{ k: 't', label: 'cut', min: 0.1, max: 5.7, step: 0.05, init: 1 }],
  derive: ({ t }) => { const L = SPL.filter(p => p[0] <= t).map(p => p[1]), Rr = SPL.filter(p => p[0] > t).map(p => p[1]); return { a: mean(L), b: mean(Rr) }; },
  plots: v => { const g = linspace(0.1, 5.7, 113), best = g.reduce((x, y) => (splitLoss(y) < splitLoss(x) ? y : x)); return [{ caption: 'Data and the two leaf means', x: [0, 6], y: [0, 6], w: 300, h: 260, label: 'piecewise constant fit', marks: [{ as: 'points', tone: 'fg', r: 3, data: SPL }, { tone: 1, data: [[0, v.a], [v.t, v.a]] }, { tone: 1, data: [[v.t, v.b], [6, v.b]] }, { drag: { x: 't', y: 5.6, tone: 4 } }] }, { caption: 'Total squared error by cut', x: [0, 6], y: [0, sse(SPL.map(p => p[1])) * 1.05], w: 300, h: 260, label: 'error against cut', marks: [{ tone: 2, data: g.map(x => [x, splitLoss(x)]) }, { as: 'rule', tone: 'fg', x: v.t }, { as: 'rule', dash: true, x: best }] }]; },
  stats: v => [`squared error ${fmt(splitLoss(v.t), 1)}`, `leaf means ${fmt(v.a)} and ${fmt(v.b)}`],
});

export const CategoricalSplitDrill = drill({
  intro: 'A split on a category sends each level left or right. Swapping the two sides gives the same split, and everything on one side is no split at all.',
  tex: R`\text{distinct splits}=2^{L-1}-1`,
  gen: r => ({ L: int(r, 2, 12) }),
  q: v => `A category has ${v.L} levels. How many distinct binary splits does it have?`,
  answer: v => 2 ** (v.L - 1) - 1, dp: 0,
  steps: v => [`$2^{${v.L - 1}}=${2 ** (v.L - 1)}$`, 'Subtract one for the empty split.'],
  takeaway: 'For two classes or squared error, you can sort the levels by mean target and test only the $L-1$ ordered cuts.',
});

export const MissingValues = lab({
  intro: 'When the data are missing **because of** the target, the fact that a value is missing is itself a signal. Compare three ways of handling it.',
  params: [{ k: 'pm', label: 'how much more often positives are missing', min: 0, max: 0.8, step: 0.02, init: 0.5 }, { k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ pm, seed }) => {
    const r = rng(seed * 97), make = n => Array.from({ length: n }, () => { const y = +(r() < 0.5), x = (y ? 0.8 : -0.8) + gauss(r), miss = r() < 0.1 + (y ? pm : 0); return { x, y, miss }; }), tr = make(200), te = make(600);
    const kept = tr.filter(d => !d.miss), mu = mean(kept.map(d => d.x));
    const fit = (D, f) => buildTree(D.map(f), D.map(d => d.y), { maxDepth: 3 }), acc = (t, f) => mean(te.map(d => +((predictTree(t, f(d)) > 0.5) === !!d.y)));
    const fm = d => [d.miss ? mu : d.x], fi = d => [d.miss ? mu : d.x, +d.miss];
    const drop = fit(kept, fm), dropAcc = mean(te.filter(d => !d.miss).map(d => +((predictTree(drop, fm(d)) > 0.5) === !!d.y)));
    return { rows: [['Drop rows, score complete rows only', dropAcc], ['Fill with the mean', acc(fit(tr, fm), fm)], ['Fill and add a missing flag', acc(fit(tr, fi), fi)]], cover: mean(te.map(d => +!d.miss)) };
  },
  plots: v => [{ x: [-0.6, 2.6], y: [0.5, 1], yl: 'test accuracy', label: 'accuracy of each missing-data strategy', marks: v.rows.map((row, i) => ({ as: 'rects', tone: i + 1, data: [[i - 0.32, 0.5, i + 0.32, row[1]]] })) }],
  table: v => ({ head: ['Strategy', 'Test accuracy'], rows: v.rows.map(([n, a]) => [n, pct(a, 1)]) }),
  stats: v => [`dropping rows can only score ${pct(v.cover, 0)} of new cases`],
  takeaway: 'Compute any fill value on the training data alone, or the imputer becomes a leak.',
});

export const DataAudit = drill({
  intro: 'Before modelling, look for what is wrong with the table. Name the problem.',
  gen: r => pick(r, [
    ['A column named discharge_date is used to predict length of hospital stay.', 0], ['A feature recorded after the outcome has an accuracy of 99% alone.', 0],
    ['Some patient ages are recorded as 240.', 1], ['Rainfall values include −15 mm.', 1],
    ['The same customer appears 40 times, spread across training and test.', 2], ['Exact copies of rows make up a tenth of the table.', 2],
    ['Income is blank mostly for people who later defaulted.', 3], ['Sensor readings go missing only during storms.', 3],
    ['Temperatures from one site are in Celsius, and from another in Fahrenheit.', 4], ['One lab reports lengths in millimetres, and another in metres.', 4],
  ]),
  q: v => v[0],
  choices: () => ['Leakage', 'Impossible value', 'Duplicates or repeated subjects', 'Informative missingness', 'Unit mismatch'],
  answer: v => v[1],
  steps: () => ['Ask when each value was recorded, whether it can exist, whether rows are independent, why values are absent, and whether units agree.'],
});

export const ShuffledLabels = lab({
  intro: 'A negative control: shuffle the labels and run the whole pipeline again. The test score must fall to chance. If it does not, something is leaking.',
  params: [{ k: 'd', label: 'tree depth', min: 1, max: 12, step: 1, init: 6 }, { k: 'seed', seed: 'New shuffle', init: 1 }],
  derive: ({ d, seed }) => { const D = dataset('blobs', 61, 80), T = dataset('blobs', 62, 300), r = rng(seed * 43), ys = D.map(p => [r(), p[2]]).sort((a, b) => a[0] - b[0]).map(p => p[1]), X = D.map(p => [p[0], p[1]]), real = buildTree(X, D.map(p => p[2]), { maxDepth: d }), fake = buildTree(X, ys, { maxDepth: d }); return { b: [['real train', accuracy(x => predictTree(real, x), D)], ['real test', accuracy(x => predictTree(real, x), T)], ['shuffled train', mean(X.map((x, i) => +((predictTree(fake, x) > 0.5) === !!ys[i])))], ['shuffled test', accuracy(x => predictTree(fake, x), T)]] }; },
  plots: v => [{ x: [-0.6, 3.6], y: [0, 1], yl: 'accuracy', label: 'real and shuffled labels', marks: [...v.b.map((row, i) => ({ as: 'rects', tone: i < 2 ? 1 : 2, data: [[i - 0.32, 0, i + 0.32, row[1]]] })), { as: 'rule', dash: true, y: 0.5 }, ...v.b.map((row, i) => ({ as: 'label', x: i, y: 0.04, text: row[0], anchor: 'middle' }))] }],
  stats: v => [`shuffled test ${pct(v.b[3][1], 0)}`, `shuffled train ${pct(v.b[2][1], 0)}`],
  takeaway: 'A flexible model can memorise random labels, so high training accuracy proves nothing. Only the held-out score means anything.',
});
