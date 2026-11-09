'use client';
// Explorers for matrix factorisation, nearest neighbours and naive Bayes.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { blobs, fmt, linspace, memo, normalPdf, pct } from '../lib/models.mjs';
import { RATINGS, SPAM, TEN_POINTS, als, bernoulliNb, kdTree, knnVote, lpDistance, lshBuckets, mfPredict, nearest } from '../lib/extended.mjs';

/* ---------- matrix factorisation ---------- */

const FILMS = ['Sholay', 'Swades', 'Batman', 'Interstellar', 'Shawshank'], USERS = ['A', 'B', 'C'];
export const RatingDotDrill = drill({
  intro: 'A predicted rating is the dot product of a user’s taste and a film’s features.',
  tex: R`\hat a_{ij}=w_i^\top h_j`,
  gen: r => ({ w: [int(r, 1, 4), int(r, 0, 3)], h: [int(r, 0, 2), int(r, 1, 3)] }),
  q: v => `Taste $(${v.w})$ and features $(${v.h})$. Find the prediction.`,
  answer: v => v.w[0] * v.h[0] + v.w[1] * v.h[1], dp: 0,
  steps: v => [`$${v.w[0]}\\times${v.h[0]}=${v.w[0] * v.h[0]}$`, `$${v.w[1]}\\times${v.h[1]}=${v.w[1] * v.h[1]}$`],
});
export const FactorCountDrill = drill({
  intro: 'Factors need $k(n+m)$ numbers, against $nm$ cells.',
  gen: r => ({ n: pick(r, [100, 500, 1000]), m: pick(r, [50, 200, 400]), k: pick(r, [5, 10, 20]) }),
  q: v => `$n=${v.n}$ users, $m=${v.m}$ films, $k=${v.k}$. How many numbers are learned?`,
  answer: v => v.k * (v.n + v.m), dp: 0,
  steps: v => [`$n+m=${v.n + v.m}$`, `Cells in the full table: $${v.n * v.m}$`],
});
export const TasteSolveDrill = drill({
  intro: 'With films fixed, a user’s taste is a small least-squares problem. Here the features are simple, so the answer is direct.',
  tex: R`w_i=(X^\top X+\lambda I)^{-1}X^\top y`,
  gen: r => ({ a: int(r, 1, 5), b: int(r, 1, 5), l: pick(r, [0, 1]) }),
  q: v => `A user rated films with features $(1,0)$ and $(0,1)$, giving $${v.a}$ and $${v.b}$, with $\\lambda=${v.l}$. Find the first taste coefficient.`,
  answer: v => v.a / (1 + v.l), dp: 3,
  steps: v => [`$X^\\top X=I$, so the matrix to invert is $(1+${v.l})I$.`, `$X^\\top y=(${v.a},${v.b})$`],
});
const mfFrames = memo((k, lam, seed) => als(RATINGS, k, { lam, sweeps: 40, seed }));
export const MatrixCompletion = lab({
  intro: 'Step through alternating least squares on the eleven known ratings. Unknown cells get their values from the two factors.',
  tex: R`\min_{W,H}\sum_{(i,j)\in\Omega}(a_{ij}-w_i^\top h_j)^2+\lambda(\|W\|^2+\|H\|^2)`,
  params: [{ k: 'k', label: 'latent features $k$', min: 1, max: 3, step: 1, init: 1 }, { k: 'lam', label: 'penalty $\\lambda$', min: 0, max: 2, step: 0.1, init: 0 }, { k: 'step', label: 'sweep', min: 0, max: 40, step: 1, init: 0, play: true }, { k: 'seed', seed: 'New start', init: 1 }],
  derive: ({ k, lam, seed, step }) => { const F = mfFrames(k, lam, seed), f = F[step]; return { F, f, P: mfPredict(f.W, f.H) }; },
  plots: v => [{ caption: 'Loss on the known cells', x: [0, 40], y: [0, Math.max(30, v.F[1].loss * 1.1)], w: 320, h: 260, xl: 'sweep', label: 'loss by sweep', marks: [{ tone: 1, data: v.F.map((f, i) => [i, Math.min(f.loss, 1e3)]) }, { as: 'rule', dash: true, tone: 4, x: v.step }] }],
  stats: v => [`loss ${fmt(v.f.loss, 3)}`, `sweep ${v.step}`],
  table: v => ({ head: ['', ...FILMS], rows: v.P.map((row, i) => [`User ${USERS[i]}`, ...row.map((p, j) => (RATINGS[i][j] === null ? `**${fmt(p, 2)}**` : `${RATINGS[i][j]} (${fmt(p, 1)})`))]) }),
  note: 'Bold cells are guesses for ratings nobody gave. Other cells show the rating, then the fit.',
  takeaway: 'Each solve is exact, so the loss never rises. With $k=3$ and no penalty, the fit is exact but the bold cells depend on the start.',
});
const STARTS = [1, 2, 3, 4, 5, 6, 7, 8];
export const StartsDisagree = lab({
  intro: 'Eight random starts, one missing cell: user B’s rating of Sholay. Add a penalty and see the guesses agree.',
  params: [{ k: 'k', label: 'latent features $k$', min: 1, max: 3, step: 1, init: 3 }, { k: 'lam', label: 'penalty $\\lambda$', min: 0, max: 2, step: 0.05, init: 0 }],
  derive: ({ k, lam }) => { const P = STARTS.map(s => { const f = mfFrames(k, lam, s).at(-1); return { p: mfPredict(f.W, f.H)[1][0], loss: f.loss }; }), ps = P.map(q => q.p); return { P, spread: Math.max(...ps) - Math.min(...ps) }; },
  plots: v => [{ x: [0, 9], y: [-2, 12], w: 400, h: 280, xl: 'random start', yl: 'guess for user B, Sholay', label: 'guess from each start', marks: [{ as: 'rects', tone: 1, data: v.P.map((q, i) => [i + 0.65, 0, i + 1.35, Math.max(-2, Math.min(12, q.p))]) }, { as: 'rule', dash: true, tone: 'muted', y: 5 }] }],
  stats: v => [`spread of guesses ${fmt(v.spread, 2)}`, `lowest loss on known cells ${fmt(Math.min(...v.P.map(q => q.loss)), 3)}`],
  status: v => (v.spread > 1 ? ['alert', 'The known cells do not pin this down: the guess depends on the start.'] : ['ok', 'The starts agree.']),
  takeaway: 'More factors than the ratings can support fit the known cells and guess the rest arbitrarily. A penalty, or fewer factors, fixes it.',
});
export const MfCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['With $H$ fixed, finding $W$ is a least-squares problem.', true, 'The loss is linear in $W$ for fixed $H$.'],
    ['The joint problem in $W$ and $H$ is convex.', false, 'It multiplies the unknowns together, so it is bilinear.'],
    ['ALS never raises the loss.', true, 'Each step solves its sub-problem exactly.'],
    ['ALS always finds the global minimum.', false, 'It can stop at a stationary point.'],
    ['A new user with no ratings can be fitted directly.', false, 'This is the cold-start problem: their taste has no equations.'],
    ['Unknown cells must be excluded from the loss.', true, 'They have no target value.'],
  ],
  takeaway: 'Fit only what you observed, and regularise what you did not.',
});

/* ---------- nearest neighbours ---------- */

const N3 = 22, RAWK = [...blobs(3, [[-2.2, -1], [1.5, 2]], N3 / 2, 0.8), ...blobs(5, [[1.8, -1.4], [-1.5, 2.2]], N3 / 2, 0.8)], YK = RAWK.map((_, i) => +(i >= N3));
const P_OPTS = [[2, 'L2'], [1, 'L1'], ['inf', 'L∞']], pOf = p => (p === 'inf' ? Infinity : +p);
const knnField = memo((K, p, weighted) => { const nx = 48, ny = 36, g = []; for (let i = 0; i < nx; i++) { g.push([]); for (let j = 0; j < ny; j++) g[i].push(knnVote(RAWK, YK, [-5 + ((i + 0.5) * 10) / nx, -4 + ((j + 0.5) * 8) / ny], K, { p: pOf(p), weighted })); } return g; });
const fieldOf = g => (x, y) => { const v = g[Math.min(47, Math.floor(((x + 5) / 10) * 48))]?.[Math.min(35, Math.floor(((y + 4) / 8) * 36))]; return v ? 0.35 : -0.35; };
export const KnnVoteLab = lab({
  intro: 'Drag the query. Its nearest points vote, and the shading shows what every location would be labelled.',
  params: [{ k: 'K', label: 'neighbours $K$', min: 1, max: 21, step: 2, init: 3 }, { k: 'p', label: 'Distance', options: P_OPTS, init: 2 }, { k: 'w', label: 'Vote', options: [['plain', 'plain vote'], ['weighted', 'weighted by 1/d']], init: 'plain' }, { k: 'qx', label: 'query, first feature', min: -5, max: 5, step: 0.05, init: 0 }, { k: 'qy', label: 'query, second feature', min: -4, max: 4, step: 0.05, init: 0 }],
  derive: ({ K, p, w, qx, qy }) => { const near = nearest(RAWK, [qx, qy], K, pOf(p)), votes = near.reduce((s, [, i]) => s + YK[i], 0); return { near, votes, g: knnField(K, p, w === 'weighted'), cls: knnVote(RAWK, YK, [qx, qy], K, { p: pOf(p), weighted: w === 'weighted' }) }; },
  plots: v => [{ x: [-5, 5], y: [-4, 4], w: 400, h: 320, label: 'nearest neighbour regions', marks: [{ as: 'field', bands: 1, nx: 48, ny: 36, tones: [2, 1], fn: fieldOf(v.g) }, { as: 'points', r: 4, data: RAWK.map((x, i) => [x[0], x[1], YK[i] + 1]) }, ...v.near.map(([, i]) => ({ tone: 'fg', soft: true, data: [[v.qx, v.qy], RAWK[i]] })), { as: 'rings', r: 8, data: v.near.map(([, i]) => RAWK[i]) }, { drag: { x: 'qx', y: 'qy', tone: 4 } }] }],
  stats: v => [`predicted class ${v.cls ? 'orange' : 'blue'}`, `${v.votes} of ${v.K} neighbours are orange`],
  takeaway: 'Small $K$ makes islands around single points. Large $K$ smooths them away.',
});
const TESTK = [...blobs(30, [[-2.2, -1], [1.5, 2]], 60, 0.8), ...blobs(31, [[1.8, -1.4], [-1.5, 2.2]], 60, 0.8)], YT = TESTK.map((_, i) => +(i >= 120));
const KS = [1, 3, 5, 7, 9, 11, 15, 19, 25, 33, 43];
const kErrors = memo(() => KS.map(K => ({ K, train: mean(RAWK.map((x, i) => +(knnVote(RAWK, YK, x, K) !== YK[i]))), test: mean(TESTK.map((x, i) => +(knnVote(RAWK, YK, x, K) !== YT[i]))) })));
export const ChooseK = lab({
  intro: 'Training error at $K=1$ is always zero, so it can never choose $K$. Held-out error can.',
  params: [{ k: 'i', label: 'neighbours $K$', min: 0, max: 10, step: 1, init: 1, show: v => KS[v] }],
  derive: ({ i }) => { const E = kErrors(); return { E, e: E[i], g: knnField(KS[i], 2, false) }; },
  plots: v => [{ caption: 'Error against K', x: [1, 43], y: [0, 0.6], w: 320, h: 280, xl: 'K', yl: 'error', label: 'error against K', marks: [{ tone: 2, data: v.E.map(e => [e.K, e.train]) }, { tone: 4, data: v.E.map(e => [e.K, e.test]) }, { as: 'rule', dash: true, tone: 'muted', x: v.e.K }] }, { caption: 'Regions', x: [-5, 5], y: [-4, 4], w: 320, h: 280, label: 'regions at this K', marks: [{ as: 'field', bands: 1, nx: 48, ny: 36, tones: [2, 1], fn: fieldOf(v.g) }, { as: 'points', r: 3, data: RAWK.map((x, i) => [x[0], x[1], YK[i] + 1]) }] }],
  stats: v => [`test error ${pct(v.e.test, 1)}`, `training error ${pct(v.e.train, 1)}`],
  legend: [['--c2', 'training error'], ['--c4', 'held-out error']],
  takeaway: 'Few neighbours mean low bias and high variance. Pick $K$ by cross-validation.',
});
export const DistanceNormDrill = drill({
  intro: 'The three common distances differ in how they combine the coordinate gaps.',
  tex: R`d_p(a,b)=\Big(\sum_j|a_j-b_j|^p\Big)^{1/p}`,
  gen: r => ({ a: [int(r, 0, 6), int(r, 0, 6)], b: [int(r, 0, 6), int(r, 0, 6)], p: pick(r, ['1', '2', '\\infty']) }),
  q: v => `Find the $L_{${v.p}}$ distance between $(${v.a})$ and $(${v.b})$.`,
  answer: v => lpDistance(v.a, v.b, v.p === '\\infty' ? Infinity : +v.p), dp: 3,
  steps: v => [`Gaps: $${Math.abs(v.a[0] - v.b[0])}$ and $${Math.abs(v.a[1] - v.b[1])}$`, v.p === '1' ? 'Add them.' : v.p === '2' ? 'Square, add, and take the root.' : 'Take the larger one.'],
});
export const NormBalls = lab({
  intro: 'All points at distance 1 from the origin. The shape shows which directions the distance treats as near.',
  params: [{ k: 'p', label: 'exponent $p$', min: 1, max: 8, step: 0.25, init: 2 }],
  plots: v => [{ x: [-1.3, 1.3], y: [-1.3, 1.3], w: 340, h: 320, label: 'unit ball', marks: [{ tone: 1, data: linspace(0, 2 * Math.PI, 241).map(t => { const c = Math.cos(t), s = Math.sin(t), d = (Math.abs(c) ** v.p + Math.abs(s) ** v.p) ** (1 / v.p); return [c / d, s / d]; }) }, { as: 'points', tone: 4, r: 4, data: [[1, 0], [0, 1]] }] }],
  stats: v => [`$(0.7,\\,0.7)$ is at distance ${fmt((2 * 0.7 ** v.p) ** (1 / v.p), 3)}`],
  takeaway: 'At $p=1$ the ball is a diamond, at $p=2$ a circle, and as $p$ grows it approaches a square.',
});
const FRUIT = (() => { const r = rng(61); return Array.from({ length: 80 }, (_, i) => { const y = i % 2; return [+(0.35 + 0.3 * y + 0.08 * gauss(r)).toFixed(3), +(180 + 40 * gauss(r)).toFixed(1), y]; }); })();
export const ScaleMatters = lab({
  intro: 'Redness separates apples from oranges, and weight in grams carries nothing. Without scaling, grams decide who is near.',
  params: [{ k: 's', label: 'Features', options: [['raw', 'raw units'], ['std', 'standardised']], init: 'raw' }],
  derive: ({ s }) => { const m = [0, 1].map(j => mean(FRUIT.map(p => p[j]))), d = [0, 1].map(j => Math.sqrt(mean(FRUIT.map(p => (p[j] - m[j]) ** 2)))), X = FRUIT.map(p => (s === 'raw' ? [p[0], p[1]] : [(p[0] - m[0]) / d[0], (p[1] - m[1]) / d[1]])), y = FRUIT.map(p => p[2]), hit = X.map((x, i) => { const rest = X.filter((_, j) => j !== i), yr = y.filter((_, j) => j !== i); return +(knnVote(rest, yr, x, 5) === y[i]); }); return { acc: mean(hit), share: (d[1] ** 2) / (d[0] ** 2 + d[1] ** 2), rawShare: s === 'raw' }; },
  plots: () => [{ x: [0.1, 0.9], y: [60, 300], w: 360, h: 280, xl: 'redness', yl: 'weight (g)', label: 'fruit data', marks: [{ as: 'points', r: 4, data: FRUIT.map(p => [p[0], p[1], p[2] + 1]) }] }],
  stats: v => [`leave-one-out accuracy, $K=5$: ${pct(v.acc, 1)}`, v.rawShare ? `weight supplies ${pct(v.share, 1)} of the squared distance` : 'each feature now counts equally'],
  takeaway: 'Scale with statistics from the training set alone, before measuring any distance.',
});
export const KdSearch = lab({
  intro: 'A kd-tree cuts space at medians, alternating axes. Drag the query and see which points it actually compares.',
  params: [{ k: 'qx', label: 'query, first feature', min: 0, max: 10, step: 0.1, init: 7 }, { k: 'qy', label: 'query, second feature', min: 0, max: 10, step: 0.1, init: 4 }, { k: 'leaf', label: 'largest leaf', min: 2, max: 5, step: 1, init: 3 }],
  derive: ({ qx, qy, leaf }) => { const T = kdTree(TEN_POINTS, leaf), L = T.leafOf([qx, qy]), found = nearest(L.idx.map(i => TEN_POINTS[i]), [qx, qy], 1)[0], best = L.idx[found[1]], tr = nearest(TEN_POINTS, [qx, qy], 1)[0]; return { T, L, best, found: found[0], true_: tr[0], tb: tr[1] }; },
  plots: v => [{ x: [0, 10], y: [0, 10], w: 360, h: 340, label: 'kd-tree cells', marks: [...v.T.cuts.map(c => ({ tone: 'muted', data: c.axis === 0 ? [[c.value, c.box[1][0]], [c.value, c.box[1][1]]] : [[c.box[0][0], c.value], [c.box[0][1], c.value]] })), { as: 'points', tone: 'fg', r: 4, data: TEN_POINTS }, { as: 'points', tone: 3, r: 6, data: v.L.idx.map(i => TEN_POINTS[i]) }, { as: 'rings', tone: 4, r: 10, data: [TEN_POINTS[v.tb]] }, { drag: { x: 'qx', y: 'qy', tone: 4 } }] }],
  stats: v => [`compared ${v.L.idx.length} of ${TEN_POINTS.length} points`, `found distance ${fmt(v.found, 3)}, true nearest ${fmt(v.true_, 3)}`],
  status: v => (v.found > v.true_ + 1e-9 ? ['alert', 'The true nearest point is just across a boundary, so it was missed.'] : ['ok', 'The leaf held a true nearest neighbour.']),
  note: 'Green points are the leaf the query reaches. The ring marks the true nearest point.',
  takeaway: 'Trees trade exactness for speed, and backtracking across nearby boxes wins the exactness back.',
});
const LSH_PTS = (() => { const r = rng(9); return Array.from({ length: 80 }, () => [+(10 * r()).toFixed(2), +(10 * r()).toFixed(2)]); })();
const line = (p, a = 0, b = 10) => { const [wx, wy] = p.w; return Math.abs(wy) > 1e-6 ? [[a, -(wx * a + p.b) / wy], [b, -(wx * b + p.b) / wy]] : [[-p.b / wx, a], [-p.b / wx, b]]; };
export const LshBuckets = lab({
  intro: 'Random lines give each point a code of bits. Near points tend to share a code, so a query only searches its own bucket.',
  params: [{ k: 'bits', label: 'bits', min: 1, max: 8, step: 1, init: 3 }, { k: 'seed', seed: 'New planes', init: 1 }, { k: 'qx', label: 'query, first feature', min: 0, max: 10, step: 0.1, init: 5 }, { k: 'qy', label: 'query, second feature', min: 0, max: 10, step: 0.1, init: 5 }],
  derive: ({ bits, seed, qx, qy }) => { const H = lshBuckets(LSH_PTS, bits, seed * 31), q = H.code([qx, qy]), ids = LSH_PTS.map((_, i) => i).filter(i => H.code(LSH_PTS[i]) === q), tr = nearest(LSH_PTS, [qx, qy], 1)[0][1]; return { H, q, ids, tr }; },
  plots: v => [{ x: [0, 10], y: [0, 10], w: 360, h: 340, label: 'hash regions', marks: [...v.H.planes.map(p => ({ tone: 'muted', data: line(p) })), { as: 'points', tone: 'muted', r: 3, data: LSH_PTS }, { as: 'points', tone: 3, r: 5, data: v.ids.map(i => LSH_PTS[i]) }, { as: 'rings', tone: 4, r: 10, data: [LSH_PTS[v.tr]] }, { drag: { x: 'qx', y: 'qy', tone: 4 } }] }],
  stats: v => [`bucket ${v.q}: ${v.ids.length} of ${LSH_PTS.length} points`, v.ids.includes(v.tr) ? 'the true nearest point is in the bucket' : 'the true nearest point was missed'],
  takeaway: 'More bits make smaller buckets and faster search, but miss more neighbours. Several tables win them back.',
});
export const CurseVolume = lab({
  intro: 'Volume behaves strangely as dimensions grow. Slide $d$ and watch where the mass goes.',
  params: [{ k: 'd', label: 'dimensions $d$', min: 1, max: 100, step: 1, init: 10 }],
  plots: v => [{ caption: 'Share in the outer 10% shell', x: [1, 100], y: [0, 1], w: 300, h: 240, xl: 'd', label: 'outer shell share', marks: [{ tone: 1, data: curve(d => 1 - 0.9 ** d, 1, 100) }, { as: 'rule', dash: true, tone: 4, x: v.d }] }, { caption: 'Edge of a cube holding 10% of the data', x: [1, 100], y: [0, 1], w: 300, h: 240, xl: 'd', label: 'cube edge', marks: [{ tone: 2, data: curve(d => 0.1 ** (1 / d), 1, 100) }, { as: 'rule', dash: true, tone: 4, x: v.d }] }],
  stats: v => [`${pct(1 - 0.9 ** v.d, 1)} of a ball lies in its outer shell`, `a cube with 10% of the data has edge ${fmt(0.1 ** (1 / v.d), 3)}`],
  takeaway: 'A "neighbourhood" in high dimensions spans most of every axis, so nearest points are not really near.',
});
export const KnnCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['KNN does its main work at prediction time.', true, 'Training only stores the data.'],
    ['A decision tree is lazy.', false, 'It is non-parametric, but it trains up front.'],
    ['At $K=1$ training error is always zero.', true, 'Each point is its own nearest neighbour, barring exact duplicates with different labels.'],
    ['Feature scaling can change which points are nearest.', true, 'Distance adds across features.'],
    ['In very high dimensions the nearest point is far closer than the farthest.', false, 'The ratio approaches one.'],
    ['A kd-tree leaf always holds the true nearest neighbour.', false, 'It can lie just across a boundary.'],
  ],
  takeaway: 'KNN is simple. The hard parts are the features, their scale and the search.',
});

/* ---------- naive Bayes ---------- */

export const BayesRuleLab = lab({
  intro: 'A rare disease and a good test. Move the numbers and see how few positives are really sick.',
  tex: R`P(D\mid+)=\frac{P(+\mid D)P(D)}{P(+\mid D)P(D)+P(+\mid\bar D)P(\bar D)}`,
  params: [{ k: 'prev', label: 'log₁₀ prevalence', min: -5, max: -0.3, step: 0.05, init: -4 }, { k: 'sens', label: 'sensitivity', min: 0.5, max: 1, step: 0.005, init: 0.99 }, { k: 'spec', label: 'specificity', min: 0.5, max: 0.999, step: 0.001, init: 0.99 }],
  derive: ({ prev, sens, spec }) => { const p = 10 ** prev, tp = sens * p, fp = (1 - spec) * (1 - p); return { p, tp, fp, post: tp / (tp + fp) }; },
  plots: v => [{ x: [0, 3], y: [0, Math.max(v.tp, v.fp) * 1.15], w: 360, h: 260, label: 'true and false positives per person tested', marks: [{ as: 'rects', tone: 3, data: [[0.3, 0, 1.3, v.tp]] }, { as: 'rects', tone: 4, data: [[1.7, 0, 2.7, v.fp]] }, { as: 'label', x: 0.8, y: v.tp + Math.max(v.tp, v.fp) * 0.03, text: 'sick', anchor: 'middle' }, { as: 'label', x: 2.2, y: v.fp + Math.max(v.tp, v.fp) * 0.03, text: 'healthy', anchor: 'middle' }] }],
  stats: v => [`$P(\\text{sick}\\mid+)=${fmt(v.post, 4)}$`, `about one person in ${fmt(1 / v.p, 0)} is sick`],
  takeaway: 'When a condition is rare, false positives can far outnumber the true ones, even for an accurate test.',
});
export const NbCountDrill = drill({
  intro: 'Fitting a Bernoulli feature is counting within a class.',
  tex: R`P(x_j=1\mid y=c)=\frac{\text{count}(x_j=1,y=c)+\alpha}{\text{count}(y=c)+2\alpha}`,
  gen: r => { const n = int(r, 4, 12); return { n, k: int(r, 0, n), a: pick(r, [0, 1]) }; },
  q: v => `A word occurs in $${v.k}$ of $${v.n}$ spam emails, with $\\alpha=${v.a}$. Find its probability given spam.`,
  answer: v => (v.k + v.a) / (v.n + 2 * v.a), dp: 3,
  steps: v => [`Numerator: $${v.k}+${v.a}$`, `Denominator: $${v.n}+2\\times${v.a}$`],
});
const FOUR = [[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]];
export const NbScoreDrill = drill({
  intro: 'Multiply the prior with one factor per word, then normalise. This uses the course’s ten emails.',
  gen: r => ({ x: pick(r, FOUR), a: pick(r, [0, 1]) }),
  q: v => `Email $(${v.x})$, with $\\alpha=${v.a}$. Find $P(\\text{spam}\\mid x)$.`,
  answer: v => bernoulliNb(SPAM.rows, SPAM.labels, v.a).posterior(v.x), dp: 3,
  steps: v => { const m = bernoulliNb(SPAM.rows, SPAM.labels, v.a), s = m.score(v.x); return [`$\\theta_{\\text{spam}}=(${m.theta[1].map(t => fmt(t, 2))})$ and $\\theta_{\\text{ham}}=(${m.theta[0].map(t => fmt(t, 2))})$`, `Scores: spam $${fmt(s[1], 4)}$ and ham $${fmt(s[0], 4)}$`]; },
});
export const GaussianNbLab = lab({
  intro: 'One normal per class. Different spreads give a boundary with two cut points, where the wider class wins both tails.',
  params: [{ k: 'm1', label: 'mean, class A', min: 3, max: 8, step: 0.1, init: 6.1 }, { k: 's1', label: 'spread, class A', min: 0.3, max: 2, step: 0.05, init: 0.6 }, { k: 'm2', label: 'mean, class B', min: 3, max: 8, step: 0.1, init: 5.3 }, { k: 's2', label: 'spread, class B', min: 0.3, max: 2, step: 0.05, init: 0.9 }, { k: 'x', label: 'value $x$', min: 2, max: 10, step: 0.05, init: 5.5 }],
  derive: ({ m1, s1, m2, s2, x }) => { const a = normalPdf(x, m1, s1), b = normalPdf(x, m2, s2); return { a, b, post: a / (a + b) }; },
  plots: v => [{ x: [2, 10], y: [0, 1], w: 420, h: 280, xl: 'x', label: 'class densities', marks: [{ as: 'area', tone: 1, soft: true, data: linspace(2, 10, 201).map(x => [x, 0, normalPdf(x, v.m1, v.s1)]) }, { as: 'area', tone: 2, soft: true, data: linspace(2, 10, 201).map(x => [x, 0, normalPdf(x, v.m2, v.s2)]) }, { tone: 4, data: curve(x => { const a = normalPdf(x, v.m1, v.s1), b = normalPdf(x, v.m2, v.s2); return a / (a + b || 1); }, 2, 10, 201) }, { as: 'rule', dash: true, tone: 'muted', x: v.x }] }],
  stats: v => [`$P(A\\mid x)=${fmt(v.post, 3)}$`, `densities ${fmt(v.a, 3)} and ${fmt(v.b, 3)}`],
  legend: [['--c1', 'class A'], ['--c2', 'class B'], ['--c4', 'P(A | x)']],
  takeaway: 'Densities can exceed one, so compare them rather than reading them as probabilities.',
});
export const LogSpaceLab = lab({
  intro: 'Multiplying many small probabilities reaches zero in floating point. Adding logs does not.',
  params: [{ k: 'n', label: 'words in the email', min: 50, max: 800, step: 10, init: 300 }, { k: 'a', label: 'per-word probability, spam', min: 0.02, max: 0.3, step: 0.005, init: 0.12 }, { k: 'b', label: 'per-word probability, ham', min: 0.02, max: 0.3, step: 0.005, init: 0.1 }],
  derive: ({ n, a, b }) => ({ pa: a ** n, pb: b ** n, la: n * Math.log10(a), lb: n * Math.log10(b) }),
  plots: v => [{ x: [50, 800], y: [-1000, 0], w: 420, h: 280, xl: 'words', yl: 'log₁₀ of the product', label: 'log of product against length', marks: [{ tone: 1, data: curve(n => n * Math.log10(v.a), 50, 800) }, { tone: 2, data: curve(n => n * Math.log10(v.b), 50, 800) }, { as: 'rule', dash: true, tone: 4, y: -323.3 }, { as: 'rule', dash: true, tone: 'muted', x: v.n }] }],
  stats: v => [`product: spam ${v.pa === 0 ? '0' : v.pa.toExponential(2)}, ham ${v.pb === 0 ? '0' : v.pb.toExponential(2)}`, `sum of logs: ${fmt(v.la, 1)} and ${fmt(v.lb, 1)}`],
  status: v => (v.pa === 0 && v.pb === 0 ? ['alert', 'Both products underflowed to zero, so they tie. The logs still pick a winner.'] : ['ok', 'The products are still representable.']),
  takeaway: 'The red line is the smallest number a double can hold. Classify on sums of logs.',
});
const P_T = 0.01, P_E = 0.02, AL = { '11': 0.95, '10': 0.9, '01': 0.3, '00': 0.01 };
const joint = (t, e) => (t ? P_T : 1 - P_T) * (e ? P_E : 1 - P_E) * AL[`${t}${e}`];
export const ExplainingAway = lab({
  intro: 'A theft or an earthquake can set off an alarm. See how the belief in theft changes as evidence comes in.',
  params: [{ k: 'ev', label: 'What you know', options: [['none', 'nothing'], ['a', 'the alarm rang'], ['ae', 'alarm, and an earthquake occurred']], init: 'a' }],
  derive: ({ ev }) => { const f = (t, e) => joint(t, e); const num = ev === 'ae' ? f(1, 1) : f(1, 0) + f(1, 1), den = ev === 'ae' ? f(1, 1) + f(0, 1) : f(0, 0) + f(0, 1) + f(1, 0) + f(1, 1); return { post: ev === 'none' ? P_T : num / den }; },
  plots: v => [{ x: [0, 2], y: [0, 1], w: 300, h: 260, label: 'probability of theft', marks: [{ as: 'rects', tone: 1, data: [[0.4, 0, 1.6, v.post]] }] }],
  stats: v => [`$P(\\text{theft}\\mid\\text{evidence})=${fmt(v.post, 3)}$`],
  table: () => ({ head: ['theft', 'earthquake', '$P(\\text{alarm})$'], rows: [['yes', 'yes', 0.95], ['yes', 'no', 0.9], ['no', 'yes', 0.3], ['no', 'no', 0.01]] }),
  takeaway: 'The alarm raises belief in theft, and then the earthquake explains the alarm away.',
});
export const NbCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Naive Bayes assumes features are independent given the class.', true, 'That is the naive assumption.'],
    ['A zero count can veto all other evidence.', true, 'One zero factor makes the whole product zero.'],
    ['A density value above 1 is impossible.', false, 'Densities can exceed one. They integrate to one.'],
    ['Naive Bayes generally needs less data than logistic regression.', true, 'It estimates simple per-feature statistics.'],
    ['Adding logs changes which class wins.', false, 'The logarithm is increasing, so the winner is the same.'],
    ['A Bayesian network factorises the joint along its graph.', true, 'One factor per node given its parents.'],
  ],
  takeaway: 'Count, smooth, add logs, and remember the assumption.',
});
