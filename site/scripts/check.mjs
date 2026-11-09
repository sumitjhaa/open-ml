import assert from 'node:assert/strict';
import { mean, median, polyfit, polyval, linfit, mse, rng, gauss } from '../lib/math.mjs';

const close = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);

// §12 of the notes: argmin of squared loss over 2,4,9 is the mean (5), of absolute loss the median (4).
close(mean([2, 4, 9]), 5);
close(median([2, 4, 9]), 4);
close(median([1, 7]), 4);

// Exact line is recovered.
const { w, b } = linfit([0, 1, 2, 3], [1, 3, 5, 7]);
close(w, 2); close(b, 1);

// §14: degree 9 through 10 points interpolates (training error ~ 0).
const r = rng(1);
const xs = Array.from({ length: 10 }, (_, i) => -1 + (2 * i) / 9);
const ys = xs.map(x => Math.sin(Math.PI * x) + 0.3 * gauss(r));
const c = polyfit(xs, ys, 9);
assert.ok(mse(x => polyval(c, x), xs, ys) < 1e-8, 'degree 9 should interpolate 10 points');

console.log('all checks passed');

// --- models.mjs: each model against an answer known by hand or by construction ---
const M = await import('../lib/models.mjs');
close(M.normalCdf(0), 0.5, 1e-6); close(M.normalCdf(1.96), 0.975, 1e-3);
close(M.lgamma(5), Math.log(24), 1e-8);
close(M.betaPdf(0.5, 2, 2), 1.5, 1e-6);
close(M.binomPmf(2, 4, 0.5), 0.375, 1e-8);
{ const e = M.eig2(2, 1, 2); close(e.l1, 3); close(e.l2, 1); close(Math.abs(e.v[0]), Math.SQRT1_2); }
close(M.soft(3, 1), 2); close(M.soft(-0.5, 1), 0);

// Lasso with orthonormal columns is exactly soft thresholding of the OLS coefficient.
{ const X = [[1, 1], [1, -1], [-1, 1], [-1, -1]], y = [3, 1, -1, -3], b = M.lasso(X, y, 0.25); close(b[0], 1.75); close(b[1], 0.75); }
{ const b = M.ridge([[1], [-1]], [2, -2], 1); close(b[0], 1); }
// Median behaviour: an outlier cannot move the LAD line through collinear points.
{ const { w, b } = M.ladFit([0, 1, 2, 3, 4], [0, 1, 2, 3, 40]); close(w, 1, 1e-2); close(b, 0, 1e-2); }

// Newton reaches the logistic MLE; on balanced overlap it must beat the start.
{ const X = [[1, -2], [1, -1], [1, 1], [1, 2], [1, -0.5], [1, 0.5]], y = [0, 0, 1, 1, 1, 0];
  let w = [0, 0]; const l0 = M.logLoss(w, X, y);
  for (let i = 0; i < 25; i++) w = M.logisticStep(w, X, y, 'newton');
  assert.ok(M.logLoss(w, X, y) < l0);
  const g = M.logisticStep(w, X, y, 'gd', 1); close(g[0], w[0], 1e-6); close(g[1], w[1], 1e-6); }

// ROC: a perfect ranking has AUC 1; all-tied scores have AUC 0.5.
close(M.rocCurve([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]).auc, 1);
close(M.rocCurve([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]).auc, 0.5);
close(M.rocCurve([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]).auc, 0.75);

// Hard-margin SVM on two points: the boundary is their perpendicular bisector with margin half the gap.
{ const X = [[-1, 0], [1, 0]], y = [-1, 1], s = M.smo(X, y, { C: 1000, k: M.kernels.linear() });
  close(s.f([0, 0]), 0, 1e-3); close(s.f([1, 0]), 1, 1e-3); close(s.alpha[0], 0.5, 1e-3); }
{ const D = M.dataset('xor', 2, 40), X = D.map(d => d.slice(0, 2)), y = D.map(d => (d[2] ? 1 : -1));
  const s = M.smo(X, y, { C: 10, k: M.kernels.rbf({ gamma: 1 }) });
  assert.ok(X.filter((x, i) => s.f(x) * y[i] > 0).length / X.length > 0.9, 'RBF SVM should solve XOR'); }

// Trees: a clean axis split is found at the midpoint; an impure one is not left unsplit.
{ const t = M.buildTree([[0], [1], [2], [3]], [0, 0, 1, 1], { maxDepth: 2 }); close(t.t, 1.5); close(t.gain, 0.5); close(M.predictTree(t, [3]), 1); }
close(M.impurity('entropy', 0.5), 1);
// A weighted stump must follow the weight.
{ const t = M.buildTree([[0], [1]], [1, 0], { maxDepth: 0, w: [3, 1] }); close(t.value, 0.75); }

// Boosting: training fit improves with rounds; a one-leaf constant model starts at the mean.
{ const X = M.linspace(0, 6, 30).map(v => [v]), y = X.map(([v]) => Math.sin(v));
  const g = M.gbRegress(X, y, { rounds: 100, lr: 0.3, maxDepth: 2 }), e = k => mse2(x => g.predict(x, k), X, y);
  close(g.predict(X[0], 0), mean(y)); assert.ok(e(100) < e(10) && e(10) < e(0)); }
function mse2(p, X, y) { return mean(X.map((x, i) => (p(x) - y[i]) ** 2)); }
for (const [name, fit] of [['gb', M.gbClass], ['xgb', M.xgbClass], ['bag', (X, y) => M.bagging(X, y, { B: 25 })], ['ada', (X, y) => M.adaboost(X, y, { rounds: 60 })]]) {
  const D = M.dataset('circles', 3, 60), X = D.map(d => d.slice(0, 2)), y = D.map(d => d[2]);
  assert.ok(M.accuracy(fit(X, y).predict, D) > 0.9, name + ' should fit circles');
}
// AdaBoost's defining identity: the stump just used scores exactly a coin toss on the updated weights.
{ const D = M.dataset('overlap', 4, 40), X = D.map(d => d.slice(0, 2)), y = D.map(d => d[2]), a = M.adaboost(X, y, { rounds: 3 });
  const [r0, r1] = a.rounds, h = X.map(x => M.predictTree(r0.tree, x) > 0.5 ? 1 : 0);
  close(r1.weights.reduce((s, v, i) => s + (h[i] !== y[i] ? v : 0), 0), 0.5, 1e-9); }
// XGBoost: leaf weight -G/(H+lambda) and gain formula by hand.
{ const t = M.xgbTree([[0], [1]], [-1, 1], [1, 1], { lambda: 1, maxDepth: 1 }); close(t.l.value, 0.5); close(t.gain, 0.5); }
console.log('model checks passed');
// Double descent: test error peaks at the interpolation threshold p = n and falls again well past it.
{ const d = M.doubleDescent(), peak = d.fit(d.n).test;
  assert.ok(peak > 5 * d.fit(4).test && peak > 5 * d.fit(240).test, 'double descent shape');
  assert.ok(d.fit(240).train < 1e-6, 'overparameterized model interpolates'); }
console.log('double descent check passed');

// SVD: the worked example in the notes, and reconstruction. k-means: the notes' worked example. Linkage: the notes' merge heights.
{ const d = M.svd([[3, 0], [4, 5]]); close(d.s[0], Math.sqrt(45), 1e-9); close(d.s[1], Math.sqrt(5), 1e-9);
  const a1 = d.rank(1); close(a1[0][0], 1.5, 1e-9); close(a1[1][1], 4.5, 1e-9); const a2 = d.rank(2); close(a2[1][0], 4, 1e-9); }
{ const f = M.kmeans([[1], [2], [3], [8], [9], [10]], 2, { seed: 1 }), last = f[f.length - 1]; assert.ok(last.inertia <= 4 + 1e-9, 'k-means should reach inertia 4 or better here');
  for (let i = 1; i < f.length; i++) assert.ok(f[i].inertia <= f[i - 1].inertia + 1e-9, 'inertia must not rise'); }
close(M.silhouette(2, 5), 0.6);
{ const X = [[1], [2], [4], [8]]; assert.deepEqual(M.agglomerate(X, 'single').map(m => m.h), [1, 2, 4]); assert.deepEqual(M.agglomerate(X, 'complete').map(m => m.h), [1, 3, 7]); }
// Pruning: alpha 0 keeps the tree, a huge alpha collapses it to one leaf.
{ const D = M.dataset('overlap', 6, 40), X = D.map(d => d.slice(0, 2)), y = D.map(d => d[2]), t = M.buildTree(X, y, { maxDepth: 6 });
  assert.ok(M.pruneTree(t, 0, X, y).leaves > 3); assert.equal(M.pruneTree(t, 10, X, y).leaves, 1); }
console.log('svd, clustering and pruning checks passed');

// Every number below is published in the course's unsupervised learning handbook or slides.
{ const near = (a, b, t = 0.006) => close(a, b, t);
  const F = M.kmeans(M.EIGHT_RUNS, 2, { init: [[2, 3], [1, 3]] }); [228, 51.22, 20.4, 15.25].forEach((j, i) => near(F[i].inertia, j, 0.01));
  const S = M.spreads(M.EIGHT_RUNS, F.at(-1).assign); near(S.W, 15.25); near(S.B, 85.625); near(S.T, 100.875);
  const E = M.em1d(M.LATENCIES, [15, 25], 4, 40); near(E[0].r[3][0], 0.223, 0.001); near(E[0].ll, -28.985, 0.002); near(E[1].mu[0], 12.69, 0.01); near(E[1].mu[1], 30.85, 0.01); near(E[40].mu[0], 14.21, 0.02); near(E[40].mu[1], 32.8, 0.02);
  for (let i = 1; i < E.length; i++) assert.ok(E[i].ll >= E[i - 1].ll - 1e-9, 'EM log-likelihood must not fall');
  [1, 1, 1, 2.118, 2.6, 2.609, 6.682].forEach((h, i) => near(M.agglomerate(M.EIGHT_RUNS, 'average')[i].h, h, 0.002));
  near(M.agglomerate(M.EIGHT_RUNS, 'ward').at(-1).h, 85.625);
  const e = M.eig2(6.859375, 5.4375, 5.75); near(e.l1, 11.7704, 0.0002); near(e.l2, 0.839, 0.0002);
  const C = M.covariance2(M.EIGHT_RUNS); near(M.mahalanobis([2, 8], C), 4.316, 0.002); near(M.mahalanobis([11, 11], C), 2.351, 0.002);
  const L = [10, 12, 14, 22, 31, 33, 36, 95]; near(M.robustScore(95, L), 4.2, 0.005); near((95 - M.stats1(L).m) / M.stats1(L).s, 2.464, 0.002);
  near(M.cFactor(256), 10.249, 0.001); near(M.cFactor(8), 3.436, 0.001); near(M.cFactor(2), 1, 1e-9);
  close(M.averagePrecision([1, 2]), 1); close(M.averagePrecision([2, 4]), 0.5);
  const D = M.dbscan([[0], [0.5], [1], [1.5], [10]], 0.6, 3); assert.deepEqual(D.kinds, ['border', 'core', 'core', 'border', 'noise']); assert.equal(D.clusters, 1);
  const F2 = M.isolationForest(L.map(v => [v]), { trees: 200, seed: 3 }); assert.ok(F2.score([95]) > F2.score([22]), 'outlier should score higher');
  const G = M.dataset('blobs', 3, 40), H = M.em2d(G.map(p => [p[0], p[1]]), [[-1, -1], [1, 1]], 25); for (let i = 1; i < H.length; i++) assert.ok(H[i].ll >= H[i - 1].ll - 1e-6, 'EM must not fall');
  const three = M.dataset('blobs', 5, 30), Y = M.tsne(three.map(p => [p[0], p[1]]), { perplexity: 8, iters: 300 }); assert.ok(Y.every(p => p.every(Number.isFinite)), 't-SNE finite');
}
console.log('handbook checks passed');

const mean2 = a => a.reduce((x, y) => x + y, 0) / a.length;
// Numbers from the ES 335 (2025) cheat sheets and notebooks.
{ const X = await import('../lib/extended.mjs'), near = (a, b, t = 0.006) => close(a, b, t), last = a => a.at(-1);
  near(last(X.als(X.RATINGS, 1, { sweeps: 60 })).loss, 3.36, 0.01);
  const P = last(X.als(X.RATINGS, 1, { sweeps: 60 })), pred = X.mfPredict(P.W, P.H); near(pred[1][0], 4.72, 0.01);
  const xs = [[0, 0], [0, 1], [1, 0], [1, 1]], hand = { W1: [[1, 1], [1, 1]], b1: [0, -1], w2: [1, -2], b2: 0 };
  assert.deepEqual(xs.map(x => X.mlpForward(hand, x).z2), [0, 1, 1, 0]);
  assert.equal(X.mlpParams([784, 128, 10]), 101770); assert.equal(X.mlpParams([4, 3, 2, 1]), 26);
  const t = [1, 1, 2].map(v => new X.Value(v)), z = t[1].mul(1).add(t[2].mul(2)).add(t[0]), L = z.neg().exp().add(1).inv().log().neg();
  L.backward(); near(L.data, 0.00248, 0.00001); near(t[0].grad, -0.00247, 0.00001); near(t[2].grad, -0.00495, 0.00001);
  assert.deepEqual(X.conv2d(X.IMG5, X.KERNEL3), [[12, 12, 17], [10, 17, 19], [9, 6, 14]]);
  assert.deepEqual(X.conv2d(X.IMG5, X.KERNEL3, 2, 1), [[6, 17, 3], [8, 17, 13], [6, 4, 4]]);
  assert.equal(X.LENET.reduce((a, r) => a + r[2], 0), 61706); assert.equal(X.convParams(6, 5, 1), 156); assert.equal(X.convParams(16, 5, 6), 2416); assert.equal(X.outSide(64, 3, 1, 2), 32);
  assert.deepEqual(X.pool2d(X.IMG5, 3, 1)[0], [3, 3, 3]);
  const b = X.bernoulliNb(X.SPAM.rows, X.SPAM.labels); near(b.posterior([0, 0, 1]), 0.771, 0.001);
  near(X.bernoulliNb(X.SPAM.rows, X.SPAM.labels, 1).posterior([0, 0, 1]), 0.703, 0.001);
  near(X.iou([0, 0, 70, 70], [20, 20, 90, 90]), 2500 / 7300, 0.0001);
  assert.deepEqual(X.nearest(X.TEN_POINTS, [7, 4], 2).map(p => p[0]), [2, 2]);
  near(X.qUpdate(20, -1, 0.99, 30, 0.2), 21.74, 0.0001); near(X.discounted([0, 0, 1], 0.9), 0.81, 1e-9);
  near(X.acf([1, 2, 3, 4, 5], 1), 0.4, 1e-9); near(X.rnnRun(0.8, 1, [1, 0, 0, 0, 0])[1], 0.544, 0.001);
  const ar = X.arFit(X.arSimulate(1, [0.6, 0.3], 500, 2), 2); assert.ok(Math.abs(ar[1] - 0.6) < 0.12 && Math.abs(ar[2] - 0.3) < 0.12, 'AR coefficients');
  const V = X.valueIteration(X.GRID, 0.9, 30), S = X.startOf(X.GRID); assert.ok(V.at(-1)[S[0]][S[1]] > 0, 'value reaches the start'); assert.deepEqual(V.at(-1), V.at(-2));
  const q = X.qLearn(X.GRID, { episodes: 400, seed: 4 }); assert.ok(mean2(q.returns.slice(-100)) > 0.2, 'Q-learning improves');
  const R0 = X.mountainCar(() => 2), R1 = X.mountainCar(t => (t <= 50 ? 0 : 2)); assert.equal(R0.reached, null); assert.ok(R1.reached && R1.reached < 200, 'backing up reaches the flag');
}
console.log('extended checks passed');

// Themes: fifteen of each kind, and every one readable.
{ const T = await import('../lib/themes.mjs'); assert.equal(T.LIGHT_THEMES.length, 15); assert.equal(T.DARK_THEMES.length, 15); assert.equal(new Set(T.THEMES.map(t => t.id)).size, 31);
  for (const t of [...T.LIGHT_THEMES, ...T.DARK_THEMES]) { const k = t.tokens; assert.ok(T.contrast(k.fg, k.bg) >= 7, t.name + ' text'); assert.ok(T.contrast(k.muted, k.bg) >= 4.4, t.name + ' muted'); assert.ok(T.contrast(k.onaccent, k.accent) >= 3, t.name + ' on accent'); assert.ok(T.contrast(k.accent, k.bg) >= 3, t.name + ' accent'); for (const c of ['c1', 'c2', 'c3', 'c4']) assert.ok(T.contrast(k[c], k.bg) >= 4.4, t.name + ' ' + c); } }
// Sidebar chips (.chip in globals.css): text is 45% tone with the foreground, on a 13% tint of the sidebar.
{ const { LIGHT_THEMES, DARK_THEMES, contrast, mix } = await import('../lib/themes.mjs'); for (const th of [...LIGHT_THEMES, ...DARK_THEMES]) for (const k of ['c1', 'c3', 'c4', 'fg']) { const t = th.tokens; assert.ok(contrast(mix(t[k], t.fg, 0.55), mix(t.side, t[k], 0.13)) >= 4.5, `chip contrast: ${th.name} ${k}`); } }
console.log('theme checks passed');

// Catalogue rules: section and explorer names carry no numbers, every section has at least ten explorers across all levels.
const { LECTURES } = await import('../lib/lectures.mjs');
const { hasGuide } = await import('../lib/guides.mjs');
const { STEPUPS } = await import('../lib/stepups.mjs');
for (const l of LECTURES) for (const n of l.notes) assert.ok(STEPUPS[n.file], `no step-up path: ${n.file}`);
const NUMBERED = /\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b/i;
for (const l of LECTURES) {
  assert.ok(l.explorers.length >= 10, `${l.title} has only ${l.explorers.length}`);
  assert.ok(!NUMBERED.test(l.title), `numbered section name: ${l.title}`);
  assert.equal(new Set(l.explorers.map(e => e.id)).size, l.explorers.length, `duplicate id in ${l.title}`);
  for (const lv of ['basics', 'maths', 'advanced']) assert.ok(l.explorers.some(e => e.level === lv), `${l.title} has no ${lv}`);
  for (const e of l.explorers) { assert.ok(hasGuide(e.component), `no guide: ${e.component}`); assert.ok(e.title && !NUMBERED.test(e.title), `numbered title: ${e.title}`); assert.ok(e.blurb, `no blurb: ${e.title}`); if (e.predict) assert.ok(e.predict.a < e.predict.o.length, `bad answer index: ${e.title}`); }
}
console.log('catalogue check passed:', LECTURES.length, 'sections,', LECTURES.reduce((s, l) => s + l.explorers.length, 0), 'interactive sections');
