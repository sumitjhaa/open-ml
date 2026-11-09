'use client';
// Machine learning topics that the course lectures leave out.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng, solve } from '../lib/math.mjs';
import { blobs, fmt, linspace, memo, pct } from '../lib/models.mjs';

export const AprioriDrill = drill({
  intro: 'Three numbers describe a rule "if A then B".',
  tex: R`\text{lift}=\frac{P(B\mid A)}{P(B)}`,
  gen: r => { const n = pick(r, [100, 200, 1000]), a = int(r, 2, 6) * n / 10, b = int(r, 2, 6) * n / 10, ab = Math.round(Math.min(a, b) * pick(r, [0.2, 0.4, 0.6])); return { n, a, b, ab, k: pick(r, ['confidence', 'lift']) }; },
  q: v => `$${v.n}$ baskets, $${v.a}$ contain A, $${v.b}$ contain B, $${v.ab}$ contain both. Find the ${v.k} of A ⇒ B.`,
  answer: v => (v.k === 'confidence' ? v.ab / v.a : (v.ab / v.a) / (v.b / v.n)), dp: 3,
  steps: v => [`$P(B\\mid A)=${fmt(v.ab / v.a, 3)}$`, `$P(B)=${fmt(v.b / v.n, 3)}$`],
});
const FULL = (() => { const r = rng(8); return Array.from({ length: 200 }, () => { const x = gauss(r); return [x, +(2 + 1.5 * x + 0.5 * gauss(r)).toFixed(3)]; }); })();
export const ImputationLab = lab({
  intro: 'Values go missing when $x$ is large. See how each way of handling it changes the estimated slope. The truth is 1.5.',
  params: [{ k: 'm', label: 'Handling', options: [['drop', 'drop the rows'], ['mean', 'fill with the mean'], ['reg', 'predict from the other column']], init: 'mean' }, { k: 'p', label: 'share missing among high values', min: 0, max: 1, step: 0.05, init: 0.7 }],
  derive: ({ m, p }) => { const r = rng(9), miss = FULL.map(([x]) => x > 0.3 && r() < p), obs = FULL.filter((_, i) => !miss[i]), line = D => { const a = mean(D.map(d => d[0])), b = mean(D.map(d => d[1])), s = D.reduce((t, d) => t + (d[0] - a) * (d[1] - b), 0) / D.reduce((t, d) => t + (d[0] - a) ** 2, 0); return [b - s * a, s]; }, mx = mean(obs.map(d => d[0])), [c, s] = line(obs.map(d => [d[1], d[0]])), D = m === 'drop' ? obs : FULL.map((d, i) => (miss[i] ? [m === 'mean' ? mx : c + s * d[1], d[1], 1] : d)); return { D, f: line(D), n: D.length }; },
  plots: v => [{ x: [-3, 3], y: [-3, 7], w: 460, h: 280, xl: 'x', yl: 'y', label: 'data and fitted line', marks: [{ as: 'points', r: 2.5, data: v.D.map(d => [d[0], d[1], d[2] ? 4 : 1]) }, { tone: 2, data: curve(x => v.f[0] + v.f[1] * x, -3, 3, 3) }] }],
  stats: v => [`estimated slope ${fmt(v.f[1], 3)}`, `${v.n} rows used`],
  takeaway: 'Filling with the mean flattens relationships, and dropping can bias them when missingness depends on the value.',
});
export const LeakTruths = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Target encoding computed on all the data leaks the label.', true, 'Compute it out of fold.'],
    ['Imputation statistics should come from training data only.', true, 'Otherwise the test set influences training.'],
    ['An indicator for missingness can be useful.', true, 'Whether a value is missing can carry information.'],
    ['One-hot encoding always suits features with thousands of categories.', false, 'It grows too wide, so use hashing or embeddings.'],
  ],
  takeaway: 'Every preprocessing step belongs inside cross-validation.',
});
export const VcDrill = drill({
  intro: 'A linear classifier in $d$ dimensions can shatter at most $d+1$ points.',
  gen: r => ({ d: int(r, 1, 20) }),
  q: v => `What is the VC dimension of a linear classifier in $${v.d}$ dimensions?`,
  answer: v => v.d + 1, dp: 0,
  steps: () => ['Weights in each dimension, plus one for the bias.'],
});
export const PacBoundDrill = drill({
  intro: 'How many examples are enough to rule out bad hypotheses in a finite class?',
  tex: R`n\ge\frac1\varepsilon\Big(\ln|H|+\ln\frac1\delta\Big)`,
  gen: r => ({ h: pick(r, [100, 1000, 1e6]), e: pick(r, [0.1, 0.05, 0.01]), d: pick(r, [0.05, 0.01]) }),
  q: v => `$|H|=${v.h}$, $\\varepsilon=${v.e}$, $\\delta=${v.d}$. Find the bound on $n$.`,
  answer: v => (Math.log(v.h) + Math.log(1 / v.d)) / v.e, dp: 1,
  steps: v => [`$\\ln|H|=${fmt(Math.log(v.h), 3)}$`, `$\\ln(1/\\delta)=${fmt(Math.log(1 / v.d), 3)}$`],
});
export const ChiSquareDrill = drill({
  intro: 'Compare what you observed with what you would expect.',
  tex: R`\chi^2=\sum\frac{(O-E)^2}{E}`,
  gen: r => { const e = pick(r, [10, 20, 50]); return { e, a: e + int(r, -8, 8), b: e + int(r, -8, 8) }; },
  q: v => `Observed $${v.a}$ and $${v.b}$, with $${v.e}$ expected in each. Find $\\chi^2$.`,
  answer: v => ((v.a - v.e) ** 2 + (v.b - v.e) ** 2) / v.e, dp: 3,
  steps: v => [`Squared differences: $${(v.a - v.e) ** 2}$ and $${(v.b - v.e) ** 2}$`],
});
export const KaplanMeierDrill = drill({
  intro: 'Multiply the fraction that survives each event time.',
  tex: R`S(t)=\prod_{t_i\le t}\Big(1-\frac{d_i}{n_i}\Big)`,
  gen: r => { const n1 = int(r, 8, 20), d1 = int(r, 1, 3), n2 = n1 - d1 - int(r, 0, 2), d2 = int(r, 1, 3); return { n1, d1, n2, d2 }; },
  q: v => `$${v.n1}$ at risk with $${v.d1}$ events, then $${v.n2}$ at risk with $${v.d2}$ events. Find the survival.`,
  answer: v => (1 - v.d1 / v.n1) * (1 - v.d2 / v.n2), dp: 3,
  steps: v => [`$1-${v.d1}/${v.n1}=${fmt(1 - v.d1 / v.n1, 3)}$`, `$1-${v.d2}/${v.n2}=${fmt(1 - v.d2 / v.n2, 3)}$`],
});
export const HammingDrill = drill({
  intro: 'In multi-label problems, each label slot is right or wrong on its own.',
  gen: r => ({ t: Array.from({ length: 6 }, () => int(r, 0, 1)), p: Array.from({ length: 6 }, () => int(r, 0, 1)) }),
  q: v => `True labels $${v.t.join('')}$ and predicted $${v.p.join('')}$. Find the Hamming loss.`,
  answer: v => v.t.filter((b, i) => b !== v.p[i]).length / 6, dp: 3,
  steps: v => [`$${v.t.filter((b, i) => b !== v.p[i]).length}$ of 6 slots differ.`],
});
const nullStats = memo(() => { const r = rng(90); return Array.from({ length: 4000 }, () => { const m1 = mean(Array.from({ length: 30 }, () => gauss(r))), m2 = mean(Array.from({ length: 30 }, () => gauss(r))); return (m1 - m2) / Math.sqrt(2 / 30); }); });
export const PValueLab = lab({
  intro: 'Differences you would see if nothing were going on. The p-value is the shaded share at least as extreme as yours.',
  params: [{ k: 't', label: 'observed statistic', min: 0, max: 4, step: 0.05, init: 2 }],
  derive: ({ t }) => { const N = nullStats(); return { N, p: mean(N.map(v => +(Math.abs(v) >= t))) }; },
  plots: v => { const b = Array(60).fill(0); v.N.forEach(x => { const i = Math.floor((x + 4) / (8 / 60)); if (i >= 0 && i < 60) b[i]++; }); return [{ x: [-4, 4], y: [0, 240], w: 480, h: 260, xl: 'statistic under the null', label: 'null distribution', marks: [{ as: 'rects', tone: 'muted', soft: true, data: b.map((c, i) => [-4 + (i * 8) / 60, 0, -4 + ((i + 0.9) * 8) / 60, c]) }, { as: 'rects', tone: 4, data: b.flatMap((c, i) => (Math.abs(-4 + ((i + 0.5) * 8) / 60) >= v.t ? [[-4 + (i * 8) / 60, 0, -4 + ((i + 0.9) * 8) / 60, c]] : [])) }] }]; },
  stats: v => [`p-value ${fmt(v.p, 4)}`],
  takeaway: 'It is the probability of data this extreme if nothing is going on. It is not the probability that nothing is going on.',
});
const A0 = blobs(13, [[-1.2, -0.6]], 60, 0.7).map(p => [p[0] + 0.5 * p[1], p[1]]), A1 = blobs(14, [[1.2, 0.9]], 60, 0.7).map(p => [p[0] + 0.5 * p[1], p[1]]);
const stats = X => { const m = [mean(X.map(p => p[0])), mean(X.map(p => p[1]))], f = (i, j) => mean(X.map(p => (p[i] - m[i]) * (p[j] - m[j]))); return { m, S: [[f(0, 0), f(0, 1)], [f(0, 1), f(1, 1)]] }; };
export const LdaLab = lab({
  intro: 'Fit a Gaussian per class with shared spread, and the boundary is a line. Skew the data and see the direction turn away from the line joining the means.',
  tex: R`w=\Sigma^{-1}(\mu_1-\mu_0)`,
  params: [{ k: 'p', label: 'prior of class one', min: 0.05, max: 0.95, step: 0.05, init: 0.5 }],
  derive: ({ p }) => { const a = stats(A0), b = stats(A1), S = a.S.map((row, i) => row.map((v, j) => (v + b.S[i][j]) / 2)), d = [b.m[0] - a.m[0], b.m[1] - a.m[1]], w = solve(S, d), mid = [(a.m[0] + b.m[0]) / 2, (a.m[1] + b.m[1]) / 2], c = w[0] * mid[0] + w[1] * mid[1] - Math.log(p / (1 - p)); return { a, b, w, c, acc: (A0.filter(x => w[0] * x[0] + w[1] * x[1] < c).length + A1.filter(x => w[0] * x[0] + w[1] * x[1] >= c).length) / 120 }; },
  plots: v => [{ x: [-5, 5], y: [-4, 4], w: 460, h: 320, label: 'linear discriminant boundary', marks: [{ as: 'field', bands: 1, tones: [2, 1], fn: (x, y) => (v.w[0] * x + v.w[1] * y >= v.c ? 0.3 : -0.3) }, { as: 'points', tone: 1, r: 3.5, data: A0 }, { as: 'points', tone: 2, r: 3.5, data: A1 }, { as: 'rings', r: 8, data: [v.a.m, v.b.m] }] }],
  stats: v => [`accuracy ${pct(v.acc, 1)}`, `direction $(${fmt(v.w[0], 2)},\\ ${fmt(v.w[1], 2)})$`],
  takeaway: 'Accounting for the shared spread rotates the boundary, and the prior only slides it.',
});
const SX = linspace(0, 10, 50), SY = (() => { const r = rng(15); return SX.map(x => +(Math.sin(x) + 0.08 * x * x - 0.5 + 0.25 * gauss(r)).toFixed(3)); })();
const splineFit = k => { const knots = Array.from({ length: k }, (_, i) => ((i + 1) * 10) / (k + 1)), basis = x => [1, x, x * x, x ** 3, ...knots.map(t => Math.max(0, x - t) ** 3)], X = SX.map(basis), n = X[0].length, G = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => X.reduce((s, r0) => s + r0[a] * r0[b], 0) + (a === b ? 1e-6 : 0))), c = solve(G, Array.from({ length: n }, (_, a) => X.reduce((s, r0, i) => s + r0[a] * SY[i], 0))); return { knots, f: x => basis(x).reduce((s, b, i) => s + b * c[i], 0) }; };
export const SplineLab = lab({
  intro: 'Polynomial pieces joined smoothly at knots. More knots follow the data more closely, and eventually follow the noise.',
  params: [{ k: 'k', label: 'knots', min: 0, max: 20, step: 1, init: 4 }],
  derive: ({ k }) => { const S = splineFit(k); return { ...S, mse: mean(SX.map((x, i) => (S.f(x) - SY[i]) ** 2)) }; },
  plots: v => [{ x: [0, 10], y: [-3, 9], w: 500, h: 280, xl: 'x', label: 'spline fit', marks: [{ as: 'points', tone: 'fg', r: 3, data: SX.map((x, i) => [x, SY[i]]) }, { tone: 1, data: curve(x => Math.max(-3, Math.min(9, v.f(x))), 0, 10, 301) }, ...v.knots.map(t => ({ as: 'rule', dash: true, tone: 'muted', x: t }))] }],
  stats: v => [`training error ${fmt(v.mse, 4)}`],
  takeaway: 'Knots control flexibility, just as depth controls a tree.',
});
const IMB = [...blobs(21, [[-0.5, 0]], 300, 1), ...blobs(22, [[1.6, 1]], 15, 0.5)], YIMB = IMB.map((_, i) => +(i >= 300));
const fitW = memo(w1 => { let a = 0, b = 0, c = 0; for (let t = 0; t < 500; t++) { let ga = 0, gb = 0, gc = 0, W = 0; IMB.forEach(([x, y], i) => { const wt = YIMB[i] ? w1 : 1, e = 1 / (1 + Math.exp(-(a * x + b * y + c))) - YIMB[i]; ga += wt * e * x; gb += wt * e * y; gc += wt * e; W += wt; }); a = +(a - (0.6 * ga) / W).toFixed(6); b = +(b - (0.6 * gb) / W).toFixed(6); c = +(c - (0.6 * gc) / W).toFixed(6); } return [a, b, c]; });
export const ImbalanceLab = lab({
  intro: 'Fifteen positives among three hundred negatives. Give positives more weight and see what recall and precision do.',
  params: [{ k: 'w', label: 'weight on positives', min: 1, max: 40, step: 1, init: 1 }],
  derive: ({ w }) => { const c = fitW(w), pred = IMB.map(([x, y]) => +(c[0] * x + c[1] * y + c[2] > 0)), tp = pred.filter((p, i) => p && YIMB[i]).length, fp = pred.filter((p, i) => p && !YIMB[i]).length; return { c, tp, fp, rec: tp / 15, prec: tp + fp ? tp / (tp + fp) : 0, acc: mean(pred.map((p, i) => +(p === YIMB[i]))) }; },
  plots: v => [{ x: [-4, 4], y: [-4, 4], w: 460, h: 320, label: 'weighted classifier', marks: [{ as: 'field', bands: 1, tones: [2, 1], fn: (x, y) => (v.c[0] * x + v.c[1] * y + v.c[2] > 0 ? 0.3 : -0.3) }, { as: 'points', r: 3, data: IMB.map((p, i) => [p[0], p[1], YIMB[i] ? 4 : 1]) }] }],
  stats: v => [`recall ${pct(v.rec, 0)}`, `precision ${pct(v.prec, 0)}`, `accuracy ${pct(v.acc, 1)}`],
  takeaway: 'Accuracy hardly moves, while recall and precision trade against each other.',
});
const ICA = (() => { const r = rng(40), n = 400; return Array.from({ length: n }, (_, t) => { const s1 = Math.sin(t / 7) > 0 ? 1 : -1, s2 = 2 * r() - 1; return [s1, s2, 0.8 * s1 + 0.6 * s2, 0.4 * s1 - 0.9 * s2]; }); })();
const whiten = (() => { const X = ICA.map(p => [p[2], p[3]]), m = [mean(X.map(p => p[0])), mean(X.map(p => p[1]))], a = mean(X.map(p => (p[0] - m[0]) ** 2)), b = mean(X.map(p => (p[0] - m[0]) * (p[1] - m[1]))), d = mean(X.map(p => (p[1] - m[1]) ** 2)), tr = a + d, det = a * d - b * b, sq = Math.sqrt(Math.max(tr * tr / 4 - det, 0)), l1 = tr / 2 + sq, l2 = tr / 2 - sq, th = 0.5 * Math.atan2(2 * b, a - d), c = Math.cos(th), s = Math.sin(th); return X.map(p => { const x = p[0] - m[0], y = p[1] - m[1]; return [(c * x + s * y) / Math.sqrt(l1), (-s * x + c * y) / Math.sqrt(l2)]; }); })();
const kurt = deg => { const t = (deg * Math.PI) / 180, z = whiten.map(p => Math.cos(t) * p[0] + Math.sin(t) * p[1]); return mean(z.map(v => v ** 4)) - 3; };
export const IcaLab = lab({
  intro: 'Two signals were mixed. Rotate the whitened data until the projection looks least Gaussian.',
  tex: R`\max_u\ |\text{kurtosis}(u^\top z)|`,
  params: [{ k: 'a', label: 'rotation (degrees)', min: 0, max: 180, step: 1, init: 20 }],
  derive: ({ a }) => { const t = (a * Math.PI) / 180, z = whiten.map(p => Math.cos(t) * p[0] + Math.sin(t) * p[1]), best = linspace(0, 180, 181).reduce((m, d) => (Math.abs(kurt(d)) > Math.abs(kurt(m)) ? d : m), 0); return { z, k: kurt(a), best }; },
  plots: v => [{ caption: 'Whitened mixtures', x: [-3, 3], y: [-3, 3], w: 300, h: 260, label: 'whitened scatter', marks: [{ as: 'points', tone: 'muted', r: 2, data: whiten }, { as: 'arrow', tone: 4, data: [[0, 0, 2.6 * Math.cos((v.a * Math.PI) / 180), 2.6 * Math.sin((v.a * Math.PI) / 180)]] }] }, { caption: 'Recovered signal', x: [0, 120], y: [-3, 3], w: 320, h: 260, xl: 'time', label: 'projected signal', marks: [{ tone: 1, data: v.z.slice(0, 120).map((x, i) => [i, x]) }] }],
  stats: v => [`kurtosis ${fmt(v.k, 3)}`, `most non-Gaussian at about ${fmt(v.best, 0)}°`],
  takeaway: 'Mixtures look more Gaussian than their sources, so separating them means finding the least Gaussian directions.',
});
const STATES = ['sunny', 'rainy'], OBS = ['walk', 'shop', 'clean'], PI = [0.6, 0.4], TR = [[0.7, 0.3], [0.4, 0.6]], EM = [[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]];
const viterbi = o => { let d = STATES.map((_, s) => PI[s] * EM[s][o[0]]); const back = []; for (let t = 1; t < o.length; t++) { const b = [], n = STATES.map((_, s) => { const c = STATES.map((__, p) => d[p] * TR[p][s]), best = c.indexOf(Math.max(...c)); b.push(best); return c[best] * EM[s][o[t]]; }); d = n; back.push(b); } let s = d.indexOf(Math.max(...d)); const path = [s]; for (let t = back.length - 1; t >= 0; t--) { s = back[t][s]; path.unshift(s); } return { path, p: Math.max(...d) }; };
export const ViterbiLab = lab({
  intro: 'Only activities are visible, and the weather is hidden. Change the sequence and see the most likely weather.',
  params: [0, 1, 2, 3].map(i => ({ k: `o${i}`, label: `Day ${['A', 'B', 'C', 'D'][i]}`, options: OBS.map((o, j) => [j, o]), init: [0, 1, 2, 2][i] })),
  derive: v => { const o = [0, 1, 2, 3].map(i => +v[`o${i}`]); return { o, ...viterbi(o) }; },
  table: v => ({ head: ['day', 'seen', 'most likely weather'], rows: v.o.map((x, i) => [['A', 'B', 'C', 'D'][i], OBS[x], STATES[v.path[i]]]) }),
  stats: v => [`probability of the best path ${v.p.toExponential(3)}`],
  plots: () => [],
  takeaway: 'Dynamic programming finds the best of exponentially many paths in linear time.',
});
const LOSSES = (() => { const r = rng(61); return Array.from({ length: 300 }, t => [0, 1, 2, 3].map(k => (r() < [0.45, 0.5, 0.3, 0.55][k] ? 1 : 0))); })();
const hedge = memo(e => { let w = [1, 1, 1, 1], tot = 0; const best = [0, 0, 0, 0], reg = []; LOSSES.forEach((l, t) => { const W = w.reduce((a, b) => a + b, 0); tot += w.reduce((s, v, k) => s + (v / W) * l[k], 0); l.forEach((x, k) => { best[k] += x; }); w = w.map((v, k) => v * Math.exp(-e * l[k])); const M = Math.max(...w); w = w.map(v => v / M); reg.push([t + 1, tot - Math.min(...best)]); }); return { reg, w }; });
export const HedgeLab = lab({
  intro: 'Four experts make predictions over time. Shift trust towards whoever is doing well, and see the regret.',
  tex: R`w_k\leftarrow w_ke^{-\eta\ell_k}`,
  params: [{ k: 'e', label: 'learning rate $\\eta$', min: 0.01, max: 2, step: 0.01, init: 0.2 }],
  derive: ({ e }) => hedge(e),
  plots: v => [{ x: [0, 300], y: [-5, 40], w: 480, h: 260, xl: 'round', yl: 'regret', label: 'regret by round', marks: [{ tone: 1, data: v.reg.filter((_, i) => i % 3 === 0) }] }],
  stats: v => [`final regret ${fmt(v.reg.at(-1)[1], 2)}`, `final trust ${v.w.map(x => fmt(x / v.w.reduce((a, b) => a + b, 0), 2)).join(', ')}`],
  takeaway: 'Regret grows slower than time, so the average loss approaches that of the best expert.',
});
const SSL = (() => { const r = rng(71); return Array.from({ length: 300 }, (_, i) => { const y = i % 2; return [+((y ? 1.2 : -1.2) + 0.7 * gauss(r)).toFixed(3), y]; }); })();
const fit1 = D => { let a = 0, b = 0; for (let t = 0; t < 300; t++) { let ga = 0, gb = 0; D.forEach(([x, y]) => { const e = 1 / (1 + Math.exp(-(a * x + b))) - y; ga += e * x; gb += e; }); a = +(a - (0.3 * ga) / D.length).toFixed(6); b = +(b - (0.3 * gb) / D.length).toFixed(6); } return [a, b]; };
const selfTrain = memo(thr => { let L = [[-1.6, 0], [-0.3, 0], [1.9, 1], [0.8, 1]], U = SSL.map(s => s[0]); const hist2 = []; for (let t = 0; t < 8; t++) { const [a, b] = fit1(L); hist2.push(mean(SSL.map(([x, y]) => +(+(a * x + b > 0) === y)))); const conf = U.map(x => 1 / (1 + Math.exp(-(a * x + b)))); L = [...L, ...U.flatMap((x, i) => (conf[i] > thr ? [[x, 1]] : conf[i] < 1 - thr ? [[x, 0]] : []))]; U = U.filter((_, i) => conf[i] <= thr && conf[i] >= 1 - thr); } return hist2; });
export const SelfTrainLab = lab({
  intro: 'Start from four labels. Repeatedly add the unlabelled points the model is most sure of.',
  params: [{ k: 't', label: 'confidence threshold', min: 0.5, max: 0.99, step: 0.01, init: 0.9 }],
  derive: ({ t }) => ({ H: selfTrain(t) }),
  plots: v => [{ x: [0, 7], y: [0.5, 1], w: 480, h: 260, xl: 'round', yl: 'accuracy', label: 'accuracy by round', marks: [{ tone: 1, data: v.H.map((a, i) => [i, a]) }, { as: 'points', tone: 4, r: 5, data: v.H.map((a, i) => [i, a]) }] }],
  stats: v => [`starting accuracy ${pct(v.H[0], 1)}`, `final accuracy ${pct(v.H.at(-1), 1)}`],
  takeaway: 'Unlabelled data can help, but confident mistakes can reinforce themselves.',
});
const NM = [[5, 3, 0, 1], [4, 2, 0, 1], [1, 1, 0, 5], [0, 0, 5, 4], [0, 1, 4, 4]];
const nmfRun = memo(k => { const r = rng(k * 3 + 2), n = NM.length, m = NM[0].length; let W = Array.from({ length: n }, () => Array.from({ length: k }, () => 0.2 + r())), H = Array.from({ length: k }, () => Array.from({ length: m }, () => 0.2 + r())); const mul = (A, B) => A.map(a => B[0].map((_, j) => a.reduce((s, v, i) => s + v * B[i][j], 0))), tr = A => A[0].map((_, j) => A.map(a => a[j])); for (let t = 0; t < 300; t++) { const WT = tr(W), N = mul(WT, NM), D = mul(mul(WT, W), H); H = H.map((row, i) => row.map((v, j) => +((v * N[i][j]) / (D[i][j] + 1e-9)).toFixed(8))); const N2 = mul(NM, tr(H)), D2 = mul(W, mul(H, tr(H))); W = W.map((row, i) => row.map((v, j) => +((v * N2[i][j]) / (D2[i][j] + 1e-9)).toFixed(8))); } const P = mul(W, H); return { H, P, err: mean(P.flatMap((row, i) => row.map((v, j) => (v - NM[i][j]) ** 2))) }; });
export const NmfLab = lab({
  intro: 'A word-count table factored into non-negative parts. Each part is a combination of words, which reads like a topic.',
  tex: R`A\approx WH,\quad W,H\ge0`,
  params: [{ k: 'k', label: 'parts', min: 1, max: 4, step: 1, init: 2 }],
  derive: ({ k }) => nmfRun(k),
  table: v => ({ head: ['part', 'ball', 'goal', 'code', 'bug'], rows: v.H.map((row, i) => [`part ${i + 1}`, ...row.map(x => fmt(x, 2))]) }),
  stats: v => [`mean squared error ${fmt(v.err, 4)}`],
  plots: () => [],
  takeaway: 'Because nothing can be subtracted, the factors tend to look like parts of a whole.',
});
export const MoreMlCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A p-value is the probability that the null hypothesis is true.', false, 'It is the probability of data at least this extreme if the null hypothesis holds.'],
    ['LDA with shared covariance gives a linear boundary.', true, 'The quadratic terms cancel.'],
    ['ICA looks for uncorrelated directions.', false, 'It seeks independent ones, by going beyond correlation.'],
    ['A multi-label model should use one sigmoid per label.', true, 'The labels are not exclusive.'],
    ['Censored subjects can simply be dropped in survival analysis.', false, 'That biases the estimate. Kaplan-Meier uses them while at risk.'],
  ],
  takeaway: 'Each method answers a particular kind of question.',
});
