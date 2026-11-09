'use client';
// Explorers for topics beyond the source courses: statistical modelling, decisions under uncertainty, structured inference,
// robust deployment, score-based generation, theory and science, and control, 3D and interpretability.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng, solve } from '../lib/math.mjs';
import { fmt, linspace, normalCdf, normalPdf, pct } from '../lib/models.mjs';
import { softmaxT } from '../lib/extended.mjs';

const r6 = x => +x.toFixed(6);
const hist = (xs, a, b, n) => { const c = Array(n).fill(0), w = (b - a) / n; xs.forEach(x => { const i = Math.floor((x - a) / w); if (i >= 0 && i < n) c[i]++; }); return c.map((v, i) => [a + i * w, 0, a + (i + 0.85) * w, v]); };
const std = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const sigmoid = z => 1 / (1 + Math.exp(-z));
const clip = (x, a, b) => Math.max(a, Math.min(b, x));
const probit = p => { let a = -8, b = 8; for (let i = 0; i < 60; i++) { const m = (a + b) / 2; if (normalCdf(m) < p) a = m; else b = m; } return (a + b) / 2; };
const factorial = n => (n < 2 ? 1 : n * factorial(n - 1));
const CHECK = 'Decide whether each statement is true, then read why.';

/* ---------- statistical modelling ---------- */
export const SmGlmLab = lab({
  intro: 'One linear score, three ways of turning it into a valid mean.',
  tex: R`g(\mathbb E[y\mid x])=wx+b`,
  params: [{ k: 'f', label: 'Family', init: 'log', options: [['id', 'Identity'], ['logit', 'Logit'], ['log', 'Log']] }, { k: 'w', label: 'slope $w$', min: -2, max: 2, step: 0.05, init: 0.8 }, { k: 'b', label: 'intercept $b$', min: -2, max: 2, step: 0.05, init: 0 }],
  derive: ({ f, w, b }) => ({ m: x => { const e = w * x + b; return f === 'id' ? e : f === 'logit' ? sigmoid(e) : Math.exp(e); }, range: f === 'logit' ? [0, 1] : f === 'log' ? [0, 12] : [-8, 8] }),
  plots: v => [{ x: [-3, 3], y: v.range, w: 480, h: 260, xl: 'x', yl: 'mean of y', label: 'mean against x', marks: [{ tone: 1, data: curve(x => clip(v.m(x), v.range[0], v.range[1]), -3, 3) }] }],
  stats: v => [`mean at x = 1: ${fmt(v.m(1), 3)}`, `at x = 2: ${fmt(v.m(2), 3)}`],
  takeaway: 'The link keeps the mean valid, so counts stay positive and probabilities stay between 0 and 1.',
});
export const SmPoissonLab = lab({
  intro: 'Counts from a Poisson distribution have a variance equal to their mean.',
  tex: R`P(y)=\frac{\lambda^ye^{-\lambda}}{y!}`,
  params: [{ k: 'l', label: 'rate $\\lambda$', min: 0.2, max: 15, step: 0.1, init: 3 }],
  plots: v => [{ x: [0, 25], y: [0, 0.9], w: 480, h: 260, xl: 'count', yl: 'probability', label: 'Poisson probabilities', marks: [{ as: 'rects', tone: 1, data: Array.from({ length: 25 }, (_, y) => [y + 0.1, 0, y + 0.9, Math.exp(y * Math.log(v.l) - v.l - Math.log(factorial(y)))]) }] }],
  stats: v => [`mean ${fmt(v.l, 2)}`, `variance ${fmt(v.l, 2)}`, `P(zero) ${pct(Math.exp(-v.l), 1)}`],
  takeaway: 'If your data have a variance well above the mean, a Poisson model is too confident.',
});
export const SmOverdispLab = lab({
  intro: 'A negative binomial adds a term that lets the variance grow faster than the mean.',
  tex: R`\text{Var}=\mu+\alpha\mu^2`,
  params: [{ k: 'a', label: 'dispersion $\\alpha$', min: 0, max: 2, step: 0.02, init: 0.3 }, { k: 'm', label: 'mean $\\mu$', min: 0.5, max: 20, step: 0.5, init: 8 }],
  plots: v => [{ x: [0, 20], y: [0, 300], w: 480, h: 260, xl: 'mean', yl: 'variance', label: 'variance against mean', marks: [{ tone: 'muted', dash: true, data: curve(m => m, 0, 20, 3) }, { tone: 1, data: curve(m => Math.min(300, m + v.a * m * m), 0, 20) }, { as: 'points', tone: 4, r: 5, data: [[v.m, Math.min(300, v.m + v.a * v.m * v.m)]] }] }],
  stats: v => [`variance ${fmt(v.m + v.a * v.m * v.m, 2)}`, `standard errors understated by × ${fmt(Math.sqrt(1 + v.a * v.m), 2)}`],
  takeaway: 'A Poisson fit to overdispersed counts gives standard errors that are too small.',
});
const QD = (() => { const r = rng(9); return Array.from({ length: 200 }, () => r6(2 + 1.5 * gauss(r) + (r() < 0.2 ? 3 : 0))); })().sort((a, b) => a - b);
const pinball = (e, t) => (e >= 0 ? t * e : (t - 1) * e);
export const SmQuantileLab = lab({
  intro: 'Drag the prediction to minimise the pinball loss on a fixed sample. The minimum lands on the quantile.',
  tex: R`\rho_\tau(e)=\max(\tau e,(\tau-1)e)`,
  params: [{ k: 't', label: 'quantile $\\tau$', min: 0.05, max: 0.95, step: 0.01, init: 0.9 }, { k: 'q', label: 'prediction $q$', min: -2, max: 9, step: 0.05, init: 2 }],
  derive: ({ t }) => ({ L: q => mean(QD.map(y => pinball(y - q, t))), best: QD[Math.min(199, Math.floor(t * 200))] }),
  plots: v => [{ x: [-2, 9], y: [0, 4], w: 480, h: 260, xl: 'prediction', yl: 'mean pinball loss', label: 'pinball loss', marks: [{ tone: 1, data: curve(q => Math.min(4, v.L(q)), -2, 9) }, { as: 'rule', dash: true, tone: 4, x: v.q }, { as: 'rule', dash: true, tone: 2, x: v.best }] }],
  stats: v => [`loss ${fmt(v.L(v.q), 3)}`, `share below ${pct(QD.filter(y => y <= v.q).length / 200, 0)}`],
  takeaway: 'Asymmetric penalties make the best constant a quantile of the data.',
});
export const SmIntervalLab = lab({
  intro: 'Noise grows with $x$. A band made of two quantile lines widens automatically.',
  params: [{ k: 'lo', label: 'lower $\\tau$', min: 0.01, max: 0.49, step: 0.01, init: 0.1 }, { k: 'hi', label: 'upper $\\tau$', min: 0.51, max: 0.99, step: 0.01, init: 0.9 }, { k: 's', label: 'noise growth', min: 0, max: 1, step: 0.05, init: 0.5 }, { k: 'seed', seed: 'New data', init: 1 }],
  derive: ({ s, seed, lo, hi }) => { const r = rng(seed * 19), P = Array.from({ length: 160 }, () => { const x = r() * 4; return [r6(x), r6(2 * x + (0.2 + s * x) * gauss(r))]; }), zl = probit(lo), zh = probit(hi), f = (x, z) => 2 * x + (0.2 + s * x) * z; return { P, zl, zh, f, cover: P.filter(([x, y]) => y >= f(x, zl) && y <= f(x, zh)).length / 160 }; },
  plots: v => [{ x: [0, 4], y: [-6, 14], w: 480, h: 280, xl: 'x', yl: 'y', label: 'data and quantile lines', marks: [{ as: 'points', tone: 'muted', r: 2, data: v.P }, { tone: 1, data: curve(x => clip(v.f(x, v.zh), -6, 14), 0, 4, 3) }, { tone: 2, data: curve(x => clip(v.f(x, v.zl), -6, 14), 0, 4, 3) }] }],
  stats: v => [`coverage ${pct(v.cover, 0)}`, `target ${pct(v.hi - v.lo, 0)}`],
  takeaway: 'Quantile lines capture changing noise, which a single standard deviation cannot.',
});
export const SmKalmanLab = lab({
  intro: 'A hidden value drifts, and is measured with noise. Tell the filter how much it should trust each.',
  tex: R`K=\frac{P^-}{P^-+R}`,
  params: [{ k: 'q', label: 'process noise $Q$', min: 0.001, max: 2, step: 0.001, init: 0.05 }, { k: 'r', label: 'assumed measurement noise $R$', min: 0.05, max: 16, step: 0.05, init: 4 }, { k: 'seed', seed: 'New data', init: 1 }],
  derive: ({ q, r, seed }) => {
    const g = rng(seed * 7 + 1); let x = 0, m = 0, P = 1; const T = [], O = [], E = [];
    for (let t = 0; t < 120; t++) { x = r6(x + 0.2 * gauss(g)); const y = r6(x + 2 * gauss(g)); const p = P + q, K = p / (p + r); m = r6(m + K * (y - m)); P = (1 - K) * p; T.push([t, x]); O.push([t, y]); E.push([t, m]); }
    return { T, O, E, K: P / r, rm: Math.sqrt(mean(E.map(([, e], i) => (e - T[i][1]) ** 2))), ro: Math.sqrt(mean(O.map(([, e], i) => (e - T[i][1]) ** 2))) };
  },
  plots: v => [{ x: [0, 120], y: [-8, 8], w: 500, h: 280, xl: 'step', label: 'truth, measurements and filter', marks: [{ as: 'points', tone: 'muted', r: 1.5, data: v.O.map(([t, y]) => [t, clip(y, -8, 8)]) }, { tone: 2, data: v.T.map(([t, y]) => [t, clip(y, -8, 8)]) }, { tone: 1, data: v.E.map(([t, y]) => [t, clip(y, -8, 8)]) }] }],
  stats: v => [`filter error ${fmt(v.rm, 3)}`, `raw error ${fmt(v.ro, 3)}`],
  legend: [['--c2', 'truth'], ['--c1', 'filter']],
  takeaway: 'The true noise here is $Q=0.04$ and $R=4$, so settings near those give the lowest error.',
});
export const SmRateDrill = drill({
  intro: 'A log link makes coefficients multiplicative.',
  tex: R`\frac{\lambda(x+1)}{\lambda(x)}=e^w`,
  gen: r => ({ w: pick(r, [-0.7, -0.3, 0.2, 0.5, 0.9]) }),
  q: v => `A Poisson coefficient is $${v.w}$. By what factor does one more unit multiply the expected count?`,
  answer: v => Math.exp(v.w), dp: 3,
  steps: v => [`Compute $e^{${v.w}}$.`],
});
export const SmExposureDrill = drill({
  intro: 'Expected count is exposure times the rate.',
  tex: R`\lambda=e\cdot\exp(x^\top w)`,
  gen: r => ({ e: int(r, 2, 40), b: pick(r, [-2, -1, -0.5, 0.5]) }),
  q: v => `Exposure $${v.e}$, linear score $${v.b}$. Find the expected count.`,
  answer: v => v.e * Math.exp(v.b), dp: 3,
  steps: v => [`Rate: $${fmt(Math.exp(v.b), 4)}$`],
});
export const SmNllDrill = drill({
  intro: 'Evaluate the Poisson negative log-likelihood.',
  tex: R`L=\lambda-y\ln\lambda+\ln y!`,
  gen: r => ({ l: pick(r, [0.5, 1, 2, 4]), y: int(r, 0, 4) }),
  q: v => `$\\lambda=${v.l}$, $y=${v.y}$. Find the negative log-likelihood.`,
  answer: v => v.l - v.y * Math.log(v.l) + Math.log(factorial(v.y)), dp: 3,
  steps: v => [`$y\\ln\\lambda=${fmt(v.y * Math.log(v.l), 3)}$`, `$\\ln y!=${fmt(Math.log(factorial(v.y)), 3)}$`],
});
export const SmPinballDrill = drill({
  intro: 'Under-prediction and over-prediction have different prices.',
  gen: r => ({ y: int(r, 0, 9), q: int(r, 0, 9), t: pick(r, [0.1, 0.5, 0.9]) }),
  q: v => `$y=${v.y}$, prediction $${v.q}$, $\\tau=${v.t}$. Find the pinball loss.`,
  answer: v => pinball(v.y - v.q, v.t), dp: 2,
  steps: v => [`Error: $${v.y - v.q}$`],
});
export const SmGainDrill = drill({
  intro: 'One scalar Kalman update.',
  tex: R`K=\frac{P^-}{P^-+R},\quad\hat x=\hat x^-+K(y-\hat x^-)`,
  gen: r => ({ p: int(r, 1, 9), r: int(r, 1, 9), m: int(r, 0, 9), y: int(r, 0, 12) }),
  q: v => `$P^-=${v.p}$, $R=${v.r}$, prediction $${v.m}$, measurement $${v.y}$. Find the updated estimate.`,
  answer: v => v.m + (v.p / (v.p + v.r)) * (v.y - v.m), dp: 3,
  steps: v => [`Gain: $${fmt(v.p / (v.p + v.r), 3)}$`],
});
export const SmDevianceDrill = drill({
  intro: 'A single row’s contribution to Poisson deviance.',
  tex: R`d=2\big(y\ln(y/\hat\lambda)-(y-\hat\lambda)\big)`,
  gen: r => ({ y: int(r, 1, 9), l: pick(r, [1, 2, 3, 5]) }),
  q: v => `$y=${v.y}$, $\\hat\\lambda=${v.l}$. Find the deviance of this row.`,
  answer: v => 2 * (v.y * Math.log(v.y / v.l) - (v.y - v.l)), dp: 3,
  steps: v => [`Log term: $${fmt(v.y * Math.log(v.y / v.l), 3)}$`],
});
export const SmCheck = truths({
  intro: CHECK,
  items: [
    ['A logistic regression is a generalised linear model.', true, 'It is a Bernoulli outcome with a logit link.'],
    ['Quantile regression assumes Gaussian noise.', false, 'It assumes nothing about the noise distribution.'],
    ['A Kalman filter uses only past and current measurements.', true, 'A smoother also uses later ones.'],
    ['A larger $R$ increases the Kalman gain.', false, 'A noisier sensor lowers the gain.'],
    ['An exposure offset has a coefficient to be learned.', false, 'It is fixed at 1.'],
  ],
  takeaway: 'Choose the distribution first, and the rest of the model follows.',
});

/* ---------- decisions: causality, bandits, tuning ---------- */
const confData = (c, seed, n = 3000, slope = 3) => { const r = rng(seed * 13 + 3); return Array.from({ length: n }, () => { const x = gauss(r), t = r() < sigmoid(c * x) ? 1 : 0; return [x, t, t + slope * x + gauss(r)]; }); };
const fitLine = D => { const X = mean(D.map(d => d[0])), Y = mean(D.map(d => d[2])), w = D.reduce((s, d) => s + (d[0] - X) * (d[2] - Y), 0) / (D.reduce((s, d) => s + (d[0] - X) ** 2, 0) || 1); return x => Y + w * (x - X); };
export const DcConfoundLab = lab({
  intro: 'Treatment truly adds exactly 1. A hidden trait raises both the chance of treatment and the outcome.',
  tex: R`\text{naive}=\text{ATE}+\text{selection bias}`,
  params: [{ k: 'c', label: 'confounding strength', min: 0, max: 3, step: 0.05, init: 1.5 }, { k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ c, seed }) => { const D = confData(c, seed), T = D.filter(d => d[1]), U = D.filter(d => !d[1]), m1 = fitLine(T), m0 = fitLine(U); return { D, m1, m0, naive: mean(T.map(d => d[2])) - mean(U.map(d => d[2])), adj: mean(D.map(d => m1(d[0]) - m0(d[0]))) }; },
  plots: v => [{ x: [-3, 3], y: [-10, 12], w: 480, h: 280, xl: 'trait x', yl: 'outcome', label: 'outcomes by treatment', marks: [{ as: 'points', r: 1.5, tone: 1, data: v.D.filter(d => d[1]).slice(0, 180).map(d => [d[0], clip(d[2], -10, 12)]) }, { as: 'points', r: 1.5, tone: 2, data: v.D.filter(d => !d[1]).slice(0, 180).map(d => [d[0], clip(d[2], -10, 12)]) }, { tone: 1, data: curve(v.m1, -3, 3, 3) }, { tone: 2, data: curve(v.m0, -3, 3, 3) }] }],
  stats: v => [`naive difference ${fmt(v.naive, 3)}`, `adjusted estimate ${fmt(v.adj, 3)}`, 'true effect 1.000'],
  legend: [['--c1', 'treated'], ['--c2', 'untreated']],
  takeaway: 'Comparing lines at the same $x$ removes the bias caused by that measured trait.',
});
export const DcSimpsonLab = lab({
  intro: 'Treatment helps in each group, and appears to hurt overall, depending on who received it.',
  params: [{ k: 'a', label: 'share of severe cases treated', min: 0, max: 1, step: 0.01, init: 0.9 }, { k: 'b', label: 'share of mild cases treated', min: 0, max: 1, step: 0.01, init: 0.1 }],
  derive: ({ a, b }) => { const s = [0.3, 0.4], m = [0.8, 0.9], tn = a + b, un = 2 - a - b; return { tr: (a * s[1] + b * m[1]) / (tn || 1), un: ((1 - a) * s[0] + (1 - b) * m[0]) / (un || 1) }; },
  table: () => ({ head: ['group', 'untreated', 'treated'], rows: [['severe', '30%', '40%'], ['mild', '80%', '90%']] }),
  stats: v => [`overall recovery, treated ${pct(v.tr, 1)}`, `untreated ${pct(v.un, 1)}`],
  status: v => (v.tr < v.un ? ['alert', 'The overall comparison points the wrong way.'] : ['ok', 'The overall comparison agrees with each group.']),
  takeaway: 'Severity affects both who gets treated and who recovers, so groups must be compared separately.',
});
export const DcOverlapLab = lab({
  intro: 'Repeat a study many times. With poor overlap, some weights become huge, and the IPW estimate swings. The doubly robust one is steadier.',
  params: [{ k: 'c', label: 'how strongly $x$ decides treatment', min: 0.2, max: 4, step: 0.1, init: 2 }, { k: 'seed', seed: 'New studies', init: 1 }],
  derive: ({ c, seed }) => {
    const I = [], A = [];
    for (let s = 0; s < 60; s++) { const D = confData(c, seed * 100 + s, 500), m1 = fitLine(D.filter(d => d[1])), m0 = fitLine(D.filter(d => !d[1])); let a = 0, b = 0; D.forEach(([x, t, y]) => { const e = sigmoid(c * x); a += t ? y / e : -y / (1 - e); b += m1(x) - m0(x) + (t ? (y - m1(x)) / e : -(y - m0(x)) / (1 - e)); }); I.push(r6(a / 500)); A.push(r6(b / 500)); }
    return { I, A };
  },
  plots: v => [{ x: [-4, 6], y: [0, 40], w: 480, h: 260, xl: 'estimate', yl: 'studies', label: 'estimates across repeated studies', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.I, -4, 6, 40).map(b => [b[0], 0, b[2], Math.min(40, b[3])]) }, { as: 'rects', tone: 2, data: hist(v.A, -4, 6, 40).map(b => [b[0], 0, b[2], Math.min(40, b[3])]) }, { as: 'rule', dash: true, tone: 4, x: 1 }] }],
  stats: v => [`IPW spread ${fmt(std(v.I), 3)}`, `doubly robust spread ${fmt(std(v.A), 3)}`],
  legend: [['--c1', 'IPW'], ['--c2', 'doubly robust']],
  takeaway: 'Always inspect overlap, since extreme propensities destroy simple weighting.',
});
export const DcUpliftLab = lab({
  intro: 'A coupon budget. Give it to those most likely to buy, or to those the coupon changes most.',
  params: [{ k: 'b', label: 'share of customers targeted', min: 0.02, max: 1, step: 0.01, init: 0.3 }],
  derive: () => { const r = rng(21), C = Array.from({ length: 2000 }, () => { const buy = r(), persuade = r() * (1 - buy) * 0.5; return { buy, up: persuade }; }); return { byBuy: [...C].sort((a, b) => b.buy - a.buy), byUp: [...C].sort((a, b) => b.up - a.up), total: C.length }; },
  plots: v => { const g = l => { let s = 0; const o = [[0, 0]]; l.forEach((c, i) => { s += c.up; if (i % 20 === 19) o.push([(i + 1) / v.total, s]); }); return o; }; return [{ x: [0, 1], y: [0, 260], w: 480, h: 260, xl: 'share targeted', yl: 'extra purchases', label: 'gain curves', marks: [{ tone: 2, data: g(v.byUp) }, { tone: 1, data: g(v.byBuy) }, { as: 'rule', dash: true, tone: 4, x: v.b }] }]; },
  stats: v => { const n = Math.round(v.b * v.total), s = l => l.slice(0, n).reduce((a, c) => a + c.up, 0); return [`by uplift: ${fmt(s(v.byUp), 1)} extra sales`, `by probability of buying: ${fmt(s(v.byBuy), 1)}`]; },
  legend: [['--c2', 'target by uplift'], ['--c1', 'target by purchase probability']],
  takeaway: 'The likeliest buyers often buy anyway, so spend on those who are persuadable.',
});
export const DcThompsonLab = lab({
  intro: 'Beliefs about two arms after their results so far. Thompson sampling plays each with the chance it is best.',
  tex: R`\theta_a\sim\text{Beta}(1+s_a,1+f_a)`,
  params: [{ k: 'a', label: 'arm A successes', min: 0, max: 40, step: 1, init: 6 }, { k: 'b', label: 'arm A failures', min: 0, max: 40, step: 1, init: 4 }, { k: 'c', label: 'arm B successes', min: 0, max: 40, step: 1, init: 1 }, { k: 'd', label: 'arm B failures', min: 0, max: 40, step: 1, init: 1 }],
  derive: ({ a, b, c, d }) => { const lg = n => { let s = 0; for (let i = 2; i < n; i++) s += Math.log(i); return s; }, beta = (x, p, q) => (x <= 0 || x >= 1 ? 0 : Math.exp(lg(p + q) - lg(p) - lg(q) + (p - 1) * Math.log(x) + (q - 1) * Math.log(1 - x))); const f = x => beta(x, a + 1, b + 1), g = x => beta(x, c + 1, d + 1), X = linspace(0.001, 0.999, 500); let best = 0, Fg = 0; X.forEach(x => { Fg += g(x) / 500; best += f(x) * Fg / 500; }); return { f, g, best }; },
  plots: v => [{ x: [0, 1], y: [0, 10], w: 480, h: 260, xl: 'success rate', yl: 'density', label: 'posterior for each arm', marks: [{ tone: 1, data: curve(x => Math.min(10, v.f(x)), 0.001, 0.999) }, { tone: 2, data: curve(x => Math.min(10, v.g(x)), 0.001, 0.999) }] }],
  stats: v => [`P(A is best) ${pct(v.best, 1)}`],
  legend: [['--c1', 'arm A'], ['--c2', 'arm B']],
  takeaway: 'An arm tried rarely keeps a wide posterior, so it still gets chosen sometimes.',
});
const MEANS = [0.3, 0.5, 0.6];
const runBandit = (kind, e, seed, T = 1500) => { const r = rng(seed), n = [0, 0, 0], s = [0, 0, 0], regret = []; let tot = 0; const gammaLike = (a, b) => { let x = 0; for (let i = 0; i < 12; i++) x += r(); const m = a / (a + b), sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))); return m + sd * (x - 6); }; for (let t = 1; t <= T; t++) { let a; if (n.includes(0)) a = n.indexOf(0); else if (kind === 'eps') a = r() < e ? Math.floor(r() * 3) : s.map((v, i) => v / n[i]).reduce((b, v, i, l) => (v > l[b] ? i : b), 0); else if (kind === 'ucb') { const u = s.map((v, i) => v / n[i] + Math.sqrt((2 * Math.log(t)) / n[i])); a = u.indexOf(Math.max(...u)); } else { const u = s.map((v, i) => gammaLike(1 + v, 1 + n[i] - v)); a = u.indexOf(Math.max(...u)); } n[a]++; s[a] += r() < MEANS[a] ? 1 : 0; tot += 0.6 - MEANS[a]; if (t % 15 === 0) regret.push([t, r6(tot)]); } return regret; };
export const DcBanditLab = lab({
  intro: 'Three arms with true rates 30%, 50% and 60%, averaged over repeated runs.',
  tex: R`R_T=T\mu^*-\sum_t\mu_{a_t}`,
  params: [{ k: 'e', label: '$\\varepsilon$ for greedy', min: 0, max: 0.5, step: 0.01, init: 0.1 }, { k: 'seed', seed: 'New runs', init: 1 }],
  derive: ({ e, seed }) => { const avg = k => { const R0 = Array.from({ length: 12 }, (_, i) => runBandit(k, e, seed * 100 + i)); return R0[0].map(([t], j) => [t, mean(R0.map(x => x[j][1]))]); }; return { E: avg('eps'), U: avg('ucb'), S: avg('ts') }; },
  plots: v => [{ x: [0, 1500], y: [0, 150], w: 480, h: 280, xl: 'round', yl: 'regret', label: 'cumulative regret', marks: [{ tone: 1, data: v.E.map(([t, y]) => [t, Math.min(150, y)]) }, { tone: 2, data: v.U.map(([t, y]) => [t, Math.min(150, y)]) }, { tone: 3, data: v.S.map(([t, y]) => [t, Math.min(150, y)]) }] }],
  stats: v => [`epsilon greedy ${fmt(v.E.at(-1)[1], 1)}`, `UCB ${fmt(v.U.at(-1)[1], 1)}`, `Thompson ${fmt(v.S.at(-1)[1], 1)}`],
  legend: [['--c1', 'epsilon greedy'], ['--c2', 'UCB'], ['--c3', 'Thompson']],
  takeaway: 'A constant exploration rate keeps paying forever, while UCB and Thompson taper off.',
});
const bof = x => -(Math.sin(3 * x) + 0.2 * (x - 2.5) ** 2);
const BOX = linspace(0, 5, 201);
export const DcBayesOptLab = lab({
  intro: 'A Gaussian process models the score, and expected improvement picks the next trial.',
  tex: R`\text{EI}=(\mu-f^+)\Phi(z)+\sigma\phi(z)`,
  params: [{ k: 'n', label: 'trials so far', min: 2, max: 14, step: 1, init: 4, play: true }],
  derive: ({ n }) => {
    const X = [0.6, 4.2], k = (a, b) => Math.exp(-((a - b) ** 2) / (2 * 0.36));
    let mu, sd, ei;
    for (let i = 2; ; i++) {
      const Y = X.map(bof), m = mean(Y), K = X.map((a, p) => X.map((b, q) => k(a, b) + (p === q ? 1e-6 : 0))), al = solve(K, Y.map(y => y - m)), best = Math.max(...Y);
      mu = BOX.map(x => m + X.reduce((s, a, j) => s + k(x, a) * al[j], 0));
      sd = BOX.map(x => { const ks = X.map(a => k(x, a)), v = solve(K, ks); return Math.sqrt(Math.max(1e-9, 1 - ks.reduce((s, c, j) => s + c * v[j], 0))); });
      ei = BOX.map((_, j) => { const d = mu[j] - best, z = d / sd[j]; return d * normalCdf(z) + sd[j] * normalPdf(z); });
      if (i >= n) break;
      X.push(BOX[ei.indexOf(Math.max(...ei))]);
    }
    return { X, mu, sd, ei };
  },
  plots: v => [{ caption: 'Model', x: [0, 5], y: [-4, 2], w: 480, h: 220, xl: 'setting', label: 'posterior and evaluations', marks: [{ tone: 'muted', dash: true, data: curve(bof, 0, 5) }, { tone: 1, data: BOX.map((x, j) => [x, clip(v.mu[j], -4, 2)]) }, { tone: 1, soft: true, data: BOX.map((x, j) => [x, clip(v.mu[j] + 2 * v.sd[j], -4, 2)]) }, { tone: 1, soft: true, data: BOX.map((x, j) => [x, clip(v.mu[j] - 2 * v.sd[j], -4, 2)]) }, { as: 'points', tone: 4, r: 4, data: v.X.map(x => [x, bof(x)]) }] }, { caption: 'Expected improvement', x: [0, 5], y: [0, Math.max(0.01, ...v.ei) * 1.1], w: 480, h: 160, xl: 'setting', label: 'expected improvement', marks: [{ tone: 2, data: BOX.map((x, j) => [x, v.ei[j]]) }] }],
  stats: v => [`best found ${fmt(Math.max(...v.X.map(bof)), 3)}`, `true best ${fmt(Math.max(...BOX.map(bof)), 3)}`],
  takeaway: 'It samples where the prediction is good or unknown, and rarely wastes a trial.',
});
export const DcIpwDrill = drill({
  intro: 'Weight each row by the inverse probability of what it received.',
  gen: r => ({ t: int(r, 0, 1), e: pick(r, [0.1, 0.2, 0.25, 0.5, 0.8]), y: int(r, 1, 9) }),
  q: v => `Row: $T=${v.t}$, $e(x)=${v.e}$, $Y=${v.y}$. Find its contribution to the IPW estimate, with the sign.`,
  answer: v => (v.t ? v.y / v.e : -v.y / (1 - v.e)), dp: 3,
  steps: v => [`Weight: $${fmt(v.t ? 1 / v.e : 1 / (1 - v.e), 3)}$`],
});
export const DcDidDrill = drill({
  intro: 'Subtract the control group’s change from the treated group’s change.',
  gen: r => ({ a: int(r, 5, 12), b: int(r, 10, 20), c: int(r, 3, 12), d: int(r, 5, 16) }),
  q: v => `Treated: from $${v.a}$ to $${v.b}$. Control: from $${v.c}$ to $${v.d}$. Find the estimate.`,
  answer: v => v.b - v.a - (v.d - v.c), dp: 0,
  steps: v => [`Treated change: $${v.b - v.a}$`, `Control change: $${v.d - v.c}$`],
});
export const DcUcbDrill = drill({
  intro: 'Add an exploration bonus that shrinks with more pulls.',
  tex: R`\hat\mu+\sqrt{\frac{2\ln t}{n}}`,
  gen: r => ({ m: pick(r, [0.2, 0.4, 0.6]), n: pick(r, [4, 10, 25, 100]), t: pick(r, [100, 500, 2000]) }),
  q: v => `$\\hat\\mu=${v.m}$, $n=${v.n}$, $t=${v.t}$. Find the UCB score.`,
  answer: v => v.m + Math.sqrt((2 * Math.log(v.t)) / v.n), dp: 3,
  steps: v => [`Bonus: $${fmt(Math.sqrt((2 * Math.log(v.t)) / v.n), 3)}$`],
});
export const DcPosteriorDrill = drill({
  intro: 'A uniform prior gives a Beta posterior.',
  tex: R`\mathbb E[\theta]=\frac{1+s}{2+s+f}`,
  gen: r => ({ s: int(r, 0, 20), f: int(r, 0, 20) }),
  q: v => `$${v.s}$ successes and $${v.f}$ failures, uniform prior. Find the posterior mean.`,
  answer: v => (1 + v.s) / (2 + v.s + v.f), dp: 3,
  steps: v => [`$\\text{Beta}(${1 + v.s},${1 + v.f})$`],
});
export const DcEiDrill = drill({
  intro: 'Expected improvement combines the gain and the uncertainty.',
  tex: R`\text{EI}=d\,\Phi(d/\sigma)+\sigma\,\phi(d/\sigma)`,
  gen: r => ({ d: pick(r, [0, 0.5, 1, -0.5]), s: pick(r, [0.5, 1, 2]) }),
  q: v => `$\\mu-f^+=${v.d}$, $\\sigma=${v.s}$. Find the expected improvement.`,
  answer: v => v.d * normalCdf(v.d / v.s) + v.s * normalPdf(v.d / v.s), dp: 2,
  steps: v => [`$z=${fmt(v.d / v.s, 3)}$`],
});
export const DcHalvingDrill = drill({
  intro: 'Each round keeps the best share, and gives the survivors more budget.',
  gen: r => { const b = pick(r, [2, 3]), k = int(r, 3, 4); return { b, k }; },
  q: v => `Start with $${v.b ** v.k}$ configurations, budget 1 each. Keep $1/${v.b}$ each round and multiply budget by $${v.b}$, until one is left. Find the total budget, excluding the last survivor’s round.`,
  answer: v => v.k * v.b ** v.k, dp: 0,
  steps: v => [`Each of the $${v.k}$ rounds costs $${v.b ** v.k}$.`],
});
export const DcCheck = truths({
  intro: CHECK,
  items: [
    ['Adjusting for a variable caused by the treatment is always safe.', false, 'It can block part of the effect you want to measure.'],
    ['A randomised experiment removes selection bias.', true, 'Treatment is independent of potential outcomes.'],
    ['A doubly robust estimate needs both models correct.', false, 'Either one correct is enough.'],
    ['UCB explores arms it knows little about.', true, 'Its bonus is large for rarely pulled arms.'],
    ['Grid search is usually better than random search.', false, 'Random search covers each important setting more densely.'],
  ],
  takeaway: 'Be explicit about what would have happened otherwise.',
});

/* ---------- structured inference, optimisers, ranking, privacy ---------- */
const chainMarg = (a, ev) => { const pair = [[a, 1 - a], [1 - a, a]], unary = ev.map(p => [1 - p, p]), n = ev.length; const f = Array.from({ length: n }, () => [1, 1]), b = Array.from({ length: n }, () => [1, 1]); for (let i = 1; i < n; i++) { const u = f[i - 1].map((x, s) => x * unary[i - 1][s]); f[i] = [0, 1].map(t => pair[0][t] * u[0] + pair[1][t] * u[1]); const z = f[i][0] + f[i][1]; f[i] = f[i].map(x => x / z); } for (let i = n - 2; i >= 0; i--) { const u = b[i + 1].map((x, s) => x * unary[i + 1][s]); b[i] = [0, 1].map(s => pair[s][0] * u[0] + pair[s][1] * u[1]); const z = b[i][0] + b[i][1]; b[i] = b[i].map(x => x / z); } return ev.map((_, i) => { const m = [0, 1].map(s => f[i][s] * unary[i][s] * b[i][s]); return m[1] / (m[0] + m[1]); }); };
export const IoChainLab = lab({
  intro: 'Eight linked variables. Evidence at the ends spreads along the chain, and coupling decides how far.',
  tex: R`m_{i\to j}(x_j)=\sum_{x_i}\psi(x_i,x_j)\,\psi_i(x_i)\,m_{\to i}(x_i)`,
  params: [{ k: 'a', label: 'coupling', min: 0.5, max: 0.99, step: 0.01, init: 0.85 }, { k: 'l', label: 'evidence at the left end', min: 0.01, max: 0.99, step: 0.01, init: 0.95 }, { k: 'r', label: 'evidence at the right end', min: 0.01, max: 0.99, step: 0.01, init: 0.5 }],
  derive: ({ a, l, r }) => ({ M: chainMarg(a, [l, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, r]) }),
  plots: v => [{ x: [0, 8], y: [0, 1], w: 480, h: 260, xl: 'variable', yl: 'P(state 1)', label: 'marginals', marks: [{ as: 'rects', tone: 1, data: v.M.map((p, i) => [i + 0.1, 0, i + 0.9, p]) }, { as: 'rule', dash: true, tone: 'muted', y: 0.5 }] }],
  stats: v => [`middle variable ${fmt(v.M[4], 3)}`],
  takeaway: 'Two passes along the chain give every exact marginal, in linear time.',
});
export const IoMessageDrill = drill({
  intro: 'Sum out one variable to make a message.',
  tex: R`m(x_2)=\sum_{x_1}\psi(x_1,x_2)\,p(x_1)`,
  gen: r => ({ a: pick(r, [0.6, 0.7, 0.8, 0.9]), p: pick(r, [0.2, 0.5, 0.75]) }),
  q: v => `$\\psi$ equals $${v.a}$ when the states match and $${fmt(1 - v.a, 1)}$ otherwise. $P(x_1=1)=${v.p}$. Find $m(x_2=1)$.`,
  answer: v => v.p * v.a + (1 - v.p) * (1 - v.a), dp: 3,
  steps: v => [`From $x_1=1$: $${fmt(v.p * v.a, 3)}$`, `From $x_1=0$: $${fmt((1 - v.p) * (1 - v.a), 3)}$`],
});
export const IoSoftDrill = drill({
  intro: 'The proximal step of the L1 norm.',
  tex: R`\text{prox}(v)=\text{sign}(v)\max(|v|-\lambda\eta,0)`,
  gen: r => ({ v: pick(r, [-3, -1.5, -0.4, 0.3, 1.2, 2.5]), t: pick(r, [0.25, 0.5, 1]) }),
  q: v => `$v=${v.v}$, $\\lambda\\eta=${v.t}$. Find the result.`,
  answer: v => Math.sign(v.v) * Math.max(Math.abs(v.v) - v.t, 0), dp: 2,
  steps: v => [`$|v|-\\lambda\\eta=${fmt(Math.abs(v.v) - v.t, 2)}$`],
});
export const IoIstaLab = lab({
  intro: 'Minimise a quadratic plus an L1 penalty. Momentum makes a large difference on ill-conditioned problems.',
  tex: R`w\leftarrow\text{prox}\big(w-\tfrac1L\nabla f(w)\big)`,
  params: [{ k: 'k', label: 'condition number', min: 1, max: 400, step: 1, init: 100 }, { k: 'l', label: 'penalty $\\lambda$', min: 0, max: 1, step: 0.01, init: 0.1 }],
  derive: ({ k, l }) => {
    const c = [1, k], b = [1, 1], soft = (v, t) => Math.sign(v) * Math.max(Math.abs(v) - t, 0), L = k, obj = w => w.reduce((s, x, i) => s + 0.5 * c[i] * x * x - b[i] * x + l * Math.abs(x), 0), best = obj(b.map((v, i) => soft(v, l) / c[i])), step = w => w.map((x, i) => soft(x - (c[i] * x - b[i]) / L, l / L));
    let w = [3, 3], u = [3, 3], y = [3, 3], t = 1; const A = [], F = [];
    for (let i = 0; i <= 150; i++) { A.push([i, Math.max(-10, Math.log10(obj(w) - best + 1e-12))]); F.push([i, Math.max(-10, Math.log10(obj(u) - best + 1e-12))]); w = step(w); const nu = step(y), t2 = (1 + Math.sqrt(1 + 4 * t * t)) / 2; y = nu.map((x, j) => x + ((t - 1) / t2) * (x - u[j])); u = nu; t = t2; }
    return { A, F };
  },
  plots: v => [{ x: [0, 150], y: [-10, 4], w: 480, h: 260, xl: 'step', yl: 'log10 gap', label: 'objective gap', marks: [{ tone: 1, data: v.A }, { tone: 2, data: v.F }] }],
  stats: v => [`ISTA gap 10^${fmt(v.A[150][1], 1)}`, `FISTA gap 10^${fmt(v.F[150][1], 1)}`],
  legend: [['--c1', 'ISTA'], ['--c2', 'FISTA']],
  takeaway: 'The same proximal step, with momentum, can save hundreds of iterations.',
});
const quad = k => ({ f: ([a, b]) => 0.5 * (a * a + k * b * b), g: ([a, b]) => [a, k * b] });
export const IoQuasiLab = lab({
  intro: 'Gradient descent zigzags along a valley. BFGS learns the curvature from its own gradients.',
  tex: R`H_{k+1}=(I-\rho sy^\top)H_k(I-\rho ys^\top)+\rho ss^\top`,
  params: [{ k: 'k', label: 'curvature ratio', min: 1, max: 100, step: 1, init: 40 }],
  derive: ({ k }) => {
    const { f, g } = quad(k);
    const lr = 1.9 / k; let x = [8, 1]; const G = [x];
    for (let i = 0; i < 24; i++) { const d = g(x); x = [x[0] - lr * d[0], x[1] - lr * d[1]]; G.push(x); }
    let w = [8, 1], H = [[1, 0], [0, 1]]; const B = [w];
    for (let i = 0; i < 24; i++) {
      const d = g(w), p = [-(H[0][0] * d[0] + H[0][1] * d[1]), -(H[1][0] * d[0] + H[1][1] * d[1])]; let t = 1;
      while (f([w[0] + t * p[0], w[1] + t * p[1]]) > f(w) + 1e-4 * t * (d[0] * p[0] + d[1] * p[1]) && t > 1e-8) t /= 2;
      const n = [w[0] + t * p[0], w[1] + t * p[1]], d2 = g(n), s = [n[0] - w[0], n[1] - w[1]], y = [d2[0] - d[0], d2[1] - d[1]], sy = s[0] * y[0] + s[1] * y[1];
      if (sy > 1e-14) { const r = 1 / sy, A = [[1 - r * s[0] * y[0], -r * s[0] * y[1]], [-r * s[1] * y[0], 1 - r * s[1] * y[1]]], m = (P, Q) => P.map(a => Q[0].map((_, c) => a[0] * Q[0][c] + a[1] * Q[1][c])), AT = [[A[0][0], A[1][0]], [A[0][1], A[1][1]]], Z = m(m(A, H), AT); H = [[Z[0][0] + r * s[0] * s[0], Z[0][1] + r * s[0] * s[1]], [Z[1][0] + r * s[1] * s[0], Z[1][1] + r * s[1] * s[1]]]; }
      w = n; B.push(w);
    }
    return { G, B, f };
  },
  plots: v => [{ x: [-9, 9], y: [-1.5, 1.5], w: 480, h: 260, label: 'optimiser paths', marks: [{ tone: 1, data: v.G.map(([a, b]) => [clip(a, -9, 9), clip(b, -1.5, 1.5)]) }, { tone: 2, data: v.B.map(([a, b]) => [clip(a, -9, 9), clip(b, -1.5, 1.5)]) }, { as: 'points', tone: 4, r: 4, data: [[0, 0]] }] }],
  stats: v => [`after 24 steps: gradient descent ${v.f(v.G[24]).toExponential(1)}`, `BFGS ${v.f(v.B[24]).toExponential(1)}`],
  legend: [['--c1', 'gradient descent'], ['--c2', 'BFGS']],
  takeaway: 'Curvature information removes the dependence on conditioning, at modest memory cost in L-BFGS.',
});
export const IoAdmmLab = lab({
  intro: 'ADMM on a small lasso problem. The penalty $\\rho$ changes speed but not the answer.',
  tex: R`u\leftarrow u+w-z`,
  params: [{ k: 'p', label: '$\\log_{10}\\rho$', min: -2, max: 2, step: 0.05, init: 0 }],
  derive: ({ p }) => { const rho = 10 ** p, a = 2, y = 3, l = 1; let w = 0, z = 0, u = 0; const H = [], star = Math.sign(y) * Math.max(Math.abs(y) - l, 0) / a; for (let i = 0; i <= 60; i++) { H.push([i, Math.max(-10, Math.log10(Math.abs(z - star) + 1e-12))]); w = (y + rho * (z - u)) / (a + rho); const v = w + u; z = Math.sign(v) * Math.max(Math.abs(v) - l / rho, 0); u += w - z; } return { H, star, z }; },
  plots: v => [{ x: [0, 60], y: [-10, 1], w: 480, h: 260, xl: 'step', yl: 'log10 error', label: 'error of z', marks: [{ tone: 1, data: v.H }] }],
  stats: v => [`final ${fmt(v.z, 4)}`, `exact ${fmt(v.star, 4)}`],
  takeaway: 'Too small or too large a $\\rho$ slows convergence, though the fixed point stays the same.',
});
export const IoPairDrill = drill({
  intro: 'RankNet compares two scores with a logistic loss.',
  tex: R`L=\ln\big(1+e^{-(s_i-s_j)}\big)`,
  gen: r => ({ a: int(r, -3, 3), b: int(r, -3, 3) }),
  q: v => `Item $i$ should rank higher. $s_i=${v.a}$, $s_j=${v.b}$. Find the loss.`,
  answer: v => Math.log(1 + Math.exp(-(v.a - v.b))), dp: 3,
  steps: v => [`Difference: $${v.a - v.b}$`],
});
export const IoDeltaDrill = drill({
  intro: 'LambdaRank weights a pair by how much a swap would change the gain.',
  tex: R`|\Delta\text{DCG}|=\big|(g_i-g_j)(d_i-d_j)\big|,\quad d_k=\frac1{\log_2(k+1)}`,
  gen: r => ({ gi: pick(r, [1, 3, 7]), gj: pick(r, [0, 1]), i: pick(r, [1, 2]), j: pick(r, [3, 5, 8]) }),
  q: v => `Gains $${v.gi}$ and $${v.gj}$ sit at ranks $${v.i}$ and $${v.j}$. Find the change in DCG from swapping them.`,
  answer: v => Math.abs((v.gi - v.gj) * (1 / Math.log2(v.i + 1) - 1 / Math.log2(v.j + 1))), dp: 3,
  steps: v => [`Discount gap: $${fmt(1 / Math.log2(v.i + 1) - 1 / Math.log2(v.j + 1), 3)}$`],
});
export const IoLaplaceLab = lab({
  intro: 'Two datasets differ by one person. Noise makes their outputs almost indistinguishable.',
  tex: R`\frac{p(a)}{p(a')}\le e^\varepsilon`,
  params: [{ k: 'e', label: 'privacy $\\varepsilon$', min: 0.05, max: 3, step: 0.05, init: 0.5 }, { k: 'x', label: 'published value', min: 90, max: 110, step: 0.5, init: 103 }],
  derive: ({ e }) => { const b = 1 / e; return { b, p: (x, m) => Math.exp(-Math.abs(x - m) / b) / (2 * b) }; },
  plots: v => [{ x: [90, 110], y: [0, Math.max(0.3, v.p(100, 100) * 1.1)], w: 480, h: 260, xl: 'published count', yl: 'density', label: 'output densities', marks: [{ tone: 1, data: curve(x => v.p(x, 100), 90, 110, 401) }, { tone: 2, data: curve(x => v.p(x, 101), 90, 110, 401) }, { as: 'rule', dash: true, tone: 4, x: v.x }] }],
  stats: v => [`likelihood ratio ${fmt(v.p(v.x, 101) / v.p(v.x, 100), 3)}`, `limit ${fmt(Math.exp(v.e), 3)}`, `noise scale ${fmt(v.b, 2)}`],
  legend: [['--c1', 'true count 100'], ['--c2', 'true count 101']],
  takeaway: 'Whatever the output, it can only slightly favour one dataset, so stronger privacy needs more noise.',
});
export const IoBudgetDrill = drill({
  intro: 'Noise scale follows sensitivity and budget, and budgets add up.',
  tex: R`b=\frac\Delta\varepsilon`,
  gen: r => ({ d: pick(r, [1, 2, 5]), e: pick(r, [0.1, 0.25, 0.5, 1]), q: int(r, 2, 8) }),
  q: v => `Sensitivity $${v.d}$, $${v.q}$ queries, each with $\\varepsilon=${v.e}$. Find the Laplace scale per query.`,
  answer: v => v.d / v.e, dp: 2,
  steps: v => [`The total budget would be $${fmt(v.q * v.e, 2)}$.`],
});
export const IoFedAvgLab = lab({
  intro: 'Clients each take several local steps on differently skewed data, and the server averages.',
  tex: R`w\leftarrow\sum_k\frac{n_k}nw_k`,
  params: [{ k: 'l', label: 'local steps', min: 1, max: 40, step: 1, init: 5 }, { k: 's', label: 'data skew', min: 0, max: 4, step: 0.1, init: 2 }],
  derive: ({ l, s }) => {
    const target = c => [c, -c], C = [-1, -0.5, 0.5, 1].map(c => target(c * s)); let w = [3, 3]; const H = [];
    for (let r = 0; r <= 40; r++) { H.push([r, Math.max(-6, Math.log10(Math.hypot(w[0], w[1]) + 1e-9))]); const u = C.map(c => { let v = w; for (let i = 0; i < l; i++) v = [v[0] - 0.1 * (v[0] - c[0]), v[1] - 0.1 * 5 * (v[1] - c[1])]; return v; }); w = [mean(u.map(a => a[0])), mean(u.map(a => a[1]))]; }
    return { H, w };
  },
  plots: v => [{ x: [0, 40], y: [-6, 1], w: 480, h: 260, xl: 'round', yl: 'log10 distance to optimum', label: 'distance to the global optimum', marks: [{ tone: 1, data: v.H }] }],
  stats: v => [`final distance ${fmt(Math.hypot(v.w[0], v.w[1]), 4)}`],
  takeaway: 'Local steps save communication, and each round moves faster. On unequal curvature the average can stop short.',
});
export const IoActiveLab = lab({
  intro: 'Find an unknown threshold, buying one label at a time. Querying where the model is unsure beats choosing at random.',
  params: [{ k: 'n', label: 'labels bought', min: 1, max: 30, step: 1, init: 10 }, { k: 'seed', seed: 'New threshold', init: 1 }],
  derive: ({ seed }) => { const run = (active, s) => { const r = rng(s), t = r(); let a = 0, b = 1; const o = []; for (let i = 1; i <= 30; i++) { const x = active ? (a + b) / 2 : r(); if (x < t) a = Math.max(a, x); else b = Math.min(b, x); o.push(b - a); } return o; }, avg = k => Array.from({ length: 30 }, (_, i) => mean(Array.from({ length: 40 }, (_, s) => run(k, seed * 50 + s)[i]))); return { A: avg(true), B: avg(false) }; },
  plots: v => [{ x: [1, 30], y: [-10, 0], w: 480, h: 260, xl: 'labels', yl: 'log2 uncertainty', label: 'remaining interval', marks: [{ tone: 1, data: v.B.map((y, i) => [i + 1, Math.max(-10, Math.log2(y))]) }, { tone: 2, data: v.A.map((y, i) => [i + 1, Math.max(-10, Math.log2(y))]) }, { as: 'rule', dash: true, tone: 4, x: v.n }] }],
  stats: v => [`uncertainty: active ${fmt(v.A[v.n - 1], 4)}`, `random ${fmt(v.B[v.n - 1], 4)}`],
  legend: [['--c1', 'random'], ['--c2', 'active']],
  takeaway: 'Each active query halves the uncertainty, where random ones shrink it only slowly.',
});
export const IoLeakLab = lab({
  intro: 'A category that is pure noise. Plain target encoding makes it look predictive when each category holds few rows.',
  params: [{ k: 'c', label: 'number of categories', min: 2, max: 400, step: 2, init: 200 }, { k: 'seed', seed: 'New data', init: 1 }],
  derive: ({ c, seed }) => { const r = rng(seed * 3), n = 1000, cat = Array.from({ length: n }, () => Math.floor(r() * c)), y = Array.from({ length: n }, () => (r() < 0.5 ? 1 : 0)), S = Array(c).fill(0), N = Array(c).fill(0); cat.forEach((k, i) => { S[k] += y[i]; N[k]++; }); const plain = cat.map(k => (S[k] + 0.5) / (N[k] + 1)), S2 = Array(c).fill(0), N2 = Array(c).fill(0), ord = cat.map((k, i) => { const v = (S2[k] + 0.5) / (N2[k] + 1); S2[k] += y[i]; N2[k]++; return v; }), auc = s => { const o = s.map((v, i) => [v, y[i]]).sort((a, b) => a[0] - b[0]); let rank = 0, pos = 0, sum = 0; for (let i = 0; i < o.length;) { let j = i; while (j < o.length && o[j][0] === o[i][0]) j++; const m = (i + j + 1) / 2; for (let q = i; q < j; q++) if (o[q][1]) { sum += m; pos++; } rank = j; i = j; } return (sum - (pos * (pos + 1)) / 2) / (pos * (rank - pos) || 1); }; return { p: auc(plain), o: auc(ord) }; },
  plots: v => [{ x: [0, 2], y: [0, 1], w: 380, h: 260, label: 'training AUC of a noise feature', marks: [{ as: 'rects', tone: 1, data: [[0.15, 0, 0.85, v.p]] }, { as: 'rects', tone: 2, data: [[1.15, 0, 1.85, v.o]] }, { as: 'rule', dash: true, tone: 'muted', y: 0.5 }] }],
  stats: v => [`plain encoding AUC ${fmt(v.p, 3)}`, `ordered encoding AUC ${fmt(v.o, 3)}`],
  legend: [['--c1', 'plain'], ['--c2', 'ordered']],
  takeaway: 'An AUC above 0.5 on pure noise is leakage, and ordered statistics remove it.',
});
export const IoHistDrill = drill({
  intro: 'Histogram trees compare a fixed number of bins rather than every row.',
  gen: r => ({ n: pick(r, [10000, 100000, 1000000]), p: int(r, 5, 50), b: pick(r, [64, 255]) }),
  q: v => `$${v.n}$ rows, $${v.p}$ features, $${v.b}$ bins. By what factor are candidate splits reduced per node?`,
  answer: v => v.n / v.b, dp: 1,
  steps: v => [`Exact candidates per feature: about $${v.n}$`],
});
export const IoCheck = truths({
  intro: CHECK,
  items: [
    ['Sum-product is exact on trees.', true, 'Each message is computed once.'],
    ['Federated learning guarantees privacy by itself.', false, 'Updates can leak data, so noise and secure aggregation are needed.'],
    ['L-BFGS stores the whole Hessian.', false, 'It stores only a few recent difference pairs.'],
    ['Privacy budgets add up across queries.', true, 'Each query spends some of the budget.'],
    ['Leaf-wise trees can overfit small datasets.', true, 'They grow deep where the gain is largest.'],
  ],
  takeaway: 'Exploit structure, and account for exactly what each step costs or leaks.',
});

/* ---------- robust, uncertain and compact networks ---------- */
export const RbLinearAttackLab = lab({
  intro: 'A linear classifier and a point near its boundary. Each coordinate may move by at most $\\epsilon$.',
  tex: R`x'=x-\epsilon\,y\,\text{sign}(w)`,
  params: [{ k: 'e', label: 'attack size $\\epsilon$', min: 0, max: 2, step: 0.02, init: 0.3 }, { k: 'a', label: 'weight angle', min: 0, max: 90, step: 1, init: 40 }],
  derive: ({ e, a }) => { const w = [Math.cos((a * Math.PI) / 180), Math.sin((a * Math.PI) / 180)], x = [1.2, 1.2], s = w[0] * x[0] + w[1] * x[1], xa = [x[0] - e * Math.sign(w[0]), x[1] - e * Math.sign(w[1] || 1)]; return { w, x, xa, s, sa: w[0] * xa[0] + w[1] * xa[1] }; },
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 360, h: 340, label: 'attack on a linear model', marks: [{ as: 'field', bands: 1, tones: [2, 1], fn: (x, y) => (v.w[0] * x + v.w[1] * y > 0 ? 0.3 : -0.3) }, { as: 'arrow', tone: 4, data: [[v.x[0], v.x[1], v.xa[0], v.xa[1]]] }, { as: 'points', tone: 'fg', r: 5, data: [v.x] }] }],
  stats: v => [`score ${fmt(v.s, 3)} to ${fmt(v.sa, 3)}`, `movement ${fmt(v.e * (Math.abs(v.w[0]) + Math.abs(v.w[1])), 3)}`],
  status: v => (v.sa <= 0 ? ['alert', 'The prediction flipped.'] : ['ok', 'The prediction held.']),
  takeaway: 'Every coordinate moves in the worst direction at once, so small changes add up.',
});
const loss1d = x => 1 + Math.sin(3 * x) + 0.3 * Math.sin(9 * x) + 0.2 * x;
const dloss = x => 3 * Math.cos(3 * x) + 2.7 * Math.cos(9 * x) + 0.2;
export const RbPgdLab = lab({
  intro: 'The loss around one input. A single large gradient step can miss the peak that repeated small steps find.',
  params: [{ k: 'x', label: 'starting input', min: -1, max: 1, step: 0.01, init: -0.1 }, { k: 'e', label: 'limit $\\epsilon$', min: 0.05, max: 1, step: 0.01, init: 0.6 }, { k: 'n', label: 'PGD steps', min: 1, max: 30, step: 1, init: 12 }],
  derive: ({ x, e, n }) => { const f = clip(x + e * Math.sign(dloss(x)), x - e, x + e); let a = x; const P = [a]; for (let i = 0; i < n; i++) { a = clip(a + (e / 4) * Math.sign(dloss(a)), x - e, x + e); P.push(a); } let best = -Infinity; for (let i = 0; i <= 400; i++) { const c = x - e + (2 * e * i) / 400; best = Math.max(best, loss1d(c)); } return { f, a, P, best }; },
  plots: v => [{ x: [-2.2, 2.2], y: [-1, 3.2], w: 480, h: 260, xl: 'input', yl: 'loss', label: 'loss around the input', marks: [{ tone: 1, data: curve(loss1d, -2.2, 2.2, 601) }, { as: 'rule', dash: true, tone: 'muted', x: v.x - v.e }, { as: 'rule', dash: true, tone: 'muted', x: v.x + v.e }, { as: 'points', tone: 2, r: 5, data: [[v.f, loss1d(v.f)]] }, { as: 'points', tone: 4, r: 5, data: [[v.a, loss1d(v.a)]] }] }],
  stats: v => [`FGSM loss ${fmt(loss1d(v.f), 3)}`, `PGD loss ${fmt(loss1d(v.a), 3)}`, `worst possible ${fmt(v.best, 3)}`],
  legend: [['--c2', 'FGSM'], ['--c4', 'PGD']],
  takeaway: 'Report robustness against strong iterative attacks, or the number is misleading.',
});
export const RbAttackSizeDrill = drill({
  intro: 'For a linear score, the worst shift is the L1 norm scaled by the limit.',
  tex: R`\Delta=\epsilon\|w\|_1`,
  gen: r => ({ d: pick(r, [100, 1000, 10000]), a: pick(r, [0.01, 0.05, 0.1]), e: pick(r, [0.01, 0.03]) }),
  q: v => `$${v.d}$ weights, each of magnitude $${v.a}$, with $\\epsilon=${v.e}$. How much can the score move?`,
  answer: v => v.d * v.a * v.e, dp: 2,
  steps: v => [`$\\|w\\|_1=${fmt(v.d * v.a, 2)}$`],
});
export const RbFgsmDrill = drill({
  intro: 'One FGSM step on one coordinate.',
  gen: r => ({ x: pick(r, [0.2, 0.5, 0.8]), g: pick(r, [-3, -0.1, 0.4, 2]), e: pick(r, [0.03, 0.1]) }),
  q: v => `$x=${v.x}$, gradient $${v.g}$, $\\epsilon=${v.e}$. Find the attacked value.`,
  answer: v => v.x + v.e * Math.sign(v.g), dp: 3,
  steps: v => [`Sign: $${Math.sign(v.g)}$`],
});
export const RbProjectDrill = drill({
  intro: 'PGD projects back into the allowed range.',
  gen: r => ({ x: pick(r, [0.2, 0.5]), e: pick(r, [0.1, 0.2]), a: pick(r, [-0.2, 0.05, 0.35, 0.9]) }),
  q: v => `Original $${v.x}$, $\\epsilon=${v.e}$, and the current attack is at $${v.a}$. Find the value after projection.`,
  answer: v => clip(v.a, v.x - v.e, v.x + v.e), dp: 3,
  steps: v => [`Allowed: $[${fmt(v.x - v.e, 2)},${fmt(v.x + v.e, 2)}]$`],
});
const ENSX = (() => { const r = rng(2); return Array.from({ length: 24 }, () => { const x = -1 + r() * 2; return [r6(x), r6(Math.sin(2 * x) + 0.1 * gauss(r))]; }); })();
const polyFit = (D, deg) => { const A = D.map(([x]) => Array.from({ length: deg + 1 }, (_, p) => x ** p)), M = Array.from({ length: deg + 1 }, (_, i) => Array.from({ length: deg + 1 }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-4 : 0))), b = Array.from({ length: deg + 1 }, (_, i) => A.reduce((s, r, k) => s + r[i] * D[k][1], 0)); return solve(M, b); };
const polyEval = (c, x) => c.reduce((s, v, p) => s + v * x ** p, 0);
export const RbEnsembleLab = lab({
  intro: 'Fit many models, each on a resample of the data. They agree where data exist, and disagree beyond.',
  params: [{ k: 'm', label: 'ensemble size', min: 2, max: 40, step: 1, init: 12 }, { k: 'd', label: 'model flexibility', min: 1, max: 7, step: 1, init: 5 }, { k: 'x', label: 'read at $x$', min: -2.5, max: 2.5, step: 0.05, init: 2 }, { k: 'seed', seed: 'New resamples', init: 1 }],
  derive: ({ m, d, seed }) => { const r = rng(seed * 41); return { C: Array.from({ length: m }, () => polyFit(Array.from({ length: 24 }, () => ENSX[Math.floor(r() * 24)]), d)) }; },
  plots: v => [{ x: [-2.5, 2.5], y: [-4, 4], w: 480, h: 280, xl: 'x', yl: 'y', label: 'ensemble members', marks: [...v.C.map(c => ({ tone: 1, soft: true, data: curve(x => clip(polyEval(c, x), -4, 4), -2.5, 2.5, 101) })), { as: 'points', tone: 'fg', r: 3, data: ENSX }, { as: 'rule', dash: true, tone: 4, x: v.x }] }],
  stats: v => [`spread at this point ${fmt(std(v.C.map(c => polyEval(c, v.x))), 3)}`, `spread at zero ${fmt(std(v.C.map(c => polyEval(c, 0))), 3)}`],
  takeaway: 'Disagreement is a useful estimate of epistemic uncertainty, and it grows away from the data.',
});
export const RbEntropyDrill = drill({
  intro: 'Split total uncertainty for two ensemble members on a binary problem, in nats.',
  tex: R`\text{epistemic}=H[\bar p]-\tfrac12\big(H[p_1]+H[p_2]\big)`,
  gen: r => ({ a: pick(r, [0.1, 0.3, 0.5]), b: pick(r, [0.6, 0.8, 0.9]) }),
  q: v => `Two members predict $P(1)=${v.a}$ and $${v.b}$. Find the epistemic part.`,
  answer: v => { const h = p => -(p * Math.log(p) + (1 - p) * Math.log(1 - p)); return h((v.a + v.b) / 2) - (h(v.a) + h(v.b)) / 2; }, dp: 3,
  steps: v => [`Average prediction: $${fmt((v.a + v.b) / 2, 2)}$`],
});
export const RbDropoutDrill = drill({
  intro: 'Rescaling keeps the mean, and adds variance.',
  tex: R`\text{Var}\Big[\frac{zb}q\Big]=z^2\,\frac{1-q}q`,
  gen: r => ({ z: int(r, 1, 6), q: pick(r, [0.5, 0.8, 0.9]) }),
  q: v => `An activation of $${v.z}$, with keep probability $${v.q}$. Find the variance added by dropout.`,
  answer: v => (v.z * v.z * (1 - v.q)) / v.q, dp: 3,
  steps: v => [`$z^2=${v.z * v.z}$`],
});
const TEACH = [4, 1.5, 0.5, -1];
export const RbDistillLab = lab({
  intro: 'A teacher’s scores for four classes. Higher temperature reveals how it ranks the wrong answers.',
  tex: R`p_i\propto e^{z_i/T}`,
  params: [{ k: 'T', label: 'temperature $T$', min: 0.5, max: 12, step: 0.1, init: 4 }],
  derive: ({ T }) => ({ P: softmaxT(TEACH, T), H: softmaxT(TEACH, 1) }),
  plots: v => [{ x: [0, 4], y: [0, 1], w: 480, h: 260, xl: 'class', yl: 'probability', label: 'teacher probabilities', marks: [{ as: 'rects', tone: 'muted', soft: true, data: v.H.map((p, i) => [i + 0.08, 0, i + 0.5, p]) }, { as: 'rects', tone: 1, data: v.P.map((p, i) => [i + 0.5, 0, i + 0.92, p]) }] }],
  stats: v => [`largest ${pct(v.P[0], 1)}`, `second ${pct(v.P[1], 1)}`, `soft gradient scale 1/${fmt(v.T * v.T, 1)}`],
  legend: [['--c1', 'at this temperature'], ['--muted', 'at T = 1']],
  takeaway: 'The relative sizes of the wrong answers are extra information a hard label cannot provide.',
});
const PRD = (() => { const r = rng(6), w = Array.from({ length: 40 }, (_, i) => (i < 6 ? 1 + r() * 2 : 0.05 * gauss(r))); const score = u => w.reduce((s, x, i) => s + (x - u[i]) ** 2, 0); return { w, score }; })();
export const RbPruneLab = lab({
  intro: 'Forty weights, of which only a few matter. Remove the smallest, and read how much is lost.',
  params: [{ k: 'p', label: 'share pruned', min: 0, max: 1, step: 0.01, init: 0.5 }],
  derive: ({ p }) => { const s = [...PRD.w].map(Math.abs).sort((a, b) => a - b), k = Math.floor(p * 40), t = k ? s[k - 1] : -1, u = PRD.w.map(x => (Math.abs(x) <= t ? 0 : x)); return { u, err: PRD.score(u) }; },
  plots: v => [{ x: [0, 40], y: [-0.5, 3.2], w: 480, h: 260, xl: 'weight', yl: 'value', label: 'weights after pruning', marks: [{ as: 'rects', tone: 'muted', soft: true, data: PRD.w.map((x, i) => [i + 0.1, Math.min(0, x), i + 0.9, Math.max(0, x)]) }, { as: 'rects', tone: 1, data: v.u.map((x, i) => [i + 0.1, Math.min(0, x), i + 0.9, Math.max(0, x)]) }] }],
  stats: v => [`loss from pruning ${fmt(v.err, 3)}`, `weights kept ${v.u.filter(x => x !== 0).length}`],
  takeaway: 'Accuracy holds until pruning reaches the weights that matter, then falls sharply.',
});
export const RbQuantLab = lab({
  intro: 'A signal stored in fewer bits. Each extra bit halves the step.',
  tex: R`S=\frac{b-a}{2^n-1}`,
  params: [{ k: 'n', label: 'bits', min: 1, max: 10, step: 1, init: 3 }],
  derive: ({ n }) => { const L = 2 ** n - 1, q = x => -1 + (Math.round(((x + 1) / 2) * L) / L) * 2; return { q, S: 2 / L, sqnr: 10 * Math.log10(mean(linspace(0, 6.283, 2000).map(t => Math.sin(t) ** 2)) / (mean(linspace(0, 6.283, 2000).map(t => (Math.sin(t) - q(Math.sin(t))) ** 2)) || 1e-12)) }; },
  plots: v => [{ x: [0, 6.283], y: [-1.1, 1.1], w: 480, h: 260, xl: 't', label: 'quantised signal', marks: [{ tone: 'muted', data: curve(Math.sin, 0, 6.283) }, { tone: 1, data: curve(t => v.q(Math.sin(t)), 0, 6.283, 1201) }] }],
  stats: v => [`step ${fmt(v.S, 4)}`, `signal to noise ${fmt(v.sqnr, 1)} dB`],
  takeaway: 'About 6 dB is gained per bit, so 8 bits are usually enough.',
});
export const RbQuantDrill = drill({
  intro: 'Find the step, and the largest rounding error.',
  tex: R`\text{error}\le\frac S2`,
  gen: r => ({ a: pick(r, [-1, -2, 0]), b: pick(r, [1, 2, 4]), n: pick(r, [4, 6, 8]) }),
  q: v => `Range $[${v.a},${v.b}]$, $${v.n}$ bits. Find the largest possible rounding error.`,
  answer: v => (v.b - v.a) / (2 ** v.n - 1) / 2, dp: 4,
  steps: v => [`$S=${fmt((v.b - v.a) / (2 ** v.n - 1), 4)}$`],
});
export const RbScaleDrill = drill({
  intro: 'A common rule scales the learning rate with the effective batch.',
  gen: r => ({ b: pick(r, [32, 64]), k: pick(r, [2, 4, 8, 16]), lr: pick(r, [0.01, 0.05, 0.1]) }),
  q: v => `A base rate of $${v.lr}$ was tuned for batch $${v.b}$. With $${v.k}$ workers, each using batch $${v.b}$, what rate does linear scaling give?`,
  answer: v => v.lr * v.k, dp: 3,
  steps: v => [`Effective batch: $${v.k * v.b}$`],
});
export const RbSpeedupLab = lab({
  intro: 'Only the parallel share speeds up, and communication grows with workers.',
  tex: R`\text{speedup}=\frac1{s+\frac{1-s}K+cK}`,
  params: [{ k: 's', label: 'serial share', min: 0, max: 0.3, step: 0.005, init: 0.02 }, { k: 'c', label: 'communication cost', min: 0, max: 0.02, step: 0.0005, init: 0.002 }, { k: 'K', label: 'workers', min: 1, max: 128, step: 1, init: 32 }],
  derive: ({ s, c }) => ({ f: K => 1 / (s + (1 - s) / K + c * (K - 1)) }),
  plots: v => [{ x: [1, 128], y: [0, 50], w: 480, h: 260, xl: 'workers', yl: 'speedup', label: 'speedup against workers', marks: [{ tone: 'muted', dash: true, data: [[1, 1], [50, 50]] }, { tone: 1, data: curve(K => Math.min(50, v.f(K)), 1, 128, 128) }, { as: 'points', tone: 4, r: 5, data: [[v.K, Math.min(50, v.f(v.K))]] }] }],
  stats: v => [`speedup ${fmt(v.f(v.K), 2)}`, `efficiency ${pct(v.f(v.K) / v.K, 0)}`],
  takeaway: 'Beyond some size, added workers slow training down.',
});
export const RbCheck = truths({
  intro: CHECK,
  items: [
    ['PGD is generally stronger than FGSM.', true, 'It iterates and stays inside the limit.'],
    ['A confident softmax means the input is familiar.', false, 'Networks can be very confident far from their data.'],
    ['More data can reduce aleatoric uncertainty.', false, 'It is the noise in the data itself.'],
    ['Structured pruning speeds up ordinary hardware.', true, 'Whole channels go, so tensors get smaller.'],
    ['Distillation needs the teacher’s logits or probabilities.', true, 'They provide the soft targets.'],
  ],
  takeaway: 'Deployment asks more than accuracy: robustness, honest uncertainty, and cost.',
});

/* ---------- flows, energy and score models ---------- */
export const FgChangeLab = lab({
  intro: 'Stretch a standard normal. The density must fall in the same proportion, so that its area stays 1.',
  tex: R`p_X(x)=p_Z\Big(\frac{x-b}a\Big)\frac1{|a|}`,
  params: [{ k: 'a', label: 'scale $a$', min: 0.2, max: 4, step: 0.05, init: 2 }, { k: 'b', label: 'shift $b$', min: -3, max: 3, step: 0.05, init: 0 }],
  plots: v => [{ x: [-8, 8], y: [0, 2.1], w: 480, h: 260, xl: 'x', yl: 'density', label: 'density before and after', marks: [{ tone: 'muted', dash: true, data: curve(x => normalPdf(x, 0, 1), -8, 8) }, { tone: 1, data: curve(x => normalPdf((x - v.b) / v.a, 0, 1) / v.a, -8, 8, 401) }] }],
  stats: v => [`log-determinant ${fmt(Math.log(v.a), 3)}`, `height at the centre ${fmt(0.3989 / v.a, 3)}`],
  takeaway: 'The log-determinant is the bookkeeping that keeps a flow’s density valid.',
});
const GRIDZ = linspace(-2, 2, 11).flatMap(a => linspace(-2, 2, 11).map(b => [a, b]));
export const FgCouplingLab = lab({
  intro: 'One coupling layer. The first coordinate stays fixed, and it controls a scale and shift of the second.',
  tex: R`y_2=x_2e^{s(x_1)}+t(x_1)`,
  params: [{ k: 'a', label: 'scale strength', min: -1, max: 1, step: 0.02, init: 0.5 }, { k: 'b', label: 'shift strength', min: -2, max: 2, step: 0.02, init: 1 }],
  derive: ({ a, b }) => ({ P: GRIDZ.map(([x, y]) => [x, y * Math.exp(a * Math.sin(x)) + b * Math.sin(2 * x)]), s: x => a * Math.sin(x) }),
  plots: v => [{ caption: 'Before', x: [-3, 3], y: [-6, 6], w: 300, h: 260, label: 'grid before', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: GRIDZ }] }, { caption: 'After', x: [-3, 3], y: [-6, 6], w: 300, h: 260, label: 'grid after', marks: [{ as: 'points', tone: 1, r: 2.5, data: v.P.map(([x, y]) => [x, clip(y, -6, 6)]) }] }],
  stats: v => [`log-determinant at $x_1=1$: ${fmt(v.s(1), 3)}`],
  takeaway: 'A triangular Jacobian makes the determinant cheap, and the network inside can be anything.',
});
export const FgLogDetDrill = drill({
  intro: 'A coupling layer’s log-determinant is the sum of its scale outputs.',
  gen: r => ({ s: Array.from({ length: 4 }, () => int(r, -8, 8) / 4) }),
  q: v => `The scale outputs are $${v.s.join(', ')}$. Find the log-determinant.`,
  answer: v => v.s.reduce((a, b) => a + b, 0), dp: 2,
  steps: () => ['Add the outputs.'],
});
export const FgDensityDrill = drill({
  intro: 'Density after an invertible scaling.',
  tex: R`p_X(x)=p_Z(x/a)/a`,
  gen: r => ({ a: pick(r, [2, 3, 4]), x: pick(r, [0, 1, 2]) }),
  q: v => `$x=${v.a}z$, with $z$ standard normal. Find $p_X(${v.x})$.`,
  answer: v => normalPdf(v.x / v.a, 0, 1) / v.a, dp: 3,
  steps: v => [`$p_Z(${fmt(v.x / v.a, 3)})=${fmt(normalPdf(v.x / v.a, 0, 1), 3)}$`],
});
export const FgLangevinLab = lab({
  intro: 'Sample from a double well using only its gradient, with some noise.',
  tex: R`x\leftarrow x+\eta\nabla\log p(x)+\sqrt{2\eta}\,\xi`,
  params: [{ k: 's', label: 'steps', min: 0, max: 800, step: 10, init: 300 }, { k: 'e', label: 'step size $\\eta$', min: 0.001, max: 0.2, step: 0.001, init: 0.01 }, { k: 'seed', seed: 'New chains', init: 1 }],
  derive: ({ s, e, seed }) => { const r = rng(seed * 29); let X = Array.from({ length: 600 }, () => 0.1 * gauss(r)); for (let i = 0; i < s; i++) X = X.map(x => clip(r6(x - 4 * e * x * (x * x - 1) + Math.sqrt(2 * e) * gauss(r)), -6, 6)); return { X }; },
  plots: v => { const g = linspace(-3, 3, 301), Z = g.reduce((a, x) => a + Math.exp(-((x * x - 1) ** 2)) * 0.02, 0); return [{ x: [-3, 3], y: [0, 1.2], w: 480, h: 260, xl: 'x', yl: 'density', label: 'samples and target', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.X, -3, 3, 40).map(b => [b[0], 0, b[2], Math.min(1.2, b[3] / (600 * 0.15))]) }, { tone: 4, data: g.map(x => [x, Math.exp(-((x * x - 1) ** 2)) / Z]) }] }]; },
  stats: v => [`second moment ${fmt(mean(v.X.map(x => x * x)), 3)}`, 'target 0.833'],
  takeaway: 'Too large a step distorts the distribution, and too few steps leave chains near their start.',
});
export const FgEnergyDrill = drill({
  intro: 'Energy differences give probability ratios, with no need for $Z$.',
  tex: R`\frac{p(a)}{p(b)}=e^{E(b)-E(a)}`,
  gen: r => ({ a: int(r, 0, 5), b: int(r, 0, 5) }),
  q: v => `$E(a)=${v.a}$, $E(b)=${v.b}$. Find $p(a)/p(b)$.`,
  answer: v => Math.exp(v.b - v.a), dp: 3,
  steps: v => [`Exponent: $${v.b - v.a}$`],
});
export const FgScoreDrill = drill({
  intro: 'The score of a Gaussian points back to its mean.',
  tex: R`s(x)=-\frac{x-\mu}{\sigma^2}`,
  gen: r => ({ x: int(r, -4, 4), m: int(r, -3, 3), s: pick(r, [0.5, 1, 2]) }),
  q: v => `$x=${v.x}$, $\\mu=${v.m}$, $\\sigma=${v.s}$. Find the score.`,
  answer: v => -(v.x - v.m) / (v.s * v.s), dp: 3,
  steps: v => [`$x-\\mu=${v.x - v.m}$`],
});
export const FgTargetDrill = drill({
  intro: 'Denoising score matching has a simple target.',
  tex: R`-\frac\varepsilon\sigma`,
  gen: r => ({ e: pick(r, [-1.5, -0.5, 0.5, 2]), s: pick(r, [0.1, 0.5, 1]) }),
  q: v => `$\\varepsilon=${v.e}$, $\\sigma=${v.s}$. Find the regression target.`,
  answer: v => -v.e / v.s, dp: 3,
  steps: () => ['Divide, and change the sign.'],
});
export const FgScoreFieldLab = lab({
  intro: 'The score of two noised peaks. More noise smooths the field, so a sampler can cross between the peaks.',
  tex: R`\nabla_x\log p_\sigma(x)`,
  params: [{ k: 's', label: 'noise $\\sigma$', min: 0.05, max: 3, step: 0.05, init: 0.5 }],
  derive: ({ s }) => { const v = 0.04 + s * s; return { score: x => { const a = Math.exp(-((x + 2) ** 2) / (2 * v)), b = Math.exp(-((x - 2) ** 2) / (2 * v)), t = a + b || 1e-300; return (a * -(x + 2) + b * -(x - 2)) / (v * t); }, dens: x => 0.5 * normalPdf(x, -2, Math.sqrt(v)) + 0.5 * normalPdf(x, 2, Math.sqrt(v)) }; },
  plots: v => [{ caption: 'Density', x: [-5, 5], y: [0, 1.1], w: 480, h: 170, xl: 'x', label: 'noised density', marks: [{ tone: 2, data: curve(v.dens, -5, 5) }] }, { caption: 'Score', x: [-5, 5], y: [-10, 10], w: 480, h: 190, xl: 'x', label: 'score', marks: [{ tone: 1, data: curve(x => clip(v.score(x), -10, 10), -5, 5, 401) }, { as: 'rule', dash: true, tone: 'muted', y: 0 }] }],
  stats: v => [`score at x = 1: ${fmt(v.score(1), 3)}`],
  takeaway: 'Sampling at high noise finds the modes, and low noise refines them.',
});
export const FgSdeLab = lab({
  intro: 'Generate from an exactly known score. Compare noisy reverse paths with the deterministic probability flow.',
  tex: R`\frac{dx}{dt}=-\tfrac12\beta x-\tfrac12\beta\,\nabla\log p_t(x)`,
  params: [{ k: 'k', label: 'Sampler', init: 'ode', options: [['ode', 'Probability flow'], ['sde', 'Reverse SDE']] }, { k: 'n', label: 'steps', min: 5, max: 400, step: 5, init: 100 }, { k: 'seed', seed: 'New noise', init: 1 }],
  derive: ({ k, n, seed }) => {
    const r = rng(seed * 31 + 2), m0 = 2, s0 = 0.5, beta = t => 0.1 + 19.9 * t, Bi = t => 0.1 * t + 9.95 * t * t, score = (x, t) => { const a = Math.exp(-0.5 * Bi(t)), v = a * a * s0 * s0 + 1 - a * a; return -(x - a * m0) / v; }, dt = 1 / n, P = [], E = [];
    for (let i = 0; i < 300; i++) { let x = gauss(r); const path = [[1, r6(x)]]; for (let j = 0; j < n; j++) { const t = 1 - j * dt, b = beta(t); x = clip(k === 'ode' ? x + dt * (0.5 * b * x + 0.5 * b * score(x, t)) : x + dt * (0.5 * b * x + b * score(x, t)) + Math.sqrt(b * dt) * gauss(r), -30, 30); if (i < 24) path.push([r6(t - dt), r6(x)]); } if (i < 24) P.push(path); E.push(r6(x)); }
    return { P, E };
  },
  plots: v => [{ x: [0, 1], y: [-4, 4], w: 480, h: 280, xl: 't (0 is data, 1 is noise)', yl: 'x', label: 'sample paths', marks: v.P.map(p => ({ tone: 1, data: p.map(([t, x]) => [t, clip(x, -4, 4)]) })) }],
  stats: v => [`mean ${fmt(mean(v.E), 3)}`, `spread ${fmt(std(v.E), 3)}`, 'target 2.000 and 0.500'],
  takeaway: 'Both share the same marginals, and the deterministic version gets by with fewer steps here.',
});
export const FgConsistencyLab = lab({
  intro: 'A consistency model maps any point on a noise trajectory straight to its endpoint. Here that map is known exactly.',
  tex: R`f(x,\sigma)=\mu+(x-\mu)\frac{s_0}{\sqrt{s_0^2+\sigma^2}}`,
  params: [{ k: 'n', label: 'evaluations', min: 1, max: 4, step: 1, init: 1 }, { k: 's', label: 'starting noise $\\sigma$', min: 1, max: 20, step: 0.5, init: 10 }, { k: 'seed', seed: 'New noise', init: 1 }],
  derive: ({ n, s, seed }) => { const r = rng(seed * 11), m = 2, s0 = 0.5, f = (x, g) => m + (x - m) * (s0 / Math.sqrt(s0 * s0 + g * g)); const X = Array.from({ length: 800 }, () => { let x = m + Math.sqrt(s0 * s0 + s * s) * gauss(r), g = s; x = f(x, g); for (let k = 1; k < n; k++) { g = s * 0.3 ** k; x = f(x + g * gauss(r), g); } return r6(x); }); return { X }; },
  plots: v => [{ x: [0, 4], y: [0, 200], w: 480, h: 260, xl: 'x', yl: 'count', label: 'generated samples', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.X, 0, 4, 40).map(b => [b[0], 0, b[2], Math.min(200, b[3])]) }, { as: 'rule', dash: true, tone: 4, x: 2 }] }],
  stats: v => [`mean ${fmt(mean(v.X), 3)}`, `spread ${fmt(std(v.X), 3)}`, 'target 2.000 and 0.500'],
  takeaway: 'Learning this map turns many sampler steps into one network evaluation.',
});
export const FgCheck = truths({
  intro: CHECK,
  items: [
    ['A normalizing flow gives an exact likelihood.', true, 'It uses the change of variables.'],
    ['Langevin sampling needs the normalising constant.', false, 'It uses only the gradient of the log density.'],
    ['A coupling layer needs an invertible network inside.', false, 'Only the coupling structure must be invertible.'],
    ['The probability flow ODE has the same marginals as the reverse SDE.', true, 'This is what allows deterministic sampling.'],
    ['Consistency models need many sampling steps.', false, 'They are designed for one or a few.'],
  ],
  takeaway: 'Density, energy, score and velocity are different handles on the same distribution.',
});

/* ---------- theory, changing tasks and science ---------- */
export const ThNtkLab = lab({
  intro: 'In the kernel regime each eigendirection is learned on its own schedule.',
  tex: R`e_i(t)=e^{-\lambda_it}\,e_i(0)`,
  params: [{ k: 'a', label: 'large eigenvalue', min: 0.5, max: 10, step: 0.1, init: 4 }, { k: 'b', label: 'small eigenvalue', min: 0.01, max: 1, step: 0.01, init: 0.1 }, { k: 't', label: 'time', min: 0, max: 20, step: 0.1, init: 2 }],
  plots: v => [{ x: [0, 20], y: [0, 1], w: 480, h: 260, xl: 'time', yl: 'remaining error', label: 'error along each direction', marks: [{ tone: 1, data: curve(t => Math.exp(-v.a * t), 0, 20) }, { tone: 2, data: curve(t => Math.exp(-v.b * t), 0, 20) }, { as: 'rule', dash: true, tone: 4, x: v.t }] }],
  stats: v => [`fast direction ${fmt(Math.exp(-v.a * v.t), 4)}`, `slow direction ${fmt(Math.exp(-v.b * v.t), 4)}`],
  legend: [['--c1', 'large eigenvalue'], ['--c2', 'small eigenvalue']],
  takeaway: 'Smooth patterns usually have large eigenvalues, which is one account of why networks learn them first.',
});
export const ThKernelDrill = drill({
  intro: 'The tangent kernel is a dot product of parameter gradients.',
  tex: R`\Theta(x,x')=\nabla_wf(x)^\top\nabla_wf(x')`,
  gen: r => ({ a: [int(r, -2, 3), int(r, -2, 3), int(r, -2, 3)], b: [int(r, -2, 3), int(r, -2, 3), int(r, -2, 3)] }),
  q: v => `The gradients at two inputs are $(${v.a})$ and $(${v.b})$. Find the kernel value.`,
  answer: v => v.a.reduce((s, x, i) => s + x * v.b[i], 0), dp: 0,
  steps: () => ['Multiply matching entries, then add.'],
});
export const ThLotteryLab = lab({
  intro: 'Iterative pruning removes a fixed share of the remaining weights each round.',
  tex: R`\text{remaining}=(1-p)^k`,
  params: [{ k: 'p', label: 'pruned per round', min: 0.05, max: 0.6, step: 0.01, init: 0.2 }, { k: 'k', label: 'rounds', min: 0, max: 25, step: 1, init: 10 }],
  plots: v => [{ x: [0, 25], y: [0, 1], w: 480, h: 260, xl: 'round', yl: 'share remaining', label: 'weights remaining', marks: [{ tone: 1, data: curve(k => (1 - v.p) ** k, 0, 25) }, { as: 'rule', dash: true, tone: 4, x: v.k }] }],
  stats: v => [`${pct((1 - v.p) ** v.k, 1)} of weights remain`],
  takeaway: 'Gentle iterative pruning finds sparser subnetworks than one large cut, at more training cost.',
});
export const ThRoundsDrill = drill({
  intro: 'Remaining weights after iterative pruning.',
  gen: r => ({ p: pick(r, [0.1, 0.2, 0.3]), k: int(r, 2, 12) }),
  q: v => `Prune $${v.p}$ of the remaining weights in each of $${v.k}$ rounds. What share remains?`,
  answer: v => (1 - v.p) ** v.k, dp: 3,
  steps: v => [`$${fmt(1 - v.p, 1)}^{${v.k}}$`],
});
export const ThDeepSetsLab = lab({
  intro: 'Shuffle the same five numbers. A sum-based model does not change, and a model that reads positions does.',
  tex: R`f(X)=\rho\Big(\sum_x\phi(x)\Big)`,
  params: [{ k: 'seed', seed: 'Shuffle', init: 1 }],
  derive: ({ seed }) => { const r = rng(seed * 7), a = [0.3, 1.1, -0.8, 2.0, -1.4]; for (let i = 4; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } const phi = x => x * x + Math.sin(x); return { a, set: Math.tanh(a.reduce((s, x) => s + phi(x), 0) / 5), pos: Math.tanh(a.reduce((s, x, i) => s + (i + 1) * phi(x), 0) / 15) }; },
  plots: v => [{ x: [0, 5], y: [-2, 2.5], w: 480, h: 240, xl: 'position', label: 'the set, in this order', marks: [{ as: 'rects', tone: 1, data: v.a.map((x, i) => [i + 0.1, Math.min(0, x), i + 0.9, Math.max(0, x)]) }] }],
  stats: v => [`sum-based output ${fmt(v.set, 4)}`, `position-based output ${fmt(v.pos, 4)}`],
  takeaway: 'Build the symmetry into the architecture, and the model never needs to learn it.',
});
export const ThMamlLab = lab({
  intro: 'Tasks are bowls with different centres and curvature. MAML finds a starting point that does well after one step.',
  tex: R`\min_w\sum_iL_i\big(w-\alpha L_i'(w)\big)`,
  params: [{ k: 'a', label: 'inner rate $\\alpha$', min: 0, max: 0.9, step: 0.01, init: 0.3 }, { k: 'w', label: 'starting point', min: -4, max: 4, step: 0.05, init: 0 }],
  derive: ({ a }) => { const T = [[-3, 1], [2, 3], [4, 0.5]], L = (w, [c, k]) => 0.5 * k * (w - c) ** 2, after = (w, t) => L(w - a * t[1] * (w - t[0]), t), M = w => mean(T.map(t => after(w, t))), B = w => mean(T.map(t => L(w, t))); return { T, M, B }; },
  plots: v => [{ x: [-5, 5], y: [0, 24], w: 480, h: 260, xl: 'start', yl: 'loss', label: 'average loss before and after adapting', marks: [{ tone: 'muted', dash: true, data: curve(w => Math.min(24, v.B(w)), -5, 5) }, { tone: 1, data: curve(w => Math.min(24, v.M(w)), -5, 5) }, { as: 'rule', dash: true, tone: 4, x: v.w }] }],
  stats: v => [`loss before adapting ${fmt(v.B(v.w), 3)}`, `after one step ${fmt(v.M(v.w), 3)}`],
  takeaway: 'The best start for later adaptation is usually not the best overall average.',
});
export const ThMetaDrill = drill({
  intro: 'The outer gradient of MAML on a scalar quadratic.',
  tex: R`\frac{\partial L(w')}{\partial w}=(1-\alpha h)\,L'(w')`,
  gen: r => ({ a: pick(r, [0.01, 0.1, 0.2]), h: int(r, 1, 4), g: int(r, -3, 3) }),
  q: v => `$\\alpha=${v.a}$, inner curvature $h=${v.h}$, gradient at the adapted weight $${v.g}$. Find the meta-gradient.`,
  answer: v => (1 - v.a * v.h) * v.g, dp: 3,
  steps: v => [`Factor: $${fmt(1 - v.a * v.h, 2)}$`],
});
export const ThEwcDrill = drill({
  intro: 'Important weights are expensive to move.',
  tex: R`\frac\lambda2F(w-w^*)^2`,
  gen: r => ({ l: pick(r, [1, 10, 100]), f: pick(r, [0.1, 0.5, 2]), d: pick(r, [0.5, 1, 3]) }),
  q: v => `$\\lambda=${v.l}$, $F=${v.f}$, weight moved by $${v.d}$. Find the penalty.`,
  answer: v => 0.5 * v.l * v.f * v.d * v.d, dp: 3,
  steps: v => [`$(w-w^*)^2=${v.d * v.d}$`],
});
export const ThForgetLab = lab({
  intro: 'Two tasks with different optimal weights. After learning the first, learn the second. A penalty protects the direction the first task needs.',
  tex: R`L=L_B(w)+\frac\lambda2\sum_iF_i(w_i-w_{A,i})^2`,
  params: [{ k: 'l', label: 'penalty $\\lambda$', min: 0, max: 40, step: 0.5, init: 0 }],
  derive: ({ l }) => { const A = [2, 0], FA = [6, 0.1], B = [-1, 3], FB = [0.1, 6]; const LA = w => 0.5 * (FA[0] * (w[0] - A[0]) ** 2 + FA[1] * (w[1] - A[1]) ** 2), LB = w => 0.5 * (FB[0] * (w[0] - B[0]) ** 2 + FB[1] * (w[1] - B[1]) ** 2); let w = [...A]; const P = [w]; for (let i = 0; i < 200; i++) { w = [0, 1].map(j => w[j] - (0.6 / (1 + l * 6 + 6)) * (FB[j] * (w[j] - B[j]) + l * FA[j] * (w[j] - A[j]))); if (i % 4 === 0) P.push(w); } return { P, w, a: LA(w), b: LB(w), A, B }; },
  plots: v => [{ x: [-2, 4], y: [-1, 4], w: 460, h: 300, label: 'path of the weights', marks: [{ tone: 1, data: v.P }, { as: 'points', tone: 2, r: 6, data: [v.A] }, { as: 'points', tone: 4, r: 6, data: [v.B] }] }],
  stats: v => [`loss on the old task ${fmt(v.a, 3)}`, `on the new task ${fmt(v.b, 3)}`],
  legend: [['--c2', 'first task'], ['--c4', 'second task']],
  takeaway: 'Each weight moves only where the old task did not care.',
});
export const ThEulerLab = lab({
  intro: 'More layers mean a smaller step along the same continuous flow.',
  tex: R`h_{t+1}=h_t+\frac1Nf(h_t)`,
  params: [{ k: 'n', label: 'layers $N$', min: 2, max: 200, step: 1, init: 12 }],
  derive: ({ n }) => { const T = 2 * Math.PI, d = T / n; let x = [1, 0]; const P = [x]; for (let i = 0; i < n; i++) { x = [x[0] - d * x[1], x[1] + d * x[0]]; P.push(x); } return { P, err: Math.hypot(x[0] - 1, x[1]) }; },
  plots: v => [{ x: [-4, 4], y: [-4, 4], w: 340, h: 320, label: 'Euler path against the exact circle', marks: [{ tone: 'muted', dash: true, data: linspace(0, 6.3, 121).map(t => [Math.cos(t), Math.sin(t)]) }, { tone: 1, data: v.P.map(([a, b]) => [clip(a, -4, 4), clip(b, -4, 4)]) }] }],
  stats: v => [`error after one turn ${fmt(v.err, 3)}`],
  takeaway: 'A residual network is a coarse ODE solver, and an ODE solver can adapt its own step size.',
});
export const ThEulerDrill = drill({
  intro: 'One Euler step is one residual block.',
  tex: R`h'=h+\Delta f(h)`,
  gen: r => ({ h: int(r, -3, 4), d: pick(r, [0.1, 0.5, 1]), f: int(r, -3, 3) }),
  q: v => `$h=${v.h}$, $f(h)=${v.f}$, step $${v.d}$. Find the next value.`,
  answer: v => v.h + v.d * v.f, dp: 2,
  steps: v => [`Change: $${fmt(v.d * v.f, 2)}$`],
});
export const ThPinnLab = lab({
  intro: 'A two-coefficient trial function for $u^\\prime=-u$, with $u(0)=1$. Reduce the equation’s residual by hand.',
  tex: R`u(t)=1+at+bt^2,\qquad r(t)=u'(t)+u(t)`,
  params: [{ k: 'a', label: '$a$', min: -2, max: 1, step: 0.01, init: 0 }, { k: 'b', label: '$b$', min: -1, max: 1, step: 0.01, init: 0 }],
  derive: ({ a, b }) => { const u = t => 1 + a * t + b * t * t, r = t => a + 2 * b * t + u(t), T = linspace(0, 1, 101); return { u, r, loss: mean(T.map(t => r(t) ** 2)) }; },
  plots: v => [{ x: [0, 1], y: [-0.5, 2], w: 480, h: 260, xl: 't', yl: 'u', label: 'trial function and exact solution', marks: [{ tone: 'muted', dash: true, data: curve(t => Math.exp(-t), 0, 1) }, { tone: 1, data: curve(t => clip(v.u(t), -0.5, 2), 0, 1) }, { tone: 4, data: curve(t => clip(v.r(t), -0.5, 2), 0, 1) }] }],
  stats: v => [`residual loss ${fmt(v.loss, 5)}`, `u(1) ${fmt(v.u(1), 3)}, exact 0.368`],
  legend: [['--c1', 'trial function'], ['--c4', 'residual']],
  takeaway: 'The loss needs no solution data, only the equation itself and its starting value.',
});
export const ThTokenDrill = drill({
  intro: 'An FT-Transformer learns a weight and a bias vector for each number column.',
  tex: R`2pd+d`,
  gen: r => ({ p: int(r, 5, 60), d: pick(r, [32, 64, 192]) }),
  q: v => `$${v.p}$ numeric features, token size $${v.d}$. How many tokeniser parameters, including one class token?`,
  answer: v => 2 * v.p * v.d + v.d, dp: 0,
  steps: v => [`Feature tokens: $${2 * v.p * v.d}$`],
});
export const ThCheck = truths({
  intro: CHECK,
  items: [
    ['In the kernel regime the features change greatly during training.', false, 'The kernel stays nearly constant.'],
    ['Deep Sets are invariant to reordering of the elements.', true, 'A sum ignores order.'],
    ['Catastrophic forgetting affects only small networks.', false, 'It is a general effect of sequential training.'],
    ['Physics-informed networks always beat classical solvers.', false, 'Classical solvers are often better on standard equations.'],
    ['Boosted trees are invariant to rescaling a feature.', true, 'They depend only on the ordering.'],
  ],
  takeaway: 'Theory explains regimes, and structure in the problem decides the design.',
});

/* ---------- control, 3D and interpretability ---------- */
const QV = [2, 1.5, 0.5, -1];
export const CiEntropyLab = lab({
  intro: 'A soft policy turns action values into probabilities. The temperature $\\alpha$ prices exploration.',
  tex: R`\pi(a)\propto e^{Q(a)/\alpha}`,
  params: [{ k: 'a', label: 'temperature $\\alpha$', min: 0.05, max: 5, step: 0.05, init: 0.5 }],
  derive: ({ a }) => { const P = softmaxT(QV, a), H = -P.reduce((s, p) => s + p * Math.log(p), 0); return { P, H, V: P.reduce((s, p, i) => s + p * QV[i], 0) + a * H }; },
  plots: v => [{ x: [0, 4], y: [0, 1], w: 480, h: 260, xl: 'action', yl: 'probability', label: 'policy', marks: [{ as: 'rects', tone: 1, data: v.P.map((p, i) => [i + 0.1, 0, i + 0.9, p]) }] }],
  stats: v => [`entropy ${fmt(v.H, 3)}`, `soft value ${fmt(v.V, 3)}`],
  takeaway: 'Higher temperature spreads actions and keeps more options open.',
});
export const CiSoftDrill = drill({
  intro: 'The soft Bellman target includes an entropy bonus.',
  tex: R`y=r+\gamma\big(Q-\alpha\log\pi\big)`,
  gen: r => ({ r: int(r, -2, 3), g: pick(r, [0.9, 0.99]), q: int(r, 0, 9), a: pick(r, [0.1, 0.2, 0.5]), l: pick(r, [-2, -1, -0.5]) }),
  q: v => `$r=${v.r}$, $\\gamma=${v.g}$, $Q=${v.q}$, $\\alpha=${v.a}$, $\\log\\pi=${v.l}$. Find the target.`,
  answer: v => v.r + v.g * (v.q - v.a * v.l), dp: 3,
  steps: v => [`Bonus: $${fmt(-v.a * v.l, 3)}$`],
});
export const CiTanhDrill = drill({
  intro: 'Squashing an action changes its density.',
  tex: R`\log\pi(a)=\log\mathcal N(u)-\log(1-\tanh^2u)`,
  gen: r => ({ u: pick(r, [0, 0.5, 1, 1.5]), n: pick(r, [-1, -1.5, -2]) }),
  q: v => `$u=${v.u}$ and $\\log\\mathcal N(u)=${v.n}$. Find $\\log\\pi(a)$.`,
  answer: v => v.n - Math.log(1 - Math.tanh(v.u) ** 2), dp: 3,
  steps: v => [`$\\tanh(u)=${fmt(Math.tanh(v.u), 3)}$`],
});
export const CiOfflineLab = lab({
  intro: 'A log contains only action A. Action B has no data, and starts with an unjustified high value.',
  tex: R`Q(s,\cdot)\leftarrow Q(s,\cdot)-\alpha\big(\pi(\cdot)-\mathbb 1_{\text{data}}\big)`,
  params: [{ k: 'q', label: 'starting value of the unseen action', min: 0, max: 20, step: 0.5, init: 15 }, { k: 'c', label: 'conservative weight', min: 0, max: 2, step: 0.05, init: 0 }],
  derive: ({ q, c }) => { let a = 0, b = q; const H = []; for (let t = 0; t <= 300; t++) { H.push([t, a, b]); const best = Math.max(a, b); a += 0.05 * (1 + 0.9 * best - a); if (c) { const m = Math.max(a, b), pa = Math.exp(a - m), pb = Math.exp(b - m), z = pa + pb; a += 0.05 * c * (1 - pa / z); b -= 0.05 * c * (pb / z); } } return { H, a, b }; },
  plots: v => [{ x: [0, 300], y: [-5, 25], w: 480, h: 260, xl: 'sweep', yl: 'value', label: 'value of each action', marks: [{ tone: 1, data: v.H.map(([t, a]) => [t, clip(a, -5, 25)]) }, { tone: 4, data: v.H.map(([t, , b]) => [t, clip(b, -5, 25)]) }] }],
  stats: v => [`value of the logged action ${fmt(v.a, 2)}`, `unseen action ${fmt(v.b, 2)}`],
  status: v => (v.b > v.a ? ['alert', 'The policy would choose the action it has no evidence for.'] : ['ok', 'The policy stays with supported actions.']),
  legend: [['--c1', 'logged action'], ['--c4', 'unseen action']],
  takeaway: 'Offline, an error in an unseen action is never corrected, so values must be held down.',
});
const dynaRun = (plan, seed) => { const r = rng(seed), N = 10, Q = Array.from({ length: N }, () => [0, 0]), model = new Map(), out = []; let total = 0; for (let ep = 0; ep < 12; ep++) { let s = 0, t = 0; for (; t < 300; t++) { const a = r() < 0.1 ? (r() < 0.5 ? 0 : 1) : Q[s][1] + 1e-6 * r() > Q[s][0] + 1e-6 * r() ? 1 : 0, n = Math.max(0, Math.min(N - 1, s + (a ? 1 : -1))), rew = n === N - 1 ? 1 : 0, done = n === N - 1; Q[s][a] += 0.5 * (rew + (done ? 0 : 0.95 * Math.max(...Q[n])) - Q[s][a]); model.set(s * 2 + a, [rew, n, done]); total++; const K = [...model.keys()]; for (let p = 0; p < plan; p++) { const key = K[Math.floor(r() * K.length)], [pr, pn, pd] = model.get(key), ps = key >> 1, pa = key & 1; Q[ps][pa] += 0.5 * (pr + (pd ? 0 : 0.95 * Math.max(...Q[pn])) - Q[ps][pa]); } s = n; if (done) break; } out.push([ep + 1, t + 1]); } return { out, total }; };
export const CiDynaLab = lab({
  intro: 'A corridor with the reward at the end. Each real step can be followed by imagined updates from a learned model.',
  params: [{ k: 'p', label: 'imagined updates per step', min: 0, max: 50, step: 1, init: 10 }, { k: 'seed', seed: 'New runs', init: 1 }],
  derive: ({ p, seed }) => { const R0 = Array.from({ length: 10 }, (_, i) => dynaRun(p, seed * 100 + i)), Z = Array.from({ length: 10 }, (_, i) => dynaRun(0, seed * 100 + i)), av = L => L[0].out.map(([e], j) => [e, mean(L.map(x => x.out[j][1]))]); return { A: av(R0), B: av(Z), total: mean(R0.map(x => x.total)), base: mean(Z.map(x => x.total)) }; },
  plots: v => [{ x: [1, 12], y: [0, 300], w: 480, h: 260, xl: 'episode', yl: 'steps', label: 'steps per episode', marks: [{ tone: 'muted', data: v.B }, { tone: 1, data: v.A }] }],
  stats: v => [`real steps with planning ${fmt(v.total, 0)}`, `without ${fmt(v.base, 0)}`],
  takeaway: 'A model lets each real experience be reused many times.',
});
const cemCost = x => (x - 1.5) ** 2 + 0.6 * Math.sin(5 * x) + 1;
export const CiCemLab = lab({
  intro: 'The cross-entropy method samples candidates, keeps the best, and refits its sampling distribution.',
  params: [{ k: 'i', label: 'iterations', min: 0, max: 12, step: 1, init: 3, play: true }, { k: 'e', label: 'share kept', min: 0.05, max: 0.5, step: 0.01, init: 0.2 }, { k: 'seed', seed: 'New run', init: 1 }],
  derive: ({ i, e, seed }) => { const r = rng(seed * 9); let m = -2, s = 2.5, X = []; for (let k = 0; k <= i; k++) { X = Array.from({ length: 120 }, () => r6(m + s * gauss(r))); if (k === i) break; const best = [...X].sort((a, b) => cemCost(a) - cemCost(b)).slice(0, Math.max(2, Math.round(e * 120))); m = mean(best); s = std(best) + 0.01; } return { m, s, X }; },
  plots: v => [{ x: [-5, 5], y: [0, 40], w: 480, h: 260, xl: 'action', yl: 'cost', label: 'cost and samples', marks: [{ tone: 1, data: curve(x => Math.min(40, cemCost(x)), -5, 5, 301) }, { as: 'points', tone: 4, r: 2.5, data: v.X.map(x => [x, Math.min(40, cemCost(x))]) }] }],
  stats: v => [`mean ${fmt(v.m, 3)}`, `spread ${fmt(v.s, 3)}`, `cost ${fmt(cemCost(v.m), 3)}`],
  takeaway: 'No gradients are needed, so this works with any learned model used for planning.',
});
export const CiRenderLab = lab({
  intro: 'A ray passing through a fog bank and a solid wall. Each segment adds light in proportion to what still reaches it.',
  tex: R`\hat C=\sum_iT_i\alpha_ic_i,\quad\alpha_i=1-e^{-\sigma_i\delta_i}`,
  params: [{ k: 'f', label: 'fog density', min: 0, max: 3, step: 0.05, init: 0.4 }, { k: 'w', label: 'wall density', min: 0, max: 40, step: 0.5, init: 25 }, { k: 'p', label: 'wall position', min: 1, max: 9, step: 0.1, init: 6 }],
  derive: ({ f, w, p }) => { const n = 200, d = 10 / n, S = [], A = []; let T = 1, C = 0; for (let i = 0; i < n; i++) { const t = (i + 0.5) * d, s = (t > 1 && t < 4 ? f : 0) + (t > p && t < p + 0.3 ? w : 0), c = t > p && t < p + 0.3 ? 1 : 0.3, a = 1 - Math.exp(-s * d); S.push([t, T]); A.push([t, s]); C += T * a * c; T *= 1 - a; } return { S, A, C, T }; },
  plots: v => [{ x: [0, 10], y: [0, 1.05], w: 480, h: 260, xl: 'distance along the ray', yl: 'transmittance', label: 'light reaching each point', marks: [{ tone: 1, data: v.S }] }],
  stats: v => [`rendered colour ${fmt(v.C, 3)}`, `remaining transmittance ${fmt(v.T, 3)}`],
  takeaway: 'Every step is differentiable, so errors in a pixel can flow back to densities and colours.',
});
export const CiAlphaDrill = drill({
  intro: 'A segment’s opacity follows from density and length.',
  tex: R`\alpha=1-e^{-\sigma\delta}`,
  gen: r => ({ s: pick(r, [0.5, 1, 2, 5]), d: pick(r, [0.1, 0.25, 0.5]) }),
  q: v => `$\\sigma=${v.s}$, $\\delta=${v.d}$. Find $\\alpha$.`,
  answer: v => 1 - Math.exp(-v.s * v.d), dp: 3,
  steps: v => [`$\\sigma\\delta=${fmt(v.s * v.d, 3)}$`],
});
export const CiCompositeDrill = drill({
  intro: 'Light from behind is blocked by what is in front.',
  tex: R`\hat C=\alpha_1c_1+(1-\alpha_1)\alpha_2c_2`,
  gen: r => ({ a: pick(r, [0.2, 0.5, 0.8]), b: pick(r, [0.4, 1]), c: pick(r, [0.2, 0.6, 1]), d: pick(r, [0, 0.5, 1]) }),
  q: v => `$\\alpha_1=${v.a}$, $c_1=${v.c}$, $\\alpha_2=${v.b}$, $c_2=${v.d}$. Find the rendered colour.`,
  answer: v => v.a * v.c + (1 - v.a) * v.b * v.d, dp: 3,
  steps: v => [`Front: $${fmt(v.a * v.c, 3)}$`],
});
const FX = linspace(0, 1, 120), FY = FX.map(x => 0.5 * Math.sin(40 * Math.PI * x) + Math.sin(4 * Math.PI * x));
export const CiFourierLab = lab({
  intro: 'A linear fit on Fourier features. Add more bands, and detail appears only when a matching frequency is available.',
  tex: R`\gamma(x)=\big(\sin(2^k\pi x),\cos(2^k\pi x)\big)`,
  params: [{ k: 'k', label: 'frequency bands', min: 1, max: 8, step: 1, init: 3 }],
  derive: ({ k }) => { const phi = x => [1, ...Array.from({ length: k }, (_, i) => [Math.sin(2 ** i * Math.PI * x), Math.cos(2 ** i * Math.PI * x)]).flat()], A = FX.map(phi), n = A[0].length, M = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-6 : 0))), b = Array.from({ length: n }, (_, i) => A.reduce((s, r, q) => s + r[i] * FY[q], 0)), c = solve(M, b), f = x => phi(x).reduce((s, v, i) => s + v * c[i], 0); return { f, mse: mean(FX.map((x, i) => (f(x) - FY[i]) ** 2)) }; },
  plots: v => [{ x: [0, 1], y: [-2, 2], w: 480, h: 260, xl: 'x', label: 'target and fit', marks: [{ tone: 'muted', data: FX.map((x, i) => [x, FY[i]]) }, { tone: 1, data: curve(x => clip(v.f(x), -2, 2), 0, 1, 481) }] }],
  stats: v => [`error ${v.mse.toExponential(2)}`],
  takeaway: 'High-frequency inputs give the network something to fit detail with.',
});
export const CiSplatDrill = drill({
  intro: 'Each Gaussian stores a position, a shape, a colour and an opacity.',
  gen: r => ({ n: pick(r, [100000, 500000, 2000000]) }),
  q: v => `$${v.n}$ Gaussians, each with 3 position, 3 scale, 4 rotation, 3 colour and 1 opacity numbers, at 4 bytes each. How many megabytes?`,
  answer: v => (v.n * 14 * 4) / 1e6, dp: 1,
  steps: () => ['Each one holds 14 numbers.'],
});
export const CiSuperpositionLab = lab({
  intro: 'Store many features in two dimensions, by spacing their directions evenly. Overlap between them is the price.',
  tex: R`\text{interference}=\max_{i\ne j}|\cos\theta_{ij}|`,
  params: [{ k: 'n', label: 'features', min: 2, max: 12, step: 1, init: 5 }, { k: 'p', label: 'chance a feature is active', min: 0.01, max: 1, step: 0.01, init: 0.05 }],
  derive: ({ n, p }) => { const D = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n), I = Math.max(...D.slice(1).map(a => Math.abs(Math.cos(a)))); return { D, I, co: 1 - (1 - p) ** (n - 1) }; },
  plots: v => [{ x: [-1.2, 1.2], y: [-1.2, 1.2], w: 340, h: 320, label: 'feature directions', marks: v.D.map(a => ({ as: 'arrow', tone: 1, data: [[0, 0, Math.cos(a), Math.sin(a)]] })) }],
  stats: v => [`largest overlap ${fmt(v.I, 3)}`, `chance another feature is also active ${pct(v.co, 1)}`],
  takeaway: 'If features are rarely active together, overlap seldom hurts, so models store more than they have dimensions.',
});
const SAEZ = (() => { const r = rng(14); return Array.from({ length: 400 }, () => Array.from({ length: 12 }, () => (r() < 0.1 ? r() : 0))); })();
export const CiSaeLab = lab({
  intro: 'Noisy feature readings are thresholded by the L1 penalty. Stronger sparsity means fewer active features, and worse reconstruction.',
  tex: R`\min_z\ \|h-z\|^2+\lambda\|z\|_1`,
  params: [{ k: 'l', label: 'penalty $\\lambda$', min: 0, max: 1, step: 0.01, init: 0.1 }, { k: 'n', label: 'read-out noise', min: 0, max: 0.2, step: 0.005, init: 0.05 }],
  derive: ({ l, n }) => { const r = rng(31), f = x => Math.max(x - l / 2, 0), H = SAEZ.map(v => v.map(x => x + n * gauss(r))), Z = H.map(v => v.map(f)); return { act: mean(Z.map(v => v.filter(x => x > 0).length)), err: mean(SAEZ.map((v, i) => mean(v.map((x, j) => (x - Z[i][j]) ** 2)))), cur: linspace(0, 1, 26).map(q => [q, mean(H.slice(0, 80).map(v => v.filter(x => x > q / 2).length))]) }; },
  plots: v => [{ x: [0, 1], y: [0, 12], w: 480, h: 260, xl: 'λ', yl: 'active features', label: 'sparsity against penalty', marks: [{ tone: 1, data: v.cur }, { as: 'rule', dash: true, tone: 4, x: v.l }, { as: 'rule', dash: true, tone: 'muted', y: 1.2 }] }],
  stats: v => [`active features ${fmt(v.act, 2)}`, `true active 1.20`, `mean squared error ${fmt(v.err, 4)}`],
  takeaway: 'The right penalty recovers about as many features as are truly active.',
});
export const CiPatchDrill = drill({
  intro: 'A patching experiment reports how much of an effect one component carries.',
  tex: R`\text{restored}=\frac{y_{\text{patched}}-y_{\text{clean}}}{y_{\text{source}}-y_{\text{clean}}}`,
  gen: r => ({ c: int(r, 0, 3), s: int(r, 6, 10), p: int(r, 3, 9) }),
  q: v => `Output: clean $${v.c}$, source $${v.s}$, patched $${v.p}$. What share of the effect is restored?`,
  answer: v => (v.p - v.c) / (v.s - v.c), dp: 3,
  steps: v => [`Total effect: $${v.s - v.c}$`],
});
export const CiCheck = truths({
  intro: CHECK,
  items: [
    ['SAC uses a deterministic policy.', false, 'Its stochastic policy is essential to the entropy objective.'],
    ['Model-predictive control replans at each step.', true, 'This corrects model errors.'],
    ['NeRF renders using a fixed mesh.', false, 'It integrates a learned density and colour field along rays.'],
    ['A successful probe proves the model uses that information.', false, 'A causal test is needed.'],
    ['Superposition makes single neurons harder to interpret.', true, 'Many features share each neuron.'],
  ],
  takeaway: 'Each area asks a causal question: what does an action, a ray or a neuron change?',
});

/* ---------- details from the lecture slides ---------- */
const MINI = (seed, n) => { const r = rng(seed * 7 + n); return Array.from({ length: n }, () => Array.from({ length: n }, () => int(r, -9, 9))); };
export const LgMinimaxLab = lab({
  intro: 'A table of values, with rows as $x$ and columns as $y$. Compare the two orders of optimising.',
  tex: R`\max_y\min_xq\le\min_x\max_yq`,
  params: [{ k: 'n', label: 'table size', min: 2, max: 6, step: 1, init: 4 }, { k: 'seed', seed: 'New table', init: 1 }],
  derive: ({ n, seed }) => { const Q = MINI(seed, n), a = Math.max(...Q[0].map((_, j) => Math.min(...Q.map(r => r[j])))), b = Math.min(...Q.map(r => Math.max(...r))); return { Q, a, b }; },
  table: v => ({ head: ['', ...v.Q[0].map((_, j) => `y${j + 1}`)], rows: v.Q.map((r, i) => [`x${i + 1}`, ...r]) }),
  stats: v => [`max-min ${v.a}`, `min-max ${v.b}`, `gap ${v.b - v.a}`],
  takeaway: 'The gap is never negative, and that is exactly weak duality.',
});
export const LgNestedDrill = drill({
  intro: 'Every outer fold repeats the whole search, then fits once more.',
  tex: R`K(JG+1)`,
  gen: r => ({ k: pick(r, [3, 5, 10]), j: pick(r, [3, 5]), g: int(r, 5, 40) }),
  q: v => `$${v.k}$ outer folds, $${v.j}$ inner folds, $${v.g}$ settings. How many fits?`,
  answer: v => v.k * (v.j * v.g + 1), dp: 0,
  steps: v => [`One outer fold: $${v.j * v.g + 1}$`],
});
export const LgWcssDrill = drill({
  intro: 'For points on a line, compute the within-cluster sum from pairs.',
  tex: R`\frac1{2n}\sum_{a,b}(a-b)^2`,
  gen: r => Array.from({ length: int(r, 2, 4) }, () => int(r, -4, 6)),
  q: v => `Points $${v.join(', ')}$. Find the within-cluster sum of squares.`,
  answer: v => { const m = mean(v); return v.reduce((s, x) => s + (x - m) ** 2, 0); }, dp: 3,
  steps: v => [`Mean: $${fmt(mean(v), 3)}$`],
});
export const LgPolyDrill = drill({
  intro: 'A polynomial kernel only needs one dot product.',
  tex: R`k(x,z)=(x^\top z+c)^d`,
  gen: r => ({ x: [int(r, -2, 3), int(r, -2, 3)], z: [int(r, -2, 3), int(r, -2, 3)], c: int(r, 0, 2), d: pick(r, [2, 3]) }),
  q: v => `$x=(${v.x})$, $z=(${v.z})$, $c=${v.c}$, $d=${v.d}$. Find the kernel value.`,
  answer: v => (v.x[0] * v.z[0] + v.x[1] * v.z[1] + v.c) ** v.d, dp: 0,
  steps: v => [`Dot product: $${v.x[0] * v.z[0] + v.x[1] * v.z[1]}$`],
});
export const LgMiDrill = drill({
  intro: 'Mutual information in nats, from a joint table.',
  tex: R`I=\sum p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}`,
  gen: r => pick(r, [[0.4, 0.1, 0.1, 0.4], [0.3, 0.2, 0.2, 0.3], [0.25, 0.25, 0.25, 0.25], [0.45, 0.05, 0.05, 0.45]]),
  q: v => `A $2\\times2$ joint table holds $${v.join(', ')}$, row by row. Find the mutual information.`,
  answer: v => { const a = [v[0] + v[1], v[2] + v[3]], b = [v[0] + v[2], v[1] + v[3]]; return v.reduce((s, p, i) => s + p * Math.log(p / (a[i >> 1] * b[i & 1])), 0); }, dp: 4,
  steps: () => ['Every marginal here is 0.5.'],
});
export const LgKappaDrill = drill({
  intro: 'Subtract the agreement expected by chance.',
  tex: R`\kappa=\frac{p_o-p_e}{1-p_e}`,
  gen: r => ({ o: pick(r, [0.7, 0.8, 0.9]), e: pick(r, [0.3, 0.5, 0.6]) }),
  q: v => `$p_o=${v.o}$, $p_e=${v.e}$. Find kappa.`,
  answer: v => (v.o - v.e) / (1 - v.e), dp: 3,
  steps: v => [`Numerator: $${fmt(v.o - v.e, 2)}$`],
});
export const LgMaseDrill = drill({
  intro: 'Scale the error by a naive seasonal forecast.',
  tex: R`\text{MASE}=\frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}`,
  gen: r => ({ a: int(r, 2, 12), b: int(r, 4, 20) }),
  q: v => `The model's MAE is $${v.a}$, and the naive seasonal MAE is $${v.b}$. Find MASE.`,
  answer: v => v.a / v.b, dp: 3,
  steps: () => ['Divide.'],
});
const HW = (() => { const r = rng(12); return Array.from({ length: 96 }, (_, t) => r6(50 + 0.5 * t + 10 * Math.sin((2 * Math.PI * t) / 12) + gauss(r))); })();
export const LgHoltLab = lab({
  intro: 'Fit level, trend and season to a monthly series, then forecast the final two years.',
  tex: R`\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1})`,
  params: [{ k: 'a', label: 'level rate $\\alpha$', min: 0.01, max: 1, step: 0.01, init: 0.3 }, { k: 'b', label: 'trend rate $\\beta$', min: 0, max: 0.5, step: 0.01, init: 0.05 }, { k: 'g', label: 'season rate $\\gamma$', min: 0, max: 1, step: 0.01, init: 0.2 }],
  derive: ({ a, b, g }) => { const m = 12, y = HW.slice(0, 72); let l = mean(y.slice(0, m)), tr = (mean(y.slice(m, 2 * m)) - l) / m; const s = y.slice(0, m).map(v => v - l); for (let i = m; i < 72; i++) { const old = l; l = a * (y[i] - s[i - m]) + (1 - a) * (l + tr); tr = b * (l - old) + (1 - b) * tr; s.push(g * (y[i] - l) + (1 - g) * s[i - m]); } const P = Array.from({ length: 24 }, (_, k) => l + (k + 1) * tr + s[72 - m + (k % m)]); return { P, mae: mean(P.map((p, i) => Math.abs(p - HW[72 + i]))) }; },
  plots: v => [{ x: [0, 96], y: [30, 120], w: 500, h: 280, xl: 'month', label: 'series and forecast', marks: [{ tone: 'muted', data: HW.map((y, t) => [t, y]) }, { tone: 1, data: v.P.map((p, i) => [72 + i, clip(p, 30, 120)]) }, { as: 'rule', dash: true, tone: 4, x: 72 }] }],
  stats: v => [`forecast MAE ${fmt(v.mae, 2)}`],
  takeaway: 'Three small update rules give a strong baseline for seasonal series.',
});
export const LgConvDrill = drill({
  intro: 'One formula covers kernel, padding, stride and dilation.',
  tex: R`\Big\lfloor\frac{n+2p-d(f-1)-1}s\Big\rfloor+1`,
  gen: r => ({ n: pick(r, [32, 64, 128]), f: pick(r, [3, 5]), p: int(r, 0, 2), s: pick(r, [1, 2]), d: pick(r, [1, 2, 3]) }),
  q: v => `Input $${v.n}$, kernel $${v.f}$, padding $${v.p}$, stride $${v.s}$, dilation $${v.d}$. Find the output size.`,
  answer: v => Math.floor((v.n + 2 * v.p - v.d * (v.f - 1) - 1) / v.s) + 1, dp: 0,
  steps: v => [`Numerator: $${v.n + 2 * v.p - v.d * (v.f - 1) - 1}$`],
});
export const LgDilatedLab = lab({
  intro: 'Stack $3\\times3$ layers. With doubling dilation, the view grows exponentially, and the weight count does not.',
  tex: R`r=1+\sum_\ell2d_\ell`,
  params: [{ k: 'n', label: 'layers', min: 1, max: 10, step: 1, init: 4 }],
  plots: v => [{ x: [1, 10], y: [0, 2100], w: 480, h: 260, xl: 'layers', yl: 'receptive field', label: 'receptive field against layers', marks: [{ tone: 'muted', data: curve(k => 1 + 2 * k, 1, 10) }, { tone: 1, data: curve(k => 1 + 2 * (2 ** k - 1), 1, 10) }, { as: 'rule', dash: true, tone: 4, x: v.n }] }],
  stats: v => [`field with doubling dilation ${1 + 2 * (2 ** v.n - 1)}`, `plain ${1 + 2 * v.n}`],
  takeaway: 'Dilation is the cheap way to see far while keeping full resolution.',
});
export const LgAdagradLab = lab({
  intro: 'A constant gradient is fed to Adagrad and to RMSProp. Compare their effective step sizes.',
  tex: R`\eta_t=\frac\eta{\sqrt{\sum_{\tau\le t}g_\tau^2}}`,
  params: [{ k: 'g', label: 'gradient size', min: 0.1, max: 4, step: 0.1, init: 1 }, { k: 'r', label: 'RMSProp decay', min: 0.5, max: 0.999, step: 0.001, init: 0.9 }],
  derive: ({ g, r }) => { let s = 0, m = 0; const A = [], B = []; for (let t = 1; t <= 300; t++) { s += g * g; m = r * m + (1 - r) * g * g; A.push([t, 1 / Math.sqrt(s)]); B.push([t, 1 / Math.sqrt(m)]); } return { A, B }; },
  plots: v => [{ x: [1, 300], y: [0, 2], w: 480, h: 260, xl: 'step', yl: 'effective rate per unit', label: 'effective step size', marks: [{ tone: 1, data: v.A.map(([t, y]) => [t, Math.min(2, y)]) }, { tone: 2, data: v.B.map(([t, y]) => [t, Math.min(2, y)]) }] }],
  stats: v => [`Adagrad at the end ${fmt(v.A[299][1], 3)}`, `RMSProp at the end ${fmt(v.B[299][1], 3)}`],
  legend: [['--c1', 'Adagrad'], ['--c2', 'RMSProp']],
  takeaway: 'Adagrad eventually stops moving, which led to averaging in RMSProp and Adam.',
});
export const LgGridDrill = drill({
  intro: 'The target is the centre’s offset within its grid cell.',
  gen: r => ({ S: pick(r, [7, 13, 19]), c: int(r, 5, 95) / 100 }),
  q: v => `Grid size $${v.S}$, centre coordinate $${v.c}$. Find the offset within its cell.`,
  answer: v => v.c * v.S - Math.floor(v.c * v.S), dp: 3,
  steps: v => [`$cS=${fmt(v.c * v.S, 3)}$`],
});
export const LgCutmixDrill = drill({
  intro: 'Label weights follow the area left unchanged.',
  tex: R`\lambda=1-\frac{wh}{WH}`,
  gen: r => ({ W: pick(r, [32, 224]), a: pick(r, [0.25, 0.5, 0.75]) }),
  q: v => `An image of side $${v.W}$. A square with side $${v.a * v.W}$ is pasted in. Find the weight of the original label.`,
  answer: v => 1 - v.a * v.a, dp: 4,
  steps: v => [`Pasted share: $${v.a * v.a}$`],
});
export const LgForcingLab = lab({
  intro: 'A decoder makes an error with some probability, and a wrong input makes the next error likelier. Teacher forcing removes that chain during training.',
  params: [{ k: 'e', label: 'error rate from a correct input', min: 0.01, max: 0.4, step: 0.01, init: 0.08 }, { k: 'p', label: 'error rate after a wrong input', min: 0.1, max: 0.95, step: 0.01, init: 0.6 }],
  derive: ({ e, p }) => { const F = [], T = []; let w = 0; for (let t = 1; t <= 30; t++) { w = w * p + (1 - w) * e; F.push([t, w]); T.push([t, e]); } return { F, T, w }; },
  plots: v => [{ x: [1, 30], y: [0, 1], w: 480, h: 260, xl: 'position', yl: 'chance of error', label: 'error against position', marks: [{ tone: 2, data: v.T }, { tone: 1, data: v.F }] }],
  stats: v => [`free-running error at the end ${pct(v.w, 1)}`, `with forcing ${pct(v.e, 1)}`],
  legend: [['--c1', 'own outputs'], ['--c2', 'teacher forcing']],
  takeaway: 'Forcing stabilises training, and hides how errors compound at test time.',
});
export const LgTdLab = lab({
  intro: 'A random walk with known values, estimated with eligibility traces.',
  tex: R`e(s)\leftarrow\gamma\lambda e(s)+\mathbb 1[s=s_t]`,
  params: [{ k: 'l', label: '$\\lambda$', min: 0, max: 1, step: 0.05, init: 0.6 }, { k: 'a', label: 'learning rate', min: 0.01, max: 0.5, step: 0.01, init: 0.1 }, { k: 'seed', seed: 'New runs', init: 1 }],
  derive: ({ l, a, seed }) => { const truth = [1, 2, 3, 4, 5].map(x => x / 6), R0 = Array(60).fill(0); for (let run = 0; run < 30; run++) { const r = rng(seed * 100 + run), V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]; for (let ep = 0; ep < 60; ep++) { const E = Array(7).fill(0); let s = 3; while (s > 0 && s < 6) { const n = s + (r() < 0.5 ? -1 : 1), d = (n === 6 ? 1 : 0) + V[n] - V[s]; E[s] += 1; for (let i = 1; i < 6; i++) { V[i] = r6(V[i] + a * d * E[i]); E[i] *= l; } s = n; } R0[ep] += Math.sqrt(mean(truth.map((t, i) => (V[i + 1] - t) ** 2))) / 30; } } return { H: R0.map((y, i) => [i + 1, y]) }; },
  plots: v => [{ x: [1, 60], y: [0, 0.4], w: 480, h: 260, xl: 'episode', yl: 'error', label: 'value error by episode', marks: [{ tone: 1, data: v.H.map(([x, y]) => [x, Math.min(0.4, y)]) }] }],
  stats: v => [`final error ${fmt(v.H[59][1], 3)}`],
  takeaway: 'An intermediate $\\lambda$ usually learns fastest, between one-step TD and full returns.',
});
export const LgVqDrill = drill({
  intro: 'A code vector snaps to its nearest codebook entry.',
  gen: r => ({ z: int(r, -9, 9) / 2, b: [-3, -1, 1, 3] }),
  q: v => `A value $${v.z}$ with codebook $${v.b.join(', ')}$. Which value is chosen? On a tie, take the smaller.`,
  answer: v => [...v.b].sort((a, c) => Math.abs(a - v.z) - Math.abs(c - v.z) || a - c)[0], dp: 0,
  steps: v => [`Distances: $${v.b.map(c => fmt(Math.abs(c - v.z), 1)).join(', ')}$`],
});
