'use client';
// Explorers added by the final coverage audit: the maths every advanced topic rests on, then the remaining named methods.
import { R, curve, drill, heatBox, heatPlot, histDensity, int, lab, pick, ridge } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { fmt, linspace, memo, normalPdf, pct } from '../lib/models.mjs';

const r6 = x => +x.toFixed(6);
const clip = (x, a, b) => Math.max(a, Math.min(b, x));
const lgam = z => { const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -0.000005395239384953]; let y = z, t = z + 5.5, s = 1.000000000190015; t -= (z + 0.5) * Math.log(t); g.forEach(c => { y += 1; s += c / y; }); return -t + Math.log((2.5066282746310005 * s) / z); };
const erf = x => { const t = 1 / (1 + 0.5 * Math.abs(x)), r = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); return x >= 0 ? r : -r; };
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
const PhiInv = p => { let a = -8, b = 8; for (let i = 0; i < 70; i++) { const m = (a + b) / 2; if (Phi(m) < p) a = m; else b = m; } return (a + b) / 2; };
const H2 = ps => -ps.reduce((s, p) => s + (p > 1e-12 ? p * Math.log2(p) : 0), 0);

/* ---------- linear algebra ---------- */
export const CvDeterminant = lab({
  intro: 'The columns of a matrix say where the two unit arrows land. The determinant is the signed area of the parallelogram they make. Drag the arrow tips.',
  tex: R`\det\begin{bmatrix}a&b\\c&d\end{bmatrix}=ad-bc`,
  params: [{ k: 'a', label: '$a$', min: -3, max: 3, step: 0.1, init: 2 }, { k: 'c', label: '$c$', min: -3, max: 3, step: 0.1, init: 0.5 }, { k: 'b', label: '$b$', min: -3, max: 3, step: 0.1, init: 0.5 }, { k: 'd', label: '$d$', min: -3, max: 3, step: 0.1, init: 1.5 }],
  derive: ({ a, b, c, d }) => ({ det: a * d - b * c, tr: a + d }),
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 480, h: 440, xl: 'x', yl: 'y', label: 'unit square under a matrix', marks: [{ tone: 'muted', dash: true, data: [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]] }, { tone: v.det < 0 ? 4 : 1, data: [[0, 0], [v.a, v.c], [v.a + v.b, v.c + v.d], [v.b, v.d], [0, 0]] }, { as: 'arrow', tone: 2, data: [[0, 0, v.a, v.c]] }, { as: 'arrow', tone: 3, data: [[0, 0, v.b, v.d]] }, { drag: { x: 'a', y: 'c', tone: 2 } }, { drag: { x: 'b', y: 'd', tone: 3 } }] }],
  stats: v => [`determinant ${fmt(v.det, 2)}`, `area scale ${fmt(Math.abs(v.det), 2)}`],
  status: v => (Math.abs(v.det) < 0.02 ? ['alert', 'The square is flattened, so this matrix has no inverse.'] : v.det < 0 ? ['warn', 'A negative determinant means the orientation flipped.'] : undefined),
  legend: [['--c2', 'first column'], ['--c3', 'second column']],
  takeaway: 'A zero determinant means a direction was lost, and nothing can bring it back.',
});
export const CvPowerIteration = lab({
  intro: 'Multiply by a symmetric matrix and renormalise. The direction swings towards the top eigenvector, faster when the two eigenvalues are far apart.',
  tex: R`x_{t+1}=\frac{Ax_t}{\lVert Ax_t\rVert}`,
  params: [{ k: 'a', label: '$A_{11}$', min: 0.5, max: 4, step: 0.1, init: 3 }, { k: 'b', label: '$A_{12}$', min: -2, max: 2, step: 0.1, init: 1 }, { k: 'd', label: '$A_{22}$', min: 0.5, max: 4, step: 0.1, init: 2 }, { k: 't', label: 'steps', min: 0, max: 12, step: 1, init: 3, play: true }],
  derive: ({ a, b, d, t }) => { const s = Math.sqrt(((a - d) / 2) ** 2 + b * b), l1 = (a + d) / 2 + s, l2 = (a + d) / 2 - s; let v1 = [b || 1e-9, l1 - a]; if (Math.abs(b) < 1e-9) v1 = a >= d ? [1, 0] : [0, 1]; const n = Math.hypot(...v1); v1 = v1.map(x => x / n); let x = [0.2, -0.98]; const P = [x]; for (let i = 0; i < t; i++) { const y = [a * x[0] + b * x[1], b * x[0] + d * x[1]], m = Math.hypot(...y); x = y.map(q => r6(q / m)); P.push(x); } return { l1, l2, v1, P, x, angle: (Math.acos(clip(Math.abs(x[0] * v1[0] + x[1] * v1[1]), 0, 1)) * 180) / Math.PI }; },
  plots: v => [{ x: [-1.4, 1.4], y: [-1.4, 1.4], w: 460, h: 440, label: 'power iteration directions', marks: [{ tone: 'muted', data: curve(t => Math.sqrt(Math.max(0, 1 - t * t)), -1, 1) }, { tone: 'muted', data: curve(t => -Math.sqrt(Math.max(0, 1 - t * t)), -1, 1) }, { tone: 3, dash: true, data: [[-v.v1[0] * 1.3, -v.v1[1] * 1.3], [v.v1[0] * 1.3, v.v1[1] * 1.3]] }, { as: 'points', tone: 2, r: 3, data: v.P }, { as: 'arrow', tone: 4, data: [[0, 0, v.x[0], v.x[1]]] }] }],
  stats: v => [`angle to the top eigenvector ${fmt(v.angle, 2)} degrees`, `eigenvalues ${fmt(v.l1, 3)} and ${fmt(v.l2, 3)}`, `shrink factor per step ${fmt(Math.abs(v.l2 / v.l1), 3)}`],
  takeaway: 'The error falls by the eigenvalue ratio each step, so near-equal eigenvalues converge slowly.',
});
export const CvGramSchmidtDrill = drill({
  intro: 'Remove from the second vector the part along the first.',
  tex: R`u=v-\frac{q^\top v}{q^\top q}\,q`,
  gen: r => ({ a: int(r, 1, 4), b: int(r, 0, 3), c: int(r, 1, 5), d: int(r, 1, 5) }),
  q: v => `With $q=(${v.a},${v.b})$ and $v=(${v.c},${v.d})$, what is the first entry of $u$?`,
  answer: v => v.c - ((v.a * v.c + v.b * v.d) / (v.a * v.a + v.b * v.b)) * v.a, dp: 3,
  steps: v => [`$q^\\top v=${v.a * v.c + v.b * v.d}$`, `$q^\\top q=${v.a * v.a + v.b * v.b}$`, `Subtract that ratio times $${v.a}$ from $${v.c}$.`],
});

/* ---------- calculus ---------- */
export const CvProductRuleDrill = drill({
  intro: 'Differentiate each factor in turn, leaving the other alone.',
  tex: R`(fg)'=f'g+fg'`,
  gen: r => ({ n: int(r, 1, 4), b: pick(r, [1, 2, -1]), x: pick(r, [0.5, 1, 2]) }),
  q: v => `Find the derivative of $x^{${v.n}}e^{${v.b}x}$ at $x=${v.x}$.`,
  answer: v => (v.n * v.x ** (v.n - 1) + v.b * v.x ** v.n) * Math.exp(v.b * v.x), dp: 3,
  steps: v => [`$f'g=${v.n}x^{${v.n - 1}}e^{${v.b}x}$`, `$fg'=${v.b}x^{${v.n}}e^{${v.b}x}$`, `Add them, and substitute $x=${v.x}$.`],
});
const TF = { exp: ['$e^x$', x => Math.exp(x), n => a => Math.exp(a)], sin: ['$\\sin x$', x => Math.sin(x), n => a => Math.sin(a + (n * Math.PI) / 2)], log: ['$\\ln(1+x)$', x => (x > -1 ? Math.log(1 + x) : NaN), n => a => (n === 0 ? Math.log(1 + a) : (((-1) ** (n - 1) * [1, 1, 1, 2, 6, 24][n]) / (1 + a) ** n))] };
export const CvTaylor = lab({
  intro: 'A polynomial built only from derivatives at a point. More terms hold on further from it.',
  tex: R`f(x)\approx\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k`,
  params: [{ k: 'f', label: 'Function', init: 'sin', options: Object.entries(TF).map(([k, o]) => [k, o[0].replace(/\$/g, '').replace('\\', '')]) }, { k: 'a', label: 'centre $a$', min: 0, max: 2, step: 0.1, init: 0.5 }, { k: 'n', label: 'order $n$', min: 0, max: 5, step: 1, init: 2 }],
  derive: ({ f, a, n }) => { const c = Array.from({ length: n + 1 }, (_, k) => TF[f][2](k)(a) / [1, 1, 2, 6, 24, 120][k]); return { c, T: x => c.reduce((s, q, k) => s + q * (x - a) ** k, 0) }; },
  plots: v => [{ x: [-2, 4], y: [-3, 5], w: 560, h: 340, xl: 'x', label: 'function and Taylor polynomial', marks: [{ tone: 'muted', data: curve(TF[v.f][1], v.f === 'log' ? -0.95 : -2, 4).filter(p => p[1] < 5 && p[1] > -3) }, { tone: 1, data: curve(v.T, -2, 4).filter(p => p[1] < 5 && p[1] > -3) }, { as: 'points', tone: 4, r: 6, data: [[v.a, TF[v.f][1](v.a)]] }] }],
  stats: v => [`error one unit away ${fmt(Math.abs(v.T(v.a + 1) - TF[v.f][1](v.a + 1)), 4)}`],
  legend: [['--muted', 'function'], ['--c1', 'polynomial']],
  takeaway: 'Newton’s method and XGBoost both use the second-order version of this polynomial.',
});
export const CvIntegral = lab({
  intro: 'An integral is the limit of thin slices. The trapezoid rule converges much faster than rectangles.',
  tex: R`\int_a^b f\,dx\approx\sum_i f(x_i)\,\Delta x`,
  params: [{ k: 'm', label: 'Rule', init: 'rect', options: [['rect', 'Rectangles'], ['trap', 'Trapezoids']] }, { k: 'n', label: 'slices', min: 1, max: 60, step: 1, init: 6 }],
  derive: ({ m, n }) => { const f = x => Math.exp(-x) * (1 + Math.sin(3 * x)), h = 3 / n, xs = Array.from({ length: n }, (_, i) => i * h), area = xs.reduce((s, x) => s + (m === 'rect' ? f(x) : (f(x) + f(x + h)) / 2) * h, 0), exact = 1 - Math.exp(-3) + (3 / 10) * (1 - Math.exp(-3) * (Math.cos(9) + Math.sin(9) / 3)); return { f, h, xs, area, exact }; },
  plots: v => [{ x: [0, 3], y: [0, 2], w: 560, h: 320, xl: 'x', label: 'area under a curve', marks: [{ as: 'rects', tone: 1, soft: true, data: v.m === 'rect' ? v.xs.map(x => [x, 0, x + v.h, v.f(x)]) : [] }, ...(v.m === 'trap' ? v.xs.map(x => ({ as: 'area', tone: 1, data: [[x, 0, v.f(x)], [x + v.h, 0, v.f(x + v.h)]] })) : []), { tone: 4, data: curve(v.f, 0, 3) }] }],
  stats: v => [`estimate ${fmt(v.area, 5)}`, `exact ${fmt(v.exact, 5)}`, `error ${fmt(Math.abs(v.area - v.exact), 5)}`],
  takeaway: 'Doubling the slices halves the rectangle error and quarters the trapezoid error.',
});
export const CvChangeVariables = lab({
  intro: 'Push a normal through $y=e^{sx}$. Without the Jacobian factor the curve no longer integrates to one.',
  tex: R`p_Y(y)=p_X(g^{-1}(y))\,\Big|\frac{d g^{-1}}{dy}\Big|`,
  params: [{ k: 's', label: 'scale $s$', min: 0.2, max: 1.5, step: 0.05, init: 0.6 }, { k: 'j', label: 'Jacobian', init: 'yes', options: [['yes', 'Included'], ['no', 'Left out']] }],
  derive: ({ s, j }) => { const ys = linspace(0.01, 6, 600), f = y => normalPdf(Math.log(y) / s, 0, 1) * (j === 'yes' ? 1 / (s * y) : 1), area = ys.reduce((t, y) => t + f(y), 0) * (ys[1] - ys[0]); return { f, area }; },
  plots: v => [{ x: [0, 6], y: [0, 1.4], w: 560, h: 320, xl: 'y', label: 'density after a transformation', marks: [{ tone: v.j === 'yes' ? 1 : 4, data: curve(y => Math.min(1.4, v.f(y)), 0.01, 6, 301) }] }],
  stats: v => [`area over the plot ${fmt(v.area, 3)}`],
  status: v => (v.j === 'no' ? ['alert', 'Without the Jacobian this is not a density.'] : ['ok', 'The density still integrates to about one.']),
  takeaway: 'Normalising flows are this formula applied many times.',
});

/* ---------- probability ---------- */
const DIST = {
  exp: { n: 'Exponential', p: [['rate $\\lambda$', 0.2, 3, 1], null], lim: [0, 6], f: (x, a) => (x < 0 ? 0 : a * Math.exp(-a * x)), m: a => 1 / a, v: a => 1 / a ** 2 },
  gamma: { n: 'Gamma', p: [['shape $\\alpha$', 0.5, 8, 2], ['rate $\\beta$', 0.3, 4, 1]], lim: [0, 10], f: (x, a, b) => (x <= 0 ? 0 : Math.exp(a * Math.log(b) - lgam(a) + (a - 1) * Math.log(x) - b * x)), m: (a, b) => a / b, v: (a, b) => a / b ** 2 },
  beta: { n: 'Beta', p: [['$a$', 0.5, 10, 2], ['$b$', 0.5, 10, 5]], lim: [0, 1], f: (x, a, b) => (x <= 0 || x >= 1 ? 0 : Math.exp(lgam(a + b) - lgam(a) - lgam(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))), m: (a, b) => a / (a + b), v: (a, b) => (a * b) / ((a + b) ** 2 * (a + b + 1)) },
  laplace: { n: 'Laplace', p: [['scale $b$', 0.2, 3, 1], null], lim: [-6, 6], f: (x, a) => Math.exp(-Math.abs(x) / a) / (2 * a), m: () => 0, v: a => 2 * a * a },
  lognormal: { n: 'Log-normal', p: [['$\\sigma$', 0.2, 1.5, 0.6], null], lim: [0, 6], f: (x, a) => (x <= 0 ? 0 : normalPdf(Math.log(x), 0, a) / x), m: a => Math.exp(a * a / 2), v: a => (Math.exp(a * a) - 1) * Math.exp(a * a) },
  t: { n: 'Student t', p: [['degrees $\\nu$', 1, 30, 3], null], lim: [-6, 6], f: (x, a) => Math.exp(lgam((a + 1) / 2) - lgam(a / 2) - 0.5 * Math.log(a * Math.PI) - ((a + 1) / 2) * Math.log(1 + (x * x) / a)), m: a => (a > 1 ? 0 : NaN), v: a => (a > 2 ? a / (a - 2) : NaN) },
  chi: { n: 'Chi-squared', p: [['degrees $k$', 1, 12, 3], null], lim: [0, 16], f: (x, a) => (x <= 0 ? 0 : Math.exp((a / 2 - 1) * Math.log(x) - x / 2 - (a / 2) * Math.LN2 - lgam(a / 2))), m: a => a, v: a => 2 * a },
};
export const CvDistributions = lab({
  intro: 'Choose a family and move its parameters. The upper chart is the density, and the lower one is the cumulative probability.',
  tex: R`F(x)=\int_{-\infty}^x p(t)\,dt`,
  params: [{ k: 'd', label: 'Family', init: 'gamma', options: Object.entries(DIST).map(([k, o]) => [k, o.n]) }, { k: 'a', label: 'first parameter (mapped to its range)', min: 0, max: 1, step: 0.01, init: 0.3 }, { k: 'b', label: 'second parameter, when there is one', min: 0, max: 1, step: 0.01, init: 0.3 }, { k: 'x', label: 'query point (mapped)', min: 0, max: 1, step: 0.01, init: 0.4 }],
  derive: ({ d, a, b, x }) => { const D = DIST[d], A = D.p[0][1] + a * (D.p[0][2] - D.p[0][1]), B = D.p[1] ? D.p[1][1] + b * (D.p[1][2] - D.p[1][1]) : 0, [lo, hi] = D.lim, X = lo + x * (hi - lo), xs = linspace(lo, hi, 800), h = xs[1] - xs[0], ps = xs.map(q => D.f(q, A, B)); let c = 0; const F = ps.map(p => (c += Math.min(p, 1e4) * h)); const i = Math.round(x * 799); return { D, A, B, X, xs, ps, F, cdf: F[i], top: Math.min(4, Math.max(...ps.filter(Number.isFinite)) * 1.1) }; },
  plots: v => { const every = v.xs.filter((_, i) => i % 3 === 0).map((x, i) => [x, Math.min(v.top, v.ps[i * 3])]); return [{ caption: 'Density', x: v.D.lim, y: [0, v.top], w: 520, h: 280, xl: 'x', label: 'probability density', marks: [{ tone: 1, data: every }, { as: 'rule', dash: true, tone: 4, x: v.X }] }, { caption: 'Cumulative probability', x: v.D.lim, y: [0, 1.05], w: 520, h: 240, xl: 'x', label: 'cumulative distribution', marks: [{ tone: 2, data: v.xs.filter((_, i) => i % 3 === 0).map((x, i) => [x, Math.min(1.05, v.F[i * 3])]) }, { as: 'points', tone: 4, r: 5, data: [[v.X, Math.min(1.05, v.cdf)]] }] }]; },
  stats: v => [`${v.D.n}, ${v.D.p[0][0].replace(/\$/g, '').replace(/\\/g, '')} = ${fmt(v.A, 2)}${v.D.p[1] ? `, ${v.D.p[1][0].replace(/\$/g, '').replace(/\\/g, '')} = ${fmt(v.B, 2)}` : ''}`, `mean ${Number.isFinite(v.D.m(v.A, v.B)) ? fmt(v.D.m(v.A, v.B), 3) : 'not defined'}, variance ${Number.isFinite(v.D.v(v.A, v.B)) ? fmt(v.D.v(v.A, v.B), 3) : 'not finite'}`, `P(X ≤ ${fmt(v.X, 2)}) is about ${fmt(Math.min(1, v.cdf), 3)}`],
  takeaway: 'A density can be above one. Only the area under it has to be one.',
});
export const CvTailBounds = lab({
  intro: 'Three answers for the chance of a large value of an exponential variable: the truth, a bound using the mean, and a bound using the variance.',
  tex: R`P(X\ge a)\le\frac{\mathbb E[X]}{a},\qquad P(|X-\mu|\ge k\sigma)\le\frac1{k^2}`,
  params: [{ k: 'a', label: 'threshold $a$', min: 2.1, max: 10, step: 0.1, init: 4 }],
  plots: v => [{ x: [2, 10], y: [-5, 0], w: 560, h: 320, xl: 'a', yl: 'log₁₀ probability', label: 'tail probability and bounds', marks: [{ tone: 3, data: curve(a => Math.log10(Math.exp(-a)), 2, 10) }, { tone: 1, data: curve(a => Math.log10(1 / a), 2, 10) }, { tone: 2, data: curve(a => Math.log10(1 / (a - 1) ** 2), 2.05, 10) }, { as: 'rule', dash: true, tone: 'muted', x: v.a }] }],
  stats: v => [`true tail ${fmt(Math.exp(-v.a), 5)}`, `Markov ${fmt(1 / v.a, 4)}`, `Chebyshev ${fmt(1 / (v.a - 1) ** 2, 4)}`],
  legend: [['--c3', 'true value'], ['--c1', 'Markov'], ['--c2', 'Chebyshev']],
  takeaway: 'More knowledge about the variable gives a tighter bound.',
});
export const CvTotalProbabilityDrill = drill({
  intro: 'Split on the cause, and add up the routes.',
  tex: R`P(A)=\sum_iP(A\mid B_i)P(B_i)`,
  gen: r => ({ p: pick(r, [0.1, 0.2, 0.3, 0.4]), a: pick(r, [0.9, 0.8, 0.7]), b: pick(r, [0.05, 0.1, 0.2]) }),
  q: v => `$P(B)=${v.p}$, $P(A\\mid B)=${v.a}$, and $P(A\\mid\\neg B)=${v.b}$. Find $P(A)$.`,
  answer: v => v.a * v.p + v.b * (1 - v.p), dp: 3,
  steps: v => [`Route through $B$: $${(v.a * v.p).toFixed(3)}$`, `Route through not $B$: $${(v.b * (1 - v.p)).toFixed(3)}$`],
});
export const CvLinearityDrill = drill({
  intro: 'Expectation passes through sums, even when the variables are dependent.',
  tex: R`\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y]`,
  gen: r => ({ a: int(r, 1, 4), b: int(r, 1, 4), n: int(r, 2, 6) }),
  q: v => `You roll $${v.n}$ fair dice. What is the expected value of ${v.a} times the total, plus ${v.b}?`,
  answer: v => v.a * 3.5 * v.n + v.b, dp: 2,
  steps: v => [`Each die has mean $3.5$, so the total has mean $${3.5 * v.n}$.`, `Scale by $${v.a}$ and add $${v.b}$.`],
});
export const CvBetaUpdate = lab({
  intro: 'A conjugate prior updates by counting. Each observation adds to one of the two parameters.',
  tex: R`\text{Beta}(a,b)\to\text{Beta}(a+s,\,b+f)`,
  params: [{ k: 'a', label: 'prior $a$', min: 0.5, max: 10, step: 0.5, init: 2 }, { k: 'b', label: 'prior $b$', min: 0.5, max: 10, step: 0.5, init: 2 }, { k: 's', label: 'heads $s$', min: 0, max: 50, step: 1, init: 7 }, { k: 'f', label: 'tails $f$', min: 0, max: 50, step: 1, init: 3 }],
  plots: v => { const d = (a, b) => x => Math.exp(lgam(a + b) - lgam(a) - lgam(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)), p = d(v.a, v.b), q = d(v.a + v.s, v.b + v.f), top = Math.max(...linspace(0.01, 0.99, 99).map(x => Math.max(p(x), q(x)))) * 1.1; return [{ x: [0, 1], y: [0, top], w: 560, h: 320, xl: 'θ', label: 'prior and posterior', marks: [{ tone: 'muted', data: curve(p, 0.005, 0.995) }, { tone: 1, data: curve(q, 0.005, 0.995) }, { as: 'rule', dash: true, tone: 4, x: (v.s + 1e-9) / (v.s + v.f + 2e-9) }] }]; },
  stats: v => [`posterior mean ${fmt((v.a + v.s) / (v.a + v.b + v.s + v.f), 3)}`, `maximum likelihood ${v.s + v.f ? fmt(v.s / (v.s + v.f), 3) : 'not defined'}`],
  legend: [['--muted', 'prior'], ['--c1', 'posterior']],
  takeaway: 'With little data the prior matters, and with plenty of data the likelihood takes over.',
});
export const CvEntropies = lab({
  intro: 'Set a joint distribution over two binary variables. Each entropy and the mutual information follow from those four cells.',
  tex: R`I(X;Y)=H(X)+H(Y)-H(X,Y)`,
  params: [{ k: 'a', label: '$p(0,0)$ share', min: 0.01, max: 1, step: 0.01, init: 0.6 }, { k: 'b', label: '$p(0,1)$ share', min: 0.01, max: 1, step: 0.01, init: 0.1 }, { k: 'c', label: '$p(1,0)$ share', min: 0.01, max: 1, step: 0.01, init: 0.1 }, { k: 'd', label: '$p(1,1)$ share', min: 0.01, max: 1, step: 0.01, init: 0.4 }],
  derive: ({ a, b, c, d }) => { const s = a + b + c + d, P = [[a / s, b / s], [c / s, d / s]], hxy = H2(P.flat()), hx = H2([P[0][0] + P[0][1], P[1][0] + P[1][1]]), hy = H2([P[0][0] + P[1][0], P[0][1] + P[1][1]]); return { P, hxy, hx, hy, mi: Math.max(0, hx + hy - hxy) }; },
  plots: v => [heatPlot(v.P, { w: 420, h: 320, label: 'joint distribution', heat: { max: 1, rows: ['X=0', 'X=1'], cols: ['Y=0', 'Y=1'], show: x => x.toFixed(3) } })],
  table: v => ({ head: ['quantity', 'bits'], rows: [['H(X)', fmt(v.hx, 3)], ['H(Y)', fmt(v.hy, 3)], ['H(X,Y)', fmt(v.hxy, 3)], ['H(Y|X)', fmt(v.hxy - v.hx, 3)], ['I(X;Y)', fmt(v.mi, 3)]] }),
  stats: v => [`mutual information ${fmt(v.mi, 3)} bits`],
  takeaway: 'Make the four cells a product of two marginals and the mutual information falls to zero.',
});

/* ---------- remaining machine learning ---------- */
const rosen = (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
export const CvRosenbrock = lab({
  intro: 'A long curved valley with its minimum at the point $(1,1)$. A learning rate large enough to make progress along the valley is too large across it.',
  tex: R`f(x,y)=(1-x)^2+100(y-x^2)^2`,
  params: [{ k: 'lr', label: 'learning rate $\\times 10^{-3}$', min: 0.2, max: 2.2, step: 0.1, init: 1.5 }, { k: 'n', label: 'steps', min: 10, max: 2000, step: 10, init: 500 }],
  derive: ({ lr, n }) => { let p = [-1.2, 1]; const P = [p]; for (let t = 0; t < n; t++) { const [x, y] = p; p = [clip(x + lr * 1e-3 * (2 * (1 - x) + 400 * x * (y - x * x)), -3, 3), clip(y - lr * 1e-3 * 200 * (y - x * x), -3, 5)].map(r6); if (t % Math.ceil(n / 300) === 0) P.push(p); } return { P, p, f: rosen(...p) }; },
  plots: v => [{ x: [-2, 2], y: [-1, 3], w: 560, h: 420, xl: 'x', yl: 'y', label: 'Rosenbrock function and path', marks: [{ as: 'field', bands: 8, tones: [1, 1], fn: (x, y) => 1 - Math.min(1, Math.log1p(rosen(x, y)) / 7.2) }, { tone: 2, data: v.P }, { as: 'rings', r: 8, data: [[1, 1]] }, { as: 'points', tone: 4, r: 5, data: [v.p] }] }],
  stats: v => [`loss ${fmt(v.f, 5)}`, `position (${fmt(v.p[0], 3)}, ${fmt(v.p[1], 3)})`],
  takeaway: 'The condition number at the minimum is about 2500, which is what makes this slow.',
});
export const CvMvnConditional = lab({
  intro: 'Fix $x_2$ and read the slice through a correlated normal. The slice is a normal again, with a shifted mean and a smaller variance.',
  tex: R`x_1\mid x_2\sim\mathcal N\big(\rho\,x_2,\ 1-\rho^2\big)`,
  params: [{ k: 'r', label: 'correlation $\\rho$', min: -0.95, max: 0.95, step: 0.05, init: 0.7 }, { k: 'x', label: 'observed $x_2$', min: -2.5, max: 2.5, step: 0.05, init: 1 }],
  plots: v => { const q = (a, b) => Math.exp(-(a * a - 2 * v.r * a * b + b * b) / (2 * (1 - v.r * v.r))); return [{ caption: 'Joint density', x: [-3, 3], y: [-3, 3], w: 460, h: 420, xl: 'x₁', yl: 'x₂', label: 'joint normal density', marks: [{ as: 'field', bands: 8, tones: [1, 1], fn: (a, b) => q(a, b) }, { as: 'rule', dash: true, tone: 4, y: v.x }, { drag: { x: 0, y: 'x', tone: 4 } }] }, { caption: 'The slice', x: [-3, 3], y: [0, 1.4], w: 460, h: 300, xl: 'x₁', label: 'conditional density', marks: [{ tone: 'muted', data: curve(a => normalPdf(a, 0, 1), -3, 3) }, { tone: 1, data: curve(a => normalPdf(a, v.r * v.x, Math.sqrt(1 - v.r * v.r)), -3, 3) }] }]; },
  stats: v => [`conditional mean ${fmt(v.r * v.x, 3)}`, `conditional variance ${fmt(1 - v.r * v.r, 3)}`],
  legend: [['--muted', 'marginal'], ['--c1', 'conditional']],
  takeaway: 'The variance does not depend on the value observed, only on the correlation.',
});
const boxMuller = memo(seed => { const r = rng(seed * 89), U = Array.from({ length: 1500 }, () => [r(), r()]); return { U, Z: U.map(([a, b]) => r6(Math.sqrt(-2 * Math.log(a + 1e-12)) * Math.cos(2 * Math.PI * b))) }; });
export const CvBoxMuller = lab({
  intro: 'Two uniform numbers go in, and a normal number comes out. The histogram matches the bell curve.',
  tex: R`Z=\sqrt{-2\ln U_1}\cos(2\pi U_2)`,
  params: [{ k: 'n', label: 'samples', min: 50, max: 1500, step: 50, init: 500 }, { k: 'seed', seed: 'New draw', init: 1 }],
  derive: ({ n, seed }) => { const B = boxMuller(seed), Z = B.Z.slice(0, n); return { Z, U: B.U.slice(0, Math.min(n, 400)), m: mean(Z), d: histDensity(Z, -4, 4, 40) }; },
  plots: v => [{ caption: 'Uniform inputs', x: [0, 1], y: [0, 1], w: 420, h: 400, xl: 'U₁', yl: 'U₂', label: 'uniform inputs', marks: [{ as: 'points', tone: 2, r: 2, data: v.U }] }, { caption: 'Normal outputs', x: [-4, 4], y: [0, 0.55], w: 420, h: 400, xl: 'z', label: 'normal outputs', marks: [{ as: 'area', tone: 1, data: linspace(-3.9, 3.9, 40).map((x, i) => [x, 0, v.d[i]]) }, { tone: 4, data: curve(x => normalPdf(x, 0, 1), -4, 4) }] }],
  stats: v => [`sample mean ${fmt(v.m, 3)}`],
  takeaway: 'Polar coordinates turn two uniform draws into two independent normal draws.',
});
const huffman = p => { let h = p.map((q, i) => [q, [i]]); const L = p.map(() => 0); while (h.length > 1) { h.sort((a, b) => a[0] - b[0]); const [a, b] = h, m = [a[0] + b[0], [...a[1], ...b[1]]]; m[1].forEach(i => L[i]++); h = [m, ...h.slice(2)]; } return L; };
export const CvHuffmanDrill = drill({
  intro: 'Merge the two least likely symbols until a single tree remains. Depth is code length.',
  tex: R`L=\sum_ip_i\ell_i`,
  gen: r => ({ p: pick(r, [[0.5, 0.25, 0.125, 0.125], [0.4, 0.3, 0.2, 0.1], [0.6, 0.2, 0.1, 0.1], [0.35, 0.3, 0.25, 0.1], [0.45, 0.25, 0.2, 0.1]]) }),
  q: v => `Build a Huffman code for probabilities $${v.p.join(', ')}$. What is the average code length in bits?`,
  answer: v => { const L = huffman(v.p); return v.p.reduce((s, q, i) => s + q * L[i], 0); }, dp: 3,
  steps: v => [`Code lengths: $${huffman(v.p).join(', ')}$`, 'Multiply each length by its probability, and add them.'],
});
const POOL = (() => { const r = rng(77); return Array.from({ length: 12 }, (_, i) => ({ t: r6(0.26 + 0.03 * gauss(r)), n: [8, 12, 15, 20, 30, 40, 60, 80, 120, 200, 300, 500][i], z: gauss(r) })); })();
export const CvPartialPooling = lab({
  intro: 'Each group has its own estimate. Shrinkage pulls every group towards the grand mean, and small groups move the most.',
  tex: R`\hat\theta_j=\frac{y_j/\sigma_j^2+\mu/\tau^2}{1/\sigma_j^2+1/\tau^2}`,
  params: [{ k: 'tau', label: 'between-group spread $\\tau$', min: 0.005, max: 0.2, step: 0.005, init: 0.03 }],
  derive: ({ tau }) => { const G = POOL.map(g => { const s2 = 0.19 / g.n, y = g.t + Math.sqrt(s2) * g.z; return { ...g, s2, y }; }), w = G.map(g => 1 / g.s2), mu = G.reduce((s, g, i) => s + g.y * w[i], 0) / w.reduce((a, b) => a + b), R2 = G.map(g => ({ ...g, post: (g.y / g.s2 + mu / tau ** 2) / (1 / g.s2 + 1 / tau ** 2) })); return { G: R2, mu, raw: mean(R2.map(g => (g.y - g.t) ** 2)), pooled: mean(R2.map(g => (g.post - g.t) ** 2)) }; },
  plots: v => [{ x: [-0.5, 11.5], y: [0, 0.6], w: 600, h: 340, xl: 'groups, smallest to largest', yl: 'estimate', label: 'raw and shrunk estimates', marks: [{ as: 'rule', dash: true, tone: 'muted', y: v.mu }, { as: 'stems', tone: 'muted', data: v.G.map((g, i) => [i, g.y, g.post]) }, { as: 'points', tone: 2, r: 5, data: v.G.map((g, i) => [i, g.y]) }, { as: 'points', tone: 1, r: 5, data: v.G.map((g, i) => [i, g.post]) }, { as: 'points', tone: 3, r: 3, data: v.G.map((g, i) => [i, g.t]) }] }],
  stats: v => [`squared error, raw ${fmt(v.raw * 1e4, 2)}`, `squared error, shrunk ${fmt(v.pooled * 1e4, 2)}`],
  legend: [['--c2', 'raw'], ['--c1', 'shrunk'], ['--c3', 'true value']],
  takeaway: 'Borrowing strength across groups reduces total error, even when each group really differs.',
});
export const CvMarkovChain = lab({
  intro: 'Rows of the matrix are current states. Repeated multiplication settles to a fixed row, unless the chain is periodic.',
  tex: R`\pi_{t+1}=\pi_tP`,
  params: [{ k: 'p', label: 'stay probability', min: 0, max: 0.9, step: 0.05, init: 0.3 }, { k: 'q', label: 'forward share', min: 0, max: 1, step: 0.05, init: 0.7 }, { k: 't', label: 'steps', min: 0, max: 40, step: 1, init: 3, play: true }],
  derive: ({ p, q, t }) => { const P = [0, 1, 2].map(i => [0, 1, 2].map(j => (i === j ? p : j === (i + 1) % 3 ? (1 - p) * q : (1 - p) * (1 - q)))); let M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; for (let k = 0; k < t; k++) M = M.map(r => [0, 1, 2].map(j => r6(r.reduce((s, x, c) => s + x * P[c][j], 0)))); return { P, M, spread: Math.max(...[0, 1, 2].map(j => Math.max(...M.map(r => r[j])) - Math.min(...M.map(r => r[j])))) }; },
  plots: v => [heatPlot(v.P, { w: 420, h: 300, caption: 'Transition matrix', label: 'transition matrix', heat: { max: 1, show: f3 } }), heatPlot(v.M, { w: 420, h: 300, caption: `Matrix to the power ${v.t}`, label: 'matrix power', heat: { max: 1, show: f3 } })],
  stats: v => [`largest disagreement between start states ${fmt(v.spread, 4)}`],
  status: v => (v.p === 0 && (v.q === 1 || v.q === 0) ? ['warn', 'This chain cycles with period three, so it never settles.'] : undefined),
  takeaway: 'Irreducible and aperiodic chains forget where they started.',
});
const f3 = x => x.toFixed(2);
export const CvAdjustedRDrill = drill({
  intro: 'Adding a feature can only raise $R^2$, so the adjusted form charges for it.',
  tex: R`\bar R^2=1-(1-R^2)\frac{n-1}{n-p-1}`,
  gen: r => ({ r2: pick(r, [0.6, 0.7, 0.8, 0.9]), n: pick(r, [21, 31, 51, 101]), p: pick(r, [2, 4, 5, 10]) }),
  q: v => `$R^2=${v.r2}$, $n=${v.n}$, and $p=${v.p}$. Find the adjusted $R^2$.`,
  answer: v => 1 - ((1 - v.r2) * (v.n - 1)) / (v.n - v.p - 1), dp: 3,
  steps: v => [`Ratio: $${v.n - 1}/${v.n - v.p - 1}$`, `$1-R^2=${(1 - v.r2).toFixed(2)}$`],
});

/* ---------- advanced maths ---------- */
export const CvFourier = lab({
  intro: 'The discrete Fourier transform says how much of each frequency a signal holds. Mix two waves, and two spikes appear.',
  tex: R`X_k=\sum_{t=0}^{n-1}x_te^{-2\pi ikt/n}`,
  params: [{ k: 'a', label: 'first frequency', min: 1, max: 15, step: 1, init: 3 }, { k: 'b', label: 'second frequency', min: 1, max: 15, step: 1, init: 9 }, { k: 'g', label: 'second amplitude', min: 0, max: 1, step: 0.05, init: 0.5 }, { k: 'n', label: 'noise', min: 0, max: 1, step: 0.05, init: 0.1 }],
  derive: ({ a, b, g, n }) => { const r = rng(9), N = 64, x = Array.from({ length: N }, (_, t) => r6(Math.sin((2 * Math.PI * a * t) / N) + g * Math.sin((2 * Math.PI * b * t) / N) + n * gauss(r))), X = Array.from({ length: 32 }, (_, k) => { let c = 0, s = 0; x.forEach((v, t) => { c += v * Math.cos((2 * Math.PI * k * t) / N); s -= v * Math.sin((2 * Math.PI * k * t) / N); }); return r6(Math.hypot(c, s) / (N / 2)); }); return { x, X, top: X.indexOf(Math.max(...X)) }; },
  plots: v => [{ caption: 'Signal', x: [0, 63], y: [-2.5, 2.5], w: 560, h: 240, xl: 'time', label: 'signal in time', marks: [{ tone: 1, data: v.x.map((y, t) => [t, y]) }] }, { caption: 'Frequencies', x: [-0.5, 31.5], y: [0, 1.3], w: 560, h: 240, xl: 'frequency', label: 'amplitude by frequency', marks: [{ as: 'stems', tone: 2, data: v.X.map((y, k) => [k, 0, y]) }, { as: 'points', tone: 2, r: 3, data: v.X.map((y, k) => [k, y]) }] }],
  stats: v => [`strongest frequency ${v.top}`, `its amplitude ${fmt(v.X[v.top], 3)}`],
  takeaway: 'Convolution in time is multiplication here, which is why filtering can be done in this domain.',
});
export const CvOdeSolvers = lab({
  intro: 'Solve $\\dot x=-kx$ with large steps. Euler overshoots and can blow up, and Runge-Kutta stays close for much longer.',
  tex: R`x_{t+h}=x_t+h\,f(x_t)`,
  params: [{ k: 'k', label: 'rate $k$', min: 0.5, max: 4, step: 0.1, init: 2 }, { k: 'h', label: 'step $h$', min: 0.05, max: 1.2, step: 0.05, init: 0.4 }],
  derive: ({ k, h }) => { const f = x => -k * x, n = Math.ceil(4 / h), E = [[0, 1]], Rk = [[0, 1]]; let e = 1, q = 1; for (let i = 1; i <= n; i++) { e = clip(e + h * f(e), -1e3, 1e3); const a = f(q), b = f(q + (h * a) / 2), c = f(q + (h * b) / 2), d = f(q + h * c); q = q + (h * (a + 2 * b + 2 * c + d)) / 6; E.push([r6(i * h), r6(e)]); Rk.push([r6(i * h), r6(q)]); } return { E, Rk, ee: Math.abs(E[1][1] - Math.exp(-k * h)), er: Math.abs(Rk[1][1] - Math.exp(-k * h)) }; },
  plots: v => [{ x: [0, 4], y: [-1.5, 1.5], w: 560, h: 340, xl: 't', yl: 'x', label: 'ODE solvers', marks: [{ tone: 'muted', data: curve(t => Math.exp(-v.k * t), 0, 4) }, { tone: 4, data: v.E.map(([t, x]) => [t, clip(x, -1.5, 1.5)]) }, { as: 'points', tone: 4, r: 3, data: v.E.map(([t, x]) => [t, clip(x, -1.5, 1.5)]) }, { tone: 1, data: v.Rk.map(([t, x]) => [t, clip(x, -1.5, 1.5)]) }] }],
  stats: v => [`$hk=${fmt(v.h * v.k, 2)}$`, `single-step error: Euler ${fmt(v.ee, 4)}, Runge-Kutta ${fmt(v.er, 5)}`],
  status: v => (v.h * v.k > 2 ? ['alert', 'Above $hk=2$ the Euler solution grows without limit.'] : undefined),
  legend: [['--muted', 'exact'], ['--c4', 'Euler'], ['--c1', 'Runge-Kutta']],
  takeaway: 'Fewer solver steps is the whole aim of fast samplers, and solver order is what buys them.',
});
const brownian = memo(seed => { const r = rng(seed * 113), n = 200, h = 1 / n, P = Array.from({ length: 400 }, () => { let w = 0; return Array.from({ length: n + 1 }, (_, i) => { if (i) w += Math.sqrt(h) * gauss(r); return r6(w); }); }); return { P, h, n }; });
export const CvBrownian = lab({
  intro: 'Random paths whose spread grows with the square root of time. The ridgelines are the distribution at each time.',
  tex: R`W_{t+s}-W_t\sim\mathcal N(0,s)`,
  params: [{ k: 't', label: 'time', min: 0.05, max: 1, step: 0.05, init: 0.8 }, { k: 'seed', seed: 'New paths', init: 1 }],
  derive: ({ t, seed }) => { const B = brownian(seed), i = Math.round(t * B.n), s = B.P.map(p => p[i]); return { B, i, sd: Math.sqrt(mean(s.map(x => x * x))), rows: [0.2, 0.4, 0.6, 0.8, 1].map(q => ({ label: `t = ${q}`, d: histDensity(B.P.map(p => p[Math.round(q * B.n)]), -3, 3, 40) })) }; },
  plots: v => [{ caption: 'Paths', x: [0, 1], y: [-3, 3], w: 500, h: 340, xl: 't', label: 'Brownian paths', marks: [...v.B.P.slice(0, 14).map((p, k) => ({ tone: (k % 4) + 1, data: p.slice(0, v.i + 1).map((y, i) => [i * v.B.h, y]) })), { tone: 'fg', dash: true, data: curve(x => 2 * Math.sqrt(x), 0, 1) }, { tone: 'fg', dash: true, data: curve(x => -2 * Math.sqrt(x), 0, 1) }] }, { caption: 'Distribution through time', x: [-3, 3], y: [0, 5.8], w: 500, h: 340, noY: true, xl: 'value', label: 'distributions through time', marks: ridge(v.rows, linspace(-2.925, 2.925, 40)) }],
  stats: v => [`measured spread ${fmt(v.sd, 3)}`, `theory $\\sqrt t=${fmt(Math.sqrt(v.t), 3)}$`],
  takeaway: 'Variance grows like time, so a step of size $dt$ moves by about $\\sqrt{dt}$.',
});
export const CvDistances = lab({
  intro: 'Two narrow bumps move apart. KL explodes, Jensen-Shannon saturates, and Wasserstein grows in proportion to the distance.',
  tex: R`W_1=\inf_\gamma\mathbb E_\gamma|x-y|`,
  params: [{ k: 'd', label: 'separation', min: 0, max: 6, step: 0.05, init: 1.5 }, { k: 's', label: 'width', min: 0.1, max: 1, step: 0.05, init: 0.2 }],
  derive: ({ d, s }) => { const xs = linspace(-4, 10, 1400), h = xs[1] - xs[0], J = D => xs.reduce((t, x) => { const p = normalPdf(x, 0, s), q = normalPdf(x, D, s), m = (p + q) / 2; return t + 0.5 * h * ((p > 1e-300 ? p * Math.log(p / m) : 0) + (q > 1e-300 ? q * Math.log(q / m) : 0)); }, 0), ds = linspace(0, 6, 49); return { js: J(d), kl: (d * d) / (2 * s * s), curve: ds.map(D => [D, J(D)]) }; },
  plots: v => [{ caption: 'The two distributions', x: [-2, 8], y: [0, 2.1], w: 560, h: 240, xl: 'x', label: 'two distributions', marks: [{ tone: 1, data: curve(x => Math.min(2.1, normalPdf(x, 0, v.s)), -2, 8, 401) }, { tone: 2, data: curve(x => Math.min(2.1, normalPdf(x, v.d, v.s)), -2, 8, 401) }] }, { caption: 'Distance against separation', x: [0, 6], y: [0, 6], w: 560, h: 280, xl: 'separation', label: 'distance measures', marks: [{ tone: 3, data: curve(D => D, 0, 6) }, { tone: 4, data: curve(D => Math.min(6, (D * D) / (2 * v.s * v.s)), 0, 6, 401) }, { tone: 1, data: v.curve }, { as: 'rule', dash: true, tone: 'muted', x: v.d }] }],
  stats: v => [`Wasserstein ${fmt(v.d, 3)}`, `KL ${fmt(v.kl, 3)}`, `Jensen-Shannon ${fmt(v.js, 4)}, limit ${fmt(Math.LN2, 4)}`],
  legend: [['--c3', 'Wasserstein'], ['--c4', 'KL'], ['--c1', 'Jensen-Shannon']],
  takeaway: 'Only the Wasserstein distance still tells a generator which way to move when the supports do not overlap.',
});
export const CvRates = lab({
  intro: 'Gradient descent on a quadratic, measured against theory. The condition number sets the speed, and momentum changes its dependence.',
  tex: R`\text{error}_t\approx\Big(\frac{\kappa-1}{\kappa+1}\Big)^t`,
  params: [{ k: 'k', label: 'condition number $\\kappa$', min: 2, max: 200, step: 2, init: 30 }, { k: 'n', label: 'steps', min: 20, max: 300, step: 10, init: 120 }],
  derive: ({ k, n }) => { const q = (k - 1) / (k + 1), m = (Math.sqrt(k) - 1) / (Math.sqrt(k) + 1); return { q, m, n }; },
  plots: v => [{ x: [0, v.n], y: [-12, 0], w: 560, h: 340, xl: 'steps', yl: 'log₁₀ error', label: 'convergence rates', marks: [{ tone: 2, data: curve(t => Math.max(-12, t * Math.log10(v.q)), 0, v.n) }, { tone: 1, data: curve(t => Math.max(-12, t * Math.log10(v.m)), 0, v.n) }] }],
  stats: v => [`rate per step: plain ${fmt(v.q, 4)}, momentum ${fmt(v.m, 4)}`, `steps for 1000 times less error: ${Math.ceil(3 / -Math.log10(v.q))} and ${Math.ceil(3 / -Math.log10(v.m))}`],
  legend: [['--c2', 'gradient descent'], ['--c1', 'heavy-ball momentum']],
  takeaway: 'Momentum replaces $\\kappa$ with $\\sqrt\\kappa$ on quadratics.',
});

/* ---------- remaining deep learning ---------- */
export const CvCausalReach = lab({
  intro: 'Rows are layers and columns are time. Shaded cells feed the output at the last step, and nothing to its right can.',
  tex: R`\text{field}=1+\sum_\ell(f-1)d_\ell`,
  params: [{ k: 'p', label: 'Dilations', init: 'double', options: [['double', 'Doubling'], ['one', 'All one']] }, { k: 'f', label: 'kernel $f$', min: 2, max: 4, step: 1, init: 2 }, { k: 'L', label: 'layers', min: 1, max: 6, step: 1, init: 5 }],
  derive: ({ p, f, L }) => { const T = 48, ds = Array.from({ length: L }, (_, i) => (p === 'double' ? 2 ** i : 1)); let on = new Set([T - 1]); const M = [Array.from({ length: T }, (_, t) => (on.has(t) ? 1 : 0))]; for (let i = L - 1; i >= 0; i--) { const nx = new Set(); on.forEach(t => { for (let k = 0; k < f; k++) if (t - k * ds[i] >= 0) nx.add(t - k * ds[i]); }); on = nx; M.push(Array.from({ length: T }, (_, t) => (on.has(t) ? 1 : 0))); } return { M, ds, field: 1 + ds.reduce((s, d) => s + (f - 1) * d, 0), n: L * f }; },
  plots: v => [heatPlot(v.M, { w: 620, h: 60 + 44 * v.M.length, label: 'inputs reaching the final output', heat: { dense: true, max: 1, rows: v.M.map((_, i) => (i ? `layer ${v.M.length - i}` : 'output')), name: (i, j, x) => `${i ? `input to layer ${v.M.length - i}` : 'output'}, time ${j}: ${x ? 'used' : 'not used'}` } })],
  stats: v => [`receptive field ${v.field}`, `only ${v.n} weights per channel path`],
  takeaway: 'Doubling dilations give an exponential reach for a linear number of layers.',
});
export const CvAttentionScoreDrill = drill({
  intro: 'Luong’s general score places a learned matrix between the two states.',
  tex: R`\text{score}=s^\top Wh`,
  gen: r => ({ s: [int(r, -2, 3), int(r, 1, 3)], h: [int(r, 1, 3), int(r, -2, 2)], W: [int(r, 1, 3), int(r, -1, 2), int(r, 0, 2), int(r, 1, 2)] }),
  q: v => `$s=(${v.s})$, $h=(${v.h})$, and $W=\\begin{bmatrix}${v.W[0]}&${v.W[1]}\\\\${v.W[2]}&${v.W[3]}\\end{bmatrix}$. Find the score.`,
  answer: v => v.s[0] * (v.W[0] * v.h[0] + v.W[1] * v.h[1]) + v.s[1] * (v.W[2] * v.h[0] + v.W[3] * v.h[1]), dp: 0,
  steps: v => [`$Wh=(${v.W[0] * v.h[0] + v.W[1] * v.h[1]},\\,${v.W[2] * v.h[0] + v.W[3] * v.h[1]})$`, 'Take the dot product with $s$.'],
});
export const CvRope = lab({
  intro: 'Rotary embeddings rotate the query by its position and the key by its own. Move both together, and the score does not change.',
  tex: R`\langle R(m\theta)q,\,R(n\theta)k\rangle=\langle R((m-n)\theta)q,\,k\rangle`,
  params: [{ k: 'm', label: 'query position $m$', min: 0, max: 40, step: 1, init: 10 }, { k: 'n', label: 'key position $n$', min: 0, max: 40, step: 1, init: 4 }, { k: 'th', label: 'angle per step $\\theta$', min: 0.02, max: 0.6, step: 0.01, init: 0.2 }],
  derive: ({ m, n, th }) => { const q = [0.9, 0.3], k = [0.5, -0.8], rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)], Q = rot(q, m * th), K = rot(k, n * th), dot = (a, b) => a[0] * b[0] + a[1] * b[1]; return { Q, K, s: dot(Q, K), rel: dot(rot(q, (m - n) * th), k), q, k, rot, dot }; },
  plots: v => [{ x: [-1.2, 1.2], y: [-1.2, 1.2], w: 440, h: 420, label: 'rotated query and key', marks: [{ as: 'arrow', tone: 1, data: [[0, 0, v.Q[0], v.Q[1]]] }, { as: 'arrow', tone: 2, data: [[0, 0, v.K[0], v.K[1]]] }] }, { caption: 'Score against relative position', x: [-40, 40], y: [-1.2, 1.2], w: 500, h: 320, xl: 'm − n', label: 'score by relative position', marks: [{ tone: 3, data: curve(d => v.dot(v.rot(v.q, d * v.th), v.k), -40, 40, 401) }, { as: 'points', tone: 4, r: 6, data: [[v.m - v.n, v.s]] }] }],
  stats: v => [`score ${fmt(v.s, 5)}`, `from relative position alone ${fmt(v.rel, 5)}`],
  legend: [['--c1', 'query'], ['--c2', 'key']],
  takeaway: 'Only relative position reaches the attention score.',
});
export const CvRoofline = lab({
  intro: 'Each step takes as long as the slower of two limits. Past the knee, a faster compute unit does not help at all.',
  tex: R`t=\max\Big(\frac{\text{FLOPs}}{\text{compute}},\ \frac{\text{bytes}}{\text{bandwidth}}\Big)`,
  params: [{ k: 'f', label: 'GFLOPs per token', min: 5, max: 400, step: 5, init: 80 }, { k: 'b', label: 'GB moved per token', min: 0.5, max: 80, step: 0.5, init: 14 }, { k: 'c', label: 'compute (TFLOP/s)', min: 5, max: 200, step: 5, init: 60 }, { k: 'w', label: 'bandwidth (GB/s)', min: 50, max: 2000, step: 50, init: 500 }],
  derive: ({ f, b, c, w }) => { const tc = f / c, tm = (b / w) * 1000; return { tc, tm, t: Math.max(tc, tm) }; },
  plots: v => [{ x: [5, 200], y: [0, Math.max(6, v.t * 2.2)], w: 560, h: 340, xl: 'compute (TFLOP/s)', yl: 'ms per token', label: 'token latency against compute', marks: [{ tone: 2, data: curve(c => Math.min(1e3, v.f / c), 5, 200) }, { tone: 3, data: [[5, v.tm], [200, v.tm]] }, { tone: 1, data: curve(c => Math.max(v.f / c, v.tm), 5, 200) }, { as: 'points', tone: 4, r: 6, data: [[v.c, v.t]] }] }],
  stats: v => [`${fmt(v.t, 2)} ms per token`, `about ${fmt(1000 / v.t, 0)} tokens per second`],
  status: v => (v.tm > v.tc ? ['warn', 'Memory-bound: only more bandwidth, or smaller weights, will help.'] : ['ok', 'Compute-bound: faster arithmetic helps.']),
  legend: [['--c2', 'compute limit'], ['--c3', 'memory limit'], ['--c1', 'actual']],
  takeaway: 'Check which limit binds before trusting a chip’s peak number.',
});
const batchSim = memo((spread, B) => { const r = rng(31), L = Array.from({ length: 1200 }, () => Math.max(1, Math.round(Math.exp(5 + spread * gauss(r))))); let st = 0; for (let i = 0; i < L.length; i += B) st += Math.max(...L.slice(i, i + B)); const s = Array(B).fill(0); let q = 0, n = 0; while (q < L.length || s.some(x => x > 0)) { for (let i = 0; i < B; i++) if (s[i] === 0 && q < L.length) s[i] = L[q++]; for (let i = 0; i < B; i++) s[i] = Math.max(0, s[i] - 1); n++; } return { st, n, tot: L.reduce((a, b) => a + b, 0) }; });
export const CvContinuousBatching = lab({
  intro: 'A static batch waits for its longest request. Continuous batching refills a slot as soon as it finishes.',
  tex: R`\text{gain}=\frac{\text{static steps}}{\text{continuous steps}}`,
  params: [{ k: 's', label: 'length spread', min: 0, max: 1.5, step: 0.05, init: 0.8 }, { k: 'b', label: 'batch slots', min: 2, max: 64, step: 2, init: 16 }],
  derive: ({ s, b }) => ({ ...batchSim(s, b), gains: [2, 4, 8, 16, 32, 64].map(B => [0, 0.3, 0.6, 0.9, 1.2, 1.5].map(S => { const R0 = batchSim(S, B); return R0.st / R0.n; })) }),
  plots: v => [heatPlot(v.gains, { w: 600, h: 330, label: 'throughput gain by slots and spread', over: [heatBox(v.gains, [2, 4, 8, 16, 32, 64].reduce((b, x, i) => (Math.abs(x - v.b) < Math.abs([2, 4, 8, 16, 32, 64][b] - v.b) ? i : b), 0), [0, 0.3, 0.6, 0.9, 1.2, 1.5].reduce((b, x, i) => (Math.abs(x - v.s) < Math.abs([0, 0.3, 0.6, 0.9, 1.2, 1.5][b] - v.s) ? i : b), 0))], heat: { min: 1, max: Math.max(...v.gains.flat()), rows: [2, 4, 8, 16, 32, 64].map(x => `${x} slots`), cols: ['0', '.3', '.6', '.9', '1.2', '1.5'], show: x => x.toFixed(1), name: (i, j, x) => `gain ${x.toFixed(2)}` } })],
  stats: v => [`gain ${fmt(v.st / v.n, 2)} times`, `steps: ${v.st.toLocaleString('en-US')} static, ${v.n.toLocaleString('en-US')} continuous`],
  takeaway: 'The gain comes from variable lengths, so it disappears if every response has the same size.',
});
export const CvCheckpointDrill = drill({
  intro: 'Store a few activations, and recompute one segment at a time on the way back.',
  tex: R`\text{peak}\approx k+\frac nk`,
  gen: r => { const n = pick(r, [64, 100, 144, 400]); return { n, k: pick(r, [Math.sqrt(n), 4, 2 * Math.sqrt(n)]) }; },
  q: v => `A network has $${v.n}$ layers split into $${v.k}$ checkpointed segments. About how many activations are held at the peak?`,
  answer: v => v.k + v.n / v.k, dp: 0,
  steps: v => [`Stored boundaries: $${v.k}$`, `One segment recomputed: $${v.n / v.k}$`],
});
const boxArea = b => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export const CvBoxLosses = lab({
  intro: 'Drag the prediction away from the target. IoU sits at zero as soon as the boxes separate, while GIoU and DIoU keep measuring.',
  tex: R`\text{GIoU}=\text{IoU}-\frac{|C\setminus(A\cup B)|}{|C|}`,
  params: [{ k: 'x', label: 'prediction $x$', min: 0.5, max: 13, step: 0.1, init: 5.5 }, { k: 'y', label: 'prediction $y$', min: 0.5, max: 11, step: 0.1, init: 6 }, { k: 'w', label: 'width', min: 1, max: 6, step: 0.1, init: 3 }, { k: 'h', label: 'height', min: 1, max: 6, step: 0.1, init: 2 }],
  derive: ({ x, y, w, h }) => { const A = [3, 3, 6, 6], B = [x - w / 2, y - h / 2, x + w / 2, y + h / 2], I = [Math.max(A[0], B[0]), Math.max(A[1], B[1]), Math.min(A[2], B[2]), Math.min(A[3], B[3])], inter = I[2] > I[0] && I[3] > I[1] ? boxArea(I) : 0, u = boxArea(A) + boxArea(B) - inter, C = [Math.min(A[0], B[0]), Math.min(A[1], B[1]), Math.max(A[2], B[2]), Math.max(A[3], B[3])], iou = inter / u, giou = iou - (boxArea(C) - u) / boxArea(C), rho = (4.5 - x) ** 2 + (4.5 - y) ** 2, diag = (C[2] - C[0]) ** 2 + (C[3] - C[1]) ** 2; return { A, B, C, iou, giou, diou: iou - rho / diag }; },
  plots: v => { const rect = (b, tone, dash) => ({ tone, dash, data: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]] }); return [{ x: [0, 14], y: [0, 12], w: 500, h: 430, label: 'target, prediction and enclosing box', marks: [rect(v.C, 'muted', true), rect(v.A, 3), rect(v.B, 2), { drag: { x: 'x', y: 'y', tone: 2 } }] }]; },
  stats: v => [`IoU ${fmt(v.iou, 3)}`, `GIoU ${fmt(v.giou, 3)}`, `DIoU ${fmt(v.diou, 3)}`],
  legend: [['--c3', 'target'], ['--c2', 'prediction'], ['--muted', 'enclosing box']],
  takeaway: 'A loss needs a slope, so it must keep changing even when the boxes do not overlap.',
});
export const CvSmoothL1 = lab({
  intro: 'Near zero the loss is a parabola, and past $\\beta$ it is a line. Outliers therefore add only a limited gradient.',
  tex: R`\text{SmoothL1}(x)=\begin{cases}\tfrac12x^2/\beta&|x|<\beta\\|x|-\tfrac12\beta&\text{otherwise}\end{cases}`,
  params: [{ k: 'b', label: 'threshold $\\beta$', min: 0.1, max: 3, step: 0.05, init: 1 }, { k: 'x', label: 'error $x$', min: -4, max: 4, step: 0.05, init: 2 }],
  derive: ({ b, x }) => { const s = q => (Math.abs(q) < b ? (0.5 * q * q) / b : Math.abs(q) - 0.5 * b); return { s, g: Math.abs(x) < b ? x / b : Math.sign(x) }; },
  plots: v => [{ x: [-4, 4], y: [0, 5], w: 560, h: 320, xl: 'error', yl: 'loss', label: 'smooth L1 compared with L1 and L2', marks: [{ tone: 'muted', data: curve(q => Math.min(5, 0.5 * q * q), -4, 4) }, { tone: 2, data: curve(Math.abs, -4, 4) }, { tone: 1, data: curve(v.s, -4, 4, 401) }, { as: 'points', tone: 4, r: 6, data: [[v.x, v.s(v.x)]] }] }],
  stats: v => [`loss ${fmt(v.s(v.x), 3)}`, `gradient ${fmt(v.g, 3)}`],
  legend: [['--muted', 'L2'], ['--c2', 'L1'], ['--c1', 'smooth L1']],
  takeaway: 'Its gradient never exceeds one in size, unlike the squared loss.',
});
export const CvPosteriorCollapse = lab({
  intro: 'A linear VAE has an exact answer. If the data variance is below $\\beta\\sigma^2$, the best encoder sends no information at all.',
  tex: R`\text{KL}^*=\tfrac12\max\Big(0,\ \ln\frac{s^2}{\beta\sigma^2}\Big)`,
  params: [{ k: 's', label: 'data variance $s^2$', min: 0.2, max: 6, step: 0.1, init: 2 }, { k: 'b', label: '$\\beta$', min: 0.1, max: 8, step: 0.1, init: 1 }, { k: 'n', label: 'decoder noise $\\sigma^2$', min: 0.1, max: 3, step: 0.1, init: 1 }],
  derive: ({ s, b, n }) => { const K = B => 0.5 * Math.max(0, Math.log(s / (B * n))), w2 = Math.max(0, s - b * n); return { K, w2, kl: K(b), edge: s / n }; },
  plots: v => [{ x: [0.1, 8], y: [0, 2.2], w: 560, h: 320, xl: 'β', yl: 'KL in nats', label: 'rate against beta', marks: [{ tone: 1, data: curve(v.K, 0.1, 8, 401) }, { as: 'rule', dash: true, tone: 4, x: Math.min(8, v.edge) }, { as: 'points', tone: 2, r: 6, data: [[v.b, v.kl]] }] }],
  stats: v => [`KL ${fmt(v.kl, 3)} nats`, `decoder weight squared ${fmt(v.w2, 3)}`, `collapse begins at $\\beta=${fmt(v.edge, 2)}$`],
  status: v => (v.kl === 0 ? ['alert', 'Posterior collapse: the latent carries nothing.'] : undefined),
  takeaway: 'Collapse can be the optimal answer, not just an optimiser failure.',
});
export const CvFidDrill = drill({
  intro: 'In one dimension, the Fréchet distance is the gap in means squared plus the gap in standard deviations squared.',
  tex: R`\text{FID}=(\mu_1-\mu_2)^2+(\sigma_1-\sigma_2)^2`,
  gen: r => ({ a: int(r, 0, 4), b: int(r, 0, 4), s: int(r, 1, 4), t: int(r, 1, 4) }),
  q: v => `Real features are $\\mathcal N(${v.a},${v.s * v.s})$ and generated features are $\\mathcal N(${v.b},${v.t * v.t})$. Find the FID.`,
  answer: v => (v.a - v.b) ** 2 + (v.s - v.t) ** 2, dp: 0,
  steps: v => [`Means: $${(v.a - v.b) ** 2}$`, `Spreads: $${(v.s - v.t) ** 2}$`],
});
export const CvSmoothingRadius = lab({
  intro: 'A smoothed classifier has a guarantee: no $\\ell_2$ perturbation within this radius can change its answer.',
  tex: R`R=\sigma\,\Phi^{-1}(p_A)`,
  params: [{ k: 's', label: 'noise $\\sigma$', min: 0.1, max: 1, step: 0.05, init: 0.5 }, { k: 'p', label: 'top-class probability $p_A$', min: 0.51, max: 0.999, step: 0.001, init: 0.9 }],
  plots: v => [{ x: [0.5, 1], y: [0, 3.2], w: 560, h: 320, xl: 'p_A', yl: 'certified radius', label: 'certified radius', marks: [{ tone: 1, data: linspace(0.501, 0.999, 120).map(p => [p, v.s * PhiInv(p)]) }, { as: 'points', tone: 4, r: 6, data: [[v.p, v.s * PhiInv(v.p)]] }] }],
  stats: v => [`radius ${fmt(v.s * PhiInv(v.p), 3)}`],
  takeaway: 'More noise certifies a larger radius, but makes the base classifier’s job harder.',
});
export const CvGae = lab({
  intro: 'Each residual further ahead counts for a factor $\\gamma\\lambda$ less. $\\lambda$ moves smoothly from a one-step estimate to a full return.',
  tex: R`\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}`,
  params: [{ k: 'g', label: 'discount $\\gamma$', min: 0.8, max: 1, step: 0.005, init: 0.99 }, { k: 'l', label: '$\\lambda$', min: 0, max: 1, step: 0.01, init: 0.95 }],
  derive: ({ g, l }) => ({ w: Array.from({ length: 40 }, (_, i) => r6((g * l) ** i)), hor: g * l < 1 ? 1 / (1 - g * l) : Infinity }),
  plots: v => [{ x: [-0.5, 39.5], y: [0, 1.05], w: 560, h: 320, xl: 'steps ahead', yl: 'weight', label: 'weight of future residuals', marks: [{ as: 'stems', tone: 1, data: v.w.map((y, i) => [i, 0, y]) }, { as: 'points', tone: 1, r: 3, data: v.w.map((y, i) => [i, y]) }] }],
  stats: v => [`effective horizon ${Number.isFinite(v.hor) ? fmt(v.hor, 1) : 'unlimited'} steps`],
  takeaway: 'Low values trust the value function, and high values trust the observed rewards.',
});
const GLYPH = { T: ['########', '########', '...##...', '...##...', '...##...', '...##...', '...##...', '...##...'], L: ['##......', '##......', '##......', '##......', '##......', '##......', '########', '########'], X: ['##....##', '.##..##.', '..####..', '...##...', '...##...', '..####..', '.##..##.', '##....##'] };
const glyph = k => GLYPH[k].flatMap(r => [...r].map(c => (c === '#' ? 1 : -1)));
const HW = (() => { const P = Object.keys(GLYPH).map(glyph), N = 64; return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i === j ? 0 : P.reduce((s, p) => s + p[i] * p[j], 0) / N))); })();
export const CvHopfield = lab({
  intro: 'A network that stores patterns in its weights. Corrupt one, and repeated updates usually fall back to the stored memory.',
  tex: R`s_i\leftarrow\text{sign}\Big(\sum_jW_{ij}s_j\Big)`,
  params: [{ k: 'g', label: 'Stored letter', init: 'T', options: Object.keys(GLYPH).map(k => [k, k]) }, { k: 'n', label: 'flipped pixels', min: 0, max: 32, step: 1, init: 16 }, { k: 't', label: 'update sweeps', min: 0, max: 5, step: 1, init: 1, play: true }, { k: 'seed', seed: 'New noise', init: 1 }],
  derive: ({ g, n, t, seed }) => { const r = rng(seed * 61), T = glyph(g), s = T.slice(), idx = T.map((_, i) => i); for (let i = 63; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } idx.slice(0, n).forEach(i => { s[i] = -s[i]; }); const start = s.slice(); for (let k = 0; k < t; k++) idx.forEach(i => { const h = HW[i].reduce((a, w, j) => a + w * s[j], 0); s[i] = h >= 0 ? 1 : -1; }); const grid = v => Array.from({ length: 8 }, (_, i) => v.slice(i * 8, i * 8 + 8).map(x => (x + 1) / 2)), E = -0.5 * s.reduce((a, x, i) => a + x * HW[i].reduce((b, w, j) => b + w * s[j], 0), 0); return { A: grid(start), B: grid(s), match: s.filter((x, i) => x === T[i]).length / 64, E }; },
  plots: v => [heatPlot(v.A, { w: 360, h: 340, caption: 'Corrupted input', label: 'corrupted input', heat: { max: 1, dense: true } }), heatPlot(v.B, { w: 360, h: 340, caption: 'After updates', label: 'state after updates', heat: { max: 1, dense: true } })],
  stats: v => [`${pct(v.match, 0)} of pixels match`, `energy ${fmt(v.E, 2)}`],
  takeaway: 'Each update can only lower the energy, so the state falls into a nearby basin.',
});
export const CvSharpness = lab({
  intro: 'Sharpness-aware minimisation optimises the worst loss within a small radius. The sharp minimum looks far worse under that view.',
  tex: R`\min_w\max_{|\epsilon|\le\rho}L(w+\epsilon)`,
  params: [{ k: 'r', label: 'radius $\\rho$', min: 0, max: 1, step: 0.05, init: 0.4 }],
  derive: ({ r }) => { const L = w => Math.min(10 * (w + 2) ** 2 - 0.2, 0.3 * (w - 2) ** 2), S = w => Math.max(...linspace(w - r, w + r, 21).map(L)), xs = linspace(-4, 5, 181); return { L, S, xs, sharp: S(-2), flat: S(2) }; },
  plots: v => [{ x: [-4, 5], y: [-0.5, 5], w: 560, h: 320, xl: 'w', yl: 'loss', label: 'loss and sharpness-aware loss', marks: [{ tone: 'muted', data: v.xs.map(w => [w, Math.min(5, v.L(w))]) }, { tone: 1, data: v.xs.map(w => [w, Math.min(5, v.S(w))]) }] }],
  stats: v => [`worst nearby loss at the sharp minimum ${fmt(v.sharp, 3)}`, `at the flat minimum ${fmt(v.flat, 3)}`],
  legend: [['--muted', 'plain loss'], ['--c1', 'worst within the radius']],
  takeaway: 'A minimum is valuable if small shifts in the weights keep the loss small.',
});
export const CvWarmRestarts = lab({
  intro: 'The learning rate falls along a cosine, and then jumps back. Each cycle can be longer than the last.',
  tex: R`\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos(\pi t_c/T_c)\big)`,
  params: [{ k: 'T', label: 'first cycle length', min: 5, max: 60, step: 5, init: 20 }, { k: 'm', label: 'length multiplier', min: 1, max: 3, step: 0.5, init: 2 }, { k: 'lo', label: 'minimum rate', min: 0, max: 0.5, step: 0.05, init: 0.05 }],
  derive: ({ T, m, lo }) => { const P = []; let t = 0, c = T, cycles = 0; while (t < 200) { for (let i = 0; i < c && t < 200; i++, t++) P.push([t, r6(lo + 0.5 * (1 - lo) * (1 + Math.cos((Math.PI * i) / c)))]); c = Math.round(c * m); cycles++; } return { P, cycles }; },
  plots: v => [{ x: [0, 200], y: [0, 1.05], w: 560, h: 300, xl: 'epoch', yl: 'relative rate', label: 'cosine schedule with warm restarts', marks: [{ tone: 1, data: v.P }] }],
  stats: v => [`${v.cycles} cycles in 200 epochs`],
  takeaway: 'A restart lets training leave one basin while still ending with a small step.',
});
