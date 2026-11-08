'use client';
// Interactive versions of the tutorial sheets. Each drill opens on the sheet's own numbers, then generates fresh ones.
import { R, curve, drill, int, lab, mat, pick } from '../components/labs';
import { mean, sum } from '../lib/math.mjs';
import { clamp, eig2, fmt, linspace, normalPdf, pct, sigmoid } from '../lib/models.mjs';

/* ---------- mathematical readiness ---------- */

export const TmShape = drill({
  intro: 'Inner dimensions agree, and outer dimensions survive. Answer like 3x4, or none.',
  gen: (r, n) => (n === 0 ? { m: 3, a: 2, b: 2, q: 4 } : { m: int(r, 1, 6), a: int(r, 1, 6), b: pick(r, [0, 0, 1]) ? int(r, 1, 6) : 0, q: int(r, 1, 6) }),
  q: v => { const b = v.b || v.a; return `$A\\in\\mathbb R^{${v.m}\\times${v.a}}$ and $B\\in\\mathbb R^{${b}\\times${v.q}}$. Shape of $AB$?`; },
  answer: v => ((v.b || v.a) === v.a ? `${v.m}x${v.q}` : 'none'),
  steps: v => [`Inner: ${v.a} and ${v.b || v.a}.`, 'If they agree, the outer dimensions give the shape.'],
});

export const TmExpressions = drill({
  intro: 'Name each expression’s shape, for $X\\in\\mathbb R^{n\\times d}$, $\\theta\\in\\mathbb R^{d}$, $y\\in\\mathbb R^{n}$.',
  gen: r => pick(r, [{ e: R`X\theta`, a: 0 }, { e: R`y-X\theta`, a: 0 }, { e: R`X^\top y`, a: 1 }, { e: R`X^\top X`, a: 2 }, { e: R`Xy`, a: 3 }, { e: R`\theta^\top X^\top y`, a: 4 }]),
  q: v => `What is the shape of $${v.e}$?`,
  choices: () => ['$n$ by one', '$d$ by one', '$d$ by $d$', 'Invalid', 'A scalar'],
  answer: v => v.a,
  steps: () => ['Write the shape under each symbol, and cancel neighbouring inner dimensions.'],
});

export const TmForwardPass = lab({
  intro: 'A two-layer network is a chain of shape changes. Change the widths, and see every intermediate shape.',
  tex: R`z=W_2(W_1x+b_1)+b_2`,
  params: [{ k: 'd', label: 'inputs', min: 1, max: 10, step: 1, init: 4 }, { k: 'h', label: 'hidden units', min: 1, max: 10, step: 1, init: 6 }, { k: 'o', label: 'outputs', min: 1, max: 10, step: 1, init: 2 }],
  plots: v => { const node = (x, n, t) => Array.from({ length: n }, (_, i) => [x, (i - (n - 1) / 2) * 0.9, t]); const a = node(0, v.d, 1), b = node(1, v.h, 2), c = node(2, v.o, 3); return [{ x: [-0.3, 2.3], y: [-5, 5], w: 380, h: 320, label: 'network layers', marks: [...a.flatMap(p => b.map(q => ({ tone: 'muted', soft: true, data: [p, q] }))), ...b.flatMap(p => c.map(q => ({ tone: 'muted', soft: true, data: [p, q] }))), { as: 'points', r: 6, data: [...a, ...b, ...c] }] }]; },
  table: v => ({ head: ['Object', 'Shape'], rows: [['$x$', `${v.d} × 1`], ['$W_1$', `${v.h} × ${v.d}`], ['$W_1x+b_1$', `${v.h} × 1`], ['$W_2$', `${v.o} × ${v.h}`], ['$z$', `${v.o} × 1`]] }),
  stats: v => [`${v.h * v.d + v.h + v.o * v.h + v.o} parameters`],
});

export const TmSigmoid = drill({
  intro: 'Use $e^{-2}\\approx0.135$ and $e^{2}\\approx7.39$.',
  tex: R`\sigma(z)=\frac{1}{1+e^{-z}}`,
  gen: (r, n) => ({ z: n === 0 ? 2 : pick(r, [-2, 0, 2]) }),
  q: v => `Compute $\\sigma(${v.z})$.`,
  answer: v => sigmoid(v.z), dp: 3,
  steps: v => [`$e^{${-v.z}}\\approx${v.z === 2 ? '0.135' : v.z === -2 ? '7.39' : '1'}$`, `$1/(1+${v.z === 2 ? '0.135' : v.z === -2 ? '7.39' : '1'})$`],
});

export const TmRelu = lab({
  intro: 'ReLU is flat, then a ramp. Slide towards the corner from each side and compare the slopes: they never agree at zero.',
  tex: R`m(z)=\max(0,z)`,
  params: [{ k: 'z', label: '$z$', min: -3, max: 3, step: 0.05, init: -1 }],
  plots: v => [{ x: [-3, 3], y: [-1, 3], xl: 'z', yl: 'm(z)', label: 'ReLU', marks: [{ tone: 1, data: curve(z => Math.max(0, z), -3, 3, 121) }, { drag: { x: 'z', y: Math.max(0, v.z) } }] }],
  stats: v => [`m(z) = ${fmt(Math.max(0, v.z))}`, `slope ${v.z === 0 ? 'undefined' : v.z > 0 ? '1' : '0'}`],
  takeaway: 'At zero the derivative does not exist. Software picks a value from the subgradient range and carries on.',
});

export const TmVertex = drill({
  intro: 'Set the derivative to zero, then read the minimum value.',
  gen: (r, n) => (n === 0 ? { a: 3, c: 2 } : { a: int(r, -5, 5), c: int(r, -3, 6) }),
  q: v => `For $h(w)=(w-(${v.a}))^2+(${v.c})$, what is the minimum value?`,
  answer: v => v.c, dp: 0,
  steps: v => [`$h'(w)=2(w-(${v.a}))=0$, so $w^\\star=${v.a}$.`, `$h''=2>0$, so this is a minimum.`, `$h(${v.a})=${v.c}$`],
});

export const TmQuartic = lab({
  intro: 'Two valleys and a hill. Drag along the curve and run the second-derivative test at each critical point.',
  tex: R`g(u)=u^4-2u^2,\quad g'(u)=4u^3-4u,\quad g''(u)=12u^2-4`,
  params: [{ k: 'u', label: '$u$', min: -1.8, max: 1.8, step: 0.01, init: 0.5 }],
  plots: v => [{ x: [-1.8, 1.8], y: [-1.5, 2], xl: 'u', yl: 'g(u)', label: 'quartic with critical points', marks: [{ tone: 1, data: curve(u => u ** 4 - 2 * u * u, -1.8, 1.8) }, { as: 'points', tone: 3, r: 5, data: [[-1, -1], [0, 0], [1, -1]] }, { drag: { x: 'u', y: v.u ** 4 - 2 * v.u * v.u } }] }],
  stats: v => [`g = ${fmt(v.u ** 4 - 2 * v.u ** 2, 3)}`, `g′ = ${fmt(4 * v.u ** 3 - 4 * v.u, 3)}`, `g″ = ${fmt(12 * v.u * v.u - 4, 3)}`],
  status: v => (Math.abs(v.u) < 0.02 ? [undefined, 'Critical point with g″ < 0: a local maximum.'] : Math.abs(Math.abs(v.u) - 1) < 0.02 ? ['ok', 'Critical point with g″ > 0: a global minimum, and there are two.'] : undefined),
});

export const TmExponent = drill({
  intro: 'Write the exponent of a Gaussian density.',
  tex: R`-\frac{(x-m)^2}{2s^2}`,
  gen: (r, n) => (n === 0 ? { x: 90, m: 120, s: 30 } : { x: pick(r, [60, 90, 120]), m: pick(r, [120, 40]), s: pick(r, [20, 30]) }),
  q: v => `Compute the exponent at $x=${v.x}$ with $m=${v.m}$ and $s=${v.s}$.`,
  answer: v => -((v.x - v.m) ** 2) / (2 * v.s ** 2), dp: 3,
  steps: v => [`$(x-m)^2=${(v.x - v.m) ** 2}$`, `$2s^2=${2 * v.s ** 2}$`],
});

export const TmFloodBayes = lab({
  intro: 'The whole calculation of Station Four on one screen. Slide the rainfall and watch the likelihood ratio and posterior respond.',
  tex: R`\frac{P(\text{flood}\mid x)}{P(\text{none}\mid x)}=\frac{p(x\mid\text{flood})}{p(x\mid\text{none})}\cdot\frac{P(\text{flood})}{P(\text{none})}`,
  params: [{ k: 'x', label: 'rainfall in mm', min: 0, max: 200, step: 1, init: 90 }, { k: 'pi', label: 'prior of flood', min: 0.01, max: 0.5, step: 0.01, init: 0.1 }],
  derive: ({ pi }) => ({ lr: x => normalPdf(x, 120, 30) / normalPdf(x, 40, 20), post: x => (pi * normalPdf(x, 120, 30)) / (pi * normalPdf(x, 120, 30) + (1 - pi) * normalPdf(x, 40, 20)) }),
  plots: v => [{ x: [0, 200], y: [0, 1], xl: 'rainfall (mm)', yl: 'probability of flood', label: 'posterior against rainfall', marks: [{ tone: 1, data: curve(v.post, 0, 200, 241) }, { drag: { x: 'x', y: v.post(v.x) } }] }],
  stats: v => [`posterior ${pct(v.post(v.x))}`, `likelihood ratio ${fmt(v.lr(v.x), 2)}`, `posterior odds ${fmt((v.lr(v.x) * v.pi) / (1 - v.pi), 2)}`],
});

export const TmOdds = drill({
  intro: 'Multiply prior odds by the likelihood ratio, then convert odds to a probability.',
  tex: R`p=\frac{\text{odds}}{1+\text{odds}}`,
  gen: (r, n) => (n === 0 ? { pr: 9, lr: 9.2 } : { pr: pick(r, [1, 4, 9, 19]), lr: pick(r, [0.5, 2, 4, 9.2, 20]) }),
  q: v => `Prior odds are 1 : ${v.pr}, and the likelihood ratio is ${v.lr}. Find the posterior probability.`,
  answer: v => (v.lr / v.pr) / (1 + v.lr / v.pr), dp: 3,
  steps: v => [`Posterior odds: $${v.lr}/${v.pr}=${fmt(v.lr / v.pr, 3)}$`, 'Then apply $p=\\text{odds}/(1+\\text{odds})$.'],
});

/* ---------- estimating theta ---------- */

const TIMES = [2, 3, 3, 4, 4, 4, 5, 7];

export const TeSkew = lab({
  intro: 'Raw times are badly skewed, and their logs are not. Slide between the two views.',
  params: [{ k: 'view', label: 'View', options: [['raw', 'time x'], ['log', 'log x']], init: 'raw' }],
  plots: v => { const d = TIMES.map(t => (v.view === 'raw' ? Math.exp(t) : t)), hi = v.view === 'raw' ? 1200 : 8; return [{ x: [0, hi], y: [0, 4], xl: v.view === 'raw' ? 'milliseconds' : 'log of milliseconds', label: 'observations on a line', marks: d.map((x, i) => ({ as: 'points', tone: 1, r: 6, data: [[x, 0.5 + TIMES.slice(0, i).filter(t => t === TIMES[i]).length]] })) }]; },
  stats: v => { const d = TIMES.map(t => (v.view === 'raw' ? Math.exp(t) : t)); return [`mean ${fmt(mean(d), 1)}`, `largest ÷ smallest ${fmt(Math.max(...d) / Math.min(...d), 1)}`]; },
  takeaway: 'A symmetric bell suits the log row. Modelling $\\log X$ is what makes the lognormal the right tool.',
});

const gLog = r => Array.from({ length: 8 }, () => int(r, 1, 7));

export const TeTheta1 = drill({
  intro: 'Setting $\\partial\\ell/\\partial\\theta_1=0$ gives the mean of the logs.',
  tex: R`\hat\theta_1=\frac1n\sum_i y_i,\quad y_i=\log x_i`,
  gen: (r, n) => ({ y: n === 0 ? TIMES : gLog(r) }),
  q: v => `The logs are $${v.y.join(',\\ ')}$. Find $\\hat\\theta_1$.`,
  answer: v => mean(v.y),
  steps: v => [`Sum: $${sum(v.y)}$. Count: $${v.y.length}$.`],
});

export const TeTheta2 = drill({
  intro: 'The variance estimate uses $S$, the sum of squared deviations from $\\hat\\theta_1$.',
  tex: R`\hat\theta_2=\frac{S}{n},\quad S=\sum_i(y_i-\hat\theta_1)^2`,
  gen: (r, n) => ({ y: n === 0 ? TIMES : gLog(r) }),
  q: v => `Logs: $${v.y.join(',\\ ')}$. Find $\\hat\\theta_2$.`,
  answer: v => { const m = mean(v.y); return mean(v.y.map(x => (x - m) ** 2)); },
  steps: v => { const m = mean(v.y); return [`$\\hat\\theta_1=${fmt(m)}$`, `$S=${fmt(sum(v.y.map(x => (x - m) ** 2)))}$`, `Divide by ${v.y.length}.`]; },
});

export const TeMoves = drill({
  intro: 'Both parts of the maximisation used the same recipe.',
  gen: r => ({ k: pick(r, [0, 1, 2]) }),
  q: v => ['What is the first move after writing $\\ell$?', 'What do you set the derivative equal to?', 'What is the last move?'][v.k],
  choices: () => ['Differentiate with respect to the unknown', 'Zero', 'Solve for the unknown', 'Integrate over the data'],
  answer: v => v.k,
  steps: () => ['Differentiate, set to zero, solve.'],
});

export const TePosteriorPowers = drill({
  intro: 'Same shape, so powers add and constants add. The result is an inverse-gamma again.',
  tex: R`\text{IG}(a,b)\ \times\ \text{data}(n,S)\ \to\ \text{IG}\big(a+\tfrac n2,\ b+\tfrac S2\big)`,
  gen: (r, n) => (n === 0 ? { a: 3, b: 4, n: 8, S: 16, k: 'a' } : { a: int(r, 2, 6), b: int(r, 1, 8), n: 2 * int(r, 2, 8), S: 2 * int(r, 2, 15), k: pick(r, ['a', 'b']) }),
  q: v => `Prior IG(${v.a}, ${v.b}) with $n=${v.n}$ and $S=${v.S}$. What is the posterior ${v.k === 'a' ? 'shape $a$' : 'scale $b$'}?`,
  answer: v => (v.k === 'a' ? v.a + v.n / 2 : v.b + v.S / 2), dp: 0,
  steps: v => [v.k === 'a' ? `$${v.a}+${v.n}/2$` : `$${v.b}+${v.S}/2$`],
});

export const TePosteriorEstimates = drill({
  intro: 'Two summaries of the inverse-gamma posterior.',
  tex: R`\text{mode}=\frac{b}{a+1},\qquad \text{mean}=\frac{b}{a-1}`,
  gen: (r, n) => (n === 0 ? { a: 7, b: 12, k: 'mode' } : { a: int(r, 3, 10), b: int(r, 2, 20), k: pick(r, ['mode', 'mean']) }),
  q: v => `For IG(${v.a}, ${v.b}), find the posterior ${v.k}.`,
  answer: v => (v.k === 'mode' ? v.b / (v.a + 1) : v.b / (v.a - 1)), dp: 3,
  steps: v => [`$${v.b}/(${v.a}${v.k === 'mode' ? '+' : '-'}1)$`],
});

export const TeContest = lab({
  intro: 'A prior note says spread near one, and the data say two. Move each side’s weight and see the compromise.',
  tex: R`\text{estimate}=\frac{w_{\text{prior}}\cdot\text{prior}+w_{\text{data}}\cdot\text{data}}{w_{\text{prior}}+w_{\text{data}}}`,
  params: [{ k: 'a', label: 'prior weight', min: 0, max: 20, step: 0.5, init: 4 }, { k: 'n', label: 'data weight $n/2$', min: 0.5, max: 40, step: 0.5, init: 4 }],
  plots: v => [{ x: [0.5, 40], y: [0.8, 2.2], xl: 'data weight', yl: 'combined estimate', label: 'prior and data compromise', marks: [{ as: 'rule', tone: 2, dash: true, y: 1 }, { as: 'rule', tone: 3, dash: true, y: 2 }, { tone: 1, data: curve(n => (v.a * 1 + n * 2) / (v.a + n), 0.5, 40) }, { drag: { x: 'n', y: (v.a + v.n * 2) / (v.a + v.n) } }] }],
  stats: v => [`combined ${fmt((v.a + v.n * 2) / (v.a + v.n), 3)}`],
  takeaway: 'At equal weights the answer is midway, at 1.5. That is the tie the sheet builds in on purpose.',
});

export const TeLognormal = lab({
  intro: 'The lognormal, in both units. The mean sits well above the median, pulled by the long right tail.',
  tex: R`\text{median}=e^{\theta_1},\qquad \text{mean}=e^{\theta_1+\theta_2/2}`,
  params: [{ k: 't1', label: '$\\theta_1$', min: 1, max: 6, step: 0.05, init: 4 }, { k: 't2', label: '$\\theta_2$', min: 0.1, max: 3, step: 0.05, init: 2 }],
  plots: v => [{ x: [0, 8], y: [0, 0.9], xl: 'log x', yl: 'density of log x', label: 'normal density of the log', marks: [{ as: 'area', tone: 1, data: linspace(0, 8, 161).map(x => [x, 0, normalPdf(x, v.t1, Math.sqrt(v.t2))]) }, { as: 'rule', dash: true, x: v.t1 }, { as: 'rule', tone: 4, x: clamp(v.t1 + v.t2 / 2, 0, 8) }] }],
  stats: v => [`median ${fmt(Math.exp(v.t1), 1)}`, `mean ${fmt(Math.exp(v.t1 + v.t2 / 2), 1)}`],
});

export const TeLogLik = drill({
  intro: 'Only two terms contain the unknowns.',
  tex: R`\ell=-\frac n2\ln\theta_2-\frac{1}{2\theta_2}\sum_i(y_i-\theta_1)^2+\text{const}`,
  gen: (r, n) => (n === 0 ? { n: 8, S: 16, t: 2 } : { n: 2 * int(r, 2, 8), S: 2 * int(r, 2, 15), t: pick(r, [1, 2, 4]) }),
  q: v => `With $n=${v.n}$, $S=${v.S}$ and $\\theta_2=${v.t}$, find the second term, $-S/(2\\theta_2)$.`,
  answer: v => -v.S / (2 * v.t),
  steps: v => [`$-${v.S}/(2\\cdot${v.t})$`],
});

export const TeMedian = drill({
  intro: 'Since $\\log X$ is symmetric, the median of $X$ is $e$ to the centre of the logs.',
  gen: (r, n) => ({ t: n === 0 ? 4 : int(r, 1, 6) }),
  q: v => `With $\\hat\\theta_1=${v.t}$, what is the estimated median time, in ms?`,
  answer: v => Math.exp(v.t), dp: 1,
  steps: v => [`$e^{${v.t}}$`],
});

/* ---------- linear regression foundations ---------- */

const X1 = [[1, 0], [1, 1], [1, 2], [1, 3]], Y1 = [2, 2, 4, 8];

export const TlResidualEntry = drill({
  intro: 'Entry $i$ of $y-X\\theta$ is one observation’s residual.',
  gen: (r, n) => ({ t: n === 0 ? [1, 2] : [int(r, -1, 3), int(r, 0, 3)], i: n === 0 ? 2 : int(r, 0, 3) }),
  q: v => `For $x_i=${X1[v.i][1]}$, $y_i=${Y1[v.i]}$ and $\\theta=(${v.t.join(',')})$, find $r_i$.`,
  answer: v => Y1[v.i] - v.t[0] - v.t[1] * X1[v.i][1], dp: 0,
  steps: v => [`Prediction: $${v.t[0]}+${v.t[1]}\\cdot${X1[v.i][1]}=${v.t[0] + v.t[1] * X1[v.i][1]}$`],
});

export const TlLossAt = drill({
  intro: 'A loss with the half in front.',
  tex: R`J(\theta)=\tfrac12\|y-X\theta\|_2^2`,
  gen: (r, n) => ({ t: n === 0 ? [1, 2] : n === 1 ? [0, 2] : [int(r, -1, 3), int(r, 0, 3)] }),
  q: v => `Runs $x=(0,1,2,3)$, $y=(2,2,4,8)$. Evaluate $J$ at $\\theta=(${v.t.join(',')})$.`,
  answer: v => 0.5 * sum(X1.map((x, i) => (Y1[i] - v.t[0] - v.t[1] * x[1]) ** 2)),
  steps: v => [`Residuals: $${X1.map((x, i) => Y1[i] - v.t[0] - v.t[1] * x[1]).join(',\\ ')}$`, 'Square, add, then halve.'],
});

export const TlLossSurface = lab({
  intro: 'The sheet’s four runs. Drag $\\theta$ on the bowl, and watch the residuals change on the right.',
  params: [{ k: 'a', label: '$\\theta_0$', min: -3, max: 5, step: 0.05, init: 0 }, { k: 'b', label: '$\\theta_1$', min: -1, max: 5, step: 0.05, init: 2 }],
  plots: v => { const J = (a, b) => 0.5 * sum(X1.map((x, i) => (Y1[i] - a - b * x[1]) ** 2)); return [{ caption: 'Coefficient space', x: [-3, 5], y: [-1, 5], w: 300, h: 280, xl: 'θ₀', yl: 'θ₁', label: 'loss bowl', marks: [{ as: 'field', bands: 10, tones: [1, 1], fn: (a, b) => Math.sqrt(clamp(J(a, b) / 200, 0, 1)) }, { as: 'points', tone: 3, r: 6, data: [[1, 2]] }, { drag: { x: 'a', y: 'b' } }] }, { caption: 'Observation view', x: [-0.5, 3.5], y: [-2, 10], w: 300, h: 280, xl: 'x', yl: 'y', label: 'line and residuals', marks: [{ as: 'stems', tone: 'muted', data: X1.map((x, i) => [x[1], Y1[i], v.a + v.b * x[1]]) }, { tone: 1, data: [[-0.5, v.a - 0.5 * v.b], [3.5, v.a + 3.5 * v.b]] }, { as: 'points', tone: 'fg', data: X1.map((x, i) => [x[1], Y1[i]]) }] }]; },
  stats: v => [`$J$ = ${fmt(0.5 * sum(X1.map((x, i) => (Y1[i] - v.a - v.b * x[1]) ** 2)), 3)}`, 'the minimum is 2 at (1, 2)'],
});

export const TlGram = drill({
  intro: 'Check entries of $X^\\top X$ and $X^\\top y$ for the sheet’s data.',
  gen: (r, n) => ({ k: n === 0 ? 3 : int(r, 0, 4) }),
  q: v => `For the runs above, find ${['$(X^\\top X)_{11}$', '$(X^\\top X)_{12}$', '$(X^\\top X)_{22}$', '$(X^\\top y)_1$', '$(X^\\top y)_2$'][v.k]}, with $x=(0,1,2,3)$ and $y=(2,2,4,8)$.`,
  answer: v => [4, 6, 14, 16, 34][v.k], dp: 0,
  steps: v => [['Count of the rows.', 'Sum of $x_i$.', 'Sum of $x_i^2$.', 'Sum of $y_i$.', 'Sum of $x_iy_i$.'][v.k]],
});

export const TlSolve = drill({
  intro: 'Solve the normal equation.',
  tex: R`4\theta_0+6\theta_1=16,\qquad 6\theta_0+14\theta_1=34`,
  gen: (r, n) => (n === 0 ? { A: [4, 6, 14], t: [1, 2], k: 1 } : (() => { const a = pick(r, [2, 4, 5]), b = pick(r, [1, 3, 6]), d = pick(r, [10, 14, 20]); return { A: [a, b, d], t: [int(r, -2, 4), int(r, -2, 4)], k: int(r, 0, 1) }; })()),
  q: v => { const [a, b, d] = v.A, c = [a * v.t[0] + b * v.t[1], b * v.t[0] + d * v.t[1]]; return `Solve $${mat([[a, b], [b, d]])}\\theta=${mat([[c[0]], [c[1]]])}$. Find $\\theta_${v.k}$.`; },
  answer: v => v.t[v.k],
  steps: v => [`Determinant: $${v.A[0] * v.A[2] - v.A[1] ** 2}$`, 'Eliminate one unknown between the two equations.'],
});

export const TlOrthogonal = drill({
  intro: 'At the solution, $X^\\top\\hat r=0$. Here $\\hat y=(1,3,5,7)$.',
  gen: (r, n) => ({ k: n === 0 ? 1 : int(r, 0, 1) }),
  q: v => `With $\\hat r=(1,-1,-1,1)$, find entry ${v.k ? 'two' : 'one'} of $X^\\top\\hat r$ for $x=(0,1,2,3)$.`,
  answer: () => 0, dp: 0,
  steps: v => [v.k ? '$0(1)+1(-1)+2(-1)+3(1)$' : '$1-1-1+1$'],
  takeaway: 'Both zeros say the same thing: no remaining linear correction is available.',
});

export const TlPythagoras = drill({
  intro: 'The fit and the residual are perpendicular, so their squared lengths add.',
  tex: R`\|y\|^2=\|\hat y\|^2+\|\hat r\|^2`,
  gen: (r, n) => ({ k: n === 0 ? 0 : int(r, 0, 2) }),
  q: v => `With $y=(2,2,4,8)$, $\\hat y=(1,3,5,7)$ and $\\hat r=(1,-1,-1,1)$, find ${['$\\|y\\|^2$', '$\\|\\hat y\\|^2$', '$\\|\\hat r\\|^2$'][v.k]}.`,
  answer: v => [88, 84, 4][v.k], dp: 0,
  steps: () => ['Square each entry, then add. Check that $84+4=88$.'],
});

export const TlGradient = drill({
  intro: 'The gradient is minus $X^\\top r$.',
  tex: R`\nabla J(\theta)=-X^\top(y-X\theta)`,
  gen: (r, n) => ({ t: n === 0 ? [0, 2] : [int(r, -1, 3), int(r, 0, 3)], k: int(r, 0, 1) }),
  q: v => `For the runs above at $\\theta=(${v.t.join(',')})$, find $\\nabla J$ entry ${v.k ? 'two' : 'one'}.`,
  answer: v => -sum(X1.map((x, i) => x[v.k] * (Y1[i] - v.t[0] - v.t[1] * x[1]))), dp: 0,
  steps: v => [`Residuals: $${X1.map((x, i) => Y1[i] - v.t[0] - v.t[1] * x[1]).join(',\\ ')}$`, `Dot with column $${v.k ? '(0,1,2,3)' : '(1,1,1,1)'}$, and negate.`],
});

export const TlCollinear = lab({
  intro: 'Two features that nearly agree. Move the last entry of the second column towards the first: the fit stays fine, and the coefficients do not.',
  tex: R`f_1=(1,2,3,4),\quad f_2=(1,2,3,s),\quad y=(3,5,5,9)`,
  params: [{ k: 's', label: 'last entry $s$', min: 3, max: 7, step: 0.01, init: 5 }],
  derive: ({ s }) => {
    const solve2 = q => { const f1 = [1, 2, 3, 4], f2 = [1, 2, 3, q], y = [3, 5, 5, 9], a = sum(f1.map(x => x * x)), b = sum(f1.map((x, i) => x * f2[i])), d = sum(f2.map(x => x * x)), c1 = sum(f1.map((x, i) => x * y[i])), c2 = sum(f2.map((x, i) => x * y[i])), det = a * d - b * b; if (Math.abs(det) < 1e-6) return { sing: true }; const t1 = (d * c1 - b * c2) / det, t2 = (a * c2 - b * c1) / det, e = eig2(a, b, d); return { t1, t2, rss: sum(y.map((v, i) => (v - t1 * f1[i] - t2 * f2[i]) ** 2)), cond: Math.sqrt(e.l1 / Math.max(e.l2, 1e-12)) }; };
    return { ...solve2(s), solve2 };
  },
  plots: v => { const side = (a, b, k) => linspace(a, b, 60).map(q => [q, clamp(v.solve2(q)[k], -30, 30)]); return [{ x: [3, 7], y: [-30, 30], xl: 's', yl: 'coefficient', label: 'coefficients as the second column changes', marks: [{ tone: 1, data: side(3, 3.97, 't1') }, { tone: 1, data: side(4.03, 7, 't1') }, { tone: 2, data: side(3, 3.97, 't2') }, { tone: 2, data: side(4.03, 7, 't2') }, { as: 'rule', dash: true, x: 4 }, { as: 'rule', tone: 4, x: v.s }] }]; },
  stats: v => (v.sing ? ['Identical columns: no unique solution.'] : [`θ = (${fmt(v.t1)}, ${fmt(v.t2)})`, `residual sum of squares ${fmt(v.rss, 3)}`, `condition number ${fmt(v.cond, 0)}`]),
  legend: [['--c1', 'first coefficient'], ['--c2', 'second coefficient']],
});

/* ---------- decisions and regularization ---------- */

const ROADS = { tp: 15, fn: 28, fp: 5, tn: 352 };
const roadQ = (name, f, how) => drill({
  intro: 'Road closures: class one means impassable.',
  gen: (r, n) => (n === 0 ? ROADS : { tp: int(r, 5, 40), fn: int(r, 5, 40), fp: int(r, 2, 40), tn: int(r, 100, 360) }),
  q: v => `TP = ${v.tp}, FN = ${v.fn}, FP = ${v.fp}, TN = ${v.tn}. Find the ${name}.`,
  answer: f, dp: 4, steps: how,
});

export const TdAccuracy = roadQ('accuracy', v => (v.tp + v.tn) / (v.tp + v.fn + v.fp + v.tn), v => [`$(${v.tp}+${v.tn})/${v.tp + v.fn + v.fp + v.tn}$`]);
export const TdBaseline = roadQ('accuracy of always predicting passable', v => (v.fp + v.tn) / (v.tp + v.fn + v.fp + v.tn), v => [`All actual negatives are correct: $(${v.fp}+${v.tn})/${v.tp + v.fn + v.fp + v.tn}$`]);
export const TdPrecision = roadQ('precision', v => v.tp / (v.tp + v.fp), v => [`$${v.tp}/(${v.tp}+${v.fp})$`]);
export const TdRecall = roadQ('recall', v => v.tp / (v.tp + v.fn), v => [`$${v.tp}/(${v.tp}+${v.fn})$`]);
export const TdFpr = roadQ('false positive rate', v => v.fp / (v.fp + v.tn), v => [`$${v.fp}/(${v.fp}+${v.tn})$`]);
export const TdF1 = roadQ('F1 score', v => (2 * v.tp) / (2 * v.tp + v.fp + v.fn), v => [`$2(${v.tp})/(2(${v.tp})+${v.fp}+${v.fn})$`]);

export const TdCost = drill({
  intro: 'Total cost adds both kinds of error, each at its own price.',
  tex: R`\text{cost}=C_{FP}\cdot FP+C_{FN}\cdot FN`,
  gen: (r, n) => (n === 0 ? { fp: 5, fn: 28, cf: 1, cn: 5 } : n === 1 ? { fp: 40, fn: 17, cf: 1, cn: 5 } : { fp: int(r, 1, 50), fn: int(r, 1, 40), cf: 1, cn: int(r, 2, 10) }),
  q: v => `FP = ${v.fp}, FN = ${v.fn}, with $C_{FP}=${v.cf}$ and $C_{FN}=${v.cn}$. Find the total cost.`,
  answer: v => v.cf * v.fp + v.cn * v.fn, dp: 0,
  steps: v => [`$${v.cf}\\cdot${v.fp}+${v.cn}\\cdot${v.fn}$`],
});

export const TdThresholdChoice = lab({
  intro: 'The sheet’s two thresholds. Change the miss cost and see which one wins.',
  params: [{ k: 'cn', label: 'cost of a miss', min: 1, max: 15, step: 0.5, init: 5 }],
  plots: v => [{ x: [1, 15], y: [0, 450], xl: 'cost of a miss', yl: 'total cost', label: 'total cost for each threshold', marks: [{ tone: 1, data: [[1, 5 + 28], [15, 5 + 28 * 15]] }, { tone: 2, data: [[1, 40 + 17], [15, 40 + 17 * 15]] }, { as: 'rule', dash: true, x: v.cn }] }],
  stats: v => [`at 0.5: ${fmt(5 + 28 * v.cn, 1)}`, `at 1/6: ${fmt(40 + 17 * v.cn, 1)}`],
  status: v => (5 + 28 * v.cn > 40 + 17 * v.cn ? ['ok', `The lower threshold saves ${fmt(5 + 28 * v.cn - 40 - 17 * v.cn, 1)} cost units.`] : [undefined, 'The higher threshold is cheaper here.']),
  legend: [['--c1', 'threshold 0.5'], ['--c2', 'threshold 1/6']],
  takeaway: 'The two lines cross at a miss cost of $35/11$. Below that, the lower-accuracy rule stops being worth it.',
});

export const TdRidgeSensors = lab({
  intro: 'Two rainfall sensors, nearly identical. Ordinary least squares can give them wild, opposite weights. Ridge splits the credit.',
  tex: R`(X^\top X+\alpha P)\beta=X^\top y`,
  params: [{ k: 'lg', label: '$\\log_{10}\\alpha$', min: -4, max: 3, step: 0.05, init: -4 }],
  derive: ({ lg }) => { const fit = L => { const a = 10 ** L, n = 300, rho = 0.9995, c = 270, A = [[n + a, n * rho], [n * rho, n + a]], b = [c + 1.5, c - 1.5], det = A[0][0] * A[1][1] - A[0][1] ** 2; return [(A[1][1] * b[0] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - A[0][1] * b[0]) / det]; }; const [b1, b2] = fit(lg); return { fit, b1, b2 }; },
  plots: v => [{ x: [-4, 3], y: [-12, 12], xl: 'log₁₀ α', yl: 'coefficient', label: 'sensor coefficients against alpha', marks: [{ tone: 1, data: curve(L => clamp(v.fit(L)[0], -12, 12), -4, 3) }, { tone: 2, data: curve(L => clamp(v.fit(L)[1], -12, 12), -4, 3) }, { as: 'rule', dash: true, x: v.lg }] }],
  stats: v => [`rain A ${fmt(v.b1, 3)}, rain B ${fmt(v.b2, 3)}`, `their sum ${fmt(v.b1 + v.b2, 3)}`],
  note: 'A stylised version of the laptop exercise, on standardized sensors with correlation 0.9995.',
  takeaway: 'The sum is steady even when the split is not. That is the difference between prediction and interpretation.',
});

export const TdPenaltyMatrix = drill({
  intro: 'The intercept stays unpenalized, so its diagonal entry is zero.',
  gen: r => ({ a: pick(r, [1, 10, 100]), xx: int(r, 3, 20) }),
  q: v => `$P=\\operatorname{diag}(0,1,1)$ and $\\alpha=${v.a}$. If $(X^\\top X)_{22}=${v.xx}$, what is entry $(2,2)$ of $X^\\top X+\\alpha P$?`,
  answer: v => v.xx + v.a, dp: 0,
  steps: v => [`$P_{22}=1$, so add $\\alpha$: $${v.xx}+${v.a}$`],
});

/* ---------- SVM tutorial ---------- */

const minmax = T => Math.min(...T.map(row => Math.max(...row)));
const maxmin = T => { const c = T[0].map((_, j) => Math.min(...T.map(row => row[j]))); return Math.max(...c); };
const gT = r => Array.from({ length: 2 }, () => [int(r, 10, 90), int(r, 10, 90)]);

export const TsMinMax = drill({
  intro: 'The parent chooses first and expects the kid to answer with the dearest toy.',
  tex: R`\min_s\max_t\,p(s,t)`,
  gen: (r, n) => ({ T: n === 0 ? [[50, 80], [70, 60]] : gT(r) }),
  q: v => `Prices: $${mat(v.T)}$. Compute min-max.`,
  answer: v => minmax(v.T), dp: 0,
  steps: v => [`Largest in each row: $${v.T.map(row => Math.max(...row)).join(',\\ ')}$`, 'Take the smaller.'],
});

export const TsMaxMin = drill({
  intro: 'Now the kid moves first, and the parent answers with the cheapest shop.',
  tex: R`\max_t\min_s\,p(s,t)`,
  gen: (r, n) => ({ T: n === 0 ? [[50, 80], [70, 60]] : gT(r) }),
  q: v => `Prices: $${mat(v.T)}$. Compute max-min.`,
  answer: v => maxmin(v.T), dp: 0,
  steps: v => [`Smallest in each column: $${v.T[0].map((_, j) => Math.min(...v.T.map(row => row[j]))).join(',\\ ')}$`, 'Take the larger.'],
});

export const TsSaddle = lab({
  intro: 'Change the prices and compare who moves first. Max-min can never exceed min-max, and equality means a saddle point exists.',
  params: [{ k: 'a', label: 'shop A, toy one', min: 10, max: 90, step: 5, init: 50 }, { k: 'b', label: 'shop A, toy two', min: 10, max: 90, step: 5, init: 80 }, { k: 'c', label: 'shop B, toy one', min: 10, max: 90, step: 5, init: 70 }, { k: 'd', label: 'shop B, toy two', min: 10, max: 90, step: 5, init: 60 }],
  plots: v => { const T = [[v.a, v.b], [v.c, v.d]], mm = minmax(T), xm = maxmin(T); return [{ x: [-0.5, 1.5], y: [0, 100], yl: 'price', label: 'weak duality bars', marks: [{ as: 'rects', tone: 2, data: [[-0.3, 0, 0.3, xm]] }, { as: 'rects', tone: 1, data: [[0.7, 0, 1.3, mm]] }, { as: 'label', x: 0, y: 95, text: 'max-min', anchor: 'middle' }, { as: 'label', x: 1, y: 95, text: 'min-max', anchor: 'middle' }] }]; },
  stats: v => { const T = [[v.a, v.b], [v.c, v.d]]; return [`max-min ${maxmin(T)}, min-max ${minmax(T)}`]; },
  status: v => { const T = [[v.a, v.b], [v.c, v.d]]; return minmax(T) === maxmin(T) ? ['ok', 'Equal: this table has a saddle point. Strong duality holds.'] : [undefined, 'A gap: moving second is an advantage.']; },
});

const PTS = [[3, 2, 1], [2, 2, 1], [2, 0, 1], [0, 1, -1]];

export const TsSlack = drill({
  intro: 'The kid raises $\\alpha_i$ when $g_i<0$, and sets it to zero when $g_i>0$.',
  tex: R`g_i=y_i(w\cdot x_i+b)-1`,
  gen: (r, n) => (n === 0 ? { w: [1, 1], b: -3, p: PTS[0] } : { w: [int(r, -1, 2), int(r, -1, 2)], b: int(r, -3, 3), p: pick(r, PTS) }),
  q: v => `For $w=(${v.w.join(',')})$, $b=${v.b}$, $x=(${v.p[0]},${v.p[1]})$ and $y=${v.p[2]}$, find $g$.`,
  answer: v => v.p[2] * (v.w[0] * v.p[0] + v.w[1] * v.p[1] + v.b) - 1, dp: 0,
  steps: v => [`$w\\cdot x+b=${v.w[0] * v.p[0] + v.w[1] * v.p[1] + v.b}$`, `Times $y$, minus one.`],
});

export const TsTwoPoint = drill({
  intro: 'Two points $x$ and $-x$ with opposite labels. The dual has one unknown.',
  tex: R`D(\alpha)=2\alpha-\tfrac12\alpha^2\|x_1-x_2\|^2`,
  gen: (r, n) => (n === 0 ? { a: 1, b: 1 } : { a: int(r, 1, 3), b: int(r, 0, 3) }),
  q: v => `Points $(${v.a},${v.b})$ with $y=+1$, and $(${-v.a},${-v.b})$ with $y=-1$. Find the optimal $\\alpha$.`,
  answer: v => 2 / ((2 * v.a) ** 2 + (2 * v.b) ** 2), dp: 3,
  steps: v => [`$\\|x_1-x_2\\|^2=${(2 * v.a) ** 2 + (2 * v.b) ** 2}$`, `Setting $D'(\\alpha)=0$ gives $\\alpha=2/\\|x_1-x_2\\|^2$.`],
});

export const TsGram = drill({
  intro: 'Fill in one Gram matrix entry.',
  gen: (r, n) => ({ i: n === 0 ? 1 : int(r, 0, 2), j: n === 0 ? 2 : int(r, 0, 2), k: n === 0 ? 'sq' : pick(r, ['lin', 'sq']) }),
  q: v => `Points $x_1=(1,0)$, $x_2=(0,1)$, $x_3=(1,1)$. With $K(x,z)=${v.k === 'lin' ? 'x\\cdot z' : '(x\\cdot z)^2'}$, find $K_{${v.i + 1}${v.j + 1}}$.`,
  answer: v => { const P = [[1, 0], [0, 1], [1, 1]], d = P[v.i][0] * P[v.j][0] + P[v.i][1] * P[v.j][1]; return v.k === 'lin' ? d : d * d; }, dp: 0,
  steps: v => { const P = [[1, 0], [0, 1], [1, 1]], d = P[v.i][0] * P[v.j][0] + P[v.i][1] * P[v.j][1]; return [`$x\\cdot z=${d}$`]; },
});

export const TsTrick = drill({
  intro: 'Compute it directly, then check against the hidden features.',
  tex: R`\phi(x)=(x_1^2,\ \sqrt2x_1x_2,\ x_2^2),\qquad K=(x\cdot z)^2`,
  gen: (r, n) => (n === 0 ? { x: [2, 1], z: [1, 3] } : { x: [int(r, -2, 3), int(r, -2, 3)], z: [int(r, -2, 3), int(r, -2, 3)] }),
  q: v => `For $x=(${v.x.join(',')})$ and $z=(${v.z.join(',')})$, find $K(x,z)$.`,
  answer: v => (v.x[0] * v.z[0] + v.x[1] * v.z[1]) ** 2, dp: 0,
  steps: v => [`$x\\cdot z=${v.x[0] * v.z[0] + v.x[1] * v.z[1]}$`, `The features give the same: $${v.x[0] ** 2 * v.z[0] ** 2}+${2 * v.x[0] * v.x[1] * v.z[0] * v.z[1]}+${v.x[1] ** 2 * v.z[1] ** 2}$`],
});

export const TsXor = lab({
  intro: 'XOR with the kernel $(x\\cdot z)^2$. The solution works out to $f(x)=x_1x_2$. Drag a query point and read its class.',
  tex: R`f(x)=\sum_i\alpha_iy_iK(x_i,x)+b=x_1x_2`,
  params: [{ k: 'a', label: '$x_1$', min: -3, max: 3, step: 0.05, init: 2 }, { k: 'b', label: '$x_2$', min: -3, max: 3, step: 0.05, init: 3 }],
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'XOR decision function', marks: [{ as: 'field', fn: (x, y) => clamp((x * y) / 4, -1, 1) }, { as: 'points', r: 7, data: [[1, 1, 1], [-1, -1, 1], [1, -1, 2], [-1, 1, 2]] }, { drag: { x: 'a', y: 'b', tone: 3 } }] }],
  stats: v => [`f = ${fmt(v.a * v.b, 2)}, so class ${v.a * v.b > 0 ? '+1' : v.a * v.b < 0 ? '−1' : 'on the boundary'}`],
});

export const TsXorGram = drill({
  intro: 'One entry of the XOR Gram matrix.',
  gen: (r, n) => ({ i: n === 0 ? 0 : int(r, 0, 3), j: n === 0 ? 1 : int(r, 0, 3) }),
  q: v => `Points $(1,1)$, $(-1,-1)$, $(1,-1)$, $(-1,1)$, with $K=(x\\cdot z)^2$. Find $K_{${v.i + 1}${v.j + 1}}$.`,
  answer: v => { const P = [[1, 1], [-1, -1], [1, -1], [-1, 1]]; return (P[v.i][0] * P[v.j][0] + P[v.i][1] * P[v.j][1]) ** 2; }, dp: 0,
  steps: v => { const P = [[1, 1], [-1, -1], [1, -1], [-1, 1]]; return [`$x\\cdot z=${P[v.i][0] * P[v.j][0] + P[v.i][1] * P[v.j][1]}$`]; },
});

export const TsRbf = drill({
  intro: 'Use $e^{-1}\\approx0.368$ when you can.',
  tex: R`K(x,z)=e^{-\gamma\|x-z\|^2}`,
  gen: (r, n) => ({ g: n === 0 ? 1 : pick(r, [0.25, 0.5, 1, 2]), x: [0, 0], z: n === 0 ? [1, 0] : [int(r, -2, 2), int(r, -2, 2)] }),
  q: v => `Find the similarity between $(0,0)$ and $(${v.z.join(',')})$, with $\\gamma=${v.g}$.`,
  answer: v => Math.exp(-v.g * (v.z[0] ** 2 + v.z[1] ** 2)), dp: 3,
  steps: v => [`$\\|x-z\\|^2=${v.z[0] ** 2 + v.z[1] ** 2}$`, `$e^{-${fmt(v.g * (v.z[0] ** 2 + v.z[1] ** 2), 2)}}$`],
});

/* ---------- boosting tutorial ---------- */

const GX = [1, 2, 3, 4, 5, 6], GY = [1, 2, 3, 8, 10, 12];

export const TbFirstGuess = drill({
  intro: 'The first guess minimises squared error, so it is the average.',
  gen: (r, n) => ({ y: n === 0 ? GY : Array.from({ length: 6 }, () => int(r, 0, 12)) }),
  q: v => `For $y=(${v.y.join(',')})$, find $F_0$.`,
  answer: v => mean(v.y),
  steps: v => [`Sum: ${sum(v.y)}. Count: 6.`],
});

const stumpSSE = (y, t) => { const L = y.filter((_, i) => GX[i] <= t), Rr = y.filter((_, i) => GX[i] > t), s = a => { const m = mean(a); return sum(a.map(v => (v - m) ** 2)); }; return s(L) + s(Rr); };

export const TbBestSplit = lab({
  intro: 'A good split makes each leaf’s leftovers similar. Slide the cut, and compare the leftover error.',
  params: [{ k: 't', label: 'split at $x\\le t$', min: 1.5, max: 5.5, step: 1, init: 2.5 }],
  derive: () => { const F0 = mean(GY); return { r: GY.map(v => v - F0) }; },
  plots: v => [{ x: [0.5, 6.5], y: [-7, 7], xl: 'x', yl: 'leftover', label: 'leftovers with a stump split', marks: [{ as: 'stems', tone: 2, data: GX.map((x, i) => [x, 0, v.r[i]]) }, { as: 'rule', tone: 'fg', x: v.t }, { tone: 1, data: [[0.5, mean(v.r.filter((_, i) => GX[i] <= v.t))], [v.t, mean(v.r.filter((_, i) => GX[i] <= v.t))]] }, { tone: 1, data: [[v.t, mean(v.r.filter((_, i) => GX[i] > v.t))], [6.5, mean(v.r.filter((_, i) => GX[i] > v.t))]] }, { drag: { x: 't', y: 6 } }] }],
  stats: v => [`leftover error ${fmt(stumpSSE(v.r, v.t), 2)}`, `leaf values ${fmt(mean(v.r.filter((_, i) => GX[i] <= v.t)), 2)} and ${fmt(mean(v.r.filter((_, i) => GX[i] > v.t)), 2)}`],
  status: v => (v.t === 3.5 ? ['ok', 'The best split: lowest leftover error.'] : undefined),
});

export const TbRounds = lab({
  intro: 'Step through the rounds on the sheet’s data, one stump at a time.',
  params: [{ k: 'm', label: 'round', min: 0, max: 12, step: 1, init: 0, play: true }, { k: 'nu', label: 'step size $\\nu$', min: 0.1, max: 1, step: 0.05, init: 0.5 }],
  derive: ({ m, nu }) => { let F = GY.map(() => mean(GY)); const hist = [sum(GY.map((y, i) => (y - F[i]) ** 2))]; for (let k = 0; k < m; k++) { const r = GY.map((y, i) => y - F[i]); let best = 1.5; [1.5, 2.5, 3.5, 4.5, 5.5].forEach(t => { if (stumpSSE(r, t) < stumpSSE(r, best) - 1e-9) best = t; }); const lv = mean(r.filter((_, i) => GX[i] <= best)), rv = mean(r.filter((_, i) => GX[i] > best)); F = F.map((f, i) => f + nu * (GX[i] <= best ? lv : rv)); hist.push(sum(GY.map((y, i) => (y - F[i]) ** 2))); } return { F, hist }; },
  plots: v => [{ caption: 'Current guess', x: [0.5, 6.5], y: [0, 14], w: 300, h: 240, xl: 'x', label: 'boosted fit', marks: [{ as: 'points', tone: 'fg', data: GX.map((x, i) => [x, GY[i]]) }, { as: 'points', tone: 1, r: 6, data: GX.map((x, i) => [x, v.F[i]]) }, { as: 'stems', tone: 'muted', data: GX.map((x, i) => [x, GY[i], v.F[i]]) }] }, { caption: 'Total leftover', x: [0, 12], y: [0, 110], w: 300, h: 240, xl: 'round', label: 'error per round', marks: [{ tone: 4, data: v.hist.map((e, i) => [i, e]) }, { as: 'points', tone: 4, r: 4, data: v.hist.map((e, i) => [i, e]) }] }],
  stats: v => [`total leftover ${fmt(v.hist[v.hist.length - 1], 2)}`],
});

export const TbUpdate = drill({
  intro: 'Add only a part of the leaf value.',
  tex: R`F_{\text{new}}=F_{\text{old}}+\nu\cdot\text{leaf}`,
  gen: (r, n) => (n === 0 ? { F: 6, nu: 0.5, l: -4 } : { F: int(r, 2, 10), nu: pick(r, [0.1, 0.25, 0.5, 1]), l: int(r, -6, 6) }),
  q: v => `$F_{\\text{old}}=${v.F}$, $\\nu=${v.nu}$ and the leaf value is $${v.l}$. Find $F_{\\text{new}}$.`,
  answer: v => v.F + v.nu * v.l,
  steps: v => [`$${v.nu}\\times(${v.l})=${fmt(v.nu * v.l)}$`],
});

export const TbLogOdds = drill({
  intro: 'For classification, the first score is the log-odds of the base rate.',
  tex: R`F_0=\ln\frac{\#\text{ones}}{\#\text{zeros}}`,
  gen: (r, n) => (n === 0 ? { a: 3, b: 3 } : { a: int(r, 1, 9), b: int(r, 1, 9) }),
  q: v => `With ${v.a} ones and ${v.b} zeros, find $F_0$.`,
  answer: v => Math.log(v.a / v.b), dp: 3,
  steps: v => [`$\\ln(${v.a}/${v.b})$`],
});

export const TbNewtonLeaf = drill({
  intro: 'Each leaf takes one Newton step.',
  tex: R`\text{leaf}=\frac{\sum r}{\sum p(1-p)}`,
  gen: (r, n) => (n === 0 ? { rs: [-0.5, -0.5, 0.5], p: 0.5 } : { rs: Array.from({ length: 3 }, () => pick(r, [-0.5, 0.5])), p: 0.5 }),
  q: v => `A leaf has leftovers $${v.rs.join(',\\ ')}$, and every point has $p=0.5$. Find the leaf value.`,
  answer: v => sum(v.rs) / (v.rs.length * 0.25), dp: 3,
  steps: v => [`$\\sum r=${sum(v.rs)}$`, `$\\sum p(1-p)=${v.rs.length}\\times0.25$`],
});

export const TbAdaAlpha = drill({
  intro: 'A stump’s vote comes from its weighted error.',
  tex: R`\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}`,
  gen: r => ({ e: pick(r, [0.1, 0.2, 0.25, 0.3, 0.4]) }),
  q: v => `A stump has $\\varepsilon=${v.e}$. Find $\\alpha$.`,
  answer: v => 0.5 * Math.log((1 - v.e) / v.e), dp: 3,
  steps: v => [`$(1-\\varepsilon)/\\varepsilon=${fmt((1 - v.e) / v.e, 3)}$`, 'Take the natural log, then halve.'],
});

export const TbXgbLeaf = drill({
  intro: 'The lowest point of $Q(w)=Gw+\\tfrac12(H+\\lambda)w^2$.',
  tex: R`w=-\frac{G}{H+\lambda}`,
  gen: (r, n) => (n === 0 ? { G: 6, H: 2, l: 1 } : { G: int(r, -8, 8), H: int(r, 1, 6), l: pick(r, [0, 1, 4]) }),
  q: v => `$G=${v.G}$, $H=${v.H}$ and $\\lambda=${v.l}$. Find the leaf weight.`,
  answer: v => -v.G / (v.H + v.l), dp: 3,
  steps: v => [`$H+\\lambda=${v.H + v.l}$`],
});

export const TbXgbGain = drill({
  intro: 'The score of each side, minus the score of the parent, halved, minus the price of a split.',
  tex: R`\text{gain}=\tfrac12\big[\text{score}_L+\text{score}_R-\text{score}_{\text{all}}\big]-\gamma`,
  gen: (r, n) => (n === 0 ? { GL: 6, HL: 2, GR: -6, HR: 2, l: 1, g: 1 } : { GL: int(r, -6, 6), HL: int(r, 1, 4), GR: int(r, -6, 6), HR: int(r, 1, 4), l: 1, g: pick(r, [0, 1]) }),
  q: v => `$G_L=${v.GL}$, $H_L=${v.HL}$, $G_R=${v.GR}$, $H_R=${v.HR}$, $\\lambda=${v.l}$ and $\\gamma=${v.g}$. Find the gain.`,
  answer: v => { const s = (g, h) => (g * g) / (h + v.l); return 0.5 * (s(v.GL, v.HL) + s(v.GR, v.HR) - s(v.GL + v.GR, v.HL + v.HR)) - v.g; },
  steps: v => { const s = (g, h) => (g * g) / (h + v.l); return [`Left score: ${fmt(s(v.GL, v.HL), 3)}`, `Right score: ${fmt(s(v.GR, v.HR), 3)}`, `Parent score: ${fmt(s(v.GL + v.GR, v.HL + v.HR), 3)}`]; },
});

const XY = [3, 5, 9, 11];

export const TbXgbSplits = lab({
  intro: 'The self-paced example. Compare all the splits, and vary the two settings.',
  params: [{ k: 'l', label: '$\\lambda$', min: 0, max: 10, step: 0.5, init: 1 }, { k: 'g', label: '$\\gamma$', min: 0, max: 15, step: 0.5, init: 1 }],
  derive: ({ l, g }) => { const F = mean(XY), gr = XY.map(y => F - y), s = (G, H) => (G * G) / (H + l); const rows = [1, 2, 3].map(t => { const GL = sum(gr.slice(0, t)), GR = sum(gr.slice(t)); return { t, gain: 0.5 * (s(GL, t) + s(GR, 4 - t) - s(GL + GR, 4)) - g, wL: -GL / (t + l), wR: -GR / (4 - t + l) }; }); return { rows, best: rows.reduce((a, b) => (b.gain > a.gain ? b : a)) }; },
  plots: v => [{ x: [0.5, 3.5], y: [-16, 20], xl: 'split x ≤ t', yl: 'gain', label: 'gain for each split', marks: [{ as: 'rects', tone: 1, data: v.rows.map(r => [r.t - 0.3, 0, r.t + 0.3, clamp(r.gain, -16, 20)]) }, { as: 'rule', tone: 4, y: 0 }] }],
  stats: v => [`best split x ≤ ${v.best.t}, gain ${fmt(v.best.gain, 2)}`, `leaf weights ${fmt(v.best.wL, 2)} and ${fmt(v.best.wR, 2)}`],
  status: v => (v.best.gain <= 0 ? ['alert', 'No split pays for $\\gamma$, so the tree stays a single leaf.'] : undefined),
});

export const TbLeftoverSign = drill({
  intro: 'Read a leftover.',
  gen: r => ({ y: int(r, 0, 12), F: int(r, 0, 12) }),
  q: v => `$y=${v.y}$ and the guess is $F=${v.F}$. The guess is:`,
  choices: () => ['Too high', 'Too low', 'Exactly right'],
  answer: v => (v.F > v.y ? 0 : v.F < v.y ? 1 : 2),
  steps: v => [`Leftover: $y-F=${v.y - v.F}$`, 'Positive means the guess must rise.'],
});

export const TlFitMeaning = drill({
  intro: 'The fitted model on the sheet is $\\hat y=1+2x$. Read what it says.',
  gen: r => ({ k: pick(r, [0, 1, 2]), x: int(r, 4, 9) }),
  q: v => [`What latency does it predict with ${v.x} extra layers, in ms?`, 'What is the fixed overhead, in ms?', 'How many ms does each extra layer add?'][v.k],
  answer: v => [1 + 2 * v.x, 1, 2][v.k], dp: 0,
  steps: v => [['Substitute into $1+2x$.', 'The intercept is the prediction at $x=0$.', 'The slope is the change per unit of $x$.'][v.k]],
});
