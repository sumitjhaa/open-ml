'use client';
// New explorers for the introduction, mathematical foundations and estimation sections.
import { R, curve, drill, int, lab, mat, pick, truths } from '../components/labs';
import { gauss, mean, rng, sum } from '../lib/math.mjs';
import { clamp, fmt, linspace, normalPdf, pct } from '../lib/models.mjs';

const circle = (cx, cy, r) => linspace(0, 2 * Math.PI, 91).map(t => [cx + r * Math.cos(t), cy + r * Math.sin(t)]);

/* ---------- introduction ---------- */

export const SetsCount = lab({
  intro: 'Before probability there is counting. Change how much the two groups overlap and watch every derived count follow.',
  tex: R`|A\cup B|=|A|+|B|-|A\cap B|,\qquad P(A\mid B)=\frac{|A\cap B|}{|B|}`,
  params: [{ k: 'a', label: '$|A|$', min: 5, max: 60, step: 1, init: 40 }, { k: 'b', label: '$|B|$', min: 5, max: 60, step: 1, init: 30 }, { k: 'o', label: 'overlap', min: 0, max: 60, step: 1, init: 12 }],
  derive: ({ a, b, o }) => { const i = Math.min(o, a, b), ra = Math.sqrt(a) / 3, rb = Math.sqrt(b) / 3; return { i, ra, rb, gap: (ra + rb) * (1 - i / Math.min(a, b)) + 0.05 }; },
  plots: v => [{ x: [-6, 6], y: [-4, 4], w: 400, h: 270, label: 'two overlapping sets', marks: [
    { tone: 1, data: circle(-v.gap / 2, 0, v.ra) }, { tone: 2, data: circle(v.gap / 2, 0, v.rb) },
    { as: 'label', x: -v.gap / 2 - v.ra, y: v.ra + 0.4, text: `A only ${v.a - v.i}` }, { as: 'label', x: v.gap / 2 + 0.2, y: v.rb + 0.4, text: `B only ${v.b - v.i}` }, { as: 'label', x: 0, y: 0, text: String(v.i), anchor: 'middle' },
  ] }],
  stats: v => [`union ${v.a + v.b - v.i}`, `of 100 people, outside both: ${Math.max(0, 100 - (v.a + v.b - v.i))}`, `$P(A\\mid B)$ = ${fmt(v.i / v.b, 3)}, $P(B\\mid A)$ = ${fmt(v.i / v.a, 3)}`],
  takeaway: 'Conditioning is just counting inside a smaller group, which is why the two conditional probabilities differ.',
});

export const FunctionMachine = lab({
  intro: 'A function takes an input and returns one output. Its **parameters** are the knobs that decide which function it is. Drag the input.',
  params: [{ k: 'kind', label: 'Function family', options: [['line', 'line'], ['parabola', 'parabola'], ['step', 'step']], init: 'line' }, { k: 'a', label: 'knob $a$', min: -2, max: 2, step: 0.1, init: 1 }, { k: 'b', label: 'knob $b$', min: -3, max: 3, step: 0.1, init: 0.5 }, { k: 'x', label: 'input $x$', min: -4, max: 4, step: 0.1, init: 1.5 }],
  derive: ({ kind, a, b }) => ({ f: kind === 'line' ? x => a * x + b : kind === 'parabola' ? x => a * x * x + b : x => (x > b ? a : 0) }),
  plots: v => [{ x: [-4, 4], y: [-6, 6], xl: 'input', yl: 'output', label: 'function graph', marks: [{ tone: 1, data: curve(v.f, -4, 4, 321) }, { as: 'rule', dash: true, x: v.x }, { as: 'points', tone: 4, r: 6, data: [[v.x, clamp(v.f(v.x), -6, 6)]] }, { drag: { x: 'x', y: -5.5, tone: 3 } }] }],
  stats: v => [`f(${fmt(v.x, 1)}) = ${fmt(v.f(v.x))}`],
  takeaway: 'A model is exactly this: a family of functions, with training choosing the knobs.',
});

export const SummationDrill = drill({
  intro: 'The sigma sign is a loop written in one symbol. Work out each term, then add.',
  gen: r => { const n = int(r, 3, 5); return { xs: Array.from({ length: n }, () => int(r, -3, 6)), kind: pick(r, ['sum', 'square', 'scaled']), c: int(r, 2, 4) }; },
  q: v => `For $x=(${v.xs.join(',\\ ')})$ compute $${v.kind === 'sum' ? R`\sum_i x_i` : v.kind === 'square' ? R`\sum_i x_i^2` : R`\sum_i ${v.c}\,x_i`}$.`,
  answer: v => sum(v.xs.map(x => (v.kind === 'sum' ? x : v.kind === 'square' ? x * x : v.c * x))),
  dp: 0,
  steps: v => [`Terms: $${v.xs.map(x => (v.kind === 'sum' ? `(${x})` : v.kind === 'square' ? `(${x})^2` : `${v.c}(${x})`)).join('+')}$`, `Values: $${v.xs.map(x => (v.kind === 'sum' ? x : v.kind === 'square' ? x * x : v.c * x)).join(',\\ ')}$`],
});

export const WeightedAverage = lab({
  intro: 'An average of averages must be weighted by group size. Drag the mix and the overall rate slides between the two group rates, never outside them.',
  tex: R`\text{overall}=(1-w)\,r_1+w\,r_2`,
  params: [{ k: 'r1', label: 'group one rate', min: 0, max: 100, step: 1, init: 90 }, { k: 'r2', label: 'group two rate', min: 0, max: 100, step: 1, init: 40 }, { k: 'w', label: 'share in group two', min: 0, max: 1, step: 0.01, init: 0.25 }],
  plots: v => [{ x: [0, 1], y: [0, 100], xl: 'share in group two', yl: 'overall rate', label: 'weighted average as a line', marks: [{ tone: 1, data: [[0, v.r1], [1, v.r2]] }, { as: 'rule', dash: true, x: v.w }, { drag: { x: 'w', y: (1 - v.w) * v.r1 + v.w * v.r2 }, r: 7 }] }],
  stats: v => [`overall ${fmt((1 - v.w) * v.r1 + v.w * v.r2, 1)}`],
  takeaway: "A shift in mix moves the overall number even when no group changes. That is the engine of Simpson's paradox.",
});

export const VectorArrows = lab({
  intro: 'A vector is both a list of numbers and an arrow. Drag either tip: addition is tip-to-tail and scaling stretches along the same line.',
  params: [{ k: 'ux', label: '$u_1$', min: -4, max: 4, step: 0.1, init: 2 }, { k: 'uy', label: '$u_2$', min: -4, max: 4, step: 0.1, init: 1 }, { k: 'vx', label: '$v_1$', min: -4, max: 4, step: 0.1, init: -1 }, { k: 'vy', label: '$v_2$', min: -4, max: 4, step: 0.1, init: 2 }, { k: 'c', label: 'scale of $u$', min: -2, max: 2, step: 0.1, init: 1 }],
  plots: v => { const ux = v.c * v.ux, uy = v.c * v.uy; return [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, xl: 'first entry', yl: 'second entry', label: 'vector addition', marks: [{ as: 'arrow', tone: 1, data: [[0, 0, ux, uy]] }, { as: 'arrow', tone: 2, data: [[ux, uy, ux + v.vx, uy + v.vy]] }, { as: 'arrow', tone: 3, data: [[0, 0, ux + v.vx, uy + v.vy]] }, { drag: { x: 'ux', y: 'uy', tone: 1 } }, { drag: { x: 'vx', y: 'vy', tone: 2 } }] }]; },
  stats: v => [`sum (${fmt(v.c * v.ux + v.vx, 1)}, ${fmt(v.c * v.uy + v.vy, 1)})`, `length of $u$: ${fmt(Math.hypot(v.ux, v.uy))}`],
  legend: [['--c1', 'scaled u'], ['--c2', 'v'], ['--c3', 'sum']],
});

export const ExpLog = lab({
  intro: 'A logarithm undoes an exponential, so their graphs are mirror images across the dashed line. Drag the point on the exponential.',
  tex: R`y=b^{x}\iff x=\log_b y`,
  params: [{ k: 'b', label: 'Base', options: [['2', 'base 2'], ['e', 'base e'], ['10', 'base 10']], init: 'e' }, { k: 'x', label: 'exponent $x$', min: -3, max: 2, step: 0.05, init: 1 }],
  derive: ({ b }) => { const B = b === 'e' ? Math.E : +b; return { B, ex: x => B ** x, lg: y => Math.log(y) / Math.log(B) }; },
  plots: v => [{ x: [-4, 8], y: [-4, 8], w: 380, h: 380, xl: 'x', yl: 'y', label: 'exponential and logarithm', marks: [{ tone: 'muted', dash: true, data: [[-4, -4], [8, 8]] }, { tone: 1, data: curve(v.ex, -4, 2.2) }, { tone: 2, data: curve(y => clamp(v.lg(y), -4, 8), 0.01, 8, 301) }, { as: 'points', tone: 2, r: 6, data: [[Math.min(v.ex(v.x), 8), v.x]] }, { drag: { x: 'x', y: Math.min(v.ex(v.x), 8), tone: 1 } }] }],
  stats: v => [`$b^x$ = ${fmt(v.ex(v.x), 3)}`, `log of that = ${fmt(v.x)}`],
  takeaway: 'Exponentials turn sums into products, and logs turn products into sums. That is why likelihoods are always handled in log form.',
});

export const IntroCheck = truths({
  intro: 'Decide before you read the reason.',
  items: [
    ['A model with 99% accuracy must be useful.', false, 'If 99% of cases are negative, predicting negative for everyone scores 99% and finds nothing.'],
    ['$\\hat R$ measures error on data the model has seen.', true, 'The hat marks an estimate from the sample. The true risk $R$ is over the whole distribution.'],
    ['$P(A\\mid B)$ and $P(B\\mid A)$ are always equal.', false, 'They share a numerator but divide by different totals.'],
    ['Adding parameters can never raise training error.', true, 'A larger model family contains the smaller one, so its best training fit is at least as good.'],
    ['A feature perfect in training is safe to deploy.', false, 'It may be a shortcut that only held in the training world.'],
  ],
});

/* ---------- mathematical foundations ---------- */

export const DotProduct = lab({
  intro: 'The dot product says how much two arrows point the same way. Drag the tips and watch its sign flip exactly at a right angle.',
  tex: R`u\cdot v=\sum_i u_iv_i=\|u\|\,\|v\|\cos\theta`,
  params: [{ k: 'ux', label: '$u_1$', min: -4, max: 4, step: 0.1, init: 3 }, { k: 'uy', label: '$u_2$', min: -4, max: 4, step: 0.1, init: 1 }, { k: 'vx', label: '$v_1$', min: -4, max: 4, step: 0.1, init: 1 }, { k: 'vy', label: '$v_2$', min: -4, max: 4, step: 0.1, init: 2.5 }],
  derive: ({ ux, uy, vx, vy }) => { const d = ux * vx + uy * vy, n = ux * ux + uy * uy || 1; return { d, sh: [(d / n) * ux, (d / n) * uy], ang: (Math.acos(clamp(d / ((Math.hypot(ux, uy) || 1) * (Math.hypot(vx, vy) || 1)), -1, 1)) * 180) / Math.PI }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, xl: 'x₁', yl: 'x₂', label: 'two vectors and a projection', marks: [{ as: 'arrow', tone: 1, data: [[0, 0, v.ux, v.uy]] }, { as: 'arrow', tone: 2, data: [[0, 0, v.vx, v.vy]] }, { as: 'arrow', tone: 3, data: [[0, 0, ...v.sh]] }, { tone: 'muted', dash: true, data: [v.sh, [v.vx, v.vy]] }, { drag: { x: 'ux', y: 'uy', tone: 1 } }, { drag: { x: 'vx', y: 'vy', tone: 2 } }] }],
  stats: v => [`$u\\cdot v$ = ${fmt(v.d)}`, `angle ${fmt(v.ang, 0)} degrees`],
  status: v => (Math.abs(v.d) < 0.15 ? ['ok', 'Perpendicular: the dot product is zero.'] : v.d > 0 ? [undefined, 'Pointing broadly the same way.'] : [undefined, 'Pointing broadly opposite ways.']),
  legend: [['--c1', 'u'], ['--c2', 'v'], ['--c3', 'shadow of v on u']],
});

export const MatrixMap = lab({
  intro: 'A matrix is a machine that moves all of space. Its columns are where the two basis arrows land, and the grid follows.',
  params: [{ k: 'a', label: '$A_{11}$', min: -2, max: 2, step: 0.1, init: 1.2 }, { k: 'b', label: '$A_{12}$', min: -2, max: 2, step: 0.1, init: 0.6 }, { k: 'c', label: '$A_{21}$', min: -2, max: 2, step: 0.1, init: 0.2 }, { k: 'd', label: '$A_{22}$', min: -2, max: 2, step: 0.1, init: 1 }],
  plots: ({ a, b, c, d }) => { const T = (x, y) => [a * x + b * y, c * x + d * y], R2 = [-2, -1, 0, 1, 2]; return [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, label: 'grid transformed by a matrix', marks: [...R2.map(t => ({ tone: 'muted', soft: true, data: [T(t, -2), T(t, 2)] })), ...R2.map(t => ({ tone: 'muted', soft: true, data: [T(-2, t), T(2, t)] })), { as: 'arrow', tone: 1, data: [[0, 0, a, c]] }, { as: 'arrow', tone: 2, data: [[0, 0, b, d]] }] }]; },
  stats: ({ a, b, c, d }) => [`determinant ${fmt(a * d - b * c)}`, 'area of each grid cell is scaled by its size'],
  status: ({ a, b, c, d }) => (Math.abs(a * d - b * c) < 0.03 ? ['alert', 'Singular: space is flattened, and the map cannot be undone.'] : a * d - b * c < 0 ? [undefined, 'Negative determinant: the map flips orientation.'] : [undefined, 'Invertible.']),
  legend: [['--c1', 'first column'], ['--c2', 'second column']],
});

export const MatrixMultiplyDrill = drill({
  intro: 'Entry $(i,j)$ of $AB$ is row $i$ of $A$ dotted with column $j$ of $B$.',
  gen: r => { const f = () => [[int(r, -3, 4), int(r, -3, 4)], [int(r, -3, 4), int(r, -3, 4)]]; return { A: f(), B: f(), i: int(r, 0, 1), j: int(r, 0, 1) }; },
  q: v => `Find entry row ${v.i ? 'two' : 'one'}, column ${v.j ? 'two' : 'one'} of $${mat(v.A)}${mat(v.B)}$.`,
  answer: v => v.A[v.i][0] * v.B[0][v.j] + v.A[v.i][1] * v.B[1][v.j],
  dp: 0,
  steps: v => [`Row of $A$: $(${v.A[v.i].join(',')})$. Column of $B$: $(${v.B[0][v.j]},${v.B[1][v.j]})$.`, `Dot product: $(${v.A[v.i][0]})(${v.B[0][v.j]})+(${v.A[v.i][1]})(${v.B[1][v.j]})$`],
});

export const ShapeDrill = drill({
  intro: 'A product $AB$ needs inner sizes to agree, and the outer sizes survive. Answer like 3x4, or none if it does not exist.',
  gen: r => { const m = int(r, 1, 5), n = int(r, 1, 5), ok = r() < 0.7; return { m, n, p: ok ? n : (n % 5) + 1, q: int(r, 1, 5) }; },
  q: v => `$A$ is $${v.m}\\times${v.n}$ and $B$ is $${v.p}\\times${v.q}$. What is the shape of $AB$?`,
  answer: v => (v.n === v.p ? `${v.m}x${v.q}` : 'none'),
  steps: v => [`Inner sizes: ${v.n} and ${v.p}. ${v.n === v.p ? 'They agree.' : 'They differ, so the product does not exist.'}`, v.n === v.p ? `Outer sizes survive: ${v.m} and ${v.q}.` : 'No shape to report.'],
});

export const DerivativeDrill = drill({
  intro: 'Power rule: bring the exponent down, and reduce it by one.',
  tex: R`\frac{d}{dx}\,a x^{n}=a\,n\,x^{n-1}`,
  gen: r => ({ a: int(r, 1, 5), n: int(r, 2, 4), b: int(r, -4, 4), x: int(r, -3, 3) }),
  q: v => `For $f(x)=${v.a}x^{${v.n}}+(${v.b})x$, find $f'(${v.x})$.`,
  answer: v => v.a * v.n * v.x ** (v.n - 1) + v.b,
  dp: 0,
  steps: v => [`$f'(x)=${v.a * v.n}x^{${v.n - 1}}+(${v.b})$`, `Substitute: $${v.a * v.n}(${v.x})^{${v.n - 1}}+(${v.b})$`],
});

export const PartialSlices = lab({
  intro: 'A partial derivative freezes every other variable. Drag the point: the arrow is the gradient, always perpendicular to the contour through the point.',
  tex: R`f=\tfrac12x^2+\tfrac12xy+y^2,\quad \nabla f=\big(x+\tfrac12y,\ \tfrac12x+2y\big)`,
  params: [{ k: 'x', label: '$x$', min: -3, max: 3, step: 0.05, init: 1.6 }, { k: 'y', label: '$y$', min: -3, max: 3, step: 0.05, init: 1 }],
  derive: ({ x, y }) => ({ gx: x + y / 2, gy: x / 2 + 2 * y }),
  plots: v => { const f = (a, b) => 0.5 * a * a + 0.5 * a * b + b * b, s = 0.45; return [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, xl: 'x', yl: 'y', label: 'contours with gradient arrow', marks: [{ as: 'field', bands: 10, tones: [1, 1], fn: (a, b) => Math.sqrt(clamp(f(a, b) / 18, 0, 1)) }, { as: 'contour', tone: 'fg', levels: [f(v.x, v.y)], fn: f }, { as: 'arrow', tone: 4, data: [[v.x, v.y, v.x + s * v.gx, v.y + s * v.gy]] }, { drag: { x: 'x', y: 'y' } }] }]; },
  stats: v => [`gradient (${fmt(v.gx)}, ${fmt(v.gy)})`, `steepness ${fmt(Math.hypot(v.gx, v.gy))}`],
  takeaway: 'The gradient points uphill, so its opposite is the steepest way down.',
});

export const ArgminInvariance = lab({
  intro: 'Training only cares **where** the minimum is. Apply a transformation and see which ones keep that location.',
  params: [{ k: 'op', label: 'Transformation', options: [['scale', 'multiply by c'], ['shift', 'add c'], ['log', 'take the log'], ['neg', 'negate']], init: 'log' }, { k: 'c', label: 'constant $c$', min: 0.2, max: 3, step: 0.1, init: 2 }],
  derive: ({ op, c }) => { const f = t => (t - 1) ** 2 + 1, g = { scale: t => c * f(t), shift: t => f(t) + c, log: t => Math.log(f(t)), neg: t => -f(t) }[op]; const ts = linspace(-3, 5, 801), a = ts.reduce((b, t) => (g(t) < g(b) ? t : b), ts[0]); return { f, g, a }; },
  plots: v => [{ x: [-3, 5], y: [-6, 12], xl: 'θ', label: 'objective and transformed objective', marks: [{ tone: 'muted', dash: true, data: curve(v.f, -3, 5) }, { tone: 1, data: curve(v.g, -3, 5) }, { as: 'rule', dash: true, x: 1 }, v.op !== 'neg' && { as: 'points', tone: 4, r: 6, data: [[v.a, v.g(v.a)]] }] }],
  status: v => (v.op === 'neg' ? ['alert', 'Negation turns the minimum into a maximum. The minimiser is gone.'] : ['ok', `The minimiser stays at ${fmt(v.a, 2)}, though the minimum value changes to ${fmt(v.g(v.a), 2)}.`]),
  takeaway: 'Positive scaling, adding constants and any increasing function all preserve the argmin. That is why dropping constants in a loss is safe.',
});

export const VarianceDrill = drill({
  intro: 'Variance is the average squared distance from the mean.',
  tex: R`\sigma^2=\frac1n\sum_i (x_i-\bar x)^2`,
  gen: r => { const m = int(r, -2, 5); return { xs: [-3, -1, 1, 3].map(d => m + d * int(r, 1, 2)) }; },
  q: v => `Compute the variance of $${v.xs.join(',\\ ')}$.`,
  answer: v => { const m = mean(v.xs); return mean(v.xs.map(x => (x - m) ** 2)); },
  steps: v => { const m = mean(v.xs); return [`Mean: $\\bar x=${fmt(m)}$`, `Deviations: $${v.xs.map(x => fmt(x - m)).join(',\\ ')}$`, `Squares: $${v.xs.map(x => fmt((x - m) ** 2)).join(',\\ ')}$, then average them.`]; },
});

export const ChainRuleDrill = drill({
  intro: 'Differentiate the outside, keep the inside, then multiply by the derivative of the inside.',
  tex: R`\frac{d}{dx}\,g(h(x))=g'(h(x))\,h'(x)`,
  gen: r => ({ a: int(r, 1, 4), b: int(r, -3, 3), n: int(r, 2, 3), x: int(r, -2, 2) }),
  q: v => `For $f(x)=(${v.a}x+(${v.b}))^{${v.n}}$, find $f'(${v.x})$.`,
  answer: v => v.n * (v.a * v.x + v.b) ** (v.n - 1) * v.a,
  dp: 0,
  steps: v => [`Inside at $x=${v.x}$: $${v.a}(${v.x})+(${v.b})=${v.a * v.x + v.b}$`, `Outside derivative: $${v.n}(${v.a * v.x + v.b})^{${v.n - 1}}$`, `Times the inside derivative, $${v.a}$.`],
  takeaway: 'Backpropagation is this rule applied one layer at a time.',
});

export const MathsCheck = truths({
  items: [
    ['A density value can exceed one.', true, 'Only the area must be at most one, and a narrow density is tall.'],
    ['Every critical point is a minimum.', false, 'Zero slope also occurs at maxima and saddle points. Check curvature.'],
    ['$AB=BA$ for square matrices.', false, 'Matrix multiplication composes maps, and the order of two maps matters.'],
    ['The gradient points towards the minimum.', false, 'It is perpendicular to the contour. It only aims at the minimum on a round bowl.'],
    ['Taking a log keeps the location of a maximum.', true, 'Log is strictly increasing, so the ordering of values is unchanged.'],
  ],
});

/* ---------- estimation ---------- */

const SOURCES = { uniform: [() => 1, r => r(), 0.5, Math.sqrt(1 / 12)], skewed: [() => 1, r => -Math.log(1 - r()), 1, 1], die: [() => 1, r => 1 + Math.floor(6 * r()), 3.5, Math.sqrt(35 / 12)] };

export const SamplingDistribution = lab({
  intro: 'Take many samples and average each one. The averages pile into a bell, whatever the shape of the original data, and the bell narrows as the sample grows.',
  tex: R`\operatorname{sd}(\bar x)=\frac{\sigma}{\sqrt n}`,
  params: [{ k: 'src', label: 'Source', options: [['uniform', 'uniform'], ['skewed', 'skewed'], ['die', 'die']], init: 'skewed' }, { k: 'n', label: 'sample size $n$', min: 1, max: 100, step: 1, init: 5 }, { k: 'seed', seed: 'Resample', init: 1 }],
  derive: ({ src, n, seed }) => {
    const [, draw, mu, s] = SOURCES[src], r = rng(seed * 53), means = Array.from({ length: 800 }, () => { let t = 0; for (let i = 0; i < n; i++) t += draw(r); return t / n; });
    const se = s / Math.sqrt(n), lo = mu - 4 * se, hi = mu + 4 * se, nb = 30, bins = Array(nb).fill(0);
    means.forEach(m => { const b = Math.floor(((m - lo) / (hi - lo)) * nb); if (b >= 0 && b < nb) bins[b]++; });
    const dx = (hi - lo) / nb, top = Math.max(...bins) / (800 * dx);
    return { mu, se, lo, hi, dx, bins, top, avg: mean(means) };
  },
  plots: v => [{ x: [v.lo, v.hi], y: [0, v.top * 1.15], xl: 'sample mean', yl: 'density', label: 'histogram of sample means', marks: [{ as: 'rects', tone: 1, data: v.bins.map((c, i) => [v.lo + i * v.dx, 0, v.lo + (i + 1) * v.dx, c / (800 * v.dx)]) }, { tone: 4, data: curve(x => normalPdf(x, v.mu, v.se), v.lo, v.hi) }] }],
  stats: v => [`average of means ${fmt(v.avg, 3)}`, `predicted spread ${fmt(v.se, 3)}`],
  takeaway: 'This is why a single number computed on data has an uncertainty you can state.',
});

export const VarianceBias = lab({
  intro: 'Dividing by $n$ underestimates spread, because the mean was fitted to the same data. Compare both estimators over many samples.',
  tex: R`\mathbb E\Big[\tfrac1n\sum(x_i-\bar x)^2\Big]=\frac{n-1}{n}\,\sigma^2`,
  params: [{ k: 'n', label: 'sample size $n$', min: 2, max: 30, step: 1, init: 4 }, { k: 'seed', seed: 'Resample', init: 1 }],
  derive: ({ n, seed }) => { const r = rng(seed * 71), v1 = [], v2 = []; for (let k = 0; k < 1500; k++) { const x = Array.from({ length: n }, () => gauss(r)), m = mean(x), s = sum(x.map(a => (a - m) ** 2)); v1.push(s / n); v2.push(s / (n - 1)); } return { a: mean(v1), b: mean(v2) }; },
  plots: v => [{ x: [2, 30], y: [0.3, 1.2], xl: 'n', yl: 'average estimate of σ²', label: 'bias of variance estimators', marks: [{ as: 'rule', tone: 3, dash: true, y: 1 }, { tone: 1, data: curve(n => (n - 1) / n, 2, 30, 57) }, { as: 'points', tone: 1, r: 6, data: [[v.n, v.a]] }, { as: 'points', tone: 2, r: 6, data: [[v.n, clamp(v.b, 0.3, 1.2)]] }] }],
  stats: v => [`divide by n: ${fmt(v.a, 3)}`, `divide by n − 1: ${fmt(v.b, 3)}`],
  legend: [['--c1', 'divide by n (MLE)'], ['--c2', 'divide by n − 1'], ['--c3', 'true value']],
});

export const LogRulesDrill = drill({
  intro: 'Break the number into known pieces. Use $\\ln 2\\approx0.693$, $\\ln 3\\approx1.099$ and $\\ln 5\\approx1.609$.',
  tex: R`\ln(ab)=\ln a+\ln b,\quad \ln(a^k)=k\ln a`,
  gen: r => { const p = int(r, 0, 3), q = int(r, 0, 2), s = int(r, 0, 1); return { p, q, s: p + q + s ? s : 1 }; },
  q: v => `Estimate $\\ln(${2 ** v.p * 3 ** v.q * 5 ** v.s})$.`,
  answer: v => 0.693 * v.p + 1.099 * v.q + 1.609 * v.s,
  steps: v => [`Factor: $2^{${v.p}}\\cdot3^{${v.q}}\\cdot5^{${v.s}}$`, `Sum: $${v.p}(0.693)+${v.q}(1.099)+${v.s}(1.609)$`],
});

export const BernoulliMleDrill = drill({
  intro: 'Maximise the log-likelihood: differentiate, set to zero, and solve.',
  tex: R`\ell(\theta)=h\ln\theta+(n-h)\ln(1-\theta)`,
  gen: r => { const n = int(r, 5, 40); return { n, h: int(r, 1, n - 1) }; },
  q: v => `${v.h} successes in ${v.n} tries. What is the MLE of $\\theta$?`,
  answer: v => v.h / v.n, dp: 3,
  steps: v => [R`$\ell'(\theta)=\frac h\theta-\frac{n-h}{1-\theta}=0$`, `Rearranged: $h(1-\\theta)=(n-h)\\theta$, so $\\theta=h/n$.`, `$${v.h}/${v.n}$`],
});

export const PoissonMleDrill = drill({
  intro: 'For Poisson counts, the same recipe gives the sample mean.',
  tex: R`\ell(\lambda)=\Big(\sum_i x_i\Big)\ln\lambda-n\lambda+\text{const}`,
  gen: r => ({ xs: Array.from({ length: int(r, 4, 7) }, () => int(r, 0, 9)) }),
  q: v => `Counts: $${v.xs.join(',\\ ')}$. What is $\\hat\\lambda$?`,
  answer: v => mean(v.xs),
  steps: v => [R`$\ell'(\lambda)=\frac{\sum x_i}{\lambda}-n=0$`, `$\\sum x_i=${sum(v.xs)}$ and $n=${v.xs.length}$.`],
});

export const MapDrill = drill({
  intro: 'A Beta prior adds pseudo-counts. The MAP is the mode of the posterior.',
  tex: R`\hat\theta_{\text{MAP}}=\frac{a+h-1}{a+b+n-2}`,
  gen: r => { const n = int(r, 3, 20); return { a: int(r, 2, 6), b: int(r, 2, 6), n, h: int(r, 0, n) }; },
  q: v => `Prior Beta(${v.a}, ${v.b}), with ${v.h} heads in ${v.n} tosses. Find the MAP estimate.`,
  answer: v => (v.a + v.h - 1) / (v.a + v.b + v.n - 2), dp: 3,
  steps: v => [`Posterior: Beta(${v.a + v.h}, ${v.b + v.n - v.h}).`, `Mode: $(${v.a + v.h}-1)/(${v.a + v.h}+${v.b + v.n - v.h}-2)$`],
});

export const PriorStrength = lab({
  intro: 'The data say 70% heads. A prior centred on 50% pulls the estimate back, and more data wins the argument.',
  tex: R`\hat\theta_{\text{mean}}=\frac{\kappa\cdot0.5+n\cdot0.7}{\kappa+n}`,
  params: [{ k: 'k', label: 'prior strength $\\kappa$', min: 0, max: 100, step: 1, init: 20 }, { k: 'n', label: 'data size $n$', min: 1, max: 300, step: 1, init: 20 }],
  plots: v => [{ x: [1, 300], y: [0.45, 0.75], xl: 'data size n', yl: 'estimate', label: 'posterior mean as data grow', marks: [{ as: 'rule', tone: 3, dash: true, y: 0.7 }, { as: 'rule', tone: 2, dash: true, y: 0.5 }, { tone: 1, data: curve(n => (v.k * 0.5 + n * 0.7) / (v.k + n), 1, 300) }, { drag: { x: 'n', y: (v.k * 0.5 + v.n * 0.7) / (v.k + v.n) } }] }],
  stats: v => [`estimate ${fmt((v.k * 0.5 + v.n * 0.7) / (v.k + v.n), 3)}`, `weight on data ${pct(v.n / (v.k + v.n), 0)}`],
  takeaway: 'A prior behaves like extra data. That is the same effect as ridge shrinking a coefficient towards zero.',
});

export const FisherCurvature = lab({
  intro: 'The sharper the log-likelihood peak, the better the data pin down $\\theta$. Curvature at the peak gives the standard error.',
  tex: R`I(\theta)=-\ell''(\theta)=\frac{n}{\theta(1-\theta)},\qquad \operatorname{se}\approx\frac{1}{\sqrt{I(\hat\theta)}}`,
  params: [{ k: 'n', label: 'tosses $n$', min: 4, max: 400, step: 2, init: 20 }, { k: 't', label: '$\\hat\\theta$', min: 0.1, max: 0.9, step: 0.01, init: 0.6 }],
  plots: ({ n, t }) => { const ll = x => n * (t * Math.log(x / t) + (1 - t) * Math.log((1 - x) / (1 - t))), q = x => (-0.5 * n * (x - t) ** 2) / (t * (1 - t)); return [{ x: [0.02, 0.98], y: [-6, 0.5], xl: 'θ', yl: 'ℓ(θ) − ℓ(θ̂)', label: 'log likelihood and quadratic approximation', marks: [{ tone: 1, data: curve(x => Math.max(ll(x), -8), 0.02, 0.98) }, { tone: 2, dash: true, data: curve(x => Math.max(q(x), -8), 0.02, 0.98) }, { as: 'rule', dash: true, x: t }] }]; },
  stats: ({ n, t }) => [`standard error ${fmt(Math.sqrt((t * (1 - t)) / n), 4)}`, `information ${fmt(n / (t * (1 - t)), 1)}`],
  legend: [['--c1', 'log-likelihood'], ['--c2', 'quadratic approximation']],
  takeaway: 'Four times the data halves the standard error. Curvature measures how much the data know.',
});

export const EstimationCheck = truths({
  items: [
    ['The likelihood integrates to one over $\\theta$.', false, 'It is a density in the data, not in the parameter.'],
    ['With enough data, MLE and MAP agree.', true, 'The likelihood grows with $n$ while the prior stays fixed.'],
    ['An unbiased estimator is always the better one.', false, 'A small bias can buy a large drop in variance, as ridge shows.'],
    ['A Beta prior with $a=b=1$ is flat.', true, 'Beta(1,1) is uniform, so MAP equals the MLE.'],
    ['The MLE of a Gaussian variance divides by $n-1$.', false, 'The MLE divides by $n$. The $n-1$ version is the unbiased correction.'],
  ],
});
