'use client';
// Explorers for neurons, multi-layer perceptrons and automatic differentiation.
import { R, curve, drill, heatBox, heatPlot, int, lab, pick, truths } from '../components/labs';
import { rng } from '../lib/math.mjs';
import { fmt, linspace, memo, moons, pct } from '../lib/models.mjs';
import { ACT, Value, mlpForward, mlpParams, trainMlp } from '../lib/extended.mjs';

const GATE = [[0, 0], [0, 1], [1, 0], [1, 1]];
const TARGETS = { and: [0, 0, 0, 1], or: [0, 1, 1, 1], xor: [0, 1, 1, 0] };
export const NeuronLab = lab({
  intro: 'A neuron adds weighted inputs and a bias, then fires if the sum is positive. Find settings for each gate, and notice that one of them is impossible.',
  tex: R`z=w_1x_1+w_2x_2+b,\qquad\hat y=[z>0]`,
  params: [{ k: 'g', label: 'Target', options: [['and', 'AND'], ['or', 'OR'], ['xor', 'XOR']], init: 'and' }, { k: 'w1', label: '$w_1$', min: -2, max: 2, step: 0.1, init: 1 }, { k: 'w2', label: '$w_2$', min: -2, max: 2, step: 0.1, init: 1 }, { k: 'b', label: 'bias $b$', min: -3, max: 3, step: 0.1, init: -1.5 }],
  derive: ({ g, w1, w2, b }) => { const pred = GATE.map(x => +(w1 * x[0] + w2 * x[1] + b > 0)), right = pred.filter((p, i) => p === TARGETS[g][i]).length; return { pred, right, T: TARGETS[g] }; },
  plots: v => [{ x: [-0.5, 1.5], y: [-0.5, 1.5], w: 340, h: 320, xl: 'x₁', yl: 'x₂', label: 'decision line', marks: [{ as: 'field', bands: 1, tones: [2, 1], fn: (a, b) => (v.w1 * a + v.w2 * b + v.b > 0 ? 0.35 : -0.35) }, { as: 'points', r: 8, data: GATE.map((x, i) => [x[0], x[1], v.T[i] + 1]) }] }],
  stats: v => [`${v.right} of 4 right`, `outputs ${v.pred.join(', ')}`],
  status: v => (v.right === 4 ? ['ok', 'All four are right.'] : v.g === 'xor' ? ['alert', 'No single line can get all four of XOR.'] : undefined),
  takeaway: 'One neuron draws one straight line, so XOR needs a hidden layer.',
});
export const NeuronDrill = drill({
  intro: 'Add up, then apply the activation.',
  gen: r => ({ w: [int(r, -3, 3), int(r, -3, 3)], x: [int(r, 0, 3), int(r, 0, 3)], b: int(r, -3, 3) }),
  q: v => `$w=(${v.w})$, $x=(${v.x})$, $b=${v.b}$, with ReLU. Find the output.`,
  answer: v => Math.max(0, v.w[0] * v.x[0] + v.w[1] * v.x[1] + v.b), dp: 0,
  steps: v => [`$z=${v.w[0] * v.x[0] + v.w[1] * v.x[1] + v.b}$`, 'ReLU keeps $z$ if positive, and otherwise gives zero.'],
});
export const ActivationCurves = lab({
  intro: 'Compare each activation and its slope. Training multiplies these slopes together, layer after layer.',
  params: [{ k: 'a', label: 'Activation', options: [['sigmoid', 'sigmoid'], ['tanh', 'tanh'], ['relu', 'ReLU']], init: 'sigmoid' }, { k: 'z', label: 'input $z$', min: -6, max: 6, step: 0.1, init: 1 }],
  plots: v => [{ caption: 'Activation', x: [-6, 6], y: [-1.2, 3], w: 300, h: 260, xl: 'z', label: 'activation', marks: [{ tone: 1, data: curve(ACT[v.a][0], -6, 6) }, { as: 'points', tone: 4, r: 6, data: [[v.z, ACT[v.a][0](v.z)]] }] }, { caption: 'Slope', x: [-6, 6], y: [0, 1.1], w: 300, h: 260, xl: 'z', label: 'slope', marks: [{ tone: 2, data: curve(ACT[v.a][1], -6, 6, 401) }, { as: 'points', tone: 4, r: 6, data: [[v.z, ACT[v.a][1](v.z)]] }] }],
  stats: v => [`value ${fmt(ACT[v.a][0](v.z), 3)}`, `slope ${fmt(ACT[v.a][1](v.z), 3)}`],
  takeaway: 'A sigmoid’s slope never exceeds 0.25 and nearly vanishes at large $|z|$, while ReLU passes a slope of 1 wherever it is active.',
});
export const ParamCountDrill = drill({
  intro: 'A weight per connection and a bias per output unit.',
  tex: R`\text{params}=\sum_l\big(n_{l-1}n_l+n_l\big)`,
  gen: r => { const n = int(r, 2, 3); return { L: [int(r, 2, 20), ...Array.from({ length: n }, () => int(r, 2, 16)), int(r, 1, 5)] }; },
  q: v => `Count the parameters of a $${v.L.join('\\text{-}')}$ network.`,
  answer: v => mlpParams(v.L), dp: 0,
  steps: v => v.L.slice(1).map((n, i) => `Layer ${i + 1}: $${v.L[i]}\\times${n}+${n}=${v.L[i] * n + n}$`),
});
const SWEEP = (() => { const out = []; let w = [0, 0, 0]; const eta = 0.2; for (let ep = 0; ep < 6; ep++) GATE.forEach((x, i) => { const inp = [...x, 1], yh = +(w.reduce((s, v, k) => s + v * inp[k], 0) > 0), y = TARGETS.or[i], nw = w.map((v, k) => +(v + eta * (y - yh) * inp[k]).toFixed(3)); out.push({ w, x, i, yh, y, nw }); w = nw; }); return out; })();
export const PerceptronSteps = lab({
  intro: 'Each misclassified point pulls the weights towards getting it right. Step through the rule on OR.',
  tex: R`w\leftarrow w+\eta\,(y-\hat y)\,x`,
  params: [{ k: 's', label: 'update', min: 0, max: SWEEP.length - 1, step: 1, init: 0, play: true }],
  derive: ({ s }) => ({ f: SWEEP[s] }),
  plots: v => [{ x: [-0.5, 1.5], y: [-0.5, 1.5], w: 340, h: 320, xl: 'x₁', yl: 'x₂', label: 'perceptron line', marks: [{ as: 'field', bands: 1, tones: [2, 1], fn: (a, b) => (v.f.nw[0] * a + v.f.nw[1] * b + v.f.nw[2] > 0 ? 0.35 : -0.35) }, { as: 'points', r: 8, data: GATE.map((x, i) => [x[0], x[1], TARGETS.or[i] + 1]) }, { as: 'rings', r: 13, data: [v.f.x] }] }],
  stats: v => [`weights $(${v.f.nw})$`, v.f.yh === v.f.y ? 'this point was right, so nothing changed' : 'this point was wrong, so the weights moved'],
  takeaway: 'If a separating line exists, this rule finds one in finitely many updates.',
});
const HAND = { W1: [[1, 1], [1, 1]], b1: [0, -1], w2: [1, -2], b2: 0 };
export const XorHidden = lab({
  intro: 'The hidden layer moves the four XOR points. In the new space, a single line separates them.',
  params: [{ k: 't', label: 'Show', options: [['x', 'original inputs'], ['h', 'after the hidden layer']], init: 'x' }],
  derive: ({ t }) => ({ pts: GATE.map(x => (t === 'x' ? x : mlpForward(HAND, x).h)) }),
  plots: v => [{ x: [-0.5, 2.5], y: [-0.5, 1.5], w: 400, h: 280, label: 'XOR points', marks: [...(v.t === 'h' ? [{ tone: 4, data: curve(a => (a - 0.5) / 2, -0.5, 2.5) }] : []), { as: 'points', r: 8, data: v.pts.map((p, i) => [p[0], p[1], TARGETS.xor[i] + 1]) }] }],
  table: () => ({ head: ['$x$', '$h$', '$\\hat y$'], rows: GATE.map(x => { const f = mlpForward(HAND, x); return [`$(${x})$`, `$(${f.h})$`, f.z2]; }) }),
  stats: v => [v.t === 'x' ? 'No line separates these colours.' : 'The line $h_1-2h_2=0.5$ separates them.'],
  takeaway: 'A hidden layer learns a new representation in which the problem becomes linear.',
});
export const ForwardPassDrill = drill({
  intro: 'One layer: multiply by the weights, add the bias, then activate.',
  gen: r => ({ W: [[int(r, -2, 2), int(r, -2, 2)], [int(r, -2, 2), int(r, -2, 2)]], x: [int(r, 0, 3), int(r, 0, 3)], b: [int(r, -2, 2), int(r, -2, 2)], k: int(r, 0, 1) }),
  q: v => `$W=\\begin{bmatrix}${v.W[0].join('&')}\\\\${v.W[1].join('&')}\\end{bmatrix}$, $x=(${v.x})$, $b=(${v.b})$, ReLU. Find hidden unit ${v.k ? 'two' : 'one'}.`,
  answer: v => Math.max(0, v.W[v.k][0] * v.x[0] + v.W[v.k][1] * v.x[1] + v.b[v.k]), dp: 0,
  steps: v => [`Row ${v.k + 1}: $${v.W[v.k][0]}\\times${v.x[0]}+${v.W[v.k][1]}\\times${v.x[1]}+(${v.b[v.k]})$`],
});
const MOONS = moons(5, 120, 0.1), XM = MOONS.map(p => [p[0], p[1]]), YM = MOONS.map(p => p[2]);
const mlpRun = memo((hidden, act) => trainMlp(XM, YM, { hidden, act, lr: 0.5, steps: 2000, every: 50, seed: 3 }));
const accM = (p, act) => mean0(XM.map((x, i) => +(mlpForward(p, x, act).p > 0.5) === YM[i]));
const mean0 = a => a.reduce((s, b) => s + +b, 0) / a.length;
export const MlpBoundary = lab({
  intro: 'Train a one-hidden-layer network on two half-moons. Remove the activation and it can only ever be a straight line.',
  params: [{ k: 'act', label: 'Activation', options: [['relu', 'ReLU'], ['tanh', 'tanh'], ['none', 'none']], init: 'relu' }, { k: 'h', label: 'hidden units', min: 1, max: 16, step: 1, init: 8 }, { k: 'i', label: 'training', min: 0, max: 40, step: 1, init: 40, show: v => `${v * 50} steps`, play: true }],
  derive: ({ act, h, i }) => { const F = mlpRun(h, act), f = F[Math.min(i, F.length - 1)]; return { F, f, a: accM(f.p, act), active: i }; },
  plots: v => [{ caption: 'Decision boundary', x: [-1.7, 2.7], y: [-1.3, 1.7], w: 340, h: 280, label: 'network decision boundary', marks: [{ as: 'field', bands: 4, nx: 48, ny: 32, tones: [2, 1], fn: (a, b) => 2 * mlpForward(v.f.p, [a, b], v.act).p - 1 }, { as: 'points', r: 3.5, data: MOONS.map(p => [p[0], p[1], p[2] + 1]) }] }, { caption: 'Loss', x: [0, 2000], y: [0, 1.5], w: 280, h: 280, xl: 'step', label: 'loss by step', marks: [{ tone: 1, data: v.F.map(f => [f.t, Math.min(1.5, f.loss)]) }, { as: 'rule', dash: true, tone: 4, x: v.f.t }] }],
  stats: v => [`accuracy ${pct(v.a, 1)}`, `loss ${fmt(v.f.loss, 3)}`, `${mlpParams([2, v.h, 1])} parameters`],
  takeaway: 'Without a nonlinearity, however many units you add, the network is one linear map.',
});
const bump = (x, c, w) => Math.max(0, 1 - Math.abs(x - c) / w);
const TGT = x => Math.sin(2 * Math.PI * x) + x / 2;
export const UniversalApprox = lab({
  intro: 'A sum of ReLU units is a piecewise-linear curve with one kink per unit. Add units and watch the worst error shrink.',
  params: [{ k: 'u', label: 'units', min: 2, max: 40, step: 1, init: 6 }],
  derive: ({ u }) => { const xs = linspace(0, 1, u + 1), ys = xs.map(TGT), f = x => { const i = Math.min(u - 1, Math.floor(x * u)); return ys[i] + ((x - xs[i]) * (ys[i + 1] - ys[i])) / (xs[i + 1] - xs[i]); }, err = Math.max(...linspace(0, 1, 801).map(x => Math.abs(f(x) - TGT(x)))); return { xs, ys, f, err }; },
  plots: v => [{ x: [0, 1], y: [-1.2, 1.8], w: 420, h: 280, xl: 'x', label: 'piecewise linear fit', marks: [{ tone: 'muted', data: curve(TGT, 0, 1, 401) }, { tone: 1, data: v.xs.map((x, i) => [x, v.ys[i]]) }, { as: 'points', tone: 4, r: 4, data: v.xs.map((x, i) => [x, v.ys[i]]) }] }],
  stats: v => [`largest error ${fmt(v.err, 3)}`],
  note: 'Each kink is one ReLU. This shows that weights exist, and not that gradient descent will find them.',
  takeaway: 'One wide hidden layer can approximate any continuous curve, at the cost of many units.',
});
export const VanishingDepth = lab({
  intro: 'Each layer multiplies the gradient by its slope. See what depth does.',
  params: [{ k: 'L', label: 'layers', min: 1, max: 20, step: 1, init: 5 }, { k: 's', label: 'slope per layer', min: 0.05, max: 1.5, step: 0.05, init: 0.25 }],
  plots: v => { const S = [1.5, 1.25, 1.1, 1, 0.9, 0.75, 0.5, 0.25, 0.1], D = Array.from({ length: 20 }, (_, i) => i + 1), M = S.map(s => D.map(L => Math.max(-8, Math.min(8, L * Math.log10(s))))), near = S.reduce((b, s, i) => (Math.abs(s - v.s) < Math.abs(S[b] - v.s) ? i : b), 0); return [heatPlot(M, { w: 560, h: 300, caption: 'Every slope against every depth', label: 'gradient factor by slope and depth', over: [heatBox(M, near, v.L - 1)], heat: { div: true, max: 8, rows: S.map(s => `${s}`), cols: D.map(L => (L % 5 ? '' : `${L}`)), name: (i, j, x) => `slope ${S[i]}, ${j + 1} layers: factor 1e${x.toFixed(1)}` } }), { x: [1, 20], y: [-14, 4], w: 420, h: 280, xl: 'layers', yl: 'log₁₀ gradient factor', label: 'gradient factor by depth', marks: [{ tone: 1, data: curve(L => L * Math.log10(v.s), 1, 20, 41) }, { as: 'rule', dash: true, tone: 'muted', y: 0 }, { as: 'points', tone: 4, r: 6, data: [[v.L, v.L * Math.log10(v.s)]] }] }]; },
  legend: [['--c1', 'gradient grows'], ['--c4', 'gradient shrinks']],
  stats: v => [`gradient factor ${(v.s ** v.L).toExponential(2)}`],
  status: v => (v.s ** v.L < 1e-3 ? ['alert', 'Early layers will barely learn.'] : v.s ** v.L > 1e3 ? ['alert', 'Gradients are exploding.'] : ['ok', 'Gradients stay usable.']),
  takeaway: 'A sigmoid’s best case is a slope of 0.25, which vanishes quickly. ReLU’s is 1.',
});

/* ---------- automatic differentiation ---------- */

export const ChainRuleDrill = drill({
  intro: 'Multiply the upstream gradient by the local one.',
  tex: R`\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}`,
  gen: r => ({ u: int(r, -5, 5), x: int(r, 1, 5), op: pick(r, ['x^2', '3x', 'x+7']) }),
  q: v => `Upstream gradient $${v.u}$ arrives at $z=${v.op}$, with $x=${v.x}$. What does $x$ receive?`,
  answer: v => v.u * (v.op === 'x^2' ? 2 * v.x : v.op === '3x' ? 3 : 1), dp: 0,
  steps: v => [`Local gradient: $${v.op === 'x^2' ? 2 * v.x : v.op === '3x' ? 3 : 1}$`],
});
export const PathsAddDrill = drill({
  intro: 'A value used on several paths adds up what comes back along each.',
  gen: r => ({ x: int(r, 2, 6), y: int(r, 1, 5) }),
  q: v => `$z=(x+y)(x-y)$ at $x=${v.x}$, $y=${v.y}$. Find $\\partial z/\\partial x$.`,
  answer: v => 2 * v.x, dp: 0,
  steps: v => [`Through $x+y$: upstream $${v.x - v.y}$, local 1.`, `Through $x-y$: upstream $${v.x + v.y}$, local 1.`],
});
export const DualNumberDrill = drill({
  intro: 'Forward mode carries a derivative beside each value, with $\\varepsilon^2=0$.',
  tex: R`(a+\dot a\varepsilon)(b+\dot b\varepsilon)=ab+(\dot ab+a\dot b)\varepsilon`,
  gen: r => ({ a: int(r, 1, 6), da: int(r, 0, 4), b: int(r, 1, 6), db: int(r, 0, 4) }),
  q: v => `Multiply $(${v.a}+${v.da}\\varepsilon)(${v.b}+${v.db}\\varepsilon)$. Find the coefficient of $\\varepsilon$.`,
  answer: v => v.da * v.b + v.a * v.db, dp: 0,
  steps: v => [`$\\dot ab=${v.da * v.b}$`, `$a\\dot b=${v.a * v.db}$`],
});
function logisticNodes(t0, t1, t2, x1, x2) {
  const T = [t0, t1, t2].map((v, i) => new Value(v, [], '', `θ${['₀', '₁', '₂'][i]}`)), f1 = T[1].mul(x1), f2 = T[2].mul(x2), f3 = f1.add(f2), f4 = f3.add(T[0]), f5 = f4.neg(), f6 = f5.exp(), f7 = f6.add(1), f8 = f7.inv(), f9 = f8.log(), L = f9.neg();
  L.backward();
  return { T, rows: [['$\\theta_1x_1$', f1], ['$\\theta_2x_2$', f2], ['$f_1+f_2$', f3], ['$+\\theta_0$', f4], ['$-z$', f5], ['$e^{(\\cdot)}$', f6], ['$+1$', f7], ['$1/(\\cdot)$', f8], ['$\\log$', f9], ['$L$', L]], L };
}
export const BackpropGraph = lab({
  intro: 'The logistic loss, node by node: values flow forward, gradients flow back. Compare the parameter gradients with the formula.',
  tex: R`L=\log(1+e^{-z}),\qquad z=\theta_0+\theta_1x_1+\theta_2x_2`,
  params: [{ k: 't0', label: '$\\theta_0$', min: -3, max: 3, step: 0.1, init: 1 }, { k: 't1', label: '$\\theta_1$', min: -3, max: 3, step: 0.1, init: 1 }, { k: 't2', label: '$\\theta_2$', min: -3, max: 3, step: 0.1, init: 2 }],
  derive: ({ t0, t1, t2 }) => logisticNodes(t0, t1, t2, 1, 2),
  plots: v => [{ x: [0, 11], y: [0, Math.max(1.05, ...v.rows.map(r => Math.abs(r[1].grad)))], w: 420, h: 220, xl: 'node, forward →', yl: '|gradient|', label: 'gradient at each node', marks: [{ as: 'rects', tone: 2, data: v.rows.map((r, i) => [i + 0.6, 0, i + 1.4, Math.abs(r[1].grad)]) }] }],
  stats: v => [`$L=${fmt(v.L.data, 5)}$`, `$\\partial L/\\partial\\theta=(${v.T.map(t => fmt(t.grad, 5))})$`],
  table: v => ({ head: ['node', 'value', '$\\partial L/\\partial$ node'], rows: v.rows.map(([n, f]) => [n, fmt(f.data, 5), fmt(f.grad, 5)]) }),
  takeaway: 'One backward sweep gives every gradient, and they equal $(\\sigma(z)-1)x_j$.',
});
const f14 = x => 3 * x * x + 2 * x + 1;
export const FiniteDifferences = lab({
  intro: 'Nudging an input estimates a slope. Too large a nudge is inaccurate, and too small a one drowns in rounding error.',
  params: [{ k: 'h', label: 'log₁₀ step $h$', min: -18, max: -1, step: 0.5, init: -6 }],
  derive: ({ h }) => { const H = 10 ** h, one = (f14(2 + H) - f14(2)) / H, two = (f14(2 + H) - f14(2 - H)) / (2 * H); return { one, two, H }; },
  plots: () => [{ x: [-18, -1], y: [-16, 2], w: 420, h: 280, xl: 'log₁₀ h', yl: 'log₁₀ error', label: 'error against step size', marks: [{ tone: 1, data: linspace(-18, -1, 69).map(h => { const H = 10 ** h, e = Math.abs((f14(2 + H) - f14(2)) / H - 14); return [h, Math.max(-16, Math.log10(e || 1e-16))]; }) }, { tone: 2, data: linspace(-18, -1, 69).map(h => { const H = 10 ** h, e = Math.abs((f14(2 + H) - f14(2 - H)) / (2 * H) - 14); return [h, Math.max(-16, Math.log10(e || 1e-16))]; }) }] }],
  stats: v => [`one-sided ${fmt(v.one, 6)}`, `two-sided ${fmt(v.two, 6)}`, 'true slope 14'],
  legend: [['--c1', 'one-sided'], ['--c2', 'two-sided']],
  takeaway: 'Numerical derivatives need two runs per parameter and are fragile. Use them to check gradients, and not to train.',
});
const EXPR = { sq: ['x² + 3x', x => x.mul(x).add(x.mul(3)), x => 2 * x + 3], cube: ['x · x · x', x => x.mul(x).mul(x), x => 3 * x * x], logp: ['log(x + 1) · x', x => x.add(1).log().mul(x), x => Math.log(x + 1) + x / (x + 1)] };
export const AutogradEngine = lab({
  intro: 'A tiny engine built from one class. Pick an expression and compare what `backward` computes with the true derivative.',
  params: [{ k: 'e', label: 'Expression', options: Object.keys(EXPR).map(k => [k, EXPR[k][0]]), init: 'sq' }, { k: 'x', label: '$x$', min: 0.1, max: 5, step: 0.1, init: 2 }],
  derive: ({ e, x }) => { const v = new Value(x), o = EXPR[e][1](v), list = o.backward(); return { val: o.data, grad: v.grad, truth: EXPR[e][2](x), nodes: list.length, ops: list.filter(n => n.op).map(n => n.op) }; },
  stats: v => [`$d f/dx=${fmt(v.grad, 6)}$`, `true value ${fmt(v.truth, 6)}`, `${v.nodes} nodes, operations ${v.ops.join(' ')}`],
  plots: v => [{ x: [0, 5], y: [-1, 40], w: 400, h: 240, xl: 'x', label: 'derivative of the expression', marks: [{ tone: 2, data: curve(EXPR[v.e][2], 0.05, 5) }, { as: 'points', tone: 4, r: 6, data: [[v.x, v.grad]] }] }],
  takeaway: 'Every library works this way: record operations forward, then multiply local gradients backward.',
});
const ACC = (() => { const g = [], acc = []; let w = 3, b = 1, gw = 0, gb = 0, w2 = 3, b2 = 1; for (let t = 0; t < 40; t++) { const e = 2 * w + b; g.push(e * e); gw += 2 * e * 2; gb += 2 * e; w -= 0.01 * gw; b -= 0.01 * gb; const e2 = 2 * w2 + b2; acc.push(e2 * e2); w2 -= 0.01 * (2 * e2 * 2); b2 -= 0.01 * (2 * e2); } return { g, acc }; })();
export const ZeroGradLab = lab({
  intro: 'Gradients accumulate until you clear them. Compare descent that zeroes them each step with descent that does not.',
  params: [{ k: 'm', label: 'Show', options: [['both', 'both runs'], ['bad', 'never zeroed']], init: 'both' }, { k: 't', label: 'step', min: 0, max: 39, step: 1, init: 39 }],
  plots: v => [{ x: [0, 39], y: [0, 60], w: 420, h: 280, xl: 'step', yl: 'loss', label: 'loss with and without zeroing', marks: [v.m === 'both' && { tone: 3, data: ACC.acc.map((l, i) => [i, l]) }, { tone: 4, data: ACC.g.map((l, i) => [i, Math.min(60, l)]) }, { as: 'rule', dash: true, tone: 'muted', x: v.t }] }],
  stats: v => [`zeroed: loss ${fmt(ACC.acc[v.t], 3)}`, `never zeroed: loss ${fmt(ACC.g[v.t], 3)}`],
  legend: [['--c3', 'zero_grad each step'], ['--c4', 'never zeroed']],
  takeaway: 'Forgetting `zero_grad()` makes each step use the sum of all past gradients.',
});
export const NeuralCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Logistic regression is a single neuron with a sigmoid.', true, 'One weighted sum, then a sigmoid.'],
    ['Stacked layers without activations are more powerful than one layer.', false, 'Their composition is one linear map.'],
    ['Reverse mode needs one sweep per parameter.', false, 'That is forward mode. Reverse mode needs one sweep per output.'],
    ['Autodiff approximates derivatives numerically.', false, 'It applies the chain rule to the actual values, exact up to rounding.'],
    ['If all weights start equal, hidden units stay identical.', true, 'They receive identical gradients.'],
    ['Universal approximation says gradient descent will find the weights.', false, 'It only says suitable weights exist.'],
    ['Gradients from different paths add.', true, 'That is the multivariate chain rule.'],
  ],
  takeaway: 'A network is simple pieces, and backpropagation is the chain rule run backwards.',
});
