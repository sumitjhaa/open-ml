'use client';
// Explorers for the deep learning course: losses, autograd on arrays, vision-language models and detection.
import { R, curve, drill, heatBox, heatPlot, histDensity, int, lab, pick, ridge, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { fmt, linspace, memo, normalPdf, pct } from '../lib/models.mjs';
import { iou, softmaxT } from '../lib/extended.mjs';

const SEQ = [1, 1, 0, 1, 1, 0, 1, 1];
export const IidLikelihoodLab = lab({
  intro: 'Eight coin flips. The likelihood of the whole sequence is a product. Its log is a sum, with the same peak.',
  tex: R`\log p(\mathcal D\mid\theta)=\sum_i\log p(y_i\mid\theta)`,
  params: [{ k: 't', label: '$\\theta$', min: 0.02, max: 0.98, step: 0.01, init: 0.5 }],
  plots: v => [{ x: [0, 1], y: [-20, 0], w: 480, h: 260, xl: 'θ', yl: 'log-likelihood', label: 'log-likelihood against theta', marks: [{ tone: 1, data: curve(t => Math.max(-20, SEQ.reduce((s, y) => s + Math.log(y ? t : 1 - t), 0)), 0.01, 0.99) }, { as: 'rule', dash: true, tone: 4, x: v.t }, { as: 'rule', dash: true, tone: 3, x: 0.75 }] }],
  stats: v => [`likelihood ${SEQ.reduce((p, y) => p * (y ? v.t : 1 - v.t), 1).toExponential(2)}`, `log-likelihood ${fmt(SEQ.reduce((s, y) => s + Math.log(y ? v.t : 1 - v.t), 0), 3)}`],
  takeaway: 'Maximising the log gives the same $\\theta$, and a negative log-likelihood is a loss.',
});
export const GaussianNllDrill = drill({
  intro: 'With fixed noise, the Gaussian negative log-likelihood is squared error plus a constant.',
  tex: R`\text{NLL}=\frac{(y-\mu)^2}{2\sigma^2}+\tfrac12\ln(2\pi\sigma^2)`,
  gen: r => ({ y: int(r, -3, 5), m: int(r, -3, 5), s: pick(r, [1, 2]) }),
  q: v => `$y=${v.y}$, $\\mu=${v.m}$, $\\sigma=${v.s}$. Find the Gaussian negative log-likelihood.`,
  answer: v => (v.y - v.m) ** 2 / (2 * v.s * v.s) + 0.5 * Math.log(2 * Math.PI * v.s * v.s), dp: 3,
  steps: v => [`Squared part: $${fmt((v.y - v.m) ** 2 / (2 * v.s * v.s), 3)}$`, `Constant part: $${fmt(0.5 * Math.log(2 * Math.PI * v.s * v.s), 3)}$`],
});
export const SoftmaxCeDrill = drill({
  intro: 'Softmax the scores, then take the negative log of the true class.',
  tex: R`L=-\log\frac{e^{z_y}}{\sum_je^{z_j}}`,
  gen: r => ({ z: [int(r, -2, 3), int(r, -2, 3), int(r, -2, 3)], y: int(r, 0, 2) }),
  q: v => `Scores $(${v.z})$, and the true class is number ${v.y + 1}. Find the cross-entropy.`,
  answer: v => -Math.log(softmaxT(v.z)[v.y]), dp: 3,
  steps: v => [`Probabilities: $(${softmaxT(v.z).map(p => fmt(p, 3))})$`],
});
export const MapPriorLab = lab({
  intro: 'A prior pulls the estimate towards zero. More data reduce its pull.',
  tex: R`\hat\mu=\frac{n\bar x/\sigma^2}{n/\sigma^2+1/\tau^2}`,
  params: [{ k: 'n', label: 'observations', min: 1, max: 100, step: 1, init: 5 }, { k: 't', label: 'prior spread $\\tau$', min: 0.1, max: 5, step: 0.1, init: 1 }, { k: 'x', label: 'sample mean', min: -4, max: 4, step: 0.1, init: 3 }],
  derive: ({ n, t, x }) => ({ map: (n * x) / (n + 1 / (t * t)) }),
  plots: v => [{ x: [1, 100], y: [-4, 4], w: 480, h: 260, xl: 'n', label: 'MAP estimate against n', marks: [{ tone: 1, data: curve(n => (n * v.x) / (n + 1 / (v.t * v.t)), 1, 100) }, { as: 'rule', dash: true, tone: 3, y: v.x }, { as: 'points', tone: 4, r: 6, data: [[v.n, v.map]] }] }],
  stats: v => [`MAP estimate ${fmt(v.map, 3)}`, `MLE ${fmt(v.x, 3)}`],
  takeaway: 'A prior acts like extra pretend observations, and weight decay is exactly this.',
});
export const HeteroscedasticLab = lab({
  intro: 'A network can also output its own uncertainty. For a given error, find the spread that costs least.',
  tex: R`L(\sigma)=\frac{e^2}{2\sigma^2}+\ln\sigma`,
  params: [{ k: 'e', label: 'error $e$', min: 0.2, max: 4, step: 0.1, init: 2 }, { k: 's', label: 'predicted spread $\\sigma$', min: 0.2, max: 6, step: 0.05, init: 1 }],
  plots: v => [{ x: [0.2, 6], y: [-2, 6], w: 480, h: 260, xl: 'σ', yl: 'loss', label: 'loss against sigma', marks: [{ tone: 1, data: curve(s => Math.min(6, (v.e * v.e) / (2 * s * s) + Math.log(s)), 0.2, 6, 241) }, { as: 'rule', dash: true, tone: 3, x: v.e }, { as: 'points', tone: 4, r: 6, data: [[v.s, Math.min(6, (v.e * v.e) / (2 * v.s * v.s) + Math.log(v.s))]] }] }],
  stats: v => [`loss ${fmt((v.e * v.e) / (2 * v.s * v.s) + Math.log(v.s), 3)}`, `best spread ${fmt(v.e, 2)}`],
  takeaway: 'Overconfidence is expensive, and blanket uncertainty is too, so the model learns where to be unsure.',
});
export const JacobianShapeDrill = drill({
  intro: 'A Jacobian has one row per output and one column per input.',
  gen: r => ({ n: int(r, 2, 9), m: int(r, 2, 9) }),
  q: v => `A function maps $\\mathbb R^{${v.n}}$ to $\\mathbb R^{${v.m}}$. How many entries does its Jacobian have?`,
  answer: v => v.m * v.n, dp: 0,
  steps: v => [`Shape: $${v.m}\\times${v.n}$`],
});
export const VjpDrill = drill({
  intro: 'Reverse mode multiplies by the transpose, and never builds the whole Jacobian.',
  tex: R`\delta_{\text{in}}=W^\top\delta_{\text{out}}`,
  gen: r => ({ W: [[int(r, -2, 3), int(r, -2, 3)], [int(r, -2, 3), int(r, -2, 3)]], d: [int(r, -2, 3), int(r, -2, 3)], k: int(r, 0, 1) }),
  q: v => `$W=\\begin{bmatrix}${v.W[0].join('&')}\\\\${v.W[1].join('&')}\\end{bmatrix}$, $\\delta=(${v.d})$. Find entry ${v.k ? 'two' : 'one'} of $W^\\top\\delta$.`,
  answer: v => v.W[0][v.k] * v.d[0] + v.W[1][v.k] * v.d[1], dp: 0,
  steps: v => [`Column ${v.k + 1} of $W$ is $(${v.W[0][v.k]},${v.W[1][v.k]})$.`],
});
const PTS = (() => { const r = rng(5); return Array.from({ length: 400 }, () => { const x = gauss(r); return [x, +(2 * x + gauss(r)).toFixed(4)]; }); })();
const gradAt = (rows, w) => mean(rows.map(([x, y]) => 2 * (w * x - y) * x));
const batchGrads = memo((b, seed) => { const r = rng(seed * 13 + b); return Array.from({ length: 300 }, () => gradAt(Array.from({ length: b }, () => PTS[Math.floor(r() * 400)]), 0)); });
export const MinibatchNoiseLab = lab({
  intro: 'A minibatch estimates the full gradient. Each estimate is right on average, and noisier for smaller batches.',
  params: [{ k: 'b', label: 'batch size', min: 1, max: 256, step: 1, init: 8 }, { k: 'seed', seed: 'New batches', init: 1 }],
  derive: ({ b, seed }) => { const G = batchGrads(b, seed), m = mean(G); return { G, m, sd: Math.sqrt(mean(G.map(v => (v - m) ** 2))), full: gradAt(PTS, 0) }; },
  plots: v => { const bins = Array(50).fill(0); v.G.forEach(g => { const i = Math.floor((g + 16) / 0.32); if (i >= 0 && i < 50) bins[i]++; }); return [{ x: [-16, 0], y: [0, Math.ceil(Math.max(...bins) * 1.15)], w: 480, h: 260, xl: 'gradient estimate', label: 'distribution of gradient estimates', marks: [{ as: 'rects', tone: 1, data: bins.map((c, i) => [-16 + i * 0.32, 0, -16 + (i + 0.85) * 0.32, c]) }, { as: 'rule', dash: true, tone: 4, x: v.full }] }, { caption: 'Batch sizes compared', x: [-16, 0], y: [0, 5.8], w: 480, h: 300, noY: true, xl: 'gradient estimate', label: 'gradient estimates by batch size', marks: [...ridge([1, 4, 16, 64, 256].map(b => ({ label: `batch ${b}`, d: histDensity(batchGrads(b, v.seed), -16, 0, 50) })), linspace(-15.84, -0.16, 50)), { as: 'rule', dash: true, tone: 4, x: v.full }] }]; },
  stats: v => [`standard deviation ${fmt(v.sd, 3)}`, `mean ${fmt(v.m, 3)}, full gradient ${fmt(v.full, 3)}`],
  takeaway: 'Noise falls like $1/\\sqrt b$, so four times the batch only halves it.',
});
const TRANS = { softplus: ['softplus', u => Math.log(1 + Math.exp(u))], exp: ['exponential', u => Math.exp(u)], sigmoid: ['sigmoid', u => 1 / (1 + Math.exp(-u))], tanh: ['tanh', u => Math.tanh(u)] };
export const ValidOutputLab = lab({
  intro: 'Let the network output any number, and pass it through a function whose range is exactly what you need.',
  params: [{ k: 'k', label: 'Transform', options: Object.keys(TRANS).map(k => [k, TRANS[k][0]]), init: 'softplus' }, { k: 'u', label: 'raw output $u$', min: -5, max: 5, step: 0.05, init: -1 }],
  plots: v => [{ x: [-5, 5], y: [-1.2, 6], w: 480, h: 260, xl: 'raw output', label: 'transform', marks: [{ tone: 1, data: curve(u => Math.min(6, TRANS[v.k][1](u)), -5, 5) }, { as: 'points', tone: 4, r: 6, data: [[v.u, Math.min(6, TRANS[v.k][1](v.u))]] }] }],
  stats: v => [`valid output ${fmt(TRANS[v.k][1](v.u), 4)}`],
  takeaway: 'Invalid outputs become unreachable, so there is nothing to clip or to punish.',
});
const propagate = memo((a, v2, seed) => { const r = rng(seed * 11 + 1); let x = Array.from({ length: 2000 }, () => gauss(r)); const out = [Math.sqrt(mean(x.map(s => s * s)))]; for (let l = 0; l < 12; l++) { x = x.map(() => { let s = 0; for (let k = 0; k < 20; k++) s += Math.sqrt(v2 / 20) * gauss(r) * x[Math.floor(r() * x.length)]; return a === 'relu' ? Math.max(0, s) : Math.tanh(s); }); out.push(Math.sqrt(mean(x.map(s => s * s)))); } return out; });
const VARS = { small: ['too small', 0.3], xavier: ['Xavier', 1], he: ['He', 2], big: ['too large', 5] };
export const InitVarianceLab = lab({
  intro: 'Send signals through twelve layers. Weight variance decides whether they die, explode or survive.',
  tex: R`\text{Var}(z)=n\,\sigma_w^2\,\text{Var}(x)`,
  params: [{ k: 'a', label: 'Activation', options: [['relu', 'ReLU'], ['tanh', 'tanh']], init: 'relu' }, { k: 'v', label: 'Initialisation', options: Object.keys(VARS).map(k => [k, VARS[k][0]]), init: 'he' }, { k: 'seed', seed: 'New weights', init: 1 }],
  derive: ({ a, v, seed }) => ({ S: propagate(a, VARS[v][1], seed), M: Object.values(VARS).map(o => propagate(a, o[1], seed).map(s => Math.max(-4, Math.min(4, Math.log10(s + 1e-9))))) }),
  plots: v => { const names = Object.values(VARS).map(o => o[0]); return [heatPlot(v.M, { w: 560, h: 230, caption: 'Every initialisation, layer by layer', label: 'log signal size by initialisation and layer', over: [heatBox(v.M, Object.keys(VARS).indexOf(v.v), null)], heat: { div: true, max: 4, rows: names, cols: v.M[0].map((_, i) => `${i}`), show: x => (Math.abs(x) < 0.05 ? '0.0' : x.toFixed(1)), name: (i, j, x) => `${names[i]}, layer ${j}: log signal ${x.toFixed(2)}` } }), { x: [0, 12], y: [-4, 3], w: 480, h: 260, xl: 'layer', yl: 'log₁₀ signal size', label: 'signal size by layer', marks: [{ tone: 1, data: v.S.map((s, i) => [i, Math.max(-4, Math.min(3, Math.log10(s + 1e-9)))]) }, { as: 'rule', dash: true, tone: 'muted', y: 0 }] }]; },
  legend: [['--c1', 'signal grows'], ['--c4', 'signal shrinks']],
  stats: v => [`signal after twelve layers ${fmt(v.S.at(-1), 4)}`],
  takeaway: 'He initialisation doubles the variance for ReLU, because ReLU discards half the signal.',
});
export const MixupDrill = drill({
  intro: 'Mixup blends two inputs, and blends their labels in the same proportion.',
  tex: R`\tilde x=\lambda x_i+(1-\lambda)x_j`,
  gen: r => ({ l: pick(r, [0.2, 0.3, 0.7, 0.9]), a: int(r, 0, 9), b: int(r, 0, 9) }),
  q: v => `$\\lambda=${v.l}$, $x_i=${v.a}$, $x_j=${v.b}$. Find $\\tilde x$.`,
  answer: v => v.l * v.a + (1 - v.l) * v.b, dp: 2,
  steps: v => [`$${v.l}\\times${v.a}+${fmt(1 - v.l, 1)}\\times${v.b}$`],
});
export const DiagnoseCurves = truths({
  intro: 'Read a pair of learning curves before reaching for a fix.',
  items: [
    ['High training loss, close to validation loss, means underfitting.', true, 'The model cannot fit even the training data.'],
    ['Low training loss, far below validation loss, means overfitting.', true, 'Regularisation or more data should help.'],
    ['Adding dropout fixes underfitting.', false, 'It reduces capacity further.'],
    ['The test set should guide hyperparameter choices.', false, 'Use validation data. Open the test set once.'],
    ['A loss that rises sharply often means the learning rate is too high.', true, 'Steps overshoot the minimum.'],
  ],
  takeaway: 'Diagnose first. The right fix depends on which curve is wrong.',
});
export const PatchEmbeddingLab = lab({
  intro: 'Cut an image into patches. Each patch becomes one token.',
  params: [{ k: 's', label: 'image side', min: 32, max: 448, step: 16, init: 224 }, { k: 'p', label: 'patch side', min: 4, max: 32, step: 4, init: 16 }],
  derive: ({ s, p }) => ({ n: Math.floor(s / p) }),
  plots: v => [{ x: [0, v.s], y: [0, v.s], w: 320, h: 320, label: 'image divided into patches', marks: [...Array.from({ length: v.n + 1 }, (_, i) => ({ tone: 'muted', data: [[i * v.p, 0], [i * v.p, v.n * v.p]] })), ...Array.from({ length: v.n + 1 }, (_, i) => ({ tone: 'muted', data: [[0, i * v.p], [v.n * v.p, i * v.p]] }))] }],
  stats: v => [`${v.n * v.n} patch tokens`, `each patch has ${v.p * v.p * 3} numbers in RGB`],
  takeaway: 'Smaller patches give more detail, and attention costs grow with the square of the number of tokens.',
});
export const InfoNceDrill = drill({
  intro: 'InfoNCE is a cross-entropy where the right answer is the matching item.',
  tex: R`L=-\log\frac{e^{s_+/\tau}}{\sum_je^{s_j/\tau}}`,
  gen: r => ({ p: int(r, 0, 3), n: int(r, -1, 2), t: pick(r, [1, 0.5]) }),
  q: v => `Positive similarity $${v.p}$, one negative $${v.n}$, $\\tau=${v.t}$. Find the loss.`,
  answer: v => -Math.log(softmaxT([v.p, v.n], v.t)[0]), dp: 3,
  steps: v => [`Divide by $\\tau$: $${v.p / v.t}$ and $${v.n / v.t}$`, 'Take a softmax over the two.'],
});
export const CrossAttentionDrill = drill({
  intro: 'Queries come from one sequence, and keys and values from another.',
  gen: r => ({ t: pick(r, [8, 12, 20]), p: pick(r, [49, 196, 256]) }),
  q: v => `$${v.t}$ text tokens attend to $${v.p}$ image patches. How many attention weights are there?`,
  answer: v => v.t * v.p, dp: 0,
  steps: v => [`The matrix is $${v.t}\\times${v.p}$.`],
});
const CONCEPTS = [['red beak', 2.2, -1.2], ['long tail', -0.8, 1.8], ['spotted wing', 1, 0.4]];
export const ConceptBottleneckLab = lab({
  intro: 'The class depends only on named concepts. Correct a concept and watch the prediction change.',
  tex: R`x\to\hat c\to\hat y`,
  params: [{ k: 'a', label: 'red beak', min: 0, max: 1, step: 0.05, init: 0.2 }, { k: 'b', label: 'long tail', min: 0, max: 1, step: 0.05, init: 0.8 }, { k: 'c', label: 'spotted wing', min: 0, max: 1, step: 0.05, init: 0.5 }],
  derive: ({ a, b, c }) => { const v = [a, b, c], s = [0, 1].map(k => CONCEPTS.reduce((t, co, i) => t + co[k + 1] * v[i], 0)); return { v, p: softmaxT(s) }; },
  plots: v => [{ x: [-0.5, 1.5], y: [0, 1], w: 320, h: 260, xl: 'cardinal, swallow', label: 'class probabilities', marks: [{ as: 'rects', tone: 1, data: v.p.map((q, i) => [i - 0.35, 0, i + 0.35, q]) }] }],
  table: v => ({ head: ['concept', 'value', 'weight, cardinal', 'weight, swallow'], rows: CONCEPTS.map((co, i) => [co[0], fmt(v.v[i], 2), co[1], co[2]]) }),
  stats: v => [`predicted: ${v.p[0] > v.p[1] ? 'cardinal' : 'swallow'}`, `confidence ${pct(Math.max(...v.p), 1)}`],
  takeaway: 'A person can fix a wrong concept and see the decision follow, which a black box does not allow.',
});
export const DiceIouDrill = drill({
  intro: 'Two common scores for how well masks overlap.',
  tex: R`\text{Dice}=\frac{2|A\cap B|}{|A|+|B|}`,
  gen: r => { const a = int(r, 20, 60), b = int(r, 20, 60); return { a, b, i: int(r, 5, Math.min(a, b)), k: pick(r, ['Dice', 'IoU']) }; },
  q: v => `$|A|=${v.a}$, $|B|=${v.b}$, intersection $${v.i}$. Find the ${v.k}.`,
  answer: v => (v.k === 'Dice' ? (2 * v.i) / (v.a + v.b) : v.i / (v.a + v.b - v.i)), dp: 3,
  steps: v => [`Union: $${v.a + v.b - v.i}$`],
});
export const DiceVsCeLab = lab({
  intro: 'A rare foreground, and a model that predicts it more or less often. Cross-entropy can look excellent while Dice shows failure.',
  params: [{ k: 'f', label: 'true foreground share', min: 0.005, max: 0.5, step: 0.005, init: 0.02 }, { k: 'r', label: 'share of foreground found', min: 0, max: 1, step: 0.02, init: 0 }],
  derive: ({ f, r }) => { const tp = f * r, fn = f - tp, tn = 1 - f, eps = 0.01, ce = -(tp * Math.log(1 - eps) + fn * Math.log(eps) + tn * Math.log(1 - eps)), dice = (2 * tp) / (2 * tp + fn || 1); return { ce, dice, acc: 1 - fn }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 480, h: 260, xl: 'share of foreground found', label: 'Dice against foreground found', marks: [{ tone: 1, data: curve(r => (2 * v.f * r) / (2 * v.f * r + (v.f - v.f * r) || 1), 0, 1) }, { as: 'rule', dash: true, tone: 4, x: v.r }] }],
  stats: v => [`pixel accuracy ${pct(v.acc, 1)}`, `Dice ${fmt(v.dice, 3)}`, `cross-entropy ${fmt(v.ce, 3)}`],
  takeaway: 'Rare classes need a loss that looks only at the foreground, or one that weights it.',
});
const GT = [3, 3, 7, 7], ANCH = Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => [i * 2 + 0.5, j * 2 + 0.5, i * 2 + 4.5, j * 2 + 4.5])).flat();
const boxL = (b, tone, dash) => ({ tone, dash, data: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]] });
export const AnchorAssignLab = lab({
  intro: 'Each candidate box is compared with the true box. High overlap trains as positive, low overlap as background.',
  params: [{ k: 'hi', label: 'positive threshold', min: 0.2, max: 0.9, step: 0.05, init: 0.5 }, { k: 'lo', label: 'background threshold', min: 0, max: 0.5, step: 0.05, init: 0.3 }],
  derive: ({ hi, lo }) => { const s = ANCH.map(a => iou(a, GT)); return { pos: ANCH.filter((_, i) => s[i] >= hi), neu: ANCH.filter((_, i) => s[i] < hi && s[i] >= lo), neg: ANCH.filter((_, i) => s[i] < lo).length, best: Math.max(...s) }; },
  plots: v => [{ x: [0, 13], y: [0, 13], w: 360, h: 340, label: 'anchors and ground truth', marks: [...v.neu.map(a => boxL(a, 'muted', true)), ...v.pos.map(a => boxL(a, 3)), boxL(GT, 4)] }],
  stats: v => [`${v.pos.length} positive, ${v.neu.length} ignored, ${v.neg} background`, `best overlap ${fmt(v.best, 3)}`],
  takeaway: 'Most candidates are background, which is why detectors need focal loss or careful sampling.',
});
export const VisionTasks = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Instance segmentation separates two touching dogs.', true, 'Semantic segmentation would call them both "dog".'],
    ['One-to-one matching removes the need for non-maximum suppression.', true, 'Duplicates are penalised while training.'],
    ['A concept bottleneck’s class depends on more than its concepts.', false, 'By construction it uses the concepts only.'],
    ['A cross-attention output has one row per query.', true, 'Queries decide the output length.'],
    ['Focal loss increases the weight of easy negatives.', false, 'It reduces it.'],
  ],
  takeaway: 'Spatial tasks reuse classification, and add rules for assignment and overlap.',
});
