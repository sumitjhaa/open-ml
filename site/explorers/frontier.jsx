'use client';
// Explorers for PyTorch practice, architectures, language models, diffusion in depth, graphs and deep control, and interview preparation.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { fmt, linspace, normalCdf, pct } from '../lib/models.mjs';
import { softmaxT } from '../lib/extended.mjs';

const r6 = x => +x.toFixed(6);
const hist = (xs, a, b, n) => { const c = Array(n).fill(0), w = (b - a) / n; xs.forEach(x => { const i = Math.floor((x - a) / w); if (i >= 0 && i < n) c[i]++; }); return c.map((v, i) => [a + i * w, 0, a + (i + 0.85) * w, v]); };
const std = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const sigmoid = z => 1 / (1 + Math.exp(-z));

/* ---------- PyTorch in practice ---------- */
export const PtBroadcastDrill = drill({
  intro: 'Line shapes up from the right. A size of 1 stretches to match.',
  gen: r => ({ a: int(r, 2, 9), b: int(r, 2, 9), c: int(r, 2, 6) }),
  q: v => `Tensors of shape $(${v.a},1,${v.c})$ and $(1,${v.b},${v.c})$ are added. How many numbers does the result hold?`,
  answer: v => v.a * v.b * v.c, dp: 0,
  steps: v => [`Result shape: $(${v.a},${v.b},${v.c})$`],
});
export const PtBatchesDrill = drill({
  intro: 'A loader keeps the short final batch unless told to drop it.',
  tex: R`\text{steps}=\Big\lceil\frac nb\Big\rceil\times\text{epochs}`,
  gen: r => ({ n: int(r, 10, 200) * 10, b: pick(r, [16, 32, 64, 128]), e: int(r, 2, 12) }),
  q: v => `$${v.n}$ rows, batch size $${v.b}$, $${v.e}$ epochs. How many optimiser steps run?`,
  answer: v => Math.ceil(v.n / v.b) * v.e, dp: 0,
  steps: v => [`Batches per epoch: $${Math.ceil(v.n / v.b)}$`],
});
export const PtParamsDrill = drill({
  intro: 'A linear layer has one weight per connection and one bias per output.',
  tex: R`(n_{\text{in}}+1)\,n_{\text{out}}`,
  gen: r => ({ i: int(r, 2, 30), h: pick(r, [16, 32, 64]), o: int(r, 2, 10) }),
  q: v => `A network with layers $${v.i}\\to${v.h}\\to${v.o}$, with biases. How many parameters?`,
  answer: v => (v.i + 1) * v.h + (v.h + 1) * v.o, dp: 0,
  steps: v => [`First layer: $${(v.i + 1) * v.h}$`, `Second layer: $${(v.h + 1) * v.o}$`],
});
export const PtClipDrill = drill({
  intro: 'Clipping rescales the whole gradient and keeps its direction.',
  tex: R`g\leftarrow g\cdot\min\Big(1,\frac c{\|g\|}\Big)`,
  gen: r => pick(r, [[3, 4], [6, 8], [5, 12], [8, 15], [9, 12]]).concat(pick(r, [1, 2, 5])),
  q: v => `$g=(${v[0]},${v[1]})$, maximum norm $${v[2]}$. Find the first entry after clipping.`,
  answer: v => (v[0] * v[2]) / Math.hypot(v[0], v[1]), dp: 3,
  steps: v => [`Norm: $${Math.hypot(v[0], v[1])}$`, `Scale: $${fmt(v[2] / Math.hypot(v[0], v[1]), 4)}$`],
});
export const PtAccumDrill = drill({
  intro: 'Accumulating gradients imitates a larger batch using less memory.',
  gen: r => ({ b: pick(r, [4, 8, 16]), a: pick(r, [2, 4, 8]), n: int(r, 8, 40) * 64 }),
  q: v => `Batch $${v.b}$, one update every $${v.a}$ batches, $${v.n}$ rows. How many full updates occur per epoch?`,
  answer: v => Math.floor(Math.ceil(v.n / v.b) / v.a), dp: 0,
  steps: v => [`Batches: $${Math.ceil(v.n / v.b)}$`, `Effective batch size: $${v.a * v.b}$`],
});
export const PtFreezeDrill = drill({
  intro: 'Frozen layers do not train, so only the new head counts.',
  gen: r => ({ b: int(r, 5, 30), h: pick(r, [512, 1024, 2048]), c: int(r, 2, 40) }),
  q: v => `A pretrained backbone of $${v.b}$ million weights is frozen. Its $${v.h}$ features feed a new linear head with $${v.c}$ classes, with bias. How many weights train?`,
  answer: v => (v.h + 1) * v.c, dp: 0,
  steps: v => [`The backbone contributes $0$.`],
});
export const PtScheduleLab = lab({
  intro: 'Three common learning-rate shapes, across one training run.',
  params: [{ k: 'kind', label: 'Schedule', init: 'warm', options: [['step', 'Step decay'], ['cos', 'Cosine'], ['warm', 'Warmup then cosine']] }, { k: 'w', label: 'warmup share', min: 0.02, max: 0.4, step: 0.01, init: 0.1 }, { k: 's', label: 'read at step', min: 0, max: 1000, step: 10, init: 50 }],
  derive: ({ kind, w }) => ({ f: s => { const t = s / 1000; if (kind === 'step') return 0.1 ** Math.floor(t * 3); if (kind === 'cos') return 0.5 * (1 + Math.cos(Math.PI * t)); return t < w ? t / w : 0.5 * (1 + Math.cos((Math.PI * (t - w)) / (1 - w))); } }),
  plots: v => [{ x: [0, 1000], y: [0, 1.05], w: 480, h: 260, xl: 'step', yl: 'share of base rate', label: 'learning rate by step', marks: [{ tone: 1, data: curve(v.f, 0, 1000, 401) }, { as: 'points', tone: 4, r: 5, data: [[v.s, v.f(v.s)]] }] }],
  stats: v => [`learning rate ${fmt(v.f(v.s), 3)} of base`],
  takeaway: 'Warmup protects the first steps from unstable gradients, and decay lets the weights settle.',
});
const lrRun = (lr, c) => { let a = 1, b = 1; for (let i = 0; i < 25; i++) { a -= lr * a; b -= lr * c * b; if (Math.abs(a) + Math.abs(b) > 1e6) return 1e3; } return a * a + c * b * b; };
export const PtLrFinderLab = lab({
  intro: 'Run a short training at many learning rates. Choose a rate on the falling side, well below where the loss blows up.',
  tex: R`w\leftarrow w-\eta\nabla L`,
  params: [{ k: 'e', label: '$\\log_{10}\\eta$', min: -3, max: 1, step: 0.05, init: -1 }, { k: 'c', label: 'steepest curvature', min: 1, max: 50, step: 1, init: 10 }],
  plots: v => [{ x: [-3, 1], y: [-6, 3], w: 480, h: 260, xl: 'log learning rate', yl: 'log loss', label: 'loss after a short run against learning rate', marks: [{ tone: 1, data: curve(e => Math.max(-6, Math.min(3, Math.log10(lrRun(10 ** e, v.c) + 1e-9))), -3, 1, 241) }, { as: 'rule', dash: true, tone: 4, x: v.e }, { as: 'rule', dash: true, tone: 'muted', x: Math.log10(2 / v.c) }] }],
  stats: v => [`loss ${lrRun(10 ** v.e, v.c).toExponential(2)}`, `stability limit ${fmt(2 / v.c, 3)}`],
  takeaway: 'Gradient descent diverges above $2/\\lambda_{\\max}$, so the sharpest direction sets the rate.',
});
const OB = (() => { const r = rng(4); return Array.from({ length: 16 }, (_, i) => { const y = i % 2, s = y ? 1.2 : -1.2; return [r6(s + 0.6 * gauss(r)), r6(s + 0.6 * gauss(r)), y]; }); })();
const SHUF = [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1];
export const PtOverfitBatchLab = lab({
  intro: 'A working model can memorise a tiny batch. If it cannot, there is a bug. Inject one and read the loss.',
  params: [{ k: 'bug', label: 'Bug', init: 'none', options: [['none', 'No bug'], ['shuffle', 'Labels shuffled'], ['nozero', 'No zero_grad'], ['sign', 'Wrong sign']] }, { k: 'lr', label: 'learning rate', min: 0.02, max: 1, step: 0.02, init: 0.3 }],
  derive: ({ bug, lr }) => {
    let w = [0, 0, 0], acc = [0, 0, 0]; const H = [];
    const ys = OB.map((p, i) => (bug === 'shuffle' ? SHUF[i] : p[2]));
    for (let t = 0; t <= 80; t++) {
      const ps = OB.map(p => Math.min(1 - 1e-6, Math.max(1e-6, sigmoid(w[0] + w[1] * p[0] + w[2] * p[1]))));
      H.push([t, r6(mean(ps.map((p, i) => -(ys[i] * Math.log(p) + (1 - ys[i]) * Math.log(1 - p)))))]);
      const g = [0, 1, 2].map(j => mean(OB.map((p, i) => (ps[i] - ys[i]) * (j ? p[j - 1] : 1))));
      acc = bug === 'nozero' ? acc.map((a, j) => a + g[j]) : g;
      w = w.map((x, j) => r6(x + (bug === 'sign' ? 1 : -1) * lr * acc[j]));
    }
    return { H };
  },
  plots: v => [{ x: [0, 80], y: [0, 3], w: 480, h: 260, xl: 'step', yl: 'loss', label: 'loss on one small batch', marks: [{ tone: 1, data: v.H.map(([t, l]) => [t, Math.min(3, l)]) }, { as: 'rule', dash: true, tone: 'muted', y: Math.log(2) }] }],
  stats: v => [`final loss ${fmt(v.H[80][1], 3)}`, `starting loss ${fmt(v.H[0][1], 3)}`],
  takeaway: 'Fit one batch before training on everything. It finds most bugs in a minute.',
});
export const PtLogitsLab = lab({
  intro: 'Computing a probability and then its log loses all precision for confident outputs. Working from logits does not.',
  tex: R`\log(1-\sigma(z))=-\log(1+e^{z})`,
  params: [{ k: 'z', label: 'logit $z$', min: 0, max: 60, step: 0.5, init: 20 }],
  derive: ({ z }) => ({ naive: Math.log(1 - sigmoid(z)), safe: -Math.log1p(Math.exp(z)) }),
  plots: v => [{ x: [0, 60], y: [-65, 0], w: 480, h: 260, xl: 'z', yl: 'log(1 - p)', label: 'stable and naive log probability', marks: [{ tone: 2, data: curve(z => -Math.log1p(Math.exp(z)), 0, 60) }, { tone: 1, data: curve(z => { const a = Math.log(1 - sigmoid(z)); return Number.isFinite(a) ? a : -65; }, 0, 60, 241) }, { as: 'rule', dash: true, tone: 4, x: v.z }] }],
  stats: v => [`from probability ${Number.isFinite(v.naive) ? fmt(v.naive, 4) : 'minus infinity'}`, `from the logit ${fmt(v.safe, 4)}`],
  legend: [['--c1', 'through the probability'], ['--c2', 'from the logit']],
  takeaway: 'Keep the final layer as logits, and use `BCEWithLogitsLoss` or `CrossEntropyLoss`.',
});
export const PytorchCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['`CrossEntropyLoss` expects probabilities.', false, 'It expects raw logits, and applies log-softmax itself.'],
    ['Without `zero_grad`, gradients from earlier steps are added to the new ones.', true, '`backward` accumulates by design.'],
    ['`model.eval()` disables gradient tracking.', false, 'It only changes layers like dropout and batch norm. Use `torch.no_grad()` as well.'],
    ['Moving a model to a GPU after creating its optimiser is always safe.', false, 'Create the optimiser afterwards, because some state depends on the parameters.'],
    ['Saving a `state_dict` is more portable than pickling the model.', true, 'It stores only the tensors, and not the class layout.'],
  ],
  takeaway: 'Most PyTorch bugs are silent, so assert shapes and test on small data.',
});

/* ---------- Architectures ---------- */
export const ArchResidualLab = lab({
  intro: 'A real one-dimensional deep network. Track how a small change at the input survives to each layer.',
  tex: R`\text{plain: }h\leftarrow\tanh(wh)\qquad\text{residual: }h\leftarrow h+\alpha\tanh(wh)`,
  params: [{ k: 'd', label: 'layers', min: 4, max: 100, step: 1, init: 40 }, { k: 'w', label: 'weight $w$', min: 0.3, max: 1.5, step: 0.01, init: 0.8 }, { k: 'a', label: 'residual scale $\\alpha$', min: 0.01, max: 0.3, step: 0.01, init: 0.1 }],
  derive: ({ d, w, a }) => {
    let p = 1, r = 1, gp = 1, gr = 1; const P = [[0, 1]], Q = [[0, 1]];
    for (let i = 1; i <= d; i++) { gp *= w * (1 - Math.tanh(w * p) ** 2); p = Math.tanh(w * p); gr *= 1 + a * w * (1 - Math.tanh(w * r) ** 2); r += a * Math.tanh(w * r); P.push([i, Math.log10(Math.max(1e-30, gp))]); Q.push([i, Math.log10(Math.max(1e-30, gr))]); }
    return { P, Q, gp, gr };
  },
  plots: v => [{ x: [0, v.d], y: [-12, 4], w: 480, h: 260, xl: 'layer', yl: 'log10 gradient', label: 'gradient by layer', marks: [{ tone: 1, data: v.P.map(([x, y]) => [x, Math.max(-12, y)]) }, { tone: 2, data: v.Q.map(([x, y]) => [x, Math.min(4, y)]) }] }],
  stats: v => [`plain ${v.gp.toExponential(1)}`, `residual ${v.gr.toExponential(1)}`],
  legend: [['--c1', 'plain'], ['--c2', 'residual']],
  takeaway: 'The added identity keeps each layer’s derivative near 1, so signals survive great depth.',
});
export const ArchBatchNormLab = lab({
  intro: 'Thirty batches from the same data. Each batch estimates the mean a little differently, and small batches estimate it badly.',
  tex: R`\hat x=\frac{x-\mu_B}{\sigma_B}`,
  params: [{ k: 'n', label: 'batch size', min: 2, max: 128, step: 1, init: 4 }, { k: 'seed', seed: 'New batches', init: 1 }],
  derive: ({ n, seed }) => { const r = rng(seed * 11 + 3); return { M: Array.from({ length: 30 }, () => r6(mean(Array.from({ length: n }, () => 2 + gauss(r))))) }; },
  plots: v => [{ x: [0, 30], y: [0, 4], w: 480, h: 260, xl: 'batch', yl: 'batch mean', label: 'batch means', marks: [{ as: 'rule', dash: true, tone: 'muted', y: 2 }, { as: 'points', tone: 1, r: 4, data: v.M.map((m, i) => [i + 0.5, Math.max(0, Math.min(4, m))]) }] }],
  stats: v => [`spread of batch means ${fmt(std(v.M), 3)}`, `theory ${fmt(1 / Math.sqrt(v.n), 3)}`],
  takeaway: 'With tiny batches the statistics are noise, so switch to group or layer normalisation.',
});
export const ArchSeparableDrill = drill({
  intro: 'One filter per channel, then a cheap mixing step.',
  tex: R`f^2c_{\text{in}}+c_{\text{in}}c_{\text{out}}`,
  gen: r => ({ f: pick(r, [3, 5]), c: pick(r, [16, 32, 64, 128]) }),
  q: v => `A $${v.f}\\times${v.f}$ depthwise separable convolution, $${v.c}$ channels in and out, no bias. How many weights?`,
  answer: v => v.f * v.f * v.c + v.c * v.c, dp: 0,
  steps: v => [`Depthwise: $${v.f * v.f * v.c}$`, `Point-wise: $${v.c * v.c}$`],
});
export const ArchBottleneckDrill = drill({
  intro: 'Shrink the channels, convolve, and restore them.',
  gen: r => ({ c: pick(r, [64, 128, 256]) }),
  q: v => `A bottleneck on $${v.c}$ channels: $1\\times1$ down to $${v.c / 4}$, $3\\times3$ at $${v.c / 4}$, $1\\times1$ back up. No biases. How many weights?`,
  answer: v => v.c * (v.c / 4) * 2 + 9 * (v.c / 4) ** 2, dp: 0,
  steps: v => [`Each $1\\times1$: $${v.c * (v.c / 4)}$`, `The $3\\times3$: $${9 * (v.c / 4) ** 2}$`, `Two plain $3\\times3$ layers would use $${18 * v.c * v.c}$.`],
});
export const ArchReceptiveDrill = drill({
  intro: 'Each layer widens the view by its kernel growth, scaled by the product of earlier strides.',
  tex: R`r=1+\sum_\ell(f_\ell-1)\prod_{j<\ell}s_j`,
  gen: r => Array.from({ length: 3 }, () => [pick(r, [3, 5, 7]), pick(r, [1, 2])]),
  q: v => `Layers as (kernel, stride): ${v.map(l => `$(${l})$`).join(', ')}. Find the receptive field.`,
  answer: v => { let R0 = 1, s = 1; v.forEach(([f, t]) => { R0 += (f - 1) * s; s *= t; }); return R0; }, dp: 0,
  steps: v => { let R0 = 1, s = 1; return v.map(([f, t]) => { R0 += (f - 1) * s; s *= t; return `After this layer: $${R0}$`; }); },
});
export const ArchLstmParamsDrill = drill({
  intro: 'An LSTM has four weight sets, each seeing the input and the hidden state.',
  tex: R`4\big(n(n+m)+n\big)`,
  gen: r => ({ n: pick(r, [32, 64, 128, 256]), m: pick(r, [10, 50, 100]) }),
  q: v => `An LSTM layer with hidden size $${v.n}$ and input size $${v.m}$. How many parameters?`,
  answer: v => 4 * (v.n * (v.n + v.m) + v.n), dp: 0,
  steps: v => [`One gate: $${v.n * (v.n + v.m) + v.n}$`],
});
export const ArchUnetLab = lab({
  intro: 'Each level halves the image and doubles the channels. Skip connections carry the fine detail across.',
  params: [{ k: 's', label: 'image side', min: 32, max: 512, step: 32, init: 256 }, { k: 'l', label: 'levels', min: 1, max: 6, step: 1, init: 4 }, { k: 'w', label: 'base channels', min: 8, max: 128, step: 8, init: 32 }],
  derive: ({ s, l, w }) => ({ T: Array.from({ length: l + 1 }, (_, i) => [s / 2 ** i, w * 2 ** i]) }),
  plots: v => [{ x: [0, v.l * 2 + 1], y: [0, Math.log2(v.s) + 1], w: 480, h: 260, xl: 'path through the network', yl: 'log2 side', label: 'U shape', marks: [{ tone: 1, data: [...v.T.map(([a], i) => [i + 0.5, Math.log2(a)]), ...v.T.slice(0, -1).reverse().map(([a], i) => [v.l + i + 1.5, Math.log2(a)])] }, ...v.T.slice(0, -1).map(([a], i) => ({ tone: 4, dash: true, data: [[i + 0.5, Math.log2(a)], [2 * v.l - i + 0.5, Math.log2(a)]] }))] }],
  table: v => ({ head: ['level', 'side', 'channels'], rows: v.T.map(([a, c], i) => [i ? `level ${i}` : 'top', Number.isInteger(a) ? a : `${fmt(a, 2)} (not whole)`, c]) }),
  takeaway: 'The image side must divide by $2^{\\text{levels}}$, or the skip shapes will not match.',
});
export const ArchRnnPowerLab = lab({
  intro: 'Backpropagation through time multiplies by the same matrix again and again.',
  tex: R`\Big\|\frac{\partial h_t}{\partial h_0}\Big\|\approx\rho^t`,
  params: [{ k: 'p', label: 'spectral radius $\\rho$', min: 0.5, max: 1.3, step: 0.01, init: 0.9 }, { k: 't', label: 'read at step', min: 1, max: 60, step: 1, init: 30 }],
  plots: v => [{ x: [0, 60], y: [-8, 8], w: 480, h: 260, xl: 'steps back', yl: 'log10 gradient', label: 'gradient against time', marks: [{ tone: 1, data: curve(t => Math.max(-8, Math.min(8, t * Math.log10(v.p))), 0, 60) }, { as: 'rule', dash: true, tone: 'muted', y: 0 }, { as: 'rule', dash: true, tone: 4, x: v.t }] }],
  stats: v => [`gradient scale ${(v.p ** v.t).toExponential(2)}`],
  status: v => (v.p < 0.97 ? ['alert', 'Vanishing: early inputs cannot be learned from.'] : v.p > 1.03 ? ['alert', 'Exploding: clip the gradient.'] : ['ok', 'Near the edge, gradients survive.']),
  takeaway: 'Only a narrow band around 1 preserves long-range gradients, which is why gates help.',
});
export const ArchLstmGateLab = lab({
  intro: 'One value is written into the cell at step 5. The forget gate decides how long it survives.',
  tex: R`c_t=f\,c_{t-1}+i\,g_t`,
  params: [{ k: 'f', label: 'forget gate $f$', min: 0.5, max: 1, step: 0.005, init: 0.98 }, { k: 'r', label: 'plain network scale', min: 0.5, max: 1, step: 0.01, init: 0.8 }],
  plots: v => [{ x: [0, 60], y: [0, 1.05], w: 480, h: 260, xl: 'step', yl: 'stored value', label: 'memory over time', marks: [{ tone: 2, data: curve(t => (t < 5 ? 0 : v.f ** (t - 5)), 0, 60, 241) }, { tone: 1, data: curve(t => (t < 5 ? 0 : v.r ** (t - 5)), 0, 60, 241) }] }],
  stats: v => [`after 50 steps: gate ${fmt(v.f ** 50, 3)}`, `plain ${fmt(v.r ** 50, 5)}`],
  legend: [['--c2', 'LSTM cell'], ['--c1', 'plain recurrent state']],
  takeaway: 'A forget gate near 1 keeps the path open, and the network learns when to close it.',
});
export const ArchCompoundLab = lab({
  intro: 'Scale depth, width and resolution together with one number.',
  tex: R`\text{cost}\propto(\alpha\beta^2\gamma^2)^\phi`,
  params: [{ k: 'p', label: 'compound $\\phi$', min: 0, max: 7, step: 0.5, init: 2 }, { k: 'a', label: 'depth $\\alpha$', min: 1, max: 1.5, step: 0.05, init: 1.2 }, { k: 'b', label: 'width $\\beta$', min: 1, max: 1.5, step: 0.05, init: 1.1 }, { k: 'g', label: 'resolution $\\gamma$', min: 1, max: 1.5, step: 0.05, init: 1.15 }],
  derive: ({ a, b, g }) => ({ base: a * b * b * g * g }),
  plots: v => [{ x: [0, 7], y: [0, 40], w: 480, h: 260, xl: 'φ', yl: 'cost multiple', label: 'cost against compound scale', marks: [{ tone: 1, data: curve(p => Math.min(40, v.base ** p), 0, 7) }, { as: 'points', tone: 4, r: 5, data: [[v.p, Math.min(40, v.base ** v.p)]] }] }],
  stats: v => [`cost × ${fmt(v.base ** v.p, 2)}`, `depth × ${fmt(v.a ** v.p, 2)}`, `width × ${fmt(v.b ** v.p, 2)}`, `side × ${fmt(v.g ** v.p, 2)}`],
  takeaway: 'Width and resolution cost quadratically, and depth only linearly.',
});
export const ArchCheckerboardLab = lab({
  intro: 'A transposed convolution stamps its kernel at each stride. Count how many stamps cover each output pixel.',
  params: [{ k: 'f', label: 'kernel size', min: 1, max: 8, step: 1, init: 3 }, { k: 's', label: 'stride', min: 1, max: 4, step: 1, init: 2 }],
  derive: ({ f, s }) => { const c = Array(40).fill(0); for (let p = 0; p + f <= 40; p += s) for (let j = 0; j < f; j++) c[p + j]++; return { c: c.slice(8, 32) }; },
  plots: v => [{ x: [0, 24], y: [0, Math.max(...v.c) + 1], w: 480, h: 240, xl: 'output pixel', yl: 'stamps', label: 'coverage', marks: [{ as: 'rects', tone: 1, data: v.c.map((n, i) => [i + 0.05, 0, i + 0.95, n]) }] }],
  status: v => (Math.max(...v.c) === Math.min(...v.c) ? ['ok', 'Even coverage: no checkerboard.'] : ['alert', 'Uneven coverage: expect a grid pattern.']),
  takeaway: 'Coverage is even when the stride divides the kernel size. Otherwise upsample by interpolation, then convolve.',
});
export const ArchitectureCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Batch norm behaves identically at training and test time.', false, 'Test time uses running averages.'],
    ['Layer norm is unaffected by batch size.', true, 'It normalises each example separately.'],
    ['Residual networks fix overfitting.', false, 'They address optimisation of very deep networks.'],
    ['U-Net skips add feature maps to the bottleneck.', false, 'They join encoder and decoder at the matching resolution.'],
    ['An LSTM has separate hidden and cell states.', true, 'A GRU merges them.'],
  ],
  takeaway: 'Each architecture idea fixes one specific failure, so name the failure first.',
});

/* ---------- Language models ---------- */
const BPE_TEXT = 'low lower lowest slow slower newest widest';
const bpe = n => {
  let t = [...BPE_TEXT]; const M = [];
  for (let k = 0; k < n; k++) {
    const c = new Map(); for (let i = 0; i + 1 < t.length; i++) if (t[i] !== ' ' && t[i + 1] !== ' ') { const key = `${t[i]}\u0001${t[i + 1]}`; c.set(key, (c.get(key) ?? 0) + 1); }
    if (!c.size) break;
    const [key, f] = [...c.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0], [a, b] = key.split('\u0001'), out = [];
    for (let i = 0; i < t.length; i++) { if (t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]); }
    t = out; M.push([a + b, f]);
  }
  return { t, M };
};
export const LlmBpeLab = lab({
  intro: 'Byte-pair encoding repeatedly merges the most frequent neighbouring pair. Spaces are kept separate here so the tokens read clearly.',
  params: [{ k: 'n', label: 'merges', min: 0, max: 16, step: 1, init: 4, play: true }],
  derive: ({ n }) => bpe(n),
  plots: () => [{ x: [0, 16], y: [0, BPE_TEXT.length], w: 480, h: 220, xl: 'merges', yl: 'tokens', label: 'tokens against merges', marks: [{ tone: 1, data: Array.from({ length: 17 }, (_, n) => [n, bpe(n).t.length]) }] }],
  stats: v => [`${v.t.length} tokens`],
  table: v => ({ head: ['text'], rows: [[v.t.filter(x => x !== ' ').join(' · ')]] }),
  takeaway: 'Common fragments become single tokens, so text shrinks as the vocabulary grows.',
});
const LOG = [3.2, 2.7, 2.1, 1.6, 0.9, 0.2, -0.6, -1.5];
export const LlmSamplingLab = lab({
  intro: 'One set of scores for eight candidate tokens. Change how the next token is sampled.',
  tex: R`p_i\propto e^{z_i/T}`,
  params: [{ k: 'T', label: 'temperature $T$', min: 0.1, max: 3, step: 0.05, init: 1 }, { k: 'k', label: 'top-k', min: 1, max: 8, step: 1, init: 8 }, { k: 'p', label: 'top-p', min: 0.1, max: 1, step: 0.01, init: 1 }],
  derive: ({ T, k, p }) => {
    let P = softmaxT(LOG, T).map((x, i) => (i < k ? x : 0)); const s1 = P.reduce((a, b) => a + b); P = P.map(x => x / s1);
    let c = 0, stop = false; P = P.map(x => { if (stop) return 0; c += x; if (c >= p - 1e-9) stop = true; return x; }); const s2 = P.reduce((a, b) => a + b); P = P.map(x => x / s2);
    return { P, H: -P.reduce((a, x) => a + (x > 0 ? x * Math.log2(x) : 0), 0) };
  },
  plots: v => [{ x: [0, 8], y: [0, 1], w: 480, h: 260, xl: 'token, most likely first', yl: 'probability', label: 'sampling distribution', marks: [{ as: 'rects', tone: 1, data: v.P.map((x, i) => [i + 0.1, 0, i + 0.9, x]) }] }],
  stats: v => [`entropy ${fmt(v.H, 3)} bits`, `${v.P.filter(x => x > 0).length} tokens possible`],
  takeaway: 'Low temperature is safe and repetitive. High temperature is varied and error-prone.',
});
export const LlmEmbedDrill = drill({
  intro: 'A token table and a position table, both $d$ wide.',
  tex: R`(V+B)\,d`,
  gen: r => ({ V: pick(r, [100, 1000, 50257]), B: pick(r, [128, 256, 1024]), d: pick(r, [192, 384, 768]) }),
  q: v => `Vocabulary $${v.V}$, block size $${v.B}$, width $${v.d}$. How many weights do the two embedding tables have?`,
  answer: v => (v.V + v.B) * v.d, dp: 0,
  steps: v => [`Tokens: $${v.V * v.d}$`, `Positions: $${v.B * v.d}$`],
});
export const LlmLoraDrill = drill({
  intro: 'LoRA trains a low-rank update and keeps the original matrix frozen.',
  tex: R`r(m+n)\quad\text{against}\quad mn`,
  gen: r => ({ m: pick(r, [768, 1024, 4096]), n: pick(r, [768, 1024]), r: pick(r, [4, 8, 16]) }),
  q: v => `A layer of $${v.m}\\times${v.n}$ weights, LoRA rank $${v.r}$. What percentage of the full weights are trained?`,
  answer: v => (100 * v.r * (v.m + v.n)) / (v.m * v.n), dp: 2,
  steps: v => [`LoRA: $${v.r * (v.m + v.n)}$`, `Full: $${v.m * v.n}$`],
});
export const LlmQuantDrill = drill({
  intro: 'Memory is the weight count times the bits per weight.',
  tex: R`\text{GB}=\frac{P\times\text{bits}}{8\times10^9}`,
  gen: r => ({ p: pick(r, [7, 13, 70]), b: pick(r, [32, 16, 8, 4]) }),
  q: v => `A model with $${v.p}$ billion weights stored at $${v.b}$ bits. How many gigabytes?`,
  answer: v => (v.p * v.b) / 8, dp: 1,
  steps: v => [`Bytes per weight: $${v.b / 8}$`],
});
export const LlmCacheDrill = drill({
  intro: 'Without a cache, each new token recomputes attention for the whole prefix.',
  tex: R`\sum_{t=1}^Nt^2\quad\text{against}\quad\sum_{t=1}^Nt`,
  gen: r => ({ n: pick(r, [10, 20, 50, 100]) }),
  q: v => `Generating $${v.n}$ tokens, counting attention scores. By what factor does a key-value cache reduce the work?`,
  answer: v => (2 * v.n + 1) / 3, dp: 2,
  steps: v => [`Without: $${(v.n * (v.n + 1) * (2 * v.n + 1)) / 6}$`, `With: $${(v.n * (v.n + 1)) / 2}$`],
});
export const LlmChunkDrill = drill({
  intro: 'Overlap repeats text at each boundary, so each chunk advances by less than its length.',
  tex: R`\Big\lceil\frac{L-o}{c-o}\Big\rceil`,
  gen: r => ({ L: int(r, 20, 90) * 100, c: pick(r, [300, 500, 800]), o: pick(r, [50, 100]) }),
  q: v => `A document of $${v.L}$ tokens, chunks of $${v.c}$ with overlap $${v.o}$. How many chunks?`,
  answer: v => Math.ceil((v.L - v.o) / (v.c - v.o)), dp: 0,
  steps: v => [`Each chunk advances $${v.c - v.o}$.`],
});
export const LlmMrrDrill = drill({
  intro: 'Reciprocal rank rewards placing the first correct passage early.',
  tex: R`\text{MRR}=\frac1Q\sum_q\frac1{\text{rank}_q}`,
  gen: r => Array.from({ length: 3 }, () => int(r, 1, 5)),
  q: v => `The first relevant passage appears at ranks $${v.join(', ')}$ for three queries. Find the MRR.`,
  answer: v => mean(v.map(x => 1 / x)), dp: 3,
  steps: v => [`Reciprocals: $${v.map(x => fmt(1 / x, 3)).join(', ')}$`],
});
const DOCS = [[15, true], [40, true], [75, false], [120, false], [200, false], [310, false]];
export const LlmRetrieveLab = lab({
  intro: 'Documents as unit vectors. Turn the query, and take the nearest passages by cosine similarity.',
  tex: R`\cos\theta=\frac{q\cdot d}{\|q\|\|d\|}`,
  params: [{ k: 'a', label: 'query angle', min: 0, max: 359, step: 1, init: 30 }, { k: 'k', label: 'passages kept $k$', min: 1, max: 6, step: 1, init: 2 }],
  derive: ({ a, k }) => { const R0 = DOCS.map(([d, rel], i) => ({ i, d, rel, c: Math.cos(((a - d) * Math.PI) / 180) })).sort((x, y) => y.c - x.c), top = R0.slice(0, k); return { R0, top, rec: top.filter(x => x.rel).length / 2, prec: top.filter(x => x.rel).length / k }; },
  plots: v => [{ x: [-1.3, 1.3], y: [-1.3, 1.3], w: 340, h: 320, label: 'query and document vectors', marks: [...DOCS.map(([d, rel]) => ({ as: 'arrow', tone: v.top.some(t => t.d === d) ? 2 : 'muted', data: [[0, 0, Math.cos((d * Math.PI) / 180), Math.sin((d * Math.PI) / 180)]] })), { as: 'arrow', tone: 4, data: [[0, 0, 1.2 * Math.cos((v.a * Math.PI) / 180), 1.2 * Math.sin((v.a * Math.PI) / 180)]] }] }],
  stats: v => [`recall ${pct(v.rec, 0)}`, `precision ${pct(v.prec, 0)}`],
  table: v => ({ head: ['rank', 'cosine', 'relevant'], rows: v.top.map((t, i) => [i + 1, fmt(t.c, 3), t.rel ? 'yes' : 'no']) }),
  takeaway: 'The first two documents are the relevant ones. Larger $k$ raises recall, and lowers precision.',
});
export const LlmEvalSizeLab = lab({
  intro: 'A score on a small test set is uncertain. How many questions do you need to tell two systems apart?',
  tex: R`\text{margin}=1.96\sqrt{\frac{p(1-p)}n}`,
  params: [{ k: 'p', label: 'accuracy', min: 0.5, max: 0.99, step: 0.01, init: 0.8 }, { k: 'n', label: 'questions', min: 10, max: 2000, step: 10, init: 100 }],
  plots: v => [{ x: [10, 2000], y: [0, 0.25], w: 480, h: 260, xl: 'questions', yl: 'margin', label: 'margin of error', marks: [{ tone: 1, data: curve(n => Math.min(0.25, 1.96 * Math.sqrt((v.p * (1 - v.p)) / n)), 10, 2000) }, { as: 'rule', dash: true, tone: 4, x: v.n }] }],
  stats: v => [`±${pct(1.96 * Math.sqrt((v.p * (1 - v.p)) / v.n), 1)}`],
  takeaway: 'Quadrupling the questions only halves the margin, so small differences need large sets.',
});
export const LlmCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A model’s context length limits its input plus output.', true, 'Both occupy the same window.'],
    ['Retrieved passages can safely be treated as instructions.', false, 'They may contain hidden instructions, so treat them as data.'],
    ['LoRA changes the original weights.', false, 'It freezes them and learns a separate low-rank update.'],
    ['Lower temperature makes the top token more likely.', true, 'Dividing by a small $T$ widens the gaps.'],
    ['Tokens correspond one-to-one with words.', false, 'A word may be several tokens.'],
  ],
  takeaway: 'Treat language models as probabilistic components, and evaluate them like any other model.',
});

/* ---------- Diffusion in depth: exact score of a two-component mixture ---------- */
const T = 200, ABAR = (() => { let a = 1; return Array.from({ length: T }, (_, i) => { a *= 1 - (1e-4 + ((0.1 - 1e-4) * i) / (T - 1)); return a; }); })();
const COMP = [[-2, 0.5], [2, 0.5]], S0 = 0.3;
const epsAt = (x, a, comps) => {
  let tot = 0, s = 0;
  comps.forEach(([m, w]) => { const v = a * S0 * S0 + 1 - a, d = x - Math.sqrt(a) * m, p = (w * Math.exp((-d * d) / (2 * v))) / Math.sqrt(v); tot += p; s += p * (-d / v); });
  return -Math.sqrt(1 - a) * (s / (tot || 1e-300));
};
const ddim = (n, steps, seed, eta, eps) => {
  const r = rng(seed * 17 + 5), ts = linspace(T - 1, 0, steps).map(Math.round);
  return Array.from({ length: n }, () => {
    let x = gauss(r);
    ts.forEach((t, k) => {
      const a = ABAR[t], a1 = k + 1 < steps ? ABAR[ts[k + 1]] : 1, e = eps(x, a), x0 = Math.max(-8, Math.min(8, (x - Math.sqrt(1 - a) * e) / Math.sqrt(a)));
      const sig = eta * Math.sqrt(((1 - a1) / (1 - a)) * (1 - a / a1));
      x = r6(Math.sqrt(a1) * x0 + Math.sqrt(Math.max(0, 1 - a1 - sig * sig)) * e + sig * gauss(r));
    });
    return x;
  });
};
export const GdDdimLab = lab({
  intro: 'Sample using the exact noise predictor for a known two-peaked distribution, so only the sampler is being tested.',
  tex: R`x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\varepsilon`,
  params: [{ k: 'n', label: 'sampling steps', min: 2, max: 100, step: 1, init: 20 }, { k: 'eta', label: 'added randomness $\\eta$', min: 0, max: 1, step: 0.05, init: 0 }, { k: 'seed', seed: 'New samples', init: 1 }],
  derive: ({ n, eta, seed }) => ({ X: ddim(600, n, seed, eta, (x, a) => epsAt(x, a, COMP)) }),
  plots: v => [{ x: [-5, 5], y: [0, 160], w: 480, h: 260, xl: 'x', yl: 'count', label: 'generated samples', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.X, -5, 5, 50).map(b => [b[0], 0, b[2], Math.min(160, b[3])]) }, { as: 'rule', dash: true, tone: 4, x: -2 }, { as: 'rule', dash: true, tone: 4, x: 2 }] }],
  stats: v => [`spread around each peak ${fmt(mean(v.X.map(x => (Math.abs(x) - 2) ** 2)) ** 0.5, 3)}`, 'target 0.300'],
  takeaway: 'A few well-chosen steps recover both peaks without retraining anything.',
});
export const GdCfgLab = lab({
  intro: 'The condition selects the right-hand peak. Larger guidance pushes samples further in that direction.',
  tex: R`\hat\varepsilon=\varepsilon_\varnothing+w(\varepsilon_c-\varepsilon_\varnothing)`,
  params: [{ k: 'w', label: 'guidance $w$', min: 0, max: 8, step: 0.1, init: 1 }, { k: 'seed', seed: 'New samples', init: 1 }],
  derive: ({ w, seed }) => ({ X: ddim(600, 40, seed, 0, (x, a) => { const u = epsAt(x, a, COMP), c = epsAt(x, a, [[2, 1]]); return u + w * (c - u); }) }),
  plots: v => [{ x: [-5, 7], y: [0, 240], w: 480, h: 260, xl: 'x', yl: 'count', label: 'guided samples', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.X, -5, 7, 60).map(b => [b[0], 0, b[2], Math.min(240, b[3])]) }, { as: 'rule', dash: true, tone: 4, x: 2 }] }],
  stats: v => [`on the right ${pct(v.X.filter(x => x > 0).length / 600, 0)}`, `mean ${fmt(mean(v.X), 2)}`, `spread ${fmt(std(v.X), 3)}`],
  takeaway: '$w=0$ ignores the condition, $w=1$ follows it, and larger values overshoot and lose variety.',
});
const flowV = (x, t) => { let tot = 0, s = 0; COMP.forEach(([m, w]) => { const S = (1 - t) ** 2 * S0 * S0 + t * t, d = x - (1 - t) * m, p = (w * Math.exp((-d * d) / (2 * S))) / Math.sqrt(S), x0 = m + ((1 - t) * S0 * S0 * d) / S; tot += p; s += p * ((x - x0) / t); }); return s / (tot || 1e-300); };
export const GdFlowLab = lab({
  intro: 'Noise moves to data along a velocity field. Here the exact field is known, so the error comes only from the number of steps.',
  tex: R`\frac{dx}{dt}=v(x,t),\qquad x_t=(1-t)x_0+tx_1`,
  params: [{ k: 'n', label: 'Euler steps', min: 1, max: 40, step: 1, init: 6 }, { k: 'seed', seed: 'New noise', init: 1 }],
  derive: ({ n, seed }) => { const r = rng(seed * 23 + 2), P = Array.from({ length: 200 }, () => { let x = gauss(r); const path = [[1, r6(x)]]; for (let i = n; i >= 1; i--) { const t = i / n; x = r6(x - (1 / n) * flowV(x, t)); path.push([(i - 1) / n, x]); } return path; }); return { P, end: P.map(p => p[p.length - 1][1]) }; },
  plots: v => [{ x: [0, 1], y: [-4, 4], w: 480, h: 280, xl: 't (0 is data, 1 is noise)', yl: 'x', label: 'sample paths', marks: v.P.slice(0, 40).map(p => ({ tone: 1, data: p })) }],
  stats: v => [`spread around each peak ${fmt(mean(v.end.map(x => (Math.abs(x) - 2) ** 2)) ** 0.5, 3)}`, 'target 0.300'],
  takeaway: 'Straighter paths tolerate coarser steps, which is the appeal of flow matching.',
});
export const GdX0Drill = drill({
  intro: 'Rearrange the forward equation to estimate the clean sample.',
  tex: R`\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}`,
  gen: r => ({ a: pick(r, [0.81, 0.64, 0.36, 0.25]), x: int(r, -3, 3), e: int(r, -2, 2) }),
  q: v => `$\\bar\\alpha_t=${v.a}$, $x_t=${v.x}$, $\\varepsilon=${v.e}$. Find $\\hat x_0$.`,
  answer: v => (v.x - Math.sqrt(1 - v.a) * v.e) / Math.sqrt(v.a), dp: 3,
  steps: v => [`$\\sqrt{\\bar\\alpha_t}=${fmt(Math.sqrt(v.a), 3)}$`, `$\\sqrt{1-\\bar\\alpha_t}=${fmt(Math.sqrt(1 - v.a), 3)}$`],
});
export const GdGuidanceDrill = drill({
  intro: 'Extrapolate from the unconditional prediction, past the conditional one.',
  tex: R`\hat\varepsilon=\varepsilon_\varnothing+w(\varepsilon_c-\varepsilon_\varnothing)`,
  gen: r => ({ u: int(r, -3, 3), c: int(r, -3, 3), w: pick(r, [1.5, 3, 7.5]) }),
  q: v => `$\\varepsilon_\\varnothing=${v.u}$, $\\varepsilon_c=${v.c}$, $w=${v.w}$. Find $\\hat\\varepsilon$.`,
  answer: v => v.u + v.w * (v.c - v.u), dp: 2,
  steps: v => [`Difference: $${v.c - v.u}$`],
});
export const GdSnrDrill = drill({
  intro: 'The signal-to-noise ratio falls as the forward process runs.',
  tex: R`\text{SNR}=\frac{\bar\alpha_t}{1-\bar\alpha_t}`,
  gen: r => ({ a: pick(r, [0.9, 0.75, 0.5, 0.2, 0.1]) }),
  q: v => `$\\bar\\alpha_t=${v.a}$. Find the signal-to-noise ratio.`,
  answer: v => v.a / (1 - v.a), dp: 3,
  steps: v => [`Noise share: $${fmt(1 - v.a, 2)}$`],
});
export const GdLatentDrill = drill({
  intro: 'Latent diffusion saves work by compressing first.',
  gen: r => ({ s: pick(r, [256, 512, 1024]), f: pick(r, [4, 8]), c: pick(r, [3, 4]) }),
  q: v => `A $${v.s}\\times${v.s}\\times3$ image, downsampled by $${v.f}$ into $${v.c}$ latent channels. How many times fewer numbers are there?`,
  answer: v => (3 * v.f * v.f) / v.c, dp: 2,
  steps: v => [`Latent shape: $${v.s / v.f}\\times${v.s / v.f}\\times${v.c}$`],
});
export const DiffusionDepthCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['DDIM requires a differently trained network.', false, 'It uses the same noise predictor and a different update.'],
    ['Predicting noise is equivalent to estimating the score.', true, 'They differ by a known scale.'],
    ['Classifier-free guidance trains a separate classifier.', false, 'It drops the condition during training instead.'],
    ['Latent diffusion runs the noise process on pixels.', false, 'It runs in the autoencoder’s compressed space.'],
    ['Flow matching integrates a learned velocity field.', true, 'Sampling solves an ordinary differential equation.'],
  ],
  takeaway: 'Noise, score and velocity are three views of one generative process.',
});

/* ---------- Graphs, self-supervision and control ---------- */
const GRAPHS = { path: Array.from({ length: 9 }, (_, i) => [i - 1, i + 1].filter(j => j >= 0 && j < 9)), ring: Array.from({ length: 9 }, (_, i) => [(i + 8) % 9, (i + 1) % 9]), star: Array.from({ length: 9 }, (_, i) => (i ? [0] : [1, 2, 3, 4, 5, 6, 7, 8])) };
export const GsMessageLab = lab({
  intro: 'One node starts with a signal. Each round, every node averages itself with its neighbours.',
  tex: R`h_v\leftarrow\frac{h_v+\sum_{u\in N(v)}h_u}{1+|N(v)|}`,
  params: [{ k: 'g', label: 'Graph', init: 'path', options: [['path', 'Path'], ['ring', 'Ring'], ['star', 'Star']] }, { k: 'n', label: 'rounds', min: 0, max: 40, step: 1, init: 2, play: true }],
  derive: ({ g, n }) => { const A = GRAPHS[g]; let h = A.map((_, i) => (i === 1 ? 1 : 0)); for (let k = 0; k < n; k++) h = h.map((x, i) => (x + A[i].reduce((s, u) => s + h[u], 0)) / (1 + A[i].length)); return { h }; },
  plots: v => [{ x: [0, 9], y: [0, 1], w: 480, h: 260, xl: 'node', yl: 'value', label: 'node values', marks: [{ as: 'rects', tone: 1, data: v.h.map((x, i) => [i + 0.1, 0, i + 0.9, x]) }] }],
  stats: v => [`differences between nodes ${fmt(std(v.h), 4)}`, `${v.h.filter(x => x > 1e-9).length} nodes reached`],
  takeaway: 'Information travels one hop per round, and after many rounds all nodes look alike.',
});
export const GsHopsDrill = drill({
  intro: 'After $k$ rounds, a node has heard from everything within $k$ hops.',
  gen: r => ({ n: int(r, 5, 20), k: int(r, 1, 8) }),
  q: v => `A path of $${v.n}$ nodes. After $${v.k}$ rounds, how many nodes have influenced the end node, including itself?`,
  answer: v => Math.min(v.n, v.k + 1), dp: 0,
  steps: v => [`Reach: $${v.k}$ hops`],
});
export const GsGcnDrill = drill({
  intro: 'With self-loops and a simple mean, a node averages its neighbourhood.',
  gen: r => ({ h: int(r, -3, 6), n: Array.from({ length: int(r, 2, 4) }, () => int(r, -3, 6)) }),
  q: v => `A node has value $${v.h}$, and its neighbours have values $${v.n.join(', ')}$. Find its new value.`,
  answer: v => (v.h + v.n.reduce((a, b) => a + b, 0)) / (1 + v.n.length), dp: 3,
  steps: v => [`Total: $${v.h + v.n.reduce((a, b) => a + b, 0)}$`, `Count: $${1 + v.n.length}$`],
});
export const GsAttentionDrill = drill({
  intro: 'Graph attention normalises the scores across the neighbours.',
  tex: R`\alpha_u=\frac{e^{s_u}}{\sum_ke^{s_k}}`,
  gen: r => ({ s: [int(r, -2, 3), int(r, -2, 3), int(r, -2, 3)], k: int(r, 0, 2) }),
  q: v => `Neighbour scores $(${v.s})$. Find the weight of neighbour number ${v.k + 1}.`,
  answer: v => softmaxT(v.s)[v.k], dp: 3,
  steps: v => [`Weights: $(${softmaxT(v.s).map(p => fmt(p, 3))})$`],
});
export const GsInfoNceLab = lab({
  intro: 'One positive pair, and a crowd of negatives. Temperature sets how sharply the loss punishes close negatives.',
  tex: R`L=-\log\frac{e^{s_+/\tau}}{e^{s_+/\tau}+Ke^{s_-/\tau}}`,
  params: [{ k: 'p', label: 'positive similarity', min: -1, max: 1, step: 0.01, init: 0.8 }, { k: 'n', label: 'negative similarity', min: -1, max: 1, step: 0.01, init: 0.2 }, { k: 'K', label: 'negatives $K$', min: 1, max: 512, step: 1, init: 64 }, { k: 't', label: 'temperature $\\tau$', min: 0.02, max: 1, step: 0.01, init: 0.2 }],
  derive: ({ p, n, K }) => ({ f: t => Math.log1p(K * Math.exp((n - p) / t)) }),
  plots: v => [{ x: [0.02, 1], y: [0, 8], w: 480, h: 260, xl: 'τ', yl: 'loss', label: 'loss against temperature', marks: [{ tone: 1, data: curve(t => Math.min(8, v.f(t)), 0.02, 1) }, { as: 'rule', dash: true, tone: 4, x: v.t }] }],
  stats: v => [`loss ${fmt(v.f(v.t), 3)}`, `chance level ${fmt(Math.log(v.K + 1), 3)}`],
  takeaway: 'More negatives raise the chance-level loss, so a loss is only meaningful alongside its $K$.',
});
export const GsMaskDrill = drill({
  intro: 'A masked autoencoder encodes only the patches it can see.',
  gen: r => ({ s: pick(r, [224, 256, 384]), p: 16, m: pick(r, [0.5, 0.75, 0.9]) }),
  q: v => `A $${v.s}\\times${v.s}$ image, patches of $${v.p}$, mask ratio $${v.m}$. How many patches are visible?`,
  answer: v => Math.round((v.s / v.p) ** 2 * (1 - v.m)), dp: 0,
  steps: v => [`Total patches: $${(v.s / v.p) ** 2}$`],
});
export const GsReturnDrill = drill({
  intro: 'A return sums rewards, discounting those further away.',
  tex: R`G_t=\sum_k\gamma^kr_{t+k}`,
  gen: r => ({ r: Array.from({ length: 3 }, () => int(r, -2, 5)), g: pick(r, [0.5, 0.9, 0.99]) }),
  q: v => `Rewards $${v.r.join(', ')}$, $\\gamma=${v.g}$. Find the return from the first step.`,
  answer: v => v.r.reduce((s, x, i) => s + x * v.g ** i, 0), dp: 3,
  steps: v => v.r.map((x, i) => `Term ${i + 1}: $${fmt(x * v.g ** i, 3)}$`),
});
export const GsAdvantageDrill = drill({
  intro: 'The advantage measures how much better the outcome was than expected.',
  tex: R`A=r+\gamma V(s')-V(s)`,
  gen: r => ({ r: int(r, -2, 5), g: pick(r, [0.9, 0.99]), a: int(r, 0, 9), b: int(r, 0, 9) }),
  q: v => `$r=${v.r}$, $\\gamma=${v.g}$, $V(s')=${v.a}$, $V(s)=${v.b}$. Find the advantage.`,
  answer: v => v.r + v.g * v.a - v.b, dp: 3,
  steps: v => [`Target: $${fmt(v.r + v.g * v.a, 3)}$`],
});
export const GsPpoLab = lab({
  intro: 'PPO’s objective stops improving once the probability ratio moves too far.',
  tex: R`\min\big(rA,\ \text{clip}(r,1-\epsilon,1+\epsilon)A\big)`,
  params: [{ k: 'A', label: 'advantage $A$', min: -2, max: 2, step: 0.05, init: 1 }, { k: 'e', label: '$\\epsilon$', min: 0.05, max: 0.5, step: 0.01, init: 0.2 }, { k: 'r', label: 'ratio $r$', min: 0, max: 2.5, step: 0.01, init: 1.1 }],
  derive: ({ A, e }) => ({ f: r => Math.min(r * A, Math.max(1 - e, Math.min(1 + e, r)) * A) }),
  plots: v => [{ x: [0, 2.5], y: [-5, 5], w: 480, h: 260, xl: 'probability ratio', yl: 'objective', label: 'objective against ratio', marks: [{ tone: 'muted', dash: true, data: curve(r => r * v.A, 0, 2.5) }, { tone: 1, data: curve(v.f, 0, 2.5, 301) }, { as: 'points', tone: 4, r: 5, data: [[v.r, v.f(v.r)]] }] }],
  stats: v => [`objective ${fmt(v.f(v.r), 3)}`, `unclipped ${fmt(v.r * v.A, 3)}`],
  takeaway: 'Beyond the clip, moving further earns nothing, which keeps each update modest.',
});
export const GsReinforceLab = lab({
  intro: 'A real REINFORCE learner choosing between two arms, averaged across runs. All rewards share a large offset, which a baseline removes.',
  tex: R`\theta\leftarrow\theta+\eta(R-b)\nabla\log\pi(a)`,
  params: [{ k: 'b', label: 'Baseline', init: 'on', options: [['on', 'Running mean'], ['off', 'None']] }, { k: 'o', label: 'reward offset', min: 0, max: 20, step: 1, init: 10 }, { k: 'lr', label: 'learning rate', min: 0.005, max: 0.2, step: 0.005, init: 0.05 }, { k: 'seed', seed: 'New runs', init: 1 }],
  derive: ({ b, o, lr, seed }) => {
    const r = rng(seed * 31 + 7), E = 200, S = Array(E).fill(0);
    for (let run = 0; run < 40; run++) { let th = 0, base = o; for (let e = 0; e < E; e++) { const p = sigmoid(th); S[e] += p; const a = r() < p ? 1 : 0, R0 = o + (a ? 1 : 0) + gauss(r), u = R0 - (b === 'on' ? base : 0); th = r6(Math.max(-30, Math.min(30, th + lr * u * (a - p)))); base = r6(base + 0.05 * (R0 - base)); } }
    return { H: S.map((s, e) => [e, s / 40]) };
  },
  plots: v => [{ x: [0, 200], y: [0, 1], w: 480, h: 260, xl: 'episode', yl: 'P(better arm)', label: 'probability of the better arm', marks: [{ as: 'rule', dash: true, tone: 'muted', y: 0.5 }, { tone: 1, data: v.H }] }],
  stats: v => [`final ${pct(v.H[199][1], 1)}`],
  takeaway: 'Without a baseline, a large offset makes updates huge and noisy. Subtracting one leaves only the signal.',
});
export const GsDoubleDqnLab = lab({
  intro: 'Every action is truly worth 0, and estimates are noisy. The max of noisy estimates is biased upward. Selecting with one estimate and evaluating with another is not.',
  tex: R`\mathbb E\big[\max_aQ(a)\big]\ge\max_a\mathbb E[Q(a)]`,
  params: [{ k: 'n', label: 'actions', min: 1, max: 40, step: 1, init: 8 }, { k: 's', label: 'noise', min: 0.1, max: 3, step: 0.1, init: 1 }, { k: 'seed', seed: 'Resample', init: 1 }],
  derive: ({ n, s, seed }) => { const r = rng(seed * 5 + 9); let m = 0, d = 0; const N = 1500; for (let i = 0; i < N; i++) { const a = Array.from({ length: n }, () => s * gauss(r)), b = Array.from({ length: n }, () => s * gauss(r)), j = a.indexOf(Math.max(...a)); m += a[j]; d += b[j]; } return { m: m / N, d: d / N }; },
  plots: v => [{ x: [0, 2], y: [-1, 7], w: 400, h: 260, label: 'bias of each estimator', marks: [{ as: 'rects', tone: 1, data: [[0.15, 0, 0.85, v.m]] }, { as: 'rects', tone: 2, data: [[1.15, 0, 1.85, v.d]] }, { as: 'rule', tone: 'muted', y: 0 }] }],
  stats: v => [`single estimator ${fmt(v.m, 3)}`, `double estimator ${fmt(v.d, 3)}`],
  legend: [['--c1', 'max of one estimate'], ['--c2', 'double estimate']],
  takeaway: 'More actions and more noise mean more overestimation, and double Q-learning removes it.',
});
export const GraphsControlCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A transformer is attention over a fully connected graph.', true, 'Every token is a neighbour of every other.'],
    ['Adding many graph layers always helps.', false, 'Over-smoothing makes the nodes alike.'],
    ['A linear probe trains the backbone.', false, 'The backbone is frozen.'],
    ['Policy gradients need a model of the environment.', false, 'The dynamics drop out of the gradient.'],
    ['A baseline biases the policy gradient.', false, 'Its expected contribution is zero.'],
  ],
  takeaway: 'Each method changes where the signal comes from: neighbours, augmentations or rewards.',
});

/* ---------- Interviews and system design ---------- */
export const CrBaseRateLab = lab({
  intro: 'A good test for a rare condition. Most positive results can still be wrong.',
  tex: R`P(D\mid+)=\frac{s\pi}{s\pi+(1-c)(1-\pi)}`,
  params: [{ k: 'p', label: 'prevalence $\\pi$', min: 0.001, max: 0.5, step: 0.001, init: 0.01 }, { k: 's', label: 'sensitivity $s$', min: 0.5, max: 1, step: 0.01, init: 0.95 }, { k: 'c', label: 'specificity $c$', min: 0.5, max: 0.999, step: 0.001, init: 0.95 }],
  derive: ({ s, c }) => ({ f: p => (s * p) / (s * p + (1 - c) * (1 - p)) }),
  plots: v => [{ x: [0, 0.5], y: [0, 1], w: 480, h: 260, xl: 'prevalence', yl: 'P(condition | positive)', label: 'positive predictive value', marks: [{ tone: 1, data: curve(v.f, 0.0005, 0.5, 301) }, { as: 'points', tone: 4, r: 5, data: [[v.p, v.f(v.p)]] }] }],
  stats: v => [`P(condition | positive) ${pct(v.f(v.p), 1)}`, `always predicting negative: accuracy ${pct(1 - v.p, 1)}`],
  takeaway: 'Always state the base rate before quoting any accuracy or precision.',
});
export const CrCapacityDrill = drill({
  intro: 'Back-of-envelope capacity estimates come up in almost every system design interview.',
  tex: R`\text{cores}=\frac{\text{QPS}\times\text{seconds per request}}{\text{target utilisation}}`,
  gen: r => ({ q: int(r, 1, 20) * 500, ms: pick(r, [5, 10, 20, 40]), u: pick(r, [0.5, 0.6, 0.8]) }),
  q: v => `$${v.q}$ requests per second, $${v.ms}$ ms per request on one core, target utilisation $${v.u}$. How many cores?`,
  answer: v => Math.ceil(+((v.q * v.ms) / 1000 / v.u).toFixed(6)), dp: 0,
  steps: v => [`Busy cores: $${(v.q * v.ms) / 1000}$`],
});
export const CrEmbedMemoryDrill = drill({
  intro: 'Embedding memory is items times dimensions times bytes.',
  gen: r => ({ n: pick(r, [10, 50, 200]), d: pick(r, [128, 256, 768]) }),
  q: v => `$${v.n}$ million items, with embeddings of $${v.d}$ dimensions at 4 bytes each. How many gigabytes?`,
  answer: v => (v.n * 1e6 * v.d * 4) / 1e9, dp: 2,
  steps: v => [`Per item: $${v.d * 4}$ bytes`],
});
export const CrF1Drill = drill({
  intro: 'The F1 score is the harmonic mean of precision and recall.',
  tex: R`F_1=\frac{2\,\text{TP}}{2\,\text{TP}+\text{FP}+\text{FN}}`,
  gen: r => ({ t: int(r, 5, 80), p: int(r, 1, 40), n: int(r, 1, 40) }),
  q: v => `TP $=${v.t}$, FP $=${v.p}$, FN $=${v.n}$. Find the F1 score.`,
  answer: v => (2 * v.t) / (2 * v.t + v.p + v.n), dp: 3,
  steps: v => [`Precision: $${fmt(v.t / (v.t + v.p), 3)}$`, `Recall: $${fmt(v.t / (v.t + v.n), 3)}$`],
});
export const CrNdcgDrill = drill({
  intro: 'NDCG divides the ranking’s score by the best achievable.',
  tex: R`\text{DCG}=\sum_i\frac{2^{r_i}-1}{\log_2(i+1)}`,
  gen: r => Array.from({ length: 3 }, () => int(r, 0, 3)),
  q: v => { const a = v.slice(0, 3); return `Results have relevance $${a.join(', ')}$, in shown order. Find the NDCG, or enter 0 if all are 0.`; },
  answer: v => { const a = v.slice(0, 3), g = l => l.reduce((s, x, i) => s + (2 ** x - 1) / Math.log2(i + 2), 0), b = g([...a].sort((x, y) => y - x)); return b ? g(a) / b : 0; }, dp: 3,
  steps: v => { const a = v.slice(0, 3), g = l => l.reduce((s, x, i) => s + (2 ** x - 1) / Math.log2(i + 2), 0); return [`DCG: $${fmt(g(a), 3)}$`, `Ideal: $${fmt(g([...a].sort((x, y) => y - x)), 3)}$`]; },
});
export const CrThresholdDrill = drill({
  intro: 'With calibrated probabilities, compare the expected costs of each decision.',
  tex: R`p^*=\frac{c_{\text{fp}}}{c_{\text{fp}}+c_{\text{fn}}}`,
  gen: r => ({ a: int(r, 1, 20), b: int(r, 5, 200) }),
  q: v => `A false positive costs $${v.a}$, and a false negative $${v.b}$. Above what probability should you act?`,
  answer: v => v.a / (v.a + v.b), dp: 3,
  steps: v => [`Total: $${v.a + v.b}$`],
});
export const CrZTestDrill = drill({
  intro: 'A two-proportion test compares conversion in two groups.',
  tex: R`z=\frac{\hat p_B-\hat p_A}{\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}}`,
  gen: r => { const n = pick(r, [1000, 2000, 5000]); return { n, a: int(r, 5, 12) * n / 100, b: int(r, 6, 14) * n / 100 }; },
  q: v => `Each group has $${v.n}$ users. Conversions: A $${v.a}$, B $${v.b}$. Find $z$.`,
  answer: v => { const p = (v.a + v.b) / (2 * v.n); return (v.b - v.a) / v.n / Math.sqrt(p * (1 - p) * (2 / v.n)); }, dp: 3,
  steps: v => [`Pooled rate: $${fmt((v.a + v.b) / (2 * v.n), 4)}$`],
});
export const CrAbSizeLab = lab({
  intro: 'How long must an experiment run? Small effects need very large samples.',
  tex: R`n\approx\frac{2(z_{\alpha/2}+z_\beta)^2\,p(1-p)}{\delta^2}`,
  params: [{ k: 'p', label: 'baseline rate', min: 0.01, max: 0.5, step: 0.01, init: 0.1 }, { k: 'd', label: 'lift to detect, in points', min: 0.002, max: 0.05, step: 0.001, init: 0.01 }, { k: 'q', label: 'daily users per group', min: 100, max: 50000, step: 100, init: 2000 }],
  derive: ({ p }) => ({ f: d => (2 * (1.96 + 0.8416) ** 2 * p * (1 - p)) / (d * d) }),
  plots: v => [{ x: [0.002, 0.05], y: [0, 100000], w: 480, h: 260, xl: 'lift', yl: 'users per group', label: 'sample size against lift', marks: [{ tone: 1, data: curve(d => Math.min(100000, v.f(d)), 0.002, 0.05, 301) }, { as: 'rule', dash: true, tone: 4, x: v.d }] }],
  stats: v => [`${Math.ceil(v.f(v.d)).toLocaleString('en-US')} users per group`, `${fmt(v.f(v.d) / v.q, 1)} days`],
  takeaway: 'Halving the effect quadruples the sample, at 5% significance and 80% power.',
});
const binsOf = sh => { const e = linspace(-3, 3, 11), P = (a, b, m) => normalCdf(b - m) - normalCdf(a - m), edges = [-Infinity, ...e.slice(1, -1), Infinity]; return edges.slice(0, -1).map((a, i) => [P(a, edges[i + 1], 0), P(a, edges[i + 1], sh)]); };
export const CrPsiLab = lab({
  intro: 'Compare a live feature with its reference distribution, bin by bin.',
  tex: R`\text{PSI}=\sum_i(p_i-q_i)\ln\frac{p_i}{q_i}`,
  params: [{ k: 's', label: 'mean shift', min: 0, max: 2, step: 0.02, init: 0.3 }],
  derive: ({ s }) => { const B = binsOf(s); return { B, psi: B.reduce((t, [q, p]) => t + (p - q) * Math.log(Math.max(p, 1e-9) / Math.max(q, 1e-9)), 0) }; },
  plots: v => [{ x: [0, 10], y: [0, 0.4], w: 480, h: 260, xl: 'bin', yl: 'share', label: 'reference and live share', marks: [{ as: 'rects', tone: 'muted', soft: true, data: v.B.map(([q], i) => [i + 0.05, 0, i + 0.5, q]) }, { as: 'rects', tone: 1, data: v.B.map(([, p], i) => [i + 0.5, 0, i + 0.95, p]) }] }],
  stats: v => [`PSI ${fmt(v.psi, 3)}`],
  status: v => (v.psi < 0.1 ? ['ok', 'Stable.'] : v.psi < 0.25 ? ['warn', 'Moderate drift: investigate.'] : ['alert', 'Large drift: act.']),
  takeaway: 'The rule of thumb is 0.1 to investigate and 0.25 to act.',
});
export const CrQueueLab = lab({
  intro: 'As a server approaches capacity, waiting time grows sharply.',
  tex: R`W=\frac1{\mu-\lambda}`,
  params: [{ k: 'm', label: 'capacity $\\mu$, per second', min: 10, max: 500, step: 10, init: 100 }, { k: 'u', label: 'utilisation', min: 0.05, max: 0.98, step: 0.01, init: 0.7 }],
  plots: v => [{ x: [0, 1], y: [0, 500], w: 480, h: 260, xl: 'utilisation', yl: 'mean latency, ms', label: 'latency against utilisation', marks: [{ tone: 1, data: curve(u => Math.min(500, 1000 / (v.m * (1 - u))), 0, 0.995, 301) }, { as: 'rule', dash: true, tone: 4, x: v.u }] }],
  stats: v => [`latency ${fmt(1000 / (v.m * (1 - v.u)), 1)} ms`, `service alone ${fmt(1000 / v.m, 1)} ms`],
  takeaway: 'Plan capacity for well below full utilisation, because queues grow without bound near it.',
});
export const CrFeedbackLab = lab({
  intro: 'A recommender with two items. It sees clicks only for what it shows. Item B is truly better, and A got lucky at the start.',
  params: [{ k: 'e', label: 'exploration share $\\varepsilon$', min: 0, max: 0.5, step: 0.01, init: 0 }, { k: 'seed', seed: 'New runs', init: 1 }],
  derive: ({ e, seed }) => {
    const r = rng(seed * 3 + 11), N = 1000, H = Array(N).fill(0);
    for (let run = 0; run < 30; run++) { const shows = [5, 5], clicks = [2, 0]; for (let t = 0; t < N; t++) { const best = clicks[1] / shows[1] > clicks[0] / shows[0] ? 1 : 0, a = r() < e ? (r() < 0.5 ? 0 : 1) : best; H[t] += a; shows[a]++; if (r() < (a ? 0.2 : 0.1)) clicks[a]++; } }
    return { H: H.map((s, t) => [t, s / 30]).filter((_, t) => t % 5 === 0) };
  },
  plots: v => [{ x: [0, 1000], y: [0, 1], w: 480, h: 260, xl: 'round', yl: 'share showing item B', label: 'share of traffic to the better item', marks: [{ tone: 1, data: v.H }] }],
  stats: v => [`final share to item B ${pct(v.H[v.H.length - 1][1], 0)}`],
  takeaway: 'Without exploration, early luck can persist, because the system never gathers contrary evidence.',
});
export const InterviewCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A p-value is the probability the null hypothesis is true.', false, 'It is the probability of such data if the null is true.'],
    ['Bagging mainly reduces variance.', true, 'Averaging decorrelated models cancels their errors.'],
    ['Scaling before the train-test split is fine.', false, 'It leaks test statistics into training.'],
    ['Trees need standardised features.', false, 'Splits depend only on order.'],
    ['ROC-AUC can look strong on very rare events.', true, 'The false-positive rate is diluted by many negatives.'],
  ],
  takeaway: 'Answer with the idea, the formula, an example and a limitation.',
});
export const DesignCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Start a design with the most advanced model.', false, 'Start with a baseline and justify complexity.'],
    ['A canary sends a small share of traffic to the new model.', true, 'Problems affect few users.'],
    ['A shadow deployment affects users.', false, 'Its output is logged only.'],
    ['A feature store reduces training-serving skew.', true, 'Both use one definition.'],
    ['Random splits are right for fraud detection over time.', false, 'Split by time, since patterns shift.'],
  ],
  takeaway: 'Interviewers grade your reasoning about goals, data and failure, more than the model name.',
});
