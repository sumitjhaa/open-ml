'use client';
// Explorers for deep network training, attention, generative models, Bayesian methods and ML systems.
import { R, curve, drill, heatBox, heatPlot, histDensity, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng, solve } from '../lib/math.mjs';
import { em2d, fmt, linspace, memo, moons, normalPdf, pct } from '../lib/models.mjs';
import { mlpForward, softmaxT, trainMlp } from '../lib/extended.mjs';

const r6 = v => +v.toFixed(6);
const hist = (a, lo, hi, n) => { const b = Array(n).fill(0), w = (hi - lo) / n; a.forEach(x => { if (x >= lo && x < hi) b[Math.floor((x - lo) / w)]++; }); return b.map((c, i) => [lo + i * w, 0, lo + (i + 0.9) * w, c / (a.length * w)]); };
const heatRows = (M, tone = 1) => { const n = M.length, peak = Math.max(1e-9, ...M.flat().map(Math.abs)); return M.flatMap((row, i) => row.flatMap((v, j) => [{ as: 'rects', tone, soft: Math.abs(v) / peak < 0.5, data: Math.abs(v) / peak < 0.03 ? [] : [[j + 0.03, n - i - 0.97, j + 0.97, n - i - 0.03]] }, { as: 'label', x: j + 0.5, y: n - i - 0.62, text: v.toFixed(2), anchor: 'middle' }])); };

/* ---------- training ---------- */

const NOISY = (() => { const r = rng(21); return Array.from({ length: 120 }, (_, t) => r6(Math.sin(t / 15) + 0.5 * gauss(r))); })();
export const EwmaLab = lab({
  intro: 'A running average smooths noise. Higher $\\beta$ remembers more, but reacts later. Correcting the start removes the early bias.',
  tex: R`v_t=\beta v_{t-1}+(1-\beta)x_t`,
  params: [{ k: 'b', label: '$\\beta$', min: 0, max: 0.99, step: 0.01, init: 0.9 }, { k: 'c', label: 'Start', options: [['raw', 'raw'], ['fix', 'bias corrected']], init: 'raw' }],
  derive: ({ b, c }) => { let v = 0; return { s: NOISY.map((x, t) => { v = b * v + (1 - b) * x; return c === 'fix' ? v / (1 - b ** (t + 1)) : v; }) }; },
  plots: v => [{ x: [0, 120], y: [-2.5, 2.5], w: 620, h: 260, xl: 'step', label: 'noisy series and its average', marks: [{ tone: 'muted', soft: true, data: NOISY.map((y, i) => [i, y]) }, { tone: 1, data: v.s.map((y, i) => [i, y]) }] }],
  stats: v => [`memory about ${fmt(1 / (1 - v.b), 1)} steps`],
  takeaway: 'Momentum and Adam are built on exactly this average.',
});
const OPTS = { sgd: 'plain', mom: 'momentum', rms: 'RMSProp', adam: 'Adam' };
const gradV = ([x, y]) => [x, 25 * y];
const runOpt = memo((k, lr) => { let p = [-4.5, 0.9], v = [0, 0], s = [0, 0]; const out = [p]; for (let t = 1; t <= 80; t++) { const g = gradV(p); if (k === 'sgd') p = p.map((a, i) => a - lr * g[i]); if (k === 'mom') { v = v.map((a, i) => 0.9 * a + g[i]); p = p.map((a, i) => a - lr * v[i]); } if (k === 'rms') { s = s.map((a, i) => 0.9 * a + 0.1 * g[i] ** 2); p = p.map((a, i) => a - (lr * 10 * g[i]) / (Math.sqrt(s[i]) + 1e-8)); } if (k === 'adam') { v = v.map((a, i) => 0.9 * a + 0.1 * g[i]); s = s.map((a, i) => 0.999 * a + 0.001 * g[i] ** 2); p = p.map((a, i) => a - (lr * 10 * (v[i] / (1 - 0.9 ** t))) / (Math.sqrt(s[i] / (1 - 0.999 ** t)) + 1e-8)); } p = p.map(r6); out.push(p); } return out; });
export const OptimizerRace = lab({
  intro: 'A long narrow valley. Compare how each optimiser travels from the same start with the same base rate.',
  params: [{ k: 'k', label: 'Optimiser', options: Object.entries(OPTS), init: 'sgd' }, { k: 'lr', label: 'learning rate', min: 0.005, max: 0.075, step: 0.005, init: 0.035 }, { k: 't', label: 'step', min: 0, max: 80, step: 1, init: 80, play: true }],
  derive: ({ k, lr, t }) => { const P = runOpt(k, lr).map(p => p.map(a => Math.max(-99, Math.min(99, a)))); return { P, pos: P[t], f: 0.5 * P[t][0] ** 2 + 12.5 * P[t][1] ** 2 }; },
  plots: v => [{ x: [-5.5, 2], y: [-1.5, 1.5], w: 620, h: 280, label: 'optimiser path', marks: [{ as: 'field', bands: 8, tones: [1, 1], fn: (x, y) => 1 - Math.sqrt(Math.min(1, (0.5 * x * x + 12.5 * y * y) / 43)) }, { as: 'contour', tone: 'muted', fn: (x, y) => 0.5 * x * x + 12.5 * y * y, levels: [0.5, 2, 5, 10, 18] }, { tone: 2, data: v.P.slice(0, v.t + 1) }, { as: 'points', tone: 4, r: 5, data: [v.pos] }, { as: 'rings', r: 8, data: [[0, 0]] }] }],
  stats: v => [`loss ${fmt(v.f, 4)}`],
  takeaway: 'Plain descent is limited by the steep direction. Momentum and per-coordinate scaling are two ways round that.',
});
export const ScheduleLab = lab({
  intro: 'The learning rate need not stay fixed. Compare common schedules.',
  params: [{ k: 'k', label: 'Schedule', options: [['step', 'step decay'], ['cos', 'cosine'], ['warm', 'warmup, then cosine']], init: 'cos' }, { k: 'w', label: 'warmup epochs', min: 1, max: 30, step: 1, init: 10 }],
  derive: ({ k, w }) => ({ f: t => (k === 'step' ? 0.1 * 0.1 ** Math.floor(t / 40) : k === 'cos' ? 0.05 * (1 + Math.cos((Math.PI * t) / 120)) : t < w ? (0.1 * t) / w : 0.05 * (1 + Math.cos((Math.PI * (t - w)) / (120 - w)))) }),
  plots: v => [{ x: [0, 120], y: [0, 0.11], w: 520, h: 260, xl: 'epoch', yl: 'learning rate', label: 'schedule', marks: [{ tone: 1, data: curve(v.f, 0, 120, 241) }] }],
  stats: v => [`at epoch 60: ${fmt(v.f(60), 4)}`],
  takeaway: 'Large early steps cover ground, and small late steps settle in.',
});
export const AdamStepDrill = drill({
  intro: 'Adam’s first step: both averages start at zero and the bias correction cancels $1-\\beta$.',
  tex: R`\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat s}}`,
  gen: r => ({ g: pick(r, [2, -3, 4, 0.5]), lr: pick(r, [0.1, 0.01, 0.5]) }),
  q: v => `The very first gradient is $${v.g}$ and $\\eta=${v.lr}$. By how much does Adam change $\\theta$? Give a signed change.`,
  answer: v => -v.lr * Math.sign(v.g), dp: 3,
  steps: v => ['After correction, $\\hat m=g$ and $\\hat s=g^2$.', `So the step is $-\\eta\\,g/|g|=-${v.lr}\\times\\text{sign}(${v.g})$.`],
});
export const WeightDecayDrill = drill({
  intro: 'Each step, weight decay shrinks a weight by a factor before the data term is applied.',
  gen: r => ({ lr: pick(r, [0.1, 0.01]), l: pick(r, [0.1, 0.5, 1]), w: int(r, 1, 9) }),
  q: v => `$\\eta=${v.lr}$, $\\lambda=${v.l}$, $w=${v.w}$. Find $w$ after the decay step.`,
  answer: v => v.w * (1 - v.lr * v.l), dp: 4,
  steps: v => [`Factor: $1-${v.lr}\\times${v.l}$`],
});
export const LabelSmoothDrill = drill({
  intro: 'Label smoothing moves a little of the target’s weight onto the other classes.',
  gen: r => ({ e: pick(r, [0.1, 0.2, 0.05]), k: pick(r, [2, 5, 10]), t: pick(r, [true, false]) }),
  q: v => `$\\varepsilon=${v.e}$ with $${v.k}$ classes. Find the target for ${v.t ? 'the true' : 'a wrong'} class.`,
  answer: v => (v.t ? 1 - v.e + v.e / v.k : v.e / v.k), dp: 4,
  steps: v => [`$\\varepsilon/K=${fmt(v.e / v.k, 4)}$`],
});
const DMO = moons(8, 40, 0.28), XD = DMO.map(p => [p[0], p[1]]), YD = DMO.map(p => p[2]), VMO = moons(9, 300, 0.28), XV = VMO.map(p => [p[0], p[1]]), YV = VMO.map(p => p[2]);
const lossOn = (p, X, Y) => mean(X.map((x, i) => { const q = Math.min(1 - 1e-6, Math.max(1e-6, mlpForward(p, x, 'relu').p)); return Y[i] ? -Math.log(q) : -Math.log(1 - q); }));
const earlyRun = memo(() => trainMlp(XD, YD, { hidden: 24, act: 'relu', lr: 0.6, steps: 6000, every: 150, seed: 5 }).map(f => ({ t: f.t, tr: f.loss, va: lossOn(f.p, XV, YV) })));
export const EarlyStopLab = lab({
  intro: 'A flexible network on few noisy points. Training loss keeps falling. Find where the held-out loss turns.',
  params: [{ k: 'i', label: 'stop at', min: 0, max: 40, step: 1, init: 40, show: v => `${v * 150} steps` }],
  derive: ({ i }) => { const F = earlyRun(), best = F.reduce((a, b) => (b.va < a.va ? b : a)); return { F, f: F[i], best }; },
  plots: v => [{ x: [0, 6000], y: [0, 1.4], w: 520, h: 280, xl: 'step', yl: 'loss', label: 'training and held-out loss', marks: [{ tone: 2, data: v.F.map(f => [f.t, f.tr]) }, { tone: 4, data: v.F.map(f => [f.t, Math.min(1.4, f.va)]) }, { as: 'rule', dash: true, tone: 'muted', x: v.f.t }] }],
  stats: v => [`held-out loss ${fmt(v.f.va, 3)}`, `lowest held-out loss ${fmt(v.best.va, 3)}, at step ${v.best.t}`],
  legend: [['--c2', 'training'], ['--c4', 'held-out']],
  takeaway: 'Keep the weights from the best validation step, not the last one.',
});
export const DropoutMasks = lab({
  intro: 'Each pass randomly drops units. Dividing the survivors by $1-p$ keeps the expected output the same.',
  params: [{ k: 'p', label: 'drop probability', min: 0, max: 0.9, step: 0.05, init: 0.5 }, { k: 'fix', label: 'Scaling', options: [['yes', 'divide by 1 − p'], ['no', 'no scaling']], init: 'yes' }, { k: 'seed', seed: 'New masks', init: 1 }],
  derive: ({ p, fix, seed }) => { const r = rng(seed * 13), a = Array.from({ length: 20 }, (_, i) => 1 + 0.5 * Math.sin(i)), full = a.reduce((s, v) => s + v, 0), passes = Array.from({ length: 200 }, () => a.reduce((s, v) => s + (r() < p ? 0 : fix === 'yes' ? v / (1 - p) : v), 0)); return { full, passes, m: mean(passes), sd: Math.sqrt(mean(passes.map(v => (v - mean(passes)) ** 2))) }; },
  plots: v => [{ x: [0, 70], y: [0, 0.3], w: 480, h: 260, xl: 'total output', label: 'distribution of output over masks', marks: [{ as: 'rects', tone: 1, data: hist(v.passes, 0, 70, 35) }, { as: 'rule', dash: true, tone: 4, x: v.full }] }],
  stats: v => [`mean over masks ${fmt(v.m, 2)}`, `full output ${fmt(v.full, 2)}`, `spread ${fmt(v.sd, 2)}`],
  takeaway: 'At test time, all units are used and the output matches the training average.',
});

/* ---------- attention ---------- */

const TEXT = 'the cat sat';
export const TokenLab = lab({
  intro: 'A model reads numbers, so text must become tokens first. The choice of token changes the vocabulary and the length.',
  params: [{ k: 'k', label: 'Token', options: [['char', 'characters'], ['word', 'words']], init: 'word' }],
  derive: ({ k }) => { const toks = k === 'char' ? [...TEXT] : TEXT.split(' '), vocab = [...new Set(toks)].sort(); return { toks, vocab, ids: toks.map(t => vocab.indexOf(t)) }; },
  table: v => ({ head: ['token', 'id'], rows: v.toks.map((t, i) => [t === ' ' ? 'space' : t, v.ids[i]]) }),
  stats: v => [`${v.toks.length} tokens`, `vocabulary of ${v.vocab.length}`],
  plots: () => [],
  takeaway: 'Larger tokens make shorter sequences and much larger vocabularies.',
});
const WORDS = ['the', 'animal', 'was', 'tired'], KEYS = [[0.2, 0.1], [1.6, 0.3], [0.3, 0.8], [0.4, 1.7]], QUERIES = { 'the': [0.3, 0.2], animal: [0.2, 1.4], was: [1.3, 0.9], tired: [1.6, 0.2] };
export const AttentionWeights = lab({
  intro: 'Choose which word is asking. Its query is compared with every key, and softmax turns the scores into weights.',
  tex: R`\alpha_{ij}=\text{softmax}_j\big(q_i^\top k_j/\sqrt d\big)`,
  params: [{ k: 'w', label: 'Query word', options: WORDS.map(w => [w, w]), init: 'tired' }, { k: 'T', label: 'sharpness', min: 0.2, max: 4, step: 0.1, init: 1 }],
  derive: ({ w, T }) => { const q = QUERIES[w], s = KEYS.map(k => (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2); return { s, a: softmaxT(s, 1 / T) }; },
  plots: v => { const M = WORDS.map(w => softmaxT(KEYS.map(k => (QUERIES[w][0] * k[0] + QUERIES[w][1] * k[1]) / Math.SQRT2), 1 / v.T)); return [heatPlot(M, { w: 460, h: 300, label: 'attention weights', over: [heatBox(M, WORDS.indexOf(v.w), null)], heat: { max: 1, rows: WORDS, cols: WORDS, show: x => x.toFixed(2).replace(/^0/, ''), name: (i, j, x) => `${WORDS[i]} attends to ${WORDS[j]}: ${x.toFixed(3)}` } })]; },
  stats: v => [`most attention on “${WORDS[v.a.indexOf(Math.max(...v.a))]}”`, `weights add up to ${fmt(v.a.reduce((x, y) => x + y, 0), 3)}`],
  takeaway: 'Weights are learned through the queries and keys, and they depend on the input.',
});
export const ScaledDotDrill = drill({
  intro: 'Scores are dot products divided by $\\sqrt d$, then softmax makes them add up to one.',
  tex: R`\alpha_j=\text{softmax}_j\big(q^\top k_j/\sqrt d\big)`,
  gen: r => ({ q: [int(r, 0, 3), int(r, 0, 3)], k1: [int(r, 0, 2), int(r, 0, 2)], k2: [int(r, 0, 2), int(r, 0, 2)] }),
  q: v => `$d=2$, $q=(${v.q})$, keys $(${v.k1})$ and $(${v.k2})$. Find the first attention weight.`,
  answer: v => { const s = [v.k1, v.k2].map(k => (v.q[0] * k[0] + v.q[1] * k[1]) / Math.SQRT2); return softmaxT(s)[0]; }, dp: 3,
  steps: v => [`Dot products: $${v.q[0] * v.k1[0] + v.q[1] * v.k1[1]}$ and $${v.q[0] * v.k2[0] + v.q[1] * v.k2[1]}$`, 'Divide by $\\sqrt2$, then take the softmax.'],
});
const randM = (r, n, m) => Array.from({ length: n }, () => Array.from({ length: m }, () => r6(gauss(r))));
export const SelfAttentionMatrix = lab({
  intro: 'Three tokens, one attention step. Each row of the weight matrix adds up to one, and each output row mixes the values.',
  tex: R`Z=\text{softmax}\big(QK^\top/\sqrt d\big)V`,
  params: [{ k: 'seed', seed: 'New tokens', init: 1 }, { k: 'mask', label: 'Mask', options: [['no', 'no mask'], ['yes', 'causal mask']], init: 'no' }],
  derive: ({ seed, mask }) => { const r = rng(seed * 23), Q = randM(r, 3, 2), K = randM(r, 3, 2), V = randM(r, 3, 2), A = Q.map((q, i) => softmaxT(K.map((k, j) => (j > i && mask === 'yes' ? -1e6 : (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2)))), Z = A.map(a => [0, 1].map(d => a.reduce((s, w, j) => s + w * V[j][d], 0))); return { A, Z, V }; },
  plots: v => { const T = ['a', 'b', 'c'], max = Math.max(...v.V.flat(), ...v.Z.flat(), ...v.V.flat().map(x => -x), ...v.Z.flat().map(x => -x)), f = x => x.toFixed(2), p = (M, caption, cols, heat) => heatPlot(M, { w: 90 + 60 * cols.length, h: 196, caption, label: caption, heat: { rows: T, cols, show: f, ...heat } }); return [p(v.A, 'Weights A', T, { max: 1 }), p(v.V, 'Values V', ['x', 'y'], { div: true, max }), p(v.Z, 'Output Z = AV', ['x', 'y'], { div: true, max })]; },
  stats: () => ['each row of weights adds up to one'],
  takeaway: 'Attention is just a softmax-weighted average, made differentiable by construction.',
});
export const PosEncoding = lab({
  intro: 'Sines at different frequencies give each position a distinct pattern.',
  tex: R`PE_{t,2i}=\sin\big(t/10000^{2i/d}\big)`,
  params: [{ k: 'd', label: 'dimensions', min: 8, max: 64, step: 8, init: 32 }, { k: 't', label: 'position', min: 0, max: 49, step: 1, init: 10 }],
  plots: v => { const f = (t, i) => (i % 2 ? Math.cos : Math.sin)(t / 10000 ** ((2 * Math.floor(i / 2)) / v.d)), M = Array.from({ length: v.d }, (_, i) => Array.from({ length: 50 }, (_, t) => r6(f(t, i)))); return [heatPlot(M, { w: 620, h: 280, caption: 'Position left to right, dimension top to bottom', label: 'positional encoding', over: [heatBox(M, null, v.t)], heat: { dense: true, div: true, max: 1, name: (i, j, x) => `position ${j}, dimension ${i}: ${x.toFixed(3)}` } }), { caption: `The code for position ${v.t}`, x: [-0.5, v.d - 0.5], y: [-1.1, 1.1], w: 620, h: 180, xl: 'dimension', label: 'encoding at the chosen position', marks: [{ as: 'stems', tone: 1, data: M.map((r, i) => [i, 0, r[v.t]]) }, { as: 'points', tone: 1, r: 3, data: M.map((r, i) => [i, r[v.t]]) }] }]; },
  stats: v => [`position ${v.t}`],
  takeaway: 'Low dimensions change quickly and high ones slowly, like the digits of a clock.',
});
export const HeadShapeDrill = drill({
  intro: 'Heads split the embedding, so together they keep the original width.',
  gen: r => ({ h: pick(r, [2, 4, 8, 12]), m: pick(r, [32, 64, 96]) }),
  q: v => `The embedding has size $${v.h * v.m}$ and there are $${v.h}$ heads. How wide is each head?`,
  answer: v => v.m, dp: 0,
  steps: v => [`Divide $${v.h * v.m}$ by $${v.h}$.`],
});
export const PatchCountDrill = drill({
  intro: 'A vision transformer treats each patch as one token.',
  gen: r => { const p = pick(r, [4, 8, 16]), n = pick(r, [8, 14, 16]); return { s: p * n, p, n }; },
  q: v => `A $${v.s}\\times${v.s}$ image with $${v.p}\\times${v.p}$ patches, and one class token. How many tokens is that?`,
  answer: v => v.n * v.n + 1, dp: 0,
  steps: v => [`Patches per side: $${v.n}$`],
});
const LOG5 = [2.4, 1.9, 1.1, 0.5, 0.1, -0.5, -1.2];
export const SamplingStrategies = lab({
  intro: 'How a language model picks its next token changes what it writes.',
  params: [{ k: 'm', label: 'Strategy', options: [['top', 'top-k'], ['nuc', 'nucleus']], init: 'nuc' }, { k: 'k', label: 'k', min: 1, max: 7, step: 1, init: 3 }, { k: 'p', label: 'nucleus p', min: 0.3, max: 1, step: 0.05, init: 0.75 }, { k: 'T', label: 'temperature', min: 0.2, max: 3, step: 0.1, init: 1 }],
  derive: ({ m, k, p, T }) => { const P = softmaxT(LOG5, T); let c = 0, n = 0; if (m === 'top') n = k; else for (const q of P) { n++; c += q; if (c >= p) break; } const kept = P.map((q, i) => (i < n ? q : 0)), t = kept.reduce((a, b) => a + b, 0); return { P, kept: kept.map(q => q / t), n }; },
  plots: v => [{ x: [-0.5, 6.5], y: [0, 1], w: 460, h: 260, xl: 'tokens, ranked', label: 'token probabilities', marks: [{ as: 'rects', tone: 'muted', soft: true, data: v.P.map((q, i) => [i - 0.4, 0, i + 0.4, q]) }, { as: 'rects', tone: 1, data: v.kept.map((q, i) => [i - 0.25, 0, i + 0.25, q]) }] }],
  stats: v => [`${v.n} tokens kept`],
  takeaway: 'Nucleus sampling adapts how many tokens it keeps to how sure the model is.',
});
const norm2 = v => { const n = Math.hypot(...v); return v.map(x => x / n); };
const CLASSES = { 'a dog': [1, 0.15], 'a cat': [0.75, 0.8], 'a car': [-0.2, 1] };
export const ClipZeroShot = lab({
  intro: 'Rotate the image embedding. Classes are only sentences, compared by cosine similarity, with no training examples.',
  params: [{ k: 'a', label: 'image direction (degrees)', min: 0, max: 130, step: 1, init: 30 }, { k: 't', label: 'temperature', min: 0.02, max: 1, step: 0.02, init: 0.1 }],
  derive: ({ a, t }) => { const img = [Math.cos((a * Math.PI) / 180), Math.sin((a * Math.PI) / 180)], names = Object.keys(CLASSES), sims = names.map(n => { const v = norm2(CLASSES[n]); return v[0] * img[0] + v[1] * img[1]; }); return { img, names, sims, p: softmaxT(sims, t) }; },
  plots: v => [{ x: [-1.3, 1.3], y: [-0.4, 1.3], w: 360, h: 280, label: 'image and text embeddings', marks: [...v.names.map((n, i) => ({ as: 'arrow', tone: i + 1, data: [[0, 0, ...norm2(CLASSES[n])]] })), ...v.names.map(n => ({ as: 'label', x: norm2(CLASSES[n])[0] * 1.08, y: norm2(CLASSES[n])[1] * 1.08, text: n, anchor: 'middle' })), { as: 'arrow', tone: 4, data: [[0, 0, ...v.img]] }] }],
  table: v => ({ head: ['class', 'similarity', 'probability'], rows: v.names.map((n, i) => [n, fmt(v.sims[i], 3), fmt(v.p[i], 3)]) }),
  stats: v => [`predicted: ${v.names[v.p.indexOf(Math.max(...v.p))]}`],
  takeaway: 'A new class costs one sentence. Lower temperatures make the choice more decisive.',
});
const DOCS = ['ridge regression shrinks weights with a penalty', 'kmeans assigns points to the nearest centroid', 'transformers use attention over tokens', 'dropout randomly removes units during training', 'a penalty on weights reduces overfitting'];
const bag = t => { const c = {}; t.split(' ').forEach(w => { c[w] = (c[w] ?? 0) + 1; }); return c; };
const cos = (a, b) => { let s = 0; for (const k in a) s += a[k] * (b[k] ?? 0); const n = o => Math.sqrt(Object.values(o).reduce((x, y) => x + y * y, 0)); return s / (n(a) * n(b) || 1); };
const QS = ['penalty on weights', 'attention tokens', 'nearest centroid points', 'units removed in training'];
export const RetrievalLab = lab({
  intro: 'Retrieval finds the most relevant documents, and adds them to the prompt. Here similarity is simple word counts.',
  params: [{ k: 'q', label: 'Question', options: QS.map(q => [q, q]), init: QS[0] }],
  derive: ({ q }) => { const b = bag(q), s = DOCS.map(d => cos(b, bag(d))); return { s, best: s.indexOf(Math.max(...s)) }; },
  table: v => ({ head: ['document', 'similarity'], rows: DOCS.map((d, i) => [i === v.best ? `**${d}**` : d, fmt(v.s[i], 3)]) }),
  stats: v => [`retrieved: “${DOCS[v.best]}”`],
  plots: () => [],
  takeaway: 'Real systems use learned embeddings, but the idea is the same: find, then answer from what was found.',
});
export const TransformerCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Attention without position information cannot tell word order.', true, 'It is equivariant to permutations.'],
    ['Each attention weight row adds up to one.', true, 'It comes from a softmax.'],
    ['A causal mask lets a token see future tokens.', false, 'It blocks them, so generation stays honest.'],
    ['Heads run on the full embedding width each.', false, 'Usually each head gets $d/h$ dimensions.'],
    ['CLIP needs labelled examples of every new class.', false, 'A class can be described with a sentence.'],
    ['Retrieved text should be treated as data, not as instructions.', true, 'Otherwise a document could hijack the model.'],
  ],
  takeaway: 'Attention lets every token decide what to read from every other.',
});

/* ---------- generative ---------- */

export const PixelVsLatent = lab({
  intro: 'A square moves across an image. Mixing pixels shows two ghosts. Moving through a good code moves the square.',
  params: [{ k: 'a', label: 'mix', min: 0, max: 1, step: 0.05, init: 0.5 }, { k: 'm', label: 'Interpolate in', options: [['p', 'pixels'], ['l', 'latent code']], init: 'p' }],
  derive: ({ a, m }) => { const n = 16, sq = c => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(Math.abs(i - 8) < 3 && Math.abs(j - c) < 3))), A = sq(3), B = sq(12); return { I: m === 'p' ? A.map((r, i) => r.map((v, j) => (1 - a) * v + a * B[i][j])) : sq(3 + 9 * a) }; },
  plots: v => [{ x: [0, 16], y: [0, 16], w: 320, h: 300, label: 'interpolated image', marks: v.I.flatMap((row, i) => row.map((c, j) => ({ as: 'rects', tone: 1, soft: c < 0.75, data: c > 0.02 ? [[j, 15 - i, j + 1, 16 - i]] : [] }))) }],
  stats: v => [v.m === 'p' ? 'halfway is two faint squares' : 'halfway is one solid square'],
  takeaway: 'A good latent space places similar things close together, which is what makes sampling meaningful.',
});
const MO = moons(4, 200, 0.08), XM = MO.map(p => [p[0], p[1]]);
const gmmFit = memo(k => em2d(XM, k === 1 ? [[0.5, 0.25]] : [[-0.5, 0.5], [1.5, -0.2]], 60).at(-1));
export const FitAndSample = lab({
  intro: 'A simple generative model: fit a distribution, then draw from it. See what one or two Gaussians can capture.',
  params: [{ k: 'k', label: 'Components', options: [[1, 'one'], [2, 'two']], init: 1 }, { k: 'seed', seed: 'New samples', init: 1 }],
  derive: ({ k, seed }) => { const f = gmmFit(+k), r = rng(seed * 7); return { S: Array.from({ length: 160 }, () => { const u = r(), c = f.pi.length === 1 || u < f.pi[0] ? 0 : 1, [a, b, d] = f.S[c], l = Math.sqrt(a), m = b / l, n = Math.sqrt(Math.max(d - m * m, 1e-9)), z1 = gauss(r), z2 = gauss(r); return [f.mu[c][0] + l * z1, f.mu[c][1] + m * z1 + n * z2]; }) }; },
  plots: v => [{ x: [-1.8, 2.8], y: [-1.3, 1.7], w: 420, h: 300, label: 'data and samples', marks: [{ as: 'points', tone: 'muted', r: 2.5, data: XM }, { as: 'points', tone: 2, r: 3, data: v.S }] }],
  stats: () => ['grey is data, and orange is generated'],
  takeaway: 'A flexible model is needed to capture curved structure, which is what neural generative models provide.',
});
const kl = (m, s) => 0.5 * (m * m + s * s - Math.log(s * s) - 1);
export const GaussKlDrill = drill({
  intro: 'The KL term of a VAE has a closed form for Gaussians.',
  tex: R`\text{KL}=\tfrac12(\mu^2+\sigma^2-\ln\sigma^2-1)`,
  gen: r => ({ m: pick(r, [0, 1, 2, -1]), s: pick(r, [1, 2, 0.5]) }),
  q: v => `Find $\\text{KL}(\\mathcal N(${v.m},${v.s}^2)\\,\\|\\,\\mathcal N(0,1))$.`,
  answer: v => kl(v.m, v.s), dp: 3,
  steps: v => [`$\\mu^2+\\sigma^2=${v.m * v.m + v.s * v.s}$`, `$\\ln\\sigma^2=${fmt(Math.log(v.s * v.s), 3)}$`],
});
export const ReparamDrill = drill({
  intro: 'Move the randomness into an input, and gradients can flow through the mean and spread.',
  tex: R`z=\mu+\sigma\varepsilon`,
  gen: r => ({ m: int(r, -3, 3), s: pick(r, [0.5, 1, 2]), e: pick(r, [-1, 0.5, 1.5, 2]) }),
  q: v => `$\\mu=${v.m}$, $\\sigma=${v.s}$, $\\varepsilon=${v.e}$. Find $z$.`,
  answer: v => v.m + v.s * v.e, dp: 3,
  steps: v => [`$\\sigma\\varepsilon=${v.s * v.e}$`],
});
const X1 = 2;
export const ElboLab = lab({
  intro: 'One observation $x=2$, with prior $z\\sim\\mathcal N(0,1)$ and $x\\mid z\\sim\\mathcal N(z,1)$. Tune $q$ until the bound is tight.',
  tex: R`\text{ELBO}=\mathbb E_q[\log p(x\mid z)]-\text{KL}(q\|p)`,
  params: [{ k: 'm', label: 'mean of $q$', min: -2, max: 3, step: 0.05, init: 0 }, { k: 's', label: 'spread of $q$', min: 0.1, max: 2, step: 0.05, init: 1 }],
  derive: ({ m, s }) => { const rec = -0.5 * Math.log(2 * Math.PI) - 0.5 * ((X1 - m) ** 2 + s * s), elbo = rec - kl(m, s), ev = -0.5 * Math.log(4 * Math.PI) - X1 ** 2 / 4; return { rec, elbo, ev, gap: ev - elbo }; },
  plots: v => [{ x: [-3, 4], y: [0, 1.2], w: 460, h: 260, xl: 'z', label: 'approximate and true posterior', marks: [{ tone: 3, data: curve(z => normalPdf(z, 1, Math.SQRT1_2), -3, 4) }, { tone: 1, data: curve(z => normalPdf(z, v.m, v.s), -3, 4) }] }, { caption: 'ELBO over every choice of q: drag the point to the peak', x: [-2, 3], y: [0.1, 2], w: 460, h: 300, xl: 'mean of q', yl: 'spread of q', label: 'ELBO surface', marks: [{ as: 'field', bands: 8, tones: [1, 1], fn: (m, s) => 1 - Math.sqrt(Math.min(1, (v.ev - (-0.5 * Math.log(2 * Math.PI) - 0.5 * ((X1 - m) ** 2 + s * s) - kl(m, s))) / 12)) }, { as: 'rings', r: 8, data: [[1, Math.SQRT1_2]] }, { drag: { x: 'm', y: 's', tone: 2 }, r: 7 }] }],
  stats: v => [`ELBO ${fmt(v.elbo, 4)}`, `log evidence ${fmt(v.ev, 4)}`, `gap ${fmt(v.gap, 4)}`],
  status: v => (v.gap < 0.002 ? ['ok', 'Tight: $q$ matches the true posterior.'] : undefined),
  legend: [['--c3', 'true posterior'], ['--c1', 'q']],
  takeaway: 'The gap between the evidence and the bound is exactly $\\text{KL}(q\\|\\text{posterior})$.',
});
export const AlphaBarDrill = drill({
  intro: 'Diffusion accumulates the survival factor at each step.',
  tex: R`\bar\alpha_t=\prod_{s\le t}(1-\beta_s)`,
  gen: r => ({ b: pick(r, [0.1, 0.2, 0.05]), t: int(r, 1, 5) }),
  q: v => `$\\beta=${v.b}$ at every step, for $${v.t}$ steps. Find $\\bar\\alpha$.`,
  answer: v => (1 - v.b) ** v.t, dp: 4,
  steps: v => [`$(1-${v.b})^{${v.t}}$`],
});
export const NoisedSampleDrill = drill({
  intro: 'Jump straight to any step of the forward process.',
  tex: R`x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon`,
  gen: r => ({ a: pick(r, [0.81, 0.64, 0.36, 0.25]), x: int(r, -3, 3), e: pick(r, [-1, 1, 2]) }),
  q: v => `$\\bar\\alpha=${v.a}$, $x_0=${v.x}$, $\\varepsilon=${v.e}$. Find $x_t$.`,
  answer: v => Math.sqrt(v.a) * v.x + Math.sqrt(1 - v.a) * v.e, dp: 3,
  steps: v => [`$\\sqrt{\\bar\\alpha}=${fmt(Math.sqrt(v.a), 3)}$ and $\\sqrt{1-\\bar\\alpha}=${fmt(Math.sqrt(1 - v.a), 3)}$`],
});
export const GanOptimalDrill = drill({
  intro: 'With the generator fixed, the best discriminator is a simple ratio.',
  tex: R`D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_G(x)}`,
  gen: r => ({ a: pick(r, [0.1, 0.2, 0.3, 0.6]), b: pick(r, [0.1, 0.2, 0.3, 0.6]) }),
  q: v => `$p_{\\text{data}}=${v.a}$ and $p_G=${v.b}$ at a point. Find $D^*$.`,
  answer: v => v.a / (v.a + v.b), dp: 3,
  steps: () => ['Divide the data density by the sum of the two.'],
});
const gan = memo((lrg, steps) => { let th = -2, w = 0.1, b = 0; const R0 = rng(5), T = []; for (let t = 0; t < steps; t++) { for (let k = 0; k < 3; k++) { let gw = 0, gb = 0; for (let i = 0; i < 40; i++) { const x = 2 + gauss(R0), z = th + gauss(R0), pr = 1 / (1 + Math.exp(-(w * x + b))), pf = 1 / (1 + Math.exp(-(w * z + b))); gw += (1 - pr) * x - pf * z; gb += 1 - pr - pf; } w = r6(w + (0.05 * gw) / 40); b = r6(b + (0.05 * gb) / 40); } let gt = 0; for (let i = 0; i < 40; i++) { const z = th + gauss(R0), pf = 1 / (1 + Math.exp(-(w * z + b))); gt += (1 - pf) * w; } th = r6(th + (lrg * gt) / 40); T.push(th); } return T; });
export const Gan1dLab = lab({
  intro: 'A generator shifts noise by one number, and tries to match data centred at 2. Watch it chase a moving judge.',
  params: [{ k: 'lr', label: 'generator learning rate', min: 0.05, max: 1.5, step: 0.05, init: 0.3 }],
  derive: ({ lr }) => ({ T: gan(lr, 300) }),
  plots: v => [{ x: [0, 300], y: [-3, 7], w: 520, h: 280, xl: 'step', yl: 'generator centre', label: 'generator centre over training', marks: [{ tone: 1, data: v.T.map((t, i) => [i, Math.max(-3, Math.min(7, t))]) }, { as: 'rule', dash: true, tone: 3, y: 2 }] }],
  stats: v => [`final centre ${fmt(v.T.at(-1), 2)}`, 'target 2'],
  takeaway: 'The two players’ gradients chase each other, so training can oscillate instead of settling.',
});
const MIX = (x, a) => 0.5 * normalPdf(x, Math.sqrt(a) * -2, Math.sqrt(1 - a + a * 0.09)) + 0.5 * normalPdf(x, Math.sqrt(a) * 2, Math.sqrt(1 - a + a * 0.09));
const SCHED = (() => { let a = 1; const o = []; for (let t = 1; t <= 100; t++) { a *= 1 - (1e-3 + (0.1 * (t - 1)) / 99); o.push(a); } return o; })();
const GRIDX = linspace(3.9, -3.9, 40), DENSITY = GRIDX.map(x => Array.from({ length: 50 }, (_, c) => r6(Math.sqrt(MIX(x, SCHED[2 * c])))));
export const DiffusionForwardLab = lab({
  intro: 'Two peaks, then repeated noise. The distribution relaxes into a standard normal.',
  tex: R`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon`,
  params: [{ k: 't', label: 'step', min: 1, max: 100, step: 1, init: 1, play: true }],
  derive: ({ t }) => ({ a: SCHED[t - 1] }),
  plots: v => [heatPlot(DENSITY, { w: 560, h: 280, caption: 'Density over the whole process: time runs right, x runs up', label: 'density by time and position', over: [heatBox(DENSITY, null, Math.floor((v.t - 1) / 2))], heat: { dense: true, name: (i, j, d) => `step ${2 * j + 1}, x ${GRIDX[i].toFixed(1)}: density ${(d * d).toFixed(3)}` } }), { x: [-4, 4], y: [0, 0.8], w: 480, h: 260, xl: 'x', label: 'noised distribution', marks: [{ tone: 1, data: curve(x => MIX(x, v.a), -4, 4) }, { tone: 'muted', dash: true, data: curve(x => normalPdf(x, 0, 1), -4, 4) }] }],
  stats: v => [`$\\bar\\alpha=${fmt(v.a, 4)}$`],
  takeaway: 'At the end nothing of the data remains, so sampling can start from plain noise.',
});
const score = (x, a) => { const s2 = 1 - a + a * 0.09, m = Math.sqrt(a) * 2, p1 = normalPdf(x, -m, Math.sqrt(s2)), p2 = normalPdf(x, m, Math.sqrt(s2)), t = p1 + p2 || 1e-300; return ((p1 * (-m - x)) / s2 + (p2 * (m - x)) / s2) / t; };
const reverseRun = memo(seed => { const r = rng(seed * 31); let X = Array.from({ length: 500 }, () => r6(gauss(r))); const keep = { 100: X }; let prev; for (let t = 100; t >= 1; t--) { const a = SCHED[t - 1], at = t > 1 ? SCHED[t - 2] : 1, b = 1 - a / at; prev = X.map(x => r6((x + b * score(x, a)) / Math.sqrt(1 - b) + (t > 1 ? Math.sqrt(b) * gauss(r) : 0))); X = prev; keep[t - 1] = X; } return keep; });
const reverseDensity = memo(seed => { const K = reverseRun(seed), C = Array.from({ length: 51 }, (_, c) => histDensity(K[100 - 2 * c], -4, 4, 40).reverse()); return C[0].map((_, i) => C.map(col => r6(Math.sqrt(col[i])))); });
export const DiffusionReverseLab = lab({
  intro: 'Start from noise and step backwards along the score. Here the score is known exactly, which a network would otherwise learn.',
  params: [{ k: 't', label: 'steps remaining', min: 0, max: 100, step: 1, init: 100 }, { k: 'seed', seed: 'New samples', init: 1 }],
  derive: ({ t, seed }) => ({ X: reverseRun(seed)[t], M: reverseDensity(seed) }),
  plots: v => [heatPlot(v.M, { w: 560, h: 280, caption: 'Samples through the reverse process: noise on the left, data on the right', label: 'sample density by time and position', over: [heatBox(v.M, null, Math.min(50, Math.floor((100 - v.t) / 2)))], heat: { dense: true, name: (i, j, d) => `${100 - 2 * j} steps remaining: density ${(d * d).toFixed(3)}` } }), { x: [-4, 4], y: [0, 0.8], w: 480, h: 260, xl: 'x', label: 'histogram of samples', marks: [{ as: 'rects', tone: 1, data: hist(v.X, -4, 4, 50) }] }],
  stats: v => [`${v.X.filter(x => x > 0).length} of 500 on the right`],
  takeaway: 'Reversing the noise, small step by small step, turns noise into samples of the data.',
});
export const GenerativeCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A linear autoencoder with squared loss learns the same subspace as PCA.', true, 'They minimise the same reconstruction error.'],
    ['The KL term in a VAE helps samples from the prior decode well.', true, 'It keeps the codes close to the prior.'],
    ['GANs maximise a likelihood directly.', false, 'They learn a judge instead.'],
    ['Diffusion networks usually predict the added noise.', true, 'Its loss is a simple squared error.'],
    ['Mode collapse means the generator covers everything.', false, 'It means the generator covers only part of the data.'],
  ],
  takeaway: 'Each family makes a different trade between stability, speed and sharpness.',
});

/* ---------- probabilistic ---------- */

export const McPi = lab({
  intro: 'Throw darts at a square. The fraction inside the circle estimates its area.',
  params: [{ k: 'n', label: 'darts', min: 50, max: 5000, step: 50, init: 500 }, { k: 'seed', seed: 'New darts', init: 1 }],
  derive: ({ n, seed }) => { const r = rng(seed * 3), pts = Array.from({ length: n }, () => [+r().toFixed(4), +r().toFixed(4)]), hit = pts.filter(p => p[0] * p[0] + p[1] * p[1] <= 1).length; return { pts, est: (4 * hit) / n }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 340, h: 320, label: 'darts in a unit square', marks: [{ as: 'points', r: 1.5, data: v.pts.slice(0, 1500).map(p => [p[0], p[1], p[0] * p[0] + p[1] * p[1] <= 1 ? 1 : 2]) }] }],
  stats: v => [`estimate ${fmt(v.est, 4)}`, `error ${fmt(Math.abs(v.est - Math.PI), 4)}`],
  takeaway: 'Monte Carlo error shrinks like $1/\\sqrt N$: a hundred times the samples for ten times the accuracy.',
});
export const InverseCdf = lab({
  intro: 'Feed a uniform number through the inverse CDF, and it becomes a sample from the target.',
  tex: R`x=F^{-1}(u)=-\ln(1-u)/\lambda`,
  params: [{ k: 'u', label: 'uniform $u$', min: 0.01, max: 0.99, step: 0.01, init: 0.5 }, { k: 'l', label: 'rate $\\lambda$', min: 0.3, max: 3, step: 0.1, init: 1 }],
  derive: ({ u, l }) => ({ x: -Math.log(1 - u) / l }),
  plots: v => [{ x: [0, 6], y: [0, 1], w: 460, h: 260, xl: 'x', label: 'exponential CDF', marks: [{ tone: 1, data: curve(x => 1 - Math.exp(-v.l * x), 0, 6) }, { tone: 4, dash: true, data: [[0, v.u], [Math.min(6, v.x), v.u], [Math.min(6, v.x), 0]] }] }],
  stats: v => [`sample $x=${fmt(v.x, 3)}$`],
  takeaway: 'Any distribution with an invertible CDF can be sampled this way.',
});
export const UcbDrill = drill({
  intro: 'An acquisition function balances a good mean against high uncertainty.',
  tex: R`\text{UCB}=\mu+\kappa\sigma`,
  gen: r => ({ m: int(r, 0, 5), s: pick(r, [0.5, 1, 2]), k: pick(r, [1, 2, 3]) }),
  q: v => `$\\mu=${v.m}$, $\\sigma=${v.s}$, $\\kappa=${v.k}$. Find the upper confidence bound.`,
  answer: v => v.m + v.k * v.s, dp: 2,
  steps: v => [`$\\kappa\\sigma=${v.k * v.s}$`],
});
export const AcceptDrill = drill({
  intro: 'With a symmetric proposal, only the ratio of densities matters.',
  tex: R`A=\min\big(1,\;p(x')/p(x)\big)`,
  gen: r => ({ a: pick(r, [0.1, 0.2, 0.4, 0.5]), b: pick(r, [0.05, 0.1, 0.3, 0.8]) }),
  q: v => `$p(x)=${v.a}$ and $p(x')=${v.b}$. Find the acceptance probability.`,
  answer: v => Math.min(1, v.b / v.a), dp: 3,
  steps: v => [`Ratio: $${fmt(v.b / v.a, 3)}$`],
});
export const ImportanceDrill = drill({
  intro: 'Reweight samples from the wrong distribution.',
  tex: R`w=p(x)/q(x)`,
  gen: r => ({ p: pick(r, [0.1, 0.3, 0.5]), q: pick(r, [0.2, 0.25, 0.4]), f: int(r, 1, 5) }),
  q: v => `$p=${v.p}$, $q=${v.q}$, $f(x)=${v.f}$. Find this sample’s contribution, $w\\,f(x)$.`,
  answer: v => (v.p / v.q) * v.f, dp: 3,
  steps: v => [`$w=${fmt(v.p / v.q, 3)}$`],
});
const BX = (() => { const r = rng(17); return Array.from({ length: 30 }, (_, i) => [+(i < 15 ? -2 + 2 * r() : 0.5 + r()).toFixed(3), 0]).map(p => [p[0], +(0.5 + 0.8 * p[0] + 0.4 * gauss(r)).toFixed(3)]); })();
const inv2 = ([[a, b], [c, d]]) => { const D = a * d - b * c; return [[d / D, -b / D], [-c / D, a / D]]; };
export const BlrLab = lab({
  intro: 'Not one best line but many plausible ones. Add data, and see the lines agree where you have looked.',
  tex: R`S^{-1}=\alpha I+\beta X^\top X,\qquad m=\beta SX^\top y`,
  params: [{ k: 'n', label: 'observations', min: 0, max: 30, step: 1, init: 4 }, { k: 'a', label: 'prior precision $\\alpha$', min: 0.1, max: 10, step: 0.1, init: 1 }, { k: 'seed', seed: 'New lines', init: 1 }],
  derive: ({ n, a, seed }) => { const beta = 1 / 0.16, D = BX.slice(0, n), XtX = [[0, 0], [0, 0]], Xty = [0, 0]; D.forEach(([x, y]) => { XtX[0][0] += 1; XtX[0][1] += x; XtX[1][0] += x; XtX[1][1] += x * x; Xty[0] += y; Xty[1] += x * y; }); const S = inv2([[a + beta * XtX[0][0], beta * XtX[0][1]], [beta * XtX[1][0], a + beta * XtX[1][1]]]), m = [beta * (S[0][0] * Xty[0] + S[0][1] * Xty[1]), beta * (S[1][0] * Xty[0] + S[1][1] * Xty[1])], l = Math.sqrt(S[0][0]), l2 = S[1][0] / l, l3 = Math.sqrt(Math.max(S[1][1] - l2 * l2, 1e-12)), r = rng(seed * 9); return { D, S, m, lines: Array.from({ length: 25 }, () => { const z1 = gauss(r), z2 = gauss(r); return [m[0] + l * z1, m[1] + l2 * z1 + l3 * z2]; }), beta }; },
  plots: v => [{ x: [-4, 4], y: [-4, 5], w: 460, h: 300, xl: 'x', yl: 'y', label: 'posterior lines', marks: [...v.lines.map(l => ({ tone: 1, soft: true, data: [[-4, l[0] - 4 * l[1]], [4, l[0] + 4 * l[1]]] })), { as: 'points', tone: 'fg', r: 4, data: v.D }] }],
  stats: v => [`posterior mean slope ${fmt(v.m[1], 3)}`, `slope uncertainty ${fmt(Math.sqrt(v.S[1][1]), 3)}`],
  takeaway: 'Uncertainty falls where data are, and stays large where they are not.',
});
const targetBi = x => 0.4 * normalPdf(x, -1.5, 0.6) + 0.6 * normalPdf(x, 1.6, 0.5);
export const RejectionLab = lab({
  intro: 'Draw under a box, and keep points that land under the curve. A tighter box wastes fewer draws.',
  params: [{ k: 'M', label: 'box height', min: 0.45, max: 1.5, step: 0.05, init: 0.8 }, { k: 'seed', seed: 'New draws', init: 1 }],
  derive: ({ M, seed }) => { const r = rng(seed * 11), pts = Array.from({ length: 800 }, () => [-4 + 8 * r(), M * r()]), acc = pts.filter(p => p[1] <= targetBi(p[0])); return { pts, acc, rate: acc.length / pts.length }; },
  plots: v => [{ x: [-4, 4], y: [0, v.M], w: 480, h: 280, xl: 'x', label: 'proposal and accepted points', marks: [{ as: 'points', tone: 'muted', r: 1.5, data: v.pts }, { as: 'points', tone: 1, r: 2, data: v.acc }, { tone: 4, data: curve(targetBi, -4, 4) }] }],
  stats: v => [`acceptance rate ${pct(v.rate, 1)}`],
  takeaway: 'Rejection sampling is exact, but it gets very wasteful as dimensions grow.',
});
const chain = memo((step, seed) => { const r = rng(seed * 5); let x = -3, acc = 0; const out = []; for (let t = 0; t < 2000; t++) { const p = r6(x + step * gauss(r)); if (r() < Math.min(1, targetBi(p) / (targetBi(x) || 1e-300))) { x = p; acc++; } out.push(x); } return { out, rate: acc / 2000 }; });
export const MetropolisLab = lab({
  intro: 'Wander with a proposal, and only sometimes accept. The step size makes the chain efficient or stuck.',
  params: [{ k: 's', label: 'step size', min: 0.05, max: 8, step: 0.05, init: 1 }, { k: 'seed', seed: 'New chain', init: 1 }],
  derive: ({ s, seed }) => chain(s, seed),
  plots: v => [{ caption: 'Trace', x: [0, 2000], y: [-4, 4], w: 320, h: 260, xl: 'step', label: 'trace of the chain', marks: [{ tone: 1, data: v.out.filter((_, i) => i % 4 === 0).map((x, i) => [i * 4, x]) }] }, { caption: 'Samples against target', x: [-4, 4], y: [0, 0.6], w: 300, h: 260, xl: 'x', label: 'histogram against target', marks: [{ as: 'rects', tone: 1, soft: true, data: hist(v.out.slice(200), -4, 4, 40) }, { tone: 4, data: curve(targetBi, -4, 4) }] }],
  stats: v => [`acceptance rate ${pct(v.rate, 1)}`],
  takeaway: 'Very small steps explore slowly, and very large ones are mostly rejected.',
});
const GPX = [-3.2, -1.9, -0.4, 1.1, 2.6], GPY = [-0.3, 0.9, 0.2, -0.8, 0.5];
const rbf = (a, b, l) => Math.exp(-((a - b) ** 2) / (2 * l * l));
function gp(xs, ys, l, nv, grid) { const n = xs.length, K = xs.map((a, i) => xs.map((b, j) => rbf(a, b, l) + (i === j ? nv : 0))); const alpha = n ? solve(K, ys) : []; return grid.map(x => { const k = xs.map(a => rbf(x, a, l)); if (!n) return [x, 0, 1]; const v = solve(K, k); return [x, k.reduce((s, c, i) => s + c * alpha[i], 0), Math.max(0, 1 - k.reduce((s, c, i) => s + c * v[i], 0))]; }); }
export const GpLab = lab({
  intro: 'A prior over functions, conditioned on points. Length-scale sets the smoothness, and uncertainty grows away from the data.',
  tex: R`\mu_*=K_*^\top(K+\sigma_n^2I)^{-1}y`,
  params: [{ k: 'l', label: 'length-scale $\\ell$', min: 0.2, max: 3, step: 0.05, init: 1 }, { k: 'n', label: 'noise variance', min: 0.0001, max: 0.5, step: 0.005, init: 0.01 }, { k: 'c', label: 'observations', min: 0, max: 5, step: 1, init: 5 }],
  derive: ({ l, n, c }) => ({ G: gp(GPX.slice(0, c), GPY.slice(0, c), l, n, linspace(-5, 5, 121)), xs: GPX.slice(0, c), ys: GPY.slice(0, c) }),
  plots: v => [{ x: [-5, 5], y: [-3, 3], w: 520, h: 280, xl: 'x', label: 'Gaussian process posterior', marks: [{ as: 'area', tone: 1, soft: true, data: v.G.map(([x, m, s]) => [x, m - 2 * Math.sqrt(s), m + 2 * Math.sqrt(s)]) }, { tone: 1, data: v.G.map(p => [p[0], p[1]]) }, { as: 'points', tone: 4, r: 5, data: v.xs.map((x, i) => [x, v.ys[i]]) }] }],
  stats: v => [`largest uncertainty ${fmt(Math.max(...v.G.map(p => 2 * Math.sqrt(p[2]))), 2)}`],
  takeaway: 'A Gaussian process gives a mean and honest uncertainty, at a cubic cost in the data.',
});
const hidden = x => Math.sin(1.6 * x) + 0.4 * Math.cos(4 * x) - 0.05 * x * x;
const boRun = memo(k => { const grid = linspace(-5, 5, 201), xs = [-2], ys = [hidden(-2)], frames = []; for (let t = 0; t < 12; t++) { const G = gp(xs, ys, 0.9, 0.001, grid), acq = G.map(([, m, s]) => m + k * Math.sqrt(s)), j = acq.indexOf(Math.max(...acq)); frames.push({ xs: [...xs], ys: [...ys], G, acq, next: grid[j] }); xs.push(+grid[j].toFixed(3)); ys.push(hidden(grid[j])); } return frames; });
export const BayesOptLab = lab({
  intro: 'Find the top of a hidden function using few tries. A larger $\\kappa$ explores more before settling.',
  tex: R`x_{\text{next}}=\arg\max_x\ \mu(x)+\kappa\sigma(x)`,
  params: [{ k: 'k', label: '$\\kappa$', min: 0, max: 5, step: 0.25, init: 2 }, { k: 't', label: 'evaluation', min: 0, max: 11, step: 1, init: 0, play: true }],
  derive: ({ k, t }) => ({ f: boRun(k)[t] }),
  plots: v => [{ x: [-5, 5], y: [-3, 3.5], w: 540, h: 300, xl: 'x', label: 'surrogate and acquisition', marks: [{ tone: 'muted', dash: true, data: curve(hidden, -5, 5) }, { as: 'area', tone: 1, soft: true, data: v.f.G.map(([x, m, s]) => [x, m - 2 * Math.sqrt(s), m + 2 * Math.sqrt(s)]) }, { tone: 1, data: v.f.G.map(p => [p[0], p[1]]) }, { as: 'rule', tone: 4, x: v.f.next }, { as: 'points', tone: 'fg', r: 4, data: v.f.xs.map((x, i) => [x, v.f.ys[i]]) }] }],
  stats: v => [`best so far ${fmt(Math.max(...v.f.ys), 3)}`, `true maximum about ${fmt(Math.max(...linspace(-5, 5, 2001).map(hidden)), 3)}`],
  takeaway: 'Bayesian optimisation suits expensive functions, such as tuning a model that takes hours to train.',
});
const gridZ = linspace(-5, 5, 401), dz = gridZ[1] - gridZ[0];
const biTarget = sep => z => 0.5 * normalPdf(z, -sep, 0.6) + 0.5 * normalPdf(z, sep, 0.6);
const klQ = (m, s, p) => gridZ.reduce((t, z) => { const q = normalPdf(z, m, s); return q > 1e-12 ? t + dz * q * Math.log(q / (p(z) + 1e-300)) : t; }, 0);
const viFit = memo(sep => { const p = biTarget(sep); let best = [0, 1, 1e9]; for (let m = -3.5; m <= 3.5; m += 0.1) for (let s = 0.2; s <= 3.5; s += 0.1) { const k = klQ(m, s, p); if (k < best[2]) best = [m, s, k]; } return best; });
export const VariationalGap = lab({
  intro: 'Fit one Gaussian to a two-peaked posterior by minimising KL. It locks on to a single peak and misses the other.',
  params: [{ k: 'sep', label: 'peak separation', min: 0, max: 3, step: 0.25, init: 2 }],
  derive: ({ sep }) => { const [m, s, k] = viFit(sep); return { m, s, k, p: biTarget(sep) }; },
  plots: v => [{ x: [-5, 5], y: [0, 0.9], w: 500, h: 260, xl: 'θ', label: 'target and variational fit', marks: [{ tone: 4, data: curve(v.p, -5, 5, 301) }, { tone: 1, data: curve(z => normalPdf(z, v.m, v.s), -5, 5, 301) }] }],
  stats: v => [`fitted mean ${fmt(v.m, 2)}, spread ${fmt(v.s, 2)}`, `KL ${fmt(v.k, 3)}`],
  legend: [['--c4', 'target'], ['--c1', 'variational fit']],
  takeaway: 'Variational inference is fast, but a too-simple family understates the uncertainty.',
});
const POOL = (() => { const r = rng(23); return Array.from({ length: 200 }, () => { const x = +(-4 + 8 * r()).toFixed(3); return [x, +(r() < 1 / (1 + Math.exp(-3 * (x - 0.7))))]; }); })();
const logistic1 = rows => { let w = 0, b = 0; for (let t = 0; t < 300; t++) { let gw = 0, gb = 0; rows.forEach(([x, y]) => { const e = 1 / (1 + Math.exp(-(w * x + b))) - y; gw += e * x; gb += e; }); w = r6(w - (0.3 * gw) / rows.length); b = r6(b - (0.3 * gb) / rows.length); } return [w, b]; };
const alRun = memo((mode, seed) => { const r = rng(seed * 29); let lab2 = [POOL[Math.floor(r() * 200)], POOL[Math.floor(r() * 200)]], left = POOL.slice(); const accs = []; for (let n = 2; n <= 30; n++) { const [w, b] = logistic1(lab2); accs.push([n, mean(POOL.map(([x, y]) => +(+(w * x + b > 0) === y)))]); const i = mode === 'rand' ? Math.floor(r() * left.length) : left.reduce((bi, p, j) => (Math.abs(w * p[0] + b) < Math.abs(w * left[bi][0] + b) ? j : bi), 0); lab2 = [...lab2, left[i]]; left = left.filter((_, j) => j !== i); } return accs; });
export const ActiveLearnLab = lab({
  intro: 'Labels are expensive. Compare asking at random with asking about what the model is least sure of.',
  params: [{ k: 'seed', seed: 'New run', init: 1 }],
  derive: ({ seed }) => ({ A: alRun('active', seed), B: alRun('rand', seed) }),
  plots: v => [{ x: [2, 30], y: [0.4, 1], w: 520, h: 280, xl: 'labels used', yl: 'accuracy', label: 'accuracy against labels', marks: [{ tone: 1, data: v.A }, { tone: 4, data: v.B }] }],
  stats: v => [`after 30 labels: uncertain ${pct(v.A.at(-1)[1], 1)}, random ${pct(v.B.at(-1)[1], 1)}`],
  legend: [['--c1', 'ask where uncertain'], ['--c4', 'ask at random']],
  takeaway: 'Choosing what to label can reach the same accuracy with far fewer labels.',
});
const EPTS = (() => { const r = rng(37); return Array.from({ length: 24 }, () => { const x = +(-2 + 4 * r()).toFixed(3); return [x, +(Math.sin(x) + 0.15 * gauss(r)).toFixed(3)]; }); })();
const polyFit = (D, deg) => { const n = deg + 1, X = D.map(([x]) => Array.from({ length: n }, (_, k) => x ** k)), G = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => X.reduce((s, r0) => s + r0[a] * r0[b], 0) + (a === b ? 1e-6 : 0))); return solve(G, Array.from({ length: n }, (_, a) => X.reduce((s, r0, i) => s + r0[a] * D[i][1], 0))); };
const ensembleRun = memo(deg => { const r = rng(deg * 41 + 1); return Array.from({ length: 20 }, () => polyFit(Array.from({ length: EPTS.length }, () => EPTS[Math.floor(r() * EPTS.length)]), deg)); });
export const EnsembleSpread = lab({
  intro: 'Fit many models on resampled data. Where they disagree, the prediction is uncertain.',
  params: [{ k: 'd', label: 'polynomial degree', min: 1, max: 9, step: 1, init: 5 }],
  derive: ({ d }) => { const C = ensembleRun(d), xs = linspace(-4, 4, 121), P = C.map(c => xs.map(x => Math.max(-6, Math.min(6, c.reduce((s, w, k) => s + w * x ** k, 0))))); return { xs, P, sdAt: x => { const i = xs.findIndex(a => a >= x), p = P.map(l => l[i]), m = mean(p); return Math.sqrt(mean(p.map(v => (v - m) ** 2))); } }; },
  plots: v => [{ x: [-4, 4], y: [-4, 4], w: 520, h: 280, xl: 'x', label: 'ensemble of fits', marks: [...v.P.map(l => ({ tone: 1, soft: true, data: l.map((y, i) => [v.xs[i], y]) })), { as: 'points', tone: 'fg', r: 3, data: EPTS }] }],
  stats: v => [`spread at x = 0: ${fmt(v.sdAt(0), 3)}`, `spread at x = 3.5: ${fmt(v.sdAt(3.5), 3)}`],
  takeaway: 'Disagreement estimates epistemic uncertainty, which is largest away from the data.',
});
export const ProbCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A Gaussian prior on weights makes MAP equal to ridge regression.', true, 'The log prior is a squared penalty.'],
    ['More data reduces aleatoric uncertainty.', false, 'It is noise in the data itself.'],
    ['MCMC needs the normalising constant.', false, 'Only ratios of the density appear.'],
    ['Monte Carlo error depends directly on the dimension.', false, 'It shrinks like $1/\\sqrt N$, though the constant can depend on the problem.'],
    ['A Gaussian process needs a cubic amount of work in the number of points.', true, 'Exact inference solves an $n\\times n$ system.'],
  ],
  takeaway: 'Treat unknowns as distributions, and use the uncertainty to decide what to do.',
});

/* ---------- systems ---------- */

const LAT = (() => { const r = rng(44); return Array.from({ length: 2000 }, () => +(Math.exp(3 + 0.6 * gauss(r)) + (r() < 0.02 ? 400 * r() : 0)).toFixed(2)).sort((a, b) => a - b); })();
export const LatencyPercentiles = lab({
  intro: 'An average can hide the worst requests. Move through the percentiles of a heavy-tailed latency.',
  params: [{ k: 'p', label: 'percentile', min: 50, max: 99.9, step: 0.1, init: 99 }],
  derive: ({ p }) => ({ v: LAT[Math.min(1999, Math.floor((p / 100) * 2000))], m: mean(LAT) }),
  plots: v => [{ x: [0, 250], y: [0, 0.04], w: 520, h: 260, xl: 'milliseconds', label: 'latency histogram', marks: [{ as: 'rects', tone: 1, data: hist(LAT, 0, 250, 60) }, { as: 'rule', tone: 4, x: Math.min(250, v.v) }, { as: 'rule', dash: true, tone: 'muted', x: v.m }] }],
  stats: v => [`P${v.p}: ${fmt(v.v, 1)} ms`, `mean ${fmt(v.m, 1)} ms`],
  takeaway: 'Set targets on tail percentiles, since those are what unhappy users experience.',
});
export const PipelineTruths = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Fixing seeds and library versions helps reproduce a result.', true, 'Both change results if left free.'],
    ['Computing features differently in training and production is harmless.', false, 'This training-serving skew can silently degrade a model.'],
    ['Most production failures come from the model architecture.', false, 'Data pipelines and integration cause many of them.'],
    ['A held-out test score guarantees future performance.', false, 'It assumes the future resembles the test set.'],
  ],
  takeaway: 'The model is one part of a system that has to be engineered.',
});
export const PsiDrill = drill({
  intro: 'One bin of the population stability index.',
  tex: R`\text{PSI}=\sum_i(p_i-q_i)\ln\frac{p_i}{q_i}`,
  gen: r => ({ p: pick(r, [0.5, 0.3, 0.2, 0.1]), q: pick(r, [0.6, 0.25, 0.35, 0.15]) }),
  q: v => `A bin has $p=${v.p}$ and $q=${v.q}$. Find its PSI term.`,
  answer: v => (v.p - v.q) * Math.log(v.p / v.q), dp: 4,
  steps: v => [`$p-q=${fmt(v.p - v.q, 2)}$`, `$\\ln(p/q)=${fmt(Math.log(v.p / v.q), 4)}$`],
});
const BINS = linspace(-3, 3, 11);
const binP = (m, s) => BINS.slice(0, -1).map((a, i) => { const Phi = x => 0.5 * (1 + Math.sign(x) * Math.sqrt(1 - Math.exp((-2 * x * x) / Math.PI))); return Math.max(1e-4, Phi((BINS[i + 1] - m) / s) - Phi((a - m) / s)); });
export const PsiLab = lab({
  intro: 'Shift the live data away from the training reference, and watch the stability index climb.',
  params: [{ k: 'm', label: 'mean shift', min: -1.5, max: 1.5, step: 0.05, init: 0 }, { k: 's', label: 'spread', min: 0.5, max: 2, step: 0.05, init: 1 }],
  derive: ({ m, s }) => { const q = binP(0, 1), p = binP(m, s); return { q, p, psi: p.reduce((t, a, i) => t + (a - q[i]) * Math.log(a / q[i]), 0) }; },
  plots: v => [{ x: [-3, 3], y: [0, 0.4], w: 480, h: 260, xl: 'bin', label: 'reference and live distributions', marks: [{ as: 'rects', tone: 'muted', soft: true, data: v.q.map((a, i) => [BINS[i] + 0.03, 0, BINS[i + 1] - 0.03, a]) }, { as: 'rects', tone: 1, soft: true, data: v.p.map((a, i) => [BINS[i] + 0.15, 0, BINS[i + 1] - 0.15, a]) }] }],
  stats: v => [`PSI ${fmt(v.psi, 3)}`],
  status: v => (v.psi > 0.25 ? ['alert', 'Major shift.'] : v.psi > 0.1 ? ['alert', 'Moderate shift: keep watching.'] : ['ok', 'Stable.']),
  takeaway: 'Labels may arrive weeks later, so input drift is often the first warning.',
});
export const AbSeDrill = drill({
  intro: 'The standard error of a difference between two rates.',
  tex: R`\text{SE}=\sqrt{\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}}`,
  gen: r => ({ a: pick(r, [0.1, 0.2, 0.5]), b: pick(r, [0.12, 0.25, 0.52]), n: pick(r, [100, 400, 1000]) }),
  q: v => `$p_1=${v.a}$, $p_2=${v.b}$, $n_1=n_2=${v.n}$. Find the standard error.`,
  answer: v => Math.sqrt((v.a * (1 - v.a)) / v.n + (v.b * (1 - v.b)) / v.n), dp: 4,
  steps: v => [`First term: $${fmt((v.a * (1 - v.a)) / v.n, 5)}$`, `Second term: $${fmt((v.b * (1 - v.b)) / v.n, 5)}$`],
});
const peeking = memo(looks => { const r = rng(looks * 3 + 7); let fp = 0; const T = 1500; for (let s = 0; s < T; s++) { let t = 0, n = 0, hit = false; for (let l = 1; l <= looks; l++) { for (let i = 0; i < 30; i++) { t += gauss(r); n++; } if (Math.abs(t / Math.sqrt(n)) > 1.96) { hit = true; break; } } if (hit) fp++; } return fp / T; });
export const PeekingLab = lab({
  intro: 'There is no real effect in this simulated test. Looking repeatedly and stopping at significance still “finds” one.',
  params: [{ k: 'k', label: 'number of looks', min: 1, max: 20, step: 1, init: 5 }],
  derive: ({ k }) => ({ rate: peeking(k), C: [1, 2, 3, 5, 8, 12, 20].map(l => [l, peeking(l)]) }),
  plots: v => [{ x: [1, 20], y: [0, 0.35], w: 480, h: 260, xl: 'looks', yl: 'false positive rate', label: 'false positive rate against looks', marks: [{ tone: 1, data: v.C }, { as: 'rule', dash: true, tone: 'muted', y: 0.05 }, { as: 'points', tone: 4, r: 6, data: [[v.k, v.rate]] }] }],
  stats: v => [`false positive rate ${pct(v.rate, 1)}`, 'nominal rate 5%'],
  takeaway: 'Fix the sample size in advance, or use a method designed for repeated looks.',
});
const G_DATA = (() => { const r = rng(52); const mk = (n, base, shift) => Array.from({ length: n }, () => { const y = +(r() < base); return [+(y ? 0.6 + shift + 0.18 * gauss(r) : 0.35 + shift + 0.18 * gauss(r)).toFixed(3), y]; }); return { a: mk(500, 0.5, 0), b: mk(500, 0.25, -0.05) }; })();
const groupRates = (rows, t) => { const tp = rows.filter(([s, y]) => y && s >= t).length, fp = rows.filter(([s, y]) => !y && s >= t).length, pos = rows.filter(r0 => r0[1]).length, neg = rows.length - pos; return { tpr: tp / pos, fpr: fp / neg, ppr: (tp + fp) / rows.length }; };
export const FairnessLab = lab({
  intro: 'Two groups with different base rates and score distributions. Move each threshold, and see which criteria hold.',
  params: [{ k: 'a', label: 'threshold, group A', min: 0, max: 1, step: 0.01, init: 0.5 }, { k: 'b', label: 'threshold, group B', min: 0, max: 1, step: 0.01, init: 0.5 }],
  derive: ({ a, b }) => ({ A: groupRates(G_DATA.a, a), B: groupRates(G_DATA.b, b) }),
  plots: v => [{ x: [0, 3], y: [0, 1], w: 460, h: 260, xl: 'selection rate, true positive rate, false positive rate', label: 'rates per group', marks: [{ as: 'rects', tone: 1, data: [[0.1, 0, 0.45, v.A.ppr], [1.1, 0, 1.45, v.A.tpr], [2.1, 0, 2.45, v.A.fpr]] }, { as: 'rects', tone: 2, data: [[0.5, 0, 0.85, v.B.ppr], [1.5, 0, 1.85, v.B.tpr], [2.5, 0, 2.85, v.B.fpr]] }] }],
  table: v => ({ head: ['', 'group A', 'group B', 'gap'], rows: [['selection rate', fmt(v.A.ppr, 3), fmt(v.B.ppr, 3), fmt(Math.abs(v.A.ppr - v.B.ppr), 3)], ['true positive rate', fmt(v.A.tpr, 3), fmt(v.B.tpr, 3), fmt(Math.abs(v.A.tpr - v.B.tpr), 3)], ['false positive rate', fmt(v.A.fpr, 3), fmt(v.B.fpr, 3), fmt(Math.abs(v.A.fpr - v.B.fpr), 3)]] }),
  stats: v => [`equal opportunity gap ${fmt(Math.abs(v.A.tpr - v.B.tpr), 3)}`],
  legend: [['--c1', 'group A'], ['--c2', 'group B']],
  takeaway: 'With different base rates, these criteria usually cannot all hold at once. Which one matters is a decision about values.',
});
export const CounterfactualDrill = drill({
  intro: 'For a linear model, the distance to the decision boundary has a closed form.',
  tex: R`d=\frac{|w^\top x+b|}{\|w\|}`,
  gen: r => { const [a, c, n] = pick(r, [[3, 4, 5], [6, 8, 10], [5, 12, 13]]); return { a, c, n, x: [int(r, -2, 4), int(r, -2, 4)], b: int(r, -6, 6) }; },
  q: v => `$w=(${v.a},${v.c})$, $b=${v.b}$, $x=(${v.x})$. Find the smallest change that reaches the boundary.`,
  answer: v => Math.abs(v.a * v.x[0] + v.c * v.x[1] + v.b) / v.n, dp: 3,
  steps: v => [`$w^\\top x+b=${v.a * v.x[0] + v.c * v.x[1] + v.b}$`, `$\\|w\\|=${v.n}$`],
});
export const CounterfactualLab = lab({
  intro: 'A loan model. Drag the applicant, and see the smallest change to the inputs that flips the decision.',
  params: [{ k: 'x', label: 'income', min: 0, max: 10, step: 0.05, init: 3 }, { k: 'y', label: 'credit score', min: 0, max: 10, step: 0.05, init: 4 }],
  derive: ({ x, y }) => { const w = [0.8, 0.6], b = -6, s = w[0] * x + w[1] * y + b; return { w, s, foot: [x - s * w[0], y - s * w[1]], d: Math.abs(s) }; },
  plots: v => [{ x: [0, 10], y: [0, 10], w: 400, h: 340, xl: 'income', yl: 'credit score', label: 'decision boundary and counterfactual', marks: [{ as: 'field', bands: 1, tones: [3, 4], fn: (a, b) => (0.8 * a + 0.6 * b - 6 > 0 ? 0.3 : -0.3) }, { tone: 'fg', data: curve(a => (6 - 0.8 * a) / 0.6, 0, 10) }, { as: 'arrow', tone: 1, data: [[v.x, v.y, v.foot[0], v.foot[1]]] }, { drag: { x: 'x', y: 'y', tone: 2 } }] }],
  stats: v => [v.s >= 0 ? 'approved' : 'denied', `distance to the boundary ${fmt(v.d, 3)}`, `move to $(${fmt(v.foot[0], 2)},\\ ${fmt(v.foot[1], 2)})$`],
  takeaway: 'A counterfactual gives a person something actionable, but it does not say that the feature causes the outcome.',
});
const WTS = (() => { const r = rng(61); return Array.from({ length: 400 }, () => +(0.6 * gauss(r)).toFixed(5)); })();
export const QuantizeLab = lab({
  intro: 'Store weights in fewer bits. Size falls linearly, and error grows slowly until very few bits remain.',
  params: [{ k: 'b', label: 'bits', min: 1, max: 16, step: 1, init: 8 }],
  derive: ({ b }) => { const lo = Math.min(...WTS), hi = Math.max(...WTS), L = 2 ** b - 1, Q = WTS.map(w => lo + (Math.round(((w - lo) / (hi - lo)) * L) / L) * (hi - lo)); return { Q, mse: mean(WTS.map((w, i) => (w - Q[i]) ** 2)), ratio: 32 / b }; },
  plots: v => [{ x: [-2.2, 2.2], y: [0, 1], w: 500, h: 260, xl: 'weight', label: 'quantised weight histogram', marks: [{ as: 'rects', tone: 1, data: hist(v.Q, -2.2, 2.2, 64) }] }],
  stats: v => [`${fmt(v.ratio, 1)} times smaller than 32-bit`, `mean squared error ${v.mse.toExponential(2)}`],
  takeaway: 'Eight bits usually cost little accuracy, and shrink a model four times.',
});
export const SystemsCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Data drift and concept drift are the same thing.', false, 'The first changes $p(x)$, the second $p(y\\mid x)$.'],
    ['Stopping an A/B test as soon as it looks significant is safe.', false, 'It inflates the false positive rate.'],
    ['A feature’s importance to a model shows it causes the outcome.', false, 'It only shows that the model relies on it.'],
    ['With different base rates, calibration and equal error rates usually conflict.', true, 'They generally cannot all hold at once.'],
    ['An agent should be given the least permissions it needs.', true, 'This limits the damage from a mistake or an injection.'],
    ['Text retrieved from the web can contain hidden instructions.', true, 'This is prompt injection, so it should be treated as data.'],
  ],
  takeaway: 'Deployment is engineering, statistics and accountability together.',
});
