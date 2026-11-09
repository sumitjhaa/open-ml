'use client';
// Explorers for convolution, invariance, class activation maps and detection.
import { R, drill, int, lab, pick, truths } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { fmt, memo, pct } from '../lib/models.mjs';
import { IMG5, KERNEL3, LENET, conv1d, conv2d, convParams, iou, nms, outSide, pool2d } from '../lib/extended.mjs';

// Draws a matrix as shaded cells with its numbers. Rows run top to bottom.
const cells = (M, x0, y0, tone, hi) => { const n = M.length, peak = Math.max(1, ...M.flat().map(Math.abs)); return M.flatMap((row, i) => row.flatMap((v, j) => [{ as: 'rects', tone: v < 0 ? 4 : tone, soft: Math.abs(v) / peak < 0.5, data: [[x0 + j + 0.04, y0 + n - i - 0.96, x0 + j + 0.96, y0 + n - i - 0.04]] }, { as: 'label', x: x0 + j + 0.5, y: y0 + n - i - 0.62, text: String(Number.isInteger(v) ? v : +v.toFixed(1)), anchor: 'middle' }])).concat(hi ? [{ tone: 'fg', data: [[x0 + hi[1], y0 + n - hi[0]], [x0 + hi[1] + hi[2], y0 + n - hi[0]], [x0 + hi[1] + hi[2], y0 + n - hi[0] - hi[2]], [x0 + hi[1], y0 + n - hi[0] - hi[2]], [x0 + hi[1], y0 + n - hi[0]]] }] : []); };

export const ConvByHand = lab({
  intro: 'Slide the filter across the image. Each output is the sum of a window multiplied element by element with the filter.',
  params: [{ k: 's', label: 'filter position', min: 0, max: 8, step: 1, init: 0, play: true }],
  derive: ({ s }) => { const r = Math.floor(s / 3), c = s % 3, O = conv2d(IMG5, KERNEL3); return { r, c, O, win: IMG5.slice(r, r + 3).map(row => row.slice(c, c + 3)) }; },
  plots: v => [{ x: [0, 13], y: [0, 5.2], w: 620, h: 250, label: 'image, filter and output', marks: [...cells(IMG5, 0, 0, 1, [v.r, v.c, 3]), ...cells(KERNEL3, 6, 1, 2), ...cells(v.O.map((row, i) => row.map((o, j) => (i * 3 + j <= v.r * 3 + v.c ? o : 0))), 10, 1, 3)] }],
  stats: v => [`output ${v.O[v.r][v.c]}`, `$${v.win.flatMap((row, i) => row.map((p, j) => `${p}\\cdot${KERNEL3[i][j]}`)).join('+')}$`],
  takeaway: 'One small filter is reused at every position. That is why convolutions need so few weights.',
});
export const ConvCellDrill = drill({
  intro: 'Multiply matching cells and add.',
  gen: r => ({ w: Array.from({ length: 2 }, () => [int(r, 0, 4), int(r, 0, 4)]), k: Array.from({ length: 2 }, () => [int(r, -2, 2), int(r, -2, 2)]) }),
  q: v => `Window $\\begin{bmatrix}${v.w[0].join('&')}\\\\${v.w[1].join('&')}\\end{bmatrix}$, filter $\\begin{bmatrix}${v.k[0].join('&')}\\\\${v.k[1].join('&')}\\end{bmatrix}$. Find the output.`,
  answer: v => v.w.flat().reduce((s, a, i) => s + a * v.k.flat()[i], 0), dp: 0,
  steps: v => [v.w.flat().map((a, i) => `${a}\\times${v.k.flat()[i]}`).map(t => `$${t}$`).join(', ')],
});
const IMG = (() => { const n = 14, g = Array.from({ length: n }, () => Array(n).fill(0)); for (let i = 2; i < 12; i++) for (let j = 2; j < 12; j++) if (j <= i) g[i][j] = 1; return g; })();
const FILTERS = { vertical: [[1, 0, -1], [1, 0, -1], [1, 0, -1]], horizontal: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] };
const heat = (M, ox, oy, tone = 1) => { const n = M.length, peak = Math.max(1e-9, ...M.flat().map(Math.abs)); return M.flatMap((row, i) => row.map((v, j) => ({ as: 'rects', tone: v < 0 ? 4 : tone, soft: Math.abs(v) / peak < 0.5, data: Math.abs(v) / peak < 0.08 ? [] : [[ox + j, oy + n - i - 1, ox + j + 1, oy + n - i]] }))); };
export const EdgeFilterLab = lab({
  intro: 'A filter is a pattern detector. Choose one and see where in the image it responds.',
  params: [{ k: 'f', label: 'Filter', options: Object.keys(FILTERS).map(k => [k, k]), init: 'vertical' }],
  derive: ({ f }) => ({ O: conv2d(IMG, FILTERS[f]), K: FILTERS[f] }),
  plots: v => [{ caption: 'Image', x: [0, 14], y: [0, 14], w: 240, h: 240, label: 'input image', marks: heat(IMG, 0, 0) }, { caption: 'Filter', x: [0, 3], y: [0, 3], w: 180, h: 240, label: 'filter values', marks: cells(v.K, 0, 0, 2) }, { caption: 'Response', x: [0, 12], y: [0, 12], w: 240, h: 240, label: 'filter response', marks: heat(v.O, 0, 0, 3) }],
  stats: v => [`largest response ${Math.max(...v.O.flat())}`, `smallest response ${Math.min(...v.O.flat())}`],
  takeaway: 'Training learns filter values like these, without anyone designing them.',
});
export const OutputSizeDrill = drill({
  intro: 'Padding adds pixels on both sides, and stride skips positions.',
  tex: R`\Big\lfloor\frac{n+2p-f}{s}\Big\rfloor+1`,
  gen: r => ({ n: pick(r, [28, 32, 64, 100, 224]), f: pick(r, [3, 5, 7]), p: int(r, 0, 3), s: pick(r, [1, 2]) }),
  q: v => `$n=${v.n}$, $f=${v.f}$, $p=${v.p}$, $s=${v.s}$. Find the output side.`,
  answer: v => outSide(v.n, v.f, v.p, v.s), dp: 0,
  steps: v => [`$n+2p-f=${v.n + 2 * v.p - v.f}$`, `Divide by $${v.s}$, round down, then add 1.`],
});
export const StridePadding = lab({
  intro: 'Padding keeps the border pixels in play, and stride takes bigger steps. Watch the output shrink or hold steady.',
  tex: R`\text{side}=\lfloor(n+2p-f)/s\rfloor+1`,
  params: [{ k: 'n', label: 'input side $n$', min: 4, max: 12, step: 1, init: 5 }, { k: 'f', label: 'filter size $f$', min: 1, max: 5, step: 2, init: 3 }, { k: 'p', label: 'padding $p$', min: 0, max: 3, step: 1, init: 0 }, { k: 's', label: 'stride $s$', min: 1, max: 3, step: 1, init: 1 }],
  derive: ({ n, f, p, s }) => { const side = outSide(n, f, p, s), starts = Array.from({ length: Math.max(0, side) }, (_, i) => i * s - p); return { side, starts }; },
  plots: v => { const m = v.n + 2 * v.p; return [{ x: [0, m], y: [0, m], w: 340, h: 340, label: 'filter positions on the grid', marks: [...Array.from({ length: m + 1 }, (_, i) => ({ tone: 'muted', soft: true, data: [[i, 0], [i, m]] })), ...Array.from({ length: m + 1 }, (_, i) => ({ tone: 'muted', soft: true, data: [[0, i], [m, i]] })), { as: 'rects', tone: 1, soft: true, data: [[v.p, v.p, v.p + v.n, v.p + v.n]] }, ...v.starts.slice(0, 1).map(c => ({ as: 'rects', tone: 2, soft: true, data: [[c + v.p, m - v.p - v.f, c + v.p + v.f, m - v.p]] })), { as: 'points', tone: 4, r: 3, data: v.starts.flatMap(a => v.starts.map(b => [a + v.p + v.f / 2, m - (b + v.p) - v.f / 2])) }] }]; },
  stats: v => [`output side ${v.side}`],
  note: 'Each red dot is one filter position. Blue is the image, and the white margin is padding.',
  takeaway: '"Same" padding, $p=(f-1)/2$ with stride 1, keeps the side.',
});
export const ConvParamsDrill = drill({
  intro: 'Each filter has $f^2c_{\\text{in}}$ weights and one bias.',
  tex: R`c_{\text{out}}(f^2c_{\text{in}}+1)`,
  gen: r => ({ c: pick(r, [6, 10, 16, 32]), f: pick(r, [3, 5]), i: pick(r, [1, 3]) }),
  q: v => `$${v.c}$ filters of size $${v.f}\\times${v.f}$ on $${v.i}$ input channel${v.i > 1 ? 's' : ''}. Count the parameters.`,
  answer: v => convParams(v.c, v.f, v.i), dp: 0,
  steps: v => [`Per filter: $${v.f}^2\\times${v.i}+1=${v.f * v.f * v.i + 1}$`],
});
export const PoolingLab = lab({
  intro: 'Pooling summarises each window with no weights at all.',
  params: [{ k: 'k', label: 'Kind', options: [['max', 'max'], ['avg', 'average']], init: 'max' }, { k: 'f', label: 'window', min: 2, max: 3, step: 1, init: 3 }, { k: 's', label: 'stride', min: 1, max: 3, step: 1, init: 1 }],
  derive: ({ k, f, s }) => ({ O: pool2d(IMG5, f, s, k) }),
  plots: v => [{ x: [0, 11], y: [0, 5.2], w: 560, h: 250, label: 'input and pooled output', marks: [...cells(IMG5, 0, 0, 1), ...cells(v.O, 7, 1, 3)] }],
  stats: v => [`output ${v.O.length}×${v.O[0].length}`],
  takeaway: 'Pooling shrinks the map and makes the result change less when a pattern shifts slightly.',
});
export const ReceptiveField = lab({
  intro: 'A deeper unit depends on a larger patch of the input. Stacks of small filters grow it slowly, and strides speed it up.',
  tex: R`r\leftarrow r+(f-1)\,j,\qquad j\leftarrow j\,s`,
  params: [{ k: 'L', label: 'layers', min: 1, max: 8, step: 1, init: 3 }, { k: 'f', label: 'filter size', min: 3, max: 7, step: 2, init: 3 }, { k: 's', label: 'stride after each layer', min: 1, max: 2, step: 1, init: 1 }],
  derive: ({ L, f, s }) => { let r = 1, j = 1; const R0 = [[0, 1]]; for (let i = 1; i <= L; i++) { r += (f - 1) * j; j *= s; R0.push([i, r]); } return { R0, r }; },
  plots: v => [{ x: [0, 8], y: [0, 300], w: 420, h: 280, xl: 'layer', yl: 'receptive field', label: 'receptive field by layer', marks: [{ tone: 1, data: v.R0 }, { as: 'points', tone: 4, r: 5, data: v.R0 }] }],
  stats: v => [`a unit sees ${v.r}×${v.r} input pixels`],
  takeaway: 'Early layers see edges. Later layers see parts, because their receptive fields are larger.',
});
export const LenetTable = lab({
  intro: 'LeNet-5, layer by layer. Choose a layer and see its share of the parameters.',
  params: [{ k: 'i', label: 'layer', min: 0, max: LENET.length - 1, step: 1, init: 1, show: v => LENET[v][0] }],
  plots: v => [{ x: [0, LENET.length], y: [0, 50000], w: 420, h: 260, xl: 'layer', yl: 'parameters', label: 'parameters per layer', marks: [{ as: 'rects', tone: 1, data: LENET.map((l, i) => [i + 0.15, 0, i + 0.85, l[2]]) }, { as: 'rects', tone: 4, data: [[v.i + 0.15, 0, v.i + 0.85, LENET[v.i][2]]] }] }],
  stats: v => [`${LENET[v.i][2].toLocaleString('en-IN')} parameters`, `output shape ${LENET[v.i][1]}`, 'total 61,706, with only 2,572 in the convolutions'],
  takeaway: 'Sharing makes convolutional layers tiny. Most of LeNet’s weights are in its dense layers.',
});
const SIG = Array.from({ length: 20 }, (_, i) => (i > 5 && i < 12 ? 1 : i >= 14 && i < 17 ? 1 : -1));
const K1 = { 'falling edge': [1, 0, -1], 'rising edge': [-1, 0, 1], 'high level': [1, 1, 1], 'low level': [-1, -1, -1] };
export const Conv1dLab = lab({
  intro: 'A filter sliding along time. Choose one and see which moments it responds to.',
  params: [{ k: 'k', label: 'Filter', options: Object.keys(K1).map(k => [k, k]), init: 'rising edge' }],
  derive: ({ k }) => ({ o: conv1d(SIG, K1[k]).map(v => Math.max(0, v)) }),
  plots: v => [{ caption: 'Signal', x: [0, 20], y: [-1.3, 1.3], w: 620, h: 150, label: 'input signal', marks: [{ tone: 1, data: SIG.map((s, i) => [i, s]) }] }, { caption: 'ReLU of the response', x: [0, 20], y: [0, 3.2], w: 620, h: 150, label: 'filter response', marks: [{ as: 'rects', tone: 2, data: v.o.map((a, i) => [i + 0.6, 0, i + 1.4, a]) }] }],
  stats: v => [`filter $[${K1[v.k]}]$`],
  takeaway: 'The same idea works along time, so convolution suits audio and sensor signals too.',
});
const SHIFTIMG = (() => { const r = Array.from({ length: 12 }, () => Array(12).fill(0)); [[3, 3], [4, 3], [5, 3], [6, 3], [6, 4], [6, 5]].forEach(([a, b]) => { r[a][b] = 1; }); return r; })();
const shift = (M, k) => M.map(row => row.map((_, j) => row[j - k] ?? 0));
const EK = [[1, 0, -1], [1, 0, -1], [1, 0, -1]];
export const EquivarianceLab = lab({
  intro: 'Shift the input. Convolving then shifting matches shifting then convolving. Max pooling breaks the match.',
  tex: R`f(S(x))=S(f(x))`,
  params: [{ k: 's', label: 'shift', min: 0, max: 3, step: 1, init: 1 }, { k: 'p', label: 'Layer', options: [['conv', 'convolution'], ['pool', 'convolution then max pool']], init: 'conv' }],
  derive: ({ s, p }) => { const f = M => (p === 'conv' ? conv2d(M, EK, 1, 1) : pool2d(conv2d(M, EK, 1, 1), 2, 2)), A = f(shift(SHIFTIMG, s)), B = p === 'conv' ? shift(f(SHIFTIMG), s) : shift(f(SHIFTIMG), Math.round(s / 2)), diff = A.flat().reduce((t, v, i) => t + Math.abs(v - B.flat()[i]), 0); return { A, B, diff }; },
  plots: v => [{ caption: 'Shift, then apply', x: [0, v.A.length], y: [0, v.A.length], w: 280, h: 260, label: 'shift then convolve', marks: heat(v.A, 0, 0, 1) }, { caption: 'Apply, then shift', x: [0, v.B.length], y: [0, v.B.length], w: 280, h: 260, label: 'convolve then shift', marks: heat(v.B, 0, 0, 3) }],
  stats: v => [`total difference ${v.diff}`],
  status: v => (v.diff === 0 ? ['ok', 'Identical: this layer is equivariant to the shift.'] : ['alert', 'Different: the shift is lost in the coarser grid.']),
  takeaway: 'Convolution moves with its input. Downsampling only approximately does.',
});
const MAPS = (() => { const n = 10, m = f => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +f(i, j).toFixed(3))); return { ears: m((i, j) => Math.exp(-((i - 2) ** 2 + (j - 2) ** 2) / 4) + Math.exp(-((i - 2) ** 2 + (j - 7) ** 2) / 4)), whiskers: m((i, j) => Math.exp(-((i - 6) ** 2) / 2 - ((j - 4.5) ** 2) / 14)), background: m((i, j) => +((i + j) / 18)) }; })();
export const CamLab = lab({
  intro: 'A class activation map is a weighted sum of feature maps. Change how much each feature counts for the class.',
  tex: R`L^c=\text{ReLU}\Big(\sum_k\alpha_k^cA_k\Big)`,
  params: [{ k: 'a', label: 'weight, ears', min: -1, max: 2, step: 0.1, init: 1.2 }, { k: 'b', label: 'weight, whiskers', min: -1, max: 2, step: 0.1, init: 0.8 }, { k: 'c', label: 'weight, background', min: -1, max: 2, step: 0.1, init: -0.4 }],
  derive: ({ a, b, c }) => ({ L: MAPS.ears.map((row, i) => row.map((v, j) => Math.max(0, a * v + b * MAPS.whiskers[i][j] + c * MAPS.background[i][j]))) }),
  plots: v => [{ caption: 'Ears', x: [0, 10], y: [0, 10], w: 170, h: 180, label: 'ears map', marks: heat(MAPS.ears, 0, 0, 1) }, { caption: 'Whiskers', x: [0, 10], y: [0, 10], w: 170, h: 180, label: 'whiskers map', marks: heat(MAPS.whiskers, 0, 0, 1) }, { caption: 'Background', x: [0, 10], y: [0, 10], w: 170, h: 180, label: 'background map', marks: heat(MAPS.background, 0, 0, 1) }, { caption: 'Class activation map', x: [0, 10], y: [0, 10], w: 240, h: 180, label: 'class activation map', marks: heat(v.L, 0, 0, 2) }],
  stats: v => [`peak ${fmt(Math.max(...v.L.flat()), 2)}`, `class score ${fmt(mean(v.L.flat()), 3)}`],
  takeaway: 'Grad-CAM picks these weights from gradients. A heat map shows where the model looked, not whether it was right.',
});
export const IouDrill = drill({
  intro: 'Overlap over combined area.',
  tex: R`\text{IoU}=\frac{|A\cap B|}{|A\cup B|}`,
  gen: r => { const s = pick(r, [10, 20, 40]); return { s, dx: int(r, 0, s / 2), dy: int(r, 0, s / 2) }; },
  q: v => `Two $${v.s}\\times${v.s}$ boxes, one shifted by $(${v.dx},${v.dy})$. Find the IoU.`,
  answer: v => iou([0, 0, v.s, v.s], [v.dx, v.dy, v.s + v.dx, v.s + v.dy]), dp: 3,
  steps: v => [`Intersection: $${(v.s - v.dx) * (v.s - v.dy)}$`, `Union: $${2 * v.s * v.s}-${(v.s - v.dx) * (v.s - v.dy)}$`],
});
const boxLine = (b, o = {}) => ({ tone: o.tone ?? 1, dash: o.dash, data: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]] });
export const IouLab = lab({
  intro: 'Drag the predicted box. A detection often counts only above an IoU threshold such as 0.5.',
  params: [{ k: 'x', label: 'prediction, left edge', min: 0, max: 8, step: 0.05, init: 3 }, { k: 'y', label: 'prediction, bottom edge', min: 0, max: 8, step: 0.05, init: 3.5 }, { k: 'thr', label: 'threshold', min: 0.3, max: 0.95, step: 0.05, init: 0.5 }],
  derive: ({ x, y }) => { const T = [2, 2, 6, 6], P = [x, y, x + 4, y + 4]; return { T, P, v: iou(T, P) }; },
  plots: v => [{ x: [0, 12], y: [0, 12], w: 360, h: 340, label: 'ground truth and prediction boxes', marks: [boxLine(v.T, { tone: 3 }), boxLine(v.P, { tone: 2 }), { as: 'rects', tone: 4, soft: true, data: [[Math.max(v.T[0], v.P[0]), Math.max(v.T[1], v.P[1]), Math.max(Math.max(v.T[0], v.P[0]), Math.min(v.T[2], v.P[2])), Math.max(Math.max(v.T[1], v.P[1]), Math.min(v.T[3], v.P[3]))]] }, { drag: { x: 'x', y: 'y', tone: 2 } }] }],
  stats: v => [`IoU ${fmt(v.v, 3)}`],
  status: v => (v.v >= v.thr ? ['ok', 'Counts as a true positive at this threshold.'] : ['alert', 'Counts as a false positive at this threshold.']),
  legend: [['--c3', 'ground truth'], ['--c2', 'prediction']],
  takeaway: 'IoU falls quickly as boxes separate, so high thresholds are demanding.',
});
const DET = (() => { const r = rng(5), out = []; [[2, 2], [6.5, 5.5]].forEach(([a, b]) => { for (let i = 0; i < 6; i++) { const dx = 0.6 * gauss(r), dy = 0.6 * gauss(r); out.push([+(a + dx).toFixed(2), +(b + dy).toFixed(2), +(a + dx + 3).toFixed(2), +(b + dy + 3).toFixed(2), +(0.5 + 0.49 * r()).toFixed(2)]); } }); return out; })();
export const NmsLab = lab({
  intro: 'A detector proposes many overlapping boxes. Keep the best, and remove whatever overlaps it too much.',
  params: [{ k: 't', label: 'overlap threshold', min: 0.05, max: 0.95, step: 0.05, init: 0.4 }],
  derive: ({ t }) => ({ keep: nms(DET, t) }),
  plots: v => [{ x: [0, 11], y: [0, 10], w: 400, h: 340, label: 'proposed and kept boxes', marks: [...DET.map((b, i) => (v.keep.includes(i) ? boxLine(b, { tone: 3 }) : boxLine(b, { tone: 'muted', dash: true })))] }],
  stats: v => [`kept ${v.keep.length} of ${DET.length}`, `scores ${v.keep.map(i => DET[i][4].toFixed(2)).join(', ')}`],
  takeaway: 'A low threshold can delete real neighbours, and a high one leaves duplicates.',
});
const featData = memo(kind => { const r = rng(kind === 'good' ? 3 : 4), rows = Array.from({ length: 60 }, (_, i) => { const y = i % 2; return kind === 'good' ? [+(1.4 * (2 * y - 1) + 0.7 * gauss(r)).toFixed(3), +(0.8 * gauss(r)).toFixed(3), y] : [+(1.6 * gauss(r)).toFixed(3), +(1.6 * gauss(r)).toFixed(3), y]; }); let w = [0, 0, 0]; const lr = 0.5; for (let t = 0; t < 300; t++) { const g = [0, 0, 0]; rows.forEach(([a, b, y]) => { const p = 1 / (1 + Math.exp(-(w[0] * a + w[1] * b + w[2]))), e = p - y; g[0] += e * a; g[1] += e * b; g[2] += e; }); w = w.map((v, i) => +(v - (lr * g[i]) / rows.length).toFixed(8)); } const acc = mean(rows.map(([a, b, y]) => +(+(w[0] * a + w[1] * b + w[2] > 0) === y))); return { rows, w, acc }; });
export const FrozenFeatures = lab({
  intro: 'A linear head on top of frozen features. Good features make the last step trivial.',
  params: [{ k: 'k', label: 'Features', options: [['good', 'pretrained model'], ['bad', 'random projection']], init: 'good' }],
  derive: ({ k }) => featData(k),
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 320, label: 'features and linear head', marks: [{ as: 'field', bands: 4, tones: [2, 1], fn: (a, b) => 2 / (1 + Math.exp(-(v.w[0] * a + v.w[1] * b + v.w[2]))) - 1 }, { as: 'points', r: 4, data: v.rows.map(p => [p[0], p[1], p[2] + 1]) }] }],
  stats: v => [`accuracy of the linear head ${pct(v.acc, 1)}`],
  takeaway: 'Transfer learning freezes a trained network and fits only the last layer, so few labels are enough.',
});
export const VisionCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['A convolution layer shares its weights across positions.', true, 'One filter is reused everywhere.'],
    ['Max pooling has learnable weights.', false, 'It has none.'],
    ['Convolution is exactly invariant to shifts.', false, 'It is equivariant: the output moves with the input.'],
    ['Stride greater than one reduces the output size.', true, 'The filter takes fewer positions.'],
    ['Grad-CAM proves a model reasons correctly.', false, 'It only shows where the evidence came from.'],
    ['An IoU of 0.5 means half of each box overlaps.', false, 'It is overlap over union, which is stricter.'],
    ['mAP averages precision over recall and classes.', true, 'AP is per class, and mAP averages them.'],
  ],
  takeaway: 'Local, shared filters are the core idea. Everything else builds on them.',
});
