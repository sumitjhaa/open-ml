'use client';
// Deep learning explorers built on chart forms suited to each quantity: heatmaps for matrices, filled contours for loss surfaces,
// ridgelines for distributions across layers or time, node-link diagrams for computation, and vector fields for scores.
import { heatBox, heatPlot, histDensity, lab, ridge } from '../components/labs';
import { gauss, mean, rng } from '../lib/math.mjs';
import { fmt, linspace, normalPdf, pct } from '../lib/models.mjs';
import { softmaxT } from '../lib/extended.mjs';

const r6 = x => +x.toFixed(6);
const clip = (x, a, b) => Math.max(a, Math.min(b, x));
const sigmoid = z => 1 / (1 + Math.exp(-z));
const std = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const f2 = v => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2).replace(/^0\./, '.').replace(/^-0\./, '-.'));

/* ---------- foundations ---------- */
export const VizGraph = lab({
  intro: 'A neuron as a computation graph. Each box shows its value on top and $\\partial L/\\partial\\cdot$ below. Stronger colour means more gradient reaches that box.',
  tex: String.raw`L=\big(\sigma(w_1x_1+w_2x_2+b)-y\big)^2`,
  params: [{ k: 'w1', label: '$w_1$', min: -3, max: 3, step: 0.1, init: 1 }, { k: 'w2', label: '$w_2$', min: -3, max: 3, step: 0.1, init: -1.5 }, { k: 'b', label: '$b$', min: -3, max: 3, step: 0.1, init: 0.5 }, { k: 'x1', label: '$x_1$', min: -2, max: 2, step: 0.1, init: 1 }, { k: 'x2', label: '$x_2$', min: -2, max: 2, step: 0.1, init: 0.5 }, { k: 'y', label: 'target $y$', min: 0, max: 1, step: 1, init: 1 }],
  derive: ({ w1, w2, b, x1, x2, y }) => { const m1 = w1 * x1, m2 = w2 * x2, z = m1 + m2 + b, a = sigmoid(z), L = (a - y) ** 2, dA = 2 * (a - y), dZ = dA * a * (1 - a); return { m1, m2, z, a, L, dA, dZ, g: { w1: dZ * x1, w2: dZ * x2, b: dZ, x1: dZ * w1, x2: dZ * w2 } }; },
  plots: v => { const lv = g => clip(Math.abs(g) / Math.max(0.05, Math.abs(v.dA)), 0.05, 1), n = (id, x, y, t, val, g) => ({ id, x, y, w: 2.1, h: 1, text: `${t} ${fmt(val, 2)}`, sub: `grad ${fmt(g, 3)}`, level: lv(g), title: `${t}: value ${fmt(val, 3)}, gradient ${fmt(g, 4)}` }); return [{ bare: true, x: [0, 12.4], y: [0, 7], w: 520, h: 340, label: 'computation graph', marks: [{ as: 'graph', nodes: [n('w1', 1.2, 6.3, 'w₁', v.w1, v.g.w1), n('x1', 1.2, 5.05, 'x₁', v.x1, v.g.x1), n('w2', 1.2, 3.55, 'w₂', v.w2, v.g.w2), n('x2', 1.2, 2.3, 'x₂', v.x2, v.g.x2), n('b', 1.2, 0.8, 'b', v.b, v.g.b), n('m1', 4, 5.4, 'w₁x₁', v.m1, v.dZ), n('m2', 4, 2.4, 'w₂x₂', v.m2, v.dZ), n('z', 6.5, 3.5, 'sum', v.z, v.dZ), n('a', 8.85, 3.5, 'σ', v.a, v.dA), { id: 'L', x: 11.2, y: 3.5, w: 2.1, h: 1, text: `L ${fmt(v.L, 3)}`, sub: 'grad 1', level: 0.75, tone: 4 }], edges: [['w1', 'm1'], ['x1', 'm1'], ['w2', 'm2'], ['x2', 'm2'], ['m1', 'z'], ['m2', 'z'], ['b', 'z'], ['z', 'a'], ['a', 'L']] }] }]; },
  stats: v => [`loss ${fmt(v.L, 4)}`, `saturation factor ${fmt(v.a * (1 - v.a), 3)}`],
  takeaway: 'Each gradient is the product of local slopes on the path back from the loss.',
});
export const VizActivationRidge = lab({
  intro: 'Real activations from a deep network of 120 units per layer. Each row is one layer, so collapse or explosion is visible straight away.',
  tex: String.raw`\text{Var}(z)=n\,\sigma_w^2\,\text{Var}(x)`,
  params: [{ k: 'act', label: 'Activation', init: 'tanh', options: [['tanh', 'Tanh'], ['relu', 'ReLU']] }, { k: 'g', label: 'weight scale', min: 0.3, max: 2.5, step: 0.05, init: 1 }, { k: 'seed', seed: 'New weights', init: 1 }],
  derive: ({ act, g, seed }) => { const r = rng(seed * 37), n = 120; let x = Array.from({ length: n }, () => gauss(r)); const L = []; for (let l = 1; l <= 8; l++) { const y = Array(n).fill(0); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += ((g * gauss(r)) / Math.sqrt(n)) * x[j]; y[i] = act === 'tanh' ? Math.tanh(s) : Math.max(0, s); } x = y.map(r6); L.push({ label: `layer ${l}`, d: histDensity(x, -3, 3, 48), sd: std(x), dead: x.filter(v => Math.abs(v) < 1e-3).length / n }); } return { L }; },
  plots: v => [{ x: [-3, 3], y: [0, 8.8], w: 560, h: 380, noY: true, xl: 'activation', label: 'activation distribution per layer', marks: ridge(v.L, linspace(-3 + 0.0625, 3 - 0.0625, 48)) }],
  stats: v => [`spread at layer 8: ${fmt(v.L[7].sd, 4)}`, `spread at layer 1: ${fmt(v.L[0].sd, 3)}`, `share near zero at layer 8: ${pct(v.L[7].dead, 0)}`],
  takeaway: 'A good initialisation keeps every row about the same width.',
});
export const VizMlpDiagram = lab({
  intro: 'Choose the width of each layer, and read how many weights each connection needs.',
  tex: String.raw`\text{weights}=(n_{\text{in}}+1)\,n_{\text{out}}`,
  params: [{ k: 'a', label: 'inputs', min: 1, max: 8, step: 1, init: 3 }, { k: 'b', label: 'first hidden', min: 1, max: 8, step: 1, init: 5 }, { k: 'c', label: 'next hidden', min: 1, max: 8, step: 1, init: 4 }, { k: 'd', label: 'outputs', min: 1, max: 8, step: 1, init: 2 }],
  derive: ({ a, b, c, d }) => { const W = [a, b, c, d]; return { W, P: W.slice(1).map((n, i) => (W[i] + 1) * n) }; },
  plots: v => { const nodes = [], edges = [], id = (l, i) => `${l}-${i}`; v.W.forEach((n, l) => { for (let i = 0; i < n; i++) nodes.push({ id: id(l, i), x: 1 + l * 3, y: 4.5 + (i - (n - 1) / 2) * 1, w: 0.62, h: 0.62, round: true, level: 0.5, tone: l === 0 ? 2 : l === 3 ? 4 : 1, title: `layer ${l + 1}, unit ${i + 1}` }); }); v.W.slice(1).forEach((n, l) => { for (let i = 0; i < v.W[l]; i++) for (let j = 0; j < n; j++) edges.push([id(l, i), id(l + 1, j), 0.15]); }); return [{ bare: true, x: [0, 11], y: [0, 9], w: 460, h: 340, label: 'network diagram', marks: [{ as: 'graph', nodes, edges }] }]; },
  table: v => ({ head: ['connection', 'weights'], rows: v.P.map((p, i) => [`layer ${i + 1} to ${i + 2}`, p]) }),
  stats: v => [`total ${v.P.reduce((s, x) => s + x, 0)} parameters`],
  takeaway: 'Parameters are dominated by the widest pair of adjacent layers.',
});

/* ---------- pytorch ---------- */
export const VizBroadcast = lab({
  intro: 'A column and a row are stretched to a common shape, and then added cell by cell.',
  tex: String.raw`(R,1)+(1,C)\to(R,C)`,
  params: [{ k: 'r', label: 'rows $R$', min: 1, max: 6, step: 1, init: 3 }, { k: 'c', label: 'columns $C$', min: 1, max: 6, step: 1, init: 4 }, { k: 'a', label: 'column step', min: -2, max: 2, step: 0.5, init: 1 }, { k: 'b', label: 'row step', min: -2, max: 2, step: 0.5, init: 0.5 }],
  derive: ({ r, c, a, b }) => { const A = Array.from({ length: r }, (_, i) => [(i + 1) * a]), B = [Array.from({ length: c }, (_, j) => (j + 1) * b)]; return { A, B, S: A.map(row => B[0].map(v => row[0] + v)) }; },
  plots: v => { const max = Math.max(1e-9, ...v.S.flat().map(Math.abs), ...v.A.flat().map(Math.abs), ...v.B.flat().map(Math.abs)), p = (M, caption, w) => heatPlot(M, { w, h: 26 + 34 * M.length + 12, caption, label: caption, heat: { div: true, max, show: f2 } }); return [p(v.A, `Shape (${v.r}, 1)`, 110), p(v.B, `Shape (1, ${v.c})`, 90 + 44 * v.c), p(v.S, `Sum, shape (${v.r}, ${v.c})`, 90 + 44 * v.c)]; },
  stats: v => [`${v.r * v.c} numbers in the result`, `no extra memory needed for the stretched inputs`],
  takeaway: 'Sizes are compared from the right, and only a size of 1 can stretch.',
});
export const VizTensorBlocks = lab({
  intro: 'A small convolutional network. Each block’s height and shade follow its spatial size and channels, and its weight count is computed from the settings.',
  tex: String.raw`\text{conv weights}=(f^2c_{\text{in}}+1)\,c_{\text{out}}`,
  params: [{ k: 's', label: 'image side', min: 16, max: 128, step: 16, init: 32 }, { k: 'c', label: 'base channels', min: 8, max: 64, step: 8, init: 16 }, { k: 'f', label: 'kernel $f$', min: 1, max: 7, step: 2, init: 3 }, { k: 'n', label: 'stages', min: 1, max: 4, step: 1, init: 3 }, { k: 'k', label: 'classes', min: 2, max: 100, step: 1, init: 10 }],
  derive: ({ s, c, f, n, k }) => { const L = [{ t: 'input', w: s, c: 3, p: 0 }]; let w = s, ch = 3; for (let i = 0; i < n; i++) { const out = c * 2 ** i; L.push({ t: `conv ${out}`, w, c: out, p: (f * f * ch + 1) * out }); ch = out; w = Math.max(1, Math.floor(w / 2)); L.push({ t: 'pool', w, c: ch, p: 0 }); } L.push({ t: 'global pool', w: 1, c: ch, p: 0 }); L.push({ t: `linear ${k}`, w: 1, c: k, p: (ch + 1) * k }); return { L, total: L.reduce((a, b) => a + b.p, 0) }; },
  plots: v => { const n = v.L.length, big = Math.max(...v.L.map(l => l.w)), maxC = Math.max(...v.L.map(l => l.c)), nodes = v.L.map((l, i) => ({ id: `n${i}`, x: i + 0.5, y: 4, w: 0.8, h: 1.4 + (5.6 * Math.sqrt(l.w)) / Math.sqrt(big), level: clip(0.15 + (0.65 * l.c) / maxC, 0, 1), tone: l.p ? 1 : 2, text: l.t.split(' ')[0].slice(0, 5), sub: `${l.w}×${l.c}`, title: `${l.t}: ${l.w} by ${l.w}, ${l.c} channels, ${l.p} weights` })); return [{ bare: true, x: [0, n], y: [0, 8], w: 520, h: 300, label: 'architecture diagram', marks: [{ as: 'graph', nodes, edges: nodes.slice(1).map((m, i) => [`n${i}`, m.id, 0.4]) }] }]; },
  table: v => ({ head: ['layer', 'side', 'channels', 'weights'], rows: v.L.filter(l => l.p).map(l => [l.t, l.w, l.c, l.p]) }),
  stats: v => [`total ${v.total.toLocaleString('en-US')} weights`],
  takeaway: 'Global pooling keeps the weight count independent of the image size.',
});

/* ---------- optimisation ---------- */
const SURF = {
  valley: { f: ([x, y]) => 0.5 * (x * x + 25 * y * y), g: ([x, y]) => [x, 25 * y], d: [-6, 6, -2, 2], s: [-5, 1.2] },
  saddle: { f: ([x, y]) => 0.5 * x * x - 0.5 * y * y + 0.06 * y ** 4, g: ([x, y]) => [x, -y + 0.24 * y ** 3], d: [-4, 4, -4, 4], s: [-3.5, 0.02] },
  bowls: { f: ([x, y]) => 2 - 1.6 * Math.exp(-((x - 1.5) ** 2 + (y - 1) ** 2)) - Math.exp(-((x + 1.5) ** 2 + (y + 1) ** 2) / 1.5), g: ([x, y]) => [3.2 * (x - 1.5) * Math.exp(-((x - 1.5) ** 2 + (y - 1) ** 2)) + (2 / 1.5) * (x + 1.5) * Math.exp(-((x + 1.5) ** 2 + (y + 1) ** 2) / 1.5), 3.2 * (y - 1) * Math.exp(-((x - 1.5) ** 2 + (y - 1) ** 2)) + (2 / 1.5) * (y + 1) * Math.exp(-((x + 1.5) ** 2 + (y + 1) ** 2) / 1.5)], d: [-4, 4, -4, 4], s: [0, -3] },
};
const trace = (g, s, kind, lr, b, n = 60) => { let p = s, v = [0, 0], q = [0, 0]; const P = [p]; for (let t = 1; t <= n; t++) { const d = g(p); if (kind === 'sgd') p = p.map((x, i) => x - lr * d[i]); if (kind === 'mom') { v = v.map((x, i) => b * x + d[i]); p = p.map((x, i) => x - lr * v[i]); } if (kind === 'adam') { v = v.map((x, i) => 0.9 * x + 0.1 * d[i]); q = q.map((x, i) => 0.999 * x + 0.001 * d[i] * d[i]); p = p.map((x, i) => x - (lr * 8 * v[i]) / (1 - 0.9 ** t) / (Math.sqrt(q[i] / (1 - 0.999 ** t)) + 1e-8)); } p = p.map(x => clip(r6(x), -50, 50)); P.push(p); } return P; };
export const VizLossSurface = lab({
  intro: 'A filled contour map of a loss, with the actual path of three optimisers. Drag the starting point.',
  tex: String.raw`v\leftarrow\beta v+\nabla L,\quad w\leftarrow w-\eta v`,
  params: [{ k: 's', label: 'Surface', init: 'valley', options: [['valley', 'Narrow valley'], ['saddle', 'Saddle'], ['bowls', 'Two basins']] }, { k: 'lr', label: 'learning rate $\\eta$', min: 0.002, max: 0.2, step: 0.002, init: 0.03 }, { k: 'b', label: 'momentum $\\beta$', min: 0, max: 0.99, step: 0.01, init: 0.9 }, { k: 'sx', label: 'start $x$', min: -6, max: 6, step: 0.05, init: -5 }, { k: 'sy', label: 'start $y$', min: -4, max: 4, step: 0.05, init: 1.2 }],
  derive: ({ s, lr, b, sx, sy }) => { const S = SURF[s], d = S.d, st = [clip(sx, d[0], d[1]), clip(sy, d[2], d[3])], P = { sgd: trace(S.g, st, 'sgd', lr, b), mom: trace(S.g, st, 'mom', lr, b), adam: trace(S.g, st, 'adam', lr, b) }; const G = linspace(d[0], d[1], 40).flatMap(x => linspace(d[2], d[3], 40).map(y => S.f([x, y]))), lo = Math.min(...G), hi = Math.max(...G); return { S, st, P, lo, hi }; },
  plots: v => { const [a, b, c, e] = v.S.d, inb = p => p.map(([x, y]) => [clip(x, a, b), clip(y, c, e)]), norm = (x, y) => Math.sqrt(clip((v.S.f([x, y]) - v.lo) / (v.hi - v.lo || 1), 0, 1)); return [{ x: [a, b], y: [c, e], w: 560, h: 380, xl: 'w₁', yl: 'w₂', label: 'loss surface with optimiser paths', marks: [{ as: 'field', bands: 8, tones: [1, 1], fn: (x, y) => 1 - norm(x, y) }, { as: 'contour', tone: 'muted', fn: norm, levels: [0.15, 0.3, 0.45, 0.6, 0.75, 0.9] }, { tone: 2, data: inb(v.P.sgd) }, { tone: 3, data: inb(v.P.mom) }, { tone: 4, data: inb(v.P.adam) }, { drag: { x: 'sx', y: 'sy', tone: 1 }, r: 7 }] }]; },
  stats: v => ['sgd', 'mom', 'adam'].map((k, i) => `${['SGD', 'Momentum', 'Adam'][i]}: loss ${fmt(v.S.f(v.P[k][60]), 4)}`),
  legend: [['--c2', 'SGD'], ['--c3', 'Momentum'], ['--c4', 'Adam']],
  takeaway: 'Conditioning, not distance, decides how fast each optimiser moves.',
});
export const VizHyperHeat = lab({
  intro: 'Each cell is a full optimisation run. Read the region that converges, and where it ends in divergence.',
  tex: String.raw`\text{cell}=\log_{10}L\big(w_{40};\eta,\beta\big)`,
  params: [{ k: 'k', label: 'curvature ratio', min: 2, max: 200, step: 2, init: 50 }, { k: 'n', label: 'steps', min: 10, max: 200, step: 5, init: 40 }],
  derive: ({ k, n }) => { const L = linspace(-3, 0, 13), B = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99], M = B.map(b => L.map(e => { const lr = 10 ** e; let a = 1, c = 1, va = 0, vc = 0; for (let t = 0; t < n; t++) { va = b * va + a; vc = b * vc + k * c; a -= lr * va; c -= lr * vc; if (Math.abs(a) + Math.abs(c) > 1e6) return 6; } return clip(Math.log10(0.5 * (a * a + k * c * c) + 1e-12), -12, 6); })); let best = [0, 0]; M.forEach((r, i) => r.forEach((v, j) => { if (v < M[best[0]][best[1]]) best = [i, j]; })); return { M, B, L, best }; },
  plots: v => [heatPlot(v.M.map(r => r.map(x => 6 - x)), { w: 580, h: 340, label: 'final loss by learning rate and momentum', heat: { min: Math.min(...v.M.flat().filter(x => x < 6).map(x => 6 - x)) * 0.8, max: Math.max(...v.M.flat().map(x => 6 - x)), rows: v.B.map(b => `β ${b}`), cols: v.L.map((e, i) => (i % 2 ? '' : `${e.toFixed(1)}`)), name: (i, j, x) => `momentum ${v.B[i]}, log lr ${v.L[j].toFixed(2)}: ${x === 0 ? 'diverged' : `log loss ${(6 - x).toFixed(2)}`}` } })],
  stats: v => [`best: β ${v.B[v.best[0]]}, η ${(10 ** v.L[v.best[1]]).toFixed(4)}`, `diverged cells: ${v.M.flat().filter(x => x >= 6).length}`],
  note: 'Column labels are the base-10 log of the learning rate. Stronger colour means a lower final loss, and blank cells diverged.',
  takeaway: 'Higher momentum needs a smaller learning rate, so tune the two together.',
});
export const VizGradFlow = lab({
  intro: 'Train a deep chain of units for real, and record the gradient at every layer on every epoch.',
  tex: String.raw`\frac{\partial L}{\partial w_k}=\frac{\partial L}{\partial h_K}\prod_{j>k}\frac{\partial h_j}{\partial h_{j-1}}\cdot\frac{\partial h_k}{\partial w_k}`,
  params: [{ k: 'act', label: 'Activation', init: 'tanh', options: [['tanh', 'Tanh'], ['relu', 'ReLU']] }, { k: 'res', label: 'Skip connections', init: 'no', options: [['no', 'None'], ['yes', 'Residual']] }, { k: 'd', label: 'layers', min: 3, max: 20, step: 1, init: 12 }, { k: 'lr', label: 'learning rate', min: 0.001, max: 0.2, step: 0.001, init: 0.02 }],
  derive: ({ act, res, d, lr }) => {
    const phi = z => (act === 'tanh' ? Math.tanh(z) : Math.max(0, z)), dphi = z => (act === 'tanh' ? 1 - Math.tanh(z) ** 2 : z > 0 ? 1 : 0), R = res === 'yes', r = rng(5), w = Array.from({ length: d }, () => 0.8 + 0.1 * gauss(r)), xs = linspace(-1, 1, 12), target = x => Math.sin(2 * x), G = []; let last = 0;
    for (let ep = 0; ep < 40; ep++) {
      const g = Array(d).fill(0); let L = 0;
      xs.forEach(x => { const h = [x], Z = []; for (let i = 0; i < d; i++) { const z = w[i] * h[i] + 0.5; Z.push(z); h.push(R ? h[i] + 0.2 * phi(z) : phi(z)); } let delta = 2 * (h[d] - target(x)); L += (h[d] - target(x)) ** 2; for (let i = d - 1; i >= 0; i--) { const slope = R ? 0.2 * dphi(Z[i]) : dphi(Z[i]); g[i] += delta * slope * h[i]; delta *= R ? 1 + slope * w[i] : slope * w[i]; } });
      last = L / 12; const gn = g.map(x => x / 12); G.push(gn.map(Math.abs)); for (let i = 0; i < d; i++) w[i] = r6(clip(w[i] - lr * gn[i], -5, 5));
    }
    const M = Array.from({ length: d }, (_, i) => G.map(row => row[i])).map(row => row.map(v => clip(Math.log10(v + 1e-12), -10, 1) + 10));
    return { M, last, first: mean(G[0]), early: G[0][0], late: G[0][d - 1], lo: Math.min(...M.flat()), hi: Math.max(...M.flat()) };
  },
  plots: v => [heatPlot(v.M, { w: 620, h: 40 + 18 * v.d + 10, label: 'gradient by layer and epoch', heat: { dense: true, min: v.lo, max: v.hi, rows: v.M.map((_, i) => (i % 2 ? '' : `L${i + 1}`)), name: (i, j, x) => `layer ${i + 1}, epoch ${j + 1}: log10 |grad| ${(x - 10).toFixed(2)}` } })],
  stats: v => [`first-layer gradient ${v.early.toExponential(1)}`, `last-layer gradient ${v.late.toExponential(1)}`, `final loss ${fmt(v.last, 4)}`],
  note: 'Rows are layers and columns are epochs, on a log scale. Colour runs from the smallest gradient in this run to the largest, so faint rows near the top mean the early layers are starving.',
  takeaway: 'A heatmap of gradient norms is the quickest diagnosis of vanishing gradients.',
});

/* ---------- architectures ---------- */
export const VizReceptiveHeat = lab({
  intro: 'Which input pixels influence the centre output? This convolves a unit impulse through the real kernel stack.',
  tex: String.raw`r=1+\sum_\ell d_\ell(f_\ell-1)`,
  params: [{ k: 'n', label: 'layers', min: 1, max: 8, step: 1, init: 4 }, { k: 'f', label: 'kernel size', min: 3, max: 7, step: 2, init: 3 }, { k: 'd', label: 'dilation', min: 1, max: 3, step: 1, init: 1 }],
  derive: ({ n, f, d }) => { const S = 41, c = 20, h = (f - 1) / 2; let G = Array.from({ length: S }, (_, i) => Array.from({ length: S }, (_, j) => (i === c && j === c ? 1 : 0))); for (let l = 0; l < n; l++) { const H = G.map(r => r.map(() => 0)); for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) { if (!G[i][j]) continue; for (let a = -h; a <= h; a++) for (let b = -h; b <= h; b++) { const x = i + a * d, y = j + b * d; if (x >= 0 && x < S && y >= 0 && y < S) H[x][y] += G[i][j] / (f * f); } } G = H; } const row = G[c], t = row.reduce((a, b) => a + b, 0), sd = Math.sqrt(row.reduce((s, v, i) => s + (v / t) * (i - c) ** 2, 0)); return { G: G.map(r => r.map(v => r6(Math.sqrt(v)))), sd, R: 1 + n * d * (f - 1) }; },
  plots: v => [heatPlot(v.G, { w: 400, h: 390, label: 'influence of each input pixel', heat: { dense: true, name: (i, j, x) => `pixel ${i - 20}, ${j - 20}: influence ${(x * x).toExponential(1)}` } })],
  stats: v => [`theoretical field ${v.R} pixels`, `effective spread ${fmt(2 * v.sd + 1, 1)} pixels`],
  takeaway: 'Influence is concentrated near the centre, so the effective field is smaller than the theoretical one.',
});
const IMG = Array.from({ length: 18 }, (_, i) => Array.from({ length: 18 }, (_, j) => ((i > 4 && i < 13 && j > 3 && j < 10) || Math.abs(i - j - 3) < 1 ? 1 : 0)));
const KERN = { sx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], sy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], blur: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map(r => r.map(v => v / 16)), sharp: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] };
export const VizFeatureMaps = lab({
  intro: 'Slide a real filter across an image. Then shift the image, and see that the output shifts with it.',
  tex: String.raw`y[i,j]=\sum_{a,b}k[a,b]\,x[i-a,j-b]`,
  params: [{ k: 'f', label: 'Filter', init: 'sx', options: [['sx', 'Vertical edges'], ['sy', 'Horizontal edges'], ['blur', 'Blur'], ['sharp', 'Sharpen']] }, { k: 's', label: 'image shift', min: 0, max: 8, step: 1, init: 0 }],
  derive: ({ f, s }) => { const K = KERN[f], X = IMG.map(r => r.map((_, j) => r[(j - s + 18) % 18])), Y = X.map((r, i) => r.map((_, j) => { let t = 0; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) t += K[a + 1][b + 1] * X[(i - a + 18) % 18][(j - b + 18) % 18]; return t; })); return { K, X, Y }; },
  plots: v => [heatPlot(v.X, { w: 260, h: 250, caption: 'Input', label: 'input image', heat: {} }), heatPlot(v.K, { w: 190, h: 170, caption: 'Kernel', label: 'kernel', heat: { div: true, show: f2 } }), heatPlot(v.Y, { w: 260, h: 250, caption: 'Output', label: 'feature map', heat: { div: true } })],
  stats: v => [`strongest response ${fmt(Math.max(...v.Y.flat().map(Math.abs)), 2)}`, `weights in this filter: 9`],
  takeaway: 'One small kernel, reused everywhere, makes the operation equivariant to translation.',
});

/* ---------- attention ---------- */
const TOK = ['the', 'cat', 'sat', 'on', 'its', 'mat'], VEC = [[0.2, 0.9], [1, 0.2], [0.3, -1], [-0.5, 0.8], [0.9, 0.5], [0.8, -0.1]];
const att = (Q, K, t, mask) => Q.map((q, i) => { const s = K.map((k, j) => (mask && j > i ? -Infinity : (q[0] * k[0] + q[1] * k[1]) / (Math.SQRT2 * t))), m = Math.max(...s.filter(Number.isFinite)), e = s.map(v => (Number.isFinite(v) ? Math.exp(v - m) : 0)), z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); });
const rot = (v, a) => v.map(([x, y]) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)]);
export const VizAttentionHeat = lab({
  intro: 'Every row is a query, and its cells are the softmax weights over all tokens. Each row adds up to 1.',
  tex: String.raw`A=\text{softmax}\Big(\frac{QK^\top}{\sqrt d\,\tau}\Big)`,
  params: [{ k: 'a', label: 'query rotation', min: -180, max: 180, step: 5, init: 0 }, { k: 't', label: 'temperature $\\tau$', min: 0.05, max: 2, step: 0.01, init: 0.3 }],
  derive: ({ a, t }) => { const A = att(rot(VEC, (a * Math.PI) / 180), VEC, t, false); return { A, H: mean(A.map(r => -r.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0))) }; },
  plots: v => [heatPlot(v.A, { w: 460, h: 410, label: 'attention matrix', heat: { rows: TOK, cols: TOK, show: f2, name: (i, j, x) => `${TOK[i]} attends to ${TOK[j]}: ${x.toFixed(3)}` } })],
  stats: v => [`mean entropy ${fmt(v.H, 2)} bits`, `strongest weight ${fmt(Math.max(...v.A.flat()), 3)}`],
  takeaway: 'Low temperature approaches a hard choice, and high temperature an even average.',
});
export const VizPosEncoding = lab({
  intro: 'Each row is a position, and each column one dimension of its code. Frequency falls as you move right.',
  tex: String.raw`\text{PE}(t,2i)=\sin\big(t/10000^{2i/d}\big)`,
  params: [{ k: 'T', label: 'positions', min: 8, max: 64, step: 4, init: 32 }, { k: 'd', label: 'dimensions', min: 8, max: 64, step: 4, init: 32 }, { k: 'p', label: 'compare position', min: 1, max: 63, step: 1, init: 5 }],
  derive: ({ T, d, p }) => { const pe = t => Array.from({ length: d }, (_, i) => { const f = t / 10000 ** ((2 * Math.floor(i / 2)) / d); return r6(i % 2 ? Math.cos(f) : Math.sin(f)); }), M = Array.from({ length: T }, (_, t) => pe(t)), a = pe(0), b = pe(Math.min(p, 63)); return { M, sim: a.reduce((s, x, i) => s + x * b[i], 0) / d }; },
  plots: v => [heatPlot(v.M, { w: 560, h: 340, label: 'positional encoding', heat: { dense: true, div: true, max: 1, name: (i, j, x) => `position ${i}, dimension ${j}: ${x.toFixed(3)}` } })],
  stats: v => [`similarity of position 0 and ${Math.min(v.p, 63)}: ${fmt(v.sim, 3)}`],
  takeaway: 'Nearby positions have similar codes, and every position stays distinct.',
});
export const VizCausalMask = lab({
  intro: 'A decoder must not read the future. Turn the mask on, and all weight above the diagonal vanishes.',
  tex: String.raw`s_{ij}\leftarrow-\infty\quad(j>i)`,
  params: [{ k: 'm', label: 'Mask', init: 'on', options: [['on', 'Causal'], ['off', 'None']] }, { k: 't', label: 'temperature $\\tau$', min: 0.05, max: 2, step: 0.01, init: 0.5 }],
  derive: ({ m, t }) => { const A = att(VEC, VEC, t, m === 'on'); return { A, leak: A.reduce((s, r, i) => s + r.slice(i + 1).reduce((a, b) => a + b, 0), 0) / 6 }; },
  plots: v => [heatPlot(v.A, { w: 460, h: 410, label: 'masked attention', heat: { rows: TOK, cols: TOK, show: f2 } })],
  stats: v => [`weight on future tokens ${pct(v.leak, 1)}`, `first token can only read itself: ${fmt(v.A[0][0], 3)}`],
  takeaway: 'Rows are renormalised after masking, so the visible tokens share all the weight.',
});
export const VizHeads = lab({
  intro: 'Each head has its own projections, so it learns a different pattern. Their dimensions divide the width, and the cost stays the same.',
  tex: String.raw`d_{\text{head}}=d/h`,
  params: [{ k: 'h', label: 'heads $h$', min: 1, max: 6, step: 1, init: 4 }, { k: 'd', label: 'width $d$', min: 24, max: 768, step: 24, init: 96 }, { k: 'seed', seed: 'New projections', init: 1 }],
  derive: ({ h, seed }) => { const r = rng(seed * 3 + 7); return { H: Array.from({ length: h }, () => { const a = r() * 6.283, b = r() * 6.283; return att(rot(VEC, a), rot(VEC, b), 0.35, false); }) }; },
  plots: v => v.H.map((A, i) => heatPlot(A, { w: 220, h: 200, caption: `Head ${i + 1}`, label: `attention of head ${i + 1}`, heat: { tone: (i % 4) + 1 } })),
  stats: v => [`each head has width ${fmt(v.d / v.h, v.d % v.h ? 1 : 0)}`, `projection weights ${(3 * v.d * v.d).toLocaleString('en-US')}`],
  status: v => (v.d % v.h ? ['alert', 'The width must divide evenly by the heads.'] : ['ok', 'The width divides evenly.']),
  takeaway: 'Splitting width across heads gives several views for about the same cost.',
});

/* ---------- language models ---------- */
const DOC = ['refunds', 'returns', 'shipping', 'delivery', 'pricing', 'security'], TV = [[1, 0.2, 0, 0.1], [0.9, 0.4, 0, 0], [0.1, 1, 0.1, 0], [0, 0.9, 0.3, 0.1], [0.3, 0, 1, 0.1], [0, 0.1, 0.1, 1]];
const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b) || 1);
export const VizSimHeat = lab({
  intro: 'Cosine similarity between document topics, with your query on the bottom row. Mix topics into the query, and see what would be retrieved.',
  tex: String.raw`\cos\theta=\frac{q\cdot d}{\|q\|\|d\|}`,
  params: [{ k: 'a', label: 'about refunds', min: 0, max: 1, step: 0.05, init: 0.7 }, { k: 'b', label: 'about delivery', min: 0, max: 1, step: 0.05, init: 0.3 }, { k: 'c', label: 'about price', min: 0, max: 1, step: 0.05, init: 0 }, { k: 'd', label: 'about security', min: 0, max: 1, step: 0.05, init: 0 }, { k: 'k', label: 'passages kept', min: 1, max: 6, step: 1, init: 2 }],
  derive: ({ a, b, c, d, k }) => { const q = [a, b, c, d], M = TV.map(u => TV.map(w => cosine(u, w))), row = TV.map(w => cosine(q, w)); return { M: [...M, row], row, top: row.map((s, i) => [s, DOC[i]]).sort((x, y) => y[0] - x[0]).slice(0, k) }; },
  plots: v => [heatPlot(v.M, { w: 500, h: 400, label: 'similarity matrix', heat: { max: 1, rows: [...DOC, 'query'], cols: DOC, show: f2 } })],
  table: v => ({ head: ['retrieved', 'similarity'], rows: v.top.map(([s, d]) => [d, fmt(s, 3)]) }),
  takeaway: 'A query mixing topics retrieves from each, so its ranking depends on the mixture.',
});
export const VizContextBudget = lab({
  intro: 'Everything shares one context window. Work out how many retrieved passages fit.',
  tex: String.raw`k_{\max}=\Big\lfloor\frac{C-s-q-a}{c}\Big\rfloor`,
  params: [{ k: 'C', label: 'context window', min: 1000, max: 32000, step: 500, init: 8000 }, { k: 's', label: 'system prompt', min: 0, max: 3000, step: 50, init: 800 }, { k: 'q', label: 'question and history', min: 0, max: 4000, step: 50, init: 900 }, { k: 'a', label: 'reserved for the answer', min: 100, max: 6000, step: 100, init: 1500 }, { k: 'c', label: 'passage size', min: 100, max: 2000, step: 50, init: 500 }, { k: 'k', label: 'passages sent', min: 0, max: 40, step: 1, init: 6 }],
  derive: ({ C, s, q, a, c, k }) => { const used = s + q + a + c * k; return { used, kmax: Math.max(0, Math.floor((C - s - q - a) / c)), parts: [['system', s, 1], ['history', q, 2], ['passages', c * k, 3], ['answer', a, 4]] }; },
  plots: v => { const nodes = []; let x = 0; const total = Math.max(v.C, v.used); v.parts.forEach(([t, n, tone]) => { const w = (n / total) * 100; if (n > 0) nodes.push({ id: t, x: x + w / 2, y: 1.5, w: Math.max(0.5, w - 0.4), h: 1.4, text: w > 8 ? t : '', tone, level: 0.7, title: `${t}: ${n} tokens` }); x += w; }); return [{ bare: true, x: [0, 100], y: [0, 3], w: 480, h: 130, label: 'context window use', marks: [{ as: 'graph', nodes }, { as: 'rule', tone: 'fg', dash: true, x: (v.C / total) * 100 }] }]; },
  stats: v => [`${v.used.toLocaleString('en-US')} of ${v.C.toLocaleString('en-US')} tokens`, `at most ${v.kmax} passages fit`],
  status: v => (v.used > v.C ? ['alert', 'Over the limit: the request would be cut off or rejected.'] : ['ok', 'Fits within the window.']),
  legend: [['--c1', 'system'], ['--c2', 'history'], ['--c3', 'passages'], ['--c4', 'answer']],
  takeaway: 'Retrieval must be planned against what is left after the prompt and the answer.',
});

/* ---------- vision ---------- */
const GTB = [[1, 1, 4, 5], [5, 2, 9, 6], [2, 6, 6, 9]], BX = (b, d) => [b[0] + d, b[1] + d * 0.5, b[2] + d, b[3] + d * 0.5], iouOf = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
const line = (b, tone, dash) => ({ tone, dash, data: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]] });
export const VizIouMatrix = lab({
  intro: 'Three predictions against three true boxes. The matrix of overlaps decides which prediction matches which object.',
  tex: String.raw`\text{IoU}=\frac{|A\cap B|}{|A\cup B|}`,
  params: [{ k: 'd', label: 'prediction offset', min: 0, max: 3, step: 0.05, init: 0.5 }, { k: 't', label: 'match threshold', min: 0.1, max: 0.9, step: 0.05, init: 0.5 }],
  derive: ({ d, t }) => { const P = GTB.map((b, i) => BX(b, d * (1 + i * 0.4))), M = P.map(p => GTB.map(g => iouOf(p, g))), used = new Set(), match = []; [...M.flatMap((r, i) => r.map((v, j) => [v, i, j]))].sort((a, b) => b[0] - a[0]).forEach(([v, i, j]) => { if (v >= t && !match.some(m => m[0] === i) && !used.has(j)) { match.push([i, j]); used.add(j); } }); return { P, M, match }; },
  plots: v => [{ caption: 'Boxes', x: [0, 13], y: [0, 11], w: 320, h: 280, label: 'predicted and true boxes', marks: [...GTB.map(b => line(b, 2)), ...v.P.map(b => line(b, 4, true))] }, heatPlot(v.M, { w: 270, h: 250, caption: 'IoU matrix', label: 'overlap matrix', heat: { max: 1, rows: ['p1', 'p2', 'p3'], cols: ['g1', 'g2', 'g3'], show: f2 } })],
  stats: v => [`${v.match.length} true positives`, `${3 - v.match.length} false positives`, `${3 - v.match.length} missed objects`],
  takeaway: 'A slight misplacement can turn a correct prediction into both a false positive and a miss.',
});
const CLS = ['cat', 'dog', 'car', 'bus', 'ship'], CTR = [[0, 2], [0.8, 2.2], [3, 0], [3.4, 0.6], [-2.5, -2]];
export const VizConfusion = lab({
  intro: 'A real nearest-centre classifier on five classes. Similar classes confuse first, so errors are not spread evenly.',
  tex: String.raw`C_{ij}=P(\hat y=j\mid y=i)`,
  params: [{ k: 's', label: 'class overlap', min: 0.1, max: 2, step: 0.05, init: 0.6 }, { k: 'seed', seed: 'New test set', init: 1 }],
  derive: ({ s, seed }) => { const r = rng(seed * 5 + 1), C = CTR.map(() => Array(5).fill(0)); CTR.forEach((c, i) => { for (let n = 0; n < 200; n++) { const x = c[0] + s * gauss(r), y = c[1] + s * gauss(r), d = CTR.map(q => (q[0] - x) ** 2 + (q[1] - y) ** 2); C[i][d.indexOf(Math.min(...d))]++; } }); const R = C.map(row => row.map(v => v / 200)), f1 = C.map((row, i) => { const tp = row[i], fn = 200 - tp, fp = C.reduce((a, b, k) => a + (k === i ? 0 : b[i]), 0); return (2 * tp) / (2 * tp + fp + fn || 1); }); return { R, f1, acc: mean(R.map((r2, i) => r2[i])) }; },
  plots: v => [heatPlot(v.R, { w: 460, h: 410, label: 'confusion matrix', heat: { max: 1, rows: CLS, cols: CLS, show: f2, name: (i, j, x) => `true ${CLS[i]}, predicted ${CLS[j]}: ${(x * 100).toFixed(1)}%` } })],
  stats: v => [`accuracy ${pct(v.acc, 1)}`, `macro F1 ${fmt(mean(v.f1), 3)}`],
  table: v => ({ head: ['class', 'F1'], rows: CLS.map((c, i) => [c, fmt(v.f1[i], 3)]) }),
  takeaway: 'Off-diagonal cells show which mistakes to fix, which a single score hides.',
});

/* ---------- robust ---------- */
const REL = (() => { const r = rng(17); return Array.from({ length: 4000 }, () => { const z = gauss(r) * 2.2; return [z * 3, r() < sigmoid(z) ? 1 : 0]; }); })();
export const VizReliability = lab({
  intro: 'An overconfident model’s predictions, binned by confidence. Bars reaching the diagonal are calibrated. Find the temperature that fixes them.',
  tex: String.raw`\text{ECE}=\sum_b\frac{n_b}n\big|\text{acc}(b)-\text{conf}(b)\big|`,
  params: [{ k: 'T', label: 'temperature $T$', min: 0.5, max: 8, step: 0.05, init: 1 }],
  derive: ({ T }) => { const B = Array.from({ length: 10 }, () => ({ n: 0, c: 0, a: 0 })); REL.forEach(([z, y]) => { const p = sigmoid(z / T), i = Math.min(9, Math.floor(p * 10)); B[i].n++; B[i].c += p; B[i].a += y; }); const R = B.map((b, i) => ({ i, n: b.n, conf: b.n ? b.c / b.n : (i + 0.5) / 10, acc: b.n ? b.a / b.n : 0 })); return { R, ece: R.reduce((s, b) => s + (b.n / REL.length) * Math.abs(b.acc - b.conf), 0) }; },
  plots: v => [{ x: [0, 1], y: [0, 1], w: 340, h: 300, xl: 'predicted probability', yl: 'observed frequency', label: 'reliability diagram', marks: [{ as: 'rects', tone: 1, data: v.R.filter(b => b.n).map(b => [b.i / 10 + 0.01, 0, (b.i + 1) / 10 - 0.01, b.acc]) }, { as: 'rects', tone: 4, soft: true, data: v.R.filter(b => b.n).map(b => [b.i / 10 + 0.01, Math.min(b.acc, b.conf), (b.i + 1) / 10 - 0.01, Math.max(b.acc, b.conf)]) }, { tone: 'muted', dash: true, data: [[0, 0], [1, 1]] }] }, { caption: 'Predictions per bin', x: [0, 1], y: [0, Math.max(...v.R.map(b => b.n)) * 1.05], w: 340, h: 300, noY: true, xl: 'predicted probability', label: 'prediction counts', marks: [{ as: 'rects', tone: 2, data: v.R.map(b => [b.i / 10 + 0.01, 0, (b.i + 1) / 10 - 0.01, b.n]) }] }],
  stats: v => [`expected calibration error ${fmt(v.ece, 4)}`],
  legend: [['--c1', 'observed frequency'], ['--c4', 'gap to confidence']],
  takeaway: 'This model’s logits are three times too large, so the error is lowest near $T=3$.',
});
const WIMG = (() => { const r = rng(23); return Array.from({ length: 14 }, (_, i) => Array.from({ length: 14 }, (_, j) => r6(0.6 * Math.sin((i * 3.1) / 14) * Math.cos((j * 3.1) / 7) + 0.25 * gauss(r)))); })();
const XIMG = WIMG.map((row, i) => row.map((_, j) => ((i - 7) ** 2 + (j - 7) ** 2 < 22 ? 0.75 : 0.2)));
export const VizAdvImage = lab({
  intro: 'A linear classifier on a small image. The attack shifts every pixel by at most $\\epsilon$, in the worst direction for that classifier.',
  tex: String.raw`x'=x-\epsilon\,\text{sign}(w)`,
  params: [{ k: 'e', label: 'attack size $\\epsilon$', min: 0, max: 0.3, step: 0.005, init: 0.05 }],
  derive: ({ e }) => { const w = WIMG.flat(), x = XIMG.flat(), base = -w.reduce((s, v, i) => s + v * x[i], 0) + 1.5, sc = u => base + w.reduce((s, v, i) => s + v * u[i], 0), adv = x.map((v, i) => v - e * Math.sign(w[i])); return { s0: sc(x), s1: sc(adv), P: WIMG.map(r => r.map(v => -e * Math.sign(v))), A: XIMG.map((r, i) => r.map((v, j) => v - e * Math.sign(WIMG[i][j]))) }; },
  plots: v => [heatPlot(XIMG, { w: 220, h: 215, caption: 'Input', label: 'input', heat: { max: 1 } }), heatPlot(WIMG, { w: 220, h: 215, caption: 'Weights', label: 'weights', heat: { div: true } }), heatPlot(v.P, { w: 220, h: 215, caption: 'Perturbation', label: 'perturbation', heat: { div: true, max: 0.3 } }), heatPlot(v.A, { w: 220, h: 215, caption: 'Attacked input', label: 'attacked input', heat: { max: 1 } })],
  stats: v => [`score ${fmt(v.s0, 3)} to ${fmt(v.s1, 3)}`, `P(class) ${pct(sigmoid(v.s0), 1)} to ${pct(sigmoid(v.s1), 1)}`],
  status: v => (v.s1 <= 0 ? ['alert', 'The prediction flipped.'] : ['ok', 'The prediction held.']),
  takeaway: 'The attacked image looks unchanged, because the change is spread thinly over every pixel.',
});
const WEIGHTS = (() => { const r = rng(44); return Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => r6((((i + j) % 5 === 0 ? 1.1 : 0) + 0.18 * gauss(r)) * (r() < 0.5 ? 1 : -1)))); })();
export const VizPruneHeat = lab({
  intro: 'A real weight matrix. Raise the threshold, and read how many weights survive and what share of the signal is kept.',
  tex: String.raw`\text{kept energy}=\frac{\sum_{\text{kept}}w^2}{\sum w^2}`,
  params: [{ k: 't', label: 'pruning threshold', min: 0, max: 1.2, step: 0.01, init: 0.3 }],
  derive: ({ t }) => { const P = WEIGHTS.map(r => r.map(v => (Math.abs(v) < t ? 0 : v))), all = WEIGHTS.flat(), kept = P.flat(); return { P, share: kept.filter(v => v).length / all.length, energy: kept.reduce((s, v) => s + v * v, 0) / all.reduce((s, v) => s + v * v, 0) }; },
  plots: v => [heatPlot(WEIGHTS, { w: 300, h: 290, caption: 'Original', label: 'original weights', heat: { div: true, max: 1.5 } }), heatPlot(v.P, { w: 300, h: 290, caption: 'After pruning', label: 'pruned weights', heat: { div: true, max: 1.5 } })],
  stats: v => [`weights kept ${pct(v.share, 0)}`, `energy kept ${pct(v.energy, 1)}`],
  takeaway: 'Most weights can be removed while losing little of the total energy.',
});

/* ---------- generative ---------- */
export const VizDiffusionRidge = lab({
  intro: 'How a two-peaked distribution dissolves under the forward process. Each row is a later time step, and the densities are exact.',
  tex: String.raw`q(x_t)=\sum_k\pi_k\,\mathcal N\big(\sqrt{\bar\alpha_t}\mu_k,\ \bar\alpha_t\sigma^2+1-\bar\alpha_t\big)`,
  params: [{ k: 'sch', label: 'Schedule', init: 'lin', options: [['lin', 'Linear'], ['cos', 'Cosine']] }, { k: 'T', label: 'total steps', min: 100, max: 1000, step: 50, init: 1000 }, { k: 'b', label: 'largest beta', min: 0.005, max: 0.04, step: 0.001, init: 0.02 }],
  derive: ({ sch, T, b }) => { const A = []; let a = 1; for (let t = 0; t < T; t++) { if (sch === 'lin') a *= 1 - (1e-4 + ((b - 1e-4) * t) / (T - 1)); else { const f = u => Math.cos(((u / T + 0.008) / 1.008) * 1.5708) ** 2; a = f(t + 1) / f(0); } A.push(a); } const X = linspace(-5, 5, 161), R = [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.999].map(p => { const t = Math.floor(p * (T - 1)), ab = A[t], s = Math.sqrt(ab * 0.09 + 1 - ab), m = Math.sqrt(ab) * 2; return { label: `t ${t + 1}`, d: X.map(x => 0.5 * normalPdf(x, -m, s) + 0.5 * normalPdf(x, m, s)), ab }; }); return { X, R, last: A[T - 1], mid: A[Math.floor(T / 2)] }; },
  plots: v => [{ x: [-5, 5], y: [0, 8.9], w: 560, h: 390, noY: true, xl: 'x', label: 'distribution across time', marks: ridge(v.R, v.X) }],
  stats: v => [`signal left at the middle ${fmt(Math.sqrt(v.mid), 3)}`, `signal left at the end ${fmt(Math.sqrt(v.last), 4)}`],
  status: v => (Math.sqrt(v.last) > 0.1 ? ['alert', 'Noise is not complete, so sampling from a pure Gaussian is wrong.'] : ['ok', 'The final step is close to pure noise.']),
  takeaway: 'A cosine schedule destroys information more evenly than a linear one.',
});
const PEAKS = [[-2, -1], [2, 1], [-1.5, 2.2]];
const score2 = (x, y, s) => { const v = 0.1 + s * s; let t = 0, a = 0, b = 0; PEAKS.forEach(([m, n]) => { const p = Math.exp(-((x - m) ** 2 + (y - n) ** 2) / (2 * v)); t += p; a += (p * (m - x)) / v; b += (p * (n - y)) / v; }); return [a / (t || 1e-300), b / (t || 1e-300), t]; };
export const VizScoreField = lab({
  intro: 'The score as a vector field over the density. Drag the point, and follow where the score carries it.',
  tex: String.raw`x\leftarrow x+\eta\,\nabla\log p_\sigma(x)`,
  params: [{ k: 's', label: 'noise $\\sigma$', min: 0.1, max: 2.5, step: 0.05, init: 0.6 }, { k: 'x', label: 'start $x$', min: -4, max: 4, step: 0.05, init: 0.5 }, { k: 'y', label: 'start $y$', min: -3, max: 3, step: 0.05, init: -2 }],
  derive: ({ s, x, y }) => { const v = 0.1 + s * s, P = [[x, y]]; let a = x, b = y; for (let i = 0; i < 60; i++) { const g = score2(a, b, s); a = r6(a + 0.3 * v * g[0]); b = r6(b + 0.3 * v * g[1]); P.push([a, b]); } const A = []; linspace(-4, 4, 17).forEach(p => linspace(-3, 3, 13).forEach(q => { const g = score2(p, q, s), m = Math.hypot(g[0], g[1]) || 1, l = 0.34 * Math.min(1, 0.4 + m / 5); A.push([p, q, p + (g[0] / m) * l, q + (g[1] / m) * l]); })); const N = Math.max(...linspace(-4, 4, 40).flatMap(p => linspace(-3, 3, 30).map(q => score2(p, q, s)[2]))); return { P, A, N }; },
  plots: v => [{ x: [-4, 4], y: [-3, 3], w: 580, h: 420, xl: 'x₁', yl: 'x₂', label: 'score vector field', marks: [{ as: 'field', bands: 7, tones: [1, 1], fn: (a, b) => clip(score2(a, b, v.s)[2] / v.N, 0, 1) * 0.9 }, { as: 'arrow', tone: 'muted', data: v.A }, { tone: 4, data: v.P }, { drag: { x: 'x', y: 'y', tone: 4 }, r: 7 }] }],
  stats: v => [`ends at (${fmt(v.P[60][0], 2)}, ${fmt(v.P[60][1], 2)})`],
  takeaway: 'At high noise the field is smooth and global, and at low noise it points to the nearest mode.',
});

/* ---------- graphs and control ---------- */
const GR = { ring: Array.from({ length: 8 }, (_, i) => [(i + 7) % 8, (i + 1) % 8]), star: Array.from({ length: 8 }, (_, i) => (i ? [0] : [1, 2, 3, 4, 5, 6, 7])), path: Array.from({ length: 8 }, (_, i) => [i - 1, i + 1].filter(j => j >= 0 && j < 8)), community: [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2, 4], [3, 5, 6, 7], [4, 6, 7], [4, 5, 7], [4, 5, 6]] };
export const VizGraphSpread = lab({
  intro: 'A real graph, with every node averaging its neighbours each round. The node-link view shows the spread, and the matrix shows who has reached whom.',
  tex: String.raw`H^{(k+1)}=\tilde D^{-1}\tilde AH^{(k)}`,
  params: [{ k: 'g', label: 'Graph', init: 'community', options: [['community', 'Two communities'], ['ring', 'Ring'], ['star', 'Star'], ['path', 'Path']] }, { k: 'k', label: 'rounds', min: 0, max: 30, step: 1, init: 2, play: true }],
  derive: ({ g, k }) => { const A = GR[g], step = M => M.map((row, i) => row.map((_, j) => (M[i][j] + A[i].reduce((s, u) => s + M[u][j], 0)) / (1 + A[i].length))); let M = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0))); for (let t = 0; t < k; t++) M = step(M); return { A, M }; },
  plots: v => { const n = v.A.length, pos = i => [5 + 3.8 * Math.cos((2 * Math.PI * i) / n), 5 + 3.8 * Math.sin((2 * Math.PI * i) / n)], nodes = v.M.map((r, i) => ({ id: `n${i}`, x: pos(i)[0], y: pos(i)[1], w: 1.4, h: 1.4, round: true, level: clip(r[0] / Math.max(...v.M.map(q => q[0])), 0.06, 1), text: `${i}`, title: `node ${i}: receives ${r[0].toFixed(3)} from node 0` })), edges = v.A.flatMap((l, i) => l.filter(j => j > i).map(j => [`n${i}`, `n${j}`, 0.35])); return [{ caption: 'Signal from node 0', bare: true, x: [0, 10], y: [0, 10], w: 330, h: 330, label: 'graph', marks: [{ as: 'graph', nodes, edges }] }, heatPlot(v.M, { w: 330, h: 320, caption: 'Influence matrix', label: 'node influence', heat: { rows: v.M.map((_, i) => `${i}`), cols: v.M.map((_, i) => `${i}`) } })]; },
  stats: v => [`difference between rows ${fmt(mean(v.M.map(r => std(r))), 4)}`],
  takeaway: 'Rows of the influence matrix become identical with depth, which is over-smoothing seen directly.',
});
const WORLD = ['......', '.##...', '.#..#.', '...#..', '.#....', '...#.G'];
export const VizGridValues = lab({
  intro: 'Value iteration on a real grid. Colour is the value of each cell, and arrows show the best move. Step through the sweeps.',
  tex: String.raw`V(s)\leftarrow\max_a\big[r+\gamma V(s')\big]`,
  params: [{ k: 'n', label: 'sweeps', min: 0, max: 20, step: 1, init: 3, play: true }, { k: 'g', label: 'discount $\\gamma$', min: 0.5, max: 0.99, step: 0.01, init: 0.9 }, { k: 'c', label: 'cost per step', min: 0, max: 0.5, step: 0.01, init: 0.02 }],
  derive: ({ n, g, c }) => { const H = 6, wall = (i, j) => i < 0 || j < 0 || i >= H || j >= H || WORLD[i][j] === '#', M = [[-1, 0], [1, 0], [0, -1], [0, 1]]; let V = WORLD.map(r => [...r].map(() => 0)); const q = (U, i, j, a) => { const x = i + M[a][0], y = j + M[a][1], [p, r] = wall(x, y) ? [i, j] : [x, y]; return (WORLD[p][r] === 'G' ? 1 : 0) - c + g * (WORLD[p][r] === 'G' ? 0 : U[p][r]); }; for (let t = 0; t < n; t++) V = V.map((r, i) => r.map((_, j) => (WORLD[i][j] === '#' || WORLD[i][j] === 'G' ? 0 : Math.max(...[0, 1, 2, 3].map(a => q(V, i, j, a)))))); const pol = []; for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) if (WORLD[i][j] === '.') { const Q = [0, 1, 2, 3].map(a => q(V, i, j, a)), a = Q.indexOf(Math.max(...Q)); if (V[i][j] > 1e-9) pol.push([j + 0.5 - M[a][1] * 0.28, 5.5 - i + M[a][0] * 0.28, j + 0.5 + M[a][1] * 0.28, 5.5 - i - M[a][0] * 0.28]); } return { V, pol, d: V.map((r, i) => r.map((v, j) => (WORLD[i][j] === '#' ? 0 : Math.max(0, v)))) }; },
  plots: v => [{ ...heatPlot(v.d, { w: 400, h: 390, label: 'state values and best moves', heat: { max: 1, name: (i, j, x) => (WORLD[i][j] === '#' ? 'wall' : `value ${x.toFixed(3)}`) } }), marks: [{ as: 'heat', data: v.d, max: 1, name: (i, j, x) => (WORLD[i][j] === '#' ? 'wall' : WORLD[i][j] === 'G' ? 'goal' : `value ${x.toFixed(3)}`) }, { as: 'rects', tone: 'muted', data: WORLD.flatMap((r, i) => [...r].flatMap((c, j) => (c === '#' ? [[j + 0.04, 5 - i + 0.04, j + 0.96, 6 - i - 0.04]] : []))) }, { as: 'arrow', tone: 'fg', data: v.pol }, { as: 'label', x: 5.5, y: 0.42, anchor: 'middle', text: 'goal' }] }],
  stats: v => [`value at the start ${fmt(v.V[0][0], 3)}`, `cells with a decided move ${v.pol.length}`],
  takeaway: 'Reward information spreads outwards one cell per sweep, so the policy fills in from the goal.',
});
export const VizNtkMatrix = lab({
  intro: 'The empirical tangent kernel of a real random network, between pairs of inputs. Wider networks concentrate around a fixed kernel.',
  tex: String.raw`\Theta(x,x')=\nabla_wf(x)^\top\nabla_wf(x')`,
  params: [{ k: 'm', label: 'network width', min: 2, max: 400, step: 2, init: 40 }, { k: 'seed', seed: 'New network', init: 1 }],
  derive: ({ m, seed }) => { const r = rng(seed * 13), X = linspace(-2, 2, 14), net = Array.from({ length: m }, () => ({ a: gauss(r), b: gauss(r), c: gauss(r) })), phi = x => net.flatMap(({ a, b, c }) => { const z = a * x + b; return [c * (1 - Math.tanh(z) ** 2) * x, c * (1 - Math.tanh(z) ** 2), Math.tanh(z)]; }), J = X.map(phi), K = J.map(u => J.map(w => u.reduce((s, x, i) => s + x * w[i], 0) / m)); const v0 = Array(14).fill(1); let v = v0, lam = 0; for (let it = 0; it < 60; it++) { const u = K.map(row => row.reduce((s, x, i) => s + x * v[i], 0)); lam = Math.hypot(...u); v = u.map(x => x / lam); } return { K, X, lam, tr: K.reduce((s, row, i) => s + row[i], 0) }; },
  plots: v => [heatPlot(v.K, { w: 420, h: 400, label: 'tangent kernel', heat: { rows: v.X.map((x, i) => (i % 3 ? '' : x.toFixed(1))), cols: v.X.map((x, i) => (i % 3 ? '' : x.toFixed(1))) } })],
  stats: v => [`largest eigenvalue ${fmt(v.lam, 3)}`, `share of the trace ${pct(v.lam / v.tr, 1)}`],
  takeaway: 'One direction dominates, so the matching smooth pattern is learned first.',
});
export const VizInterference = lab({
  intro: 'Features squeezed into two dimensions. The matrix shows how much each feature leaks into every other.',
  tex: String.raw`\hat x=W^\top Wx`,
  params: [{ k: 'n', label: 'features', min: 2, max: 12, step: 1, init: 6 }, { k: 'r', label: 'rotation', min: 0, max: 90, step: 1, init: 0 }, { k: 'p', label: 'chance a feature is active', min: 0.01, max: 1, step: 0.01, init: 0.05 }],
  derive: ({ n, r, p }) => { const D = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n + (r * Math.PI) / 180), M = D.map(a => D.map(b => Math.cos(a - b))); return { D, M, I: Math.max(0, ...M.flatMap((row, i) => row.filter((_, j) => j !== i).map(Math.abs))), both: 1 - (1 - p) ** (n - 1) }; },
  plots: v => [{ caption: 'Directions', x: [-1.3, 1.3], y: [-1.3, 1.3], w: 300, h: 300, label: 'feature directions', marks: v.D.map((a, i) => ({ as: 'arrow', tone: (i % 4) + 1, data: [[0, 0, Math.cos(a), Math.sin(a)]] })) }, heatPlot(v.M, { w: 320, h: 300, caption: 'Interference', label: 'interference matrix', heat: { div: true, max: 1, show: v.n <= 8 ? f2 : undefined } })],
  stats: v => [`largest interference ${fmt(v.I, 3)}`, `chance of overlap when a feature is active ${pct(v.both, 1)}`],
  takeaway: 'Sparse features rarely collide, so the off-diagonal cost is rarely paid.',
});
export const VizRayStrip = lab({
  intro: 'One camera ray, broken into segments. Each row is one quantity along the ray, scaled to its own peak, so the render equation reads from top to bottom.',
  tex: String.raw`\hat C=\sum_iT_i\alpha_ic_i`,
  params: [{ k: 'f', label: 'fog density', min: 0, max: 2, step: 0.05, init: 0.3 }, { k: 'w', label: 'object density', min: 0, max: 20, step: 0.5, init: 8 }, { k: 'p', label: 'object position', min: 3, max: 26, step: 1, init: 16 }],
  derive: ({ f, w, p }) => { const n = 30, d = 1 / 3, S = [], A = [], T = [], C = []; let tr = 1, total = 0; for (let i = 0; i < n; i++) { const s = (i >= 3 && i < 11 ? f : 0) + (i >= p && i < p + 3 ? w : 0), c = i >= p && i < p + 3 ? 1 : 0.25, a = 1 - Math.exp(-s * d); S.push(s); A.push(a); T.push(tr); C.push(tr * a * c); total += tr * a * c; tr *= 1 - a; } return { S, A, T, C, total, tr }; },
  plots: v => { const norm = x => { const m = Math.max(1e-9, ...x); return x.map(u => u / m); }; return [heatPlot([norm(v.S), norm(v.A), norm(v.T), norm(v.C)], { w: 560, h: 190, label: 'quantities along the ray', heat: { dense: true, max: 1, rows: ['density', 'opacity', 'transmit', 'colour'], name: (i, j, x) => `segment ${j + 1}, ${['density', 'opacity', 'transmittance', 'contribution'][i]}: ${x.toFixed(3)} of this row’s peak` } })]; },
  stats: v => [`rendered colour ${fmt(v.total, 3)}`, `light passing through ${fmt(v.tr, 3)}`],
  takeaway: 'Dense matter blocks everything behind it, so it wins the pixel.',
});

/* ---------- first steps in vision ---------- */
const PICS = { ring: (i, j) => Math.exp(-((Math.hypot(i - 5.5, j - 5.5) - 3.6) ** 2) / 1.2), cross: (i, j) => Math.exp(-(Math.min(Math.abs(i - 5.5), Math.abs(j - 5.5)) ** 2) / 1.5), ramp: (i, j) => (i + j) / 22 };
const pic = s => Array.from({ length: 12 }, (_, i) => Array.from({ length: 12 }, (_, j) => +PICS[s](i, j).toFixed(2)));
export const VizPixelGrid = lab({
  intro: 'An image is a grid of numbers. Each cell shows its pixel value from 0 to 255, and stronger colour is brighter.',
  tex: String.raw`x'=\text{clip}\big(c\,(x-\tfrac12)+\tfrac12+b,\;0,\;1\big)`,
  params: [{ k: 's', label: 'Picture', init: 'ring', options: [['ring', 'Ring'], ['cross', 'Cross'], ['ramp', 'Ramp']] }, { k: 'b', label: 'brightness $b$', min: -0.5, max: 0.5, step: 0.05, init: 0 }, { k: 'c', label: 'contrast $c$', min: 0.2, max: 3, step: 0.1, init: 1 }],
  derive: ({ s, b, c }) => { const M = pic(s).map(r => r.map(x => +clip(c * (x - 0.5) + 0.5 + b, 0, 1).toFixed(3))), xs = M.flat(); return { M, m: mean(xs), sd: std(xs), cut: xs.filter(x => x === 0 || x === 1).length }; },
  plots: v => [heatPlot(v.M, { w: 470, h: 440, label: 'pixel values', heat: { max: 1, show: x => Math.round(x * 255), name: (i, j, x) => `row ${i + 1}, column ${j + 1}: ${Math.round(x * 255)}` } })],
  stats: v => ['12 by 12 is 144 numbers, or 432 with colour', `mean ${fmt(v.m, 3)}, spread ${fmt(v.sd, 3)}`, `${v.cut} pixels at the limits`],
  takeaway: 'Pixels pushed to 0 or 255 lose their detail for good, so check the range before you normalise.',
});
const LETTER = ['............', '..########..', '..##........', '..##........', '..##........', '..######....', '..##........', '..##........', '..##........', '..##........', '..##........', '............'].map(r => [...r].map(c => (c === '#' ? 1 : 0)));
export const VizAugmentGrid = lab({
  intro: 'Augmentation changes the pixels and keeps the label. Flip, shift and crop the picture, and see how much of it survives.',
  tex: String.raw`x'_{i,j}=x_{\,i-d_y,\;j-d_x}`,
  params: [{ k: 'f', label: 'Flip', init: 'no', options: [['no', 'None'], ['yes', 'Left to right']] }, { k: 'dx', label: 'shift right', min: -6, max: 6, step: 1, init: 0 }, { k: 'dy', label: 'shift down', min: -6, max: 6, step: 1, init: 0 }, { k: 'c', label: 'centre crop size', min: 4, max: 12, step: 2, init: 12 }],
  derive: ({ f, dx, dy, c }) => { const n = 12, lo = (n - c) / 2, M = LETTER.map((row, i) => row.map((_, j) => { if (i < lo || i >= lo + c || j < lo || j >= lo + c) return 0; const a = i - dy, b = j - dx; if (a < 0 || a >= n || b < 0 || b >= n) return 0; return LETTER[a][f === 'yes' ? n - 1 - b : b]; })), kept = M.flat().reduce((s, x) => s + x, 0), all = LETTER.flat().reduce((s, x) => s + x, 0); return { M, kept, all, share: kept / all }; },
  plots: v => [heatPlot(LETTER, { w: 300, h: 290, caption: 'Original', label: 'original picture', heat: { max: 1, name: (i, j, x) => `row ${i + 1}, column ${j + 1}: ${x}` } }), heatPlot(v.M, { w: 300, h: 290, caption: 'Augmented', label: 'augmented picture', heat: { max: 1, name: (i, j, x) => `row ${i + 1}, column ${j + 1}: ${x}` } })],
  stats: v => [`${v.kept} of ${v.all} letter pixels remain`, `${pct(v.share, 0)} survives`],
  status: v => (v.share < 0.5 ? ['alert', 'So much is gone that the label may no longer hold.'] : v.f === 'yes' ? ['warn', 'A flipped letter is not the same letter, though a flipped cat is still a cat.'] : ['ok', 'The object is still recognisable.']),
  takeaway: 'An augmentation is valid only when the label is still true afterwards.',
});
const SCORE_CLASSES = ['cat', 'dog', 'car', 'ship', 'bird'];
export const VizClassScores = lab({
  intro: 'A classifier outputs one raw score per class. Softmax turns them into probabilities, and the loss looks only at the true class.',
  tex: String.raw`p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\qquad L=-\ln p_{\text{true}}`,
  params: [{ k: 'y', label: 'True class', init: 0, options: SCORE_CLASSES.map((c, i) => [i, c]) }, ...SCORE_CLASSES.map((c, i) => ({ k: `z${i}`, label: `score for ${c}`, min: -4, max: 6, step: 0.1, init: [2, 1.5, -1, -2, 0.5][i] }))],
  derive: v => { const p = softmaxT(SCORE_CLASSES.map((_, i) => v[`z${i}`]), 1), y = +v.y, top = p.indexOf(Math.max(...p)); return { p, top, Y: y, L: -Math.log(p[y]) }; },
  plots: v => [{ x: [-0.5, 4.5], y: [0, 1.1], w: 500, h: 280, xl: 'class', yl: 'probability', label: 'class probabilities', marks: [...v.p.map((q, i) => ({ as: 'rects', tone: i === v.Y ? 3 : 1, data: [[i - 0.32, 0, i + 0.32, q]] })), ...v.p.map((q, i) => ({ as: 'label', x: i, y: q + 0.04, anchor: 'middle', text: `${SCORE_CLASSES[i]} ${q.toFixed(2)}` }))] }],
  stats: v => [`predicted: ${SCORE_CLASSES[v.top]}`, `probability of the true class ${fmt(v.p[v.Y], 3)}`, `cross-entropy ${fmt(v.L, 3)}`],
  status: v => (v.top === v.Y ? ['ok', 'The prediction is correct.'] : ['warn', 'The prediction is wrong, and the loss is large.']),
  legend: [['--c3', 'true class'], ['--c1', 'other classes']],
  takeaway: 'Adding the same amount to every score changes nothing. Only the differences matter.',
});

/* ---------- first steps in training ---------- */
export const VizGradStep = lab({
  intro: 'A loss shaped like a bowl with its minimum at 3. Drag the start and follow each update, then read the arithmetic in the table.',
  tex: String.raw`w_{t+1}=w_t-\eta\,a\,(w_t-3)`,
  params: [{ k: 'a', label: 'curvature $a$', min: 0.2, max: 3, step: 0.1, init: 1 }, { k: 'lr', label: 'learning rate $\\eta$', min: 0.05, max: 2.5, step: 0.05, init: 0.5 }, { k: 'w0', label: 'start $w_0$', min: -5, max: 11, step: 0.1, init: -3 }],
  derive: ({ a, lr, w0 }) => { const L = w => 0.5 * a * (w - 3) ** 2, W = [w0]; for (let t = 0; t < 8; t++) W.push(r6(clip(W[t] - lr * a * (W[t] - 3), -1e6, 1e6))); return { W, L, r: lr * a, top: Math.max(1, 1.25 * L(w0)) }; },
  plots: v => { const inb = v.W.filter(w => w >= -6 && w <= 12 && v.L(w) <= v.top), steps = v.W.slice(0, 8).map((w, i) => [w, v.W[i + 1]]).filter(([a, b]) => inb.includes(a) && inb.includes(b)); return [{ x: [-6, 12], y: [0, v.top], w: 560, h: 320, xl: 'w', yl: 'loss', label: 'gradient descent on a parabola', marks: [{ tone: 1, data: linspace(-6, 12, 361).filter(w => v.L(w) <= v.top).map(w => [w, v.L(w)]) }, { as: 'arrow', tone: 2, data: steps.map(([a, b]) => [a, v.L(a), b, v.L(b)]) }, { as: 'points', tone: 2, r: 4, data: inb.map(w => [w, v.L(w)]) }, { drag: { x: 'w0', y: v.L(v.w0), tone: 4 }, r: 7 }] }]; },
  table: v => ({ head: ['step', '$w$', 'gradient', 'next $w$'], rows: v.W.slice(0, 6).map((w, i) => [i, fmt(w, 3), fmt(v.a * (w - 3), 3), fmt(v.W[i + 1], 3)]) }),
  stats: v => [`$\\eta a=${fmt(v.r, 2)}$`, `loss after eight steps ${fmt(v.L(v.W[8]), 4)}`],
  status: v => (v.r >= 2 ? ['alert', 'Above $\\eta a=2$ every step lands further away, so training diverges.'] : v.r > 1 ? ['warn', 'Between 1 and 2 each step overshoots, but it still closes in.'] : ['ok', 'At or below 1 the steps approach without crossing the minimum.']),
  takeaway: 'The safe learning rate depends on the curvature, so steeper losses need smaller steps.',
});
export const VizBatchEpoch = lab({
  intro: 'Each row is an epoch and each column a training example. The colour and number say which batch used that example.',
  tex: String.raw`\text{steps per epoch}=\lceil N/B\rceil`,
  params: [{ k: 'N', label: 'examples $N$', min: 6, max: 30, step: 1, init: 18 }, { k: 'B', label: 'batch size $B$', min: 1, max: 16, step: 1, init: 4 }, { k: 'E', label: 'epochs', min: 1, max: 6, step: 1, init: 3 }, { k: 'd', label: 'Last small batch', init: 'keep', options: [['keep', 'Keep it'], ['drop', 'Drop it']] }, { k: 'seed', seed: 'Reshuffle', init: 1 }],
  derive: ({ N, B, E, d, seed }) => { const r = rng(seed * 97), n = d === 'drop' ? Math.floor(N / B) : Math.ceil(N / B), M = Array.from({ length: E }, () => { const o = Array.from({ length: N }, (_, i) => i); for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } const row = Array(N).fill(0); o.forEach((s, k) => { const b = Math.floor(k / B); row[s] = b < n ? b + 1 : 0; }); return row; }); return { M, n, unused: N - n * Math.min(B, N) > 0 && d === 'drop' ? N - n * B : 0 }; },
  plots: v => [heatPlot(v.M, { w: 620, h: 60 + 46 * v.E, label: 'batch assignment by epoch and example', heat: { min: 0, max: Math.max(1, v.n), rows: v.M.map((_, i) => `epoch ${i + 1}`), show: x => x || '', name: (i, j, x) => `epoch ${i + 1}, example ${j + 1}: ${x ? `batch ${x}` : 'not used'}` } })],
  stats: v => [`${v.n} steps per epoch`, `${v.n * v.E} updates in total`, `${v.unused} examples unused per epoch`],
  note: 'Blank cells were left out when the last small batch was dropped.',
  takeaway: 'Models improve per update, so halving the batch size doubles the updates in every epoch.',
});
export const VizLrGrid = lab({
  intro: 'Each row is a learning rate and each column a step. Cells show how far the loss has moved from where it started.',
  tex: String.raw`\frac{L_t}{L_0}=(1-\eta a)^{2t}`,
  params: [{ k: 'a', label: 'curvature $a$', min: 0.5, max: 4, step: 0.1, init: 2 }, { k: 'n', label: 'steps', min: 10, max: 40, step: 2, init: 24 }],
  derive: ({ a, n }) => { const E = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3], M = E.map(lr => Array.from({ length: n + 1 }, (_, t) => { const q = Math.abs(1 - lr * a); return t && q ? clip(2 * t * Math.log10(q), -6, 6) : t ? -6 : 0; })); return { E, M }; },
  plots: v => [heatPlot(v.M, { w: 600, h: 320, label: 'relative loss by learning rate and step', heat: { div: true, max: 6, tone: 4, neg: 1, rows: v.E.map(e => `η ${e}`), cols: v.M[0].map((_, t) => (t % 5 ? '' : `${t}`)), name: (i, j, x) => `learning rate ${v.E[i]}, step ${j}: loss ratio 1e${x.toFixed(1)}` } })],
  stats: v => [`fastest at $\\eta=1/a=${fmt(1 / v.a, 3)}$`, `diverges above $\\eta=2/a=${fmt(2 / v.a, 3)}$`],
  legend: [['--c1', 'loss fell'], ['--c4', 'loss grew']],
  takeaway: 'Too small crawls, too large explodes, and the useful band moves when the curvature changes.',
});

/* ---------- serving language models ---------- */
const stepProbs = (path, seed) => { const r = rng(seed * 53 + [...path].reduce((h, c) => (h * 7 + c.charCodeAt(0)) % 99991, 1)); return softmaxT([gauss(r) * 1.4, gauss(r) * 1.4, gauss(r) * 1.4], 1); };
const TOKS = ['a', 'b', 'c'];
const expand = (b, seed) => stepProbs(b.path, seed).map((p, k) => ({ path: b.path + TOKS[k], lp: b.lp + Math.log(p), parent: b.path }));
export const VizBeamTree = lab({
  intro: 'Beam search keeps only the best partial sequences at each step. Strong boxes survive, and faint ones are discarded. Each box shows its total log probability.',
  tex: String.raw`\text{score}(s)=\sum_t\log p(s_t\mid s_{<t})`,
  params: [{ k: 'W', label: 'beam width', min: 1, max: 3, step: 1, init: 2 }, { k: 'seed', seed: 'New model', init: 3 }],
  derive: ({ W, seed }) => { let F = [{ path: '', lp: 0 }]; const levels = []; for (let d = 0; d < 3; d++) { const C = F.flatMap(b => expand(b, seed)), keep = new Set([...C].sort((a, b) => b.lp - a.lp).slice(0, W).map(b => b.path)); levels.push(C.map(b => ({ ...b, kept: keep.has(b.path) }))); F = C.filter(b => keep.has(b.path)); } const found = F.reduce((a, b) => (b.lp > a.lp ? b : a)); let all = [{ path: '', lp: 0 }]; for (let d = 0; d < 3; d++) all = all.flatMap(b => expand(b, seed)); const best = all.reduce((a, b) => (b.lp > a.lp ? b : a)); return { levels, found, best, cost: levels.reduce((s, l) => s + l.length, 0) }; },
  plots: v => { const nodes = [{ id: 'root', x: 0.9, y: 5, w: 1.3, h: 0.8, text: 'start', level: 0.6, tone: 2 }], edges = []; v.levels.forEach((l, d) => l.forEach((b, i) => { nodes.push({ id: b.path, x: 3.5 + d * 3, y: 9.5 - (i + 0.5) * (9.5 / l.length) + 0.25, w: 2.1, h: Math.min(0.9, 8 / l.length), text: `${b.path}  ${b.lp.toFixed(2)}`, level: b.kept ? 0.75 : 0.08, tone: b.path === v.found.path ? 3 : 1, title: `${b.path}: probability ${Math.exp(b.lp).toFixed(3)}, ${b.kept ? 'kept' : 'discarded'}` }); edges.push([b.parent || 'root', b.path, b.kept ? 0.8 : 0.08]); })); return [{ bare: true, x: [0, 11.2], y: [0, 10], w: 580, h: 430, label: 'beam search tree', marks: [{ as: 'graph', nodes, edges }] }]; },
  stats: v => [`beam returns ${v.found.path} with probability ${fmt(Math.exp(v.found.lp), 3)}`, `best of all 27 sequences: ${v.best.path} at ${fmt(Math.exp(v.best.lp), 3)}`, `${v.cost} candidates scored instead of 39`],
  status: v => (v.found.path === v.best.path ? ['ok', 'The beam found the most likely sequence.'] : ['warn', 'The beam discarded the start of the most likely sequence.']),
  takeaway: 'A wider beam costs more and misses less, and it still gives no guarantee.',
});
export const VizServing = lab({
  intro: 'How many GPUs a steady load needs. Rows are requests per second, columns are output tokens per request, and the outlined cell is nearest your load.',
  tex: String.raw`\text{GPUs}=\Big\lceil\frac{\text{requests/s}\times\text{tokens}}{\text{tokens/s per GPU}\times\text{utilisation}}\Big\rceil`,
  params: [{ k: 'rps', label: 'requests per second', min: 1, max: 200, step: 1, init: 20 }, { k: 'tok', label: 'tokens per request', min: 50, max: 2000, step: 50, init: 500 }, { k: 'tps', label: 'tokens/s per GPU', min: 100, max: 5000, step: 100, init: 1000 }, { k: 'u', label: 'target utilisation', min: 0.3, max: 0.9, step: 0.05, init: 0.6 }, { k: 'price', label: 'dollars per GPU hour', min: 0.5, max: 8, step: 0.25, init: 2.5 }],
  derive: ({ rps, tok, tps, u, price }) => { const need = (r, t) => Math.ceil((r * t) / (tps * u) - 1e-9), R = [1, 2, 5, 10, 20, 50, 100, 200], T = [50, 100, 200, 500, 1000, 2000], G = R.map(r => T.map(t => need(r, t))), near = (A, x) => A.reduce((b, a, i) => (Math.abs(Math.log(a / x)) < Math.abs(Math.log(A[b] / x)) ? i : b), 0), g = need(rps, tok); return { R, T, G, M: G.map(r => r.map(x => Math.log10(x) + 0.3)), i: near(R, rps), j: near(T, tok), g, cost: g * price * 730, per: (g * price * 1000) / (3600 * rps) }; },
  plots: v => [heatPlot(v.M, { w: 560, h: 340, label: 'GPUs needed by load', over: [heatBox(v.M, v.i, v.j)], heat: { rows: v.R.map(r => `${r}/s`), cols: v.T.map(t => `${t}`), show: (_, i, j) => v.G[i][j], name: (i, j) => `${v.R[i]} requests/s, ${v.T[j]} tokens: ${v.G[i][j]} GPUs` } })],
  stats: v => [`${v.g} GPUs`, `about ${v.cost.toLocaleString('en-US', { maximumFractionDigits: 0 })} dollars a month`, `${v.per.toFixed(3)} dollars per thousand requests`],
  note: 'This assumes steady traffic. Real peaks need headroom, and batching raises the tokens per second of each GPU.',
  takeaway: 'Capacity follows total tokens per second, so long answers cost as much as extra users.',
});
export const VizSpeculative = lab({
  intro: 'A small model drafts several tokens, and the large model checks them in a single pass. Cells show the speed-up over using the large model alone.',
  tex: String.raw`S=\frac{1+\sum_{i=1}^{k}p^{\,i}}{1+k\,c}`,
  params: [{ k: 'c', label: 'draft cost per token $c$', min: 0.01, max: 0.5, step: 0.01, init: 0.1 }, { k: 'p', label: 'acceptance $p$', min: 0.5, max: 0.98, step: 0.01, init: 0.8 }, { k: 'k', label: 'drafted tokens $k$', min: 1, max: 12, step: 1, init: 4 }],
  derive: ({ c, p, k }) => { const tok = (q, m) => 1 + Array.from({ length: m }, (_, i) => q ** (i + 1)).reduce((s, x) => s + x, 0), S = (q, m) => tok(q, m) / (1 + m * c), P = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98], K = Array.from({ length: 12 }, (_, i) => i + 1), near = P.reduce((b, q, i) => (Math.abs(q - p) < Math.abs(P[b] - p) ? i : b), 0); return { P, K, near, M: P.map(q => K.map(m => S(q, m) - 1)), s: S(p, k), t: tok(p, k), best: K.reduce((b, m) => (S(p, m) > S(p, b) ? m : b), 1) }; },
  plots: v => [heatPlot(v.M, { w: 600, h: 340, label: 'speed-up by acceptance and draft length', over: [heatBox(v.M, v.near, v.k - 1)], heat: { div: true, max: 2.5, rows: v.P.map(q => `p ${q}`), cols: v.K.map(m => `${m}`), show: x => (x + 1).toFixed(1), name: (i, j, x) => `acceptance ${v.P[i]}, ${j + 1} drafted: ${(x + 1).toFixed(2)} times faster` } })],
  stats: v => [`${fmt(v.t, 2)} tokens per check`, `speed-up ${fmt(v.s, 2)}`, `best draft length here: ${v.best}`],
  status: v => (v.s < 1 ? ['alert', 'Drafting costs more than it saves at these settings.'] : undefined),
  note: 'This assumes that checking the drafts costs one forward pass of the large model. The output tokens themselves are unchanged.',
  legend: [['--c1', 'faster'], ['--c4', 'slower']],
  takeaway: 'Drafting pays only while most drafts are accepted, so the best length grows with acceptance.',
});
