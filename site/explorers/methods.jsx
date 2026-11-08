'use client';
// New explorers for the SVM, tree and practical sections.
import { R, curve, drill, int, lab, pick, truths } from '../components/labs';
import { mean, rng, gauss, polyfit, polyval, sum } from '../lib/math.mjs';
import { binomPmf, dataset, fmt, impurity, kernels, linspace, pct, smo } from '../lib/models.mjs';

/* ---------- support vector machines ---------- */

const STREET = [[-2, -1, -1], [-1, -2, -1], [-1.5, 0.5, -1], [1, 2, 1], [2, 0.5, 1], [0.6, 1, 1]];

export const WidestStreet = lab({
  intro: 'Many lines separate these points. Rotate and shift yours, trying to make the empty street as wide as you can.',
  params: [{ k: 'th', label: 'angle of the normal', min: 0, max: 180, step: 1, init: 90 }, { k: 'c', label: 'offset', min: -2, max: 2, step: 0.02, init: 0 }],
  derive: ({ th, c }) => {
    const margin = (t, o) => { const n = [Math.cos(t), Math.sin(t)]; return Math.min(...STREET.map(p => p[2] * (n[0] * p[0] + n[1] * p[1] + o))); };
    let best = -9;
    for (let a = 0; a < 180; a += 1) for (let o = -2; o <= 2; o += 0.02) best = Math.max(best, margin((a * Math.PI) / 180, o));
    const t = (th * Math.PI) / 180;
    return { n: [Math.cos(t), Math.sin(t)], m: margin(t, c), best };
  },
  plots: v => { const f = (a, b) => v.n[0] * a + v.n[1] * b + v.c; return [{ x: [-3.5, 3.5], y: [-3.5, 3.5], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'separating line with street', marks: [{ as: 'contour', tone: 'fg', fn: f }, v.m > 0 && { as: 'contour', tone: 3, dash: true, levels: [-v.m, v.m], fn: f }, { as: 'points', r: 6, data: STREET.map(p => [p[0], p[1], p[2] > 0 ? 1 : 2]) }] }]; },
  stats: v => [v.m > 0 ? `street half-width ${fmt(v.m, 3)}` : 'Not separating yet', `best possible ${fmt(v.best, 3)}`],
  status: v => (v.m > v.best - 0.02 ? ['ok', 'That is the maximum margin.'] : undefined),
  takeaway: 'A wide street tolerates new points landing a little away from the old ones, and only the points touching it decide where it goes.',
});

export const DistanceDrill = drill({
  intro: 'Plug the point in, then divide by the length of $w$.',
  tex: R`\text{distance}=\frac{|w^\top x+b|}{\|w\|}`,
  gen: r => { const [a, c] = pick(r, [[3, 4], [4, 3], [6, 8], [5, 12], [1, 0], [0, 2]]); return { a, c, b: int(r, -6, 6), x: int(r, -3, 3), y: int(r, -3, 3) }; },
  q: v => `Distance from $(${v.x},${v.y})$ to the line $${v.a}x_1+${v.c}x_2+(${v.b})=0$?`,
  answer: v => Math.abs(v.a * v.x + v.c * v.y + v.b) / Math.hypot(v.a, v.c), dp: 3,
  steps: v => [`Score: $${v.a}(${v.x})+${v.c}(${v.y})+(${v.b})=${v.a * v.x + v.c * v.y + v.b}$`, `$\\|w\\|=${Math.hypot(v.a, v.c)}$`],
});

export const MarginDrill = drill({
  intro: 'With the scale fixed so that support vectors score one, the full street width is two over the length of $w$.',
  tex: R`\text{width}=\frac{2}{\|w\|}`,
  gen: r => { const [a, c] = pick(r, [[3, 4], [1, 0], [0, 2], [6, 8], [5, 12]]); return { a, c }; },
  q: v => `For $w=(${v.a},${v.c})$, how wide is the street?`,
  answer: v => 2 / Math.hypot(v.a, v.c), dp: 3,
  steps: v => [`$\\|w\\|=\\sqrt{${v.a * v.a}+${v.c * v.c}}=${Math.hypot(v.a, v.c)}$`],
});

export const HingeDrill = drill({
  intro: 'A point inside the street, or on the wrong side, pays in proportion to how far it is from safety.',
  tex: R`\ell(y,f)=\max(0,\ 1-yf)`,
  gen: r => ({ y: pick(r, [-1, 1]), f: int(r, -30, 30) / 10 }),
  q: v => `For $y=${v.y}$ and score $f=${v.f}$, find the hinge loss.`,
  answer: v => Math.max(0, 1 - v.y * v.f),
  steps: v => [`Margin: $yf=${fmt(v.y * v.f)}$`, `$1-yf=${fmt(1 - v.y * v.f)}$, floored at zero.`],
});

export const KernelDrill = drill({
  intro: 'A kernel computes a dot product in a hidden feature space, without ever going there.',
  tex: R`k(x,z)=(1+x\cdot z)^2`,
  gen: r => ({ x: [int(r, -2, 3), int(r, -2, 3)], z: [int(r, -2, 3), int(r, -2, 3)] }),
  q: v => `Compute $k(x,z)$ for $x=(${v.x.join(',')})$ and $z=(${v.z.join(',')})$.`,
  answer: v => (1 + v.x[0] * v.z[0] + v.x[1] * v.z[1]) ** 2, dp: 0,
  steps: v => [`$x\\cdot z=${v.x[0] * v.z[0] + v.x[1] * v.z[1]}$`, `$(1+${v.x[0] * v.z[0] + v.x[1] * v.z[1]})^2$`],
});

export const RbfSimilarity = lab({
  intro: 'The Gaussian kernel is a similarity that decays with distance, and $\\gamma$ sets how quickly. Drag the second point away.',
  tex: R`k(x,z)=\exp\big(-\gamma\|x-z\|^2\big)`,
  params: [{ k: 'x', label: 'first coordinate of z', min: -3, max: 3, step: 0.05, init: 1 }, { k: 'y', label: 'second coordinate of z', min: -3, max: 3, step: 0.05, init: 0.6 }, { k: 'g', label: '$\\gamma$', min: 0.05, max: 6, step: 0.05, init: 1 }],
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, label: 'similarity field around a point', marks: [{ as: 'field', tones: [1, 1], bands: 10, fn: (a, b) => Math.exp(-v.g * (a * a + b * b)) }, { as: 'points', tone: 'fg', r: 6, data: [[0, 0]] }, { drag: { x: 'x', y: 'y', tone: 2 } }] }],
  stats: v => [`similarity ${fmt(Math.exp(-v.g * (v.x * v.x + v.y * v.y)), 4)}`, `reach about ${fmt(1 / Math.sqrt(v.g))}`],
  takeaway: 'An SVM with this kernel sums one bump per support vector. Large $\\gamma$ makes each bump narrow, so the model memorises.',
});

export const SupportVsC = lab({
  intro: 'Sweep the price of a violation. Fewer points stay support vectors as the model stops tolerating errors, and the margin narrows.',
  params: [{ k: 'kind', label: 'Data', options: [['overlap', 'Overlap'], ['blobs', 'Separable']], init: 'overlap' }, { k: 'lc', label: '$\\log_{10}C$', min: -2, max: 3, step: 0.5, init: 0 }],
  derive: ({ kind }) => { const D = dataset(kind, 3, 30), X = D.map(d => [d[0], d[1]]), y = D.map(d => (d[2] ? 1 : -1)), k = kernels.linear(); const rows = linspace(-2, 3, 11).map(l => { const C = 10 ** l, s = smo(X, y, { C, k }), w = [0, 1].map(j => sum(s.alpha.map((a, i) => a * y[i] * X[i][j]))); return { l, sv: s.alpha.filter(a => a > 1e-5).length, width: 2 / (Math.hypot(...w) || 1) }; }); return { rows }; },
  plots: v => { const cur = v.rows.find(r => Math.abs(r.l - v.lc) < 0.01); return [{ caption: 'Support vectors', x: [-2, 3], y: [0, 30], w: 300, h: 230, xl: 'log₁₀ C', label: 'support vectors against C', marks: [{ tone: 1, data: v.rows.map(r => [r.l, r.sv]) }, { as: 'points', tone: 4, r: 6, data: [[cur.l, cur.sv]] }] }, { caption: 'Street width', x: [-2, 3], y: [0, Math.min(12, Math.max(...v.rows.map(r => r.width)) * 1.1)], w: 300, h: 230, xl: 'log₁₀ C', label: 'margin width against C', marks: [{ tone: 2, data: v.rows.map(r => [r.l, r.width]) }, { as: 'points', tone: 4, r: 6, data: [[cur.l, cur.width]] }] }]; },
  stats: v => { const cur = v.rows.find(r => Math.abs(r.l - v.lc) < 0.01); return [`${cur.sv} support vectors, width ${fmt(cur.width)}`]; },
});

export const SvmCheck = truths({
  items: [
    ['Removing a non-support vector changes the solution.', false, 'Its $\\alpha$ is zero, so the solution does not depend on it.'],
    ['A point with $\\alpha=C$ may be misclassified.', true, 'It is inside the margin, or past it, and paying slack.'],
    ['A kernel has to compute $\\phi(x)$ first.', false, 'That is the trick: only the pairwise values are needed.'],
    ['Larger $C$ means a wider margin.', false, 'Larger $C$ punishes violations more, which narrows the street.'],
    ['SVM scores are probabilities.', false, 'They are signed distances, and need calibration to become probabilities.'],
  ],
});

/* ---------- trees and ensembles ---------- */

export const FollowTheTree = drill({
  intro: 'A tree is a sequence of questions. Follow the answers to a leaf.',
  gen: r => ({ t1: int(r, 2, 6), t2: int(r, 2, 6), t3: int(r, 2, 6), a: int(r, 1, 9), b: int(r, 1, 9), l: [int(r, 1, 9), int(r, 1, 9), int(r, 1, 9), int(r, 1, 9)] }),
  q: v => <>
    <div>For x₁ = {v.a} and x₂ = {v.b}, which value does this tree predict?</div>
    <ul className="rules"><li>is x₁ ≤ {v.t1}?<ul><li>yes: is x₂ ≤ {v.t2}? yes predicts {v.l[0]}, no predicts {v.l[1]}</li><li>no: is x₂ ≤ {v.t3}? yes predicts {v.l[2]}, no predicts {v.l[3]}</li></ul></li></ul>
  </>,
  answer: v => (v.a <= v.t1 ? (v.b <= v.t2 ? v.l[0] : v.l[1]) : v.b <= v.t3 ? v.l[2] : v.l[3]), dp: 0,
  steps: v => [`Is ${v.a} at most ${v.t1}? ${v.a <= v.t1 ? 'Yes.' : 'No.'}`, `Is ${v.b} at most ${v.a <= v.t1 ? v.t2 : v.t3}? ${v.b <= (v.a <= v.t1 ? v.t2 : v.t3) ? 'Yes.' : 'No.'}`],
});

export const EntropyDrill = drill({
  intro: 'Entropy is highest when a group is evenly mixed. Gini is a cheaper cousin.',
  tex: R`H=-p\log_2p-(1-p)\log_2(1-p),\qquad G=2p(1-p)`,
  gen: r => { const n = pick(r, [4, 8, 10, 16]), k = int(r, 1, n - 1); return { n, k, m: pick(r, ['entropy', 'gini']) }; },
  q: v => `A node holds ${v.k} positives out of ${v.n}. Find its ${v.m === 'gini' ? 'Gini impurity' : 'entropy'}.`,
  answer: v => impurity(v.m, v.k / v.n), dp: 3,
  steps: v => [`$p=${v.k}/${v.n}=${fmt(v.k / v.n, 3)}$`, v.m === 'gini' ? R`Substitute in $2p(1-p)$.` : R`Substitute in $-p\log_2p-(1-p)\log_2(1-p)$.`],
});

export const MajorityVote = lab({
  intro: 'A committee of independent, slightly-better-than-chance voters is right far more often than any one of them.',
  tex: R`P(\text{majority right})=\sum_{k>B/2}\binom Bk p^k(1-p)^{B-k}`,
  params: [{ k: 'p', label: 'each voter is right with $p$', min: 0.3, max: 0.9, step: 0.01, init: 0.6 }, { k: 'B', label: 'voters $B$', min: 1, max: 101, step: 2, init: 21 }],
  derive: ({ p }) => { const acc = B => { let s = 0; for (let k = Math.ceil(B / 2); k <= B; k++) s += binomPmf(k, B, p); return s; }; return { acc }; },
  plots: v => [{ x: [1, 101], y: [0, 1], xl: 'voters', yl: 'committee accuracy', label: 'majority vote accuracy', marks: [{ tone: 1, data: Array.from({ length: 51 }, (_, i) => [2 * i + 1, v.acc(2 * i + 1)]) }, { as: 'rule', dash: true, y: v.p }, { drag: { x: 'B', y: v.acc(v.B) } }] }],
  stats: v => [`committee ${pct(v.acc(v.B))}`, `single voter ${pct(v.p, 0)}`],
  status: v => (v.p < 0.5 ? ['alert', 'Voters worse than chance: the committee gets worse too.'] : undefined),
  takeaway: 'Real trees are correlated rather than independent, which is why random forests work to decorrelate them.',
});

export const EnsembleCheck = truths({
  items: [
    ['Bagging mainly reduces variance.', true, 'Averaging smooths the instability of deep trees.'],
    ['Boosting uses each model independently of the others.', false, 'Each round depends on the errors of those before it.'],
    ['More trees always overfit a random forest.', false, 'Extra trees stop helping, but do not cause overfitting.'],
    ['In AdaBoost, a mislabeled point gains weight.', true, 'It keeps being misclassified, so its weight keeps growing.'],
    ['Gradient boosting fits each tree to the original target.', false, 'It fits the residual, the negative gradient of the loss.'],
  ],
});

/* ---------- practical machine learning ---------- */

function splitErrors(frac, seed) {
  const out = [], base = rng(91), xs = Array.from({ length: 80 }, () => 2 * base() - 1), ys = xs.map(x => Math.sin(Math.PI * x) + 0.3 * gauss(base));
  for (let k = 0; k < 60; k++) {
    const r = rng(seed * 100 + k), idx = xs.map((_, i) => [r(), i]).sort((a, b) => a[0] - b[0]).map(p => p[1]), cut = Math.round(frac * 80), tr = idx.slice(0, cut), te = idx.slice(cut);
    const c = polyfit(tr.map(i => xs[i]), tr.map(i => ys[i]), 3);
    out.push(mean(te.map(i => (polyval(c, xs[i]) - ys[i]) ** 2)));
  }
  return out;
}

export const TrainTestSplit = lab({
  intro: 'How much data should you hold out? Repeat the split many times and see how much the test score moves.',
  params: [{ k: 'f', label: 'share used for training', min: 0.2, max: 0.95, step: 0.05, init: 0.8 }, { k: 'seed', seed: 'Resplit', init: 1 }],
  derive: ({ f, seed }) => { const e = splitErrors(f, seed), m = mean(e); return { e, m, sd: Math.sqrt(mean(e.map(v => (v - m) ** 2))) }; },
  plots: v => [{ x: [0, 60], y: [0, 0.5], xl: 'repeat', yl: 'test error', label: 'test error across repeated splits', marks: [{ as: 'stems', tone: 1, data: v.e.map((x, i) => [i + 0.5, 0, Math.min(x, 0.5)]) }, { as: 'rule', tone: 4, y: Math.min(v.m, 0.5) }] }],
  stats: v => [`average test error ${fmt(v.m, 3)}`, `spread between splits ${fmt(v.sd, 3)}`],
  takeaway: 'Too little training data biases the model, and too little test data makes the score noisy. Cross-validation is the way out of this choice.',
});

export const RSquaredDrill = drill({
  intro: 'R-squared compares your errors with those of always guessing the mean.',
  tex: R`R^2=1-\frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}`,
  gen: r => ({ tot: int(r, 40, 200), res: int(r, 2, 60) }),
  q: v => `$\\text{SS}_{\\text{res}}=${v.res}$ and $\\text{SS}_{\\text{tot}}=${v.tot}$. Find $R^2$.`,
  answer: v => 1 - v.res / v.tot, dp: 3,
  steps: v => [`Ratio: $${v.res}/${v.tot}=${fmt(v.res / v.tot, 3)}$`],
});

export const MacroMicroDrill = drill({
  intro: 'Macro averages over classes, and micro averages over examples.',
  gen: r => ({ n: [int(r, 50, 100), int(r, 10, 30), int(r, 2, 10)], k: pick(r, ['macro', 'micro']), seed: r() }),
  q: v => { const c = hits(v); return `Three classes have sizes ${v.n.join(', ')} and correct counts ${c.join(', ')}. Find the ${v.k} recall.`; },
  answer: v => { const c = hits(v); return v.k === 'macro' ? mean(c.map((x, i) => x / v.n[i])) : sum(c) / sum(v.n); }, dp: 3,
  steps: v => { const c = hits(v); return [v.k === 'macro' ? `Class recalls: $${c.map((x, i) => fmt(x / v.n[i], 3)).join(',\\ ')}$, then the plain average.` : `Total correct $${sum(c)}$ over total $${sum(v.n)}$.`]; },
});
function hits(v) { const r = rng(Math.floor(v.seed * 1e6)); return v.n.map(n => int(r, Math.ceil(n * 0.3), n)); }

export const KnnDrill = drill({
  intro: 'Find the nearest points, then take the majority label.',
  gen: r => { const q = int(r, 0, 12), xs = [...new Set(Array.from({ length: 9 }, () => int(r, 0, 20)))].filter(x => x !== q).slice(0, 5); while (xs.length < 3) xs.push(30 + xs.length); return { q, xs, ys: xs.map(() => int(r, 0, 1)) }; },
  q: v => `Points on a line: ${v.xs.map((x, i) => `$${x}$ (class ${v.ys[i] ? 'B' : 'A'})`).join(', ')}. With three neighbours, what class does $${v.q}$ get?`,
  choices: () => ['Class A', 'Class B'],
  answer: v => { const o = v.xs.map((x, i) => [Math.abs(x - v.q), v.ys[i], x]).sort((a, b) => a[0] - b[0] || a[2] - b[2]).slice(0, 3); return +(sum(o.map(p => p[1])) >= 2); },
  steps: v => [`Distances: $${v.xs.map(x => Math.abs(x - v.q)).join(',\\ ')}$`, 'Take the three smallest (ties to the smaller position), and count labels.'],
});

export const PracticeCheck = truths({
  items: [
    ['A random split detects distribution shift.', false, 'Both halves come from the same world.'],
    ['Scaling features on the full dataset first is harmless.', false, 'The scaler has seen test data, which is a mild leak.'],
    ['Higher $R^2$ always means a better model.', false, 'Training $R^2$ cannot fall as features are added.'],
    ['With $k$-NN, feature scale matters.', true, 'Distances are dominated by whichever feature has the largest range.'],
    ['Learning curves can tell you whether more data would help.', true, 'A persistent gap suggests variance, and converged high curves suggest bias.'],
  ],
});
