'use client';
// New explorers for the linear regression, logistic regression, regularization and evaluation sections.
import { R, curve, drill, int, lab, mat, pick, truths } from '../components/labs';
import { gauss, mean, rng, polyval, sum } from '../lib/math.mjs';
import { clamp, eig2, fmt, linspace, normalPdf, pct, ridge, sigmoid, soft } from '../lib/models.mjs';

const ints = (r, n, a, b) => Array.from({ length: n }, () => int(r, a, b));

/* ---------- linear regression and PCA ---------- */

export const ResidualDrill = drill({
  intro: 'A residual is truth minus prediction, and the sum of squares adds them up after squaring.',
  gen: r => ({ w: int(r, 1, 3), b: int(r, -2, 3), xs: ints(r, 3, 0, 4), es: ints(r, 3, -2, 2) }),
  q: v => { const ys = v.xs.map((x, i) => v.w * x + v.b + v.es[i]); return `Line $\\hat y=${v.w}x+(${v.b})$. Points: ${v.xs.map((x, i) => `$(${x},${ys[i]})$`).join(', ')}. Find the sum of squared residuals.`; },
  answer: v => sum(v.es.map(e => e * e)), dp: 0,
  steps: v => [`Predictions: $${v.xs.map(x => v.w * x + v.b).join(',\\ ')}$`, `Residuals: $${v.es.join(',\\ ')}$`, `Squares: $${v.es.map(e => e * e).join('+')}$`],
});

const SLICE = (() => { const r = rng(33); return Array.from({ length: 140 }, () => { const x = -3 + 6 * r(); return [x, gauss(r)]; }); })();

export const ConditionalMean = lab({
  intro: 'Regression estimates the **average of $y$ at each $x$**. Drag the slice and read the mean of the points inside it.',
  tex: R`f^*(x)=\mathbb E[Y\mid X=x]`,
  params: [{ k: 'kind', label: 'World', options: [['lin', 'linear'], ['curve', 'curved']], init: 'curve' }, { k: 'x0', label: 'slice at $x$', min: -2.6, max: 2.6, step: 0.05, init: 1 }, { k: 'h', label: 'slice half-width', min: 0.1, max: 1, step: 0.05, init: 0.4 }],
  derive: ({ kind, x0, h }) => { const f = kind === 'lin' ? x => 0.8 * x : x => 0.6 * x * x - 1, pts = SLICE.map(([x, e]) => [x, f(x) + 0.5 * e]), ins = pts.filter(p => Math.abs(p[0] - x0) < h); return { f, pts, ins, m: ins.length ? mean(ins.map(p => p[1])) : NaN }; },
  plots: v => [{ x: [-3, 3], y: [-3, 5], xl: 'x', yl: 'y', label: 'scatter with a vertical slice', marks: [{ as: 'rects', tone: 1, data: [[v.x0 - v.h, -3, v.x0 + v.h, 5]] }, { as: 'points', tone: 'muted', r: 2.5, data: v.pts }, { as: 'points', tone: 'fg', r: 3, data: v.ins }, { tone: 3, data: curve(v.f, -3, 3) }, Number.isFinite(v.m) && { as: 'points', tone: 4, r: 7, data: [[v.x0, v.m]] }, { drag: { x: 'x0', y: -2.6, tone: 1 } }] }],
  stats: v => [`slice mean ${fmt(v.m)}`, `true mean ${fmt(v.f(v.x0))}, using ${v.ins.length} points`],
  takeaway: 'A narrow slice is unbiased but noisy, and a wide one is steady but blurred. Models assume a shape to borrow strength across slices.',
});

export const SlopeDrill = drill({
  intro: 'Simple regression has a closed form using deviations from the means.',
  tex: R`\hat w=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},\qquad \hat b=\bar y-\hat w\bar x`,
  gen: r => ({ xs: [0, 1, 2, 3], ys: ints(r, 4, 0, 9) }),
  q: v => `Points: ${v.xs.map((x, i) => `$(${x},${v.ys[i]})$`).join(', ')}. Find the slope $\\hat w$.`,
  answer: v => { const mx = mean(v.xs), my = mean(v.ys); return sum(v.xs.map((x, i) => (x - mx) * (v.ys[i] - my))) / sum(v.xs.map(x => (x - mx) ** 2)); },
  steps: v => { const mx = mean(v.xs), my = mean(v.ys); return [`$\\bar x=${fmt(mx)}$, $\\bar y=${fmt(my)}$`, `Numerator: $${fmt(sum(v.xs.map((x, i) => (x - mx) * (v.ys[i] - my))))}$`, `Denominator: $${fmt(sum(v.xs.map(x => (x - mx) ** 2)))}$`]; },
});

export const GramDrill = drill({
  intro: 'Entry $(i,j)$ of $X^\\top X$ is column $i$ dotted with column $j$.',
  gen: r => ({ X: Array.from({ length: 3 }, () => ints(r, 2, -2, 3)), i: int(r, 0, 1), j: int(r, 0, 1) }),
  q: v => `For $X=${mat(v.X)}$, find entry row ${v.i ? 'two' : 'one'}, column ${v.j ? 'two' : 'one'} of $X^\\top X$.`,
  answer: v => sum(v.X.map(row => row[v.i] * row[v.j])), dp: 0,
  steps: v => [`Column pair: $(${v.X.map(x => x[v.i]).join(',')})$ and $(${v.X.map(x => x[v.j]).join(',')})$`, `Dot product: $${v.X.map(x => `(${x[v.i]})(${x[v.j]})`).join('+')}$`],
});

export const NormalEquationDrill = drill({
  intro: 'Solve the two equations by elimination or by the two-by-two inverse.',
  tex: R`X^\top X\,\hat\theta=X^\top y`,
  gen: r => { const a = int(r, 2, 6), b = int(r, 0, 3), d = int(r, b + 1, 7), t = [int(r, -2, 3), int(r, -2, 3)]; return { a, b, d, t, c: [a * t[0] + b * t[1], b * t[0] + d * t[1]], k: int(r, 0, 1) }; },
  q: v => `Solve $${mat([[v.a, v.b], [v.b, v.d]])}\\theta=${mat([[v.c[0]], [v.c[1]]])}$. What is the ${v.k ? 'second' : 'first'} entry of $\\theta$?`,
  answer: v => v.t[v.k], dp: 2,
  steps: v => [`Determinant: $${v.a}\\cdot${v.d}-${v.b}^2=${v.a * v.d - v.b * v.b}$`, `$\\theta_1=(${v.d}\\cdot${v.c[0]}-${v.b}\\cdot${v.c[1]})/${v.a * v.d - v.b * v.b}$, $\\theta_2=(${v.a}\\cdot${v.c[1]}-${v.b}\\cdot${v.c[0]})/${v.a * v.d - v.b * v.b}$`],
});

export const EigenDrill = drill({
  intro: 'For a symmetric two-by-two matrix, the eigenvalues come from the trace and determinant.',
  tex: R`\lambda=\frac{\operatorname{tr}}{2}\pm\sqrt{\Big(\frac{\operatorname{tr}}{2}\Big)^2-\det}`,
  gen: r => { const a = int(r, 1, 6), b = int(r, -3, 3); return { a, b, d: a + 2 * int(r, 0, 2) }; },
  q: v => `Find the larger eigenvalue of $${mat([[v.a, v.b], [v.b, v.d]])}$.`,
  answer: v => eig2(v.a, v.b, v.d).l1,
  steps: v => [`Trace: $${v.a + v.d}$. Determinant: $${v.a * v.d - v.b * v.b}$.`, `$\\lambda_1=${(v.a + v.d) / 2}+\\sqrt{${((v.a + v.d) / 2) ** 2}-(${v.a * v.d - v.b * v.b})}$`],
});

export const ScreePlot = lab({
  intro: 'How many components do you need? With $p$ equally correlated features, one strong component appears and the rest share what remains.',
  tex: R`\lambda_1=1+(p-1)\rho,\qquad \lambda_{2..p}=1-\rho`,
  params: [{ k: 'p', label: 'features $p$', min: 2, max: 20, step: 1, init: 8 }, { k: 'rho', label: 'correlation $\\rho$', min: 0, max: 0.95, step: 0.01, init: 0.6 }, { k: 'k', label: 'components kept', min: 1, max: 20, step: 1, init: 2 }],
  derive: ({ p, rho, k }) => { const L = [1 + (p - 1) * rho, ...Array(p - 1).fill(1 - rho)], kk = Math.min(k, p); return { L, kk, kept: sum(L.slice(0, kk)) / p }; },
  plots: v => [{ x: [0.4, v.p + 0.6], y: [0, v.L[0] * 1.1], xl: 'component', yl: 'variance', label: 'scree plot', marks: [{ as: 'rects', tone: 1, data: v.L.map((l, i) => [i + 0.7, 0, i + 1.3, l]) }, { as: 'rule', tone: 4, x: v.kk + 0.5 }] }],
  stats: v => [`variance kept ${pct(v.kept, 0)}`, `error left ${fmt(v.p * (1 - v.kept), 2)} of ${v.p}`],
  takeaway: 'Reconstruction error is exactly the sum of the discarded eigenvalues.',
});

export const ScaleSensitivity = lab({
  intro: 'Change one feature from metres to millimetres and PCA changes its mind. Scale the first feature and watch the top direction swing.',
  params: [{ k: 'lg', label: '$\\log_{10}$ scale of the first feature', min: -1, max: 1, step: 0.02, init: 0 }, { k: 'rho', label: 'correlation', min: 0, max: 0.95, step: 0.01, init: 0.5 }],
  derive: ({ lg, rho }) => { const s = 10 ** lg, e = eig2(s * s, rho * s, 1); return { s, e, ang: (Math.atan2(e.v[1], e.v[0]) * 180) / Math.PI }; },
  plots: v => [{ x: [-4, 4], y: [-4, 4], w: 380, h: 360, xl: 'scaled x₁', yl: 'x₂', label: 'principal direction under rescaling', marks: [{ tone: 2, data: linspace(0, 6.3, 91).map(t => { const a = Math.sqrt(v.e.l1) * 1.5, b = Math.sqrt(Math.max(v.e.l2, 0)) * 1.5; return [a * Math.cos(t) * v.e.v[0] + b * Math.sin(t) * v.e.u[0], a * Math.cos(t) * v.e.v[1] + b * Math.sin(t) * v.e.u[1]]; }) }, { as: 'arrow', tone: 1, data: [[0, 0, 3 * v.e.v[0], 3 * v.e.v[1]]] }] }],
  stats: v => [`PC1 angle ${fmt(Math.abs(v.ang) > 90 ? 180 - Math.abs(v.ang) : Math.abs(v.ang), 0)} degrees from x₁`],
  takeaway: 'Standardize first, unless the units are truly comparable.',
});

export const LinearCheck = truths({
  items: [
    ['The residual vector is perpendicular to every column of $X$.', true, 'That is exactly the normal equation $X^\\top(y-X\\hat\\theta)=0$.'],
    ['A zero residual sum means a perfect fit.', false, 'Positive and negative misses cancel, and with an intercept the sum is always zero.'],
    ['$X^\\top X$ is always invertible.', false, 'Duplicate columns or fewer rows than columns make it singular.'],
    ['PCA needs labels.', false, 'It uses only the covariance of the features.'],
    ['Eigenvectors of a symmetric matrix are perpendicular.', true, 'Distinct eigenvalues give orthogonal eigenvectors, by the spectral theorem.'],
  ],
});

/* ---------- logistic regression ---------- */

const ONE = [[-3, 0], [-2.2, 0], [-1.5, 0], [-0.8, 1], [-0.3, 0], [0.4, 0], [0.9, 1], [1.6, 1], [2.3, 1], [3, 1]];

export const SigmoidFit = lab({
  intro: 'Slope sets how sharp the S is, and intercept slides it. Each stem is a point’s log loss, and confident mistakes tower over everything.',
  tex: R`p=\sigma(wx+b),\qquad \text{loss}=-y\ln p-(1-y)\ln(1-p)`,
  params: [{ k: 'w', label: 'slope $w$', min: -3, max: 6, step: 0.05, init: 1 }, { k: 'b', label: 'intercept $b$', min: -4, max: 4, step: 0.05, init: 0 }],
  derive: ({ w, b }) => { const L = ONE.map(([x, y]) => { const p = clamp(sigmoid(w * x + b), 1e-6, 1 - 1e-6); return -(y ? Math.log(p) : Math.log(1 - p)); }); return { L, loss: mean(L) }; },
  plots: v => [{ x: [-4, 4], y: [0, 2], xl: 'x', yl: 'p and loss', label: 'sigmoid fit with loss stems', marks: [{ as: 'stems', tone: 4, data: ONE.map(([x], i) => [x, 0, Math.min(v.L[i], 2)]) }, { tone: 1, data: curve(x => sigmoid(v.w * x + v.b), -4, 4) }, { as: 'points', data: ONE.map(([x, y]) => [x, y, y ? 1 : 2]) }] }],
  stats: v => [`average log loss ${fmt(v.loss, 3)}`, `boundary at x = ${v.w ? fmt(-v.b / v.w) : 'none'}`],
});

const TWO_D = [[-2, -1, 0], [-1.5, 1, 0], [-1, -2, 0], [-0.5, 0.5, 0], [0.2, -1.5, 0], [0.5, 1.5, 1], [1, 0.2, 1], [1.5, -0.5, 1], [2, 1.5, 1], [0.8, 2.3, 1]];

export const BoundaryLine = lab({
  intro: 'A linear classifier is a line and a direction. Drag the arrow tip to aim it, and shift the line with the offset.',
  tex: R`w_1x_1+w_2x_2+b>0\ \Rightarrow\ \text{class one}`,
  params: [{ k: 'w1', label: '$w_1$', min: -3, max: 3, step: 0.05, init: 1 }, { k: 'w2', label: '$w_2$', min: -3, max: 3, step: 0.05, init: -0.5 }, { k: 'b', label: 'offset $b$', min: -4, max: 4, step: 0.05, init: 0 }],
  plots: v => [{ x: [-3, 3], y: [-3, 3], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'linear decision boundary', marks: [{ as: 'field', fn: (a, b) => 2 * sigmoid(v.w1 * a + v.w2 * b + v.b) - 1 }, { as: 'contour', tone: 'fg', fn: (a, b) => v.w1 * a + v.w2 * b + v.b }, { as: 'points', r: 5, data: TWO_D.map(d => [d[0], d[1], d[2] ? 1 : 2]) }, { as: 'arrow', tone: 3, data: [[0, 0, v.w1, v.w2]] }, { drag: { x: 'w1', y: 'w2', tone: 3 } }] }],
  stats: v => [`accuracy ${pct(mean(TWO_D.map(d => +((v.w1 * d[0] + v.w2 * d[1] + v.b > 0) === !!d[2]))), 0)}`],
  takeaway: 'The weight vector is perpendicular to the boundary and points to the positive side.',
});

export const BceDrill = drill({
  intro: 'Log loss looks only at the probability given to the true class.',
  tex: R`\text{loss}=-\ln(\text{probability of the true class})`,
  gen: r => ({ y: int(r, 0, 1), p: pick(r, [0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9]) }),
  q: v => `The model says $p=P(y{=}1)=${v.p}$ and the true label is $y=${v.y}$. What is the log loss? (natural log)`,
  answer: v => -Math.log(v.y ? v.p : 1 - v.p), dp: 3,
  steps: v => [`Probability of the true class: ${v.y ? v.p : fmt(1 - v.p)}`, `$-\\ln(${v.y ? v.p : fmt(1 - v.p)})$`],
});

export const GradientDrill = drill({
  intro: 'For one example the slope of the loss has a tidy form: error times feature.',
  tex: R`\frac{\partial\,\text{loss}}{\partial w}=(p-y)\,x`,
  gen: r => { const z = pick(r, [0, Math.log(3), -Math.log(3), Math.log(4), -Math.log(4)]); return { p: sigmoid(z), y: int(r, 0, 1), x: int(r, -3, 3) || 2 }; },
  q: v => `With $p=${fmt(v.p, 2)}$, $y=${v.y}$ and $x=${v.x}$, find the gradient with respect to $w$.`,
  answer: v => (v.p - v.y) * v.x,
  steps: v => [`Error: $p-y=${fmt(v.p - v.y, 2)}$`, `Multiply by $x=${v.x}$.`],
  takeaway: 'A correct, confident prediction has almost no error, and so almost no pull.',
});

export const HessianTypeDrill = drill({
  intro: 'Read the shape from the determinant and trace of a symmetric matrix.',
  gen: r => ({ a: int(r, -4, 5) || 2, b: int(r, -4, 4), d: int(r, -4, 5) || 1 }),
  q: v => `What surface does the Hessian $${mat([[v.a, v.b], [v.b, v.d]])}$ describe?`,
  choices: () => ['Bowl', 'Dome', 'Saddle', 'Flat in one direction'],
  answer: v => { const det = v.a * v.d - v.b * v.b; return det === 0 ? 3 : det < 0 ? 2 : v.a > 0 ? 0 : 1; },
  steps: v => [`Determinant: $${v.a}\\cdot${v.d}-(${v.b})^2=${v.a * v.d - v.b * v.b}$`, 'Negative means mixed signs (saddle). Positive means the same sign, and the trace gives which.'],
});

export const IrlsWeights = lab({
  intro: 'Newton’s method weights each example by $p(1-p)$. Drag the score: points near the boundary carry the most curvature, and decided points carry almost none.',
  tex: R`H=\tfrac1n\sum_i p_i(1-p_i)\,x_ix_i^\top`,
  params: [{ k: 'z', label: 'score $z$', min: -6, max: 6, step: 0.05, init: 1 }],
  plots: v => [{ x: [-6, 6], y: [0, 1], xl: 'score z', yl: 'p and weight', label: 'probability and curvature weight', marks: [{ tone: 1, data: curve(sigmoid, -6, 6) }, { tone: 2, data: curve(z => 4 * sigmoid(z) * (1 - sigmoid(z)), -6, 6) }, { drag: { x: 'z', y: sigmoid(v.z) } }] }],
  stats: v => [`weight ${fmt(sigmoid(v.z) * (1 - sigmoid(v.z)), 4)}`, `largest possible is 0.25, at z = 0`],
  legend: [['--c1', 'p'], ['--c2', 'weight, scaled up by four']],
});

export const SeparableRidge = lab({
  intro: 'On separable data the loss has no finite minimum. A small penalty brings one back, and sets how confident the model may be.',
  tex: R`\min_w\ \frac1n\sum_i\text{loss}_i+\frac\lambda2w^2`,
  params: [{ k: 'lg', label: '$\\log_{10}\\lambda$', min: -5, max: 0, step: 0.05, init: -2 }],
  derive: ({ lg }) => { const xs = [-2, -1.2, -0.5, 0.5, 1.2, 2], ys = [0, 0, 0, 1, 1, 1]; const fit = l => { let w = 0.5; for (let i = 0; i < 80; i++) { let g = l * w, h = l; xs.forEach((x, k) => { const p = sigmoid(w * x); g += ((p - ys[k]) * x) / 6; h += (p * (1 - p) * x * x) / 6; }); w -= clamp(g / h, -2, 2); } return w; }; return { w: fit(10 ** lg), xs, ys, fit }; },
  plots: v => [{ caption: 'Fit', x: [-3, 3], y: [0, 1], w: 300, h: 240, xl: 'x', label: 'separable data with sigmoid', marks: [{ tone: 1, data: curve(x => sigmoid(v.w * x), -3, 3) }, { as: 'points', data: v.xs.map((x, i) => [x, v.ys[i], v.ys[i] ? 1 : 2]) }] }, { caption: 'Weight against penalty', x: [-5, 0], y: [0, 30], w: 300, h: 240, xl: 'log₁₀ λ', yl: 'w', label: 'weight against penalty', marks: [{ tone: 2, data: linspace(-5, 0, 41).map(l => [l, Math.min(v.fit(10 ** l), 40)]) }, { as: 'rule', dash: true, x: v.lg }] }],
  stats: v => [`w = ${fmt(v.w)}`],
});

export const LossShapes = lab({
  intro: 'Why not squared error for probabilities? Compare the two as the prediction for a true positive gets worse.',
  params: [{ k: 'p', label: 'predicted probability', min: 0.01, max: 0.99, step: 0.01, init: 0.2 }],
  plots: v => [{ x: [0, 1], y: [0, 5], xl: 'probability given to the true class', yl: 'loss', label: 'log loss and squared loss', marks: [{ tone: 1, data: curve(p => -Math.log(p), 0.005, 1) }, { tone: 2, data: curve(p => (1 - p) ** 2, 0.005, 1) }, { drag: { x: 'p', y: 0.3 } }] }],
  stats: v => [`log loss ${fmt(-Math.log(v.p), 3)}`, `squared loss ${fmt((1 - v.p) ** 2, 3)}`],
  legend: [['--c1', 'log loss'], ['--c2', 'squared loss']],
  takeaway: 'Squared error tops out at one, however wrong the model is. Log loss keeps pushing, and it stays convex in the weights.',
});

export const LogisticCheck = truths({
  items: [
    ['A coefficient of 0.7 raises the probability by 0.7.', false, 'It adds 0.7 to the log-odds, multiplying the odds by about two.'],
    ['Logistic loss is convex in the weights.', true, 'Its Hessian $X^\\top WX$ is positive semidefinite.'],
    ['Separable data give the best possible logistic fit.', false, 'The MLE does not exist: weights grow without bound.'],
    ['The decision boundary of logistic regression is linear in the features.', true, '$p=0.5$ exactly when the linear score is zero.'],
    ['Newton’s method needs a learning rate to converge on logistic loss.', false, 'A plain step is usually enough, although damping can protect against overshoot.'],
  ],
});

/* ---------- regularization ---------- */

const OVER = (() => { const r = rng(5), xs = linspace(-1, 1, 11); return { xs, ys: xs.map(x => Math.sin(2.4 * x) + 0.25 * gauss(r)) }; })();

export const PenaltySlider = lab({
  intro: 'A flexible polynomial chases the noise. Turn up the penalty and watch it calm down, without removing a single feature.',
  tex: R`\min_\theta\ \tfrac1n\|y-X\theta\|^2+\lambda\|\theta\|^2`,
  params: [{ k: 'lg', label: '$\\log_{10}\\lambda$', min: -8, max: 1, step: 0.1, init: -6 }],
  derive: ({ lg }) => { const X = OVER.xs.map(x => Array.from({ length: 10 }, (_, j) => x ** j)), c = ridge(X, OVER.ys, 10 ** lg), truth = x => Math.sin(2.4 * x); return { c, truth, tr: mean(OVER.xs.map((x, i) => (polyval(c, x) - OVER.ys[i]) ** 2)), te: mean(linspace(-1, 1, 201).map(x => (polyval(c, x) - truth(x)) ** 2)), norm: Math.hypot(...c) }; },
  plots: v => [{ x: [-1, 1], y: [-2.5, 2.5], xl: 'x', yl: 'y', label: 'ridge polynomial fit', marks: [{ tone: 3, dash: true, data: curve(v.truth, -1, 1) }, { tone: 1, data: curve(x => clamp(polyval(v.c, x), -4, 4), -1, 1, 301) }, { as: 'points', tone: 'fg', data: OVER.xs.map((x, i) => [x, OVER.ys[i]]) }] }],
  stats: v => [`error to the truth ${fmt(v.te, 3)}`, `training error ${fmt(v.tr, 3)}`, `size of coefficients ${fmt(v.norm, 1)}`],
  takeaway: 'Too little penalty overfits, and too much underfits. The best setting is somewhere between, and it has to be found on held-out data.',
});

export const RidgeDrill = drill({
  intro: 'With a single feature, ridge just adds $\\lambda$ to the denominator.',
  tex: R`\hat\beta=\frac{\sum_i x_iy_i}{\sum_i x_i^2+\lambda}`,
  gen: r => ({ xy: int(r, 2, 40), xx: int(r, 2, 30), l: int(r, 0, 10) }),
  q: v => `$\\sum x_iy_i=${v.xy}$, $\\sum x_i^2=${v.xx}$ and $\\lambda=${v.l}$. Find $\\hat\\beta$.`,
  answer: v => v.xy / (v.xx + v.l), dp: 3,
  steps: v => [`Denominator: $${v.xx}+${v.l}=${v.xx + v.l}$`, `$${v.xy}/${v.xx + v.l}$`],
});

export const SoftDrill = drill({
  intro: 'Move towards zero by $\\lambda$, and stop at zero.',
  tex: R`\operatorname{soft}(z,\lambda)=\operatorname{sign}(z)\max(|z|-\lambda,0)`,
  gen: r => ({ z: int(r, -40, 40) / 10, l: int(r, 1, 25) / 10 }),
  q: v => `Compute the lasso estimate for $z=${v.z}$ and $\\lambda=${v.l}$.`,
  answer: v => soft(v.z, v.l),
  steps: v => [`$|z|-\\lambda=${fmt(Math.abs(v.z) - v.l)}$`, Math.abs(v.z) <= v.l ? 'That is not positive, so the coefficient is zero.' : `Keep the sign of $z$.`],
});

export const EigenFilter = lab({
  intro: 'In the eigenbasis of $X^\\top X$, ridge multiplies each direction by its own factor. Strong directions pass, and weak ones are silenced.',
  tex: R`\hat\beta_j^{\text{ridge}}=\frac{\mu_j}{\mu_j+\lambda}\,\hat\beta_j^{\text{OLS}}`,
  params: [{ k: 'lg', label: '$\\log_{10}\\lambda$', min: -2, max: 2, step: 0.02, init: 0 }, { k: 'mu', label: '$\\log_{10}\\mu$ of a direction', min: -2, max: 2, step: 0.02, init: -0.5 }],
  plots: v => { const l = 10 ** v.lg; return [{ x: [-2, 2], y: [0, 1], xl: 'log₁₀ μ', yl: 'kept fraction', label: 'ridge shrinkage factor', marks: [{ tone: 1, data: curve(m => 10 ** m / (10 ** m + l), -2, 2) }, { as: 'rule', dash: true, x: v.lg }, { drag: { x: 'mu', y: 10 ** v.mu / (10 ** v.mu + l) } }] }]; },
  stats: v => [`kept ${pct(10 ** v.mu / (10 ** v.mu + 10 ** v.lg), 0)}`, `half is kept where μ = λ`],
  takeaway: 'OLS amplifies noise by $1/\\mu$ in weak directions, and ridge replaces this with $\\mu/(\\mu^2+\\lambda\\mu)$, which stays bounded.',
});

export const PenaltyScale = lab({
  intro: 'Penalties are unfair to features in small units. Shrink a feature’s scale and its contribution is shrunk harder, for no real reason.',
  tex: R`\text{kept}=\frac{s^2}{s^2+\lambda}`,
  params: [{ k: 'lg', label: '$\\log_{10}$ feature scale $s$', min: -2, max: 2, step: 0.02, init: 0 }, { k: 'l', label: '$\\lambda$', min: 0.01, max: 10, step: 0.01, init: 1 }],
  plots: v => [{ x: [-2, 2], y: [0, 1], xl: 'log₁₀ s', yl: 'fraction of the effect kept', label: 'effect of feature scale under ridge', marks: [{ tone: 1, data: curve(g => 10 ** (2 * g) / (10 ** (2 * g) + v.l), -2, 2) }, { drag: { x: 'lg', y: 10 ** (2 * v.lg) / (10 ** (2 * v.lg) + v.l) } }] }],
  stats: v => [`kept ${pct(10 ** (2 * v.lg) / (10 ** (2 * v.lg) + v.l), 0)}`],
  takeaway: 'Standardize features before penalizing, and leave the intercept unpenalized.',
});

export const RegularizationCheck = truths({
  items: [
    ['Ridge usually sets some coefficients exactly to zero.', false, 'It only shrinks. Exact zeros need the corners of the lasso.'],
    ['Lasso is safest when features are highly correlated.', false, 'It picks among correlated features unstably. Elastic net handles them better.'],
    ['Regularization increases bias.', true, 'It deliberately pulls estimates towards zero, in exchange for lower variance.'],
    ['The intercept should normally be penalized.', false, 'Penalizing it makes the fit depend on where $y$ is centred.'],
    ['Larger $\\lambda$ always lowers test error.', false, 'Past the optimum, the model underfits and error rises.'],
  ],
});

/* ---------- evaluation ---------- */

export const ConfusionCounts = lab({
  intro: 'Every metric is a ratio of the same four counts, with a different denominator. Move the counts and see which metrics react.',
  params: [{ k: 'tp', label: 'true positives', min: 0, max: 100, step: 1, init: 30 }, { k: 'fn', label: 'false negatives', min: 0, max: 100, step: 1, init: 10 }, { k: 'fp', label: 'false positives', min: 0, max: 100, step: 1, init: 20 }, { k: 'tn', label: 'true negatives', min: 0, max: 400, step: 1, init: 140 }],
  plots: v => { const bars = [v.tp, v.fn, v.fp, v.tn]; return [{ x: [-0.6, 3.6], y: [0, Math.max(10, ...bars) * 1.1], yl: 'count', label: 'confusion counts', marks: bars.map((b, i) => ({ as: 'rects', tone: [3, 4, 2, 'muted'][i], data: [[i - 0.35, 0, i + 0.35, b]] })).concat(['TP', 'FN', 'FP', 'TN'].map((t, i) => ({ as: 'label', x: i, y: Math.max(10, ...bars) * 1.02, text: t, anchor: 'middle' }))) }]; },
  table: ({ tp, fn, fp, tn }) => { const f = (a, b) => (b ? fmt(a / b, 3) : 'undefined'); return { head: ['Metric', 'Value'], rows: [['Accuracy', f(tp + tn, tp + fn + fp + tn)], ['Precision', f(tp, tp + fp)], ['Recall', f(tp, tp + fn)], ['Specificity', f(tn, tn + fp)], ['FPR', f(fp, fp + tn)], ['F1', f(2 * tp, 2 * tp + fp + fn)]] }; },
});

export const MetricsDrill = drill({
  intro: 'Pick the correct denominator first.',
  gen: r => ({ tp: int(r, 5, 60), fn: int(r, 1, 30), fp: int(r, 1, 30), tn: int(r, 20, 150), k: pick(r, ['precision', 'recall', 'FPR', 'accuracy', 'F1']) }),
  q: v => `TP = ${v.tp}, FN = ${v.fn}, FP = ${v.fp}, TN = ${v.tn}. Find the **${v.k}**.`,
  answer: v => ({ precision: v.tp / (v.tp + v.fp), recall: v.tp / (v.tp + v.fn), FPR: v.fp / (v.fp + v.tn), accuracy: (v.tp + v.tn) / (v.tp + v.fn + v.fp + v.tn), F1: (2 * v.tp) / (2 * v.tp + v.fp + v.fn) }[v.k]), dp: 3,
  steps: v => [{ precision: R`Precision is $TP/(TP+FP)$: of those flagged, how many were real.`, recall: R`Recall is $TP/(TP+FN)$: of the real ones, how many were caught.`, FPR: R`FPR is $FP/(FP+TN)$: of the negatives, how many were flagged.`, accuracy: R`Accuracy is $(TP+TN)/N$.`, F1: R`$F_1=2TP/(2TP+FP+FN)$.` }[v.k]],
});

export const BaseRateDrill = drill({
  intro: 'Precision depends on how common the positive class is.',
  tex: R`\text{precision}=\frac{\text{TPR}\,\pi}{\text{TPR}\,\pi+\text{FPR}\,(1-\pi)}`,
  gen: r => ({ tpr: pick(r, [0.8, 0.9, 0.95, 0.99]), fpr: pick(r, [0.01, 0.05, 0.1, 0.2]), pi: pick(r, [0.01, 0.05, 0.1, 0.3, 0.5]) }),
  q: v => `TPR = ${v.tpr}, FPR = ${v.fpr}, prevalence = ${v.pi}. Find the precision.`,
  answer: v => (v.tpr * v.pi) / (v.tpr * v.pi + v.fpr * (1 - v.pi)), dp: 3,
  steps: v => [`True positives per case: $${v.tpr}\\times${v.pi}=${fmt(v.tpr * v.pi, 4)}$`, `False positives per case: $${v.fpr}\\times${fmt(1 - v.pi)}=${fmt(v.fpr * (1 - v.pi), 4)}$`],
});

export const CostDrill = drill({
  intro: 'Compare the two expected costs. Flag when acting is cheaper.',
  tex: R`t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}`,
  gen: r => ({ fp: int(r, 1, 10), fn: int(r, 1, 30) }),
  q: v => `A false alarm costs ${v.fp}, and a miss costs ${v.fn}. What is the best probability threshold?`,
  answer: v => v.fp / (v.fp + v.fn), dp: 3,
  steps: v => [`Flag if $p\\cdot${v.fn}>(1-p)\\cdot${v.fp}$`, `Solve for $p$: $${v.fp}/(${v.fp}+${v.fn})$`],
});

export const AucPairsDrill = drill({
  intro: 'AUC is the share of positive-negative pairs where the positive scores higher, with ties counted as half.',
  gen: r => ({ pos: ints(r, 3, 1, 9), neg: ints(r, 3, 1, 9) }),
  q: v => `Positive scores: $${v.pos.join(',\\ ')}$. Negative scores: $${v.neg.join(',\\ ')}$. Find the AUC.`,
  answer: v => sum(v.pos.flatMap(p => v.neg.map(n => (p > n ? 1 : p === n ? 0.5 : 0)))) / (v.pos.length * v.neg.length), dp: 3,
  steps: v => [`There are ${v.pos.length * v.neg.length} pairs.`, `Winning pairs: ${sum(v.pos.flatMap(p => v.neg.map(n => (p > n ? 1 : p === n ? 0.5 : 0))))}`],
});

export const ClassWeights = lab({
  intro: 'Weighting positives in training moves the boundary the same way as lowering the threshold. Compare the two.',
  tex: R`\text{weight }k\text{ on positives}\iff\text{threshold }\frac{1}{1+k}`,
  params: [{ k: 'lk', label: '$\\log_2$ of the weight $k$', min: -3, max: 5, step: 0.05, init: 2 }, { k: 'pi', label: 'prevalence', min: 0.05, max: 0.8, step: 0.01, init: 0.3 }],
  derive: ({ lk, pi }) => { const k = 2 ** lk, post = x => (pi * normalPdf(x, 1.5)) / (pi * normalPdf(x, 1.5) + (1 - pi) * normalPdf(x, 0)), wt = x => (k * pi * normalPdf(x, 1.5)) / (k * pi * normalPdf(x, 1.5) + (1 - pi) * normalPdf(x, 0)), xs = linspace(-4, 6, 2001), cross = g => xs.find(x => g(x) >= 0.5) ?? 6, t = 1 / (1 + k); return { k, post, wt, t, a: cross(wt), b: xs.find(x => post(x) >= t) ?? 6 }; },
  plots: v => [{ x: [-4, 6], y: [0, 1], xl: 'score x', yl: 'probability', label: 'weighted and unweighted posteriors', marks: [{ tone: 1, data: curve(v.post, -4, 6) }, { tone: 2, data: curve(v.wt, -4, 6) }, { as: 'rule', tone: 2, x: v.a }, { as: 'rule', dash: true, y: v.t }, { as: 'rule', dash: true, x: v.b }] }],
  stats: v => [`weighted boundary at ${fmt(v.a)}`, `threshold ${fmt(v.t, 3)} on the plain curve gives ${fmt(v.b)}`],
  legend: [['--c1', 'plain posterior'], ['--c2', 'weighted posterior']],
  takeaway: 'Weighting distorts the probabilities. If you need them calibrated, keep the plain model and move the threshold.',
});

export const EvaluationCheck = truths({
  items: [
    ['AUC changes if you square all the scores.', false, 'Squaring positive scores keeps their order, and AUC depends only on order.'],
    ['Precision can fall when prevalence falls.', true, 'False positives grow relative to true positives.'],
    ['A well-calibrated model has a high AUC.', false, 'Always predicting the base rate is calibrated but ranks nothing.'],
    ['Lowering the threshold never reduces recall.', true, 'More cases are flagged, so no true positive can be lost.'],
    ['The best threshold is 0.5 when the probabilities are calibrated.', false, 'Only when the two errors cost the same.'],
  ],
});
