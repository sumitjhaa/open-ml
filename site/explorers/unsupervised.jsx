'use client';
// Explorers for SVD, representation learning and clustering.
import { R, drill, int, lab, mat, pick, truths } from '../components/labs';
import { gauss, mean, rng, sum } from '../lib/math.mjs';
import { agglomerate, clamp, eig2, fmt, kmeans, linspace, meanSilhouette, pct, silhouette, svd } from '../lib/models.mjs';

const tone = a => (a % 4) + 1;
const blobs = (seed, centres, n = 18, s = 0.55) => { const r = rng(seed), out = []; centres.forEach(c => { for (let i = 0; i < n; i++) out.push([+(c[0] + s * gauss(r)).toFixed(3), +(c[1] + s * gauss(r)).toFixed(3)]); }); return out; };
const THREE = blobs(4, [[-2.2, -1.2], [2, -1.5], [0.2, 2.2]]);
const circle = linspace(0, 2 * Math.PI, 121);

/* ---------- k-means ---------- */

export const AssignDrill = drill({
  intro: 'A point joins whichever centre is closest.',
  gen: r => ({ x: int(r, 0, 20), a: int(r, 0, 9), b: int(r, 11, 20) }),
  q: v => `Centres are at $${v.a}$ and $${v.b}$. Which one does the point $${v.x}$ join?`,
  choices: v => [`The centre at ${v.a}`, `The centre at ${v.b}`, 'Exactly tied'],
  answer: v => { const d = Math.abs(v.x - v.a) - Math.abs(v.x - v.b); return d < 0 ? 0 : d > 0 ? 1 : 2; },
  steps: v => [`Distances: $${Math.abs(v.x - v.a)}$ and $${Math.abs(v.x - v.b)}$`],
});

export const CentroidDrill = drill({
  intro: 'A centre moves to the mean of its points, which is what minimises the squared distances.',
  tex: R`c=\frac{1}{|C|}\sum_{x\in C}x`,
  gen: r => { const n = int(r, 3, 5); return { pts: Array.from({ length: n }, () => [int(r, -4, 8), int(r, -4, 8)]), k: int(r, 0, 1) }; },
  q: v => `A cluster holds ${v.pts.map(p => `$(${p[0]},${p[1]})$`).join(', ')}. What is coordinate ${v.k ? 'two' : 'one'} of its new centre?`,
  answer: v => mean(v.pts.map(p => p[v.k])),
  steps: v => [`Add that coordinate: $${v.pts.map(p => p[v.k]).join('+')}=${sum(v.pts.map(p => p[v.k]))}$`, `Divide by ${v.pts.length}.`],
});

export const KMeansSteps = lab({
  intro: 'Step through the two moves. Each **assign** recolours the points, each **update** moves the centres, and the total squared distance can only fall.',
  tex: R`J=\sum_i\|x_i-c_{z_i}\|^2`,
  params: [{ k: 'init', label: 'Starting centres', options: [['random', 'random'], ['plus', 'k-means++']], init: 'random' }, { k: 'seed', seed: 'New start', init: 1 }, { k: 'k', label: 'clusters $k$', min: 2, max: 6, step: 1, init: 3 }, { k: 'step', label: 'step', min: 0, max: 12, step: 1, init: 0, play: true }],
  derive: ({ k, seed, init, step }) => { const frames = kmeans(THREE, k, { seed: seed * 11, init }); return { frames, f: frames[Math.min(step, frames.length - 1)], done: step >= frames.length - 1 }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 360, xl: 'x₁', yl: 'x₂', label: 'k-means assignment and centres', marks: [...THREE.map((x, i) => ({ tone: 'muted', soft: true, data: [x, v.f.centres[v.f.assign[i]]] })), { as: 'points', r: 4, data: THREE.map((x, i) => [x[0], x[1], tone(v.f.assign[i])]) }, { as: 'rings', r: 11, data: v.f.centres }, { as: 'points', tone: 'fg', r: 5, data: v.f.centres }] }],
  stats: v => [`inertia ${fmt(v.f.inertia)}`, `round ${Math.min(v.step, v.frames.length - 1)} of ${v.frames.length - 1}`],
  status: v => (v.done ? ['ok', 'Converged: no point changed cluster.'] : undefined),
  takeaway: 'Each move solves its own small problem exactly, which is why the objective never rises and why it can still stop at a poor answer.',
});

const ELBOW = (() => Array.from({ length: 8 }, (_, i) => { const k = i + 1, best = [1, 2, 3, 4, 5].map(s => kmeans(THREE, k, { seed: s * 13, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a)); return { k, inertia: best.inertia, sil: k > 1 ? meanSilhouette(THREE, best.assign) : 0 }; }))();

export const ChoosingK = lab({
  intro: 'Inertia always falls as $k$ grows, so read its shape rather than its size. Silhouette can go down as well as up.',
  params: [{ k: 'k', label: 'clusters $k$', min: 1, max: 8, step: 1, init: 3 }],
  plots: v => [{ caption: 'Inertia', x: [1, 8], y: [0, ELBOW[0].inertia * 1.05], w: 300, h: 240, xl: 'k', label: 'inertia against k', marks: [{ tone: 1, data: ELBOW.map(e => [e.k, e.inertia]) }, { as: 'rule', dash: true, x: v.k }] }, { caption: 'Average silhouette', x: [1, 8], y: [0, 1], w: 300, h: 240, xl: 'k', label: 'silhouette against k', marks: [{ tone: 2, data: ELBOW.slice(1).map(e => [e.k, e.sil]) }, { as: 'rule', dash: true, x: v.k }] }],
  stats: v => [`inertia ${fmt(ELBOW[v.k - 1].inertia, 1)}`, v.k > 1 ? `silhouette ${fmt(ELBOW[v.k - 1].sil, 3)}` : 'silhouette needs at least two groups'],
  takeaway: 'Three real groups give a clear bend at three, and the silhouette peaks there.',
});

export const SilhouetteDrill = drill({
  intro: '$a$ is the mean distance to your own cluster, and $b$ the mean distance to the nearest other one.',
  tex: R`s=\frac{b-a}{\max(a,b)}`,
  gen: r => ({ a: int(r, 1, 9), b: int(r, 1, 9) }),
  q: v => `$a=${v.a}$ and $b=${v.b}$. Find the silhouette.`,
  answer: v => silhouette(v.a, v.b), dp: 3,
  steps: v => [`$b-a=${v.b - v.a}$`, `$\\max(a,b)=${Math.max(v.a, v.b)}$`],
});

const failures = () => {
  const r = rng(29), g = () => gauss(r), out = { long: [], rings: [], unequal: [], scale: [] };
  for (let i = 0; i < 90; i++) { const y = i % 2; out.long.push([3.2 * g(), (y ? 1.3 : -1.3) + 0.25 * g(), y]); const t = 2 * Math.PI * r(), R0 = y ? 4 : 1.5; out.rings.push([R0 * Math.cos(t) + 0.15 * g(), R0 * Math.sin(t) + 0.15 * g(), y]); out.scale.push([(y ? 1 : -1) * 0.6 + 0.2 * g(), 3.5 * g(), y]); }
  for (let i = 0; i < 90; i++) out.unequal.push([-1.5 + 1.4 * g(), 1.4 * g(), 0]);
  for (let i = 0; i < 12; i++) out.unequal.push([3.6 + 0.25 * g(), 3 + 0.25 * g(), 1]);
  for (const k in out) out[k] = out[k].map(p => [clamp(+p[0].toFixed(3), -5.8, 5.8), clamp(+p[1].toFixed(3), -5.8, 5.8), p[2]]);
  return out;
};
const FAIL = failures();
const standardize = X => [0, 1].map(j => { const c = X.map(p => p[j]), m = mean(c), s = Math.sqrt(mean(c.map(v => (v - m) ** 2))); return { m, s }; });

export const KMeansFailures = lab({
  intro: 'k-means assumes round clusters of similar size. Pick a shape that breaks that, and compare the true groups with what it finds.',
  params: [{ k: 'kind', label: 'Shape', options: [['long', 'elongated'], ['rings', 'rings'], ['unequal', 'unequal sizes'], ['scale', 'unscaled']], init: 'long' }, { k: 'std', label: 'Scaling', options: [['raw', 'raw features'], ['std', 'standardized']], init: 'raw' }],
  derive: ({ kind, std }) => {
    const D = FAIL[kind], st = standardize(D), X = D.map(p => (std === 'std' ? [(p[0] - st[0].m) / st[0].s, (p[1] - st[1].m) / st[1].s] : [p[0], p[1]]));
    const f = [1, 2, 3, 4, 5, 6].map(s => kmeans(X, 2, { seed: s * 7, init: 'plus' }).at(-1)).reduce((a, b) => (b.inertia < a.inertia ? b : a)), agree = mean(D.map((p, i) => +(p[2] === f.assign[i])));
    return { D, a: f.assign, agree: Math.max(agree, 1 - agree) };
  },
  plots: v => [{ caption: 'True groups', x: [-6, 6], y: [-6, 6], w: 300, h: 280, label: 'true groups', marks: [{ as: 'points', r: 3, data: v.D.map(p => [p[0], p[1], p[2] + 1]) }] }, { caption: 'What k-means finds', x: [-6, 6], y: [-6, 6], w: 300, h: 280, label: 'k-means result', marks: [{ as: 'points', r: 3, data: v.D.map((p, i) => [p[0], p[1], v.a[i] + 1]) }] }],
  stats: v => [`agreement with the truth ${pct(v.agree, 0)}`],
  takeaway: 'Standardizing rescues the unscaled case and nothing else. Other shapes need a different method or different features.',
});

const SIX = blobs(8, [[-3.5, -3], [0, -3.2], [3.5, -3], [-3.5, 3], [0, 3.2], [3.5, 3]], 12, 0.4);

export const InitMatters = lab({
  intro: 'Twelve runs on the same data. Lloyd’s algorithm stops at a local optimum, so where it starts decides where it ends.',
  params: [{ k: 'init', label: 'Starting centres', options: [['random', 'random'], ['plus', 'k-means++']], init: 'random' }, { k: 'seed', seed: 'Run again', init: 1 }],
  derive: ({ init, seed }) => { const runs = Array.from({ length: 12 }, (_, i) => kmeans(SIX, 6, { seed: seed * 100 + i, init }).at(-1)), best = runs.reduce((a, b) => (b.inertia < a.inertia ? b : a)); return { runs, best, worst: Math.max(...runs.map(r => r.inertia)) }; },
  plots: v => [{ caption: 'Final inertia of each run', x: [0, 13], y: [0, Math.max(v.worst * 1.1, 10)], w: 300, h: 260, xl: 'run', label: 'final inertia per run', marks: [{ as: 'rects', tone: 1, data: v.runs.map((r, i) => [i + 0.6, 0, i + 1.4, r.inertia]) }] }, { caption: 'Best run', x: [-6, 6], y: [-6, 6], w: 300, h: 260, label: 'best clustering', marks: [{ as: 'points', r: 4, data: SIX.map((x, i) => [x[0], x[1], tone(v.best.assign[i])]) }, { as: 'rings', r: 10, data: v.best.centres }] }],
  stats: v => [`best ${fmt(v.best.inertia, 1)}, worst ${fmt(v.worst, 1)}`],
  takeaway: 'Restart many times and keep the lowest inertia. Spreading the first centres out makes bad starts rarer.',
});

export const PhantomClusters = lab({
  intro: 'k-means returns $k$ groups no matter what. Compare real clusters with **uniform noise**, and notice that the picture alone can fool you.',
  params: [{ k: 'kind', label: 'Data', options: [['real', 'real clusters'], ['noise', 'uniform noise']], init: 'noise' }, { k: 'k', label: 'clusters $k$', min: 2, max: 6, step: 1, init: 3 }, { k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ kind, k, seed }) => { const r = rng(seed * 37), X = kind === 'real' ? THREE : Array.from({ length: 54 }, () => [+(-4.5 + 9 * r()).toFixed(3), +(-4.5 + 9 * r()).toFixed(3)]), f = kmeans(X, k, { seed: seed * 3, init: 'plus' }).at(-1); return { X, f, sil: meanSilhouette(X, f.assign) }; },
  plots: v => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 360, label: 'clusters found in the data', marks: [{ as: 'points', r: 4, data: v.X.map((x, i) => [x[0], x[1], tone(v.f.assign[i])]) }, { as: 'rings', r: 10, data: v.f.centres }] }],
  stats: v => [`silhouette ${fmt(v.sil, 3)}`],
  status: v => (v.sil < 0.5 ? ['alert', 'Weak structure: these groups may be an artefact.'] : ['ok', 'Well-separated groups.']),
  takeaway: 'Colours make any partition look meaningful. Check against a random baseline before naming anything.',
});

const TREE_PTS = blobs(15, [[-3, -2], [3, -2], [0, 3]], 4, 0.7);
function layout(merges, n) {
  const kids = new Map(merges.map(m => [m.id, m])), order = [], root = merges.at(-1).id;
  const visit = id => { if (id < n) order.push(id); else { visit(kids.get(id).a); visit(kids.get(id).b); } };
  visit(root);
  const pos = new Map(order.map((id, i) => [id, { x: i, h: 0 }])), lines = [];
  for (const m of merges) { const A = pos.get(m.a), B = pos.get(m.b); lines.push([[A.x, A.h], [A.x, m.h], [B.x, m.h], [B.x, B.h]]); pos.set(m.id, { x: (A.x + B.x) / 2, h: m.h }); }
  return { order, lines };
}
const cutLabels = (merges, n, cut) => { const p = Array.from({ length: 2 * n }, (_, i) => i), f = x => (p[x] === x ? x : (p[x] = f(p[x]))); merges.forEach(m => { if (m.h <= cut) { p[f(m.a)] = m.id; p[f(m.b)] = m.id; } }); const roots = [...new Set(Array.from({ length: n }, (_, i) => f(i)))]; return Array.from({ length: n }, (_, i) => roots.indexOf(f(i))); };

export const Dendrogram = lab({
  intro: 'Merge the closest groups, again and again. Drag the cut through the tree and read how many clusters fall out. Then change the linkage.',
  params: [{ k: 'link', label: 'Linkage', options: [['single', 'single'], ['complete', 'complete'], ['average', 'average']], init: 'single' }, { k: 'cut', label: 'cut height', min: 0, max: 10, step: 0.05, init: 3 }],
  derive: ({ link, cut }) => { const merges = agglomerate(TREE_PTS, link), { lines } = layout(merges, TREE_PTS.length), labels = cutLabels(merges, TREE_PTS.length, cut); return { merges, lines, labels, top: merges.at(-1).h, count: new Set(labels).size }; },
  plots: v => [{ caption: 'Dendrogram', x: [-0.5, 11.5], y: [0, Math.max(10, v.top) * 1.02], w: 320, h: 280, yl: 'merge height', label: 'dendrogram', marks: [...v.lines.map(l => ({ tone: 1, data: l })), { as: 'rule', tone: 4, y: v.cut }, { drag: { x: 11, y: 'cut', tone: 4 } }] }, { caption: 'Clusters at this cut', x: [-6, 6], y: [-6, 6], w: 300, h: 280, label: 'clusters at the cut', marks: [{ as: 'points', r: 6, data: TREE_PTS.map((x, i) => [x[0], x[1], tone(v.labels[i])]) }] }],
  stats: v => [`${v.count} cluster${v.count === 1 ? '' : 's'}`, `last merge at height ${fmt(v.top)}`],
  takeaway: 'You choose the number of clusters after seeing the tree. A long vertical gap suggests a natural place to cut.',
});

export const LinkageDrill = drill({
  intro: 'The distance between two groups depends on the linkage.',
  gen: r => { const a = [int(r, 0, 3), int(r, 4, 6)], b = [int(r, 8, 11), int(r, 12, 15)]; return { a, b, m: pick(r, ['single', 'complete', 'average']) }; },
  q: v => `Groups $\\{${v.a.join(',')}\\}$ and $\\{${v.b.join(',')}\\}$ on a line. Find their ${v.m}-linkage distance.`,
  answer: v => { const d = v.a.flatMap(x => v.b.map(y => Math.abs(x - y))); return v.m === 'single' ? Math.min(...d) : v.m === 'complete' ? Math.max(...d) : mean(d); },
  steps: v => [`All pair distances: $${v.a.flatMap(x => v.b.map(y => Math.abs(x - y))).join(',\\ ')}$`, { single: 'Single takes the smallest.', complete: 'Complete takes the largest.', average: 'Average takes the mean.' }[v.m]],
});

/* ---------- SVD ---------- */

export const SvdGeometry = lab({
  intro: 'Any matrix sends the unit circle to an ellipse. The blue input directions land on the orange axes, stretched by the singular values.',
  tex: R`A=U\Sigma V^\top,\qquad Av_i=\sigma_iu_i`,
  params: [{ k: 'a', label: '$A_{11}$', min: -3, max: 3, step: 0.1, init: 2 }, { k: 'b', label: '$A_{12}$', min: -3, max: 3, step: 0.1, init: 1 }, { k: 'c', label: '$A_{21}$', min: -3, max: 3, step: 0.1, init: 0.5 }, { k: 'd', label: '$A_{22}$', min: -3, max: 3, step: 0.1, init: 1.5 }],
  derive: ({ a, b, c, d }) => ({ S: svd([[a, b], [c, d]]) }),
  plots: ({ a, b, c, d, S }) => [{ x: [-5, 5], y: [-5, 5], w: 380, h: 380, label: 'unit circle mapped to an ellipse', marks: [{ tone: 'muted', dash: true, data: circle.map(t => [Math.cos(t), Math.sin(t)]) }, { tone: 3, data: circle.map(t => [a * Math.cos(t) + b * Math.sin(t), c * Math.cos(t) + d * Math.sin(t)]) }, { as: 'arrow', tone: 1, data: [0, 1].map(j => [0, 0, S.V[0][j], S.V[1][j]]) }, { as: 'arrow', tone: 2, data: [0, 1].map(j => [0, 0, S.s[j] * S.U[0][j], S.s[j] * S.U[1][j]]) }] }],
  stats: v => [`singular values ${fmt(v.S.s[0])} and ${fmt(v.S.s[1])}`, `rank ${v.S.s.filter(s => s > 1e-6).length}`],
  legend: [['--c1', 'input directions v'], ['--c2', 'stretched output directions σu']],
});

export const SvdDrill = drill({
  intro: 'Singular values are square roots of the eigenvalues of $A^\\top A$.',
  gen: (r, n) => (n === 0 ? { a: 3, c: 4, d: 5 } : { a: int(r, 1, 5), c: int(r, 0, 5), d: int(r, 1, 5) }),
  q: v => `Find the larger singular value of $${mat([[v.a, 0], [v.c, v.d]])}$.`,
  answer: v => Math.sqrt(eig2(v.a * v.a + v.c * v.c, v.c * v.d, v.d * v.d).l1), dp: 3,
  steps: v => [`$A^\\top A=${mat([[v.a * v.a + v.c * v.c, v.c * v.d], [v.c * v.d, v.d * v.d]])}$`, `Its larger eigenvalue is $${fmt(eig2(v.a * v.a + v.c * v.c, v.c * v.d, v.d * v.d).l1, 3)}$. Take the square root.`],
});

export const EnergyDrill = drill({
  intro: 'A rank-$k$ approximation keeps the first $k$ squared singular values.',
  tex: R`\text{kept}=\frac{\sum_{i\le k}\sigma_i^2}{\sum_i\sigma_i^2}`,
  gen: r => { const s = [int(r, 6, 12), int(r, 3, 5), int(r, 1, 2)]; return { s, k: int(r, 1, 2) }; },
  q: v => `Singular values are $${v.s.join(',\\ ')}$. What fraction is kept by rank $${v.k}$?`,
  answer: v => sum(v.s.slice(0, v.k).map(x => x * x)) / sum(v.s.map(x => x * x)), dp: 3,
  steps: v => [`Squares: $${v.s.map(x => x * x).join(',\\ ')}$`, `Kept: $${sum(v.s.slice(0, v.k).map(x => x * x))}$ of $${sum(v.s.map(x => x * x))}$`],
});

const N = 22, field = M => (x, y) => { const i = clamp(Math.floor(N - y - 0.0001), 0, N - 1), j = clamp(Math.floor(x), 0, N - 1); return clamp(M[i][j], -1, 1); };
const IMAGE = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { const u = (j - 10.5) / 11, v = (i - 10.5) / 11, ring = Math.abs(Math.hypot(u, v) - 0.6) < 0.16; return ring ? 0.95 : Math.abs(v + 0.05) < 0.08 ? -0.8 : u > 0.65 ? -0.5 : 0; }));
const IMG_SVD = svd(IMAGE);

export const LowRankImage = lab({
  intro: 'An image is a matrix. Build it from rank-one layers, biggest first, and watch how few are needed for the outline.',
  tex: R`A_k=\sum_{i\le k}\sigma_iu_iv_i^\top`,
  params: [{ k: 'k', label: 'layers $k$', min: 1, max: 22, step: 1, init: 2 }],
  derive: ({ k }) => { const total = sum(IMG_SVD.s.map(s => s * s)); return { M: IMG_SVD.rank(k), kept: sum(IMG_SVD.s.slice(0, k).map(s => s * s)) / total }; },
  plots: v => [{ caption: 'Original', x: [0, N], y: [0, N], w: 280, h: 270, label: 'original matrix', marks: [{ as: 'field', nx: N, ny: N, fn: field(IMAGE) }] }, { caption: `Rank ${v.k}`, x: [0, N], y: [0, N], w: 280, h: 270, label: 'low-rank approximation', marks: [{ as: 'field', nx: N, ny: N, fn: field(v.M) }] }],
  stats: v => [`energy kept ${pct(v.kept)}`, `numbers stored about ${v.k * (2 * N)} of ${N * N}`],
  takeaway: 'Eckart-Young says no other rank-$k$ matrix is closer. That is compression with a guarantee.',
});

export const Denoising = lab({
  intro: 'A rank-two pattern, buried in noise. Compare the error against the noisy matrix with the error against the truth you are really after.',
  params: [{ k: 'sig', label: 'noise level', min: 0.05, max: 0.8, step: 0.01, init: 0.35 }, { k: 'k', label: 'rank kept', min: 1, max: 12, step: 1, init: 2 }, { k: 'seed', seed: 'New noise', init: 1 }],
  derive: ({ sig, seed, k }) => {
    const clean = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => 0.7 * Math.sin((i / N) * 6.3) * Math.cos((j / N) * 3.1) + 0.5 * ((i + j) / N - 1)));
    const r = rng(seed * 59), noisy = clean.map(row => row.map(v => v + sig * gauss(r))), S = svd(noisy), rmse = (A, B) => Math.sqrt(mean(A.flatMap((row, i) => row.map((v, j) => (v - B[i][j]) ** 2))));
    const curve = Array.from({ length: 12 }, (_, q) => { const M = S.rank(q + 1); return { k: q + 1, truth: rmse(M, clean), noisy: rmse(M, noisy) }; }), best = curve.reduce((a, b) => (b.truth < a.truth ? b : a));
    return { noisy, M: S.rank(k), curve, best, cur: curve[k - 1] };
  },
  plots: v => [{ caption: 'Noisy input', x: [0, N], y: [0, N], w: 230, h: 220, label: 'noisy matrix', marks: [{ as: 'field', nx: N, ny: N, fn: field(v.noisy) }] }, { caption: `Rank ${v.k} reconstruction`, x: [0, N], y: [0, N], w: 230, h: 220, label: 'reconstruction', marks: [{ as: 'field', nx: N, ny: N, fn: field(v.M) }] }, { caption: 'Error by rank', x: [1, 12], y: [0, 0.8], w: 230, h: 220, xl: 'rank', label: 'error against rank', marks: [{ tone: 3, data: v.curve.map(c => [c.k, c.noisy]) }, { tone: 4, data: v.curve.map(c => [c.k, c.truth]) }, { as: 'rule', dash: true, x: v.k }] }],
  stats: v => [`error to the truth ${fmt(v.cur.truth, 3)}`, `lowest at rank ${v.best.k}`],
  legend: [['--c3', 'error against the noisy matrix'], ['--c4', 'error against the truth']],
  takeaway: 'The error to the noisy data only falls, so on its own it cannot choose the rank. This is overfitting wearing a linear algebra costume.',
});

export const Embedding = lab({
  intro: 'Three kinds of item, described by twelve noisy features. Two numbers per item recover the groups, because the data really live near a low-dimensional plane.',
  params: [{ k: 'sig', label: 'noise', min: 0.1, max: 3, step: 0.05, init: 0.8 }, { k: 'seed', seed: 'New sample', init: 1 }],
  derive: ({ sig, seed }) => { const r = rng(seed * 83), proto = [0, 1, 2].map(() => Array.from({ length: 12 }, () => 2 * gauss(r))), X = Array.from({ length: 60 }, (_, i) => proto[i % 3].map(v => v + sig * gauss(r))), m = X[0].map((_, j) => mean(X.map(row => row[j]))), Z = X.map(row => row.map((v, j) => v - m[j])), S = svd(Z), tot = sum(S.s.map(s => s * s)); return { pts: Z.map((_, i) => [S.U[i][0] * S.s[0], S.U[i][1] * S.s[1], (i % 3) + 1]), kept: (S.s[0] ** 2 + S.s[1] ** 2) / tot, s: S.s }; },
  plots: v => { const R0 = Math.max(4, ...v.pts.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])])) * 1.05; return [{ x: [-R0, R0], y: [-R0, R0], w: 380, h: 360, xl: 'first coordinate', yl: 'second coordinate', label: 'two-dimensional embedding', marks: [{ as: 'points', r: 4, data: v.pts }] }]; },
  stats: v => [`variance in the embedding ${pct(v.kept, 0)}`],
  takeaway: 'The colours were not used to build the map. Here unsupervised compression reveals the groups, but that is not guaranteed in general.',
});

export const UnsupervisedCheck = truths({
  items: [
    ['k-means always finds the global optimum.', false, 'It stops at a local optimum, which is why restarts matter.'],
    ['Inertia can be used alone to choose $k$.', false, 'It always falls as $k$ grows.'],
    ['Standardising features can change which clusters k-means finds.', true, 'Squared distance adds across features, so units decide which ones dominate.'],
    ['k-means boundaries between clusters are straight.', true, 'The boundary is the perpendicular bisector of the two centroids.'],
    ['If clusters appear, they must exist in the world.', false, 'k-means will partition pure noise.'],
    ['Single linkage tends to form long chains.', true, 'One close pair is enough to join two groups.'],
  ],
});
