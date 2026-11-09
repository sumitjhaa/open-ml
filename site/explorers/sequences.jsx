'use client';
// Explorers for next-token prediction, time series, recurrent networks and reinforcement learning.
import { R, curve, drill, heatBox, heatPlot, int, lab, pick, truths } from '../components/labs';
import { gauss, rng } from '../lib/math.mjs';
import { fmt, linspace, memo, pct } from '../lib/models.mjs';
import { BASELINES, GRID, MOVES, NAMES, VOCAB, acf, arFit, arForecast, difference, discounted, entropyNats, forecastScores, gridStep, isTerminal, monthly, mountainCar, ngram, qLearn, qUpdate, rnnRun, softmaxT, startOf, valueIteration } from '../lib/extended.mjs';

/* ---------- next-token prediction ---------- */

const TRAIN = NAMES.filter((_, i) => i % 5), HELD = NAMES.filter((_, i) => !(i % 5));
const WORDS = ['abid', 'priya', 'rahul', 'neha', 'kabir'];
export const PairsFromName = lab({
  intro: 'A name becomes (context, next) pairs. Padding marks the start, and the final target marks the end.',
  params: [{ k: 'w', label: 'Name', options: WORDS.map(w => [w, w]), init: 'abid' }, { k: 'c', label: 'context length', min: 1, max: 5, step: 1, init: 3 }],
  derive: ({ w, c }) => { const s = '.'.repeat(c) + w + '.'; return { rows: Array.from({ length: w.length + 1 }, (_, i) => [s.slice(i, i + c), s[i + c]]) }; },
  table: v => ({ head: ['context', 'next'], rows: v.rows.map(r => [r[0], r[1]]) }),
  stats: v => [`${v.rows.length} pairs from a name of length ${v.w.length}`],
  takeaway: 'Every position is one classification example with 27 possible answers.',
});
export const PairCountDrill = drill({
  intro: 'A name of length $L$ gives $L+1$ pairs, because the end of the name is also predicted.',
  gen: r => ({ w: pick(r, NAMES) }),
  q: v => `How many (context, next) pairs does “${v.w}” give?`,
  answer: v => v.w.length + 1, dp: 0,
  steps: v => [`Its length is $${v.w.length}$.`],
});
const models = memo(order => ngram(TRAIN, order));
const bigram = models(1);
const LETTERS = [...VOCAB], BIGRAMS = LETTERS.map(c => bigram.probs(c).map(p => Math.sqrt(p)));
const LET = 'abcdefghijklmnopqrstuvwxyz'.split('');
export const BigramTable = lab({
  intro: 'Counting letter pairs gives a first model. Each row of the matrix is a previous letter, each column a next letter, and stronger cells are likelier pairs. The outlined row is drawn as bars below.',
  params: [{ k: 'p', label: 'previous letter', min: 0, max: 26, step: 1, init: 1, show: v => VOCAB[v] }],
  derive: ({ p }) => { const P = bigram.probs(VOCAB[p]), top = P.map((v, i) => [v, VOCAB[i]]).sort((a, b) => b[0] - a[0]).slice(0, 5); return { P, top }; },
  plots: v => [heatPlot(BIGRAMS, { w: 600, h: 560, label: 'letter pair probabilities', over: [heatBox(BIGRAMS, v.p, null)], heat: { rows: LETTERS, cols: LETTERS, name: (i, j, x) => `${VOCAB[i]} then ${VOCAB[j]}: ${(x * x).toFixed(3)}` } }), { x: [-0.5, 26.5], y: [0, Math.max(...v.P) * 1.15], w: 620, h: 230, xl: '. a b c … z', label: 'next character distribution', marks: [{ as: 'rects', tone: 1, data: v.P.map((q, i) => [i - 0.4, 0, i + 0.4, q]) }] }],
  stats: v => [`most likely next: ${v.top.map(t => `${t[1]} ${t[0].toFixed(2)}`).join(', ')}`, `entropy ${fmt(entropyNats(v.P), 2)} nats`],
  takeaway: 'Counting is cheap, but the table grows exponentially with the context length.',
});
export const CrossEntropyDrill = drill({
  intro: 'Each pair costs the negative log of the probability given to the true next character.',
  tex: R`\text{loss}=-\ln p_{\text{target}}`,
  gen: r => ({ p: pick(r, [0.5, 0.25, 0.1, 0.05, 0.75, 0.02]) }),
  q: v => `The model gives the true character probability $${v.p}$. Find the loss in nats.`,
  answer: v => -Math.log(v.p), dp: 3,
  steps: () => ['Take the natural logarithm, then flip the sign.'],
});
export const EmbedWidthDrill = drill({
  intro: 'Concatenating keeps each slot separate, so the widths multiply.',
  gen: r => ({ b: int(r, 2, 8), k: pick(r, [2, 4, 8, 10]), h: pick(r, [32, 64, 100]) }),
  q: v => `Context $${v.b}$, embedding size $${v.k}$, hidden layer $${v.h}$. How many weights connect the input to the hidden layer, not counting biases?`,
  answer: v => v.b * v.k * v.h, dp: 0,
  steps: v => [`Input width: $${v.b}\\times${v.k}=${v.b * v.k}$`],
});
const orderScores = memo(() => [1, 2, 3, 4].map(o => { const m = models(o); return { o, tr: m.nll(TRAIN), he: m.nll(HELD) }; }));
export const ContextLength = lab({
  intro: 'A longer context always fits the training names better. Held-out names tell the real story.',
  params: [{ k: 'o', label: 'context length', min: 1, max: 4, step: 1, init: 2 }],
  derive: () => ({ S: orderScores() }),
  plots: v => [{ x: [1, 4], y: [0, 3.6], w: 420, h: 280, xl: 'context length', yl: 'nats', label: 'loss against context length', marks: [{ tone: 2, data: v.S.map(s => [s.o, s.tr]) }, { tone: 4, data: v.S.map(s => [s.o, s.he]) }, { as: 'rule', dash: true, tone: 'muted', y: Math.log(27) }, { as: 'rule', dash: true, tone: 1, x: v.o }] }],
  stats: v => [`training ${fmt(v.S[v.o - 1].tr, 2)} nats`, `held-out ${fmt(v.S[v.o - 1].he, 2)} nats`, `uniform guess ${fmt(Math.log(27), 2)} nats`],
  legend: [['--c2', 'training'], ['--c4', 'held-out']],
  takeaway: 'Counts cannot generalise between similar contexts. Embeddings and a network can.',
});
const LOGITS = [3.2, 2.4, 2.0, 1.1, 0.4, -0.2, -1];
export const TemperatureLab = lab({
  intro: 'Temperature divides the logits. Low values play safe, and high values take risks.',
  tex: R`p_k(T)\propto e^{z_k/T}`,
  params: [{ k: 'T', label: 'temperature $T$', min: 0.1, max: 5, step: 0.05, init: 1 }],
  derive: ({ T }) => ({ p: softmaxT(LOGITS, T) }),
  plots: v => { const Ts = [0.1, 0.25, 0.5, 1, 2, 5], M = Ts.map(T => softmaxT(LOGITS, T)); return [{ x: [-0.5, 6.5], y: [0, 1], w: 420, h: 260, xl: 'characters, ranked', label: 'probabilities at this temperature', marks: [{ as: 'rects', tone: 1, data: v.p.map((q, i) => [i - 0.35, 0, i + 0.35, q]) }] }, heatPlot(M, { w: 460, h: 270, caption: 'Temperatures compared, ranked characters left to right', label: 'probability by temperature and rank', heat: { max: 1, rows: Ts.map(T => `T ${T}`), cols: LOGITS.map((_, i) => `#${i + 1}`), show: x => x.toFixed(2).replace(/^0/, ''), name: (i, j, x) => `temperature ${Ts[i]}, rank ${j + 1}: ${x.toFixed(3)}` } })]; },
  stats: v => [`entropy ${fmt(entropyNats(v.p), 3)} nats`, `top probability ${fmt(v.p[0], 3)}`],
  takeaway: 'The ranking never changes, only the gaps between the characters.',
});
export const TemperatureDrill = drill({
  intro: 'At temperature $T$, probabilities become proportional to $p^{1/T}$.',
  gen: r => ({ a: pick(r, [0.64, 0.81, 0.36, 0.49]), T: 2 }),
  q: v => `Two characters have probabilities $${v.a}$ and $${fmt(1 - v.a, 2)}$. Find the first at $T=2$.`,
  answer: v => Math.sqrt(v.a) / (Math.sqrt(v.a) + Math.sqrt(1 - v.a)), dp: 3,
  steps: v => [`Square roots: $${fmt(Math.sqrt(v.a), 3)}$ and $${fmt(Math.sqrt(1 - v.a), 3)}$`],
});
export const NameSampler = lab({
  intro: 'Sample a character, append it, slide the window, and repeat. Names generated from the counts of real names.',
  params: [{ k: 'o', label: 'context length', min: 1, max: 4, step: 1, init: 2 }, { k: 'T', label: 'temperature', min: 0.3, max: 3, step: 0.1, init: 1 }, { k: 'seed', seed: 'New names', init: 1 }],
  derive: ({ o, T, seed }) => { const m = models(o), r = rng(seed * 97), names = Array.from({ length: 14 }, () => m.sample(r, T)); return { names, copied: names.filter(n => NAMES.includes(n)).length }; },
  stats: v => [v.names.join(', '), `${v.copied} of ${v.names.length} are copies of real names`],
  plots: () => [],
  takeaway: 'Low temperature and long context copy the data. High temperature makes noise.',
});

/* ---------- time series ---------- */

const Y = monthly(), TR = Y.slice(0, 96), TE = Y.slice(96);
const B_NAMES = { mean: 'Mean', naive: 'Naive', seasonal: 'Seasonal naive', drift: 'Drift' };
export const BaselineForecast = lab({
  intro: 'Four forecasts that need no training. Every model you build must beat these.',
  params: [{ k: 'b', label: 'Baseline', options: Object.entries(B_NAMES), init: 'seasonal' }],
  derive: ({ b }) => { const f = BASELINES[b](TR, 24), all = Object.keys(B_NAMES).map(k => [B_NAMES[k], forecastScores(TE, BASELINES[k](TR, 24))]); return { f, all, s: forecastScores(TE, f) }; },
  plots: v => [{ x: [48, 120], y: [60, 160], w: 620, h: 260, xl: 'month', label: 'series and forecast', marks: [{ tone: 'muted', data: Y.slice(48, 96).map((y, i) => [48 + i, y]) }, { tone: 1, data: TE.map((y, i) => [96 + i, y]) }, { tone: 4, data: v.f.map((y, i) => [96 + i, y]) }, { as: 'rule', dash: true, tone: 'muted', x: 96 }] }],
  stats: v => [`MAE ${fmt(v.s.mae, 2)}`, `RMSE ${fmt(v.s.rmse, 2)}`, `MAPE ${fmt(v.s.mape, 2)}%`],
  table: v => ({ head: ['baseline', 'MAE', 'RMSE'], rows: v.all.map(([n, s]) => [n, fmt(s.mae, 2), fmt(s.rmse, 2)]) }),
  takeaway: 'Seasonal naive keeps the shape but misses the rise. Drift follows the rise but misses the shape.',
});
export const NaiveForecastDrill = drill({
  intro: 'Forecast by rule, from the last values of the series.',
  gen: r => ({ y: Array.from({ length: 5 }, () => int(r, 10, 30)), k: pick(r, ['naive', 'mean', 'drift']) }),
  q: v => `Series $${v.y.join(',\\ ')}$. Give the ${v.k === 'naive' ? 'naive' : v.k === 'mean' ? 'mean' : 'drift'} forecast one step ahead.`,
  answer: v => BASELINES[v.k](v.y, 1)[0], dp: 2,
  steps: v => [v.k === 'naive' ? 'Repeat the last value.' : v.k === 'mean' ? 'Average all the values.' : `Slope: $(${v.y[4]}-${v.y[0]})/4$. Add it to the last value.`],
});
export const ForecastErrorDrill = drill({
  intro: 'MAE averages the absolute errors. RMSE squares first, so it punishes big misses more.',
  gen: r => ({ e: Array.from({ length: 4 }, () => int(r, -6, 6)), k: pick(r, ['MAE', 'RMSE']) }),
  q: v => `Errors $${v.e.join(',\\ ')}$. Find the ${v.k}.`,
  answer: v => (v.k === 'MAE' ? v.e.reduce((s, x) => s + Math.abs(x), 0) / 4 : Math.sqrt(v.e.reduce((s, x) => s + x * x, 0) / 4)), dp: 3,
  steps: v => [v.k === 'MAE' ? 'Add the absolute values and divide by four.' : `Sum of squares: $${v.e.reduce((s, x) => s + x * x, 0)}$`],
});
export const TimeSplit = lab({
  intro: 'Each row trains on everything before an origin and tests the step just after it. Shuffled folds would train on the future.',
  params: [{ k: 'h', label: 'forecast horizon', min: 1, max: 5, step: 1, init: 1 }],
  plots: v => [{ x: [0, 24], y: [0, 8], w: 620, h: 240, xl: 'time', label: 'rolling origin splits', marks: Array.from({ length: 7 }, (_, f) => { const end = 8 + f * 2, y = 7 - f; return [{ as: 'rects', tone: 1, data: Array.from({ length: end }, (_, i) => [i + 0.08, y + 0.15, i + 0.92, y + 0.85]) }, { as: 'rects', tone: 4, data: [[end + v.h - 1 + 0.08, y + 0.15, end + v.h - 1 + 0.92, y + 0.85]] }]; }).flat() }],
  legend: [['--c1', 'training'], ['--c4', 'scored']],
  stats: () => ['seven training origins'],
  takeaway: 'Always validate forward in time. The average over origins estimates how well you forecast new data.',
});
const WALK = (() => { const r = rng(13), y = [50]; for (let i = 0; i < 199; i++) y.push(+(y.at(-1) + gauss(r)).toFixed(4)); return y; })();
const SERIES = { walk: ['random walk', WALK], noise: ['white noise', (() => { const r = rng(7); return Array.from({ length: 200 }, () => +gauss(r).toFixed(4)); })()], month: ['monthly, with trend and season', Y] };
export const AcfLab = lab({
  intro: 'Autocorrelation measures how much a series resembles itself shifted. Difference it and see the memory disappear.',
  tex: R`r_k=\frac{\sum_t(y_t-\bar y)(y_{t+k}-\bar y)}{\sum_t(y_t-\bar y)^2}`,
  params: [{ k: 's', label: 'Series', options: Object.keys(SERIES).map(k => [k, SERIES[k][0]]), init: 'walk' }, { k: 'd', label: 'differences', min: 0, max: 1, step: 1, init: 0 }],
  derive: ({ s, d }) => { const y = d ? difference(SERIES[s][1]) : SERIES[s][1]; return { y, A: Array.from({ length: 24 }, (_, k) => acf(y, k + 1)), band: 1.96 / Math.sqrt(y.length) }; },
  plots: v => [{ caption: 'Series', x: [0, v.y.length], y: [Math.min(...v.y) - 1, Math.max(...v.y) + 1], w: 310, h: 220, label: 'the series', marks: [{ tone: 1, data: v.y.map((a, i) => [i, a]) }] }, { caption: 'Autocorrelation', x: [0, 25], y: [-1, 1], w: 310, h: 220, xl: 'lag', label: 'autocorrelation', marks: [{ as: 'stems', tone: 2, data: v.A.map((a, i) => [i + 1, 0, a]) }, { as: 'rule', dash: true, tone: 'muted', y: v.band }, { as: 'rule', dash: true, tone: 'muted', y: -v.band }] }],
  stats: v => [`$r_1=${fmt(v.A[0], 3)}$`, `${v.A.filter(a => Math.abs(a) > v.band).length} of 24 lags outside the band`],
  takeaway: 'A random walk looks predictable, but its differences are noise.',
});
export const DifferencingLab = lab({
  intro: 'One difference removes a trend. A seasonal difference removes a repeating pattern.',
  params: [{ k: 'm', label: 'Difference', options: [[0, 'none'], [1, 'first'], [12, 'seasonal']], init: 0 }],
  derive: ({ m }) => ({ y: +m ? difference(Y, +m) : Y }),
  plots: v => [{ x: [0, 120], y: [Math.min(...v.y) - 2, Math.max(...v.y) + 2], w: 620, h: 260, xl: 'month', label: 'differenced series', marks: [{ tone: 1, data: v.y.map((a, i) => [i, a]) }] }],
  stats: v => [`mean ${fmt(v.y.reduce((a, b) => a + b, 0) / v.y.length, 2)}`, `lag twelve autocorrelation ${fmt(acf(v.y, 12), 3)}`],
  takeaway: 'A seasonal difference leaves about twelve times the monthly slope.',
});
export const ArPredictDrill = drill({
  intro: 'Autoregression is a weighted sum of recent values, plus a constant.',
  tex: R`\hat y_t=c+\sum_k\varphi_ky_{t-k}`,
  gen: r => ({ c: int(r, 0, 3), a: pick(r, [0.5, 0.6, 0.3, 0.2]), b: pick(r, [0.1, 0.2, 0.3]), y: int(r, 5, 12), z: int(r, 5, 12) }),
  q: v => `$c=${v.c}$, $\\varphi_1=${v.a}$, $\\varphi_2=${v.b}$, with last value $${v.y}$ and the one before it $${v.z}$. Forecast the next value.`,
  answer: v => v.c + v.a * v.y + v.b * v.z, dp: 2,
  steps: v => [`$${v.a}\\times${v.y}=${fmt(v.a * v.y, 2)}$`, `$${v.b}\\times${v.z}=${fmt(v.b * v.z, 2)}$`],
});
const fits = memo(p => { const c = arFit(TR, p); return { c, f: arForecast(TR, c, 24) }; });
export const ArFitLab = lab({
  intro: 'Fit by least squares on the lags, then forecast by feeding each prediction back in as a lag.',
  params: [{ k: 'p', label: 'lags $p$', min: 1, max: 24, step: 1, init: 13 }],
  derive: ({ p }) => { const { c, f } = fits(p); return { c, f, s: forecastScores(TE, f) }; },
  plots: v => [{ x: [48, 120], y: [60, 170], w: 620, h: 260, xl: 'month', label: 'autoregression forecast', marks: [{ tone: 'muted', data: Y.slice(48, 96).map((y, i) => [48 + i, y]) }, { tone: 1, data: TE.map((y, i) => [96 + i, y]) }, { tone: 4, data: v.f.map((y, i) => [96 + i, y]) }, { as: 'rule', dash: true, tone: 'muted', x: 96 }] }],
  stats: v => [`MAE ${fmt(v.s.mae, 2)}`, `compare the baselines: 9 to 11`],
  takeaway: 'With enough lags to see a whole season, autoregression beats every simple baseline here.',
});
export const RnnMemory = lab({
  intro: 'One input pulse, then zeros. The hidden state carries a memory of it, and $W$ sets how long that lasts.',
  tex: R`h_t=\tanh(Wh_{t-1}+Ux_t)`,
  params: [{ k: 'W', label: '$W$', min: 0, max: 1.5, step: 0.05, init: 0.8 }, { k: 'U', label: '$U$', min: 0.2, max: 2, step: 0.1, init: 1 }],
  derive: ({ W, U }) => ({ h: rnnRun(W, U, [1, ...Array(24).fill(0)]) }),
  plots: v => [{ x: [0, 25], y: [0, 1.05], w: 420, h: 260, xl: 'step', yl: 'hidden state', label: 'hidden state over time', marks: [{ tone: 1, data: v.h.map((a, i) => [i + 1, a]) }, { as: 'points', tone: 4, r: 3, data: v.h.map((a, i) => [i + 1, a]) }] }],
  stats: v => [`after five steps ${fmt(v.h[4], 3)}`, `after twenty steps ${fmt(v.h[19], 3)}`],
  takeaway: 'Below one the memory fades. Above one it can lock on.',
});
export const RnnGradient = lab({
  intro: 'Backpropagation through time multiplies one factor per step. Small changes in the factor have huge effects.',
  params: [{ k: 'rho', label: 'spectral radius $\\rho$', min: 0.5, max: 1.5, step: 0.01, init: 0.9 }, { k: 'k', label: 'steps back', min: 1, max: 60, step: 1, init: 40 }],
  plots: v => [{ x: [1, 60], y: [-12, 12], w: 420, h: 280, xl: 'steps back', yl: 'log₁₀ gradient size', label: 'gradient by steps back', marks: [{ tone: 1, data: curve(k => k * Math.log10(v.rho), 1, 60, 60) }, { as: 'rule', dash: true, tone: 'muted', y: 0 }, { as: 'points', tone: 4, r: 6, data: [[v.k, v.k * Math.log10(v.rho)]] }] }],
  stats: v => [`gradient scale ${(v.rho ** v.k).toExponential(2)}`],
  status: v => (v.rho ** v.k < 1e-3 ? ['alert', 'Vanishing: early steps stop influencing learning.'] : v.rho ** v.k > 1e3 ? ['alert', 'Exploding: clip the gradients.'] : ['ok', 'Usable.']),
  takeaway: 'LSTMs and GRUs add gated memory, so this product does not rule every path.',
});
export const SequenceCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Random folds are fine for time series.', false, 'They train on the future.'],
    ['A model should always be compared with a naive baseline.', true, 'Otherwise a score has no reference.'],
    ['A random walk is stationary.', false, 'Its first difference is.'],
    ['Temperature changes the ranking of characters.', false, 'It only changes the gaps.'],
    ['Autoregression is linear regression on lags.', true, 'Each row holds recent values, and the target is the next one.'],
    ['An embedding table is trained like any other weights.', true, 'Backpropagation updates the rows that were used.'],
    ['A language model predicts the next token.', true, 'The same recipe at a much larger scale.'],
  ],
  takeaway: 'Forecasts and language models both predict what comes next from what came before.',
});

/* ---------- reinforcement learning ---------- */

const SY = GRID.length;
export const ReturnDrill = drill({
  intro: 'Each later reward is multiplied by another $\\gamma$.',
  tex: R`G=r_0+\gamma r_1+\gamma^2r_2+\dots`,
  gen: r => ({ rw: [int(r, -1, 2), int(r, -1, 2), int(r, 0, 3)], g: pick(r, [0.5, 0.9, 0.8]) }),
  q: v => `Rewards $${v.rw.join(',\\ ')}$ with $\\gamma=${v.g}$. Find the return.`,
  answer: v => discounted(v.rw, v.g), dp: 3,
  steps: v => [`$${v.rw[0]}+${v.g}\\times${v.rw[1]}+${v.g}^2\\times${v.rw[2]}$`],
});
export const DiscountLab = lab({
  intro: 'How much is a reward worth if it arrives later? The discount sets the agent’s horizon.',
  params: [{ k: 'g', label: 'discount $\\gamma$', min: 0.1, max: 0.999, step: 0.001, init: 0.9 }],
  plots: v => [{ x: [0, 60], y: [0, 1], w: 420, h: 260, xl: 'steps ahead', yl: 'weight', label: 'discount weights', marks: [{ tone: 1, data: curve(k => v.g ** k, 0, 60, 61) }] }],
  stats: v => [`horizon about ${fmt(1 / (1 - v.g), 1)} steps`],
  takeaway: 'A higher $\\gamma$ cares about the far future. The horizon is about $1/(1-\\gamma)$.',
});
export const MountainCarLab = lab({
  intro: 'The engine is too weak to climb straight up. Choose how long to back up first.',
  params: [{ k: 'n', label: 'steps backing up', min: 0, max: 80, step: 1, init: 40 }],
  derive: ({ n }) => { const a = mountainCar(() => 2), b = mountainCar(t => (t <= n ? 0 : 2)); return { a, b }; },
  plots: v => [{ x: [0, 200], y: [-1.2, 0.6], w: 620, h: 280, xl: 'step', yl: 'position', label: 'car position over time', marks: [{ tone: 'muted', data: v.a.path }, { tone: 1, data: v.b.path }, { as: 'rule', dash: true, tone: 3, y: 0.5 }] }],
  stats: v => [v.b.reached ? `reaches the flag at step ${v.b.reached}` : 'never reaches the flag', 'always pushing right never does'],
  takeaway: 'The best action now can be the opposite of the goal, because reward comes later.',
});
export const GreedyDrill = drill({
  intro: 'Exploration picks at random with probability $\\varepsilon$, which can still land on the best action.',
  gen: r => ({ n: pick(r, [2, 3, 4]), e: pick(r, [0.1, 0.2, 0.3]) }),
  q: v => `With $${v.n}$ actions and $\\varepsilon=${v.e}$, what is the probability of taking the greedy action?`,
  answer: v => 1 - v.e + v.e / v.n, dp: 3,
  steps: v => [`Greedy directly: $${fmt(1 - v.e, 2)}$`, `Randomly picked: $${v.e}/${v.n}$`],
});
export const BellmanDrill = drill({
  intro: 'Reward now, plus the discounted best value afterwards.',
  tex: R`Q(s,a)=r+\gamma\max_{a'}Q(s',a')`,
  gen: r => ({ r: int(r, -1, 2), g: pick(r, [0.9, 0.99, 0.5]), q: [int(r, 0, 10), int(r, 0, 10), int(r, 0, 10)] }),
  q: v => `Reward $${v.r}$, $\\gamma=${v.g}$, and next-state values $${v.q.join(',\\ ')}$. Find $Q(s,a)$.`,
  answer: v => v.r + v.g * Math.max(...v.q), dp: 3,
  steps: v => [`The best next value is $${Math.max(...v.q)}$.`],
});
export const QUpdateDrill = drill({
  intro: 'Move the old value part of the way towards the target.',
  tex: R`Q\leftarrow Q+\alpha\,(r+\gamma\max Q'-Q)`,
  gen: (r, n) => (n === 0 ? { q: 20, r: -1, g: 0.99, b: 30, a: 0.2 } : { q: int(r, 0, 20), r: int(r, -2, 3), g: pick(r, [0.9, 0.99]), b: int(r, 0, 30), a: pick(r, [0.1, 0.2, 0.5]) }),
  q: v => `$Q=${v.q}$, $r=${v.r}$, $\\gamma=${v.g}$, $\\max Q'=${v.b}$, $\\alpha=${v.a}$. Find the new $Q$.`,
  answer: v => qUpdate(v.q, v.r, v.g, v.b, v.a), dp: 3,
  steps: v => [`Target: $${fmt(v.r + v.g * v.b, 3)}$`, `TD error: $${fmt(v.r + v.g * v.b - v.q, 3)}$`],
});
const VI = memo(g => valueIteration(GRID, g, 16));
const gridMarks = (V, q) => { const n = GRID.length, m = GRID[0].length, marks = []; GRID.forEach((row, a) => [...row].forEach((c, b) => { const x = b, y = n - a - 1, v = V[a][b]; if (c === '#') marks.push({ as: 'rects', tone: 'fg', data: [[x + 0.05, y + 0.05, x + 0.95, y + 0.95]] }); else if (c === 'G') marks.push({ as: 'rects', tone: 3, data: [[x + 0.05, y + 0.05, x + 0.95, y + 0.95]] }, { as: 'label', x: x + 0.5, y: y + 0.4, text: '+1', anchor: 'middle' }); else if (c === 'P') marks.push({ as: 'rects', tone: 4, data: [[x + 0.05, y + 0.05, x + 0.95, y + 0.95]] }, { as: 'label', x: x + 0.5, y: y + 0.4, text: '−1', anchor: 'middle' }); else { if (v > 0.02) marks.push({ as: 'rects', tone: 1, soft: v < 0.6, data: [[x + 0.05, y + 0.05, x + 0.95, y + 0.95]] }); marks.push({ as: 'label', x: x + 0.5, y: y + 0.4, text: v.toFixed(2), anchor: 'middle' }); } })); void m; void q; return marks; };
const arrows = (V, g) => GRID.flatMap((row, a) => [...row].flatMap((c, b) => { if (isTerminal(c)) return []; const Qv = MOVES.map((_, m) => { const o = gridStep(GRID, [a, b], m); return o.r + (o.done ? 0 : g * V[o.s[0]][o.s[1]]); }), best = Qv.indexOf(Math.max(...Qv)), x = b + 0.5, y = SY - a - 0.5; return Math.max(...Qv) > 0 ? [[x + MOVES[best][1] * 0.1, y - MOVES[best][0] * 0.1, x + MOVES[best][1] * 0.38, y - MOVES[best][0] * 0.38]] : []; }));
export const ValueIterLab = lab({
  intro: 'Apply the Bellman equation to every state. Value spreads outward from the goal, one step per sweep.',
  tex: R`V(s)\leftarrow\max_a[R+\gamma V(s')]`,
  params: [{ k: 'g', label: 'discount $\\gamma$', min: 0.5, max: 0.99, step: 0.01, init: 0.9 }, { k: 's', label: 'sweep', min: 0, max: 16, step: 1, init: 0, play: true }],
  derive: ({ g, s }) => { const F = VI(g), V = F[s], st = startOf(GRID); return { V, vs: V[st[0]][st[1]] }; },
  plots: v => [{ x: [0, 5], y: [0, 5], w: 400, h: 380, label: 'state values on the grid', marks: [...gridMarks(v.V), { as: 'arrow', tone: 2, data: arrows(v.V, v.g) }] }],
  stats: v => [`value of the start state ${fmt(v.vs, 3)}`],
  takeaway: 'When the rules are known, repeating Bellman is enough. Reinforcement learning handles unknown rules.',
});
const qRun = memo((eps, alpha, seed) => qLearn(GRID, { eps, alpha, gamma: 0.9, episodes: 300, seed: seed * 19 + 3 }));
const smooth = (a, w = 15) => a.map((_, i) => { const s = a.slice(Math.max(0, i - w), i + 1); return s.reduce((x, y) => x + y, 0) / s.length; });
export const QLearnLab = lab({
  intro: 'The agent knows nothing of the rules, and learns only from what happens when it acts.',
  tex: R`Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]`,
  params: [{ k: 'eps', label: 'exploration $\\varepsilon$', min: 0, max: 0.6, step: 0.02, init: 0.1 }, { k: 'alpha', label: 'learning rate $\\alpha$', min: 0.05, max: 1, step: 0.05, init: 0.5 }, { k: 'seed', seed: 'New run', init: 1 }],
  derive: ({ eps, alpha, seed }) => { const { Q, returns } = qRun(eps, alpha, seed), V = Q.map(row => row.map(q => Math.max(...q))); return { V, S: smooth(returns), last: returns.slice(-50).reduce((a, b) => a + b, 0) / 50 }; },
  plots: v => [{ caption: 'Learned values', x: [0, 5], y: [0, 5], w: 320, h: 320, label: 'learned state values', marks: gridMarks(v.V) }, { caption: 'Return per episode', x: [0, 300], y: [-1, 1], w: 300, h: 320, xl: 'episode', label: 'return by episode', marks: [{ tone: 1, data: v.S.map((r, i) => [i, r]) }] }],
  stats: v => [`mean return, last 50 episodes ${fmt(v.last, 3)}`],
  status: v => (v.eps === 0 ? ['alert', 'With no exploration and a zero table, it may never find the goal.'] : undefined),
  takeaway: 'Each step moves one entry towards reward plus the discounted best next value.',
});
const EXPL = [0, 0.05, 0.1, 0.2, 0.3, 0.5];
const exploreRuns = memo(() => EXPL.map(e => { let t = 0; for (let s = 1; s <= 8; s++) { const r = qLearn(GRID, { eps: e, alpha: 0.5, gamma: 0.9, episodes: 300, seed: s * 7 }).returns.slice(-100); t += r.reduce((a, b) => a + b, 0) / 100; } return [e, t / 8]; }));
export const ExploreTradeoff = lab({
  intro: 'Too little exploration never discovers the goal. Too much keeps falling in the pit. This averages eight runs per setting.',
  params: [{ k: 'i', label: 'exploration $\\varepsilon$', min: 0, max: 5, step: 1, init: 2, show: v => EXPL[v] }],
  derive: () => ({ R0: exploreRuns() }),
  plots: v => [{ x: [0, 0.5], y: [-0.2, 0.8], w: 420, h: 280, xl: 'ε', yl: 'mean return', label: 'return against exploration', marks: [{ tone: 1, data: v.R0 }, { as: 'points', tone: 4, r: 6, data: [v.R0[v.i]] }] }],
  stats: v => [`mean return ${fmt(v.R0[v.i][1], 3)}`],
  takeaway: 'Exploration has a cost while it runs. Many agents reduce $\\varepsilon$ over time.',
});
const EDGES = linspace(-1.2, 0.6, 25);
export const BinningLab = lab({
  intro: 'A table needs discrete states, so continuous positions are cut into bins.',
  params: [{ k: 'b', label: 'edges per axis', min: 3, max: 40, step: 1, init: 25 }, { k: 'x', label: 'position', min: -1.2, max: 0.6, step: 0.01, init: -0.5 }],
  derive: ({ b, x }) => { const E = linspace(-1.2, 0.6, b), idx = E.filter(e => e <= x).length; return { E, idx, states: (b + 1) * (b + 1) * 3 }; },
  plots: v => [{ x: [-1.2, 0.6], y: [0, 1], w: 620, h: 130, xl: 'position', label: 'bin edges', marks: [...v.E.map(e => ({ as: 'rule', tone: 'muted', x: e })), { as: 'rule', tone: 4, x: v.x }] }],
  stats: v => [`bin ${v.idx}`, `${v.states.toLocaleString('en-IN')} table entries with two axes and three actions`, `the course’s setting uses ${EDGES.length} edges`],
  takeaway: 'Coarse bins merge states that need different actions. Fine bins leave many unvisited.',
});
export const RlCheck = truths({
  intro: 'Decide whether each statement is true, then read why.',
  items: [
    ['Q-learning needs a model of the environment.', false, 'It learns from observed transitions alone.'],
    ['Q-learning is off-policy.', true, 'Its target uses the best next action, even while exploring.'],
    ['With $\\alpha=1$, the new value equals the target.', true, 'The old value is fully replaced.'],
    ['A higher discount makes the agent more short-sighted.', false, 'A higher $\\gamma$ values the future more.'],
    ['A target network keeps the regression target steadier.', true, 'It updates slowly compared with the online network.'],
    ['Replay memory helps by breaking the correlation between consecutive steps.', true, 'Training uses random past transitions.'],
  ],
  takeaway: 'Value the future, explore enough, and update towards reward plus best next value.',
});
