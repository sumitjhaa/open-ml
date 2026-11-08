'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Mark, Plot, moved } from '../components/Plot';
import { Button, Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from '../components/ui';
import { gauss, mean, rng } from '../lib/math.mjs';
import { accuracy, adaboost, bagging, buildTree, clamp, dataset, fmt, gbClass, gbRegress, impurity, linspace, pct, predictTree, xgbClass, xgbScore } from '../lib/models.mjs';

const KINDS = { blobs: 'Blobs', overlap: 'Overlap', xor: 'XOR', circles: 'Circles' };
const split2 = (D, f) => [D.map(d => [d[0], d[1]]), D.map(f ?? (d => d[2]))];
const toned = D => D.map(d => [d[0], d[1], d[2] + 1]);
const prob = p => 2 * p - 1;
const dataChoice = (value, onChange) => <Choice label="Dataset" value={value} onChange={onChange} options={Object.entries(KINDS).map(([v, label]) => ({ value: v, label }))} />;

function Rules({ t, names = ['x₁', 'x₂'] }) {
  if (!t.l) return <li>predict {fmt(t.value, 2)} <span>(n = {t.n})</span></li>;
  return <li>{names[t.f]} ≤ {fmt(t.t)}? <span>(gain {fmt(t.gain, 3)})</span><ul><Rules t={t.l} names={names} /><Rules t={t.r} names={names} /></ul></li>;
}

/* ---------- L12a ---------- */

export function TreeGrow() {
  const [kind, setKind] = useState('xor'), [pts, setPts] = useState(() => dataset('xor', 1, 40)), [depth, setDepth] = useState(2), [crit, setCrit] = useState('gini');
  const [X, y] = split2(pts), t = useMemo(() => buildTree(X, y, { maxDepth: depth, criterion: crit }), [pts, depth, crit]); // eslint-disable-line
  const test = useMemo(() => dataset(kind, 77, 200), [kind]), p = x => predictTree(t, x);
  return (
    <>
      <p>Every split is one axis-aligned question. Raise the depth and <b>drag a single point</b> near a boundary, then watch whole regions rearrange. That instability is the reason ensembles exist.</p>
      <Toolbar>{dataChoice(kind, k => { setKind(k); setPts(dataset(k, 1, 40)); })}<Choice label="Criterion" value={crit} onChange={setCrit} options={[{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }, { value: 'error', label: 'Error rate' }]} /></Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} w={400} h={380} xl="x₁" yl="x₂" label="decision regions with draggable points">
          <Mark as="field" fn={(a, b) => prob(p([a, b]))} />
          <Mark as="handles" r={5} data={toned(pts)} onDrag={(i, a, b) => setPts(q => q.map((d, j) => (j === i ? [a, b, d[2]] : d)))} />
        </Plot>
        <Stack>
          <Slider label="max depth" value={depth} min={0} max={8} step={1} onChange={setDepth} />
          <Stat big>train {pct(accuracy(p, pts), 0)} · test {pct(accuracy(p, test), 0)}</Stat>
          <ul className="rules"><Rules t={t} /></ul>
        </Stack>
      </Row>
      <Takeaway>Try Error rate on XOR at depth 2. No first question reduces the error, so the greedy rule stalls while Gini and entropy still see progress. Depth 8 memorises: training reaches 100% while test falls behind.</Takeaway>
    </>
  );
}

export function ImpurityLab() {
  const [crit, setCrit] = useState('entropy'), [pl, setPl] = useState(0.2), [pr, setPr] = useState(0.9), [w, setW] = useState(0.5), g = linspace(0, 1);
  const pp = w * pl + (1 - w) * pr, I = v => impurity(crit, v), gain = I(pp) - (w * I(pl) + (1 - w) * I(pr)), top = crit === 'entropy' ? 1.1 : 0.62;
  return (
    <>
      <p>A split's gain is the <b>vertical gap</b> between the curve at the parent and the chord joining the children. Concave curves make that gap positive for any useful split. The tent-shaped error rate has straight sides, so the gap can vanish.</p>
      <M d t="\text{gain}=I(p)-\big[w\,I(p_L)+(1-w)\,I(p_R)\big],\qquad p=w\,p_L+(1-w)\,p_R" />
      <Toolbar><Choice label="Impurity" value={crit} onChange={setCrit} options={[{ value: 'entropy', label: 'Entropy' }, { value: 'gini', label: 'Gini' }, { value: 'error', label: 'Error rate' }]} /></Toolbar>
      <Row>
        <Plot x={[0, 1]} y={[0, top]} xl="fraction positive p" yl="impurity" label="impurity curve with chord">
          <Mark tone={1} data={g.map(v => [v, I(v)])} />
          <Mark tone={2} data={[[pl, I(pl)], [pr, I(pr)]]} />
          <Mark as="stems" tone={4} data={[[pp, I(pp), w * I(pl) + (1 - w) * I(pr)]]} />
          <Mark as="handles" lockY data={[[pl, I(pl), 2], [pr, I(pr), 2]]} onDrag={(i, x) => (i ? setPr(clamp(x, 0, 1)) : setPl(clamp(x, 0, 1)))} />
        </Plot>
        <Stack>
          <Slider label="share going left" value={w} min={0.05} max={0.95} step={0.01} onChange={setW} show={v => pct(v, 0)} />
          <Stat big>gain {fmt(gain, 4)}</Stat>
          <Stat>parent p = {fmt(pp, 3)}</Stat>
          <Status as="div" tone={gain < 1e-6 ? 'alert' : 'ok'}>{gain < 1e-6 ? 'Zero gain, even though the children differ.' : 'This split reduces impurity.'}</Status>
          <Note>Put both children on the same side of 0.5 with Error rate to reach zero.</Note>
        </Stack>
      </Row>
      <Takeaway>This is why trees grow with Gini or entropy and are only pruned by error rate.</Takeaway>
    </>
  );
}

const SPLIT1 = (() => { const r = rng(6); return Array.from({ length: 24 }, (_, i) => { const x = -2.6 + (5.2 * i) / 23 + 0.05 * gauss(r); return [x, +(x + 0.7 * gauss(r) > 0.3)]; }); })();

export function SplitScan() {
  const [t, setT] = useState(-1), [crit, setCrit] = useState('gini'), g = linspace(-3, 3, 241);
  const gainAt = c => { const L = SPLIT1.filter(p => p[0] <= c), R = SPLIT1.filter(p => p[0] > c); if (!L.length || !R.length) return 0; const a = S => mean(S.map(p => p[1])), P = a(SPLIT1), w = L.length / SPLIT1.length; return impurity(crit, P) - w * impurity(crit, a(L)) - (1 - w) * impurity(crit, a(R)); };
  const best = g.reduce((b, c) => (gainAt(c) > gainAt(b) ? c : b), g[0]), top = Math.max(...g.map(gainAt)) * 1.15 || 0.1;
  return (
    <>
      <p>A tree tries every threshold. <b>Drag the cut</b> across the one feature and read the gain beneath it. The gain only changes when the cut passes a data point, so only midpoints need checking.</p>
      <Toolbar><Choice label="Criterion" value={crit} onChange={setCrit} options={[{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }]} /></Toolbar>
      <Row>
        <Plot caption="Data on one feature" x={[-3, 3]} y={[-0.2, 1.2]} w={380} h={170} xl="feature" label="data strip with a cut">
          <Mark as="points" data={SPLIT1.map(([x, c]) => [x, c, c + 1])} />
          <Mark as="rule" tone="fg" x={t} />
          <Mark as="handles" lockY data={[[t, 0.5]]} onDrag={(_, x) => setT(x)} />
        </Plot>
        <Plot caption="Gain at every threshold" x={[-3, 3]} y={[0, +top.toFixed(3)]} w={380} h={170} xl="threshold" label="gain curve">
          <Mark tone={1} data={g.map(c => [c, gainAt(c)])} />
          <Mark as="rule" tone="fg" x={t} />
          <Mark as="rule" dash x={best} />
        </Plot>
      </Row>
      <Stat big>gain {fmt(gainAt(t), 4)} here · best {fmt(gainAt(best), 4)} at {fmt(best)}</Stat>
      <Toolbar><Button onClick={() => setT(best)}>Jump to the best cut</Button></Toolbar>
      <Takeaway>The search is exhaustive but greedy: the best split now need not lead to the best tree.</Takeaway>
    </>
  );
}

/* ---------- L12b ---------- */

export function CorrelatedAverage() {
  const [rho, setRho] = useState(0.4), [sig, setSig] = useState(1), [B, setB] = useState(25), g = Array.from({ length: 200 }, (_, i) => i + 1), v = b => rho * sig * sig + ((1 - rho) * sig * sig) / b;
  return (
    <>
      <p>Average B predictions, each with variance σ² and pairwise correlation ρ. Independent models cancel their errors, while <b>correlated models hit a floor</b> no amount of averaging can break.</p>
      <M d t="\operatorname{Var}\Big(\frac1B\sum_b \hat f_b\Big)=\rho\sigma^2+\frac{1-\rho}{B}\,\sigma^2" />
      <Row>
        <Plot x={[1, 200]} y={[0, +(sig * sig * 1.05).toFixed(2)]} xl="number of models B" yl="variance of the average" label="variance against ensemble size">
          <Mark tone={1} data={g.map(b => [b, v(b)])} />
          <Mark as="rule" tone={4} dash y={rho * sig * sig} />
          <Mark as="handles" lockY data={[[B, v(B)]]} onDrag={(_, x) => setB(Math.round(clamp(x, 1, 200)))} />
        </Plot>
        <Stack>
          <Slider label={<M t="\rho" />} value={rho} min={0} max={1} step={0.01} onChange={setRho} show={v2 => fmt(v2)} />
          <Slider label={<M t="\sigma^2" />} value={sig} min={0.2} max={2} step={0.01} onChange={setSig} show={v2 => fmt(v2 * v2)} />
          <Stat big>variance at B = {B}: {fmt(v(B), 3)}</Stat>
          <Stat>floor ρσ² = {fmt(rho * sig * sig, 3)}</Stat>
        </Stack>
      </Row>
      <Takeaway>Past a few dozen trees, more trees barely help. A random forest attacks the floor directly, by lowering ρ with random feature subsets.</Takeaway>
    </>
  );
}

export function Bootstrap() {
  const [seed, setSeed] = useState(1), n = 40, draw = useMemo(() => { const r = rng(seed * 101); return Array.from({ length: n }, () => Math.floor(r() * n)); }, [seed]);
  const counts = Array(n).fill(0);
  draw.forEach(i => counts[i]++);
  const inb = counts.filter(c => c).length, g = Array.from({ length: 100 }, (_, i) => i + 1);
  return (
    <>
      <p>Draw n times <b>with replacement</b>. Some rows repeat, and about a third are never picked. Those form a free validation set for that tree.</p>
      <M d t="P(\text{row never drawn})=\Big(1-\frac1n\Big)^{n}\ \longrightarrow\ e^{-1}\approx0.368" />
      <Toolbar><Button variant="solid" onClick={() => setSeed(s => s + 1)}>Draw a new bootstrap sample</Button></Toolbar>
      <Row>
        <Plot caption="Times each of the 40 rows was drawn" x={[-1, 40]} y={[0, 6]} w={400} h={200} xl="row" yl="count" label="bootstrap counts per row">
          <Mark as="rects" tone={1} data={counts.map((c, i) => [i - 0.4, 0, i + 0.4, c])} />
          <Mark as="points" tone={4} r={3} data={counts.map((c, i) => [i, 0.15, c ? undefined : 4]).filter(d => d[2])} />
        </Plot>
        <Plot caption="Share left out, as n grows" x={[1, 100]} y={[0, 0.5]} w={300} h={200} xl="n" label="out-of-bag share against n">
          <Mark tone={2} data={g.map(k => [k, (1 - 1 / k) ** k])} />
          <Mark as="rule" dash y={Math.exp(-1)} />
        </Plot>
      </Row>
      <Stat big>{inb} rows in the bag ({pct(inb / n, 0)}), {n - inb} out of it (red dots)</Stat>
      <Takeaway>Out-of-bag error scores each row using only the trees that did not train on it, so a random forest comes with its own honest estimate.</Takeaway>
    </>
  );
}

const WAVE = (() => { const r = rng(8), xs = Array.from({ length: 40 }, () => 6 * r()); return { X: xs.map(x => [x]), y: xs.map(x => Math.sin(x) + 0.35 * gauss(r)) }; })();

export function BaggedTrees() {
  const [B, setB] = useState(10), [depth, setDepth] = useState(6), g = linspace(0, 6, 241);
  const bag = useMemo(() => bagging(WAVE.X, WAVE.y, { B: 60, maxDepth: depth, criterion: 'mse', seed: 3 }), [depth]);
  const avg = x => bag.predict([x], B), mseOf = k => mean(g.map(x => (bag.predict([x], k) - Math.sin(x)) ** 2));
  const ks = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60];
  return (
    <>
      <p>Each thin line is one tree on one bootstrap sample: jagged, and overfit in its own way. <b>Their average is smooth</b> without any tree being simpler.</p>
      <Row>
        <Plot x={[0, 6]} y={[-2.2, 2.2]} xl="x" yl="y" label="individual bagged trees and their average">
          {bag.trees.slice(0, Math.min(B, 15)).map((t, i) => <Mark key={i} tone="muted" soft data={g.map(x => [x, predictTree(t, [x])])} />)}
          <Mark as="points" tone="fg" r={2.5} data={WAVE.X.map((x, i) => [x[0], WAVE.y[i]])} />
          <Mark tone={3} dash data={g.map(x => [x, Math.sin(x)])} />
          <Mark tone={1} data={g.map(x => [x, avg(x)])} />
        </Plot>
        <Stack>
          <Slider label="trees B" value={B} min={1} max={60} step={1} onChange={setB} />
          <Slider label="tree depth" value={depth} min={1} max={10} step={1} onChange={setDepth} />
          <Plot caption="Error to the truth" x={[1, 60]} y={[0, 0.25]} w={340} h={150} xl="B" label="error against number of trees">
            <Mark tone={4} data={ks.map(k => [k, mseOf(k)])} />
            <Mark as="points" tone={4} r={3} data={[[B, mseOf(B)]]} />
          </Plot>
          <Legend items={[['--c1', 'bagged average'], ['--c3', 'truth'], ['--muted', 'single trees']]} />
        </Stack>
      </Row>
      <Takeaway>Bagging cuts variance and leaves bias alone. So use deep, low-bias trees. Set depth 1 and see that averaging stumps cannot help.</Takeaway>
    </>
  );
}

/* ---------- L12c ---------- */

const TEST_WAVE = linspace(0.02, 5.98, 120).map(x => [x]);

export function BoostRegression() {
  const [k, setK] = useState(5), [lr, setLr] = useState(0.3), [depth, setDepth] = useState(1), g = linspace(0, 6, 241);
  const gb = useMemo(() => gbRegress(WAVE.X, WAVE.y, { rounds: 200, lr, maxDepth: depth }), [lr, depth]);
  const res = WAVE.X.map((x, i) => [x[0], 0, WAVE.y[i] - gb.predict(x, k)]);
  const err = useMemo(() => { const c = [0, 1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 80, 115, 160, 200]; return c.map(q => ({ q, tr: mean(WAVE.X.map((x, i) => (gb.predict(x, q) - WAVE.y[i]) ** 2)), te: mean(TEST_WAVE.map(x => (gb.predict(x, q) - Math.sin(x[0])) ** 2)) })); }, [gb]);
  return (
    <>
      <p>Each round fits a small tree to <b>what is still wrong</b> and adds a fraction of it. Watch the residuals below the fit lose their pattern.</p>
      <M d t="F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad h_m\approx y-F_{m-1}(x)" />
      <Row>
        <Plot caption={`Fit after ${k} rounds`} x={[0, 6]} y={[-2.2, 2.2]} w={360} h={230} xl="x" yl="y" label="boosted fit">
          <Mark as="points" tone="fg" r={2.5} data={WAVE.X.map((x, i) => [x[0], WAVE.y[i]])} />
          <Mark tone={3} dash data={g.map(x => [x, Math.sin(x)])} />
          <Mark tone={1} data={g.map(x => [x, gb.predict([x], k)])} />
        </Plot>
        <Plot caption="Residuals: the next tree's target" x={[0, 6]} y={[-2.2, 2.2]} w={360} h={230} xl="x" yl="r" label="residuals">
          <Mark as="stems" tone={2} data={res} />
        </Plot>
      </Row>
      <Row>
        <Stack>
          <Slider label="rounds" value={k} min={0} max={200} step={1} onChange={setK} />
          <Slider label={<>learning rate <M t="\nu" /></>} value={lr} min={0.02} max={1} step={0.01} onChange={setLr} show={v => fmt(v)} />
          <Slider label="tree depth" value={depth} min={1} max={4} step={1} onChange={setDepth} />
        </Stack>
        <Plot caption="Error by round" x={[0, 200]} y={[0, 0.8]} w={360} h={170} xl="rounds" label="train and test error by round">
          <Mark tone={3} data={err.map(o => [o.q, o.tr])} />
          <Mark tone={4} data={err.map(o => [o.q, o.te])} />
          <Mark as="rule" dash x={k} />
        </Plot>
      </Row>
      <Takeaway>Boosting reduces bias, the opposite of bagging. Too many rounds at a high rate starts fitting noise: the red test curve turns up while the green train curve keeps falling.</Takeaway>
    </>
  );
}

export function BoostClass() {
  const [kind, setKind] = useState('circles'), [k, setK] = useState(10), [lr, setLr] = useState(0.3);
  const D = useMemo(() => dataset(kind, 5, 60), [kind]), T = useMemo(() => dataset(kind, 55, 200), [kind]);
  const gb = useMemo(() => { const [X, y] = split2(D); return gbClass(X, y, { rounds: 80, lr, maxDepth: 2 }); }, [D, lr]);
  const p = x => gb.predict(x, k), ll = (S, q) => mean(S.map(d => { const v = clamp(gb.predict([d[0], d[1]], q), 1e-9, 1 - 1e-9); return d[2] ? -Math.log(v) : -Math.log(1 - v); }));
  const qs = [0, 1, 2, 4, 7, 11, 16, 24, 35, 50, 65, 80];
  return (
    <>
      <p>For classification the trees model the <b>log-odds</b>. The residual is <M t="y-p" />, and each leaf takes one Newton step.</p>
      <M d t="r_i=y_i-p_i,\qquad \text{leaf}=\frac{\sum r_i}{\sum p_i(1-p_i)}" />
      <Toolbar>{dataChoice(kind, setKind)}</Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} w={380} h={360} xl="x₁" yl="x₂" label="boosted classifier probability field">
          <Mark as="field" nx={44} ny={32} fn={(a, b) => prob(p([a, b]))} />
          <Mark as="points" r={4} data={toned(D)} />
        </Plot>
        <Stack>
          <Slider label="rounds" value={k} min={0} max={80} step={1} onChange={setK} />
          <Slider label="learning rate" value={lr} min={0.03} max={1} step={0.01} onChange={setLr} show={v => fmt(v)} />
          <Stat big>train {pct(accuracy(p, D), 0)} · test {pct(accuracy(p, T), 0)}</Stat>
          <Plot caption="Log loss by round" x={[0, 80]} y={[0, 0.8]} w={340} h={150} xl="rounds" label="log loss by round">
            <Mark tone={3} data={qs.map(q => [q, ll(D, q)])} />
            <Mark tone={4} data={qs.map(q => [q, ll(T, q)])} />
            <Mark as="rule" dash x={k} />
          </Plot>
        </Stack>
      </Row>
      <Takeaway>Round 0 is just the base rate. The picture sharpens as the sum of log-odds grows, which is the same recipe as regression with a different loss.</Takeaway>
    </>
  );
}

/* ---------- L12d ---------- */

export function AdaLab() {
  const [kind, setKind] = useState('circles'), [k, setK] = useState(1);
  const D = useMemo(() => dataset(kind, 9, 40), [kind]), ada = useMemo(() => { const [X, y] = split2(D); return adaboost(X, y, { rounds: 40 }); }, [D]);
  const r = ada.rounds[k - 1], err = q => D.filter(d => (ada.score([d[0], d[1]], q) > 0) !== !!d[2]).length / D.length;
  const qs = Array.from({ length: 40 }, (_, i) => i + 1);
  let bound = 1;
  const bs = qs.map(q => (bound *= ada.rounds[q - 1].Z, [q, bound]));
  const score = (a, b) => ada.score([a, b], k), max = Math.max(1e-6, ...ada.rounds.slice(0, k).map(o => o.alpha)) * 1.5;
  return (
    <>
      <p>Dot size is <b>the weight each point has this round</b>. Misclassified points grow, so the next stump must attend to them. Step through and watch the attention move.</p>
      <M d t="\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t},\qquad w_i\leftarrow\frac{w_i\,e^{-\alpha_t y_i h_t(x_i)}}{Z_t}" />
      <Toolbar>{dataChoice(kind, setKind)}<Button onClick={() => setK(v => Math.max(1, v - 1))}>Previous round</Button><Button variant="solid" onClick={() => setK(v => Math.min(40, v + 1))}>Next round</Button></Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} w={380} h={360} xl="x₁" yl="x₂" label="weighted points and committee decision">
          <Mark as="field" nx={40} ny={30} fn={(a, b) => clamp(score(a, b) / max, -1, 1)} />
          {r.tree.l && <Mark as="rule" tone="fg" {...(r.tree.f === 0 ? { x: r.tree.t } : { y: r.tree.t })} />}
          <Mark as="points" data={D.map((d, i) => [d[0], d[1], d[2] + 1, clamp(2 + r.weights[i] * 40 * 4, 2, 14)])} />
        </Plot>
        <Stack>
          <Slider label="round" value={k} min={1} max={40} step={1} onChange={setK} />
          <Stat big>round {k}: ε = {fmt(r.eps, 3)}, α = {fmt(r.alpha, 3)}</Stat>
          <Stat>committee training error {pct(err(k), 0)}</Stat>
          <Plot caption="Training error stays under ΠZₜ" x={[1, 40]} y={[0, 1]} w={340} h={150} xl="rounds" label="training error and its bound">
            <Mark tone={1} data={bs} />
            <Mark tone={4} data={qs.map(q => [q, err(q)])} />
            <Mark as="rule" dash x={k} />
          </Plot>
          <Note>The black line is the stump added this round.</Note>
        </Stack>
      </Row>
      <Takeaway>After each update the stump just used scores exactly 50% on the new weights, so it can never be chosen twice in a row. Mislabeled points gain weight without limit, and that is AdaBoost's weakness.</Takeaway>
    </>
  );
}

export function AlphaZ() {
  const [eps, setEps] = useState(0.25), [a, setA] = useState(0.2), g = linspace(-1, 2.5), Z = v => (1 - eps) * Math.exp(-v) + eps * Math.exp(v), best = 0.5 * Math.log((1 - eps) / eps), es = linspace(0.01, 0.99);
  return (
    <>
      <p><M t="\alpha" /> is not chosen by taste. It is the exact minimiser of the normalizer <M t="Z" />. Drag along the curve to the bottom.</p>
      <M d t="Z(\alpha)=(1-\varepsilon)e^{-\alpha}+\varepsilon e^{\alpha},\qquad \frac{dZ}{d\alpha}=0\Rightarrow\alpha^*=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}" />
      <Row>
        <Plot caption="Z as a function of α" x={[-1, 2.5]} y={[0, 3]} w={340} h={260} xl="α" yl="Z" label="Z against alpha">
          <Mark tone={1} data={g.map(v => [v, Z(v)])} />
          <Mark as="rule" dash x={best} />
          <Mark as="handles" lockY data={[[a, Z(a)]]} onDrag={(_, x) => setA(x)} />
        </Plot>
        <Plot caption="Best α against error" x={[0, 1]} y={[-2, 2]} w={340} h={260} xl="ε" yl="α*" label="optimal alpha against epsilon">
          <Mark tone={2} data={es.map(v => [v, 0.5 * Math.log((1 - v) / v)])} />
          <Mark as="points" tone={4} data={[[eps, best]]} />
        </Plot>
      </Row>
      <Slider label={<M t="\varepsilon" />} value={eps} min={0.01} max={0.99} step={0.01} onChange={setEps} show={v => fmt(v)} />
      <Stat big>Z({fmt(a)}) = {fmt(Z(a), 4)}; minimum {fmt(Z(best), 4)} = 2√(ε(1−ε))</Stat>
      <Takeaway>ε = 0.5 gets α = 0: a coin toss has no vote. ε above 0.5 gets a negative vote, so the committee simply reverses it.</Takeaway>
    </>
  );
}

export function WeightedMedian() {
  const init = [[1, 1], [2.5, 2], [4, 1], [6, 0.5], [8.5, 3]];
  const [pts, setPts] = useState(init), s = [...pts].sort((a, b) => a[0] - b[0]), total = s.reduce((a, p) => a + p[1], 0);
  let c = 0;
  const steps = s.map(p => { c += p[1]; return [p[0], c / total]; }), wm = steps.find(q => q[1] >= 0.5)[0], line = [[0, 0], ...steps.flatMap((q, i) => [[q[0], i ? steps[i - 1][1] : 0], q]), [10, 1]];
  return (
    <>
      <p>AdaBoost.R2 combines models by a weighted median: <b>walk along the sorted predictions until you have collected half the weight</b>. Drag stem tops to change the weights, and the position to change a prediction.</p>
      <Row>
        <Plot caption="Predictions and their weights" x={[0, 10]} y={[0, 4]} w={360} h={250} xl="prediction" yl="weight" label="predictions with weights">
          <Mark as="stems" tone={1} data={pts.map(p => [p[0], 0, p[1]])} />
          <Mark as="rule" tone={4} x={wm} />
          <Mark as="handles" data={pts} onDrag={(i, x, y) => setPts(moved(i, x, Math.max(y, 0.05)))} />
        </Plot>
        <Plot caption="Cumulative weight" x={[0, 10]} y={[0, 1]} w={300} h={250} xl="prediction" label="cumulative weight staircase">
          <Mark tone={2} data={line} />
          <Mark as="rule" dash y={0.5} />
          <Mark as="rule" tone={4} x={wm} />
        </Plot>
      </Row>
      <Stat big>weighted median {fmt(wm)} · weighted mean {fmt(pts.reduce((a, p) => a + p[0] * p[1], 0) / total)}</Stat>
      <Takeaway>Move one prediction far right. The median holds and the mean does not, which makes this combination robust to a few wild models.</Takeaway>
    </>
  );
}

/* ---------- L12e ---------- */

export function XgbGain() {
  const [gl, setGl] = useState(-6), [hl, setHl] = useState(3), [gr, setGr] = useState(4), [hr, setHr] = useState(4), [lam, setLam] = useState(1), [gam, setGam] = useState(0.5);
  const S = (g, h) => xgbScore(g, h, lam), gain = 0.5 * (S(gl, hl) + S(gr, hr) - S(gl + gr, hl + hr)) - gam, wl = -gl / (hl + lam), wr = -gr / (hr + lam);
  const g = linspace(0, 20), curve = l => 0.5 * (xgbScore(gl, hl, l) + xgbScore(gr, hr, l) - xgbScore(gl + gr, hl + hr, l)) - gam;
  return (
    <>
      <p>XGBoost scores a split from four numbers: the summed slopes and curvatures on each side. Move them, and see <M t="\lambda" /> shrink leaves and <M t="\gamma" /> <b>veto splits that do not pay their way</b>.</p>
      <M d t="w^*=-\frac{G}{H+\lambda},\qquad \text{gain}=\tfrac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma" />
      <Row>
        <Plot caption="Gain as λ grows" x={[0, 20]} y={[-2, 12]} w={340} h={260} xl="λ" yl="gain" label="gain against lambda">
          <Mark tone={1} data={g.map(l => [l, curve(l)])} />
          <Mark as="rule" tone={4} y={0} />
          <Mark as="points" tone={4} r={6} data={[[lam, clamp(gain, -2, 12)]]} />
        </Plot>
        <Stack>
          <Slider label={<M t="G_L" />} value={gl} min={-10} max={10} step={0.5} onChange={setGl} show={v => fmt(v, 1)} />
          <Slider label={<M t="H_L" />} value={hl} min={0.25} max={10} step={0.25} onChange={setHl} show={v => fmt(v)} />
          <Slider label={<M t="G_R" />} value={gr} min={-10} max={10} step={0.5} onChange={setGr} show={v => fmt(v, 1)} />
          <Slider label={<M t="H_R" />} value={hr} min={0.25} max={10} step={0.25} onChange={setHr} show={v => fmt(v)} />
          <Slider label={<M t="\lambda" />} value={lam} min={0} max={20} step={0.25} onChange={setLam} show={v => fmt(v)} />
          <Slider label={<M t="\gamma" />} value={gam} min={0} max={5} step={0.05} onChange={setGam} show={v => fmt(v)} />
          <Stat big>gain {fmt(gain, 3)}</Stat>
          <Status as="div" tone={gain > 0 ? 'ok' : 'alert'}>{gain > 0 ? 'Keep the split.' : 'Prune: the split does not cover γ.'}</Status>
          <Stat>leaf weights {fmt(wl, 3)} and {fmt(wr, 3)}</Stat>
        </Stack>
      </Row>
      <Takeaway>With λ = 0 and squared loss, H counts the rows and the score reduces to (Σr)²/n from gradient boosting. XGBoost generalizes that formula to any twice-differentiable loss.</Takeaway>
    </>
  );
}

export function Compare() {
  const [kind, setKind] = useState('circles'), [noise, setNoise] = useState(0.35), [seed, setSeed] = useState(2);
  const models = useMemo(() => {
    const D = dataset(kind, seed, 80, noise), T = dataset(kind, seed + 50, 300, noise), [X, y] = split2(D);
    const tree = buildTree(X, y, { maxDepth: 12 });
    const all = [['Single tree', x => predictTree(tree, x)], ['Bagging', bagging(X, y, { B: 25, maxDepth: 12 }).predict], ['Random forest', bagging(X, y, { B: 25, maxDepth: 12, m: 1, seed: 5 }).predict],
      ['Gradient boosting', gbClass(X, y, { rounds: 40, lr: 0.2 }).predict], ['AdaBoost', adaboost(X, y, { rounds: 40 }).predict], ['XGBoost', xgbClass(X, y, { rounds: 40, lr: 0.2, lambda: 1, gamma: 0.1 }).predict]];
    return { D, list: all.map(([name, fn]) => ({ name, fn, tr: accuracy(fn, D), te: accuracy(fn, T) })) };
  }, [kind, seed, noise]);
  const best = Math.max(...models.list.map(m => m.te));
  return (
    <>
      <p>The same training points, six learners, one held-out set. Compare <b>the shape of the boundary</b> as much as the score.</p>
      <Toolbar>{dataChoice(kind, setKind)}<Button onClick={() => setSeed(s => s + 1)}>New sample</Button></Toolbar>
      <Slider label="noise in the features" value={noise} min={0.1} max={0.9} step={0.05} onChange={setNoise} show={v => fmt(v)} />
      <div className="minis">
        {models.list.map(m => (
          <Plot key={m.name} caption={`${m.name} · test ${pct(m.te, 0)}`} x={[-3, 3]} y={[-3, 3]} w={240} h={220} label={m.name}>
            <Mark as="field" nx={30} ny={24} fn={(a, b) => prob(m.fn([a, b]))} />
            <Mark as="points" r={2.5} data={toned(models.D)} />
          </Plot>
        ))}
      </div>
      <Table>
        <thead><tr><th>Model</th><th>Train</th><th>Test</th></tr></thead>
        <tbody>{models.list.map(m => <tr key={m.name}><th>{m.name}</th><td>{pct(m.tr, 0)}</td><td className={m.te === best ? 'win' : ''}>{pct(m.te, 0)}</td></tr>)}</tbody>
      </Table>
      <Takeaway>A single deep tree has the highest training score and jagged regions. Ensembles give up a little of that to generalize, which is the whole point of the lecture.</Takeaway>
    </>
  );
}

