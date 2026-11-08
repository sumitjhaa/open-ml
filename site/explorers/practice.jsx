'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Bars, Mark, Plot, moved } from '../components/Plot';
import { Button, Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from '../components/ui';
import { gauss, linfit, mean, polyfit, polyval, rng, sum } from '../lib/math.mjs';
import { accuracy, clamp, dataset, fmt, linspace, pct } from '../lib/models.mjs';

/* ---------- data leakage ---------- */

function leakRun(p, seed, leak) {
  const r = rng(seed), n = 40, top = 10;
  const X = Array.from({ length: n }, () => Array.from({ length: p }, () => gauss(r))), y = Array.from({ length: n }, () => +(r() < 0.5));
  const tr = Array.from({ length: n / 2 }, (_, i) => i), te = tr.map(i => i + n / 2), basis = leak ? [...tr, ...te] : tr;
  const score = j => { const a = basis.filter(i => y[i]), b = basis.filter(i => !y[i]); return Math.abs(mean(a.map(i => X[i][j])) - mean(b.map(i => X[i][j]))); };
  const keep = Array.from({ length: p }, (_, j) => [score(j), j]).sort((a, b) => b[0] - a[0]).slice(0, top).map(q => q[1]);
  const centre = c => { const s = tr.filter(i => y[i] === c); return keep.map(j => mean(s.map(i => X[i][j]))); };
  const c1 = centre(1), c0 = centre(0), d = (i, c) => keep.reduce((s, j, k) => s + (X[i][j] - c[k]) ** 2, 0);
  return mean(te.map(i => +((d(i, c1) < d(i, c0)) === !!y[i])));
}

export function Leakage() {
  const [p, setP] = useState(500), [mode, setMode] = useState('leak');
  const acc = useMemo(() => { const a = [], b = []; for (let s = 1; s <= 12; s++) { a.push(leakRun(p, s, true)); b.push(leakRun(p, s, false)); } return { leak: a, clean: b }; }, [p]);
  const shown = acc[mode], avg = mean(shown);
  return (
    <>
      <p>Every feature is <b>pure noise</b> and the labels are coin tosses, so no method can truly beat 50%. Choose the 10 "best" features using the whole dataset, and see what the test set reports.</p>
      <Toolbar><Choice label="Feature selection" value={mode} onChange={setMode} options={[{ value: 'leak', label: 'Select on all data' }, { value: 'clean', label: 'Select on training only' }]} /></Toolbar>
      <Row>
        <Plot caption="Test accuracy on 12 datasets" x={[0, 13]} y={[0.2, 1]} xl="dataset" yl="accuracy" label="test accuracy per repeat">
          <Mark as="rects" tone={mode === 'leak' ? 4 : 3} data={shown.map((v, i) => [i + 0.6, 0.2, i + 1.4, v])} />
          <Mark as="rule" dash y={0.5} />
        </Plot>
        <Stack>
          <Slider label="noise features" value={p} min={10} max={1500} step={10} onChange={setP} />
          <Stat big>average test accuracy {pct(avg, 0)}</Stat>
          <Status as="div" tone={avg > 0.6 ? 'alert' : 'ok'}>{avg > 0.6 ? 'A confident result from pure noise.' : 'The honest answer: nothing is learnable here.'}</Status>
          <Note>The model never saw test labels in training, but the feature filter did. Every preprocessing step must be fit inside the training fold.</Note>
        </Stack>
      </Row>
      <Takeaway>Any step that looks at test data is part of the training. Scalers, imputers, filters and target encoders all count.</Takeaway>
    </>
  );
}

/* ---------- distribution shift ---------- */

const SHIFT = (() => { const r = rng(14), xs = Array.from({ length: 40 }, () => -1 + 0.6 * gauss(r)); return { xs, ys: xs.map(x => Math.sin(1.2 * x) + 0.12 * gauss(r)) }; })();

export function Shift() {
  const [mu, setMu] = useState(0.5), [deg, setDeg] = useState(3), g = linspace(-3.5, 3.5, 281), truth = x => Math.sin(1.2 * x);
  const c = polyfit(SHIFT.xs.map(x => x / 3.5), SHIFT.ys, deg), pred = x => polyval(c, x / 3.5);
  const test = useMemo(() => { const r = rng(44); return Array.from({ length: 300 }, () => gauss(r) * 0.6); }, []);
  const err = m => mean(test.map(z => (pred(z + m) - truth(z + m)) ** 2)), train = mean(SHIFT.xs.map((x, i) => (pred(x) - SHIFT.ys[i]) ** 2));
  const dens = (x, m) => Math.exp(-((x - m) ** 2) / 0.72);
  return (
    <>
      <p>The model trains where the blue hump is. <b>Drag the deployment distribution</b> away and see what extrapolation does, although the true rule never changed.</p>
      <Row>
        <Plot x={[-3.5, 3.5]} y={[-2, 2]} xl="x" yl="y" label="training and deployment distributions with model and truth">
          <Mark as="area" tone={1} data={g.map(x => [x, -2, -2 + dens(x, -1)])} />
          <Mark as="area" tone={2} data={g.map(x => [x, -2, -2 + dens(x, mu)])} />
          <Mark tone={3} dash data={g.map(x => [x, truth(x)])} />
          <Mark tone={4} data={g.map(x => [x, clamp(pred(x), -5, 5)])} />
          <Mark as="points" tone="fg" r={2.5} data={SHIFT.xs.map((x, i) => [x, SHIFT.ys[i]])} />
          <Mark as="handles" lockY data={[[mu, -1.3, 2]]} onDrag={(_, x) => setMu(x)} />
        </Plot>
        <Stack>
          <Slider label="model degree" value={deg} min={1} max={9} step={1} onChange={setDeg} />
          <Stat big>training error {fmt(train, 3)}</Stat>
          <Stat big>deployment error {fmt(err(mu), 3)}</Stat>
          <Legend items={[['--c1', 'train inputs'], ['--c2', 'deploy inputs'], ['--c3', 'truth'], ['--c4', 'model']]} />
        </Stack>
      </Row>
      <Takeaway>This is covariate shift: p(x) moved while p(y|x) did not. A random split can never reveal it, because a random test set is drawn from the training world. Split by time, site or device.</Takeaway>
    </>
  );
}

/* ---------- learning curves ---------- */

function curves(deg, seed) {
  const r = rng(seed), xs = Array.from({ length: 200 }, () => 2 * r() - 1), ys = xs.map(x => Math.sin(Math.PI * x) + 0.3 * gauss(r)), tx = Array.from({ length: 300 }, () => 2 * r() - 1), ty = tx.map(x => Math.sin(Math.PI * x) + 0.3 * gauss(r));
  return [deg + 2, 12, 16, 22, 30, 40, 60, 80, 120, 160, 200].filter((n, i, a) => n > deg + 1 && a.indexOf(n) === i).sort((a, b) => a - b).map(n => {
    const c = polyfit(xs.slice(0, n), ys.slice(0, n), deg), e = (a, b) => mean(a.map((x, i) => (polyval(c, x) - b[i]) ** 2));
    return { n, tr: e(xs.slice(0, n), ys.slice(0, n)), te: e(tx, ty) };
  });
}

export function LearningCurve() {
  const [deg, setDeg] = useState(1), c = useMemo(() => curves(deg, 7), [deg]), last = c[c.length - 1], gap = last.te - last.tr;
  return (
    <>
      <p>Before buying more data, plot error against training size. The two curves <b>diagnose the problem</b> for you.</p>
      <Row>
        <Plot x={[0, 200]} y={[0, 0.8]} xl="training examples" yl="squared error" label="learning curves">
          <Mark tone={3} data={c.map(o => [o.n, Math.min(o.tr, 2)])} />
          <Mark tone={4} data={c.map(o => [o.n, Math.min(o.te, 2)])} />
          <Mark as="rule" dash y={0.09} />
        </Plot>
        <Stack>
          <Slider label="polynomial degree" value={deg} min={0} max={9} step={1} onChange={setDeg} />
          <Stat big>final train {fmt(last.tr, 3)} · test {fmt(last.te, 3)}</Stat>
          <Status as="div" tone={last.te > 0.2 ? 'alert' : 'ok'}>{last.te > 0.2 ? 'High bias: curves meet high up, and more data will not help. Use a richer model.' : gap > 0.05 ? 'High variance: a gap remains, so more data should help.' : 'Close to the noise floor: both curves meet near 0.09.'}</Status>
          <Legend items={[['--c3', 'train'], ['--c4', 'test']]} />
        </Stack>
      </Row>
      <Takeaway>Training error rises and test error falls as data grows, converging towards the best this model class can do.</Takeaway>
    </>
  );
}

/* ---------- SGD and batch size ---------- */

const SG = (() => { const r = rng(12), xs = Array.from({ length: 60 }, () => -2 + 4 * r()); return { xs, ys: xs.map(x => x * 1.2 + 0.6 + 0.8 * gauss(r)) }; })();

export function SgdLab() {
  const [batch, setBatch] = useState('8'), [eta, setEta] = useState(0.1), [seed, setSeed] = useState(1), [decay, setDecay] = useState(false);
  const L = (w, b) => mean(SG.xs.map((x, i) => (SG.ys[i] - w * x - b) ** 2)), best = linfit(SG.xs, SG.ys), max = L(-3, 3);
  const path = useMemo(() => {
    const r = rng(seed * 37), k = +batch, n = SG.xs.length, p = [[-2.5, 2.5]];
    for (let t = 0; t < 120; t++) {
      const [w, b] = p[t], idx = Array.from({ length: k }, (_, i) => (k >= n ? i : Math.floor(r() * n)));
      let gw = 0, gb = 0;
      for (const i of idx) { const e = w * SG.xs[i] + b - SG.ys[i]; gw += (2 * e * SG.xs[i]) / k; gb += (2 * e) / k; }
      const s = decay ? eta / (1 + 0.05 * t) : eta;
      p.push([clamp(w - s * gw, -8, 8), clamp(b - s * gb, -8, 8)]);
    }
    return p;
  }, [batch, eta, seed, decay]);
  const tail = path.slice(-30), noise = mean(tail.map(q => Math.hypot(q[0] - best.w, q[1] - best.b)));
  return (
    <>
      <p>A mini-batch gives a <b>noisy but cheap</b> gradient. Compare the path shapes, and see why SGD with a fixed step never quite settles.</p>
      <Toolbar>
        <Choice label="Batch size" value={batch} onChange={setBatch} options={[['1', '1 (SGD)'], ['8', '8'], ['60', '60 (full)']].map(([value, label]) => ({ value, label }))} />
        <Choice label="Schedule" value={decay ? 'decay' : 'fixed'} onChange={v => setDecay(v === 'decay')} options={[{ value: 'fixed', label: 'fixed η' }, { value: 'decay', label: 'decaying η' }]} />
        <Button onClick={() => setSeed(s => s + 1)}>Run again</Button>
      </Toolbar>
      <Row>
        <Plot x={[-3, 4]} y={[-3, 4]} w={380} h={360} xl="w" yl="b" label="optimizer path on the loss surface">
          <Mark as="field" bands={9} tones={[1, 1]} fn={(a, b) => Math.sqrt(clamp(L(a, b) / max, 0, 1))} />
          <Mark tone="fg" data={path} />
          <Mark as="points" tone={3} r={6} data={[[best.w, best.b]]} />
        </Plot>
        <Stack>
          <Slider label={<M t="\eta" />} value={eta} min={0.005} max={0.6} step={0.005} onChange={setEta} show={v => fmt(v, 3)} />
          <Stat big>final loss {fmt(L(...path[path.length - 1]), 3)}</Stat>
          <Stat>best possible {fmt(L(best.w, best.b), 3)}</Stat>
          <Stat>average distance from the optimum over the last 30 steps: {fmt(noise, 3)}</Stat>
        </Stack>
      </Row>
      <Takeaway>Noise is proportional to step size, so a fixed η leaves a jitter around the optimum. A decaying step removes it at the price of slower progress.</Takeaway>
    </>
  );
}

/* ---------- R-squared and residuals ---------- */

const WORLDS = {
  linear: { name: 'Linear', f: (x, e) => 0.8 * x + 1 + 0.8 * e },
  curved: { name: 'Curved', f: (x, e) => 0.18 * (x - 5) ** 2 + 1 + 0.25 * e },
  funnel: { name: 'Funnel', f: (x, e) => 0.8 * x + 1 + 0.25 * x * e },
};

export function Residuals() {
  const [w, setW] = useState('curved');
  const pts = useMemo(() => { const r = rng(19); return Array.from({ length: 50 }, (_, i) => { const x = (10 * i) / 49; return [x, WORLDS[w].f(x, gauss(r))]; }); }, [w]);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), { w: m, b } = linfit(xs, ys), res = pts.map(([x, y]) => [x, y - m * x - b]), my = mean(ys);
  const R2 = 1 - sum(res.map(r => r[1] ** 2)) / sum(ys.map(y => (y - my) ** 2));
  return (
    <>
      <p>R² alone cannot tell these apart. <b>The residual plot can.</b> A good model leaves nothing but shapeless noise behind.</p>
      <M d t="R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}" />
      <Toolbar><Choice label="World" value={w} onChange={setW} options={Object.entries(WORLDS).map(([value, v]) => ({ value, label: v.name }))} /></Toolbar>
      <Row>
        <Plot caption="Data and line" x={[0, 10]} y={[-2, 14]} w={340} h={240} xl="x" yl="y" label="data and least squares line">
          <Mark as="points" tone="fg" r={3} data={pts} />
          <Mark tone={1} data={[[0, b], [10, m * 10 + b]]} />
        </Plot>
        <Plot caption="Residuals against x" x={[0, 10]} y={[-5, 5]} w={340} h={240} xl="x" yl="residual" label="residual plot">
          <Mark as="rule" dash y={0} />
          <Mark as="points" tone={2} r={3} data={res} />
        </Plot>
      </Row>
      <Stat big>R² = {fmt(R2, 3)}</Stat>
      <Takeaway>A curve in the residuals means a wrong model shape. A funnel means the noise grows with x, so standard errors and intervals cannot be trusted.</Takeaway>
    </>
  );
}

/* ---------- softmax and averaging ---------- */

export function Softmax() {
  const [z, setZ] = useState([2, 1, 0.1]), [T, setT] = useState(1), [rare, setRare] = useState(0.2);
  const e = z.map(v => Math.exp(v / T)), s = sum(e), p = e.map(v => v / s);
  const n = [900, 90, 10], rec = [0.98, 0.7, rare], tp = n.map((v, i) => v * rec[i]), micro = sum(tp) / sum(n), macro = mean(rec);
  return (
    <>
      <p>Softmax turns any real scores into probabilities. Below it, see why a <b>rare class vanishes from micro-averaged scores</b>.</p>
      <M d t="p_k=\frac{e^{z_k/T}}{\sum_j e^{z_j/T}}" />
      <Row>
        <Stack>
          {z.map((v, i) => <Slider key={i} label={<M t={`z_${i + 1}`} />} value={v} min={-4} max={4} step={0.05} show={u => fmt(u)} onChange={u => setZ(q => q.map((o, j) => (j === i ? u : o)))} />)}
          <Slider label="temperature T" value={T} min={0.1} max={5} step={0.05} onChange={setT} show={v => fmt(v)} />
          <Note>Adding a constant to every z changes nothing, which is why one class's score can be fixed at zero.</Note>
        </Stack>
        <Stack>
          <Bars data={p.map((v, i) => ({ label: `class ${i + 1}`, title: pct(v), height: v * 100 }))} />
          <Stat big>{p.map(v => pct(v, 0)).join(' · ')}</Stat>
        </Stack>
      </Row>
      <Row>
        <Stack>
          <Slider label="recall on the rare class" value={rare} min={0} max={1} step={0.01} onChange={setRare} show={v => pct(v, 0)} />
          <Stat big>micro {pct(micro)} · macro {pct(macro)}</Stat>
        </Stack>
        <Table>
          <thead><tr><th>Class</th><th>Support</th><th>Recall</th></tr></thead>
          <tbody>{n.map((v, i) => <tr key={i}><th>{i + 1}</th><td>{v}</td><td>{pct(rec[i], 0)}</td></tr>)}</tbody>
        </Table>
      </Row>
      <Takeaway>Micro-average weights by examples, macro weights every class equally. Report macro whenever the small classes are the ones you care about.</Takeaway>
    </>
  );
}

/* ---------- K-nearest neighbours ---------- */

export function Knn() {
  const [kind, setKind] = useState('overlap'), [pts, setPts] = useState(() => dataset('overlap', 2, 36).map(d => [d[0], d[1], d[2] + 1])), [k, setK] = useState(5), [q, setQ] = useState([0.3, 0.2]);
  const near = (a, b) => pts.map((p, i) => [(p[0] - a) ** 2 + (p[1] - b) ** 2, i]).sort((u, v) => u[0] - v[0]).slice(0, k);
  const prob = (a, b) => mean(near(a, b).map(n => +(pts[n[1]][2] === 2)));
  const test = useMemo(() => dataset(kind, 90, 150), [kind]), hood = near(...q);
  return (
    <>
      <p>No training: just store the points and ask the <M t="k" /> nearest. <b>Drag the query</b> or any point. Small <M t="k" /> means islands, and large means blur.</p>
      <Toolbar><Choice label="Data" value={kind} onChange={v => { setKind(v); setPts(dataset(v, 2, 36).map(d => [d[0], d[1], d[2] + 1])); }} options={[{ value: 'overlap', label: 'Overlap' }, { value: 'xor', label: 'XOR' }, { value: 'circles', label: 'Circles' }]} /></Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} w={380} h={360} xl="x₁" yl="x₂" label="k-nearest-neighbour decision regions">
          <Mark as="field" nx={46} ny={34} fn={(a, b) => 2 * prob(a, b) - 1} />
          {hood.map(n => <Mark key={n[1]} tone="fg" data={[q, pts[n[1]]]} />)}
          <Mark as="handles" r={4.5} data={pts} onDrag={(i, a, b) => setPts(moved(i, a, b))} />
          <Mark as="handles" r={8} data={[[...q, 3]]} onDrag={(_, a, b) => setQ([a, b])} />
        </Plot>
        <Stack>
          <Slider label={<M t="k" />} value={k} min={1} max={35} step={2} onChange={setK} />
          <Stat big>P(class 1) at the query = {fmt(prob(...q))}</Stat>
          <Stat>test accuracy {pct(accuracy(x => prob(...x), test), 0)}</Stat>
          <Note>Because it is distance-based, kNN needs scaled features.</Note>
        </Stack>
      </Row>
      <Takeaway>k = 1 gets every training point right by construction, which says nothing. k is a capacity knob, read backwards.</Takeaway>
    </>
  );
}

/* ---------- curse of dimensionality ---------- */

export function Curse() {
  const [d, setD] = useState(2);
  const h = useMemo(() => {
    const r = rng(23), q = Array.from({ length: d }, () => r()), D = Array.from({ length: 400 }, () => Math.sqrt(q.reduce((s, v) => s + (r() - v) ** 2, 0)));
    const lo = Math.min(...D), hi = Math.max(...D), bins = Array(24).fill(0);
    D.forEach(v => bins[Math.min(23, Math.floor((v / (hi * 1.0001)) * 24))]++);
    return { lo, hi, bins, m: mean(D) };
  }, [d]);
  const ratio = (h.hi - h.lo) / h.lo, top = Math.max(...h.bins);
  return (
    <>
      <p>Distances from a query to 400 uniform points, measured relative to the farthest. Raise the dimension, and <b>nearest and farthest become almost the same</b>.</p>
      <Row>
        <Plot x={[0, 1]} y={[0, top]} xl="distance ÷ farthest distance" yl="points" label="histogram of distances">
          <Mark as="rects" tone={1} data={h.bins.map((c, i) => [i / 24, 0, (i + 1) / 24, c])} />
        </Plot>
        <Stack>
          <Slider label="dimensions" value={d} min={1} max={1000} step={1} onChange={setD} />
          <Stat big>(farthest − nearest) ÷ nearest = {fmt(ratio, 3)}</Stat>
          <Stat>nearest {fmt(h.lo, 2)}, farthest {fmt(h.hi, 2)}</Stat>
          <Note>To keep 10 points per axis you would need 10ᵈ points in all.</Note>
        </Stack>
      </Row>
      <Takeaway>When every point is about equally far, "nearest" stops meaning anything, so methods based on neighbours, kernels and density all weaken. Real data helps by living near a lower-dimensional surface.</Takeaway>
    </>
  );
}
