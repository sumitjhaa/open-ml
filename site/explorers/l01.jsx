'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Bars, DotGrid, Mark, Plot } from '../components/Plot';
import { Button, Choice, Field, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from '../components/ui';
import { mean, median, sum, rng, gauss, polyfit, polyval, mse, linfit } from '../lib/math.mjs';
import { doubleDescent } from '../lib/models.mjs';

const f = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const pct = (v, d = 1) => (Number.isFinite(v) ? (100 * v).toFixed(d) + '%' : 'undefined (0/0)');

/* ---------- 1. Turn the knobs: f_θ(x) = wx + b and empirical risk ---------- */

const KNOB_DATA = (() => {
  const r = rng(7), xs = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4];
  return { xs, ys: xs.map(x => 2 * x + 1 + 0.8 * gauss(r)) };
})();

export function Knobs() {
  const [w, setW] = useState(0.5), [b, setB] = useState(3);
  const { xs, ys } = KNOB_DATA;
  const best = useMemo(() => linfit(xs, ys), [xs, ys]);
  const risk = mse(x => w * x + b, xs, ys);
  const bestRisk = mse(x => best.w * x + best.b, xs, ys);
  return (
    <>
      <p>A model is a <b>machine with knobs</b>: <M t="f_\theta(x) = wx + b,\ \theta=(w,b)" />. Each knob setting is one line. Drag the line's two ends, or use the sliders. The
        data never changes; only the knobs do. We score a setting by the average squared miss — the <b>empirical risk</b>:</p>
      <M d t="\hat R(\theta) = \frac{1}{n}\sum_{i=1}^{n} \big(f_\theta(x_i) - y_i\big)^2" />
      <Row>
        <Plot x={[0, 4.5]} y={[-1, 11]} xl="x (input)" yl="y (target)" label="data points with the current line and residuals">
          <Mark as="stems" tone="muted" data={xs.map((x, i) => [x, ys[i], w * x + b])} />
          <Mark tone={1} data={[[0, b], [4.5, w * 4.5 + b]]} />
          <Mark as="points" tone="fg" data={xs.map((x, i) => [x, ys[i]])} />
          <Mark as="handles" lockX data={[[0, b, 1], [4.5, w * 4.5 + b, 1]]} onDrag={(i, _, y) => { if (i) setW((y - b) / 4.5); else { setW((w * 4.5 + b - y) / 4.5); setB(y); } }} />
        </Plot>
        <Stack>
          <Slider label={<M t="w" />} value={w} min={-3} max={4} step={0.05} onChange={setW} show={v => f(v)} />
          <Slider label={<M t="b" />} value={b} min={-3} max={11} step={0.05} onChange={setB} show={v => f(v)} />
          <Stat big><M t={String.raw`\hat R(\theta) = ${f(risk, 3)}`} /></Stat>
          <Stat>Lowest possible: <M t={String.raw`\min_\theta \hat R = ${f(bestRisk, 3)}`} /></Stat>
          <Button onClick={() => { setW(+best.w.toFixed(2)); setB(+best.b.toFixed(2)); }}>
            Jump to <M t="\hat\theta = \arg\min_\theta \hat R(\theta)" />
          </Button>
          <Note>Each grey segment is a residual <M t="r_i = y_i - \hat y_i" />. <M t="\hat R" /> is the average of their squares.</Note>
        </Stack>
      </Row>
      <Takeaway>Training = <M t="\arg\min" /> over the knobs. But a small <M t="\hat R" /> is only a promise about <em>these 8 points</em>; the
        rest of the lecture is about when that promise fails to carry over to <M t="R" />.</Takeaway>
    </>
  );
}

/* ---------- 2. Simpson's paradox ---------- */

const SIMPSON_NOISE = (() => {
  const r = rng(3);
  return Array.from({ length: 24 }, (_, i) => ({ g: i < 12 ? 'A' : 'B', u: r(), e: 0.3 * gauss(r) }));
})();

export function Simpson() {
  const [wa, setWa] = useState(263 / 350), [wb, setWb] = useState(80 / 350), [gap, setGap] = useState(2);
  const rate = { A: [93.1, 73.0], B: [86.7, 68.8] };
  const overall = (r, w) => (1 - w) * r[0] + w * r[1];
  const oa = overall(rate.A, wa), ob = overall(rate.B, wb);
  const pts = SIMPSON_NOISE.map(({ g, u, e }) =>
    g === 'A' ? { g, x: 1 + 3 * u, y: 5 + 0.6 * 3 * u + e } : { g, x: 5 + 3 * u, y: 5 - gap + 0.6 * 3 * u + e });
  const fit = g => { const s = pts.filter(p => !g || p.g === g); return linfit(s.map(p => p.x), s.map(p => p.y)); };
  const A = fit('A'), B = fit('B'), all = fit();
  const group = g => pts.filter(p => p.g === g).map(p => [p.x, p.y]);
  return (
    <>
      <h4>Part 1 — the kidney-stone numbers are weighted averages</h4>
      <p>Each treatment's overall success rate is a <b>weighted average</b> of its two group rates, weighted by the share of its patients in each group:</p>
      <M d t="\text{overall} = (1-w_{\text{large}})\cdot\text{rate}_{\text{small}} + w_{\text{large}}\cdot\text{rate}_{\text{large}}" />
      <Row>
        <Stack>
          <Slider label="A: share with large stones" value={wa} min={0} max={1} step={0.01} onChange={setWa} show={v => pct(v, 0)} />
          <Slider label="B: share with large stones" value={wb} min={0} max={1} step={0.01} onChange={setWb} show={v => pct(v, 0)} />
          <Button onClick={() => { setWa(263 / 350); setWb(80 / 350); }}>Reset to the real study</Button>
        </Stack>
        <Stack>
          <Table>
            <thead><tr><th /><th>Small</th><th>Large</th><th>Overall</th></tr></thead>
            <tbody>
              <tr><th>A</th><td className="win">93.1%</td><td className="win">73.0%</td><td className={oa > ob ? 'win' : ''}>{f(oa, 1)}%</td></tr>
              <tr><th>B</th><td>86.7%</td><td>68.8%</td><td className={ob > oa ? 'win' : ''}>{f(ob, 1)}%</td></tr>
            </tbody>
          </Table>
          <Status tone={oa < ob ? 'alert' : 'ok'}>
            {oa < ob ? 'Reversal! A wins in every group but loses overall.' : 'No reversal: the mixes are similar enough.'}
          </Status>
        </Stack>
      </Row>
      <h4>Part 2 — the same thing with trend lines (slide 14)</h4>
      <p>Within each group <M t="y" /> rises with <M t="x" />. Slide the gap between the groups and watch the single pooled line.</p>
      <Row>
        <Plot x={[0, 9]} y={[0, 9]} xl="feature x" yl="outcome y" label="two groups with within-group and pooled trend lines">
          <Mark as="points" soft tone={1} data={group('A')} />
          <Mark as="points" soft tone={2} data={group('B')} />
          <Mark tone={1} data={[[1, A.b + A.w], [4, A.b + 4 * A.w]]} />
          <Mark tone={2} data={[[5, B.b + 5 * B.w], [8, B.b + 8 * B.w]]} />
          <Mark tone={4} dash data={[[0.5, all.b + 0.5 * all.w], [8.5, all.b + 8.5 * all.w]]} />
        </Plot>
        <Stack>
          <Slider label="How far B sits below A" value={gap} min={0} max={4} step={0.05} onChange={setGap} show={v => f(v)} />
          <Stat>slope inside A: <b>{f(A.w)}</b></Stat>
          <Stat>slope inside B: <b>{f(B.w)}</b></Stat>
          <Stat>pooled slope: <b className={all.w < 0 ? 'neg' : ''}>{f(all.w)}</b></Stat>
          <Legend items={[['--c1', 'Group A'], ['--c2', 'Group B'], ['--c4', 'Pooled fit']]} />
        </Stack>
      </Row>
      <Takeaway>By total probability, <M t="p(Y\mid X)=\sum_g p(Y\mid X,g)\,p(g\mid X)" />. The weights <M t="p(g\mid X)" /> — who ended
        up in which group — can overpower the within-group effect and flip its sign. Always ask: does a third variable <M t="G" /> change the story?</Takeaway>
    </>
  );
}

/* ---------- 3. Conditional probability on the rain/flood table ---------- */

export function Conditional() {
  const [c, setC] = useState({ hf: 8, hn: 12, lf: 2, ln: 78 });
  const [given, setGiven] = useState('heavy');
  const N = c.hf + c.hn + c.lf + c.ln;
  const rows = { heavy: ['hf', 'hn'], light: ['lf', 'ln'] }, cols = { flood: ['hf', 'lf'], noflood: ['hn', 'ln'] };
  const universe = rows[given] || cols[given] || ['hf', 'hn', 'lf', 'ln'];
  const U = sum(universe.map(k => c[k]));
  const name = { heavy: '\\text{heavy}', light: '\\text{light}', flood: '\\text{flood}', noflood: '\\text{no flood}' };
  const pFlood = (c.hf + c.lf) / N, pHeavy = (c.hf + c.hn) / N, pFloodHeavy = c.hf / (c.hf + c.hn);
  const cell = k => (
    <td className={universe.includes(k) ? 'in' : 'out'}>
      <Field type="number" min={0} value={c[k]} aria-label={k} onChange={e => setC({ ...c, [k]: Math.max(0, +e.target.value || 0) })} />
    </td>
  );
  const answers = rows[given]
    ? [['\\text{flood}', c[rows[given][0]]], ['\\text{no flood}', c[rows[given][1]]]]
    : cols[given] ? [['\\text{heavy}', c[cols[given][0]]], ['\\text{light}', c[cols[given][1]]]] : null;
  return (
    <>
      <p>Conditioning on <M t="B" /> means <b>throwing away every outcome where <M t="B" /> didn't happen</b> and counting again inside what's left:</p>
      <M d t="P(A\mid B) = \frac{P(A,B)}{P(B)} = \frac{\#(A \text{ and } B)}{\#B}" />
      <Toolbar>
        <Choice label="Condition on" value={given} onChange={setGiven}
          options={['none', 'heavy', 'light', 'flood', 'noflood'].map(g => ({ value: g, label: g === 'none' ? 'no condition' : 'given ' + g.replace('noflood', 'no flood') }))} />
      </Toolbar>
      <Row>
        <Table className="cond">
          <thead><tr><th>{N} days</th><th>Flood</th><th>No flood</th><th>Total</th></tr></thead>
          <tbody>
            <tr><th>Heavy rain</th>{cell('hf')}{cell('hn')}<td>{c.hf + c.hn}</td></tr>
            <tr><th>Light rain</th>{cell('lf')}{cell('ln')}<td>{c.lf + c.ln}</td></tr>
            <tr><th>Total</th><td>{c.hf + c.lf}</td><td>{c.hn + c.ln}</td><td>{N}</td></tr>
          </tbody>
        </Table>
        <Stack>
          <Note>The highlighted cells are your new universe ({U} days). Edit any count.</Note>
          {answers ? answers.map(([a, n]) => (
            <Stat key={a}><M t={String.raw`P(${a}\mid ${name[given]}) = \frac{${n}}{${U}} = ${f(n / U, 3)}`} /></Stat>
          )) : (
            <>
              <Stat><M t={String.raw`P(\text{heavy},\text{flood}) = \frac{${c.hf}}{${N}} = ${f(c.hf / N, 3)}`} /></Stat>
              <Stat><M t={String.raw`P(\text{flood}) = \frac{${c.hf}+${c.lf}}{${N}} = ${f(pFlood, 3)}`} /> (marginal)</Stat>
            </>
          )}
        </Stack>
      </Row>
      <Stat>Independence test: <M t={String.raw`P(\text{flood}\mid\text{heavy}) = ${f(pFloodHeavy, 3)}`} /> vs <M t={String.raw`P(\text{flood}) = ${f(pFlood, 3)}`} /> →{' '}
        {Math.abs(pFloodHeavy - pFlood) < 0.005 ? <Status as="b" tone="ok">independent: rain tells you nothing</Status> : <b>dependent: information changes probability</b>}
        <Note as="span"> (equivalently <M t={String.raw`P(\text{heavy},\text{flood})=${f(c.hf / N, 3)}`} /> vs <M t={String.raw`P(\text{heavy})P(\text{flood})=${f(pHeavy * pFlood, 3)}`} />)</Note>
      </Stat>
      <Takeaway>Try "given heavy" then "given flood": <M t="P(\text{flood}\mid\text{heavy})" /> and <M t="P(\text{heavy}\mid\text{flood})" /> use
        different denominators, so they're different numbers. A predictive model estimates <M t="P(y\mid x)" /> — exactly this "shrink the universe" step.</Takeaway>
    </>
  );
}

/* ---------- 4. Accuracy paradox + Bayes base rate ---------- */

export function RareEvents() {
  const [prev, setPrev] = useState(0.01), [rec, setRec] = useState(0), [fpr, setFpr] = useState(0);
  const N = 1000, P = Math.round(N * prev), TP = Math.round(P * rec), FN = P - TP, FP = Math.round((N - P) * fpr), TN = N - P - FP;
  const acc = (TP + TN) / N, recall = P ? TP / P : NaN, precision = TP + FP ? TP / (TP + FP) : NaN;
  const bayes = (rec * prev) / (rec * prev + fpr * (1 - prev));
  const cells = [...Array(TP).fill('tp'), ...Array(FN).fill('fn'), ...Array(FP).fill('fp'), ...Array(TN).fill('tn')];
  const preset = (a, b, c) => { setPrev(a); setRec(b); setFpr(c); };
  return (
    <>
      <p>Every yes/no prediction lands in one of four boxes. Each metric is a fraction with a <b>different denominator</b> — a different conditional probability:</p>
      <M d t="\text{Accuracy}=\tfrac{TP+TN}{N},\quad \text{Recall}=\tfrac{TP}{TP+FN}=P(\hat y{=}1\mid y{=}1),\quad \text{Precision}=\tfrac{TP}{TP+FP}=P(y{=}1\mid \hat y{=}1)" />
      <Toolbar>
        <Button onClick={() => preset(0.01, 0, 0)}>Slide 16: always predict 0</Button>
        <Button onClick={() => preset(0.01, 0.99, 0.05)}>Medical test (§8.8)</Button>
        <Button onClick={() => preset(0.001, 0.95, 0.02)}>Fraud detector (Practice Q5)</Button>
      </Toolbar>
      <Row>
        <Stack>
          <Slider label="Base rate: share that are positive" value={prev} min={0.001} max={0.5} step={0.001} onChange={setPrev} show={v => pct(v)} />
          <Slider label="Recall: positives caught" value={rec} min={0} max={1} step={0.01} onChange={setRec} show={v => pct(v, 0)} />
          <Slider label="False-alarm rate on negatives" value={fpr} min={0} max={0.3} step={0.005} onChange={setFpr} show={v => pct(v)} />
          <Table className="cm">
            <thead><tr><th>1000 cases</th><th>pred 1</th><th>pred 0</th></tr></thead>
            <tbody>
              <tr><th>actual 1</th><td className="tp">TP {TP}</td><td className="fn">FN {FN}</td></tr>
              <tr><th>actual 0</th><td className="fp">FP {FP}</td><td className="tn">TN {TN}</td></tr>
            </tbody>
          </Table>
          <Stat big>Accuracy {pct(acc)}</Stat>
          <Stat>Recall {pct(recall)} · Precision {pct(precision)}</Stat>
        </Stack>
        <Stack>
          <DotGrid cells={cells} label="1000 cases coloured by confusion-matrix box" />
          <Legend items={[['--c3', 'TP caught'], ['--c4', 'FN missed'], ['--c2', 'FP false alarm'], ['--muted2', 'TN']]} />
          <Note>Precision is Bayes' rule in disguise — the chance a flagged case is really positive:</Note>
          <M t={String.raw`P(y{=}1\mid +)=\frac{${f(rec)}\times${f(prev, 3)}}{${f(rec)}\times${f(prev, 3)}+${f(fpr, 3)}\times${f(1 - prev, 3)}}=${Number.isFinite(bayes) ? f(bayes, 3) : '\\text{undefined}'}`} />
        </Stack>
      </Row>
      <Takeaway>When positives are rare the majority class owns almost all of accuracy's weight, so "always say 0" scores 99% while catching nothing.
        And even a good test drowns in false alarms when the base rate is tiny. <b>The metric must match the decision.</b></Takeaway>
    </>
  );
}

/* ---------- 5. Shortcut learning ---------- */

const SHORTCUT = (() => {
  const r = rng(11);
  return Array.from({ length: 200 }, (_, i) => {
    const y = i % 2, s = y ? 1 : -1;
    return { y, x1: 0.8 * s + gauss(r), x1d: 0.8 * s + gauss(r), m: 0.3 + 1.2 * r(), md: 0.3 + 1.2 * r(), u: r() };
  });
})();

export function Shortcut() {
  const [rho, setRho] = useState(0);
  const train = SHORTCUT.map(p => ({ y: p.y, x1: p.x1, x2: (p.y ? 1 : -1) * p.m }));
  const deploy = SHORTCUT.map(p => {
    const agrees = p.u < 0.5 + 0.5 * rho, s = p.y ? 1 : -1; // rho=0: sign of x2 is a coin flip; rho=1: same as training
    return { y: p.y, x1: p.x1d, x2: (agrees ? s : -s) * p.md };
  });
  const acc = (d, k) => mean(d.map(p => +((p[k] > 0) === !!p.y)));
  const panel = (d, title) => (
    <Plot caption={title} x={[-3.5, 3.5]} y={[-2, 2]} w={300} h={220} xl="x₁ (real signal)" yl="x₂ (spurious)" label={title}>
      <Mark as="points" soft tone={1} r={3} data={d.filter(p => p.y).map(p => [p.x1, p.x2])} />
      <Mark as="points" soft tone={2} r={3} data={d.filter(p => !p.y).map(p => [p.x1, p.x2])} />
    </Plot>
  );
  return (
    <>
      <p>Two features: <M t="x_1" /> is the real (noisy) signal, <M t="x_2" /> is a spurious one — like the snow behind every wolf photo. In training,
        <M t="x_2" /> separates the classes <em>perfectly</em>, so the easiest rule to learn is <M t="\hat y = \mathbb 1\{x_2 > 0\}" />.</p>
      <Row>{panel(train, 'Training world')}{panel(deploy, 'Deployment world')}</Row>
      <Slider label="How much x₂ still tracks the label at deployment" value={rho} min={0} max={1} step={0.05} onChange={setRho} show={v => pct(v, 0)} />
      <Table>
        <thead><tr><th>Rule</th><th>Training accuracy</th><th>Deployment accuracy</th></tr></thead>
        <tbody>
          <tr><th>Shortcut: <M t="x_2 > 0" /></th><td className="win">{pct(acc(train, 'x2'))}</td><td className={acc(deploy, 'x2') < 0.6 ? 'neg' : ''}>{pct(acc(deploy, 'x2'))}</td></tr>
          <tr><th>Real signal: <M t="x_1 > 0" /></th><td>{pct(acc(train, 'x1'))}</td><td>{pct(acc(deploy, 'x1'))}</td></tr>
        </tbody>
      </Table>
      <Legend items={[['--c1', 'class 1'], ['--c2', 'class 0']]} />
      <Takeaway>The training data rewarded the shortcut, so <M t="\hat R" /> looked perfect. But the pattern lived in <M t="\mathcal D_{\text{train}}" />, not
        in <M t="\mathcal D_{\text{deploy}}" />. Ask of every feature: <em>will it still be available, valid and causal at deployment time?</em></Takeaway>
    </>
  );
}

/* ---------- 6. Capacity and overfitting ---------- */

function makeCapacityData(seed) {
  const r = rng(seed), truth = x => Math.sin(Math.PI * x);
  const xs = Array.from({ length: 10 }, (_, i) => -1 + (2 * i) / 9 + (i % 9 ? 0.06 * (r() - 0.5) : 0));
  const ys = xs.map(x => truth(x) + 0.3 * gauss(r));
  const tx = Array.from({ length: 200 }, () => 2 * r() - 1), ty = tx.map(x => truth(x) + 0.3 * gauss(r));
  const fits = Array.from({ length: 10 }, (_, d) => {
    const c = polyfit(xs, ys, d), g = x => polyval(c, x);
    return { c, train: mse(g, xs, ys), test: mse(g, tx, ty) };
  });
  return { xs, ys, truth, fits };
}

export function Capacity() {
  const [seed, setSeed] = useState(5), [deg, setDeg] = useState(3);
  const { xs, ys, truth, fits } = useMemo(() => makeCapacityData(seed), [seed]);
  const fit = fits[deg], grid = Array.from({ length: 201 }, (_, i) => -1 + i / 100);
  const lg = v => Math.max(-4, Math.log10(v));
  return (
    <>
      <p>A polynomial of degree <M t="d" /> has <M t="d+1" /> knobs: <M t="f_\theta(x)=\theta_0+\theta_1x+\dots+\theta_dx^d" />. With 10 points, degree 9
        (10 knobs) can pass through every point exactly — the <b>interpolation threshold</b>. Training picks <M t="\hat\theta=\arg\min_\theta \hat R(\theta)" />; we then
        check on 200 fresh points from the same world.</p>
      <Row>
        <Plot x={[-1.05, 1.05]} y={[-2.2, 2.2]} xl="x" yl="y" label="training points, true curve and fitted polynomial">
          <Mark tone={3} dash data={grid.map(x => [x, truth(x)])} />
          <Mark tone={1} data={grid.map(x => [x, polyval(fit.c, x)])} />
          <Mark as="points" tone="fg" data={xs.map((x, i) => [x, ys[i]])} />
        </Plot>
        <Plot x={[0, 9]} y={[-4, 1]} xl="degree (capacity)" yl="log₁₀ error" label="train and test error versus degree">
          <Mark as="rule" dash x={deg} />
          <Mark tone={3} data={fits.map((q, d) => [d, lg(q.train)])} />
          <Mark tone={4} data={fits.map((q, d) => [d, lg(q.test)])} />
          <Mark as="points" tone={4} r={3} data={fits.map((q, d) => [d, lg(q.test)])} />
        </Plot>
      </Row>
      <Row>
        <Stack>
          <Slider label="Polynomial degree" value={deg} min={0} max={9} step={1} onChange={setDeg} />
          <Button onClick={() => setSeed(s => s + 1)}>New noisy sample</Button>
        </Stack>
        <Stack>
          <Stat>training error <M t={String.raw`\hat R = ${fit.train < 1e-6 ? '\\approx 0' : f(fit.train, 4)}`} /></Stat>
          <Stat>test error <M t={String.raw`\approx R = ${f(fit.test, 4)}`} /></Stat>
          <Status as="div" tone={deg <= 1 || deg >= 8 ? 'alert' : 'ok'}>{deg <= 1 ? 'Underfit: too stiff for the curve.' : deg >= 8 ? 'Overfit: chasing the noise between points.' : 'Good fit: bends like the real curve.'}</Status>
          <Legend items={[['--c1', 'fitted f'], ['--c3', 'true curve / train error'], ['--c4', 'test error']]} />
        </Stack>
      </Row>
      <Takeaway>Training error only ever goes down as capacity grows; test error makes the classical U. Slide 20 adds a twist — far past the threshold
        (many more knobs than points), training methods pick the smoothest interpolating curve and test error can <em>fall again</em> (double descent).
        Neither "small is safe" nor "big is better" is a law — only held-out data tells you where you are.</Takeaway>
    </>
  );
}

/* ---------- 7. Loss functions ---------- */

export function Loss() {
  const [y, setY] = useState(4), [yh, setYh] = useState(7), [p, setP] = useState(0.9);
  const r = y - yh, grid = Array.from({ length: 201 }, (_, i) => i / 20), pg = Array.from({ length: 200 }, (_, i) => 0.005 + i / 201);
  return (
    <>
      <p>The residual <M t="r = y-\hat y" /> can be negative, so we can't just average it (+3 and −3 would cancel). A <b>loss</b> <M t="\ell(\hat y, y)\ge 0" /> turns
        each miss into a cost — and <b>choosing it is choosing which mistakes matter</b>.</p>
      <Row>
        <Plot x={[0, 10]} y={[0, 10]} xl="prediction ŷ" yl="loss" label="squared, absolute and 0-1 loss versus prediction">
          <Mark tone={1} data={grid.map(v => [v, (v - y) ** 2])} />
          <Mark tone={2} data={grid.map(v => [v, Math.abs(v - y)])} />
          <Mark tone={3} data={grid.map(v => [v, Math.abs(v - y) < 0.025 ? 0 : 1])} />
          <Mark as="rule" dash x={yh} />
        </Plot>
        <Stack>
          <Slider label={<>truth <M t="y" /></>} value={y} min={0} max={10} step={0.5} onChange={setY} />
          <Slider label={<>prediction <M t="\hat y" /></>} value={yh} min={0} max={10} step={0.5} onChange={setYh} />
          <Stat><M t={String.raw`(\hat y-y)^2 = ${f(r * r)}`} /> <Note as="span">squared — big misses cost a lot</Note></Stat>
          <Stat><M t={String.raw`|\hat y-y| = ${f(Math.abs(r))}`} /> <Note as="span">absolute — grows steadily</Note></Stat>
          <Stat><M t={String.raw`\mathbb 1\{\hat y\ne y\} = ${r === 0 ? 0 : 1}`} /> <Note as="span">0-1 — only right/wrong</Note></Stat>
          <Legend items={[['--c1', 'squared'], ['--c2', 'absolute'], ['--c3', '0-1']]} />
        </Stack>
      </Row>
      <h4>Log loss: surprise at the true answer</h4>
      <p>A probabilistic classifier gives the true class probability <M t="\hat p_y" />. Its loss is <M t="-\log \hat p_y" />, where <M t="\log" /> is the natural log (the inverse of <M t="e^z" />).</p>
      <Row>
        <Plot x={[0, 1]} y={[0, 5]} xl="probability given to the true class" yl="−log p" label="log loss curve">
          <Mark tone={4} data={pg.map(v => [v, -Math.log(v)])} />
          <Mark as="points" tone={4} r={5} data={[[p, -Math.log(p)]]} />
        </Plot>
        <Stack>
          <Slider label={<M t="\hat p_y" />} value={p} min={0.01} max={1} step={0.01} onChange={setP} />
          <Stat big><M t={String.raw`-\log(${f(p)}) = ${f(-Math.log(p), 3)}`} /></Stat>
          <Note>{p > 0.8 ? '"I knew it" — tiny penalty.' : p > 0.4 ? 'Unsure — moderate penalty.' : 'Confident and wrong — the penalty explodes toward ∞ as p → 0.'}</Note>
          <Note>Logs turn products into sums: <M t="\log(ab)=\log a+\log b" />, and since <M t="\log" /> is increasing, maximising <M t="P" /> is the same as maximising <M t="\log P" />.</Note>
        </Stack>
      </Row>
    </>
  );
}

/* ---------- 8. min vs argmin ---------- */

export function Argmin() {
  const [text, setText] = useState('2, 4, 9'), [c, setC] = useState(3);
  const xs = text.split(/[\s,]+/).filter(Boolean).map(Number).filter(Number.isFinite);
  const ok = xs.length > 0;
  const lo = ok ? Math.min(...xs) - 2 : 0, hi = ok ? Math.max(...xs) + 2 : 10;
  const Lsq = v => sum(xs.map(x => (v - x) ** 2)), Labs = v => sum(xs.map(x => Math.abs(v - x)));
  const grid = Array.from({ length: 201 }, (_, i) => lo + ((hi - lo) * i) / 200);
  const m = ok ? mean(xs) : 0, md = ok ? median(xs) : 0;
  const cc = Math.min(hi, Math.max(lo, c));
  const ymax = ok ? Math.max(Lsq(lo), Lsq(hi)) : 1, yabs = ok ? Math.max(Labs(lo), Labs(hi)) : 1;
  return (
    <>
      <p>Which single number <M t="c" /> is "closest" to all your measurements? It depends on the loss. <M t="\min" /> is the lowest <em>height</em> of the curve;
        <M t="\arg\min" /> is <em>where</em> it happens. Training wants the argmin — the knob setting.</p>
      <M d t={String.raw`L_{\text{sq}}(c)=\sum_i (c-x_i)^2 = nc^2-2\Big(\sum_i x_i\Big)c+\sum_i x_i^2 \;\Rightarrow\; \arg\min_c = -\tfrac{b}{2a} = \tfrac{1}{n}\sum_i x_i = \bar x`} />
      <label className="control"><span>Your data</span><Field value={text} onChange={e => setText(e.target.value)} aria-label="data values" /></label>
      {ok && (
        <>
          <Slider label={<M t="c" />} value={cc} min={+lo.toFixed(1)} max={+hi.toFixed(1)} step={0.05} onChange={setC} show={v => f(v)} />
          <Row>
            <Plot caption={<>Squared: <M t={String.raw`L_{\text{sq}}(${f(cc)}) = ${f(Lsq(cc))}`} /></>} x={[+lo.toFixed(1), +hi.toFixed(1)]} y={[0, +ymax.toFixed(0)]} w={340} h={220} xl="c" label="total squared loss">
              <Mark tone={1} data={grid.map(v => [v, Lsq(v)])} />
              <Mark as="rule" dash x={m} />
              <Mark as="points" tone={4} r={5} data={[[cc, Lsq(cc)]]} />
            </Plot>
            <Plot caption={<>Absolute: <M t={String.raw`L_{\text{abs}}(${f(cc)}) = ${f(Labs(cc))}`} /></>} x={[+lo.toFixed(1), +hi.toFixed(1)]} y={[0, +yabs.toFixed(0)]} w={340} h={220} xl="c" label="total absolute loss">
              <Mark tone={2} data={grid.map(v => [v, Labs(v)])} />
              <Mark as="rule" dash x={md} />
              <Mark as="points" tone={4} r={5} data={[[cc, Labs(cc)]]} />
            </Plot>
          </Row>
          <Stat>Squared: <M t={String.raw`\arg\min = \bar x = ${f(m)},\ \min = ${f(Lsq(m))}`} /> · Absolute: <M t={String.raw`\arg\min = \text{median} = ${f(md)},\ \min = ${f(Labs(md))}`} /></Stat>
          <Takeaway>Add an outlier like 100 to your data: the mean (squared loss) jumps, the median (absolute loss) barely moves. Squared loss punishes big misses, so one far point drags the answer toward it.</Takeaway>
        </>
      )}
    </>
  );
}

/* ---------- 9. Law of large numbers ---------- */

export function LLN() {
  const [rolls, setRolls] = useState([]);
  const roll = k => setRolls(r => [...r, ...Array.from({ length: k }, () => 1 + Math.floor(6 * Math.random()))].slice(0, 20000));
  const n = rolls.length;
  const run = useMemo(() => { let s = 0; return rolls.map((v, i) => (s += v) / (i + 1)); }, [rolls]);
  const step = Math.max(1, Math.floor(n / 400)), pts = run.map((v, i) => [i + 1, v]).filter((_, i) => i % step === 0 || i === n - 1);
  const counts = [1, 2, 3, 4, 5, 6].map(k => rolls.filter(v => v === k).length);
  return (
    <>
      <p>The <b>expectation</b> is a probability-weighted average of all possible values — computable only if you know the distribution:</p>
      <M d t="\mathbb E[X]=\sum_x x\,p(x) = 1\cdot\tfrac16+2\cdot\tfrac16+\dots+6\cdot\tfrac16 = 3.5" />
      <p>The <b>sample mean</b> <M t="\frac1n\sum_{i=1}^n x_i" /> needs only the rolls you've seen. The law of large numbers says it settles on <M t="\mathbb E[X]" /> as <M t="n" /> grows.</p>
      <Toolbar>
        {[1, 10, 100, 1000].map(k => <Button key={k} onClick={() => roll(k)}>Roll {k}</Button>)}
        <Button onClick={() => setRolls([])}>Reset</Button>
      </Toolbar>
      <Row>
        <Plot x={[1, Math.max(10, n)]} y={[1, 6]} xl="number of rolls n" yl="running average" label="running average of die rolls">
          <Mark as="rule" tone={3} dash y={3.5} />
          {n > 0 && <Mark tone={1} data={pts} />}
        </Plot>
        <Stack>
          <Stat big>n = {n} · sample mean = {n ? f(run[n - 1], 3) : '—'}</Stat>
          <Stat>gap to <M t="\mathbb E[X]=3.5" />: {n ? f(Math.abs(run[n - 1] - 3.5), 3) : '—'}</Stat>
          <Bars data={counts.map((k, i) => ({ label: i + 1, title: `${i + 1}: ${k}`, height: n ? (300 * k) / n : 0 }))} />
          <Note>Each face's bar approaches 1/6 of the rolls.</Note>
        </Stack>
      </Row>
      <Takeaway>Swap "die roll" for "loss on one example": the true risk <M t="R(f)=\mathbb E[\ell(f(X),Y)]" /> is the expectation, the empirical
        risk <M t="\hat R(f)" /> is the sample mean. The catch: we <em>choose</em> <M t="f" /> to make <M t="\hat R" /> small on these very examples, so the law of large numbers alone
        doesn't protect us — that gap is overfitting.</Takeaway>
    </>
  );
}

/* ---------- 10. Double descent ---------- */

export function DoubleDescent() {
  const [p, setP] = useState(6);
  const d = useMemo(() => doubleDescent(), []);
  const sweep = useMemo(() => [1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 22, 28, 40, 60, 90, 140, 240].map(k => d.fit(k)), [d]);
  const cur = d.fit(p), lg = v => Math.min(1.5, Math.max(-3, Math.log10(v))), lx = Math.log10;
  return (
    <>
      <p>The classical U-curve ends at the <b>interpolation threshold</b>, where the model has just enough knobs to hit all {d.n} points. Keep adding knobs. Among all perfect fits, this method picks the one with the smallest weights, and test error <b>falls a second time</b>.</p>
      <M d t="\hat\theta=\arg\min\|\theta\|\ \text{ subject to }\ \Phi\theta=y\quad(p>n)" />
      <Row>
        <Plot caption={`Fit with ${p} random features`} x={[-1, 1]} y={[-2.5, 2.5]} w={340} h={260} xl="x" yl="y" label="fitted curve">
          <Mark tone={3} dash data={d.grid.map(x => [x, d.truth(x)])} />
          <Mark tone={1} data={d.grid.map(x => [x, Math.max(-4, Math.min(4, cur.g(x)))])} />
          <Mark as="points" tone="fg" data={d.xs.map((x, i) => [x, d.ys[i]])} />
        </Plot>
        <Plot caption="Error against capacity" x={[0, 2.4]} y={[-3, 1.5]} w={340} h={260} xl="log₁₀ features p" yl="log₁₀ error" label="train and test error against number of features">
          <Mark tone={3} data={sweep.map(s => [lx(s.p), lg(s.train)])} />
          <Mark tone={4} data={sweep.map(s => [lx(s.p), lg(s.test)])} />
          <Mark as="rule" dash x={lx(d.n)} />
          <Mark as="handles" lockY data={[[lx(p), 1.2]]} onDrag={(_, x) => setP(Math.min(240, Math.max(1, Math.round(10 ** x))))} />
        </Plot>
      </Row>
      <Slider label="features p" value={p} min={1} max={240} step={1} onChange={setP} />
      <Stat big>test error {f(cur.test, 3)} · weight norm {f(cur.norm, 2)}</Stat>
      <Takeaway>At p = n exactly one solution fits, and it has to contort. Beyond that, many fit and the smallest is smooth. This does not overturn bias-variance: variance spikes at the threshold, then a benign selection rule brings it down.</Takeaway>
    </>
  );
}
