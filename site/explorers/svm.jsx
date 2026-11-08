'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Mark, Plot, moved } from '../components/Plot';
import { Button, Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Takeaway, Toolbar } from '../components/ui';
import { clamp, dataset, fmt, kernels, linspace, smo } from '../lib/models.mjs';

/* ---------- distance from a point to a line ---------- */

export function Distance() {
  const [w, setW] = useState([1, 2]), [b, setB] = useState(-1), [p, setP] = useState([2, 2]), [scale, setScale] = useState(1);
  const n = Math.hypot(...w) || 1, g = (w[0] * p[0] + w[1] * p[1] + b) / n, foot = [p[0] - (g * w[0]) / n, p[1] - (g * w[1]) / n];
  const line = Math.abs(w[1]) > 1e-3 ? [-4, 4].map(x => [x, (-b - w[0] * x) / w[1]]) : [[-b / w[0], -4], [-b / w[0], 4]];
  const raw = scale * (w[0] * p[0] + w[1] * p[1] + b);
  return (
    <>
      <p>The line <M t="w^\top x+b=0" />. Drag the normal <M t="w" /> and the point. Then scale <M t="(w,b)" />: <b>the line does not move</b> yet the raw score does, which is why the geometric margin divides by the norm.</p>
      <M d t="\text{dist}(x)=\frac{w^\top x+b}{\|w\|}" />
      <Row>
        <Plot x={[-4, 4]} y={[-4, 4]} w={380} h={360} xl="x₁" yl="x₂" label="line with normal vector and a point">
          <Mark as="field" fn={(x, y) => clamp((w[0] * x + w[1] * y + b) / (n * 4), -1, 1) * 0.6} />
          <Mark tone="fg" data={line} />
          <Mark as="arrow" tone={4} data={[[...foot, ...p]]} />
          <Mark as="arrow" tone={3} data={[[0, 0, ...w]]} />
          <Mark as="handles" data={[[...w, 3]]} onDrag={(_, x, y) => setW([x, y])} />
          <Mark as="handles" data={[p]} onDrag={(_, x, y) => setP([x, y])} />
        </Plot>
        <Stack>
          <Slider label={<M t="b" />} value={b} min={-6} max={6} step={0.1} onChange={setB} show={v => fmt(v, 1)} />
          <Slider label="scale (w, b) by" value={scale} min={0.1} max={10} step={0.1} onChange={setScale} show={v => fmt(v, 1)} />
          <Stat big>signed distance {fmt(g, 3)}</Stat>
          <Stat>raw score {fmt(raw, 3)}</Stat>
          <Note>Because scale is free, we can fix the closest point's raw score at 1 and maximise the margin by minimising ‖w‖.</Note>
        </Stack>
      </Row>
      <Takeaway>That scale choice turns "maximise the smallest distance" into a convex problem, the whole trick behind the SVM primal.</Takeaway>
    </>
  );
}

/* ---------- the full SVM lab ---------- */

const PRESETS = { blobs: 'Separable', overlap: 'Overlap', xor: 'XOR', circles: 'Circles' };
const load = (k, seed) => dataset(k, seed, 30).map(([a, b, c]) => [a, b, c + 1]);

export function MarginLab() {
  const [data, setData] = useState('overlap'), [seed, setSeed] = useState(1), [pts, setPts] = useState(() => load('overlap', 1));
  const [kind, setKind] = useState('linear'), [lc, setLc] = useState(0), [gamma, setGamma] = useState(1);
  const C = 10 ** lc, reset = (d, s) => { setData(d); setSeed(s); setPts(load(d, s)); };
  const model = useMemo(() => {
    const X = pts.map(p => [p[0], p[1]]), y = pts.map(p => (p[2] === 2 ? 1 : -1));
    return { y, ...smo(X, y, { C, k: kernels[kind]({ gamma, degree: 3 }) }) };
  }, [pts, C, kind, gamma]);
  const { alpha, f, y } = model;
  const onMargin = alpha.map(a => a > 1e-5 && a < C - 1e-5), bound = alpha.map(a => a >= C - 1e-5);
  const sv = alpha.map(a => a > 1e-5), correct = pts.filter((p, i) => f([p[0], p[1]]) * y[i] > 0).length;
  const field = (x, v) => clamp(f([x, v]), -1.4, 1.4) / 1.4;
  return (
    <>
      <p>This runs a real soft-margin SVM solver every time you move something. <b>Drag any point</b>: only the ringed support vectors change the answer, and dragging an unringed point well behind the margin leaves the boundary where it is.</p>
      <M d t="\min_{w,b,\xi}\ \tfrac12\|w\|^2+C\sum_i\xi_i\quad\text{s.t.}\ \ y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i" />
      <Toolbar>
        <Choice label="Data" value={data} onChange={d => reset(d, seed)} options={Object.entries(PRESETS).map(([value, label]) => ({ value, label }))} />
        <Choice label="Kernel" value={kind} onChange={setKind} options={[{ value: 'linear', label: 'Linear' }, { value: 'poly', label: 'Poly 3' }, { value: 'rbf', label: 'Gaussian' }]} />
        <Button onClick={() => reset(data, seed + 1)}>New sample</Button>
      </Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} w={420} h={400} xl="x₁" yl="x₂" label="SVM decision function with draggable points">
          <Mark as="field" nx={60} ny={48} fn={field} />
          <Mark as="contour" tone={3} dash levels={[-1, 1]} fn={(x, v) => f([x, v])} />
          <Mark as="contour" tone="fg" fn={(x, v) => f([x, v])} />
          <Mark as="rings" data={pts.filter((_, i) => sv[i])} />
          <Mark as="handles" r={5} data={pts} onDrag={(i, a, b) => setPts(moved(i, a, b))} />
        </Plot>
        <Stack>
          <Slider label={<M t="\log_{10}C" />} value={lc} min={-2} max={3} step={0.05} onChange={setLc} show={v => fmt(v)} />
          {kind === 'rbf' && <Slider label={<M t="\gamma" />} value={gamma} min={0.05} max={8} step={0.05} onChange={setGamma} show={v => fmt(v)} />}
          <Stat big>{sv.filter(Boolean).length} support vectors of {pts.length}</Stat>
          <Stat>{onMargin.filter(Boolean).length} exactly on the margin (0 &lt; α &lt; C)</Stat>
          <Stat>{bound.filter(Boolean).length} inside or wrong (α = C)</Stat>
          <Stat>training accuracy {fmt((100 * correct) / pts.length, 0)}%</Stat>
          <Legend items={[['--c1', 'score > 0'], ['--c2', 'score < 0'], ['--c3', 'margin ±1'], ['--fg', 'boundary']]} />
        </Stack>
      </Row>
      <Takeaway>C prices each violation: small C buys a wide margin that tolerates errors, large C insists on fitting. With a Gaussian kernel, large γ gives each point its own tiny island, which is overfitting you can see.</Takeaway>
    </>
  );
}

/* ---------- the kernel idea ---------- */

const ONE_D = [[-2.5, 0], [-2.1, 0], [-1.7, 0], [-1.2, 1], [-0.7, 1], [-0.2, 1], [0.3, 1], [0.8, 1], [1.3, 1], [1.8, 0], [2.2, 0], [2.6, 0]];

export function Lift() {
  const [lift, setLift] = useState(1), [cut, setCut] = useState(2.6);
  const wrong = ONE_D.filter(([x, c]) => (x * x < cut) !== !!c).length;
  return (
    <>
      <p>These points cannot be split by one threshold on a line. Add the feature <M t="x^2" /> and they can. Slide the lift, then move the horizontal cut.</p>
      <M d t="\phi(x)=(x,\ x^2),\qquad k(x,z)=\phi(x)^\top\phi(z)=xz+x^2z^2" />
      <Row>
        <Plot x={[-3, 3]} y={[-0.5, 9]} xl="x" yl="lifted feature" label="points lifted onto a parabola">
          {lift > 0.95 && <Mark as="field" fn={(_, y) => (y < cut ? 0.5 : -0.5)} />}
          <Mark tone="muted" dash data={linspace(-3, 3, 81).map(x => [x, lift * x * x])} />
          <Mark as="points" r={5} data={ONE_D.map(([x, c]) => [x, lift * x * x, c + 1])} />
          {lift > 0.95 && <Mark as="handles" lockX data={[[0, cut, 3]]} onDrag={(_, __, y) => setCut(y)} />}
        </Plot>
        <Stack>
          <Slider label="lift into the new dimension" value={lift} min={0} max={1} step={0.01} onChange={setLift} show={v => fmt(v)} />
          <Stat big>{lift > 0.95 ? `${wrong} misclassified by the cut` : 'Finish the lift to place a cut'}</Stat>
          <Status as="div" tone={lift > 0.95 && !wrong ? 'ok' : undefined}>{lift > 0.95 && !wrong ? 'Linearly separable in the new space.' : 'No single threshold on x alone can do this.'}</Status>
          <Note>The SVM dual only ever uses dot products, so a kernel supplies them without building φ, even when φ is infinite-dimensional.</Note>
        </Stack>
      </Row>
      <Takeaway>A straight cut in the lifted space is a curved boundary in the original. Here it is the pair of thresholds |x| = √cut.</Takeaway>
    </>
  );
}

/* ---------- four losses ---------- */

const LOSSES = [['0-1', m => (m <= 0 ? 1 : 0), 4], ['hinge', m => Math.max(0, 1 - m), 1], ['logistic', m => Math.log2(1 + Math.exp(-m)), 2], ['exponential', m => Math.exp(-m), 3]];

export function FourLosses() {
  const [m, setM] = useState(0.4), g = linspace(-2.5, 3, 221);
  return (
    <>
      <p>Everything is a function of the <b>margin</b> <M t="m=y\,f(x)" />: positive means right, large means confident. Drag the point and compare what each loss charges.</p>
      <Row>
        <Plot x={[-2.5, 3]} y={[0, 4]} xl="margin y·f(x)" yl="loss" label="four classification losses against margin">
          {LOSSES.map(([n, fn, t]) => <Mark key={n} tone={t} data={g.map(v => [v, Math.min(fn(v), 5)])} />)}
          <Mark as="rule" dash x={m} />
          <Mark as="handles" lockY data={[[m, 3.6]]} onDrag={(_, x) => setM(x)} />
        </Plot>
        <Stack>
          {LOSSES.map(([n, fn]) => <Stat key={n}>{n}: <b>{fmt(fn(m), 3)}</b></Stat>)}
          <Legend items={[['--c4', '0-1'], ['--c1', 'hinge'], ['--c2', 'logistic'], ['--c3', 'exponential']]} />
          <Note>Logistic is scaled by 1/log 2 so all the convex losses touch 0-1 at m = 0.</Note>
        </Stack>
      </Row>
      <Takeaway>Hinge is exactly zero past margin 1, so those points stop mattering, and that is where sparse support vectors come from. Exponential explodes on mistakes, which is AdaBoost's outlier problem.</Takeaway>
    </>
  );
}

/* ---------- duality on a one-variable toy ---------- */

export function Duality() {
  const [lam, setLam] = useState(0.8), g = linspace(-2, 3), ls = linspace(0, 4);
  const L = x => x * x + lam * (1 - x), xs = lam / 2, dual = l => l - (l * l) / 4;
  return (
    <>
      <p>Minimise <M t="x^2" /> subject to <M t="x\ge1" />, whose answer is 1. Price the constraint with <M t="\lambda" />, minimise freely, then <b>raise the price until the bound is tightest</b>.</p>
      <M d t="L(x,\lambda)=x^2+\lambda(1-x),\qquad g(\lambda)=\min_x L=\lambda-\tfrac{\lambda^2}{4}" />
      <Row>
        <Plot caption="Lagrangian in x at this price" x={[-2, 3]} y={[-1, 6]} w={340} h={260} xl="x" label="objective and Lagrangian">
          <Mark as="field" nx={50} ny={1} tones={[3, 3]} fn={x => (x >= 1 ? 0.35 : 0)} />
          <Mark tone="muted" dash data={g.map(x => [x, x * x])} />
          <Mark tone={1} data={g.map(x => [x, L(x)])} />
          <Mark as="points" tone={1} data={[[xs, L(xs)]]} />
        </Plot>
        <Plot caption="Dual: the bound as a function of price" x={[0, 4]} y={[-1, 1.5]} w={340} h={260} xl="λ" yl="g(λ)" label="dual function">
          <Mark as="rule" dash y={1} />
          <Mark tone={2} data={ls.map(l => [l, dual(l)])} />
          <Mark as="handles" data={[[lam, dual(lam), 2]]} lockY={false} onDrag={(_, x) => setLam(x)} />
        </Plot>
      </Row>
      <Stat big>g({fmt(lam)}) = {fmt(dual(lam), 3)} ≤ 1, free minimiser x = {fmt(xs, 3)}</Stat>
      <Status as="div" tone={Math.abs(lam - 2) < 0.08 ? 'ok' : undefined}>{Math.abs(lam - 2) < 0.08 ? 'Strong duality: at λ = 2, the bound meets the true optimum and x reaches the constraint.' : 'Every price gives a lower bound on the true minimum.'}</Status>
      <Takeaway>λ &gt; 0 only when the constraint binds. In an SVM that complementary slackness says: only points on or past the margin get a nonzero α.</Takeaway>
    </>
  );
}

