'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Mark, Plot, moved } from '../components/Plot';
import { Button, Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Takeaway, Toolbar } from '../components/ui';
import { mean } from '../lib/math.mjs';
import { betaPdf, binomPmf, clamp, fmt, linspace, normalCdf, normalPdf, pct, poissonPmf } from '../lib/models.mjs';

/* ---------- L02: derivative as a limit ---------- */

const CURVES = {
  cubic: { name: 'x³/4 − x', f: x => x ** 3 / 4 - x, d: x => 0.75 * x * x - 1 },
  wave: { name: 'sin x + x/2', f: x => Math.sin(x) + x / 2, d: x => Math.cos(x) + 0.5 },
  bowl: { name: 'x²/2', f: x => x * x / 2, d: x => x },
};

export function Tangent() {
  const [x, setX] = useState(1), [h, setH] = useState(1.5), [k, setK] = useState('cubic');
  const { f, d } = CURVES[k], g = linspace(-3, 3), sec = (f(x + h) - f(x)) / h;
  const through = (m, a = -3, b = 3) => [[a, f(x) + m * (a - x)], [b, f(x) + m * (b - x)]];
  return (
    <>
      <p>A derivative is a <b>rise over run with the run shrunk to nothing</b>. Drag the point along the curve, then shrink <M t="h" /> and watch the orange chord become the green tangent.</p>
      <M d t="f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}" />
      <Toolbar><Choice label="Curve" value={k} onChange={setK} options={Object.entries(CURVES).map(([v, c]) => ({ value: v, label: c.name }))} /></Toolbar>
      <Row>
        <Plot x={[-3, 3]} y={[-4, 4]} xl="x" yl="f(x)" label="curve with secant and tangent lines">
          <Mark tone={1} data={g.map(v => [v, f(v)])} />
          <Mark tone={3} dash data={through(d(x))} />
          <Mark tone={2} data={through(sec)} />
          <Mark as="points" tone={2} data={[[x + h, f(x + h)]]} />
          <Mark as="handles" lockY data={[[x, clamp(f(x), -4, 4)]]} onDrag={(_, v) => setX(v)} />
        </Plot>
        <Stack>
          <Slider label={<M t="h" />} value={h} min={0.01} max={2} step={0.01} onChange={setH} show={v => fmt(v)} />
          <Stat big>chord slope {fmt(sec, 3)}</Stat>
          <Stat>true slope <M t={`f'(${fmt(x)}) = ${fmt(d(x), 3)}`} /></Stat>
          <Stat>gap {fmt(Math.abs(sec - d(x)), 3)}</Stat>
          <Legend items={[['--c1', 'f'], ['--c2', 'chord'], ['--c3', 'tangent']]} />
        </Stack>
      </Row>
      <Takeaway>The sign of the slope says which way is uphill, so stepping against it goes down. That one fact drives every training loop in this course.</Takeaway>
    </>
  );
}

/* ---------- L02: gradient descent on a bowl ---------- */

export function Descent() {
  const [s, setS] = useState([2.6, 2.2]), [eta, setEta] = useState(0.3), [kappa, setKappa] = useState(5);
  const L = (x, y) => 0.5 * (x * x + kappa * y * y), max = L(3, 3);
  const path = useMemo(() => {
    const p = [[...s]];
    for (let i = 0; i < 40; i++) { const [x, y] = p[i], n = [x - eta * x, y - eta * kappa * y]; if (Math.hypot(...n) > 1e4) break; p.push(n); }
    return p;
  }, [s, eta, kappa]);
  const end = path[path.length - 1], blown = Math.abs(1 - eta * kappa) > 1;
  return (
    <>
      <p>Filled contours of <M t="L=\tfrac12(x^2+\kappa y^2)" />. The gradient is perpendicular to the contours and <b>does not point at the minimum</b> unless the bowl is round. Drag the start point.</p>
      <M d t="\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)" />
      <Row>
        <Plot x={[-3, 3]} y={[-3, 3]} h={330} xl="θ₁" yl="θ₂" label="loss contours with the descent path">
          <Mark as="field" bands={9} tones={[1, 1]} fn={(x, y) => Math.sqrt(clamp(L(x, y) / max, 0, 1))} />
          <Mark tone="fg" data={path} />
          <Mark as="points" tone="fg" r={2.5} data={path} />
          <Mark as="handles" data={[s]} onDrag={(_, x, y) => setS([x, y])} />
        </Plot>
        <Stack>
          <Slider label={<>step size <M t="\eta" /></>} value={eta} min={0.01} max={1.2} step={0.01} onChange={setEta} show={v => fmt(v)} />
          <Slider label={<>steepness <M t="\kappa" /></>} value={kappa} min={1} max={20} step={0.5} onChange={setKappa} show={v => fmt(v, 1)} />
          <Stat big>after 40 steps: L = {fmt(L(...end), 4)}</Stat>
          <Status as="div" tone={blown ? 'alert' : 'ok'}>{blown ? `Oscillating or diverging: η > 2/κ = ${fmt(2 / kappa, 3)} along the steep axis.` : `Stable: η is below the limit 2/κ = ${fmt(2 / kappa, 3)}.`}</Status>
          <Note>Each axis multiplies its error by <M t="1-\eta\mu" /> per step. The steepest axis caps <M t="\eta" />, and the flattest axis then crawls.</Note>
        </Stack>
      </Row>
      <Takeaway>Round bowls are easy and stretched ones are slow. Conditioning, feature scaling and Newton's method (Lecture 6) all exist to fix this stretch.</Takeaway>
    </>
  );
}

/* ---------- L02: convexity and local minima ---------- */

const SHAPES = {
  convex: { name: 'convex', f: x => 0.35 * x * x + 0.3, d: x => 0.7 * x },
  bumpy: { name: 'not convex', f: x => 0.18 * x * x + Math.sin(2 * x) + 1.5, d: x => 0.36 * x + 2 * Math.cos(2 * x) },
};

export function Convexity() {
  const [a, setA] = useState(-2.2), [b, setB] = useState(1.6), [k, setK] = useState('bumpy'), [start, setStart] = useState(2.8);
  const { f, d } = SHAPES[k], g = linspace(-4, 4);
  const inside = linspace(a, b, 41), broken = inside.some(x => a !== b && f(a) + ((f(b) - f(a)) * (x - a)) / (b - a) < f(x) - 1e-6);
  let x = start;
  const trail = [[x, f(x)]];
  for (let i = 0; i < 60; i++) { x -= 0.08 * d(x); trail.push([x, f(x)]); }
  return (
    <>
      <p>A function is convex when <b>every chord stays on or above the curve</b>. Drag the two chord ends looking for a violation, then drag the ball's start to see where descent lands.</p>
      <Toolbar><Choice label="Shape" value={k} onChange={setK} options={Object.entries(SHAPES).map(([v, c]) => ({ value: v, label: c.name }))} /></Toolbar>
      <Row>
        <Plot x={[-4, 4]} y={[0, 6]} xl="θ" yl="L(θ)" label="curve with a draggable chord and a descent ball">
          <Mark tone={1} data={g.map(v => [v, f(v)])} />
          <Mark tone={broken ? 4 : 3} data={[[a, f(a)], [b, f(b)]]} />
          <Mark as="points" tone={2} r={3} data={trail} />
          <Mark as="handles" lockY data={[[a, f(a)], [b, f(b)]]} onDrag={(i, v) => (i ? setB(v) : setA(v))} />
          <Mark as="handles" lockY data={[[start, f(start), 2]]} onDrag={(_, v) => setStart(v)} />
        </Plot>
        <Stack>
          <Status as="div" tone={broken ? 'alert' : 'ok'}>{broken ? 'This chord dips below the curve, so the function is not convex.' : 'This chord stays above the curve.'}</Status>
          <Stat>descent from θ = {fmt(start)} stops at θ = {fmt(x)}</Stat>
          <Stat big>L = {fmt(f(x), 3)}</Stat>
          <Note>On the convex shape every start reaches the same bottom. On the bumpy one, the start decides the answer.</Note>
        </Stack>
      </Row>
      <Takeaway>Convexity is why linear and logistic regression have one answer regardless of initialization, and why they are the safe place to start.</Takeaway>
    </>
  );
}

/* ---------- L02: Gaussian, density vs probability ---------- */

export function GaussianArea() {
  const [mu, setMu] = useState(0), [sd, setSd] = useState(1), [a, setA] = useState(-1), [b, setB] = useState(1);
  const g = linspace(-6, 6, 301), lo = Math.min(a, b), hi = Math.max(a, b), peak = normalPdf(mu, mu, sd);
  const top = Math.max(0.5, peak * 1.1);
  return (
    <>
      <p>A density's height is <b>not</b> a probability. Only area is. Drag the two edges, then make <M t="\sigma" /> small and notice the height go above 1.</p>
      <M d t="p(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)" />
      <Row>
        <Plot x={[-6, 6]} y={[0, +top.toFixed(2)]} xl="x" yl="density" label="normal density with a shaded interval">
          <Mark as="area" tone={1} data={g.filter(x => x >= lo && x <= hi).map(x => [x, 0, normalPdf(x, mu, sd)])} />
          <Mark tone={1} data={g.map(x => [x, normalPdf(x, mu, sd)])} />
          <Mark as="handles" lockY data={[[a, 0.02], [b, 0.02]]} onDrag={(i, x) => (i ? setB(x) : setA(x))} />
        </Plot>
        <Stack>
          <Slider label={<M t="\mu" />} value={mu} min={-3} max={3} step={0.05} onChange={setMu} show={v => fmt(v)} />
          <Slider label={<M t="\sigma" />} value={sd} min={0.15} max={3} step={0.01} onChange={setSd} show={v => fmt(v)} />
          <Stat big>P({fmt(lo)} &lt; X &lt; {fmt(hi)}) = {pct(normalCdf(hi, mu, sd) - normalCdf(lo, mu, sd))}</Stat>
          <Stat>peak height {fmt(peak, 3)} {peak > 1 && <Status as="b" tone="alert">above 1, and still legal</Status>}</Stat>
          <Toolbar>
            {[1, 2, 3].map(k => <Button key={k} onClick={() => { setA(mu - k * sd); setB(mu + k * sd); }}>±{k}σ</Button>)}
          </Toolbar>
        </Stack>
      </Row>
      <Takeaway>Every Gaussian is the same curve moved and stretched, so ±1σ always holds 68% whatever the units.</Takeaway>
    </>
  );
}

/* ---------- L02: Bayes' rule with densities ---------- */

export function BayesDensity() {
  const [m0, setM0] = useState(-1), [m1, setM1] = useState(1.2), [prior, setPrior] = useState(0.5), [sd, setSd] = useState(1);
  const g = linspace(-5, 5, 241), post = x => { const a = prior * normalPdf(x, m1, sd), b = (1 - prior) * normalPdf(x, m0, sd); return a / (a + b); };
  const cross = m0 !== m1 ? (m0 + m1) / 2 + (sd * sd * Math.log((1 - prior) / prior)) / (m1 - m0) : NaN;
  return (
    <>
      <p>A generative story gives each class a density. Weight each by its prior and normalize, and you get a posterior. Drag the class means; note that the <b>posterior is an S-curve</b> even though nothing logistic was assumed.</p>
      <M d t="P(c{=}1\mid x)=\frac{\pi\,p(x\mid 1)}{\pi\,p(x\mid 1)+(1-\pi)\,p(x\mid 0)}" />
      <Row>
        <Plot x={[-5, 5]} y={[0, 1]} xl="x" yl="prior × density, posterior" label="class densities and posterior">
          <Mark as="area" tone={2} data={g.map(x => [x, 0, (1 - prior) * normalPdf(x, m0, sd)])} />
          <Mark as="area" tone={1} data={g.map(x => [x, 0, prior * normalPdf(x, m1, sd)])} />
          <Mark tone={3} data={g.map(x => [x, post(x)])} />
          {Number.isFinite(cross) && <Mark as="rule" dash x={clamp(cross, -5, 5)} />}
          <Mark as="handles" lockY data={[[m0, 0.04, 2], [m1, 0.04, 1]]} onDrag={(i, x) => (i ? setM1(x) : setM0(x))} />
        </Plot>
        <Stack>
          <Slider label={<>prior <M t="\pi" /></>} value={prior} min={0.02} max={0.98} step={0.01} onChange={setPrior} show={v => fmt(v)} />
          <Slider label={<M t="\sigma" />} value={sd} min={0.3} max={2} step={0.01} onChange={setSd} show={v => fmt(v)} />
          <Stat big>decide 1 when x &gt; {fmt(cross)}</Stat>
          <Legend items={[['--c1', 'class 1'], ['--c2', 'class 0'], ['--c3', 'P(1 | x)']]} />
        </Stack>
      </Row>
      <Takeaway>Raising a class's prior moves the boundary away from it: rarer classes must earn a more convincing x. Equal-variance Gaussians give exactly the sigmoid of Lecture 6.</Takeaway>
    </>
  );
}

/* ---------- L03: distribution zoo ---------- */

const ZOO = {
  bernoulli: { name: 'Bernoulli', params: [['p', 0, 1, 0.3]], discrete: k => ({ xs: [0, 1], p: x => (x ? k[0] : 1 - k[0]) }), m: k => k[0], v: k => k[0] * (1 - k[0]), use: 'one yes/no outcome' },
  binomial: { name: 'Binomial', params: [['n', 1, 30, 10, 1], ['p', 0, 1, 0.4]], discrete: k => ({ xs: Array.from({ length: k[0] + 1 }, (_, i) => i), p: x => binomPmf(x, k[0], k[1]) }), m: k => k[0] * k[1], v: k => k[0] * k[1] * (1 - k[1]), use: 'count of successes in n tries' },
  poisson: { name: 'Poisson', params: [['λ', 0.2, 15, 3]], discrete: k => ({ xs: Array.from({ length: 31 }, (_, i) => i), p: x => poissonPmf(x, k[0]) }), m: k => k[0], v: k => k[0], use: 'counts per interval' },
  gaussian: { name: 'Gaussian', params: [['μ', -3, 3, 0], ['σ', 0.3, 3, 1]], pdf: (x, k) => normalPdf(x, k[0], k[1]), m: k => k[0], v: k => k[1] ** 2, use: 'sums of many small effects' },
  exponential: { name: 'Exponential', params: [['λ', 0.2, 3, 1]], pdf: (x, k) => (x < 0 ? 0 : k[0] * Math.exp(-k[0] * x)), m: k => 1 / k[0], v: k => 1 / k[0] ** 2, use: 'waiting times' },
  beta: { name: 'Beta', params: [['a', 0.5, 12, 2], ['b', 0.5, 12, 5]], pdf: (x, k) => betaPdf(x, k[0], k[1]), m: k => k[0] / (k[0] + k[1]), v: k => (k[0] * k[1]) / ((k[0] + k[1]) ** 2 * (k[0] + k[1] + 1)), use: 'a probability that is itself unknown' },
};

export function Zoo() {
  const [name, setName] = useState('binomial'), [vals, setVals] = useState(Object.fromEntries(Object.entries(ZOO).map(([k, d]) => [k, d.params.map(p => p[3])])));
  const d = ZOO[name], k = vals[name], D = d.discrete?.(k);
  const X = name === 'beta' ? [0, 1] : name === 'gaussian' ? [-7, 7] : name === 'exponential' ? [0, 8] : [-0.5, Math.max(...D.xs) + 0.5];
  const g = linspace(X[0] + 1e-3, X[1] - 1e-3, 301), top = Math.min(4, Math.max(0.1, ...(D ? D.xs.map(D.p) : g.map(x => d.pdf(x, k)))) * 1.12);
  return (
    <>
      <p>Each distribution is a <b>shape with knobs</b>, and bars and curves mean different things: a bar's height is a probability, a curve's area is.</p>
      <Toolbar><Choice label="Distribution" value={name} onChange={setName} options={Object.entries(ZOO).map(([v, z]) => ({ value: v, label: z.name }))} /></Toolbar>
      <Row>
        <Plot x={X} y={[0, +top.toFixed(3)]} xl="x" yl={D ? 'P(X = x)' : 'density p(x)'} label={`${d.name} distribution`}>
          {D ? <Mark as="rects" tone={1} data={D.xs.map(x => [x - 0.35, 0, x + 0.35, D.p(x)])} />
            : <><Mark as="area" tone={1} data={g.map(x => [x, 0, d.pdf(x, k)])} /><Mark tone={1} data={g.map(x => [x, d.pdf(x, k)])} /></>}
          <Mark as="rule" dash x={d.m(k)} />
        </Plot>
        <Stack>
          {d.params.map(([l, lo, hi, , step], i) => <Slider key={l} label={l} value={k[i]} min={lo} max={hi} step={step ?? 0.01} show={v => fmt(v, step ? 0 : 2)} onChange={v => setVals(s => ({ ...s, [name]: k.map((o, j) => (j === i ? v : o)) }))} />)}
          <Stat big>mean {fmt(d.m(k), 3)}</Stat>
          <Stat>variance {fmt(d.v(k), 3)}</Stat>
          <Note>Models: {d.use}.</Note>
        </Stack>
      </Row>
      <Takeaway>Picking a distribution is picking a loss. Gaussian noise leads to squared error, Laplace to absolute error, Bernoulli to log loss.</Takeaway>
    </>
  );
}

/* ---------- L03: likelihood vs log-likelihood ---------- */

export function Likelihood() {
  const [h, setH] = useState(7), [n, setN] = useState(10), [t, setT] = useState(0.5);
  const heads = Math.min(h, n), g = linspace(0.002, 0.998, 301), L = p => p ** heads * (1 - p) ** (n - heads), ll = p => heads * Math.log(p) + (n - heads) * Math.log(1 - p);
  const hat = heads / n, Lmax = Math.max(...g.map(L)), floor = Math.min(ll(hat) - 12, -1), topLL = Math.max(...g.map(ll));
  return (
    <>
      <p>Likelihood holds the <b>data fixed and varies θ</b>. It is not a probability over θ: its area need not be 1. Drag θ and compare the two curves.</p>
      <M d t="L(\theta)=\theta^{h}(1-\theta)^{n-h},\qquad \ell(\theta)=h\log\theta+(n-h)\log(1-\theta),\qquad \hat\theta=\tfrac hn" />
      <Row>
        <Plot caption="Likelihood" x={[0, 1]} y={[0, +Lmax.toExponential(2) * 1.1]} w={340} h={220} xl="θ" label="likelihood curve">
          <Mark tone={1} data={g.map(p => [p, L(p)])} />
          <Mark as="rule" dash x={hat} />
          <Mark as="handles" lockY data={[[t, Lmax * 0.05]]} onDrag={(_, x) => setT(clamp(x, 0.01, 0.99))} />
          <Mark as="points" tone={4} data={[[t, L(t)]]} />
        </Plot>
        <Plot caption="Log-likelihood" x={[0, 1]} y={[+floor.toFixed(1), +(topLL + 1).toFixed(1)]} w={340} h={220} xl="θ" label="log-likelihood curve">
          <Mark tone={2} data={g.map(p => [p, Math.max(ll(p), floor)])} />
          <Mark as="rule" dash x={hat} />
          <Mark as="points" tone={4} data={[[t, Math.max(ll(t), floor)]]} />
        </Plot>
      </Row>
      <Slider label={<>tosses <M t="n" /></>} value={n} min={1} max={200} step={1} onChange={setN} />
      <Slider label={<>heads <M t="h" /></>} value={heads} min={0} max={n} step={1} onChange={setH} />
      <Stat big><M t={String.raw`\hat\theta=${fmt(hat, 3)},\quad L(${fmt(t)})=${L(t).toExponential(2).replace('e', '\\times10^{') + '}'},\quad \ell(${fmt(t)})=${fmt(ll(t), 2)}`} /></Stat>
      <Takeaway>Raise n to 200: L collapses towards 10⁻⁶⁰ and would underflow a computer, while ℓ stays ordinary. The log keeps the peak in the same place, and turns products into sums.</Takeaway>
    </>
  );
}

/* ---------- L03: Beta-Bernoulli updating ---------- */

export function BetaBayes() {
  const [a, setA] = useState(2), [b, setB] = useState(2), [h, setH] = useState(0), [t, setT] = useState(0);
  const g = linspace(0.002, 0.998, 301), pa = a + h, pb = b + t, n = h + t;
  const prior = x => betaPdf(x, a, b), post = x => betaPdf(x, pa, pb);
  const like = x => (n ? betaPdf(x, h + 1, t + 1) : 0), top = Math.min(14, Math.max(...g.map(prior), ...g.map(post), ...g.map(like)) * 1.1);
  const mle = n ? h / n : NaN, map = pa > 1 && pb > 1 ? (pa - 1) / (pa + pb - 2) : NaN, pm = pa / (pa + pb);
  return (
    <>
      <p>Toss the coin and watch belief update. Conjugacy means the update is just <b>adding counts</b>: the prior acts like pseudo-tosses seen earlier.</p>
      <M d t="\text{Beta}(a,b)\ \xrightarrow{\ h\text{ heads},\ t\text{ tails}\ }\ \text{Beta}(a+h,\ b+t)" />
      <Toolbar>
        <Button variant="solid" onClick={() => setH(v => v + 1)}>Heads</Button>
        <Button variant="solid" onClick={() => setT(v => v + 1)}>Tails</Button>
        <Button onClick={() => { setH(v => v + 7); setT(v => v + 3); }}>+7 heads, +3 tails</Button>
        <Button onClick={() => { setH(0); setT(0); }}>Reset</Button>
      </Toolbar>
      <Row>
        <Plot x={[0, 1]} y={[0, +top.toFixed(2)]} xl="θ" yl="density" label="prior, normalized likelihood and posterior">
          <Mark tone={2} dash data={g.map(x => [x, Math.min(prior(x), top)])} />
          {n > 0 && <Mark tone={3} dash data={g.map(x => [x, Math.min(like(x), top)])} />}
          <Mark as="area" tone={1} data={g.map(x => [x, 0, Math.min(post(x), top)])} />
          <Mark tone={1} data={g.map(x => [x, Math.min(post(x), top)])} />
        </Plot>
        <Stack>
          <Slider label="prior a" value={a} min={0.5} max={30} step={0.5} onChange={setA} show={v => fmt(v, 1)} />
          <Slider label="prior b" value={b} min={0.5} max={30} step={0.5} onChange={setB} show={v => fmt(v, 1)} />
          <Stat big>{h} heads, {t} tails</Stat>
          <Stat>MLE {fmt(mle, 3)} · MAP {fmt(map, 3)} · posterior mean {fmt(pm, 3)}</Stat>
          <Legend items={[['--c2', 'prior'], ['--c3', 'likelihood'], ['--c1', 'posterior']]} />
        </Stack>
      </Row>
      <Takeaway>With little data the prior holds the estimate back from extremes; one toss gives an MLE of 0 or 1, the posterior mean never does. With much data, all three estimates agree.</Takeaway>
    </>
  );
}

/* ---------- L03: fit a Gaussian to draggable data ---------- */

export function GaussianFit() {
  const [pts, setPts] = useState([-1.5, -0.6, 0.1, 0.4, 1.2].map(x => [x, 0.03, 1]));
  const [mu, setMu] = useState(1), [sd, setSd] = useState(1.5);
  const xs = pts.map(p => p[0]), hatM = mean(xs), hatS = Math.max(Math.sqrt(mean(xs.map(x => (x - hatM) ** 2))), 0.05);
  const LL = (m, s) => xs.reduce((a, x) => a + Math.log(normalPdf(x, m, s)), 0), g = linspace(-6, 6, 241);
  return (
    <>
      <p>Your curve and the data. Each stem's height is that point's density, and the log-likelihood adds up their logs. <b>Drag the data</b>, then try to beat the MLE with the sliders.</p>
      <M d t="\hat\mu=\tfrac1n\sum x_i,\qquad \hat\sigma^2=\tfrac1n\sum(x_i-\hat\mu)^2" />
      <Row>
        <Plot x={[-6, 6]} y={[0, 1]} xl="x" yl="density" label="Gaussian with draggable data points">
          <Mark tone={3} dash data={g.map(x => [x, normalPdf(x, hatM, hatS)])} />
          <Mark tone={1} data={g.map(x => [x, normalPdf(x, mu, sd)])} />
          <Mark as="stems" tone="muted" data={xs.map(x => [x, 0, normalPdf(x, mu, sd)])} />
          <Mark as="handles" lockY data={pts} onDrag={(i, x, y) => setPts(moved(i, x, y))} />
        </Plot>
        <Stack>
          <Slider label={<M t="\mu" />} value={mu} min={-4} max={4} step={0.05} onChange={setMu} show={v => fmt(v)} />
          <Slider label={<M t="\sigma" />} value={sd} min={0.2} max={4} step={0.05} onChange={setSd} show={v => fmt(v)} />
          <Stat big>your log-likelihood {fmt(LL(mu, sd), 3)}</Stat>
          <Stat>the MLE's {fmt(LL(hatM, hatS), 3)}</Stat>
          <Button onClick={() => { setMu(+hatM.toFixed(2)); setSd(+hatS.toFixed(2)); }}>Jump to the MLE</Button>
          <Legend items={[['--c1', 'your curve'], ['--c3', 'MLE']]} />
        </Stack>
      </Row>
      <Takeaway>Drag one point far away. Both the mean and the width chase it, because the log-density punishes an unlikely point quadratically. That same force makes squared loss fragile to outliers.</Takeaway>
    </>
  );
}
