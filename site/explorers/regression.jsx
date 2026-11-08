'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { DotGrid, Mark, Plot, moved } from '../components/Plot';
import { Button, Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from '../components/ui';
import { gauss, linfit, mean, polyval, rng, sum } from '../lib/math.mjs';
import { clamp, eig2, fmt, ladFit, lasso, linspace, logLoss, logisticStep, pct, ridge, sigmoid, soft } from '../lib/models.mjs';

const TONES = [1, 2, 3, 4];

/* ---------- L04: residuals are literal squares ---------- */

const SQUARE_DATA = [[1, 2.2], [2.5, 3.1], [4, 5.8], [5.5, 4.6], [7, 7.4]];

export function Squares() {
  const [ends, setEnds] = useState([[0, 4], [9, 5]]), [pts, setPts] = useState(SQUARE_DATA);
  const w = (ends[1][1] - ends[0][1]) / 9, b = ends[0][1], at = x => w * x + b;
  const best = linfit(pts.map(p => p[0]), pts.map(p => p[1])), sse = sum(pts.map(([x, y]) => (y - at(x)) ** 2)), low = sum(pts.map(([x, y]) => (y - best.w * x - best.b) ** 2));
  return (
    <>
      <p>Squared error is not a metaphor. Each miss is <b>a real square whose side is the residual</b>, and least squares minimises their total area. Drag the line's ends and the data.</p>
      <Row>
        <Plot x={[0, 9]} y={[0, 9]} w={400} h={400} xl="x" yl="y" label="line with residual squares">
          <Mark as="rects" tone={2} data={pts.map(([x, y]) => { const r = y - at(x); return [x, y, x + (r > 0 ? -1 : 1) * Math.abs(r), at(x)]; })} />
          <Mark tone={1} data={[[0, at(0)], [9, at(9)]]} />
          <Mark as="handles" data={ends} onDrag={(i, _, y) => setEnds(moved(i, ends[i][0], y))} />
          <Mark as="handles" r={5} data={pts.map(p => [...p, 2])} onDrag={(i, x, y) => setPts(moved(i, x, y))} />
        </Plot>
        <Stack>
          <Stat big>total area {fmt(sse)}</Stat>
          <Stat>smallest possible {fmt(low)}</Stat>
          <Button onClick={() => setEnds([[0, best.b], [9, best.w * 9 + best.b]])}>Show the least-squares line</Button>
          <Note>Doubling a residual quadruples its square. One far point can dominate the sum, so the line bends to it.</Note>
        </Stack>
      </Row>
      <Takeaway>The optimum makes residuals balance out: they sum to zero and are uncorrelated with x. If a pattern is left in them, the line missed something.</Takeaway>
    </>
  );
}

/* ---------- L04: noise model picks the loss ---------- */

export function NoiseLoss() {
  const [pts, setPts] = useState([[0.5, 1], [1.5, 1.6], [2.5, 3.2], [3.5, 3.7], [4.5, 5.2], [5.5, 5.6], [6.5, 7.1], [8, 1]]);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), ols = linfit(xs, ys), lad = ladFit(xs, ys);
  const nll = (s, o) => sum(xs.map((x, i) => (o === 'g' ? (ys[i] - s.w * x - s.b) ** 2 : Math.abs(ys[i] - s.w * x - s.b))));
  const ln = s => [[0, s.b], [9, s.w * 9 + s.b]];
  return (
    <>
      <p>Assume Gaussian noise and the negative log-likelihood is squared loss. Assume Laplace noise and it is absolute loss. Drag the outlier in the corner up and down.</p>
      <M d t="-\log p(y\mid x,\theta)=\begin{cases}\frac{1}{2\sigma^2}(y-\hat y)^2+\text{const}&\text{Gaussian}\\[2pt]\frac1b|y-\hat y|+\text{const}&\text{Laplace}\end{cases}" />
      <Row>
        <Plot x={[0, 9]} y={[0, 9]} xl="x" yl="y" label="least squares and least absolute fits">
          <Mark tone={1} data={ln(ols)} />
          <Mark tone={2} data={ln(lad)} />
          <Mark as="handles" r={5} data={pts} onDrag={(i, x, y) => setPts(moved(i, x, y))} />
        </Plot>
        <Stack>
          <Stat big>squared loss: slope {fmt(ols.w)}</Stat>
          <Stat big>absolute loss: slope {fmt(lad.w)}</Stat>
          <Stat>sum of squares {fmt(nll(ols, 'g'))} · sum of absolutes {fmt(nll(lad, 'l'))}</Stat>
          <Legend items={[['--c1', 'Gaussian noise (OLS)'], ['--c2', 'Laplace noise (LAD)']]} />
        </Stack>
      </Row>
      <Takeaway>The absolute-loss line behaves like a median and barely notices the outlier. Choosing a loss is a claim about how your data is corrupted.</Takeaway>
    </>
  );
}

/* ---------- L04: projection in observation space ---------- */

export function Projection() {
  const [xv, setX] = useState([2.5, 1]), [y, setY] = useState([1, 3]);
  const c = (xv[0] * y[0] + xv[1] * y[1]) / (xv[0] ** 2 + xv[1] ** 2 || 1), yh = [c * xv[0], c * xv[1]], e = [y[0] - yh[0], y[1] - yh[1]];
  const span = [[-4 * xv[0], -4 * xv[1]], [4 * xv[0], 4 * xv[1]]];
  return (
    <>
      <p>Stop looking at points on an x-y scatter. Here <b>each axis is one observation</b>, so the whole column <M t="y" /> is a single arrow. Fitting means finding the closest point to <M t="y" /> on the line spanned by the feature column.</p>
      <M d t="\hat y=X(X^\top X)^{-1}X^\top y,\qquad X^\top(y-\hat y)=0" />
      <Row>
        <Plot x={[-4, 4]} y={[-4, 4]} w={380} h={380} xl="observation 1" yl="observation 2" label="vector y projected onto the column of X">
          <Mark tone="muted" dash data={span} />
          <Mark as="arrow" tone={1} data={[[0, 0, ...xv]]} />
          <Mark as="arrow" tone={3} data={[[0, 0, ...yh]]} />
          <Mark as="arrow" tone={4} data={[[...yh, ...y]]} />
          <Mark as="arrow" tone="fg" data={[[0, 0, ...y]]} />
          <Mark as="handles" data={[[...xv, 1]]} onDrag={(_, a, b) => setX([a, b])} />
          <Mark as="handles" data={[y]} onDrag={(_, a, b) => setY([a, b])} />
        </Plot>
        <Stack>
          <Stat big><M t={`\\hat\\beta = ${fmt(c, 3)}`} /></Stat>
          <Stat>residual length {fmt(Math.hypot(...e), 3)}</Stat>
          <Stat>residual · x = {fmt(e[0] * xv[0] + e[1] * xv[1], 4)}</Stat>
          <Legend items={[['--c1', 'column of X'], ['--fg', 'y'], ['--c3', 'ŷ (the fit)'], ['--c4', 'residual']]} />
          <Note>The dot product stays at zero however you drag: the residual is always perpendicular to everything the model can make.</Note>
        </Stack>
      </Row>
      <Takeaway>The normal equations are a geometry statement. The residual must be orthogonal to every column of X, and that condition alone gives the formula.</Takeaway>
    </>
  );
}

/* ---------- L05: coefficient space and data space, linked ---------- */

const TWO_DATA = (() => { const r = rng(9), xs = linspace(-2, 2, 12); return { xs, ys: xs.map(x => 0.8 * x + 1 + 0.5 * gauss(r)) }; })();

export function TwoSpaces() {
  const [th, setTh] = useState([-1, -1]), { xs, ys } = TWO_DATA;
  const L = (w, b) => mean(xs.map((x, i) => (ys[i] - w * x - b) ** 2)), best = linfit(xs, ys), max = L(3, 3);
  return (
    <>
      <p>One idea, two pictures. A <b>single point on the left is an entire line on the right</b>. Drag it down the bowl.</p>
      <Row>
        <Plot caption="Coefficient space: each point is a model" x={[-3, 3]} y={[-3, 3]} w={340} h={320} xl="slope w" yl="intercept b" label="loss surface in coefficient space">
          <Mark as="field" bands={9} tones={[1, 1]} fn={(w, b) => Math.sqrt(clamp(L(w, b) / max, 0, 1))} />
          <Mark as="points" tone={3} data={[[best.w, best.b]]} />
          <Mark as="handles" data={[th]} onDrag={(_, a, b) => setTh([a, b])} />
        </Plot>
        <Plot caption="Data space: the same model as a line" x={[-2.5, 2.5]} y={[-4, 6]} w={340} h={320} xl="x" yl="y" label="the corresponding line and data">
          <Mark as="stems" tone="muted" data={xs.map((x, i) => [x, ys[i], th[0] * x + th[1]])} />
          <Mark tone={1} data={[[-2.5, -2.5 * th[0] + th[1]], [2.5, 2.5 * th[0] + th[1]]]} />
          <Mark as="points" tone="fg" data={xs.map((x, i) => [x, ys[i]])} />
        </Plot>
      </Row>
      <Stat big><M t={String.raw`L(${fmt(th[0])},\,${fmt(th[1])}) = ${fmt(L(...th), 3)}\qquad \text{minimum } ${fmt(L(best.w, best.b), 3)}`} /></Stat>
      <Takeaway>Training moves one point through coefficient space. Every optimizer in the course is a rule for that walk.</Takeaway>
    </>
  );
}

/* ---------- L05: eigenvectors by direct search ---------- */

export function Eigen() {
  const [a, setA] = useState(2), [b, setB] = useState(1), [d, setD] = useState(1.5), [v, setV] = useState([0.8, 0.6]);
  const n = Math.hypot(...v) || 1, u = [v[0] / n, v[1] / n], Au = [a * u[0] + b * u[1], b * u[0] + d * u[1]];
  const e = eig2(a, b, d), ang = Math.abs(Math.atan2(u[0] * Au[1] - u[1] * Au[0], u[0] * Au[0] + u[1] * Au[1]));
  const align = Math.min(ang, Math.PI - ang) < 0.03, c = linspace(0, 2 * Math.PI, 121);
  return (
    <>
      <p>A matrix usually turns a vector. Eigenvectors are the rare directions it <b>only stretches</b>. Drag the blue arrow round the circle until the orange arrow lines up with it.</p>
      <M d t="A\,v=\lambda\,v" />
      <Row>
        <Plot x={[-4, 4]} y={[-4, 4]} w={380} h={380} xl="x₁" yl="x₂" label="unit circle mapped to an ellipse by a symmetric matrix">
          <Mark tone="muted" dash data={c.map(t => [Math.cos(t), Math.sin(t)])} />
          <Mark tone={3} data={c.map(t => [a * Math.cos(t) + b * Math.sin(t), b * Math.cos(t) + d * Math.sin(t)])} />
          <Mark as="arrow" tone={2} data={[[0, 0, ...Au]]} />
          <Mark as="arrow" tone={1} data={[[0, 0, ...u]]} />
          <Mark as="handles" data={[[...u, 1]]} onDrag={(_, x, y) => setV([x, y])} />
        </Plot>
        <Stack>
          <Slider label={<M t="A_{11}" />} value={a} min={-3} max={3} step={0.05} onChange={setA} show={v2 => fmt(v2)} />
          <Slider label={<M t="A_{12}" />} value={b} min={-2} max={2} step={0.05} onChange={setB} show={v2 => fmt(v2)} />
          <Slider label={<M t="A_{22}" />} value={d} min={-3} max={3} step={0.05} onChange={setD} show={v2 => fmt(v2)} />
          <Status as="div" tone={align ? 'ok' : 'alert'}>{align ? 'Aligned: this is an eigenvector.' : 'Not aligned yet.'}</Status>
          <Stat big><M t={`\\lambda_1=${fmt(e.l1, 3)},\\ \\lambda_2=${fmt(e.l2, 3)}`} /></Stat>
          <Toolbar>
            <Button onClick={() => setV(e.v)}>Snap to v₁</Button>
            <Button onClick={() => setV(e.u)}>Snap to v₂</Button>
          </Toolbar>
        </Stack>
      </Row>
      <Takeaway>Symmetric matrices have perpendicular eigenvectors, which are the ellipse's axes. Curvature, conditioning, ridge and PCA all reduce to reading these axes.</Takeaway>
    </>
  );
}

/* ---------- L05: PCA by hand ---------- */

function pcaCloud(seed) { const r = rng(seed); return Array.from({ length: 50 }, () => [gauss(r), gauss(r)]); }

export function Pca() {
  const [t, setT] = useState(10), [rho, setRho] = useState(0.8), [seed, setSeed] = useState(4);
  const pts = useMemo(() => pcaCloud(seed).map(([a, b]) => [1.5 * a, 1.5 * (rho * a + Math.sqrt(1 - rho * rho) * b)]), [seed, rho]);
  const mx = mean(pts.map(p => p[0])), my = mean(pts.map(p => p[1])), Z = pts.map(([x, y]) => [x - mx, y - my]);
  const s = (f, g) => mean(Z.map(z => f(z) * g(z))), a = s(z => z[0], z => z[0]), b = s(z => z[0], z => z[1]), d = s(z => z[1], z => z[1]), e = eig2(a, b, d);
  const rad = (t * Math.PI) / 180, u = [Math.cos(rad), Math.sin(rad)], proj = Z.map(z => z[0] * u[0] + z[1] * u[1]), kept = mean(proj.map(v => v * v)), total = a + d;
  return (
    <>
      <p>Pick a direction and project everything on to it. PCA wants the direction <b>keeping the most variance</b>, which is the same one <b>losing the least reconstruction error</b>. Rotate and watch the two trade.</p>
      <M d t="\max_{\|u\|=1}\ u^\top S\,u\quad\Longrightarrow\quad S\,u=\lambda u" />
      <Row>
        <Plot x={[-5, 5]} y={[-5, 5]} w={380} h={380} xl="x₁" yl="x₂" label="scatter projected on a rotating direction">
          <Mark tone="muted" dash data={[[-6 * u[0], -6 * u[1]], [6 * u[0], 6 * u[1]]]} />
          {Z.map((z, i) => <Mark key={i} tone="muted" data={[z, [proj[i] * u[0], proj[i] * u[1]]]} />)}
          <Mark as="points" tone={1} r={3} data={Z} />
          <Mark as="points" tone={2} r={3} data={proj.map(v => [v * u[0], v * u[1]])} />
          <Mark as="handles" data={[[3.5 * u[0], 3.5 * u[1], 4]]} onDrag={(_, x, y) => setT((Math.atan2(y, x) * 180) / Math.PI)} />
        </Plot>
        <Stack>
          <Slider label="correlation" value={rho} min={0} max={0.98} step={0.01} onChange={setRho} show={v => fmt(v)} />
          <Stat big>variance kept {fmt(kept, 3)} ({pct(kept / total, 0)})</Stat>
          <Stat>reconstruction error {fmt(total - kept, 3)}</Stat>
          <Stat>kept + lost = {fmt(total, 3)}, always</Stat>
          <Toolbar>
            <Button variant="solid" onClick={() => setT((Math.atan2(e.v[1], e.v[0]) * 180) / Math.PI)}>Snap to PC1</Button>
            <Button onClick={() => setSeed(v => v + 1)}>New sample</Button>
          </Toolbar>
          <Note>λ₁ = {fmt(e.l1, 3)} equals the variance kept at PC1.</Note>
        </Stack>
      </Row>
      <Takeaway>By Pythagoras, maximising the projection minimises the gap. Scale matters: PCA on unstandardized features just finds whichever has the largest units.</Takeaway>
    </>
  );
}

/* ---------- L06: curvature and the eigenvalue test ---------- */

export function Curvature() {
  const [a, setA] = useState(2), [b, setB] = useState(0.5), [d, setD] = useState(1), [t, setT] = useState(30);
  const e = eig2(a, b, d), q = (x, y) => 0.5 * (a * x * x + 2 * b * x * y + d * y * y), max = Math.max(1, ...[[2, 2], [2, -2]].map(p => Math.abs(q(...p))));
  const rad = (t * Math.PI) / 180, u = [Math.cos(rad), Math.sin(rad)], g = linspace(-2, 2, 81), curv = a * u[0] ** 2 + 2 * b * u[0] * u[1] + d * u[1] ** 2;
  const kind = e.l2 > 1e-3 ? 'A bowl: strictly convex, one minimum.' : e.l1 < -1e-3 ? 'A dome: a maximum.' : e.l1 > 1e-3 && e.l2 < -1e-3 ? 'A saddle: not convex.' : 'A valley: flat along one direction, with many minimizers.';
  return (
    <>
      <p>The Hessian's eigenvalues name the shape of a surface. Slide its entries and see <b>bowl, saddle, dome and valley</b>. The right plot slices the surface along the arrow.</p>
      <M d t="f(\theta)=\tfrac12\theta^\top H\theta,\qquad \left.\frac{d^2 f}{ds^2}\right|_{u}=u^\top H u" />
      <Row>
        <Plot caption="Contours (blue up, orange down)" x={[-2, 2]} y={[-2, 2]} w={340} h={320} xl="θ₁" yl="θ₂" label="quadratic form contours">
          <Mark as="field" bands={8} fn={(x, y) => clamp(q(x, y) / max, -1, 1)} />
          <Mark as="arrow" tone="fg" data={[[0, 0, 1.6 * u[0], 1.6 * u[1]]]} />
          <Mark as="handles" data={[[1.6 * u[0], 1.6 * u[1]]]} onDrag={(_, x, y) => setT((Math.atan2(y, x) * 180) / Math.PI)} />
        </Plot>
        <Plot caption="Slice along the arrow" x={[-2, 2]} y={[-4, 4]} w={340} h={320} xl="s" yl="f(s·u)" label="one-dimensional slice">
          <Mark tone={3} data={g.map(s => [s, 0.5 * curv * s * s])} />
        </Plot>
      </Row>
      <Slider label={<M t="H_{11}" />} value={a} min={-3} max={3} step={0.05} onChange={setA} show={v => fmt(v)} />
      <Slider label={<M t="H_{12}" />} value={b} min={-3} max={3} step={0.05} onChange={setB} show={v => fmt(v)} />
      <Slider label={<M t="H_{22}" />} value={d} min={-3} max={3} step={0.05} onChange={setD} show={v => fmt(v)} />
      <Stat big><M t={`\\mu_1=${fmt(e.l1, 2)},\\ \\mu_2=${fmt(e.l2, 2)},\\quad u^\\top Hu=${fmt(curv, 2)}`} /></Stat>
      <Status as="div" tone={e.l2 > 1e-3 ? 'ok' : 'alert'}>{kind}</Status>
      <Takeaway>A direction's curvature always lies between μ₂ and μ₁. Every eigenvalue positive means every slice curves up, and that is what positive definite means.</Takeaway>
    </>
  );
}

/* ---------- L06: Taylor and Newton ---------- */

export function Taylor() {
  const [x, setX] = useState(2.2);
  const f = t => Math.log(1 + t * t) + 0.1 * t * t, d = t => (2 * t) / (1 + t * t) + 0.2 * t, h = t => (2 * (1 - t * t)) / (1 + t * t) ** 2 + 0.2;
  const g = linspace(-4, 4), q = t => f(x) + d(x) * (t - x) + 0.5 * h(x) * (t - x) ** 2, nx = x - d(x) / h(x), bad = h(x) <= 0;
  return (
    <>
      <p>Newton's method replaces the curve with its <b>best-fitting parabola</b> and jumps to that parabola's bottom. Drag the point. Where curvature is negative the parabola opens down and the jump goes uphill.</p>
      <M d t="f(\theta+d)\approx f+f'd+\tfrac12 f''d^2\quad\Rightarrow\quad d=-\frac{f'}{f''}" />
      <Row>
        <Plot x={[-4, 4]} y={[-1, 5]} xl="θ" yl="f(θ)" label="function with quadratic approximation">
          <Mark tone={1} data={g.map(t => [t, f(t)])} />
          <Mark tone={2} data={g.map(t => [t, q(t)])} />
          <Mark as="points" tone={2} data={[[clamp(nx, -4, 4), clamp(q(nx), -1, 5)]]} />
          <Mark as="handles" lockY data={[[x, f(x)]]} onDrag={(_, v) => setX(v)} />
        </Plot>
        <Stack>
          <Stat big>f″({fmt(x)}) = {fmt(h(x), 3)}</Stat>
          <Stat>Newton sends θ to {fmt(nx, 3)}, loss {fmt(f(nx), 3)} (was {fmt(f(x), 3)})</Stat>
          <Status as="div" tone={bad ? 'alert' : 'ok'}>{bad ? 'Negative curvature: the step is towards a maximum of the parabola.' : 'Positive curvature: the step is towards the parabola’s minimum.'}</Status>
          <Toolbar><Button onClick={() => setX(clamp(nx, -4, 4))}>Take the step</Button></Toolbar>
          <Legend items={[['--c1', 'f'], ['--c2', 'second-order model']]} />
        </Stack>
      </Row>
      <Takeaway>Logistic loss is convex, so its curvature is never negative. That is why Newton (IRLS) is safe there, and takes few steps.</Takeaway>
    </>
  );
}

/* ---------- L06: the odds ladder ---------- */

export function OddsLadder() {
  const [p, setP] = useState(0.8), z = Math.log(p / (1 - p)), g = linspace(-6, 6);
  return (
    <>
      <p>Probability, odds and log-odds are <b>one number in three outfits</b>. Only the last can be any real number, which is why a linear score can model it.</p>
      <M d t="p\ \to\ \frac{p}{1-p}\ \to\ \log\frac{p}{1-p}=z\qquad\Longleftrightarrow\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}" />
      <Row>
        <Plot x={[-6, 6]} y={[0, 1]} xl="log-odds z" yl="probability p" label="sigmoid with a draggable point">
          <Mark tone={1} data={g.map(v => [v, sigmoid(v)])} />
          <Mark as="rule" dash y={0.5} />
          <Mark as="handles" lockY={false} data={[[z, p]]} onDrag={(_, a) => setP(clamp(sigmoid(a), 0.0025, 0.9975))} />
        </Plot>
        <Stack>
          <Slider label="probability" value={p} min={0.0025} max={0.9975} step={0.0025} onChange={setP} show={v => fmt(v, 3)} />
          <Stat big>odds {fmt(p / (1 - p), 2)} to 1</Stat>
          <Stat big>log-odds {fmt(z, 2)}</Stat>
          <Note>A unit increase in z multiplies the odds by e ≈ 2.72 wherever you are, but changes p a lot near 0.5 and hardly at all near the ends.</Note>
        </Stack>
      </Row>
      <Takeaway>Coefficients in logistic regression are additive in log-odds, so reading them as direct changes in probability is the classic mistake.</Takeaway>
    </>
  );
}

/* ---------- L06: logistic regression, GD vs Newton ---------- */

const LOGI = (() => { const r = rng(21); return Array.from({ length: 28 }, (_, i) => { const y = i % 2; return [(y ? 1 : -1) * 1.1 + gauss(r) * 0.9, (y ? 0.7 : -0.7) + gauss(r) * 0.9, y + 1]; }); })();

export function LogisticLab() {
  const [pts, setPts] = useState(LOGI), [w, setW] = useState([0, 0, 0]), [hist, setHist] = useState([]), [eta, setEta] = useState(1);
  const X = pts.map(p => [1, p[0], p[1]]), y = pts.map(p => p[2] - 1), loss = logLoss(w, X, y), acc = mean(X.map((r, i) => +((r[0] * w[0] + r[1] * w[1] + r[2] * w[2] > 0) === !!y[i])));
  const step = mode => { const n = logisticStep(w, X, y, mode, eta); setW(n); setHist(h => [...h, logLoss(n, X, y)]); };
  const reset = () => { setW([0, 0, 0]); setHist([]); };
  const series = [Math.log(2), ...hist], norm = Math.hypot(w[1], w[2]);
  return (
    <>
      <p>The shading is the fitted probability. <b>Step by step</b>, compare a gradient step with a Newton step on the same data, and drag points across the boundary to make the classes separable.</p>
      <M d t="\nabla L=\tfrac1n X^\top(p-y),\qquad H=\tfrac1n X^\top W X,\quad W=\operatorname{diag}\big(p_i(1-p_i)\big)" />
      <Toolbar>
        <Button variant="solid" onClick={() => step('gd')}>Gradient step</Button>
        <Button variant="solid" onClick={() => step('newton')}>Newton step</Button>
        <Button onClick={reset}>Reset</Button>
      </Toolbar>
      <Row>
        <Plot x={[-4, 4]} y={[-4, 4]} w={360} h={340} xl="x₁" yl="x₂" label="probability field with draggable points">
          <Mark as="field" fn={(a, b) => 2 * sigmoid(w[0] + w[1] * a + w[2] * b) - 1} />
          <Mark as="handles" r={5} data={pts} onDrag={(i, a, b) => setPts(moved(i, a, b))} />
        </Plot>
        <Stack>
          <Plot caption="Loss by step" x={[0, Math.max(5, hist.length)]} y={[0, 0.8]} w={340} h={170} xl="step" label="loss per step">
            <Mark tone={4} data={series.map((v, i) => [i, v])} />
            <Mark as="points" tone={4} r={3} data={series.map((v, i) => [i, v])} />
          </Plot>
          <Slider label={<>gradient step <M t="\eta" /></>} value={eta} min={0.1} max={5} step={0.1} onChange={setEta} show={v => fmt(v, 1)} />
          <Stat big>loss {fmt(loss, 4)} · accuracy {pct(acc, 0)}</Stat>
          <Stat>‖w‖ = {fmt(norm, 2)} {norm > 15 && <Status as="b" tone="alert">growing without limit: the classes are separable</Status>}</Stat>
        </Stack>
      </Row>
      <Takeaway>Newton uses curvature and gets there in a handful of steps. With perfectly separable classes no finite minimum exists, and the weights grow for ever. Regularization fixes that.</Takeaway>
    </>
  );
}

/* ---------- L07: bias-variance by re-sampling the world ---------- */

function bvData(deg, lam) {
  const truth = x => Math.sin(Math.PI * x), g = linspace(-1, 1, 81), fits = [];
  for (let s = 1; s <= 60; s++) {
    const r = rng(s * 13), xs = Array.from({ length: 15 }, () => 2 * r() - 1), ys = xs.map(x => truth(x) + 0.35 * gauss(r));
    const X = xs.map(x => Array.from({ length: deg + 1 }, (_, j) => x ** j));
    const c = ridge(X, ys, lam);
    fits.push(g.map(x => polyval(c, x)));
  }
  const avg = g.map((_, i) => mean(fits.map(f => f[i])));
  const bias2 = mean(g.map((x, i) => (avg[i] - truth(x)) ** 2)), variance = mean(g.map((_, i) => mean(fits.map(f => (f[i] - avg[i]) ** 2))));
  return { g, fits, avg, truth, bias2, variance };
}

export function BiasVariance() {
  const [deg, setDeg] = useState(5), [logl, setLogl] = useState(-6);
  const d = useMemo(() => bvData(deg, 10 ** logl), [deg, logl]);
  return (
    <>
      <p>Bias and variance are facts about <b>many possible training sets</b>, so here are 60 of them, each fit separately. Spread is variance. The bold curve's miss from the dashed truth is bias.</p>
      <M d t="\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\big(\bar f(x)-f^*(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb E\big[(\hat f(x)-\bar f(x))^2\big]}_{\text{variance}}+\sigma^2" />
      <Row>
        <Plot x={[-1, 1]} y={[-2.5, 2.5]} xl="x" yl="y" label="fitted curves from many resampled datasets">
          {d.fits.slice(0, 30).map((f, i) => <Mark key={i} tone="muted" soft data={d.g.map((x, j) => [x, clamp(f[j], -3, 3)])} />)}
          <Mark tone={3} dash data={d.g.map(x => [x, d.truth(x)])} />
          <Mark tone={1} data={d.g.map((x, i) => [x, clamp(d.avg[i], -3, 3)])} />
        </Plot>
        <Stack>
          <Slider label="polynomial degree" value={deg} min={0} max={9} step={1} onChange={setDeg} />
          <Slider label={<>ridge <M t="\log_{10}\lambda" /></>} value={logl} min={-6} max={1} step={0.1} onChange={setLogl} show={v => fmt(v, 1)} />
          <Stat big>bias² {fmt(d.bias2, 3)}</Stat>
          <Stat big>variance {fmt(d.variance, 3)}</Stat>
          <Stat>sum {fmt(d.bias2 + d.variance, 3)} (noise floor adds 0.123)</Stat>
          <Legend items={[['--c1', 'average fit'], ['--c3', 'truth'], ['--muted', 'individual fits']]} />
        </Stack>
      </Row>
      <Takeaway>More capacity lowers bias and raises variance, and λ pulls the other way. Set degree 9 and raise λ: a flexible model can be tamed without removing a feature.</Takeaway>
    </>
  );
}

/* ---------- L07: the geometry of penalties ---------- */

export function PenaltyBalls() {
  const [c, setC] = useState([2.1, 1]), [t, setT] = useState(1.2), [a, setA] = useState(1);
  const e = [[2, 0.9], [0.9, 1]], L = (x, y) => { const u = x - c[0], v = y - c[1]; return e[0][0] * u * u + 2 * e[0][1] * u * v + e[1][1] * v * v; };
  const radius = s => (a > 0.999 ? t / s : a < 0.001 ? Math.sqrt(t) : (-a * s + Math.sqrt(a * a * s * s + 4 * (1 - a) * t)) / (2 * (1 - a)));
  const ring = useMemo(() => linspace(0, 2 * Math.PI, 1441).map(p => { const co = Math.cos(p), si = Math.sin(p), r = radius(Math.abs(co) + Math.abs(si)); return [r * co, r * si]; }), [t, a]); // eslint-disable-line
  const penalty = (x, y) => a * (Math.abs(x) + Math.abs(y)) + (1 - a) * (x * x + y * y);
  const sol = penalty(...c) <= t ? c : ring.reduce((b, p) => (L(...p) < L(...b) ? p : b), ring[0]);
  const sparse = Math.abs(sol[0]) < 0.012 || Math.abs(sol[1]) < 0.012, max = 30;
  return (
    <>
      <p>Penalized regression is <b>the first point where the loss contours touch the budget</b>. Drag the unpenalized solution and watch which point of the ball is touched. Corners are what create exact zeros.</p>
      <M d t="\min_\beta\ \|y-X\beta\|^2\ \ \text{s.t.}\ \ \alpha\|\beta\|_1+(1-\alpha)\|\beta\|_2^2\le t" />
      <Row>
        <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} w={380} h={380} xl="β₁" yl="β₂" label="loss contours touching the constraint region">
          <Mark as="field" bands={12} tones={[2, 2]} fn={(x, y) => Math.sqrt(clamp(L(x, y) / max, 0, 1))} />
          <Mark tone={1} data={ring} />
          <Mark as="points" tone={4} r={6} data={[sol]} />
          <Mark as="handles" data={[c]} onDrag={(_, x, y) => setC([x, y])} />
        </Plot>
        <Stack>
          <Slider label={<>budget <M t="t" /></>} value={t} min={0.1} max={4} step={0.05} onChange={setT} show={v => fmt(v)} />
          <Slider label={<>mix <M t="\alpha" /></>} value={a} min={0} max={1} step={0.05} onChange={setA} show={v => (v === 1 ? 'lasso' : v === 0 ? 'ridge' : fmt(v))} />
          <Stat big><M t={`\\hat\\beta=(${fmt(sol[0])},\\ ${fmt(sol[1])})`} /></Stat>
          <Status as="div" tone={sparse ? 'ok' : undefined}>{sparse ? 'Sparse: one coefficient is exactly zero.' : 'Both coefficients are nonzero.'}</Status>
          <Note>The draggable point is the OLS solution, and the red dot is the penalized one.</Note>
        </Stack>
      </Row>
      <Takeaway>A round ball has no preferred axis, so ridge shrinks and rarely zeros. The diamond's corners sit on the axes, so contours often touch there. Elastic net keeps rounded edges and sharp corners.</Takeaway>
    </>
  );
}

/* ---------- L07: coefficient paths ---------- */

function regData() {
  const r = rng(17), n = 60, raw = Array.from({ length: n }, () => { const z = gauss(r); return [z + 0.25 * gauss(r), z + 0.25 * gauss(r), gauss(r), gauss(r)]; });
  const X = raw.map(row => row.map((v, j) => { const c = raw.map(q => q[j]), m = mean(c), s = Math.sqrt(mean(c.map(u => (u - m) ** 2))); return (v - m) / s; }));
  const y0 = X.map(row => 2 * row[0] + 1.2 * row[2] + 0.5 * gauss(r)), my = mean(y0);
  return { X, y: y0.map(v => v - my) };
}
const REG = regData();

export function Paths() {
  const [k, setK] = useState('lasso'), [lg, setLg] = useState(-1);
  const grid = useMemo(() => linspace(-3, 1, 61), []);
  const path = useMemo(() => grid.map(l => (k === 'ridge' ? ridge(REG.X, REG.y, 10 ** l) : lasso(REG.X, REG.y, 10 ** l))), [k, grid]);
  const cur = k === 'ridge' ? ridge(REG.X, REG.y, 10 ** lg) : lasso(REG.X, REG.y, 10 ** lg);
  const ols = ridge(REG.X, REG.y, 1e-6);
  return (
    <>
      <p>Features 1 and 2 are near-duplicates, 3 matters, 4 is pure noise. Slide λ and see how each method <b>handles the duplicates</b> and when each coefficient dies.</p>
      <Toolbar><Choice label="Penalty" value={k} onChange={setK} options={[{ value: 'lasso', label: 'Lasso' }, { value: 'ridge', label: 'Ridge' }]} /></Toolbar>
      <Row>
        <Plot x={[-3, 1]} y={[-0.5, 2.5]} xl="log₁₀ λ" yl="coefficient" label="coefficient paths against penalty">
          {[0, 1, 2, 3].map(j => <Mark key={j} tone={TONES[j]} data={grid.map((l, i) => [l, path[i][j]])} />)}
          <Mark as="rule" dash x={lg} />
          <Mark as="handles" lockY data={[[lg, 2.3]]} onDrag={(_, x) => setLg(x)} />
        </Plot>
        <Stack>
          <Table>
            <thead><tr><th /><th>β₁</th><th>β₂</th><th>β₃</th><th>β₄</th></tr></thead>
            <tbody>
              <tr><th>OLS</th>{ols.map((v, i) => <td key={i}>{fmt(v)}</td>)}</tr>
              <tr><th>{k}</th>{cur.map((v, i) => <td key={i}>{fmt(v)}</td>)}</tr>
            </tbody>
          </Table>
          <Stat big>{cur.filter(v => Math.abs(v) > 1e-9).length} of 4 nonzero</Stat>
          <Legend items={[['--c1', 'x₁ (twin)'], ['--c2', 'x₂ (twin)'], ['--c3', 'x₃ (real)'], ['--c4', 'x₄ (noise)']]} />
        </Stack>
      </Row>
      <Takeaway>Ridge splits credit evenly between twins. Lasso tends to pick one arbitrarily. Neither view is the truth about which twin causes y.</Takeaway>
    </>
  );
}

/* ---------- L07: soft thresholding ---------- */

export function SoftThreshold() {
  const [lam, setLam] = useState(1), [z, setZ] = useState(2), g = linspace(-4, 4);
  return (
    <>
      <p>For a single standardized feature every method is a function of the OLS estimate <M t="z" />. Compare a <b>scale</b> with a <b>shift-and-kill</b>.</p>
      <M d t="\hat\beta^{\text{ridge}}=\frac{z}{1+\lambda},\qquad \hat\beta^{\text{lasso}}=\operatorname{sign}(z)\max(|z|-\lambda,\,0)" />
      <Row>
        <Plot x={[-4, 4]} y={[-4, 4]} w={380} h={360} xl="OLS estimate z" yl="penalized estimate" label="shrinkage functions">
          <Mark tone="muted" dash data={[[-4, -4], [4, 4]]} />
          <Mark tone={1} data={g.map(v => [v, v / (1 + lam)])} />
          <Mark tone={2} data={g.map(v => [v, soft(v, lam)])} />
          <Mark as="points" tone={1} data={[[z, z / (1 + lam)]]} />
          <Mark as="points" tone={2} data={[[z, soft(z, lam)]]} />
          <Mark as="handles" lockY data={[[z, -3.6]]} onDrag={(_, x) => setZ(x)} />
        </Plot>
        <Stack>
          <Slider label={<M t="\lambda" />} value={lam} min={0} max={3} step={0.05} onChange={setLam} show={v => fmt(v)} />
          <Stat big>ridge {fmt(z / (1 + lam), 3)}</Stat>
          <Stat big>lasso {fmt(soft(z, lam), 3)}</Stat>
          <Legend items={[['--c1', 'ridge'], ['--c2', 'lasso'], ['--muted', 'OLS (no shrinkage)']]} />
        </Stack>
      </Row>
      <Takeaway>Lasso has a dead zone of width 2λ and shifts the survivors by a constant. Ridge has no dead zone, and hits big coefficients hardest in absolute size.</Takeaway>
    </>
  );
}

/* ---------- L07: cross-validation ---------- */

function cvCurve(K, seed) {
  const r = rng(seed), xs = Array.from({ length: 30 }, () => 2 * r() - 1), ys = xs.map(x => Math.sin(Math.PI * x) + 0.3 * gauss(r));
  const folds = xs.map((_, i) => i % K), out = [];
  for (let d = 0; d <= 9; d++) {
    const tr = [], va = [];
    for (let k = 0; k < K; k++) {
      const tx = xs.filter((_, i) => folds[i] !== k), ty = ys.filter((_, i) => folds[i] !== k), vx = xs.filter((_, i) => folds[i] === k), vy = ys.filter((_, i) => folds[i] === k);
      const X = tx.map(x => Array.from({ length: d + 1 }, (_, j) => x ** j)), c = ridge(X, ty, 1e-7), e = (a, b) => mean(a.map((x, i) => (polyval(c, x) - b[i]) ** 2));
      tr.push(e(tx, ty)); va.push(e(vx, vy));
    }
    const m = mean(va), sd = Math.sqrt(mean(va.map(v => (v - m) ** 2)) / K);
    out.push({ d, tr: mean(tr), va: m, se: sd });
  }
  return { out, folds };
}

export function CrossVal() {
  const [K, setK] = useState(5), [seed, setSeed] = useState(3), [show, setShow] = useState(0);
  const { out, folds } = useMemo(() => cvCurve(K, seed), [K, seed]);
  const best = out.reduce((a, b) => (b.va < a.va ? b : a)), oneSe = out.find(o => o.va <= best.va + best.se), lg = v => Math.log10(Math.max(v, 1e-3));
  const fold = Math.min(show, K - 1);
  return (
    <>
      <p>Each point takes a turn in the held-out set. Pick the fold to see who is held out, then read the validation curve. The <b>one-standard-error rule</b> prefers the simplest model that is not measurably worse.</p>
      <Toolbar><Choice label="Folds" value={String(K)} onChange={v => setK(+v)} options={['2', '5', '10', '30'].map(v => ({ value: v, label: v === '30' ? 'LOOCV' : `${v}-fold` }))} /><Button onClick={() => setSeed(s => s + 1)}>New data</Button></Toolbar>
      <Row>
        <Plot x={[0, 9]} y={[-2, 1]} xl="degree" yl="log₁₀ error" label="train and validation error against degree">
          <Mark tone={3} data={out.map(o => [o.d, lg(o.tr)])} />
          <Mark tone={4} data={out.map(o => [o.d, lg(o.va)])} />
          <Mark as="points" tone={4} r={3} data={out.map(o => [o.d, lg(o.va)])} />
          <Mark as="rule" dash x={best.d} />
          <Mark as="rule" tone={1} x={oneSe.d} />
        </Plot>
        <Stack>
          <Slider label="held-out fold" value={fold} min={0} max={K - 1} step={1} onChange={setShow} show={v => v + 1} />
          <DotGrid cells={folds.map(f => (f === fold ? 'fn' : 'tp'))} label="fold assignment of the thirty points" />
          <Stat big>lowest error at degree {best.d}</Stat>
          <Stat>one-SE choice: degree {oneSe.d}</Stat>
          <Legend items={[['--c3', 'train'], ['--c4', 'validation'], ['--c1', 'one-SE pick']]} />
        </Stack>
      </Row>
      <Takeaway>Validation picks the model and test reports the result. If the test score ever influences a choice, it stops being a test.</Takeaway>
    </>
  );
}

/* ---------- L08: near-duplicate columns as nearly parallel lines ---------- */

export function TwoLines() {
  const [e, setE] = useState(0.2), [d, setD] = useState(0.1);
  const y1 = 2, y2 = 2 + e + d, b2 = (y2 - y1) / e, b1 = y1 - b2, g = linspace(-6, 8, 2);
  const c = eig2(2, 2 + e, 1 + (1 + e) ** 2), cond = Math.sqrt(Math.abs(c.l1 / c.l2));
  return (
    <>
      <p>Each equation is a line, and the solution is where they cross. As the two columns become duplicates the lines go parallel, so <b>a tiny change in y sends the crossing a long way</b>.</p>
      <M d t="\beta_1+\beta_2=y_1,\qquad \beta_1+(1+\varepsilon)\beta_2=y_2" />
      <Row>
        <Plot x={[-6, 8]} y={[-6, 8]} w={380} h={360} xl="β₁" yl="β₂" label="two lines crossing at the solution">
          <Mark tone={1} data={g.map(a => [a, y1 - a])} />
          <Mark tone={2} data={g.map(a => [a, (y2 - a) / (1 + e)])} />
          <Mark as="points" tone={4} r={6} data={[[clamp(b1, -6, 8), clamp(b2, -6, 8)]]} />
          <Mark as="points" tone={3} r={5} data={[[1, 1]]} />
        </Plot>
        <Stack>
          <Slider label={<>column gap <M t="\varepsilon" /></>} value={e} min={0.01} max={1} step={0.01} onChange={setE} show={v => fmt(v)} />
          <Slider label="noise in y₂" value={d} min={-0.2} max={0.2} step={0.005} onChange={setD} show={v => fmt(v, 3)} />
          <Stat big><M t={`\\hat\\beta=(${fmt(b1)},\\ ${fmt(b2)})`} /></Stat>
          <Stat>condition number ≈ {fmt(cond, 1)}</Stat>
          <Note>The green dot is the noise-free answer (1, 1).</Note>
        </Stack>
      </Row>
      <Takeaway>Nothing is wrong with the fit: predictions barely change. It is the coefficients that cannot be trusted, and they cannot be read as importance.</Takeaway>
    </>
  );
}

/* ---------- L08: the 1 - eta*mu factor ---------- */

export function GdFactor() {
  const [m1, setM1] = useState(4), [m2, setM2] = useState(0.5), [eta, setEta] = useState(0.2);
  const r1 = 1 - eta * m1, r2 = 1 - eta * m2, opt = 2 / (m1 + m2), k = Array.from({ length: 31 }, (_, i) => i);
  let p = [2.4, 2.4];
  const path = [p];
  for (let i = 0; i < 30; i++) { p = [r1 * p[0], r2 * p[1]]; path.push(p); }
  const rate = Math.max(Math.abs(r1), Math.abs(r2));
  return (
    <>
      <p>Rotate into eigen-axes and gradient descent splits into <b>independent one-dimensional problems</b>. Each coordinate is multiplied by its own factor at every step.</p>
      <M d t="c_j^{(t+1)}=(1-\eta\mu_j)\,c_j^{(t)},\qquad \text{stable iff }\ 0<\eta<\frac{2}{\mu_{\max}}" />
      <Row>
        <Plot caption="Path in eigen-coordinates" x={[-3, 3]} y={[-3, 3]} w={340} h={320} xl="along μ₁" yl="along μ₂" label="descent path">
          <Mark as="field" bands={9} tones={[1, 1]} fn={(x, y) => Math.sqrt(clamp((m1 * x * x + m2 * y * y) / (9 * (m1 + m2)), 0, 1))} />
          <Mark tone="fg" data={path} />
          <Mark as="points" tone="fg" r={2.5} data={path} />
        </Plot>
        <Plot caption="Remaining error per coordinate" x={[0, 30]} y={[0, 1.5]} w={340} h={320} xl="step" yl="|1 − ημ|ᵗ" label="decay curves">
          <Mark tone={2} data={k.map(i => [i, Math.min(Math.abs(r1) ** i, 2)])} />
          <Mark tone={1} data={k.map(i => [i, Math.min(Math.abs(r2) ** i, 2)])} />
        </Plot>
      </Row>
      <Slider label={<M t="\mu_1" />} value={m1} min={0.5} max={10} step={0.1} onChange={setM1} show={v => fmt(v, 1)} />
      <Slider label={<M t="\mu_2" />} value={m2} min={0.05} max={2} step={0.05} onChange={setM2} show={v => fmt(v)} />
      <Slider label={<M t="\eta" />} value={eta} min={0.01} max={1} step={0.01} onChange={setEta} show={v => fmt(v)} />
      <Stat big>factors {fmt(r1, 3)} and {fmt(r2, 3)}; best η = {fmt(opt, 3)}</Stat>
      <Status as="div" tone={rate >= 1 ? 'alert' : 'ok'}>{rate >= 1 ? 'A factor exceeds 1 in size, so that coordinate diverges.' : `Converging: the slower coordinate shrinks to ${pct(rate, 0)} of itself each step.`}</Status>
      <Takeaway>A negative factor means zig-zag across the valley. The condition number decides how many steps the slow direction costs.</Takeaway>
    </>
  );
}

/* ---------- L08: lasso KKT ---------- */

export function LassoKkt() {
  const [lg, setLg] = useState(-0.6), lam = 10 ** lg, b = useMemo(() => lasso(REG.X, REG.y, lam), [lam]);
  const n = REG.X.length, r = REG.y.map((v, i) => v - REG.X[i].reduce((s, x, j) => s + x * b[j], 0));
  const corr = [0, 1, 2, 3].map(j => sum(REG.X.map((row, i) => row[j] * r[i])) / n), top = Math.max(2.2, lam * 1.2);
  return (
    <>
      <p>Lasso's optimality has a clean reading: a coefficient stays at zero <b>exactly when its feature's correlation with the residual is under λ</b>. Bars at the line are active. Bars below it are dead.</p>
      <M d t="\beta_j\ne0\ \Rightarrow\ \tfrac1n x_j^\top r=\lambda\operatorname{sign}\beta_j,\qquad \beta_j=0\ \Rightarrow\ \big|\tfrac1n x_j^\top r\big|\le\lambda" />
      <Row>
        <Plot x={[-0.5, 3.5]} y={[0, +top.toFixed(2)]} xl="feature index (0 to 3)" yl="|correlation with residual|" label="feature correlations against the lasso threshold">
          <Mark as="rects" tone={1} data={corr.map((v, j) => [j - 0.3, 0, j + 0.3, Math.abs(v)])} />
          <Mark as="rule" tone={4} y={lam} />
        </Plot>
        <Stack>
          <Slider label={<M t="\log_{10}\lambda" />} value={lg} min={-3} max={0.4} step={0.02} onChange={setLg} show={v => fmt(v)} />
          <Table>
            <thead><tr><th /><th>β₁</th><th>β₂</th><th>β₃</th><th>β₄</th></tr></thead>
            <tbody>
              <tr><th>value</th>{b.map((v, i) => <td key={i}>{fmt(v, 3)}</td>)}</tr>
              <tr><th>|corr|</th>{corr.map((v, i) => <td key={i}>{fmt(Math.abs(v), 3)}</td>)}</tr>
            </tbody>
          </Table>
          <Note>Above the largest starting correlation, every coefficient is zero: that λ is the top of the path.</Note>
        </Stack>
      </Row>
      <Takeaway>This condition is also how you check a solver: if any zero coefficient's correlation exceeds λ, the solution is wrong.</Takeaway>
    </>
  );
}
