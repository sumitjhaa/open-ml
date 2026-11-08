'use client';
import { useMemo, useState } from 'react';
import M from '../components/M';
import { Mark, Plot, moved } from '../components/Plot';
import { Choice, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from '../components/ui';
import { mean, rng } from '../lib/math.mjs';
import { clamp, fmt, linspace, normalCdf, normalPdf, pct, rocCurve, sigmoid } from '../lib/models.mjs';

const rates = (t, d) => ({ tpr: 1 - normalCdf(t, d), fpr: 1 - normalCdf(t, 0) });
const rocPts = d => linspace(-5, d + 5, 161).reverse().map(t => { const r = rates(t, d); return [r.fpr, r.tpr]; });
const aucOf = d => normalCdf(d / Math.SQRT2);

/* ---------- L09: three linked views of one threshold ---------- */

export function ThresholdLab() {
  const [t, setT] = useState(0.8), [d, setD] = useState(1.5), [prev, setPrev] = useState(0.3);
  const g = linspace(-4, 6, 241), N = 1000, { tpr, fpr } = rates(t, d), P = prev * N, Ne = N - P;
  const TP = tpr * P, FN = P - TP, FP = fpr * Ne, TN = Ne - FP, prec = TP / (TP + FP), acc = (TP + TN) / N, f1 = (2 * prec * tpr) / (prec + tpr);
  const pr = linspace(-5, d + 5, 161).map(x => { const r = rates(x, d); const q = (r.tpr * prev) / (r.tpr * prev + r.fpr * (1 - prev)); return [r.tpr, Number.isFinite(q) ? q : 1]; });
  const dens = x => [prev * normalPdf(x, d), (1 - prev) * normalPdf(x, 0)];
  return (
    <>
      <p>One threshold, three views. <b>Drag the threshold</b> on the score plot and see the same decision move on the ROC and precision-recall curves.</p>
      <div className="minis">
        <Plot caption="Scores, weighted by prevalence" x={[-4, 6]} y={[0, 0.42]} w={280} h={250} xl="model score" label="class score densities with threshold">
          <Mark as="area" tone={4} data={g.filter(x => x < t).map(x => [x, 0, dens(x)[0]])} />
          <Mark as="area" tone={2} data={g.filter(x => x >= t).map(x => [x, 0, dens(x)[1]])} />
          <Mark tone={3} data={g.map(x => [x, dens(x)[0]])} />
          <Mark tone="muted" data={g.map(x => [x, dens(x)[1]])} />
          <Mark as="rule" tone="fg" x={t} />
          <Mark as="handles" lockY data={[[t, 0.39]]} onDrag={(_, x) => setT(x)} />
        </Plot>
        <Plot caption={`ROC (AUC ${fmt(aucOf(d), 3)})`} x={[0, 1]} y={[0, 1]} w={280} h={250} xl="FPR" yl="TPR" label="ROC curve">
          <Mark tone="muted" dash data={[[0, 0], [1, 1]]} />
          <Mark tone={1} data={rocPts(d)} />
          <Mark as="points" tone={4} r={6} data={[[fpr, tpr]]} />
        </Plot>
        <Plot caption="Precision-recall" x={[0, 1]} y={[0, 1]} w={280} h={250} xl="recall" yl="precision" label="precision recall curve">
          <Mark as="rule" dash y={prev} />
          <Mark tone={2} data={pr} />
          <Mark as="points" tone={4} r={6} data={[[tpr, prec]]} />
        </Plot>
      </div>
      <Row>
        <Stack>
          <Slider label="class separation" value={d} min={0} max={4} step={0.05} onChange={setD} show={v => fmt(v)} />
          <Slider label="prevalence" value={prev} min={0.01} max={0.9} step={0.01} onChange={setPrev} show={v => pct(v, 0)} />
          <Legend items={[['--c3', 'positives'], ['--muted', 'negatives'], ['--c4', 'false negatives'], ['--c2', 'false positives']]} />
        </Stack>
        <Stack>
          <Table className="cm">
            <thead><tr><th>1000 cases</th><th>pred +</th><th>pred −</th></tr></thead>
            <tbody>
              <tr><th>actual +</th><td className="tp">{TP.toFixed(0)}</td><td className="fn">{FN.toFixed(0)}</td></tr>
              <tr><th>actual −</th><td className="fp">{FP.toFixed(0)}</td><td>{TN.toFixed(0)}</td></tr>
            </tbody>
          </Table>
          <Stat>accuracy {pct(acc)} · precision {pct(prec)} · recall {pct(tpr)} · F1 {fmt(f1, 3)}</Stat>
        </Stack>
      </Row>
      <Takeaway>A model has one ROC curve but a threshold is a business decision. Change only prevalence: the ROC stays put while precision moves.</Takeaway>
    </>
  );
}

/* ---------- L09: calibration ---------- */

const CAL = (() => { const r = rng(31); return Array.from({ length: 4000 }, () => { const p = clamp(0.02 + 0.96 * r(), 0.001, 0.999); return { z: Math.log(p / (1 - p)), y: +(r() < p) }; }); })();

export function Calibration() {
  const [T, setT] = useState(2), [shift, setShift] = useState(0), [fix, setFix] = useState('raw');
  const q = z => sigmoid(fix === 'fixed' ? z : T * z + shift);
  const bins = useMemo(() => Array.from({ length: 10 }, (_, b) => { const s = CAL.filter(c => { const v = q(c.z); return v >= b / 10 && v < (b + 1) / 10; }); return { n: s.length, conf: s.length ? mean(s.map(c => q(c.z))) : NaN, acc: s.length ? mean(s.map(c => c.y)) : NaN }; }), [T, shift, fix]); // eslint-disable-line
  const used = bins.filter(b => b.n), ece = used.reduce((a, b) => a + (b.n / CAL.length) * Math.abs(b.conf - b.acc), 0);
  const AUC = useMemo(() => { const s = CAL.slice(0, 800); return rocCurve(s.map(c => q(c.z)), s.map(c => c.y)).auc; }, [T, shift, fix]); // eslint-disable-line
  return (
    <>
      <p>When a calibrated model says 70%, it should be right 70% of the time. Each dot is a bin of predictions: <b>distance from the diagonal is the lie</b>.</p>
      <M d t="\text{ECE}=\sum_b \frac{n_b}{N}\,\big|\text{accuracy}_b-\text{confidence}_b\big|" />
      <Toolbar><Choice label="Calibration" value={fix} onChange={setFix} options={[{ value: 'raw', label: 'as trained' }, { value: 'fixed', label: 'after recalibration' }]} /></Toolbar>
      <Row>
        <Plot x={[0, 1]} y={[0, 1]} w={360} h={340} xl="predicted probability" yl="observed frequency" label="reliability diagram">
          <Mark tone="muted" dash data={[[0, 0], [1, 1]]} />
          <Mark as="stems" tone={4} data={used.map(b => [b.conf, b.conf, b.acc])} />
          <Mark tone={1} data={used.map(b => [b.conf, b.acc])} />
          <Mark as="points" tone={1} data={used.map(b => [b.conf, b.acc, undefined, 3 + 5 * Math.sqrt(b.n / CAL.length)])} />
        </Plot>
        <Stack>
          <Slider label="overconfidence" value={T} min={0.3} max={4} step={0.05} onChange={setT} show={v => fmt(v)} />
          <Slider label="bias shift" value={shift} min={-2} max={2} step={0.05} onChange={setShift} show={v => fmt(v)} />
          <Stat big>ECE {fmt(ece, 3)}</Stat>
          <Stat>AUC {fmt(AUC, 3)}</Stat>
          <Note>Overconfidence keeps the ranking, so AUC does not budge even as probabilities go badly wrong.</Note>
        </Stack>
      </Row>
      <Takeaway>Ranking and calibration are different skills. If decisions use the probabilities themselves, as in expected cost, you must check the second.</Takeaway>
    </>
  );
}

/* ---------- L09: expected cost ---------- */

export function CostThreshold() {
  const [cfp, setCfp] = useState(1), [cfn, setCfn] = useState(5), [prev, setPrev] = useState(0.2), d = 1.8;
  const xOf = p => (Math.log(p / (1 - p)) - Math.log(prev / (1 - prev)) + (d * d) / 2) / d;
  const cost = p => { const r = rates(xOf(p), d); return prev * (1 - r.tpr) * cfn + (1 - prev) * r.fpr * cfp; };
  const g = linspace(0.005, 0.995, 199), star = cfp / (cfp + cfn), r = rates(xOf(star), d), slope = ((1 - prev) * cfp) / (prev * cfn);
  const iso = [[0, r.tpr - slope * r.fpr], [1, r.tpr + slope * (1 - r.fpr)]], top = Math.max(...g.map(cost)) * 1.1, naive = cost(0.5);
  return (
    <>
      <p>Act when the expected cost of doing nothing beats the expected cost of acting. <b>The threshold falls out of the costs</b>, with no search required.</p>
      <M d t="\text{flag if }\ p\,C_{FN}>(1-p)\,C_{FP}\quad\Longleftrightarrow\quad p>t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}" />
      <Row>
        <Plot caption="Expected cost per case" x={[0, 1]} y={[0, +top.toFixed(3)]} w={340} h={260} xl="probability threshold" label="expected cost against threshold">
          <Mark tone={1} data={g.map(p => [p, cost(p)])} />
          <Mark as="rule" tone={3} x={star} />
          <Mark as="rule" dash x={0.5} />
        </Plot>
        <Plot caption="The same choice on the ROC" x={[0, 1]} y={[0, 1]} w={260} h={260} xl="FPR" yl="TPR" label="ROC with iso-cost line">
          <Mark tone="muted" data={rocPts(d)} />
          <Mark tone={2} dash data={iso} />
          <Mark as="points" tone={4} r={6} data={[[r.fpr, r.tpr]]} />
        </Plot>
      </Row>
      <Slider label="false-alarm cost" value={cfp} min={0.1} max={10} step={0.1} onChange={setCfp} show={v => fmt(v, 1)} />
      <Slider label="missed-case cost" value={cfn} min={0.1} max={50} step={0.1} onChange={setCfn} show={v => fmt(v, 1)} />
      <Slider label="prevalence" value={prev} min={0.01} max={0.9} step={0.01} onChange={setPrev} show={v => pct(v, 0)} />
      <Stat big>t* = {fmt(star, 3)} · cost {fmt(cost(star), 3)} vs {fmt(naive, 3)} at the default 0.5</Stat>
      <Takeaway>0.5 is only right when the two errors cost the same. On the ROC, the best point is where a line of slope (1−π)C_FP / (πC_FN) just touches the curve.</Takeaway>
    </>
  );
}

/* ---------- L10: walk a ranked list ---------- */

const RANK = [[0.95, 1], [0.88, 1], [0.8, 0], [0.74, 1], [0.66, 1], [0.59, 0], [0.5, 1], [0.44, 0], [0.37, 0], [0.3, 1], [0.21, 0], [0.12, 0]].map(([s, y]) => [s, y ? 0.7 : 0.3, y ? 3 : 4]);

export function RocWalk() {
  const [pts, setPts] = useState(RANK), [k, setK] = useState(5);
  const scores = pts.map(p => p[0]), labels = pts.map(p => p[2] === 3), { pts: curve, auc } = rocCurve(scores, labels);
  const order = pts.map((_, i) => i).sort((a, b) => scores[b] - scores[a]), step = Math.min(k, curve.length - 1), cut = step === 0 ? 1.01 : pts[order[Math.min(step, order.length) - 1]][0] - 1e-6;
  const P = labels.filter(Boolean).length, N = labels.length - P;
  let pairs = 0;
  labels.forEach((y, i) => { if (y) labels.forEach((z, j) => { if (!z) pairs += scores[i] > scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0; }); });
  return (
    <>
      <p>Sort by score and walk from the top. Each positive steps <b>up</b>, each negative steps <b>right</b>. Drag scores to reorder the list and the staircase redraws.</p>
      <M d t="\text{AUC}=P\big(\text{score}(+)>\text{score}(-)\big)" />
      <Row>
        <Plot caption="Scores (top lane positive, bottom negative)" x={[0, 1]} y={[0, 1]} w={360} h={210} xl="score" label="scores along a line">
          <Mark as="rule" tone="fg" x={clamp(cut, 0, 1)} />
          <Mark as="handles" r={7} lockY data={pts} onDrag={(i, x, y) => setPts(moved(i, clamp(x, 0, 1), y))} />
        </Plot>
        <Plot caption="ROC staircase" x={[0, 1]} y={[0, 1]} w={260} h={260} xl="FPR" yl="TPR" label="ROC staircase">
          <Mark tone="muted" dash data={[[0, 0], [1, 1]]} />
          <Mark tone="muted" data={curve} />
          <Mark tone={1} data={curve.slice(0, step + 1)} />
          <Mark as="points" tone={4} r={5} data={[curve[step]]} />
        </Plot>
      </Row>
      <Slider label="walk down the list" value={Math.min(k, 12)} min={0} max={12} step={1} onChange={setK} />
      <Stat big>AUC {fmt(auc, 3)} = {fmt(pairs, 1)} winning pairs of {P * N}</Stat>
      <Takeaway>Only the order matters. Apply any monotone squeeze to the scores and the AUC is unchanged, which is why it says nothing about calibration.</Takeaway>
    </>
  );
}

/* ---------- L10: prevalence ---------- */

export function PrevalencePR() {
  const [prev, setPrev] = useState(0.5), [d, setD] = useState(1.5);
  const g = linspace(-5, d + 5, 161), p = (x, pi) => { const r = rates(x, d); const v = (r.tpr * pi) / (r.tpr * pi + r.fpr * (1 - pi)); return [r.tpr, Number.isFinite(v) ? v : 1]; };
  const ap = pi => { const c = g.map(x => p(x, pi)); return c.slice(1).reduce((s, q, i) => s + (c[i][0] - q[0]) * q[1], 0); };
  return (
    <>
      <p>Same model, same scores, only the mix of classes changes. Slide prevalence and compare the two curves' reactions.</p>
      <M d t="\text{precision}=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}" />
      <Row>
        <Plot caption="ROC is blind to prevalence" x={[0, 1]} y={[0, 1]} w={300} h={280} xl="FPR" yl="TPR" label="ROC curve">
          <Mark tone="muted" dash data={[[0, 0], [1, 1]]} />
          <Mark tone={1} data={rocPts(d)} />
        </Plot>
        <Plot caption="Precision-recall is not" x={[0, 1]} y={[0, 1]} w={300} h={280} xl="recall" yl="precision" label="precision recall curves">
          <Mark tone="muted" dash data={g.map(x => p(x, 0.5))} />
          <Mark tone={2} data={g.map(x => p(x, prev))} />
          <Mark as="rule" dash y={prev} />
        </Plot>
      </Row>
      <Slider label="prevalence" value={prev} min={0.005} max={0.9} step={0.005} onChange={setPrev} show={v => pct(v)} />
      <Slider label="class separation" value={d} min={0.2} max={4} step={0.05} onChange={setD} show={v => fmt(v)} />
      <Stat big>AUC {fmt(aucOf(d), 3)} · average precision {fmt(ap(prev), 3)} · random baseline {fmt(prev, 3)}</Stat>
      <Takeaway>On rare events a high AUC can hide a precision of a few percent. Report average precision against its baseline, which is the prevalence and not 0.5.</Takeaway>
    </>
  );
}

/* ---------- L10: means of precision and recall ---------- */

export function FMean() {
  const [P, setP] = useState(0.9), [R, setR] = useState(0.2), [beta, setBeta] = useState(1);
  const am = (P + R) / 2, fb = ((1 + beta * beta) * P * R) / (beta * beta * P + R || 1), Fb = (p, r) => ((1 + beta * beta) * p * r) / (beta * beta * p + r || 1);
  return (
    <>
      <p>The shading is <M t="F_\beta" /> over every precision-recall pair. <b>Drag the point</b>: a harmonic mean is dragged towards the smaller of its two inputs.</p>
      <M d t="F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2P+R}" />
      <Row>
        <Plot x={[0, 1]} y={[0, 1]} w={360} h={340} xl="recall" yl="precision" label="F-beta contours">
          <Mark as="field" bands={10} tones={[1, 1]} fn={(r, p) => Fb(p, r)} />
          <Mark as="handles" data={[[R, P]]} onDrag={(_, r, p) => { setR(r); setP(p); }} />
        </Plot>
        <Stack>
          <Slider label={<M t="\beta" />} value={beta} min={0.25} max={4} step={0.05} onChange={setBeta} show={v => fmt(v)} />
          <Stat big><M t={`F_{${fmt(beta, 2)}}=${fmt(fb, 3)}`} /></Stat>
          <Stat>arithmetic mean {fmt(am, 3)}</Stat>
          <Status as="div" tone={am - fb > 0.2 ? 'alert' : 'ok'}>{am - fb > 0.2 ? 'The arithmetic mean is flattering a lopsided model.' : 'Balanced: the two means roughly agree.'}</Status>
          <Note>β &gt; 1 weighs recall more, and β &lt; 1 weighs precision more.</Note>
        </Stack>
      </Row>
      <Takeaway>A flag-everything model has recall 1 and precision equal to prevalence. The harmonic mean refuses to let one good number cover for a bad one.</Takeaway>
    </>
  );
}
