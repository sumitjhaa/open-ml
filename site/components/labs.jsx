'use client';
// Builders for the three explorer formats that repeat across the course, so each explorer is a short description:
//   lab     relationships and algorithms: sliders, choices, draggable points, linked plots, step-through
//   drill   formulas and procedures: a fresh problem each time, with the working revealed one step at a time
//   truths  common confusions: judge each statement, then read why
import { useDeferredValue, useMemo, useState } from 'react';
import M from './M';
import { rich } from './rich';
import { Mark, Plot } from './Plot';
import { Button, Choice, Field, Legend, Note, Row, Slider, Stack, Stat, Status, Table, Takeaway, Toolbar } from './ui';
import { rng } from '../lib/math.mjs';
import { fmt, linspace } from '../lib/models.mjs';

export { rich };
export const R = String.raw;
export const curve = (f, a, b, n = 201) => linspace(a, b, n).map(x => [x, f(x)]);
export const int = (r, a, b) => a + Math.floor(r() * (b - a + 1));
export const pick = (r, a) => a[Math.floor(r() * a.length)];
export const mat = A => R`\begin{bmatrix}${A.map(row => row.join('&')).join(R`\\`)}\end{bmatrix}`;
export const heatPlot = (M, o = {}) => ({ bare: true, x: [0, M[0].length], y: [0, M.length], w: o.w ?? 400, h: o.h ?? Math.round(30 + (o.w ?? 400) * (M.length / M[0].length) * 0.86), label: o.label ?? 'heatmap', caption: o.caption, marks: [{ as: 'heat', data: M, ...o.heat }, ...(o.over ?? [])] });
// Outline a cell of a heatPlot; a null index outlines the whole row or column.
export const heatBox = (M, i, j) => { const R = M.length, C = M[0].length, x0 = j ?? 0, x1 = j == null ? C : j + 1, y1 = i == null ? R : R - i, y0 = i == null ? 0 : R - i - 1; return { tone: 'fg', data: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] }; };
// Stacked densities: one baseline per row, so a change across layers or time reads at a glance.
export const ridge = (rows, x, h = 0.9) => { const top = Math.max(1e-9, ...rows.flatMap(r => r.d)), r6 = v => +v.toFixed(6); return rows.flatMap((r, i) => { const base = rows.length - 1 - i; return [{ as: 'area', tone: r.tone ?? 1, data: x.map((v, k) => [v, base, r6(base + (h * r.d[k] * 1.5) / top)]) }, { tone: r.tone ?? 1, data: x.map((v, k) => [v, r6(base + (h * r.d[k] * 1.5) / top)]) }, { as: 'label', x: x[0], y: base + 0.1, text: r.label }]; }); };
const smooth = c => c.map((_, i) => (0.25 * (c[i - 1] ?? 0) + 0.5 * c[i] + 0.25 * (c[i + 1] ?? 0)));
export const histDensity = (xs, a, b, n) => { const c = Array(n).fill(0), w = (b - a) / n; xs.forEach(v => { const i = Math.floor((v - a) / w); if (i >= 0 && i < n) c[i]++; }); return smooth(smooth(smooth(c))).map(v => v / (xs.length * w)); };


const places = step => (String(step).split('.')[1] ?? '').length;

export function lab({ intro, tex, params = [], derive, plots, stats, status, table, legend, note, takeaway }) {
  return function Lab() {
    const [p, setP] = useState(() => Object.fromEntries(params.map(q => [q.k, q.init ?? 1])));
    // Sliders follow the pointer at once; the sums and charts catch up when the browser is free.
    const shown = useDeferredValue(p);
    const set = (k, val) => setP(o => ({ ...o, [k]: val }));
    const snap = (k, val) => {
      const q = params.find(x => x.k === k);
      if (!q) return;
      set(k, +Math.min(q.max, Math.max(q.min, q.min + Math.round((val - q.min) / q.step) * q.step)).toFixed(places(q.step)));
    };
    const { figs, lines, st, t } = useMemo(() => {
    const v = { ...shown, ...(derive ? derive(shown) : {}) };
    const renderMark = (m, i) => {
      if (!m.drag) return <Mark key={i} {...m} />;
      const { drag: { x, y, tone }, ...rest } = m, X = typeof x === 'string' ? v[x] : x, Y = typeof y === 'string' ? v[y] : y;
      return <Mark key={i} as="handles" lockX={typeof x !== 'string'} lockY={typeof y !== 'string'} data={[[X, Y, tone]]}
        onDrag={(_, a, b) => { if (typeof x === 'string') snap(x, a); if (typeof y === 'string') snap(y, b); }} {...rest} />;
    };
    return { figs: (plots?.(v) ?? []).map(({ marks, ...f }, i) => <Plot key={i} {...f}>{marks.filter(Boolean).map(renderMark)}</Plot>), lines: stats?.(v) ?? [], st: status?.(v), t: table?.(v) };
    }, [shown]); // eslint-disable-line react-hooks/exhaustive-deps
    const choices = params.filter(q => q.options), seeds = params.filter(q => q.seed), sliders = params.filter(q => q.min !== undefined);
    const panel = (
      <Stack>
        {sliders.map(q => (
          <div key={q.k}>
            <Slider label={rich(q.label)} value={p[q.k]} min={q.min} max={q.max} step={q.step} onChange={val => set(q.k, val)} show={val => (q.show ? q.show(val) : fmt(val, places(q.step)))} />
            {q.play && <Toolbar><Button onClick={() => snap(q.k, p[q.k] - q.step)}>Previous</Button><Button variant="solid" onClick={() => snap(q.k, p[q.k] + q.step)}>Next</Button></Toolbar>}
          </div>
        ))}
        {lines.map((s, i) => <Stat key={i} big={i === 0}>{rich(s)}</Stat>)}
        {st && <Status as="div" tone={st[0]}>{rich(st[1])}</Status>}
        {t && <Table><thead><tr>{t.head.map((h, i) => <th key={i}>{rich(h)}</th>)}</tr></thead><tbody>{t.rows.map((r, i) => <tr key={i}>{r.map((c, j) => (j ? <td key={j}>{rich(c)}</td> : <th key={j}>{rich(c)}</th>))}</tr>)}</tbody></Table>}
        {legend && <Legend items={legend} />}
        {note && <Note>{rich(note)}</Note>}
      </Stack>
    );
    return (
      <>
        {intro && <p>{rich(intro)}</p>}
        {tex && <M d t={tex} />}
        {(choices.length > 0 || seeds.length > 0) && (
          <Toolbar>
            {choices.map(q => <Choice key={q.k} label={q.label} value={p[q.k]} onChange={val => set(q.k, val)} options={q.options.map(([value, label]) => ({ value, label }))} />)}
            {seeds.map(q => <Button key={q.k} onClick={() => set(q.k, p[q.k] + 1)}>{q.seed}</Button>)}
          </Toolbar>
        )}
        {figs.length === 1 ? <Row>{figs}{panel}</Row> : <><div className="minis">{figs}</div>{panel}</>}
        {takeaway && <Takeaway>{rich(takeaway)}</Takeaway>}
      </>
    );
  };
}

const norm = s => String(s).toLowerCase().replace(/[\s()]/g, '').replace(/×/g, 'x');
function parse(s) {
  const t = s.trim(), m = t.match(/^(-?[\d.]+)\s*\/\s*(-?[\d.]+)$/);
  return m ? +m[1] / +m[2] : t === '' ? NaN : +t;
}

// gen(r, n): n = 0 is the first problem shown, so a tutorial can open on its own numbers.
// answer: a number, a string (matched loosely, e.g. "3x4"), or with `choices` the index of the right option.
export function drill({ intro, tex, gen, q, answer, steps, choices, dp = 2, unit, takeaway }) {
  return function Drill() {
    const [n, setN] = useState(0), [text, setText] = useState(''), [result, setResult] = useState(null), [shown, setShown] = useState(0), [streak, setStreak] = useState(0), [tries, setTries] = useState(0);
    const v = useMemo(() => gen(rng(1000 + n * 7919), n), [n]);
    const a = answer(v), opts = choices?.(v), all = steps(v);
    const grade = ok => { setResult(ok); setTries(t => t + 1); setStreak(s => (ok ? s + 1 : 0)); };
    const check = () => {
      if (typeof a === 'string') return grade(norm(text) === norm(a));
      const x = parse(text);
      grade(Number.isFinite(x) && Math.abs(x - a) <= Math.max(0.51 * 10 ** -dp, Math.abs(a) * 0.002));
    };
    const next = () => { setN(i => i + 1); setText(''); setResult(null); setShown(0); setTries(0); };
    const shownAnswer = opts ? opts[a] : typeof a === 'string' ? a : fmt(a, dp);
    return (
      <>
        {intro && <p>{rich(intro)}</p>}
        {tex && <M d t={tex} />}
        <div className="drill">
          <div className="drill-q">{rich(q(v))}</div>
          {opts ? (
            <Toolbar>{opts.map((o, i) => <Button key={i} onClick={() => grade(i === a)}>{rich(o)}</Button>)}</Toolbar>
          ) : (
            <form className="drill-a" onSubmit={e => { e.preventDefault(); check(); }}>
              <Field value={text} onChange={e => setText(e.target.value)} placeholder={typeof a === 'string' ? 'your answer' : 'a number or fraction'} aria-label="Your answer" autoComplete="off" />
              {unit && <span className="note">{unit}</span>}
              <Button type="submit" variant="solid">Check</Button>
            </form>
          )}
          {result !== null && <Status as="div" tone={result ? 'ok' : 'alert'} role="status">{result ? <>Correct: {rich(shownAnswer)}</> : tries > 1 ? 'Not yet. Reveal a step below and try again.' : 'Not yet. Try once more.'}</Status>}
          <Toolbar>
            <Button disabled={shown > all.length} onClick={() => setShown(s => s + 1)}>{shown === 0 ? 'Show the first step' : shown > all.length ? 'All steps shown' : 'Show the next step'}</Button>
            <Button variant="ghost" onClick={next}>New problem</Button>
            {streak > 1 && <span className="note drill-streak">{streak} correct in a row</span>}
          </Toolbar>
          {shown > 0 && (
            <ol className="drill-steps">
              {all.slice(0, shown).map((s, i) => <li key={i}>{rich(s)}</li>)}
              {shown > all.length && <li><b>Answer:</b> {rich(shownAnswer)}</li>}
            </ol>
          )}
        </div>
        {takeaway && <Takeaway>{rich(takeaway)}</Takeaway>}
      </>
    );
  };
}

// items: [[statement, isTrue, why], ...]
export function truths({ intro, items, takeaway }) {
  return function Truths() {
    const [picks, setPicks] = useState({});
    const done = Object.keys(picks).length, right = items.filter(([, t], i) => picks[i] === t).length;
    return (
      <>
        {intro && <p>{rich(intro)}</p>}
        <div className="truths">
          {items.map(([s, t, why], i) => (
            <div key={i} className="truth">
              <p>{rich(s)}</p>
              <Toolbar>
                {[[true, 'True'], [false, 'False']].map(([val, label]) => <Button key={label} variant={picks[i] === val ? 'solid' : 'push'} aria-pressed={picks[i] === val} onClick={() => setPicks(o => ({ ...o, [i]: val }))}>{label}</Button>)}
              </Toolbar>
              {picks[i] !== undefined && <p className="truth-why"><Status as="b" tone={picks[i] === t ? 'ok' : 'alert'}>{picks[i] === t ? 'Right.' : `It is ${t ? 'true' : 'false'}.`}</Status> {rich(why)}</p>}
            </div>
          ))}
        </div>
        <Stat big>{right} of {done} right, {items.length - done} left</Stat>
        {takeaway && <Takeaway>{rich(takeaway)}</Takeaway>}
      </>
    );
  };
}

// Short orientation above an explorer: the idea you need, and how to work through it.
export function Guide({ idea, how }) {
  return <div className="guide"><p><b>The idea.</b> {rich(idea)}</p><p><b>How to work through it.</b> {rich(how)}</p></div>;
}
