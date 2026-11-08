'use client';
import { createContext, useContext, useId } from 'react';

const Scale = createContext(null);

// Axes and scales for a chart; draw inside it with <Mark>. Pass `caption` to wrap it in a titled <figure>.
export function Plot({ x: [x0, x1], y: [y0, y1], w = 420, h = 260, xl, yl, label, caption, children }) {
  const id = 'c' + useId().replace(/[^a-zA-Z0-9]/g, '');
  const box = { l: 42, r: w - 10, t: 10, b: h - 34 };
  // Rounded: Node and the browser can differ in the last float digit of Math.log/cos, which breaks hydration.
  const sx = v => +(box.l + ((v - x0) / (x1 - x0)) * (box.r - box.l)).toFixed(2);
  const sy = v => +(box.b - ((v - y0) / (y1 - y0)) * (box.b - box.t)).toFixed(2);
  const ix = p => x0 + ((p - box.l) / (box.r - box.l)) * (x1 - x0), iy = p => y0 + ((box.b - p) / (box.b - box.t)) * (y1 - y0);
  const tk = (a, b) => [a, (a + b) / 2, b].map(v => (Math.abs(b - a) < 0.1 ? +v.toPrecision(2) : +v.toFixed(2)));
  const svg = (
    <svg viewBox={`0 0 ${w} ${h}`} className="plot" role="group" aria-label={label}>
      <defs><clipPath id={id}><rect x={box.l} y={box.t} width={box.r - box.l} height={box.b - box.t} /></clipPath></defs>
      {tk(y0, y1).map((v, i) => <line key={i} className="gridline" x1={box.l} x2={box.r} y1={sy(v)} y2={sy(v)} />)}
      {x0 < 0 && x1 > 0 && <line className="zero" x1={sx(0)} x2={sx(0)} y1={box.t} y2={box.b} />}
      {y0 < 0 && y1 > 0 && <line className="zero" x1={box.l} x2={box.r} y1={sy(0)} y2={sy(0)} />}
      {tk(x0, x1).map((v, i) => <text key={i} className="tick" x={sx(v)} y={box.b + 14} textAnchor="middle">{v}</text>)}
      {tk(y0, y1).map((v, i) => <text key={i} className="tick" x={box.l - 6} y={sy(v) + 4} textAnchor="end">{v}</text>)}
      {xl && <text className="axl" x={(box.l + box.r) / 2} y={h - 4} textAnchor="middle">{xl}</text>}
      {yl && <text className="axl" x={12} y={(box.t + box.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(box.t + box.b) / 2})`}>{yl}</text>}
      <g clipPath={`url(#${id})`}><Scale.Provider value={{ sx, sy, box, ix, iy, dom: [x0, x1, y0, y1] }}>{children}</Scale.Provider></g>
    </svg>
  );
  return caption ? <figure><figcaption>{caption}</figcaption>{svg}</figure> : svg;
}

const GEOMETRY = {
  // data: [[x, y], ...]
  line: ({ data }, { sx, sy }) => <path d={'M' + data.map(([a, b]) => `${sx(a).toFixed(1)},${sy(b).toFixed(1)}`).join('L')} />,
  // data: [[x, y, tone?, radius?], ...]
  points: ({ data, r = 4 }, { sx, sy }) => data.map(([a, b, t, s], i) => <circle key={i} cx={sx(a)} cy={sy(b)} r={s ?? r} style={t ? { fill: `var(--c${t})` } : undefined} />),
  rings: ({ data, r = 8 }, { sx, sy }) => data.map(([a, b], i) => <circle key={i} cx={sx(a)} cy={sy(b)} r={r} />),
  // data: [[x, y0, y1], ...]
  stems: ({ data }, { sx, sy }) => data.map(([a, b, c], i) => <line key={i} x1={sx(a)} x2={sx(a)} y1={sy(b)} y2={sy(c)} />),
  // x or y: a full-height or full-width guide
  rule: ({ x, y }, { sx, sy, box }) => (x !== undefined
    ? <line x1={sx(x)} x2={sx(x)} y1={box.t} y2={box.b} />
    : <line x1={box.l} x2={box.r} y1={sy(y)} y2={sy(y)} />),
  // data: [[x, y0, y1], ...] shaded between the two curves
  area: ({ data }, { sx, sy }) => <path d={'M' + data.map(d => `${sx(d[0]).toFixed(1)},${sy(d[2]).toFixed(1)}`).join('L') + 'L' + [...data].reverse().map(d => `${sx(d[0]).toFixed(1)},${sy(d[1]).toFixed(1)}`).join('L') + 'Z'} />,
  // data: [[x0, y0, x1, y1], ...]
  rects: ({ data }, { sx, sy }) => data.map(([a, b, c, d], i) => <rect key={i} x={Math.min(sx(a), sx(c))} y={Math.min(sy(b), sy(d))} width={Math.abs(sx(c) - sx(a))} height={Math.abs(sy(d) - sy(b))} />),
  arrow: ({ data }, { sx, sy }) => data.map(([a, b, c, d], i) => {
    const X = sx(c), Y = sy(d), t = Math.atan2(Y - sy(b), X - sx(a)), at = o => `${(X - 9 * Math.cos(t + o)).toFixed(1)},${(Y - 9 * Math.sin(t + o)).toFixed(1)}`;
    return <g key={i}><line x1={sx(a)} y1={sy(b)} x2={X} y2={Y} /><polygon points={`${X},${Y} ${at(0.45)} ${at(-0.45)}`} /></g>;
  }),
  label: ({ x, y, text, anchor = 'start' }, { sx, sy }) => <text x={sx(x)} y={sy(y)} textAnchor={anchor}>{text}</text>,
  // fn(x, y) in [-1, 1]: positive shades tones[0], negative tones[1]. `bands` quantizes it into filled contours.
  field: ({ fn, bands, nx = bands ? 84 : 54, ny = bands ? 68 : 36, tones = [1, 2] }, { sx, sy, dom: [x0, x1, y0, y1] }) => {
    const dx = (x1 - x0) / nx, dy = (y1 - y0) / ny, W = (Math.abs(sx(x0 + dx) - sx(x0)) + 0.6).toFixed(1), H = (Math.abs(sy(y0 + dy) - sy(y0)) + 0.6).toFixed(1), out = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      let v = fn(x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy);
      if (bands) v = Math.round(v * bands) / bands;
      if (Math.abs(v) > 0.01) out.push(<rect key={i + '-' + j} x={sx(x0 + i * dx).toFixed(1)} y={sy(y0 + (j + 1) * dy).toFixed(1)} width={W} height={H} fill={`color-mix(in srgb, var(--c${v > 0 ? tones[0] : tones[1]}) ${Math.round(Math.min(Math.abs(v), 1) * 62)}%, var(--bg))`} />);
    }
    return out;
  },
  // Level curves of fn(x, y) by marching squares. levels: the values to trace.
  contour: ({ fn, levels = [0], nx = 90, ny = 72 }, { sx, sy, dom: [x0, x1, y0, y1] }) => {
    const X = i => x0 + ((x1 - x0) * i) / nx, Y = j => y0 + ((y1 - y0) * j) / ny, G = Array.from({ length: nx + 1 }, (_, i) => Array.from({ length: ny + 1 }, (_, j) => fn(X(i), Y(j))));
    let d = '';
    for (const L of levels) for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const cross = [], edge = (a, b, ax, ay, bx, by) => { if ((a - L) * (b - L) < 0) { const t = (L - a) / (b - a); cross.push(`${sx(ax + t * (bx - ax)).toFixed(1)},${sy(ay + t * (by - ay)).toFixed(1)}`); } };
      const a = G[i][j], b = G[i + 1][j], c = G[i + 1][j + 1], e = G[i][j + 1];
      edge(a, b, X(i), Y(j), X(i + 1), Y(j)); edge(b, c, X(i + 1), Y(j), X(i + 1), Y(j + 1)); edge(c, e, X(i + 1), Y(j + 1), X(i), Y(j + 1)); edge(e, a, X(i), Y(j + 1), X(i), Y(j));
      for (let k = 0; k + 1 < cross.length; k += 2) d += `M${cross[k]}L${cross[k + 1]}`;
    }
    return <path d={d} />;
  },
  // Draggable points, also movable with the arrow keys. data as for points; onDrag(i, x, y).
  handles: ({ data, onDrag, r = 6, lockX, lockY }, scale) => data.map((d, i) => <Handle key={i} i={i} d={d} r={r} onDrag={onDrag} lockX={lockX} lockY={lockY} scale={scale} />),
};

function Handle({ i, d, r, onDrag, lockX, lockY, scale: { sx, sy, ix, iy, dom: [x0, x1, y0, y1] } }) {
  const emit = (x, y) => onDrag(i, lockX ? d[0] : Math.min(x1, Math.max(x0, x)), lockY ? d[1] : Math.min(y1, Math.max(y0, y)));
  const fromEvent = e => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(e.currentTarget.ownerSVGElement.getScreenCTM().inverse());
    emit(ix(p.x), iy(p.y));
  };
  const key = e => {
    const sX = (x1 - x0) / 50, sY = (y1 - y0) / 50, m = { ArrowLeft: [-sX, 0], ArrowRight: [sX, 0], ArrowUp: [0, sY], ArrowDown: [0, -sY] }[e.key];
    if (m) { e.preventDefault(); emit(d[0] + m[0], d[1] + m[1]); }
  };
  return (
    <g className="handle" role="slider" tabIndex={0} aria-label={`Draggable point ${i + 1}`} aria-valuetext={`${d[0].toFixed(2)}, ${d[1].toFixed(2)}`} onKeyDown={key}
      onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); fromEvent(e); }}
      onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) fromEvent(e); }}
      style={d[2] ? { '--tone': `var(--c${d[2]})` } : undefined}>
      <circle className="hit" cx={sx(d[0])} cy={sy(d[1])} r={r + 9} />
      <circle className="dot" cx={sx(d[0])} cy={sy(d[1])} r={d[3] ?? r} />
    </g>
  );
}

// For a points array in state: onDrag={(i, x, y) => setPts(moved(i, x, y))}. Keeps tone/radius columns.
export const moved = (i, x, y) => p => p.map((q, j) => (j === i ? [x, y, ...q.slice(2)] : q));

// One polymorphic chart element. as: a GEOMETRY key. tone: 1-4 | 'fg' | 'muted'.
export function Mark({ as = 'line', tone, dash, soft, ...geometry }) {
  const scale = useContext(Scale);
  const c = ['mark', `mark-${as}`, tone && `tone-${tone}`, dash && 'dash', soft && 'soft'].filter(Boolean).join(' ');
  return <g className={c}>{GEOMETRY[as](geometry, scale)}</g>;
}

// Small HTML charts that don't need axes. data: [{ label, title, height }], height in %.
export const Bars = ({ data }) => (
  <div className="bars">{data.map(d => <div key={d.label} title={d.title}><i style={{ height: `${d.height.toFixed(2)}%` }} /><span>{d.label}</span></div>)}</div>
);

// cells: one class name per square ('tp' | 'fn' | 'fp' | 'tn')
export const DotGrid = ({ cells, label }) => (
  <div className="dots" role="img" aria-label={label}>{cells.map((k, i) => <i key={i} className={k} />)}</div>
);
