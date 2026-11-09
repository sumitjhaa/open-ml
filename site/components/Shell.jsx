'use client';
import { Fragment, useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LEVELS, getLecture, inCourse } from '../lib/lectures.mjs';
import { notebooksOf } from '../lib/notebooks.mjs';
import ThemePicker from './ThemePicker';
import { NavItem } from './ui';

const DEFAULTS = { course: { w: 156, folded: false }, lectures: { w: 212, folded: false }, topics: { w: 216, folded: false } };
const MIN = 120, MAX = 480;
const limit = w => Math.min(MAX, Math.max(MIN, w));

// Layout is a per-viewer convenience kept in localStorage; anything malformed falls back to the default.
function useColumns() {
  const [cols, setCols] = useState(DEFAULTS);
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('columns')) ?? {};
      setCols(Object.fromEntries(Object.entries(DEFAULTS).map(([id, d]) => [id, {
        w: Number.isFinite(saved[id]?.w) ? limit(saved[id].w) : d.w,
        folded: saved[id]?.folded === true,
      }])));
    } catch {}
  }, []);
  const update = (id, patch) => setCols(c => {
    const next = { ...c, [id]: { ...c[id], ...patch } };
    try { localStorage.setItem('columns', JSON.stringify(next)); } catch {}
    return next;
  });
  return [cols, update];
}

function Column({ id, title, head, col, update, footer, className = '', children }) {
  const drag = e => {
    const el = e.currentTarget, x0 = e.clientX, w0 = col.w;
    el.setPointerCapture(e.pointerId);
    const move = ev => update(id, { w: limit(w0 + ev.clientX - x0) });
    const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); el.blur(); };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };
  const key = e => {
    const d = { ArrowLeft: -16, ArrowRight: 16 }[e.key];
    if (d) { e.preventDefault(); update(id, { w: limit(col.w + d) }); }
  };
  return (
    <section className={`side-col ${col.folded ? 'is-folded' : ''} ${className}`} style={{ '--w': `${col.w}px` }} aria-label={title}>
      <header className="col-head">
        <span className="col-title">{head ?? title}</span>
        <button type="button" className="col-fold" aria-expanded={!col.folded} aria-label={`${col.folded ? 'Unfold' : 'Fold'} ${title}`} title={col.folded ? 'Unfold' : 'Fold'} onClick={() => update(id, { folded: !col.folded })}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="10" rx="2" /><path d="M6.5 3v10" /></svg>
        </button>
      </header>
      <div className="col-body">{children}</div>
      {footer}
      {!col.folded && (
        <div className="col-resize" role="separator" aria-orientation="vertical" aria-label={`Resize ${title}`} aria-valuenow={col.w} aria-valuemin={MIN} aria-valuemax={MAX} tabIndex={0}
          title="Drag to resize, double-click to reset" onPointerDown={drag} onKeyDown={key} onDoubleClick={() => update(id, { w: DEFAULTS[id].w })} />
      )}
    </section>
  );
}

// Same artwork as public/icons/ml-infinite.svg (and the favicon, app/icon.svg), drawn with the theme colour and a thicker line for small sizes.
const LogoMark = () => <svg className="brand-mark" viewBox="0 0 48 48" aria-hidden="true"><path d="M17.3083,23.5331,15.374,25.8175a2.0435,2.0435,0,0,1-2.6666.3511,2.0705,2.0705,0,0,1-.7638-2.5791,2.0039,2.0039,0,0,1,2.4276-1.1577L30.495,28.134A9.6947,9.6947,0,1,0,26.3779,12.79l-6.4834,7.68M26.8715,26.9216,22.4146,34.11A9.6923,9.6923,0,0,1,5.3862,24.9308,9.8217,9.8217,0,0,1,17.67,19.9262L32.3816,25.024a2.8755,2.8755,0,0,0,3.57-1.7291,2.9183,2.9183,0,0,0-1.4542-3.6916,2.958,2.958,0,0,0-3.79,1.1709L28.851,23.7533M37.5454,21.6788l4.5962-.8617M23.7747,23.2319l-.5257,1.186M28.6034,24.8885l-.5517,1.1907M33.1215,26.2533l-.1128,1.3658M36.6794,24.8885l1.4938,1.4646M35.164,18.1773l2.532-6.6886M30.6177,18.9585,26.543,14.7643M23.6241,29.4065l-2.9348-3.3386M20.64,33.9528l-4.3919-7.0566M15.8493,37.21l-1.458-9.4126M9.45,35.873l3.141-8.3217M5.9557,30.7525l5.2118-4.2573M6.8377,24.8226l3.7754-.0562M10.8606,21.114l.61.9234M15.0492,20.4646l-.3522.97M19.37,21.66l-.5611,1.18M32.6038,18.0737l-.8583-6.8966M36.6324,19.721l4.6639-4.344M27.0786,21.8388l-3.46-3.5956" /></svg>;
// Self-attention: one token attending to the others in a sequence.
const AttentionIcon = () => (
  <svg className="nav-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M9.5 16Q6.5 9 3.5 16M9.5 16Q12 9 14.5 16M9.5 16Q15 1 20.5 16" />
    <g fill="currentColor" stroke="none"><circle cx="3.5" cy="18.5" r="2" /><circle cx="9.5" cy="18.5" r="2" /><circle cx="14.5" cy="18.5" r="2" /><circle cx="20.5" cy="18.5" r="2" /></g>
  </svg>
);
// An attention matrix: a grid of weights, where stronger links are larger.
const AttentionMatrixIcon = () => (
  <svg className="nav-icon" viewBox="0 0 24 24" aria-hidden="true">
    <g fill="currentColor" stroke="none">
      {[[5, 5, 1.1], [12, 5, 2.7], [19, 5, 1.1], [5, 12, 2.7], [12, 12, 1.1], [19, 12, 1.8], [5, 19, 1.8], [12, 19, 1.1], [19, 19, 2.7]].map(([x, y, r]) => <circle key={`${x}-${y}`} cx={x} cy={y} r={r} />)}
    </g>
  </svg>
);
const LEVEL_TONE = { basics: 3, maths: 1, advanced: 4 };
const GroupHead = ({ tone, children }) => <p className={`grouphead tone-${tone}`}><span className="chip">{children}</span></p>;
const ColumnHead = ({ tone, children }) => <span className={`chip tone-${tone}`}>{children}</span>;

export default function Shell({ children }) {
  const [, root, lid, kind, eid] = usePathname().split('/');
  const [cols, update] = useColumns();
  const lec = root === 'lectures' ? getLecture(lid) : null;
  const course = lec ? lec.course ?? 'ml' : root === 'dl' ? 'dl' : 'ml', sections = inCourse(course);
  const shown = lec ?? sections[0]; // outside a lecture (home), the topics column still has something to show
  const active = lec && ['notes', 'explore', 'sheet', 'notebook', 'resources'].includes(kind) ? `${kind}/${eid}` : null;
  const notebooks = notebooksOf(shown.id);
  const row = l => <NavItem key={l.id} as={Link} href={`/lectures/${l.id}`} active={lec?.id === l.id}>{l.title}</NavItem>;
  return (
    <div className="app">
      {/* A mouse click should not leave focus behind, or a later key press draws a focus ring on the old tab. Keyboard activation has detail 0. */}
      <aside className="sidebar" onClick={e => { if (e.detail > 0) e.target.closest('a, button')?.blur(); }}>
        <Column id="course" title="ML" head={<Link href="/" className="brand"><LogoMark />Epocher</Link>} col={cols.course} update={update} className="course" footer={<ThemePicker />}>
          <NavItem as={Link} href="/" active={course === 'ml'} leading={<AttentionIcon />}>ML</NavItem>
          <NavItem as={Link} href="/dl" active={course === 'dl'} leading={<AttentionMatrixIcon />}>DL</NavItem>
        </Column>
        <Column id="lectures" title="Lectures" head={<ColumnHead tone="1">Lectures</ColumnHead>} col={cols.lectures} update={update} className="lectures">
          <nav className="group" aria-label="Lectures">{sections.filter(l => !l.sheets).map(row)}</nav>
          {sections.some(l => l.sheets) && <GroupHead tone="3">Tutorials</GroupHead>}
          {sections.some(l => l.sheets) && <nav className="group" aria-label="Tutorials">{sections.filter(l => l.sheets).map(row)}</nav>}
        </Column>
        <Column id="topics" title="Topics" head={<ColumnHead tone="1">Topics</ColumnHead>} col={cols.topics} update={update} className="topics">
          <nav className="group" aria-label={`Topics in ${shown.title}`}>
            {shown.notes.length > 1 && <GroupHead tone="fg">Notes</GroupHead>}
            {shown.notes.map(n => <NavItem key={n.id} as={Link} href={`/lectures/${shown.id}/notes/${n.id}`} active={active === `notes/${n.id}`}>{n.label}</NavItem>)}
            {shown.sheets && <GroupHead tone="fg">Worksheets</GroupHead>}
            {shown.sheets?.map(s => <NavItem key={s.id} as={Link} href={`/lectures/${shown.id}/sheet/${s.id}`} active={active === `sheet/${s.id}`}>{s.label}</NavItem>)}
            {LEVELS.map(([level, label]) => (
              <Fragment key={level}>
                <GroupHead tone={LEVEL_TONE[level]}>{label}</GroupHead>
                {shown.explorers.filter(e => e.level === level).map(e => (
                  <NavItem key={e.id} as={Link} href={`/lectures/${shown.id}/explore/${e.id}`} active={active === `explore/${e.id}`}>{e.title}</NavItem>
                ))}
              </Fragment>
            ))}
            <GroupHead tone="fg">Code and resources</GroupHead>
            <NavItem as={Link} href={`/lectures/${shown.id}/resources`} active={kind === 'resources' && lec?.id === shown.id}>Slides, cheat sheets and videos</NavItem>
            {notebooks.map(n => <NavItem key={n.id} as={Link} href={`/lectures/${shown.id}/notebook/${n.id}`} active={active === `notebook/${n.id}`}>{n.title}</NavItem>)}
          </nav>
        </Column>
      </aside>
      <main className="pane">
        <div className="content note-wrap">{children}</div>
      </main>
    </div>
  );
}
