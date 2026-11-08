'use client';
import { Fragment, useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LECTURES, LEVELS, getLecture } from '../lib/lectures.mjs';
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

export default function Shell({ children }) {
  const [, root, lid, kind, eid] = usePathname().split('/');
  const [cols, update] = useColumns();
  const lec = root === 'lectures' ? getLecture(lid) : null;
  const shown = lec ?? LECTURES[0]; // outside a lecture (home), the topics column still has something to show
  const active = lec && (kind === 'notes' || kind === 'explore' || kind === 'sheet') ? `${kind}/${eid}` : null;
  const row = l => <NavItem key={l.id} as={Link} href={`/lectures/${l.id}`} active={lec?.id === l.id}>{l.title}</NavItem>;
  return (
    <div className="app">
      {/* A mouse click should not leave focus behind, or a later key press draws a focus ring on the old tab. Keyboard activation has detail 0. */}
      <aside className="sidebar" onClick={e => { if (e.detail > 0) e.target.closest('a, button')?.blur(); }}>
        <Column id="course" title="machine-learning" head={<Link href="/" className="brand">ml-sensei</Link>} col={cols.course} update={update} className="course" footer={<ThemePicker />}>
          <NavItem as={Link} href="/" active>machine-learning</NavItem>
        </Column>
        <Column id="lectures" title="Lectures" col={cols.lectures} update={update} className="lectures">
          <nav className="group" aria-label="Lectures">{LECTURES.filter(l => !l.sheets).map(row)}</nav>
          <p className="grouphead">Tutorials</p>
          <nav className="group" aria-label="Tutorials">{LECTURES.filter(l => l.sheets).map(row)}</nav>
        </Column>
        <Column id="topics" title="Topics" col={cols.topics} update={update} className="topics">
          <nav className="group" aria-label={`Topics in ${shown.title}`}>
            {shown.notes.length > 1 && <p className="grouphead">Notes</p>}
            {shown.notes.map(n => <NavItem key={n.id} as={Link} href={`/lectures/${shown.id}/notes/${n.id}`} active={active === `notes/${n.id}`}>{n.label}</NavItem>)}
            {shown.sheets && <p className="grouphead">Worksheets</p>}
            {shown.sheets?.map(s => <NavItem key={s.id} as={Link} href={`/lectures/${shown.id}/sheet/${s.id}`} active={active === `sheet/${s.id}`}>{s.label}</NavItem>)}
            {LEVELS.map(([level, label]) => (
              <Fragment key={level}>
                <p className="grouphead">{label}</p>
                {shown.explorers.filter(e => e.level === level).map(e => (
                  <NavItem key={e.id} as={Link} href={`/lectures/${shown.id}/explore/${e.id}`} active={active === `explore/${e.id}`}>{e.title}</NavItem>
                ))}
              </Fragment>
            ))}
          </nav>
        </Column>
      </aside>
      <main className="pane">
        <div className="content note-wrap">{children}</div>
      </main>
    </div>
  );
}
