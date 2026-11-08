'use client';
import { useEffect, useState } from 'react';
import { DARK_THEMES, LIGHT_THEMES } from '../lib/themes.mjs';
import Modal from './Modal';
import { Choice, NavItem } from './ui';

// A tiny editor window drawn in the theme's own colours.
const Preview = ({ t }) => (
  <span className="thumb" style={{ background: t.bg }}>
    <span className="thumb-side" style={{ background: t.side }} />
    <span className="thumb-code">
      <i style={{ background: t.c1, width: '70%' }} />
      <i style={{ background: t.c2, width: '45%' }} />
      <i style={{ background: t.c3, width: '85%' }} />
      <i style={{ background: t.c4, width: '30%' }} />
    </span>
  </span>
);
const options = list => list.map(t => ({ value: t.id, label: t.name, preview: <Preview t={t.tokens} /> }));
const SYSTEM = [{
  value: 'system', label: 'System',
  preview: <span className="thumb"><span className="thumb-split" style={{ background: `linear-gradient(135deg, ${LIGHT_THEMES[0].tokens.bg} 50%, ${DARK_THEMES[0].tokens.bg} 50%)` }} /></span>,
}];
const LIGHT = options(LIGHT_THEMES), DARK = options(DARK_THEMES);

// Choice is a per-viewer convenience kept in localStorage; the inline script in layout.jsx applies it before paint.
export default function ThemePicker() {
  const [theme, setTheme] = useState('system'), [open, setOpen] = useState(false);
  useEffect(() => { try { setTheme(localStorage.getItem('theme') || 'system'); } catch {} }, []);
  useEffect(() => {
    const onKey = e => {
      if ((e.key !== 't' && e.key !== 'T') || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      // Only block T where it would type text; sliders and radios don't take any.
      if (e.target.closest?.('input:not([type=range],[type=radio],[type=checkbox]),textarea,select,[contenteditable]')) return;
      setOpen(o => !o);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const pick = id => {
    setTheme(id);
    const apply = () => {
      if (id === 'system') delete document.documentElement.dataset.theme;
      else document.documentElement.dataset.theme = id;
    };
    document.startViewTransition ? document.startViewTransition(apply) : apply();
    try { id === 'system' ? localStorage.removeItem('theme') : localStorage.setItem('theme', id); } catch {}
  };
  return (
    <div className="themes">
      <NavItem as="button" type="button" aria-haspopup="dialog" trailing={<kbd>T</kbd>} onClick={() => setOpen(true)}>Theme</NavItem>
      <Modal open={open} onClose={() => setOpen(false)} title="Theme" style={{ width: 'min(1040px, calc(100vw - 32px))' }}>
        <div className="theme-scroll">
          <Choice variant="tiles" label="Follow the system" value={theme} onChange={pick} options={SYSTEM} />
          <p className="theme-group">Light</p>
          <Choice variant="tiles" label="Light themes" value={theme} onChange={pick} options={LIGHT} />
          <p className="theme-group">Dark</p>
          <Choice variant="tiles" label="Dark themes" value={theme} onChange={pick} options={DARK} />
        </div>
      </Modal>
    </div>
  );
}
