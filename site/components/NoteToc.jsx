'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { rich } from './labs';

// items: [{ id, depth, text }] from the rendered note. Depth 1 is a part; its sections only unfold while you are in it.
export default function NoteToc({ items }) {
  const parts = useMemo(() => {
    const out = [{ id: items[0]?.id, text: 'Overview', kids: [] }];
    items.slice(1).forEach(h => { if (h.depth === 1) out.push({ ...h, kids: [] }); else out.at(-1).kids.push(h); });
    return out;
  }, [items]);
  const [current, setCurrent] = useState(items[0]?.id), [open, setOpen] = useState(false);
  useEffect(() => {
    let raf = 0;
    const scan = () => { raf = 0; let id = items[0]?.id; for (const h of items) { const el = document.getElementById(h.id); if (el && el.getBoundingClientRect().top < 110) id = h.id; else if (el) break; } setCurrent(id); };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(scan); };
    document.addEventListener('scroll', onScroll, true);
    scan();
    return () => { document.removeEventListener('scroll', onScroll, true); cancelAnimationFrame(raf); };
  }, [items]);
  const nav = useRef(null);
  // Keep the active entry visible: scroll only the rail, never the page.
  useEffect(() => {
    const rail = nav.current, el = rail?.querySelector('a.is-on') ?? rail?.querySelector('a.is-here');
    if (!el || rail.scrollHeight <= rail.clientHeight) return;
    const r = rail.getBoundingClientRect(), a = el.getBoundingClientRect();
    rail.scrollTo({ top: rail.scrollTop + a.top - r.top - (r.height - a.height) / 2, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }, [current]);
  const inPart = p => p.id === current || p.kids.some(k => k.id === current);
  return (
    <nav ref={nav} className={`note-toc ${open ? 'is-open' : ''}`} aria-label="On this page">
      <button type="button" className="btn btn-push toc-toggle" aria-expanded={open} onClick={() => setOpen(o => !o)}>On this page</button>
      <p className="grouphead">On this page</p>
      <ul>
        {parts.map(p => (
          <li key={p.id}>
            <a href={`#${p.id}`} className={`toc-part ${inPart(p) ? 'is-here' : ''}`}>{rich(p.text)}</a>
            {inPart(p) && p.kids.length > 0 && <ul>{p.kids.map(k => <li key={k.id}><a href={`#${k.id}`} className={k.id === current ? 'is-on' : ''}>{rich(k.text)}</a></li>)}</ul>}
          </li>
        ))}
      </ul>
    </nav>
  );
}
