'use client';
import { useEffect, useRef } from 'react';

// The one popup: a native <dialog>, so focus trapping and Esc come from the browser.
export default function Modal({ open, onClose, title, wide, style, children }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (open && !d.open) { d.showModal(); d.querySelector(':checked')?.focus(); }
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className={wide ? 'modal modal-wide' : 'modal'} aria-label={title} style={style} onClose={onClose} onClick={e => e.target === e.currentTarget && onClose()}>
      <header className="modal-head">
        <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>×</button>
        <h2>{title}</h2>
      </header>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}
