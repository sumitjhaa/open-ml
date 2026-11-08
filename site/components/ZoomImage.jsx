'use client';
import { useState } from 'react';
import Modal from './Modal';

// A figure that opens large in the site's one popup.
export default function ZoomImage({ alt, ...img }) {
  const [open, setOpen] = useState(false), [used, setUsed] = useState(false);
  return (
    <>
      <button type="button" className="zoom" aria-label={`Enlarge figure: ${alt}`} onClick={() => { setUsed(true); setOpen(true); }}><img alt={alt} {...img} /></button>
      {used && <Modal wide open={open} onClose={() => setOpen(false)} title="Figure"><img className="zoomed" alt={alt} src={img.src} /><p className="note">{alt}</p></Modal>}
    </>
  );
}
