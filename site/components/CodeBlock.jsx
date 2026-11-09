'use client';
import { useRef, useState } from 'react';

// A code block with a Copy button. The code itself still renders on the server.
export default function CodeBlock({ children }) {
  const pre = useRef(null), [state, setState] = useState('Copy');
  const copy = async () => {
    try { await navigator.clipboard.writeText(pre.current.textContent); setState('Copied'); }
    catch { setState('Copy failed'); }
    setTimeout(() => setState('Copy'), 1500);
  };
  return (
    <div className="code-block">
      <pre ref={pre}>{children}</pre>
      <button type="button" className="btn btn-push code-copy" onClick={copy} aria-live="polite">{state}</button>
    </div>
  );
}
