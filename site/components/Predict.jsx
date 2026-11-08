'use client';
import { useState } from 'react';
import { rich } from './labs';

// Commit to an answer before exploring. It never blocks the explorer: a prediction is a prompt to think, not a gate.
export default function Predict({ q, o, a, why }) {
  const [pick, setPick] = useState(null);
  return (
    <aside className="predict" aria-label="Predict first">
      <p className="predict-q"><b>Predict first.</b> {rich(q)}</p>
      <div className="predict-o">
        {o.map((t, i) => (
          <button key={t} type="button" className={pick === null ? '' : i === a ? 'right' : i === pick ? 'wrong' : ''} aria-pressed={pick === i} onClick={() => setPick(i)}>{rich(t)}</button>
        ))}
      </div>
      {pick !== null && <p className="predict-why" role="status"><b>{pick === a ? 'Yes.' : 'Not quite.'}</b> {rich(why)} Check it below.</p>}
    </aside>
  );
}
