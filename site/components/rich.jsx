import M from './M';

// "plain **bold** and $maths$" to elements.
export function rich(s) {
  if (typeof s !== 'string') return s;
  return s.split(/(\$[^$]+\$|\*\*[^*]+\*\*)/).map((p, i) => (p[0] === '$' && p.length > 1 ? <M key={i} t={p.slice(1, -1)} /> : p.startsWith('**') ? <b key={i}>{rich(p.slice(2, -2))}</b> : p));
}
