// Polymorphic UI primitives: each takes `as` (a tag name or a component such as next/link's Link)
// so buttons, links, cards and nav rows share one look wherever they're used.

const cx = (...c) => c.filter(Boolean).join(' ');

const box = (base, tag = 'div') => function Box({ as: C = tag, className, ...props }) {
  return <C className={cx(base, className)} {...props} />;
};

export const Card = box('card');
export const Tag = box('tag', 'span');
export const Row = box('row');
export const Stack = box('stack');
export const Toolbar = box('toolbar');
export const Note = box('note', 'p');
export const Takeaway = box('takeaway', 'p');
export const Table = box('tbl', 'table');
export const Field = box('field', 'input');

export function Button({ as: C = 'button', variant = 'push', className, ...props }) {
  if (C === 'button' && !props.type) props.type = 'button';
  return <C className={cx('btn', `btn-${variant}`, className)} {...props} />;
}

export function Stat({ as: C = 'div', big, className, ...props }) {
  return <C className={cx('stat', big && 'big', className)} {...props} />;
}

// tone: 'ok' | 'alert'
export function Status({ as: C = 'p', tone, className, ...props }) {
  return <C className={cx(tone, className)} {...props} />;
}

export function NavItem({ as: C = 'a', active, leading, trailing, className, children, ...props }) {
  return (
    <C className={cx('nav-item', active && 'is-on', className)} aria-current={active ? 'page' : undefined} {...props}>
      {leading}
      <span className="label">{children}</span>
      {trailing}
    </C>
  );
}

// Single-choice selector on native radios (arrow keys for free). variant: 'segmented' | 'tiles'.
// `label` doubles as the radio group's name, so keep it unique on the page.
export function Choice({ variant = 'segmented', label, value, onChange, options, className }) {
  return (
    <div role="radiogroup" aria-label={label} className={cx('choice', `choice-${variant}`, className)}>
      {options.map(o => (
        <label key={o.value}>
          <input type="radio" name={label} checked={value === o.value} onChange={() => onChange(o.value)} />
          <span>{o.preview}{o.label}</span>
        </label>
      ))}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, show = v => v }) {
  return (
    <label className="control">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} style={{ '--p': `${(100 * (value - min)) / (max - min)}%` }} onChange={e => onChange(+e.target.value)} />
      <output>{show(value)}</output>
    </label>
  );
}

// items: [['--c1', 'label'], ...]
export const Legend = ({ items }) => (
  <div className="legend">{items.map(([c, t]) => <span key={t}><i style={{ background: `var(${c})` }} />{t}</span>)}</div>
);
