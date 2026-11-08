import Link from 'next/link';
import { LECTURES } from '../lib/lectures.mjs';
import { Card, Tag } from '../components/ui';

const Grid = ({ items, tag }) => (
  <div className="grid">
    {items.map(l => (
      <Card as={Link} key={l.id} href={`/lectures/${l.id}`}>
        {tag && <Tag>{tag}</Tag>}
        <h3>{l.title}</h3>
        <p>{l.explorers.length} interactive sections</p>
      </Card>
    ))}
  </div>
);

export default function Home() {
  return (
    <div className="welcome">
      <p className="eyebrow">ES 335 / ES 678 · Machine Learning · Sem I 2026</p>
      <h1>Pick a section</h1>
      <p className="lead">Each section has full notes and interactive sections that climb from the basics, through the mathematics, to advanced ideas.</p>
      <Grid items={LECTURES.filter(l => !l.sheets)} />
      <h2 className="level-head">Tutorials</h2>
      <Grid items={LECTURES.filter(l => l.sheets)} tag="Tutorial" />
    </div>
  );
}
