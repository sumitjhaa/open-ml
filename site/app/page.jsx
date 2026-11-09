import Link from 'next/link';
import { inCourse } from '../lib/lectures.mjs';
import { CourseGrid as Grid } from '../components/CourseGrid';

const LECTURES = inCourse('ml');

export default function Home() {
  return (
    <div className="welcome">
      <p className="eyebrow">ES 335 / ES 678 · Machine Learning · Sem I 2026</p>
      <h1>Pick a section</h1>
      <p className="lead">Each section has full notes and interactive sections that climb from the basics, through the mathematics, to advanced ideas.</p>
      <Grid items={LECTURES.filter(l => !l.sheets)} />
      <h2 className="level-head">Tutorials</h2>
      <Grid items={LECTURES.filter(l => l.sheets)} tag="Tutorial" />
      <p className="note"><Link href="/references">References</Link></p>
    </div>
  );
}
