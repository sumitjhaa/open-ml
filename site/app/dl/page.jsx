import Link from 'next/link';
import { CourseGrid } from '../../components/CourseGrid';
import { inCourse } from '../../lib/lectures.mjs';

export const metadata = { title: 'Epocher - Deep Learning' };

export default function DeepLearning() {
  return (
    <div className="welcome">
      <p className="eyebrow">Deep Learning</p>
      <h1>Pick a section</h1>
      <p className="lead">Losses, gradients, attention, vision with language, and generative models, with notes, interactive sections, slides and notebooks.</p>
      <CourseGrid items={inCourse('dl')} />
      <p className="note"><Link href="/references">References</Link></p>
    </div>
  );
}
