import { LECTURES } from '../../../../../../lib/lectures.mjs';
import Notes, { generateMetadata as meta } from '../render';

// What scripts/make-pdf.mjs prints: the same note, with practice answers open and the inline contents kept.
export const generateStaticParams = () => LECTURES.flatMap(l => l.notes.map(n => ({ lecture: l.id, part: n.id })));
export const generateMetadata = meta;
export default function Page({ params }) {
  return <Notes params={params} print />;
}
