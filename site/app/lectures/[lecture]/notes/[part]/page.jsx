import { LECTURES } from '../../../../../lib/lectures.mjs';
import Notes, { generateMetadata as meta } from './render';

// Built once at deploy time, so a note is a static file and no request has to parse Markdown or maths.
export const generateStaticParams = () => LECTURES.flatMap(l => l.notes.map(n => ({ lecture: l.id, part: n.id })));
export const generateMetadata = meta;
export default function Page({ params }) {
  return <Notes params={params} />;
}
