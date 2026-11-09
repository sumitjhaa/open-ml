import Notes, { generateMetadata as meta } from '../render';

// What scripts/make-pdf.mjs prints: the same note, with practice answers open and the inline contents kept.
export const generateMetadata = meta;
export default function Page({ params }) {
  return <Notes params={params} print />;
}
