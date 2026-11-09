import { notFound } from 'next/navigation';
import { LECTURES, getLecture } from '../../../../../lib/lectures.mjs';
import { Button, Tag } from '../../../../../components/ui';

export const generateStaticParams = () => LECTURES.flatMap(l => (l.sheets ?? []).map(s => ({ lecture: l.id, id: s.id })));
export async function generateMetadata({ params }) {
  const { lecture, id } = await params;
  const s = getLecture(lecture)?.sheets?.find(x => x.id === id);
  return { title: s ? `Epocher - ${s.label}` : 'Epocher' };
}

export default async function Sheet({ params }) {
  const { lecture, id } = await params;
  const lec = getLecture(lecture), s = lec?.sheets?.find(x => x.id === id);
  if (!s) notFound();
  const href = `/tutorials/${s.file}`;
  return (
    <div>
      <Tag>Tutorial · {s.label}</Tag>
      <h1>{lec.title}</h1>
      <iframe className="sheet" src={href} title={`${lec.title}: ${s.label}`} />
      <p><Button as="a" href={href} download>Download PDF</Button></p>
    </div>
  );
}
