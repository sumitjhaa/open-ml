import Link from 'next/link';
import { notFound } from 'next/navigation';
import { LECTURES, LEVELS, getLecture } from '../../../../../lib/lectures.mjs';
import { Button, Tag } from '../../../../../components/ui';
import Predict from '../../../../../components/Predict';
import ExplorerView from '../../../../../components/ExplorerView';
import { Guide } from '../../../../../components/labs';
import { guideFor } from '../../../../../lib/guides.mjs';

export const generateStaticParams = () => LECTURES.flatMap(l => l.explorers.map(e => ({ lecture: l.id, id: e.id })));
export async function generateMetadata({ params }) {
  const { lecture, id } = await params;
  const e = getLecture(lecture)?.explorers.find(x => x.id === id);
  return { title: e ? `Epocher - ${e.title}` : 'Epocher' };
}

export default async function Explorer({ params }) {
  const { lecture, id } = await params;
  const lec = getLecture(lecture);
  const i = lec?.explorers.findIndex(e => e.id === id) ?? -1;
  if (i < 0) notFound();
  const e = lec.explorers[i], prev = lec.explorers[i - 1], next = lec.explorers[i + 1], base = `/lectures/${lec.id}/explore`;
  return (
    <div className="explorer">
      <Tag>{LEVELS.find(([k]) => k === e.level)[1]}</Tag>
      <h1>{e.title}</h1>
      <Guide {...guideFor(e.component)} />
      {e.predict && <Predict {...e.predict} />}
      <ExplorerView module={e.module} component={e.component} />
      <div className="pager">
        {prev ? <Button as={Link} variant="ghost" href={`${base}/${prev.id}`}>‹ {prev.title}</Button> : <span />}
        {next && <Button as={Link} variant="ghost" href={`${base}/${next.id}`}>{next.title} ›</Button>}
      </div>
    </div>
  );
}
