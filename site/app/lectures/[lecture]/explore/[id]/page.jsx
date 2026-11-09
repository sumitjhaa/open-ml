import Link from 'next/link';
import { notFound } from 'next/navigation';
import { LECTURES, LEVELS, getLecture } from '../../../../../lib/lectures.mjs';
import { Button, Tag } from '../../../../../components/ui';
import Predict from '../../../../../components/Predict';
import { Guide } from '../../../../../components/labs';
import { guideFor } from '../../../../../lib/guides.mjs';
import * as L01 from '../../../../../explorers/l01';
import * as Maths from '../../../../../explorers/maths';
import * as Regression from '../../../../../explorers/regression';
import * as Evaluation from '../../../../../explorers/evaluation';
import * as Svm from '../../../../../explorers/svm';
import * as Trees from '../../../../../explorers/trees';
import * as Practice from '../../../../../explorers/practice';
import * as Foundations from '../../../../../explorers/foundations';
import * as Modelling from '../../../../../explorers/modelling';
import * as Methods from '../../../../../explorers/methods';
import * as Tutorials from '../../../../../explorers/tutorials';
import * as Unsupervised from '../../../../../explorers/unsupervised';
import * as Clusters from '../../../../../explorers/clusters';
import * as Reduction from '../../../../../explorers/reduction';
import * as Anomaly from '../../../../../explorers/anomaly';
import * as Classical from '../../../../../explorers/classical';
import * as Neuralnets from '../../../../../explorers/neuralnets';
import * as Vision from '../../../../../explorers/vision';
import * as Sequences from '../../../../../explorers/sequences';
import * as Additions from '../../../../../explorers/additions';
import * as Modern from '../../../../../explorers/modern';
import * as Beyond from '../../../../../explorers/beyond';
import * as Deeplearning from '../../../../../explorers/deeplearning';
import * as Moreml from '../../../../../explorers/moreml';
import * as Frontier from '../../../../../explorers/frontier';
import * as Extensions from '../../../../../explorers/extensions';
import * as Dlviz from '../../../../../explorers/dlviz';
import * as Coverage from '../../../../../explorers/coverage';
import * as Gaps from '../../../../../explorers/gaps';
import * as Completions from '../../../../../explorers/completions';

// Each catalogue row in lib/lectures.mjs names one of these modules.
const MODULES = { Coverage, Dlviz, Extensions, Frontier, L01, Maths, Regression, Evaluation, Svm, Trees, Practice, Foundations, Modelling, Methods, Tutorials, Unsupervised, Clusters, Reduction, Anomaly, Classical, Neuralnets, Vision, Sequences, Additions, Modern, Beyond, Deeplearning, Moreml, Gaps, Completions };

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
  const e = lec.explorers[i], C = MODULES[e.module][e.component], prev = lec.explorers[i - 1], next = lec.explorers[i + 1], base = `/lectures/${lec.id}/explore`;
  return (
    <div className="explorer">
      <Tag>{LEVELS.find(([k]) => k === e.level)[1]}</Tag>
      <h1>{e.title}</h1>
      <Guide {...guideFor(e.component)} />
      {e.predict && <Predict {...e.predict} />}
      <C />
      <div className="pager">
        {prev ? <Button as={Link} variant="ghost" href={`${base}/${prev.id}`}>‹ {prev.title}</Button> : <span />}
        {next && <Button as={Link} variant="ghost" href={`${base}/${next.id}`}>{next.title} ›</Button>}
      </div>
    </div>
  );
}
