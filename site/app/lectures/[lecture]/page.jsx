import Link from 'next/link';
import { notFound } from 'next/navigation';
import M from '../../../components/M';
import { Button, Card, Toolbar } from '../../../components/ui';
import { LECTURES, LEVELS, getLecture } from '../../../lib/lectures.mjs';

export const generateStaticParams = () => LECTURES.map(l => ({ lecture: l.id }));
export async function generateMetadata({ params }) {
  const l = getLecture((await params).lecture);
  return { title: l ? `${l.title} · ml-sensei` : 'ml-sensei' };
}

export default async function Overview({ params }) {
  const lec = getLecture((await params).lecture);
  if (!lec) notFound();
  const base = `/lectures/${lec.id}`;
  return (
    <div>
      {lec.sheets && <p className="eyebrow">Tutorial</p>}
      <h1>{lec.title}</h1>
      <p className="lead">Guiding question: <em>{lec.question}</em></p>
      {lec.id === 'introduction' && (
        <>
          <p>The whole lecture lives in the gap between two numbers — the error we can measure,</p>
          <M d t="\hat R(f)=\frac1n\sum_{i=1}^n \ell(f(x_i),y_i)" />
          <p>and the error we actually care about,</p>
          <M d t="R(f)=\mathbb{E}_{(X,Y)\sim\mathcal D}[\ell(f(X),Y)]" />
          <p>Each interactive section lets you poke one piece of the maths until it clicks.</p>
        </>
      )}
      {lec.notes.map(n => (
        <Toolbar key={n.id}>
          <Button as={Link} variant="solid" href={`${base}/notes/${n.id}`}>{n.label === 'Notes' ? 'Read the notes' : n.label}</Button>
          <Button as="a" href={`/notes/${n.file}.pdf`} download>PDF</Button>
          <Button as="a" href={`/notes/${n.file}.md`} download>Markdown</Button>
        </Toolbar>
      ))}
      {lec.sheets && <Toolbar>{lec.sheets.map(s => <Button key={s.id} as={Link} href={`${base}/sheet/${s.id}`}>{s.label}</Button>)}</Toolbar>}
      {LEVELS.map(([level, label]) => (
        <section key={level}>
          <h2 className="level-head">{label}</h2>
          <div className="grid">
            {lec.explorers.filter(e => e.level === level).map(e => (
              <Card as={Link} key={e.id} href={`${base}/explore/${e.id}`}>
                <h3>{e.title}</h3>
                <p>{e.blurb}</p>
              </Card>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
