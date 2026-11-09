import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Button, Toolbar } from '../../../../components/ui';
import { LECTURES, getLecture } from '../../../../lib/lectures.mjs';
import { RESOURCES } from '../../../../lib/resources.mjs';
import { MORE_VIDEOS } from '../../../../lib/videos.mjs';

const ext = { target: '_blank', rel: 'noopener noreferrer' };
export const generateStaticParams = () => LECTURES.filter(l => MORE_VIDEOS[l.id]?.length).map(l => ({ lecture: l.id }));
export async function generateMetadata({ params }) {
  const l = getLecture((await params).lecture);
  return { title: l ? `Epocher - ${l.title} resources` : 'Epocher' };
}

export default async function Resources({ params }) {
  const lec = getLecture((await params).lecture), rows = lec && (RESOURCES[lec.id] ?? []), more = lec && MORE_VIDEOS[lec.id];
  if (!more?.length) notFound();
  return (
    <div>
      <p className="eyebrow">Resources</p>
      <h1>{lec.title}</h1>
      <p className="note">Slides, cheat sheets, handouts and one-minute videos. Original sources in <Link href="/references">References</Link>.</p>
      {rows.map(r => (
        <section key={r.title}>
          <h2 className="level-head">{r.title}</h2>
          {r.sub && <p>{r.sub}</p>}
          {(r.slides.length > 0 || r.cheats.length > 0 || r.labs.length > 0) && (
            <Toolbar>
              {r.slides.map(s => <Button as="a" key={s.href} href={s.href} {...ext}>{s.label}</Button>)}
              {r.cheats.map(s => <Button as="a" key={s.href} href={s.href} {...ext}>{s.label}</Button>)}
              {r.labs.map(s => <Button as={Link} key={s.href} href={s.href}>{s.label}</Button>)}
            </Toolbar>
          )}
          {r.videos.length > 0 && <ul>{r.videos.map(v => <li key={v.href}><a href={v.href} {...ext}>{v.label}</a></li>)}</ul>}
        </section>
      ))}
      <section>
        <h2 className="level-head">Full lectures from other courses</h2>
        <p>Stanford, MIT, Berkeley and other open courses on this topic. These open on YouTube.</p>
        <ul>{more.map(v => <li key={v.href}><a href={v.href} {...ext}>{v.label}</a></li>)}</ul>
      </section>
    </div>
  );
}
