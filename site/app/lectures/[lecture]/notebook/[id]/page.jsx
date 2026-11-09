import { readFileSync } from 'node:fs';
import path from 'node:path';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import ZoomImage from '../../../../../components/ZoomImage';
import { highlight } from '../../../../../components/Code';
import CodeBlock from '../../../../../components/CodeBlock';
import { Button, Toolbar } from '../../../../../components/ui';
import { LECTURES, getLecture } from '../../../../../lib/lectures.mjs';
import { notebooksOf } from '../../../../../lib/notebooks.mjs';

const textOf = c => (typeof c === 'string' ? c : Array.isArray(c) ? c.map(textOf).join('') : '');
// Only video links leave this site, and they open in a new tab. Other links in the notebook text become plain words.
const mdLink = ({ href, children }) => (/^https:\/\/(www\.)?(youtube\.com|youtu\.be)\//.test(href ?? '') ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <>{children}</>);
const mdComponents = { a: mdLink, pre: ({ children }) => <CodeBlock>{children}</CodeBlock>, code: ({ className, children }) => <code className={className}>{/language-(python|py)/.test(className ?? '') ? highlight(textOf(children)) : children}</code> };
const find = (lecture, id) => (getLecture(lecture) ? notebooksOf(lecture).find(n => n.id === id) : undefined);

export const generateStaticParams = () => LECTURES.flatMap(l => notebooksOf(l.id).map(n => ({ lecture: l.id, id: n.id })));
export async function generateMetadata({ params }) {
  const { lecture, id } = await params, n = find(lecture, id);
  return { title: n ? `Epocher - ${n.title}` : 'Epocher' };
}

export default async function Notebook({ params }) {
  const { lecture, id } = await params, n = find(lecture, id);
  if (!n) notFound();
  const { cells } = JSON.parse(readFileSync(path.join(process.cwd(), `public/notebooks/${id}.json`), 'utf8'));
  return (
    <div>
      <p className="eyebrow">Notebook</p>
      <h1>{n.title}</h1>
      <p className="note">Original source in <Link href="/references">References</Link>.</p>
      <Toolbar className="note-actions">
        <Button as="a" variant="solid" href={`/notebooks/files/${id}.ipynb`} download>Download notebook</Button>
      </Toolbar>
      <article className="prose">
        {cells.map((c, i) => (c.md !== undefined ? (
          // Notebook text comes from outside this site: raw HTML and images in it are dropped.
          <Markdown key={i} skipHtml disallowedElements={['img']} unwrapDisallowed remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]} components={mdComponents}>{c.md}</Markdown>
        ) : (
          <div key={i} className="nb-cell">
            <CodeBlock><code>{highlight(c.code)}</code></CodeBlock>
            {c.out.map((o, j) => (o.img ? (
              <figure key={j} className="fig fig-mid"><ZoomImage src={o.img} alt={`Output of cell ${i + 1}`} width={o.w} height={o.h} loading="lazy" /></figure>
            ) : <pre key={j} className="nb-out">{o.text}</pre>))}
          </div>
        )))}
      </article>
    </div>
  );
}
