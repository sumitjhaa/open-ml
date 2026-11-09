import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { notFound } from 'next/navigation';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import rehypeKatex from 'rehype-katex';
import Link from 'next/link';
import { findNote, getLecture } from '../../../../../lib/lectures.mjs';
import NoteToc from '../../../../../components/NoteToc';
import { rich } from '../../../../../components/rich';
import ZoomImage from '../../../../../components/ZoomImage';
import { highlight } from '../../../../../components/Code';
import CodeBlock from '../../../../../components/CodeBlock';
import { Button, Toolbar } from '../../../../../components/ui';

export async function generateMetadata({ params }) {
  const { lecture, part } = await params, l = getLecture(lecture), n = l?.notes.find(x => x.id === part);
  return { title: n ? `Epocher - ${n.label === 'Notes' ? l.title : n.label}` : 'Epocher' };
}

// Notes image paths are relative to notes/; sync.mjs copies them to /images.
function fixUrl(u) {
  if (/^(https?:|#|\/|mailto:)/.test(u)) return u;
  const [file, hash] = u.split('#'), hit = file.endsWith('.md') && findNote(file.slice(0, -3)); // links between lecture notes
  return hit ? `/lectures/${hit.lecture}/notes/${hit.note}${hash ? '#' + hash : ''}` : '/' + u;
}

// Intrinsic size from the PNG/GIF header, so layout doesn't jump and we can pick a size class by shape.
function imgSize(src) {
  try {
    const b = readFileSync(path.join(process.cwd(), 'public', src));
    if (b.toString('ascii', 1, 4) === 'PNG') return [b.readUInt32BE(16), b.readUInt32BE(20)];
    if (b.toString('ascii', 0, 3) === 'GIF') return [b.readUInt16LE(6), b.readUInt16LE(8)];
  } catch {}
  return null;
}

// slides/ = lecture slide captures, web/ = small reference diagrams, figures/ = charts (wide strips vs squarer plots).
function figClass(src) {
  if (src.endsWith('.svg')) return 'fig fig-wide';
  if (src.includes('/slides/')) return 'fig fig-slide';
  if (src.includes('/web/')) return 'fig fig-small';
  const s = imgSize(src);
  return s && s[0] / s[1] > 2.1 ? 'fig fig-wide' : 'fig fig-mid';
}

const label = file => { const hit = findNote(file); if (!hit) return null; const l = getLecture(hit.lecture), n = l.notes.find(x => x.id === hit.note); return n.label === 'Notes' ? l.title : n.label; };
const textOf = c => (typeof c === 'string' ? c : Array.isArray(c) ? c.map(textOf).join('') : c?.props ? textOf(c.props.children) : '');

const plain = n => (n.type === 'text' ? n.value : (n.children ?? []).map(plain).join(''));
// Bold openers that mark what a paragraph is doing. Cards for the ones worth stopping at, coloured lead-ins for the rest.
function calloutOf(t) {
  if (/^(careful|warning|common (mistake|confusion)|watch out|trap|pitfall)/i.test(t)) return 'callout callout-careful';
  if (/^(insight|intuition|key|takeaway|rule|why ml cares|ml lesson|the idea)/i.test(t)) return 'callout callout-idea';
  if (/^(worked( example)?|example|numbers)/i.test(t)) return 'callout callout-example';
  if (/^(summary|recipe|how to use this file|what this (file|lecture))/i.test(t)) return 'callout callout-summary';
  if (/^(question|exercise|your turn|try it)/i.test(t)) return 'run-in run-in-question';
  if (/^(check|verify|answer)/i.test(t)) return 'run-in run-in-check';
  if (/^step \d/i.test(t)) return 'run-in';
  return null;
}

const components = {
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  },
  code({ className, children }) {
    return <code className={className}>{/language-(python|py)/.test(className ?? '') ? highlight(textOf(children)) : children}</code>;
  },
  // A paragraph holding an image (+ its *caption* line) becomes a sized <figure>.
  p({ node, children }) {
    const img = node.children.find(c => c.tagName === 'img');
    if (img) return <figure className={figClass(fixUrl(img.properties.src))}>{children}</figure>;
    const kids = node.children.filter(c => c.type !== 'text' || c.value.trim()), first = kids[0];
    if (kids.length === 1 && first.tagName === 'strong') return <h3>{children}</h3>; // a bold line on its own is a subheading
    const lead = first?.tagName === 'strong' ? calloutOf(plain(first)) : null;
    return <p className={lead ?? undefined}>{children}</p>;
  },
  img({ src, alt }) {
    const s = imgSize(src);
    return <ZoomImage src={src} alt={alt} width={s?.[0]} height={s?.[1]} loading="lazy" />;
  },
  a({ node, href, children }) {
    const t = textOf(children);
    if (href?.startsWith('/lectures/') && /interactive/i.test(t)) return <Button as={Link} href={href}>{t}</Button>;
    const raw = node.children[0]?.tagName === 'code' && t.match(/^(L\w+?)\.md$/); // a file name as link text: show the note's name
    return <a href={href}>{raw ? label(raw[1]) ?? t : children}</a>;
  },
  summary({ children }) {
    const noun = /answers/i.test(textOf(children)) ? 'answers' : 'answer';
    return <summary className="btn btn-push"><span className="if-closed">Show {noun}</span><span className="if-open">Hide {noun}</span></summary>;
  },
};

// Puts each part in a coloured <section>, and turns heading numbers ("A4.", "Slide 6 —") into one consistent label.
function decorate() {
  const NUMBER = /^((?:Part [\w]+|Slide \d+|Slides \d+[–-]\d+|Page \d+|Pages \d+[–-]\d+|[AB]\d+|\d+))\s*(?:[.:—–]\s*)/;
  return tree => {
    const walk = n => {
      for (const c of n.children ?? []) {
        if (/^h[12]$/.test(c.tagName ?? '') && c.children[0]?.type === 'text') {
          const m = c.children[0].value.match(NUMBER);
          if (m) { c.children[0].value = c.children[0].value.slice(m[0].length); c.children.unshift({ type: 'element', tagName: 'span', properties: { className: ['h-tag'] }, children: [{ type: 'text', value: m[1] }] }); }
        }
        walk(c);
      }
    };
    walk(tree);
    const out = [];
    let seen = false, cur = null;
    for (const c of tree.children) {
      if (c.tagName === 'h1' && seen) { const part = plain(c).match(/^Part ([A-Z])/)?.[1]?.toLowerCase(); cur = { type: 'element', tagName: 'section', properties: { className: ['part', part ? `part-${part}` : 'part-x'] }, children: [] }; out.push(cur); }
      if (c.tagName === 'h1') seen = true;
      (cur ? cur.children : out).push(c);
    }
    tree.children = out;
  };
}

// Runs after rehype-slug: lists the parts and sections for the navigator, keeping any maths in headings.
function collectHeadings(list) {
  return tree => {
    const text = n => (n.type === 'text' ? n.value : n.tagName === 'code' && n.properties?.className?.includes('language-math') ? `$${(n.children ?? []).map(text).join('')}$` : (n.children ?? []).map(text).join(''));
    const walk = n => { for (const c of n.children ?? []) { if (c.tagName === 'h1' || c.tagName === 'h2') list.push({ id: c.properties.id, depth: +c.tagName[1], text: text(c) }); walk(c); } };
    walk(tree);
  };
}

export default async function Notes({ params, print }) {
  const { lecture, part } = await params, lec = getLecture(lecture), i = lec?.notes.findIndex(x => x.id === part) ?? -1;
  if (i < 0) notFound();
  const note = lec.notes[i], prev = lec.notes[i - 1], next = lec.notes[i + 1], base = `/lectures/${lec.id}`;
  let md = await fs.readFile(path.join(process.cwd(), `public/notes/${note.file}.md`), 'utf8');
  if (print) md = md.replaceAll('<details>', '<details open>'); // PDF should show practice answers
  else md = md.replace(/^## Contents\n[\s\S]*?(?=^---$|^# )/m, ''); // the navigator replaces the inline list
  const headings = [];
  const body = Markdown({ // called directly so `headings` is filled before the navigator renders
    children: md,
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeRaw, rehypeSlug, [collectHeadings, headings], [rehypeKatex, { strict: false, throwOnError: false }], decorate],
    urlTransform: fixUrl,
    components,
  });
  return (
    <div className="note-layout">
      <div className="note-main">
        <Toolbar className="note-actions">
          <Button as="a" href={`/notes/${note.file}.pdf`} download>PDF</Button>
          <Button as="a" href={`/notes/${note.file}.md`} download>Markdown</Button>
          <Button as={Link} href={base}>Interactive sections</Button>
        </Toolbar>
        <article className="prose">{body}</article>
        <div className="pager">
          {prev ? <Button as={Link} variant="ghost" href={`${base}/notes/${prev.id}`}>‹ {prev.label}</Button> : <span />}
          {next && <Button as={Link} variant="ghost" href={`${base}/notes/${next.id}`}>{next.label} ›</Button>}
        </div>
      </div>
      <NoteToc items={headings.map(h => ({ ...h, label: rich(h.text) }))} />
    </div>
  );
}
