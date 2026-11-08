import katex from 'katex';

// <M t="\hat R" /> inline maths, <M d t="..." /> display maths.
export default function M({ t, d = false }) {
  const html = katex.renderToString(t, { displayMode: d, throwOnError: false, strict: false });
  return <span className={d ? 'mdisp' : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
}
