// Prints each lecture's notes page to PDF with headless Chrome so maths renders with KaTeX exactly as on the site.
// Usage:  yarn pdf http://localhost:3000   (print from a running dev/prod server)
//         yarn build && yarn pdf            (starts its own `next start` on a free port)
// Set CHROME=/path/to/chrome if not on macOS.
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'node:net';
import { LECTURES } from '../lib/lectures.mjs';

const chrome = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const freePort = () => new Promise(res => { const s = createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });

let base = process.argv[2], server;
if (!base) {
  const port = await freePort();
  base = `http://127.0.0.1:${port}`;
  server = spawn(process.execPath, ['node_modules/next/dist/bin/next', 'start', '-H', '127.0.0.1', '-p', String(port)], { stdio: 'inherit' });
}
try {
  for (let i = 0; ; i++) {
    try {
      const html = await (await fetch(`${base}/lectures/${LECTURES[0].id}/notes/${LECTURES[0].notes[0].id}`)).text();
      if (html.includes('Epocher')) break; // make sure it's this site, not another app on the port
    } catch {}
    if (i > 60) throw new Error(`no Epocher site at ${base} — run \`yarn build\` first or pass a running server URL`);
    await new Promise(r => setTimeout(r, 500));
  }
  for (const [l, n] of LECTURES.flatMap(x => x.notes.map(y => [x, y]))) {
    const out = `public/notes/${n.file}.pdf`;
    execFileSync(chrome, ['--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=15000',
      `--print-to-pdf=${out}`, `${base}/lectures/${l.id}/notes/${n.id}/print`], { stdio: 'inherit' });
    console.log('wrote', out);
  }
} finally {
  server?.kill();
}
