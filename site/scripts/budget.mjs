// Fails when a page's JavaScript grows past its budget (gzipped KB). Run after `next build`.
import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const BUDGET = { '/layout': 150, '/lectures/[lecture]/explore/[id]/page': 220, '/lectures/[lecture]/notes/[part]/page': 125 };
const { pages } = JSON.parse(readFileSync('.next/app-build-manifest.json', 'utf8'));
let over = 0;
for (const [route, limit] of Object.entries(BUDGET)) {
  if (!pages[route]) throw new Error(`no such route in the build: ${route}`);
  const kb = pages[route].filter(f => f.endsWith('.js')).reduce((a, f) => a + gzipSync(readFileSync(`.next/${f}`)).length, 0) / 1024;
  console.log(`${route}: ${kb.toFixed(0)} KB of ${limit} KB`);
  if (kb > limit) over++;
}
if (over) process.exit(1);
