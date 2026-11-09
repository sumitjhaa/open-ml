// Python syntax colouring drawn from the theme's own colours. Renders on the server, so it adds no client code.
const KEYWORDS = 'and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield';
const BUILTINS = 'abs all any bool dict enumerate float int isinstance len list map max min open print range reversed round set sorted str sum super tuple type zip';
const TOKEN = new RegExp(`(#[^\\n]*)|("""[\\s\\S]*?"""|'''[\\s\\S]*?'''|[rbf]?"(?:[^"\\\\\\n]|\\\\.)*"|[rbf]?'(?:[^'\\\\\\n]|\\\\.)*')|\\b(${KEYWORDS.replace(/ /g, '|')})\\b|\\b(True|False|None|\\d+(?:\\.\\d+)?(?:e-?\\d+)?)\\b|\\b(${BUILTINS.replace(/ /g, '|')})\\b(?=\\()|\\b([A-Za-z_]\\w*)(?=\\()`, 'g');
const KINDS = [null, 'hl-com', 'hl-str', 'hl-kw', 'hl-num', 'hl-bi', 'hl-fn'];

export function highlight(src) {
  const out = [];
  let last = 0;
  for (const m of src.matchAll(TOKEN)) {
    if (m.index > last) out.push(src.slice(last, m.index));
    const kind = KINDS.findIndex((_, i) => i && m[i] !== undefined);
    out.push(<span key={m.index} className={KINDS[kind]}>{m[0]}</span>);
    last = m.index + m[0].length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}
