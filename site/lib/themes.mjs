// Every theme in the popup, using the palettes of popular VS Code themes. "system" = no data-theme attribute: light or dark from the OS.
// A row is [id, name, background, text, sidebar, accent, blue, orange, green, red]. Everything else is derived, and colours too faint
// to read on their background are nudged until they pass the contrast limits (tested in scripts/check.mjs).
const LIGHT = [
  ['light', 'Light Modern', '#ffffff', '#3b3b3b', '#f8f8f8', '#005fb8', '#0451a5', '#a31515', '#098658', '#cd3131'],
  ['light-plus', 'Light+', '#ffffff', '#000000', '#f3f3f3', '#007acc', '#0000ff', '#795e26', '#008000', '#a31515'],
  ['quiet-light', 'Quiet Light', '#f5f5f5', '#333333', '#f2f2f2', '#4b69c6', '#4b69c6', '#ab6526', '#448c27', '#aa3731'],
  ['solarized-light', 'Solarized Light', '#fdf6e3', '#586e75', '#eee8d5', '#268bd2', '#268bd2', '#cb4b16', '#859900', '#dc322f'],
  ['github-light', 'GitHub Light', '#ffffff', '#1f2328', '#f6f8fa', '#0969da', '#0550ae', '#953800', '#116329', '#cf222e'],
  ['one-light', 'One Light', '#fafafa', '#383a42', '#eaeaeb', '#526fff', '#4078f2', '#c18401', '#50a14f', '#e45649'],
  ['ayu-light', 'Ayu Light', '#fcfcfc', '#5c6166', '#f3f4f5', '#f29718', '#399ee6', '#fa8d3e', '#86b300', '#f07171'],
  ['catppuccin-latte', 'Catppuccin Latte', '#eff1f5', '#4c4f69', '#e6e9ef', '#8839ef', '#1e66f5', '#fe640b', '#40a02b', '#d20f39'],
  ['rose-pine-dawn', 'Rosé Pine Dawn', '#faf4ed', '#575279', '#fffaf3', '#907aa9', '#286983', '#ea9d34', '#56949f', '#b4637a'],
  ['gruvbox-light', 'Gruvbox Light', '#fbf1c7', '#3c3836', '#ebdbb2', '#076678', '#076678', '#af3a03', '#79740e', '#9d0006'],
  ['tokyo-night-light', 'Tokyo Night Light', '#d5d6db', '#343b58', '#cbccd1', '#34548a', '#34548a', '#965027', '#485e30', '#8c4351'],
  ['night-owl-light', 'Night Owl Light', '#fbfbfb', '#403f53', '#f0f0f0', '#4876d6', '#4876d6', '#c96765', '#08916a', '#de3d3b'],
  ['material-lighter', 'Material Lighter', '#fafafa', '#546e7a', '#f2f2f2', '#39adb5', '#6182b8', '#f76d47', '#91b859', '#e53935'],
  ['everforest-light', 'Everforest Light', '#fdf6e3', '#5c6a72', '#f4f0d9', '#8da101', '#3a94c5', '#f57d26', '#8da101', '#f85552'],
  ['horizon-bright', 'Horizon Bright', '#fdf0ed', '#16161c', '#fadad1', '#da103f', '#1d8991', '#f6661e', '#07a06a', '#da103f'],
];
const DARK = [
  ['dark', 'Dark Modern', '#1f1f1f', '#cccccc', '#181818', '#0078d4', '#569cd6', '#ce9178', '#4ec9b0', '#f14c4c'],
  ['dark-plus', 'Dark+', '#1e1e1e', '#d4d4d4', '#252526', '#007acc', '#9cdcfe', '#ce9178', '#6a9955', '#f44747'],
  ['monokai', 'Monokai', '#272822', '#f8f8f2', '#1e1f1c', '#a6e22e', '#66d9ef', '#fd971f', '#a6e22e', '#f92672'],
  ['dracula', 'Dracula', '#282a36', '#f8f8f2', '#21222c', '#bd93f9', '#8be9fd', '#ffb86c', '#50fa7b', '#ff5555'],
  ['one-dark-pro', 'One Dark Pro', '#282c34', '#abb2bf', '#21252b', '#61afef', '#61afef', '#d19a66', '#98c379', '#e06c75'],
  ['nord', 'Nord', '#2e3440', '#d8dee9', '#292e39', '#88c0d0', '#81a1c1', '#d08770', '#a3be8c', '#bf616a'],
  ['solarized', 'Solarized Dark', '#002b36', '#93a1a1', '#00212b', '#2aa198', '#268bd2', '#cb4b16', '#859900', '#dc322f'],
  ['github-dark', 'GitHub Dark', '#0d1117', '#e6edf3', '#010409', '#2f81f7', '#79c0ff', '#ffa657', '#7ee787', '#ff7b72'],
  ['tokyo-night', 'Tokyo Night', '#1a1b26', '#a9b1d6', '#16161e', '#7aa2f7', '#7aa2f7', '#ff9e64', '#9ece6a', '#f7768e'],
  ['catppuccin-mocha', 'Catppuccin Mocha', '#1e1e2e', '#cdd6f4', '#181825', '#cba6f7', '#89b4fa', '#fab387', '#a6e3a1', '#f38ba8'],
  ['gruvbox-dark', 'Gruvbox Dark', '#282828', '#ebdbb2', '#1d2021', '#83a598', '#83a598', '#fe8019', '#b8bb26', '#fb4934'],
  ['ayu-mirage', 'Ayu Mirage', '#1f2430', '#cccac2', '#1c212b', '#ffcc66', '#73d0ff', '#ffad66', '#d5ff80', '#f28779'],
  ['night-owl', 'Night Owl', '#011627', '#d6deeb', '#01111d', '#82aaff', '#82aaff', '#f78c6c', '#addb67', '#ef5350'],
  ['rose-pine', 'Rosé Pine', '#191724', '#e0def4', '#1f1d2e', '#c4a7e7', '#9ccfd8', '#f6c177', '#56949f', '#eb6f92'],
  ['palenight', 'Palenight', '#292d3e', '#a6accd', '#232635', '#82aaff', '#82aaff', '#f78c6c', '#c3e88d', '#ff5370'],
];

const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const hex = c => '#' + c.map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
export const mix = (a, b, t) => { const A = rgb(a), B = rgb(b); return hex(A.map((v, i) => v + (B[i] - v) * t)); };
const lum = h => { const [r, g, b] = rgb(h).map(v => { const s = v / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
export const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
// Move a colour towards black or white, whichever is away from the background, until it reaches the ratio.
function readable(c, bg, ratio) {
  const end = lum(bg) > 0.18 ? '#000000' : '#ffffff';
  let out = c;
  for (let t = 0.04; contrast(out, bg) < ratio && t <= 1; t += 0.04) out = mix(c, end, t);
  return out;
}

function tokens([, , bg, fgRaw, side, accentRaw, b, o, g, r], scheme) {
  const fg = readable(fgRaw, bg, 7);
  const accent = readable(accentRaw, bg, 3), c = [b, o, g, r].map(x => readable(x, bg, 4.5));
  const onaccent = contrast('#ffffff', accent) >= contrast('#000000', accent) ? '#ffffff' : '#000000';
  return { 'color-scheme': scheme, bg, fg, muted: readable(mix(fg, bg, 0.38), bg, 4.5), muted2: mix(bg, fg, 0.2), line: mix(bg, fg, 0.13), card: mix(bg, fg, 0.045), side, selgray: mix(side, fg, 0.13), accent, onaccent, in: mix(bg, accent, 0.16), c1: c[0], c2: c[1], c3: c[2], c4: c[3], alert: c[3], okc: c[2] };
}

const build = (row, scheme) => ({ id: row[0], name: row[1], tokens: tokens(row, scheme) });
export const LIGHT_THEMES = LIGHT.map(r => build(r, 'light'));
export const DARK_THEMES = DARK.map(r => build(r, 'dark'));
export const THEMES = [{ id: 'system', name: 'System' }, ...LIGHT_THEMES, ...DARK_THEMES];

const block = t => Object.entries(t.tokens).map(([k, v]) => (k === 'color-scheme' ? `color-scheme:${v}` : `--${k}:${v}`)).join(';');
export const themeCss = [
  `:root{${block(LIGHT_THEMES[0])}}`,
  `@media (prefers-color-scheme:dark){:root:not([data-theme]){${block(DARK_THEMES[0])}}}`,
  ...[...LIGHT_THEMES, ...DARK_THEMES].map(t => `:root[data-theme="${t.id}"]{${block(t)}}`),
].join('\n');
