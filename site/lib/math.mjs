// Tiny numeric toolkit for the explorers. Pure functions so scripts/check.mjs can test them in Node.
export const sum = a => a.reduce((s, x) => s + x, 0);
export const mean = a => sum(a) / a.length;
export function median(a) {
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Seeded PRNG (mulberry32) so every visitor sees the same "random" data.
export function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

// Gaussian elimination with partial pivoting: solves A x = b.
export function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// Least-squares polynomial fit via normal equations; returns coefficients c0..c_deg.
// shortcut: normal equations square the condition number; fine for deg <= 9 on [-1,1], use QR beyond that.
export function polyfit(xs, ys, deg) {
  const k = deg + 1;
  const A = Array.from({ length: k }, () => Array(k).fill(0)), b = Array(k).fill(0);
  xs.forEach((x, i) => {
    const p = Array.from({ length: k }, (_, j) => x ** j);
    for (let r = 0; r < k; r++) {
      b[r] += p[r] * ys[i];
      for (let c = 0; c < k; c++) A[r][c] += p[r] * p[c];
    }
  });
  return solve(A, b);
}
export const polyval = (c, x) => c.reduceRight((acc, ci) => acc * x + ci, 0);
export const mse = (f, xs, ys) => mean(xs.map((x, i) => (f(x) - ys[i]) ** 2));
export function linfit(xs, ys) {
  const [b, w] = polyfit(xs, ys, 1);
  return { w, b };
}
