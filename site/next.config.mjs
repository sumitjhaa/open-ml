import { PHASE_DEVELOPMENT_SERVER } from 'next/constants.js';

// Separate output dirs so `yarn build` / `yarn pdf` never overwrite a running `yarn dev` (which caused 500s).
// The production build is plain files in out/, so any static host can serve it. Cache headers live in vercel.json.
export default phase => ({ trailingSlash: true, ...(phase === PHASE_DEVELOPMENT_SERVER ? { distDir: '.next-dev' } : { output: 'export' }) });
