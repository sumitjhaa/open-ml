import { PHASE_DEVELOPMENT_SERVER } from 'next/constants.js';

// Separate output dirs so `yarn build` / `yarn pdf` never overwrite a running `yarn dev` (which caused 500s).
export default phase => ({ distDir: phase === PHASE_DEVELOPMENT_SERVER ? '.next-dev' : '.next' });
