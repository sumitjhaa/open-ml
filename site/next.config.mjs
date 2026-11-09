import { PHASE_DEVELOPMENT_SERVER } from 'next/constants.js';

// Separate output dirs so `yarn build` / `yarn pdf` never overwrite a running `yarn dev` (which caused 500s).
// Figures keep their names between deploys, so browsers reuse them for a day and then refresh in the background.
const assetCache = [{ key: 'Cache-Control', value: 'public, max-age=86400, stale-while-revalidate=604800' }];
const headers = async () => ['/images/:path*', '/notebooks/img/:path*', '/icons/:path*'].map(source => ({ source, headers: assetCache }));

export default phase => ({ distDir: phase === PHASE_DEVELOPMENT_SERVER ? '.next-dev' : '.next', headers });
