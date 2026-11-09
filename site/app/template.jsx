import { Suspense } from 'react';

// Next remounts a template on every navigation, which replays the page's fade-in.
// The boundary lets React wait for the rest of a very long page's data instead of discarding the server's HTML.
export default function Template({ children }) {
  return <div className="page"><Suspense fallback={null}>{children}</Suspense></div>;
}
