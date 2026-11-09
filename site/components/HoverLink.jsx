'use client';
import { useState } from 'react';
import Link from 'next/link';

// Long link lists would otherwise fetch every page they show. This fetches a page only when it is about to be opened.
export default function HoverLink(props) {
  const [soon, setSoon] = useState(false);
  return <Link prefetch={soon ? null : false} onMouseEnter={() => setSoon(true)} onFocus={() => setSoon(true)} onTouchStart={() => setSoon(true)} {...props} />;
}
