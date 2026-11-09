import 'katex/dist/katex.min.css';
import './globals.css';
import Shell from '../components/Shell';
import { THEMES, themeCss } from '../lib/themes.mjs';

export const metadata = {
  title: 'Epocher',
  description: 'ES 335 / ES 678 Machine Learning — interactive explorers, maths from the ground up, full lecture notes.',
};

// Runs before paint so a saved light/dark choice doesn't flash.
const ids = JSON.stringify(THEMES.map(t => t.id).filter(id => id !== 'system'));
// Registered only in production so development never serves stale files.
const swScript = `if('serviceWorker'in navigator)addEventListener('load',function(){navigator.serviceWorker.register('/sw.js').catch(function(){})})`;
const themeScript = `try{var t=localStorage.getItem('theme');if(${ids}.indexOf(t)>=0)document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><style dangerouslySetInnerHTML={{ __html: themeCss }} /><script dangerouslySetInnerHTML={{ __html: themeScript }} />{process.env.NODE_ENV === 'production' && <script dangerouslySetInnerHTML={{ __html: swScript }} />}</head>
      <body><Shell>{children}</Shell></body>
    </html>
  );
}
