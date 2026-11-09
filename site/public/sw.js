// Hashed build files never change, so they come from the cache. Pages come from the network and fall back to the last copy when offline.
const LIMIT = { 'epocher-static': 400, 'epocher-pages': 100 };

async function keep(name, request, response) {
  const cache = await caches.open(name);
  await cache.put(request, response);
  const keys = await cache.keys();
  await Promise.all(keys.slice(0, -LIMIT[name]).map(k => cache.delete(k)));
}

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const r = e.request, url = new URL(r.url);
  if (r.method !== 'GET' || url.origin !== location.origin) return;
  const save = (name, res) => { if (res.ok) e.waitUntil(keep(name, r, res.clone())); return res; };
  if (url.pathname.startsWith('/_next/static/')) e.respondWith(caches.match(r).then(hit => hit ?? fetch(r).then(res => save('epocher-static', res))));
  else if (r.mode === 'navigate' && !url.pathname.endsWith('.pdf')) e.respondWith(fetch(r).then(res => save('epocher-pages', res)).catch(() => caches.match(r).then(hit => hit ?? Response.error())));
});
