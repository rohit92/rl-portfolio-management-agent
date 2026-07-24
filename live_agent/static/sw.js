// Minimal service worker — required for "Install app" on Chrome/Android.
// Network-first passthrough: live market data must never be served stale,
// so we do NOT cache API responses. The shell loads from network too (the
// server is on your own machine, so there's no latency to hide).
self.addEventListener("install", (e) => self.skipWaiting());
self.addEventListener("activate", (e) => self.clients.claim());
self.addEventListener("fetch", (e) => {
  e.respondWith(fetch(e.request).catch(() =>
    new Response("Offline — start the dashboard server on your Mac "
      + "(python webapp.py) and reload.", {status: 503,
      headers: {"Content-Type": "text/plain"}})));
});
