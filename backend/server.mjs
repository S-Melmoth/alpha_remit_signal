import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), 'public');
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.png': 'image/png', '.svg': 'image/svg+xml', '.pdf': 'application/pdf' };
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    const routes = { '/': '/index.html', '/payments': '/payments.html', '/payments/': '/payments.html' };
    const featureRoute = /^\/features\/(behaviour|news|recommendation|prefill|scheduler)\/?$/.test(url.pathname);
    const file = path.resolve(root, '.' + decodeURIComponent(featureRoute ? '/feature.html' : routes[url.pathname] || url.pathname));
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  } catch { res.writeHead(404).end('Not found'); }
});
server.listen(Number(process.env.PORT) || 5173, '0.0.0.0', () => console.log(`Прототип: http://localhost:${server.address().port}`));
