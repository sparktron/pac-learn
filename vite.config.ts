import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { createReadStream, readdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { join } from 'path';
import type { NormalizedOutputOptions } from 'rollup';

const removeScriptCrossorigin = () => ({
  name: 'remove-script-crossorigin',
  writeBundle(options: NormalizedOutputOptions) {
    const htmlPath = join(options.dir, 'index.html');
    let html = readFileSync(htmlPath, 'utf-8');
    html = html.replace(/<script([^>]*)\scrossorigin/g, '<script$1');
    writeFileSync(htmlPath, html);
  },
});

// Walks bench-out/ for the most recently modified policy file (prefers merged
// federated outputs, falls back to per-run snapshots) and streams it to the
// browser as /trained-policy.json so the UI's "trained" preset can resume
// from the latest training checkpoint without manual file uploads.
const serveTrainedPolicy = () => ({
  name: 'serve-trained-policy',
  configureServer(server: { middlewares: { use: (path: string, handler: (req: unknown, res: { statusCode: number; setHeader: (k: string, v: string) => void; end: (b?: string) => void; }) => void) => void } }) {
    server.middlewares.use('/trained-policy.json', (_req, res) => {
      const root = join(process.cwd(), 'bench-out');
      let best: { path: string; mtime: number; rank: number } | null = null;
      const walk = (dir: string, depth: number): void => {
        if (depth > 3) return;
        let entries;
        try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }
        for (const e of entries) {
          if (e.name.startsWith('.') || e.name === '_archive' || e.name === 'node_modules') continue;
          const p = join(dir, e.name);
          if (e.isDirectory()) { walk(p, depth + 1); continue; }
          if (!e.isFile()) continue;
          // Rank: merged > latest. Lets a merged file win even when an
          // individual worker snapshot has a fractionally newer mtime.
          const rank = e.name === 'policy-merged.json' ? 2
                     : e.name === 'policy-latest.json' ? 1 : 0;
          if (rank === 0) continue;
          // Guard statSync: a worker can delete/rename a file mid-walk
          // (atomic-write tmpfile patterns do this). Without try/catch
          // the dev server returns 500 and breaks the UI's load flow.
          let mtime: number;
          try { mtime = statSync(p).mtimeMs; } catch { continue; }
          if (!best || rank > best.rank || (rank === best.rank && mtime > best.mtime)) {
            best = { path: p, mtime, rank };
          }
        }
      };
      walk(root, 0);
      if (!best) {
        res.statusCode = 404;
        res.setHeader('content-type', 'text/plain');
        res.end('No policy-merged.json or policy-latest.json found under bench-out/');
        return;
      }
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.setHeader('x-policy-source', best.path);
      createReadStream(best.path).pipe(res as unknown as NodeJS.WritableStream);
    });
  },
});

export default defineConfig({
  plugins: [react(), removeScriptCrossorigin(), serveTrainedPolicy()],
});
