import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync, writeFileSync } from 'fs';
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

export default defineConfig({
  plugins: [react(), removeScriptCrossorigin()],
});
