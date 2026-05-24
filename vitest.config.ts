import { defineConfig } from 'vitest/config';

// Vitest's default include glob (**/*.test.ts) pulls in stale test files
// from agent worktrees under .claude/worktrees/, causing duplicated runs of
// out-of-date environment.test.ts copies. Pin the include to src/ and
// explicitly exclude the worktree tree so CI runs only the live tests.
export default defineConfig({
  test: {
    include: ['src/**/*.{test,spec}.?(c|m)[jt]s?(x)'],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '.claude/worktrees/**',
      'bench-out/**',
    ],
  },
});
