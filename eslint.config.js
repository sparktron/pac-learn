import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

export default tseslint.config(
  // Build output, deps, and run artifacts are not linted.
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'coverage/**',
      'bench-out/**',
      'notebooklm_diagnostics/**',
      '*.config.js',
      '*.config.ts',
    ],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  // App source: browser globals + React hooks rules.
  {
    files: ['src/**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    languageOptions: {
      globals: { ...globals.browser },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // The codebase already annotates every intentional dependency-array
      // omission with an inline disable + rationale; keep the rule on so new
      // ones must be justified the same way.
      'react-hooks/exhaustive-deps': 'warn',
      // console.log is debug noise in the app; console.warn/error are
      // legitimate operator-facing diagnostics (e.g. the agents' load guards).
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      // Native dialogs are used deliberately in a couple of spots, each with an
      // inline disable — flag any new, unannotated ones.
      'no-alert': 'error',
      // Allow intentionally-unused args/vars prefixed with _ (e.g. wrapPosition's
      // _height, drawWall placeholders).
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },

  // Headless scripts + tests run under Node and print to the console freely.
  {
    files: ['scripts/**/*.ts', 'src/**/*.test.ts', '*.config.ts'],
    languageOptions: {
      globals: { ...globals.node },
    },
    rules: {
      'no-console': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
);
