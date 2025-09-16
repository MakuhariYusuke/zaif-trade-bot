import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      '@contracts': path.resolve(__dirname, 'src', 'contracts'),
  '@utils': path.resolve(__dirname, 'src', 'utils'),
  '@adapters': path.resolve(__dirname, 'src', 'adapters'),
  '@core': path.resolve(__dirname, 'src', 'core'),
    }
  },
  test: {
    environment: 'node',
  setupFiles: ['src/test/setup.ts'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      reportsDirectory: 'coverage',
      thresholds: {
        statements: 70,
      },
      exclude: [
        'src/index.ts',
        'src/app/**',
        'src/api/**',
        'src/tools/**',
      ]
    },
  include: ['__tests__/unit/**/*.test.ts'],
  exclude: ['__tests__/**/*.test.js', '__tests__/*.test.ts']
  }
});
