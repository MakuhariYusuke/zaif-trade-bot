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
      enabled: true,
      provider: 'v8',
      reporter: ['text', 'text-summary', 'lcov', 'json'],
      reportsDirectory: 'coverage',
      thresholds: {
        global: {
          statements: 80,
          branches: 80,
          functions: 80,
          lines: 80
        },
        perFile: true
      },
      exclude: [
        'src/index.ts',
        'src/app/**',
        'src/api/**',
        'src/tools/**',
        'dist/**',
        'vitest.config.*',
        'tsconfig.json',
        'package.json',
        // Keep strategy app wrappers and unused stubs out of coverage calc
        'src/application/strategies/**',
        // Generated or aggregated contract/type barrels
        'src/contracts/**',
        'src/types/**',
        '__tests__/helpers/**',
        '__tests__/mocks/**',
        'src/**/__mocks__/**',
      ]
    },
    include: ['__tests__/**/*.test.ts'],
    exclude: ['__tests__/**/*.test.js', '__tests__/*.test.ts']
  }
});
