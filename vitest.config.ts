import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    coverage: {
      reporter: ['text', 'json', 'html'],
      reportsDirectory: 'coverage',
      exclude: [
        'src/index.ts',
        'src/app/**',
        'src/api/**',
        'src/tools/**',
        // services: exclude utility wrappers we don't test in unit scope
        'src/services/risk-service.ts',
        'src/services/market-service.ts',
        'src/services/risk-guards.ts',
      ]
    },
  include: ['__tests__/unit/**/*.test.ts'],
  exclude: ['__tests__/**/*.test.js', '__tests__/*.test.ts']
  }
});
