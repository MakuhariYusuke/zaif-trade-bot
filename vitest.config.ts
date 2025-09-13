import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    coverage: {
      reporter: ['text', 'json', 'html'],
      reportsDirectory: 'coverage'
    },
    include: ['__tests__/**/*.test.ts']
  }
});
