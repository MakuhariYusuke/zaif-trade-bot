"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const config_1 = require("vitest/config");
exports.default = (0, config_1.defineConfig)({
    test: {
        environment: 'node',
        coverage: {
            reporter: ['text', 'json', 'html'],
            reportsDirectory: 'coverage'
        },
        include: ['archived/ts_legacy/__tests__/**/*.test.ts']
    }
});
