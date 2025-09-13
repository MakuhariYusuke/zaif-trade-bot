import path from 'path';

type ToolMap = Record<string, string>;

const tools: ToolMap = {
  // live
  'live:health': 'live/health.ts',
  'live:test-minimal-live': 'live/test-minimal-live.ts',
  'live:test-minimal-trade': 'live/test-minimal-trade.ts',
  'live:test-coincheck': 'live/test-coincheck.ts',
  'live:test-coincheck-flow': 'live/test-coincheck-flow.ts',
  'live:test-zaif': 'live/test-zaif.ts',
  // paper
  'paper:mock-scenario': 'paper/mock-scenario.ts',
  'paper:mock-smoke': 'paper/mock-smoke.ts',
  'paper:smoke-once': 'paper/smoke-once.ts',
  'paper:reset': 'paper/paper-reset.ts',
  // ml
  'ml:export': 'ml/ml-export.ts',
  'ml:search': 'ml/ml-search.ts',
  'ml:simulate': 'ml/ml-simulate.ts',
  // stats
  'stats:today': 'stats/stats-today.ts',
  'stats:graph': 'stats/stats-graph.ts',
  // tests/util
  'tests:flow': 'tests/test-flow.ts',
  'balance': 'balance.ts',
  'nonce:bump': 'nonce-bump.ts'
};

function printHelp() {
  const entries = Object.keys(tools).sort();
  console.log('Usage: npm run tool -- <name> [-- ...args]\n');
  console.log('Available tools:');
  for (const k of entries) console.log(`  - ${k}`);
}

(async () => {
  const args = process.argv.slice(2);
  const name = args[0];
  if (!name || name === 'help' || name === '--help' || name === '-h') {
    printHelp();
    process.exit(0);
  }
  const rel = tools[name];
  if (!rel) {
    console.error(`[tool] unknown name: ${name}`);
    printHelp();
    process.exit(1);
  }
  const file = path.resolve(process.cwd(), 'src', 'tools', rel);
  // Delegate execution to the selected tool (ts-node context)
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require(file);
})();
