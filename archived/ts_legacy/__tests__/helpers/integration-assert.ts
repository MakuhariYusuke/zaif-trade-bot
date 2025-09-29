import fs from 'fs';

function fail(msg: string): never { console.error(msg); process.exit(1); }

try {
  const diffPath = fs.existsSync('stats-diff-2.json') ? 'stats-diff-2.json' : 'stats-diff.json';
  if (!fs.existsSync(diffPath)) fail(`${diffPath} missing`);
  const diff = JSON.parse(fs.readFileSync(diffPath,'utf8'));
  let inc = 0;
  if (Array.isArray(diff.pairsDiff)) {
    for (const p of diff.pairsDiff) {
      const d = p?.diff || {};
      inc += (d.buyEntries||0) + (d.sellEntries||0);
    }
  } else {
    inc = (diff?.diff?.buyEntries||0) + (diff?.diff?.sellEntries||0);
  }
  if (!(inc > 0)) fail(`entries diff not > 0: ${inc}`);
  if (!fs.existsSync('ml-search-top.json')) fail('ml-search-top.json missing');
  const top = JSON.parse(fs.readFileSync('ml-search-top.json','utf8'));
  if (!(Array.isArray(top.top) && top.top.length && top.top[0].winRate >= 0)) fail('ml top invalid');
  console.log(JSON.stringify({ ok: true, inc, winRate: top.top[0].winRate }));
} catch (e: any) {
  fail(`integration assert error: ${e?.message||String(e)}`);
}
