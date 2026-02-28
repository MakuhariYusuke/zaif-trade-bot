import json, collections

records = []
for line in open("results/v460/fill_test/fill_records_20260225.jsonl", encoding="utf-8"):
    records.append(json.loads(line.strip()))

# cancel_reason 分布
reasons = collections.Counter(r.get("cancel_reason", "filled") for r in records)
print(f"Total records: {len(records)}")
for reason, count in reasons.most_common(20):
    print(f"  {reason}: {count}")

# blank git_sha count
blank_sha = sum(1 for r in records if not r.get("git_sha"))
print(f"\nBlank git_sha: {blank_sha}/{len(records)}")

# spread_too_narrow count
narrow = sum(1 for r in records if r.get("cancel_reason") == "spread_too_narrow")
print(f"Spread too narrow: {narrow}")

# fast_fill_defense統計
ffd = [r for r in records if r.get("cancel_reason") == "filled" and r.get("wait_sec", 999) < 15]
print(f"\nFast fill (wait<15s): {len(ffd)}")
if ffd:
    waits = [r["wait_sec"] for r in ffd if "wait_sec" in r]
    if waits:
        print(f"  avg wait: {sum(waits)/len(waits):.1f}s, min: {min(waits):.1f}s, max: {max(waits):.1f}s")
