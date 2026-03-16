import os
from collections import defaultdict

base = "tests/unit/trading"
dups = defaultdict(list)
for root, dirs, files in os.walk(base):
    for f in files:
        if f.startswith("test_") and f.endswith(".py"):
            dups[f].append(os.path.join(root, f))
for k, v in sorted(dups.items()):
    if len(v) > 1:
        print("\nDUPLICATE: " + k)
        for p in v:
            print("    ", p)
        # Suggest canonical file (prefer deeper namespaced tests)
        canonical = sorted(v, key=lambda x: (-len(x.split(os.sep)), x))[0]
        print("    Suggested canonical: ", canonical)
        print("    Suggested actions:")
        for p in v:
            if p != canonical:
                print("        - Archive or convert to shim:", p)
