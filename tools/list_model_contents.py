import sys
import zipfile

p = "models/sac_v444_advanced_regime_adaptation.zip"
try:
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        print("entries=", len(names))
        print("all entries:")
        for n in names:
            print("  ", n)
        matches = [
            n
            for n in names
            if "scaler" in n.lower()
            or n.endswith(".npz")
            or "vecnormalize" in n.lower()
        ]
        print("matches:", matches)
except Exception as e:
    print("ERROR", e)
    sys.exit(2)
