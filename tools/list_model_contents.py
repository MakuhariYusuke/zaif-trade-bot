import zipfile

from ztb.utils.cli import run_main


def main():
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
            return 0
    except Exception as e:
        print("ERROR", e)
        return 2


if __name__ == "__main__":
    run_main(main)
