"""Download MicroSAM's light-microscopy SAM weights (vit_b_lm, 375 MB) used for mask verification.

    python backend/scripts/download_sam.py
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt"
DEST = Path(__file__).resolve().parent.parent / "app" / "model" / "microsam_vit_b_lm.pt"


def main():
    if DEST.exists():
        print(f"Already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")

    def hook(blocks, size, total):
        done = blocks * size
        if total > 0:
            sys.stdout.write(f"\r{min(done, total) / 1e6:.0f} / {total / 1e6:.0f} MB")
            sys.stdout.flush()

    print(f"Downloading MicroSAM vit_b_lm → {DEST}")
    urllib.request.urlretrieve(URL, tmp, hook)
    tmp.replace(DEST)
    print("\nDone.")


if __name__ == "__main__":
    main()
