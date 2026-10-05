"""
data/download_tcia.py
─────────────────────────────────────────────────────────────────────────────
Download the Matek et al. 2021 Bone Marrow Cytomorphology dataset from TCIA.

Dataset details:
  Title:   An Expert-Annotated Dataset of Bone Marrow Cytology in
           Hematologic Malignancies
  Authors: Matek, Schwarz, Marr, Spiekermann (2021)
  DOI:     https://doi.org/10.7937/TCIA.AXH3-T579
  License: CC BY 4.0
  TCIA URL: https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/

Requirements:
  pip install tcia_utils requests tqdm

Usage:
  python data/download_tcia.py --out data/raw --format png

Notes:
  • TCIA registration is required before downloading.
    Create a free account at https://www.cancerimagingarchive.net/
  • Attribution: Please cite the dataset using the DOI above in any
    publication that uses this data.
  • This script uses the NBIA Data Retriever API (no Java client required).
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from tcia_utils import nbia
except ImportError:
    sys.exit(
        "tcia_utils is not installed. Run:\n"
        "  pip install tcia_utils\n"
        "then re-run this script."
    )

try:
    import requests
    from tqdm import tqdm
except ImportError:
    sys.exit("requests and tqdm are required. Run: pip install requests tqdm")


# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer"
TCIA_DOI        = "https://doi.org/10.7937/TCIA.AXH3-T579"
LICENSE         = "CC BY 4.0"

CITATION = """
If you use this dataset, please cite:
  Matek C, Schwarz S, Marr C, Spiekermann K.
  "Human-level recognition of blast cells in aggressive haematological
  malignancies."
  Nature Machine Intelligence (2019).
  DOI: 10.7937/TCIA.AXH3-T579
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".tcia_download"
    marker.write_text(
        f"Collection: {COLLECTION_NAME}\n"
        f"DOI: {TCIA_DOI}\n"
        f"License: {LICENSE}\n"
        f"Downloaded: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )


def get_series_list(collection: str) -> list[dict]:
    """Return list of series dicts for the given TCIA collection."""
    print(f"Fetching series list for collection: {collection!r} …")
    series = nbia.getSeries(collection=collection)
    if not series:
        sys.exit(
            f"No series found for collection {collection!r}.\n"
            "Check that:\n"
            "  1. You are authenticated (run nbia.getToken() first if needed).\n"
            "  2. The collection name is correct.\n"
            f"  3. The dataset is accessible at {TCIA_DOI}"
        )
    print(f"Found {len(series)} series.")
    return series


def download_series(series_list: list[dict], out_dir: Path) -> None:
    """Download each series into out_dir/<SeriesInstanceUID>/."""
    failed = []
    for i, s in enumerate(tqdm(series_list, desc="Downloading series", unit="series")):
        uid = s.get("SeriesInstanceUID", f"series_{i}")
        series_dir = out_dir / uid
        if series_dir.exists() and any(series_dir.iterdir()):
            # Already downloaded — skip
            continue
        series_dir.mkdir(parents=True, exist_ok=True)
        try:
            nbia.downloadSeries(series_data=[s], path=str(series_dir), format="png")
        except Exception as exc:
            print(f"\n⚠️  Failed to download {uid}: {exc}")
            failed.append(uid)

    if failed:
        print(f"\n{len(failed)} series failed to download. UIDs saved to: {out_dir}/failed_series.txt")
        (out_dir / "failed_series.txt").write_text("\n".join(failed))
    else:
        print("\n✅ All series downloaded successfully.")


def write_manifest(series_list: list[dict], out_dir: Path) -> None:
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(series_list, f, indent=2)
    print(f"Manifest written to {manifest_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Matek 2021 Bone Marrow Cytomorphology dataset from TCIA."
    )
    parser.add_argument(
        "--out", default="data/raw",
        help="Output directory for downloaded images (default: data/raw)"
    )
    parser.add_argument(
        "--collection", default=COLLECTION_NAME,
        help=f"TCIA collection name (default: {COLLECTION_NAME})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List series without downloading."
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("=" * 70)
    print("  TCIA Bone Marrow Cytomorphology Downloader")
    print(f"  Collection : {args.collection}")
    print(f"  DOI        : {TCIA_DOI}")
    print(f"  License    : {LICENSE}")
    print("=" * 70)
    print(CITATION)
    print(
        "⚠️  You must have a TCIA account to download data.\n"
        "   Register at: https://www.cancerimagingarchive.net/\n"
        "   After registration, authenticate below.\n"
    )

    # Authenticate (prompts for username/password if not cached)
    try:
        nbia.getToken()
    except Exception as exc:
        sys.exit(f"Authentication failed: {exc}")

    series_list = get_series_list(args.collection)

    if args.dry_run:
        print("\nDry run — no files downloaded. Series list:")
        for s in series_list[:10]:
            print(f"  {s.get('SeriesInstanceUID', 'N/A')}")
        if len(series_list) > 10:
            print(f"  … and {len(series_list) - 10} more.")
        return

    check_output_dir(out_dir)
    write_manifest(series_list, out_dir)
    download_series(series_list, out_dir)

    print(f"\n✅ Download complete. Images saved to: {out_dir.resolve()}")
    print(CITATION)


if __name__ == "__main__":
    main()
