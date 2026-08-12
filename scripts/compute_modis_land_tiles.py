"""
Compute the MODIS sinusoidal land-tile allowlist from a land-polygon layer.

docs/design/07a-modis-band-reference.md flags the commonly-cited "~317
land-covering tiles" figure as UNVERIFIED and says to "confirm against the
actual sinusoidal tile grid + a land mask before finalizing tile lists."
This script is that confirmation: it reprojects a land-polygon vector layer
into the MODIS sinusoidal grid's CRS and tests each of the 36x18 `hXXvYY`
tiles for overlap (`src.data.sources.modis.tiles.compute_land_tiles`).

Input: the `osm` source's PREPARE output, `land_polygons_simplified.gpkg`
(run `python run.py data run --source osm --step prepare` first if it
doesn't exist yet). GADM's `gadm_levelADM_0_simplified.gpkg` works too --
any polygon layer distinguishing land from ocean is fine.

Output: a `land_tiles:` YAML list, printed to stdout, ready to paste into
`orchestration/configs/data.yaml`'s `modis`/`modis_robustness_11a1` source
blocks (consumed by `ModisSource.__init__`, `src/data/sources/modis/source.py`).

Usage:
  python scripts/compute_modis_land_tiles.py \\
      --land-polygons /path/to/misc/prepared/osm/land_polygons_simplified.gpkg

  python scripts/compute_modis_land_tiles.py \\
      --land-polygons /path/to/land_polygons_simplified.gpkg \\
      --lat-clip-deg 60.0 --out land_tiles.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sources.modis.tiles import compute_land_tiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--land-polygons", required=True, help="Path to a land-polygon vector file (e.g. .gpkg)")
    parser.add_argument("--lat-clip-deg", type=float, default=60.0, help="Latitude clip (default: 60.0, matches data.yaml's default)")
    parser.add_argument("--out", default=None, help="Optional: also write the tile list as JSON to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    land_tiles = sorted(compute_land_tiles(args.land_polygons, lat_clip_deg=args.lat_clip_deg))

    print(f"# {len(land_tiles)} land tiles within |lat| <= {args.lat_clip_deg} degrees")
    print("land_tiles:")
    for tile in land_tiles:
        print(f'  - "{tile}"')

    if args.out:
        with open(args.out, "w") as f:
            json.dump(land_tiles, f, indent=2)
        print(f"\nWrote {len(land_tiles)} tile ids to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
