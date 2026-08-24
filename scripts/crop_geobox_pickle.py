#!/usr/bin/env python3
"""Crop a pickled GeoBox to a small window.

Extracted from orchestration/scripts/validate-hard-gate-acag.sh's original
inline snippet once a second hard-gate pilot (GADM) needed the same crop
logic -- one shared, tested implementation instead of two copies that could
silently drift (the exact class of bug jobs.yaml's consolidation fixed
earlier in this migration).

Why this exists: hard-gate pilots (docs/design/09-integrated-pipeline.md
step 9) compare OLD vs NEW code on the same input, but both reproject/
rasterize onto the *production* target grid via `get_or_create_geobox()`,
which is huge (~86,401x33,601px). That's pure overhead for the pilot's
actual purpose -- the reprojection/rasterization logic itself is unchanged
by the migration, so a tiny window proves equivalence just as well as the
whole planet (confirmed on ACAG: 50+ minutes with dask memory-pressure
churn -> ~90 seconds, both EQUIVALENT either way).

`GeoBox` supports plain numpy-style slicing (`gbox[y0:y1, x0:x1]`); the
result keeps the same CRS/resolution/affine alignment, just fewer pixels, so
`get_or_create_geobox()` (which only needs `.shape`/`.affine`/`.crs` off
whatever it unpickles) works on the cropped output with no other code
changes needed anywhere.

Usage:
    python scripts/crop_geobox_pickle.py INPUT_PKL OUTPUT_PKL --window-px 300
    python scripts/crop_geobox_pickle.py INPUT_PKL OUTPUT_PKL --window-px 300 --lon 10.0 --lat 50.0

Without --lon/--lat, the window is centered on the full geobox's own pixel
midpoint. For a VIIRS-derived global equirectangular grid that's (lon=0,
lat=0) -- the Gulf of Guinea -- fine for a globally-continuous field (e.g.
ACAG's PM2.5), but wrong for a source whose content is sparse/discrete in
space (e.g. GADM's country polygons): pass a real-content coordinate or the
window may land somewhere with nothing to rasterize/compare.
"""

import argparse
import pickle
import sys


def crop_geobox(full_geobox, window_px: int, lon: float = None, lat: float = None):
    """Return a window_px x window_px crop of full_geobox, clamped to its bounds."""
    ny, nx = full_geobox.shape
    if window_px > min(ny, nx):
        raise ValueError(f"window_px={window_px} exceeds full geobox shape {full_geobox.shape}")

    if lon is not None and lat is not None:
        col, row = ~full_geobox.affine * (lon, lat)
        cy, cx = int(round(row)), int(round(col))
        if not (0 <= cy < ny and 0 <= cx < nx):
            raise ValueError(
                f"(lon={lon}, lat={lat}) -> pixel ({cx}, {cy}) is outside the geobox ({nx}x{ny})"
            )
    else:
        cy, cx = ny // 2, nx // 2

    half = window_px // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Clamp into bounds, preserving window size where possible by shifting
    # rather than shrinking (only shrinks if window_px > the grid itself,
    # already rejected above).
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y1 > ny:
        y0 -= y1 - ny
        y1 = ny
    if x1 > nx:
        x0 -= x1 - nx
        x1 = nx
    y0, x0 = max(0, y0), max(0, x0)

    return full_geobox[y0:y1, x0:x1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_pkl")
    parser.add_argument("output_pkl")
    parser.add_argument("--window-px", type=int, default=300)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--lat", type=float, default=None)
    args = parser.parse_args()

    if (args.lon is None) != (args.lat is None):
        parser.error("--lon and --lat must be given together")

    with open(args.input_pkl, "rb") as f:
        full_geobox = pickle.load(f)

    window = crop_geobox(full_geobox, args.window_px, args.lon, args.lat)

    with open(args.output_pkl, "wb") as f:
        pickle.dump(window, f)

    center_desc = f", centered on lon={args.lon} lat={args.lat}" if args.lon is not None else ", centered"
    print(f"cropped target geobox: {full_geobox.shape} -> {window.shape} (window={args.window_px}px{center_desc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
