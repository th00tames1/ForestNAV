"""
Utilities for downloading OpenStreetMap raster tiles over HTTP.

This module is a copy of the top‑level ``tile_downloader.py`` so that
it is available within the ``ForestNAV_src`` package when the
integrated ForestNAV software is redistributed.  See the root
``tile_downloader.py`` for detailed documentation.
"""

import os
import math
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Iterable, Callable, Optional, Tuple, List

USER_AGENT = "ForestNAV/1.0 (you@example.com)"

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Converts geographic coordinates to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )
    return xtile, ytile

def download_tiles_multi_zoom(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom_levels: Iterable[int],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Downloads map tiles for multiple zoom levels for a given geographic bounding box.

    Args:
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        zoom_levels: An iterable of integer zoom levels to download.
        progress_callback: An optional function to call with progress updates.
                           It receives the current index and the total number of tiles.
        cancel_callback: An optional function that returns True to cancel the download.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    tiles_root = os.path.join(base_dir, 'tiles')
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": USER_AGENT})

    coords: List[Tuple[int, int, int]] = []
    for z in zoom_levels:
        x_min, y_max = deg2num(lat_min, lon_min, z)
        x_max, y_min = deg2num(lat_max, lon_max, z)
        for x in range(min(x_min, x_max), max(x_min, x_max) + 1):
            for y in range(min(y_min, y_max), max(y_min, y_max) + 1):
                coords.append((z, x, y))

    total = len(coords)
    for idx, (z, x, y) in enumerate(coords, start=1):
        # Check if the download should be cancelled
        if cancel_callback and cancel_callback():
            print("Tile download cancelled.")
            break

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                # Ignore exceptions from the callback to not interrupt downloads
                pass

        out_dir = os.path.join(tiles_root, str(z), str(x))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{y}.png")

        if os.path.exists(out_path):
            continue

        url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Skipping tile {z}/{x}/{y} → {e}")
            continue

        with open(out_path, 'wb') as f:
            f.write(resp.content)

        # A small delay to respect the tile server's usage policy.
        time.sleep(0.05)
