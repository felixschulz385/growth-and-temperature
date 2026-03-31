"""Extract claim geometries from Property Profile map tiles."""

from __future__ import annotations

import io
import re
from typing import Any
from urllib.parse import urljoin

from time import sleep
import cv2
import numpy as np
import requests
from PIL import Image as PILImage
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from ..config import (
    MAP_LOAD_WAIT_SECONDS,
)

def make_hsv_range(h: int, s: int, v: int, h_tol: int = 12, s_tol: int = 60, v_tol: int = 60):
    lo = np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)])
    hi = np.array([min(180, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])
    return [(lo, hi)]


COLOR_RANGES = {
    "property_fill": make_hsv_range(107, 189, 213, h_tol=10, s_tol=60, v_tol=60),
    "property_border": [(np.array([0, 0, 0]), np.array([180, 255, 30]))],
    "linked": make_hsv_range(140, 126, 205, h_tol=15, s_tol=60, v_tol=60),
    "all_claims": make_hsv_range(20, 126, 234, h_tol=12, s_tol=60, v_tol=60),
}

COMPOSITE_TARGETS = {
    "property": COLOR_RANGES["property_fill"],
    "linked": COLOR_RANGES["linked"],
}


def build_color_mask(img_rgba, target: str = "property") -> np.ndarray:
    if isinstance(img_rgba, PILImage.Image):
        img_rgba = np.array(img_rgba.convert("RGBA"))

    alpha = img_rgba[:, :, 3]
    bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    ranges = COMPOSITE_TARGETS.get(target)
    if ranges is None:
        raise ValueError(f"Unknown target '{target}'. Choose from: {list(COMPOSITE_TARGETS)}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)

    mask[alpha == 0] = 0
    return mask


def get_map_element_id(driver: WebDriver) -> str:
    map_id = driver.execute_script(
        """
        const candidates = [...document.querySelectorAll('div.leaflet-container')]
            .filter(el => el.offsetParent !== null);
        if (candidates.length === 0) return null;
        const withId = candidates.find(el => el.id && el.id.trim().length > 0);
        if (withId) return withId.id;
        candidates[0].id = 'snf_dynamic_leaflet_map';
        return candidates[0].id;
        """
    )
    if not map_id:
        raise ValueError("No visible leaflet map found on Property Profile page.")
    return str(map_id)


def _build_authenticated_session(driver: WebDriver) -> requests.Session:
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path"),
        )

    session.headers.update(
        {
            "User-Agent": driver.execute_script("return navigator.userAgent;") or "Mozilla/5.0",
            "Referer": driver.current_url,
        }
    )
    return session


def fetch_tile(driver: WebDriver, src: str, session: requests.Session | None = None) -> np.ndarray:
    resolved_src = urljoin(driver.current_url, src)
    active_session = session or _build_authenticated_session(driver)
    response = active_session.get(resolved_src, timeout=30)
    response.raise_for_status()
    image = PILImage.open(io.BytesIO(response.content)).convert("RGBA")
    return np.array(image)


def get_wms_tiles(driver: WebDriver, components: str = "1") -> list[dict[str, Any]]:
    _wait_for_map_tiles(driver)

    map_id = get_map_element_id(driver)
    tile_imgs = driver.find_elements(
        By.CSS_SELECTOR,
        f"#{map_id} .leaflet-overlay-pane img.leaflet-image-layer",
    )

    tiles: list[dict[str, Any]] = []
    for img in tile_imgs:
        src = img.get_attribute("src") or ""
        style = img.get_attribute("style") or ""

        if "SNL.Services.GIS.Service/v1/WMS/get" not in src:
            continue
        if f"components={components}" not in src:
            continue
        if "display: none" in style:
            continue
        if "visibility: hidden" in style:
            continue
        if "bb_mininglocation" not in src:
            continue

        bbox_m = re.search(r"bbox=([-\d.,]+)", src)
        w_m = re.search(r"width=(\d+)", src)
        h_m = re.search(r"height=(\d+)", src)

        if not (bbox_m and w_m and h_m):
            continue

        tiles.append(
            {
                "src": src,
                "bbox": list(map(float, bbox_m.group(1).split(","))),
                "width": int(w_m.group(1)),
                "height": int(h_m.group(1)),
            }
        )

    return tiles


def _wait_for_map_tiles(driver: WebDriver) -> None:
    """Wait for the Leaflet map and overlay tiles to become visible and settle."""
    stable_samples = 0
    last_state: tuple[int, int] | None = None

    def _tiles_ready(current_driver: WebDriver) -> bool:
        nonlocal stable_samples, last_state
        base_tiles = len(current_driver.find_elements(By.CSS_SELECTOR, ".leaflet-tile-loaded"))
        overlay_tiles = len(
            current_driver.find_elements(
                By.CSS_SELECTOR,
                ".leaflet-overlay-pane img.leaflet-image-layer:not([style*='visibility: hidden']):not([style*='display: none'])",
            )
        )
        current_state = (base_tiles, overlay_tiles)
        if overlay_tiles == 0:
            stable_samples = 0
            last_state = current_state
            return False
        if current_state == last_state:
            stable_samples += 1
        else:
            stable_samples = 1
            last_state = current_state
        return stable_samples >= 3

    WebDriverWait(driver, MAP_LOAD_WAIT_SECONDS, poll_frequency=1.0).until(_tiles_ready)
    sleep(min(5, MAP_LOAD_WAIT_SECONDS / 6))


def tiles_to_polygons(driver: WebDriver, target: str = "property", components: str = "1"):
    tiles = get_wms_tiles(driver, components=components)
    if not tiles:
        raise ValueError("No WMS tiles found.")

    session = _build_authenticated_session(driver)
    all_polys_3857 = []
    for tile in tiles:
        arr = fetch_tile(driver, tile["src"], session=session)
        minx, miny, maxx, maxy = tile["bbox"]
        w, h = tile["width"], tile["height"]

        if arr[:, :, 3].max() == 0:
            continue

        mask = build_color_mask(arr, target=target)
        if mask.max() == 0:
            continue

        transform = from_bounds(minx, miny, maxx, maxy, w, h)
        for geom, val in shapes(mask, mask=(mask > 0), transform=transform):
            if val > 0:
                all_polys_3857.append(shape(geom))

    return all_polys_3857


def extract_claim_geometry_from_tiles(driver: WebDriver, target: str = "property", components: str = "1"):
    polys_3857 = tiles_to_polygons(driver, target=target, components=components)
    if not polys_3857:
        raise ValueError(f"No polygons found for target='{target}'.")

    unioned_3857 = unary_union(polys_3857)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return shp_transform(transformer.transform, unioned_3857)


def extract_property_and_linked_geometries(driver: WebDriver, components: str = "1") -> dict[str, Any]:
    return {
        "linked": extract_claim_geometry_from_tiles(driver, target="linked", components=components),
        "property": extract_claim_geometry_from_tiles(driver, target="property", components=components),
    }
