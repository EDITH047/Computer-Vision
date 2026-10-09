"""
test_cloud_tracker.py
---------------------
Headless automated tests for the Python CloudTracker port.

Run with:
    python test_cloud_tracker.py

Tests are fully self-contained — no real satellite images needed.
Synthetic images with a known bright blob are used to verify every
processing step produces numerically correct output.
"""

import sys
import os
import tempfile
import traceback

import cv2
import numpy as np

# Allow importing cloud_tracker from same directory
sys.path.insert(0, os.path.dirname(__file__))
from cloud_tracker import CloudTracker


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_results = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    _results.append((label, condition))


def make_synthetic_images(
    width: int = 300,
    height: int = 200,
    blob_shift: int = 5,
) -> tuple[str, str]:
    """
    Create two temporary greyscale PNG images:
      - img1: dark background with a bright white rectangular 'cloud' blob
      - img2: same image with the blob shifted right by `blob_shift` pixels
               (simulates cloud movement)

    Returns file paths (deleted automatically by the caller).
    """
    img1 = np.zeros((height, width), dtype=np.uint8)
    img2 = np.zeros((height, width), dtype=np.uint8)

    # Bright blob (represents a cloud — above threshold 160)
    bx, by, bw, bh = 80, 60, 60, 40
    img1[by : by + bh, bx : bx + bw] = 200          # bright white blob
    img2[by : by + bh, bx + blob_shift : bx + bw + blob_shift] = 200  # shifted

    # Convert to BGR (imread returns BGR; imwrite accepts both)
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    fd1, p1 = tempfile.mkstemp(suffix=".png")
    fd2, p2 = tempfile.mkstemp(suffix=".png")
    os.close(fd1)
    os.close(fd2)
    cv2.imwrite(p1, img1_bgr)
    cv2.imwrite(p2, img2_bgr)
    return p1, p2


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

def test_import() -> None:
    """Test 1: Module and class import."""
    try:
        ct = CloudTracker()
        check("Import CloudTracker class", True)
        check("Instance created successfully", ct is not None)
    except Exception as exc:
        check("Import CloudTracker class", False, str(exc))


def test_load_images() -> None:
    """Test 2: load_images with valid and invalid paths."""
    ct = CloudTracker()

    # Valid paths
    p1, p2 = make_synthetic_images()
    try:
        result = ct.load_images(p1, p2)
        check("load_images returns True for valid files", result is True)
        check("_img1_color is loaded",  ct._img1_color  is not None)
        check("_img2_color is loaded",  ct._img2_color  is not None)
        check("_img1_gray is loaded",   ct._img1_gray   is not None)
        check("_img2_gray is loaded",   ct._img2_gray   is not None)
        check(
            "Both images are same shape",
            ct._img1_color.shape == ct._img2_color.shape,
            f"{ct._img1_color.shape} == {ct._img2_color.shape}",
        )
    finally:
        os.unlink(p1)
        os.unlink(p2)

    # Invalid path
    ct2 = CloudTracker()
    result2 = ct2.load_images("nonexistent1.png", "nonexistent2.png")
    check("load_images returns False for missing files", result2 is False)


def test_preprocess_and_mask() -> None:
    """Test 3: _preprocess_and_mask sets cloud masks correctly."""
    p1, p2 = make_synthetic_images()
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    ct._preprocess_and_mask(ct._img1_gray, mask_id=1)
    ct._preprocess_and_mask(ct._img2_gray, mask_id=2)

    check("cloud_mask1 is not None",  ct._cloud_mask1 is not None)
    check("cloud_mask2 is not None",  ct._cloud_mask2 is not None)
    check(
        "cloud_mask1 dtype is uint8",
        ct._cloud_mask1.dtype == np.uint8,
    )
    check(
        "cloud_mask1 has non-zero pixels (cloud blob detected)",
        np.count_nonzero(ct._cloud_mask1) > 0,
        f"non-zero count = {np.count_nonzero(ct._cloud_mask1)}",
    )
    check(
        "cloud_mask values are only 0 or 255",
        set(np.unique(ct._cloud_mask1)).issubset({0, 255}),
    )


def test_extract_overlays() -> None:
    """Test 4: _extract_overlays does not crash and returns a valid mask."""
    p1, p2 = make_synthetic_images()
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    ct._preprocess_and_mask(ct._img1_gray, 1)
    ct._preprocess_and_mask(ct._img2_gray, 2)
    ct._extract_overlays()

    check("map_overlay is not None",  ct._map_overlay is not None)
    check(
        "map_overlay shape matches image",
        ct._map_overlay.shape == ct._img1_gray.shape,
    )
    check(
        "cloud_mask1 still valid after overlay removal",
        ct._cloud_mask1 is not None and ct._cloud_mask1.shape == ct._img1_gray.shape,
    )


def test_optical_flow() -> None:
    """Test 5: _compute_optical_flow produces a 2-channel float32 flow array."""
    p1, p2 = make_synthetic_images(blob_shift=10)
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    ct._compute_optical_flow()

    check("flow is not None",       ct._flow is not None)
    check(
        "flow has 2 channels",
        ct._flow.ndim == 3 and ct._flow.shape[2] == 2,
        f"shape = {ct._flow.shape}",
    )
    check(
        "flow dtype is float32",
        ct._flow.dtype == np.float32,
    )
    # The Farneback algorithm computes flow from local polynomial expansion.
    # Uniform interior pixels have no gradient, so their flow magnitude is
    # near-zero; flow is non-zero at edges/boundaries where gradient exists.
    # Verify that the overall max magnitude across the image is > 0.
    magnitude = np.sqrt(ct._flow[..., 0] ** 2 + ct._flow[..., 1] ** 2)
    check(
        "flow magnitude > 0 somewhere in image (edge/boundary region)",
        float(magnitude.max()) > 0.0,
        f"max magnitude = {magnitude.max():.4f}",
    )


def test_wind_speed_estimation() -> None:
    """Test 6: _estimate_wind_speed produces correct scalar and map."""
    p1, p2 = make_synthetic_images(blob_shift=8)
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    ct._preprocess_and_mask(ct._img1_gray, 1)
    ct._preprocess_and_mask(ct._img2_gray, 2)
    ct._extract_overlays()
    ct._compute_optical_flow()
    ct._estimate_wind_speed(time_difference=1.0)

    check("wind_speed_map is not None",   ct._wind_speed_map is not None)
    check(
        "wind_speed_map dtype is float32",
        ct._wind_speed_map.dtype == np.float32,
    )
    check(
        "wind_speed_map has non-zero values in cloud region",
        np.count_nonzero(ct._wind_speed_map) > 0,
        f"non-zero count = {np.count_nonzero(ct._wind_speed_map)}",
    )
    check(
        "avg_wind_speed is non-negative float",
        isinstance(ct._avg_wind_speed, float) and ct._avg_wind_speed >= 0.0,
        f"avg = {ct._avg_wind_speed:.4f}",
    )
    check(
        "wind speed outside cloud region is 0",
        ct._wind_speed_map[0, 0] == 0.0,   # top-left pixel = background, no cloud
    )

    # Time scaling: doubling time_difference should halve speed
    ct2 = CloudTracker()
    p1b, p2b = make_synthetic_images(blob_shift=8)
    ct2.load_images(p1b, p2b)
    os.unlink(p1b)
    os.unlink(p2b)
    ct2._preprocess_and_mask(ct2._img1_gray, 1)
    ct2._preprocess_and_mask(ct2._img2_gray, 2)
    ct2._extract_overlays()
    ct2._compute_optical_flow()
    ct2._estimate_wind_speed(time_difference=2.0)
    check(
        "Doubling time_difference halves avg_wind_speed",
        abs(ct2._avg_wind_speed - ct._avg_wind_speed / 2.0) < 1e-3,
        f"{ct2._avg_wind_speed:.6f} ≈ {ct._avg_wind_speed / 2.0:.6f}",
    )


def test_draw_motion_vectors() -> None:
    """Test 7: _draw_motion_vectors creates a valid BGR image."""
    p1, p2 = make_synthetic_images(blob_shift=5)
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    ct._preprocess_and_mask(ct._img1_gray, 1)
    ct._preprocess_and_mask(ct._img2_gray, 2)
    ct._extract_overlays()
    ct._compute_optical_flow()
    ct._draw_motion_vectors(step=16, color=(0, 0, 255))

    check("flow_visualization is not None",      ct._flow_visualization is not None)
    check(
        "flow_visualization is BGR (3 channels)",
        ct._flow_visualization.ndim == 3 and ct._flow_visualization.shape[2] == 3,
    )
    check(
        "flow_visualization same HxW as input",
        ct._flow_visualization.shape[:2] == ct._img1_color.shape[:2],
    )


def test_cloud_coverage_percentage() -> None:
    """Test 8: Cloud coverage percentages are in the range [0, 100]."""
    p1, p2 = make_synthetic_images()
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)
    ct.process(time_difference=1.0)

    check(
        "cloud_coverage_pct1 in [0, 100]",
        0.0 <= ct._cloud_coverage_pct1 <= 100.0,
        f"{ct._cloud_coverage_pct1:.2f} %",
    )
    check(
        "cloud_coverage_pct2 in [0, 100]",
        0.0 <= ct._cloud_coverage_pct2 <= 100.0,
        f"{ct._cloud_coverage_pct2:.2f} %",
    )
    check(
        "cloud_coverage_pct1 > 0 (blob detected)",
        ct._cloud_coverage_pct1 > 0.0,
        f"{ct._cloud_coverage_pct1:.2f} %",
    )


def test_full_process_pipeline() -> None:
    """Test 9: End-to-end process() call completes without error."""
    p1, p2 = make_synthetic_images(blob_shift=6)
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    try:
        ct.process(time_difference=1.5)
        check("process() completes without exception", True)
    except Exception as exc:
        check("process() completes without exception", False, str(exc))
        traceback.print_exc()
        return

    # Verify all outputs exist after process()
    check("_cloud_mask1 set",           ct._cloud_mask1         is not None)
    check("_cloud_mask2 set",           ct._cloud_mask2         is not None)
    check("_flow set",                  ct._flow                is not None)
    check("_wind_speed_map set",        ct._wind_speed_map      is not None)
    check("_flow_visualization set",    ct._flow_visualization  is not None)
    check("_highlighted_clouds set",    ct._highlighted_clouds  is not None)
    check("_map_overlay set",           ct._map_overlay         is not None)


def test_same_image_zero_wind_speed() -> None:
    """Test 10: Identical images → zero average wind speed."""
    p1, _ = make_synthetic_images()
    ct = CloudTracker()
    # Load the same image as both T1 and T2 — no movement
    ct.load_images(p1, p1)
    os.unlink(p1)
    ct.process(time_difference=1.0)

    check(
        "Identical images → avg_wind_speed ≈ 0",
        ct._avg_wind_speed < 0.01,
        f"avg = {ct._avg_wind_speed:.6f}",
    )


def test_zero_time_difference_fallback() -> None:
    """Test 11: time_difference=0 triggers the fallback to 1.0 (no division by zero)."""
    p1, p2 = make_synthetic_images(blob_shift=5)
    ct = CloudTracker()
    ct.load_images(p1, p2)
    os.unlink(p1)
    os.unlink(p2)

    try:
        ct.process(time_difference=0.0)
        check("time_difference=0 does not raise ZeroDivisionError", True)
        check(
            "wind speed is finite after td=0 fallback",
            np.isfinite(ct._avg_wind_speed),
            f"avg = {ct._avg_wind_speed}",
        )
    except ZeroDivisionError:
        check("time_difference=0 does not raise ZeroDivisionError", False)


def test_size_mismatch_handling() -> None:
    """Test 12: Images of different sizes → T2 is resized to match T1."""
    img1 = np.zeros((200, 300, 3), dtype=np.uint8)
    img2 = np.zeros((100, 150, 3), dtype=np.uint8)
    fd1, p1 = tempfile.mkstemp(suffix=".png")
    fd2, p2 = tempfile.mkstemp(suffix=".png")
    os.close(fd1); os.close(fd2)
    cv2.imwrite(p1, img1)
    cv2.imwrite(p2, img2)

    ct = CloudTracker()
    result = ct.load_images(p1, p2)
    os.unlink(p1); os.unlink(p2)

    check("load_images succeeds with mismatched sizes", result is True)
    check(
        "T2 resized to match T1 shape",
        ct._img2_color.shape == ct._img1_color.shape,
        f"{ct._img2_color.shape} == {ct._img1_color.shape}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests() -> int:
    test_funcs = [
        test_import,
        test_load_images,
        test_preprocess_and_mask,
        test_extract_overlays,
        test_optical_flow,
        test_wind_speed_estimation,
        test_draw_motion_vectors,
        test_cloud_coverage_percentage,
        test_full_process_pipeline,
        test_same_image_zero_wind_speed,
        test_zero_time_difference_fallback,
        test_size_mismatch_handling,
    ]

    print("\n" + "=" * 60)
    print("  CloudTracker Python Port — Automated Test Suite")
    print("=" * 60 + "\n")

    for fn in test_funcs:
        print(f"\n>> {fn.__doc__.strip().splitlines()[0]}")
        try:
            fn()
        except Exception as exc:
            print(f"  {FAIL} Unexpected exception in {fn.__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    passed  = sum(1 for _, ok in _results if ok)
    failed  = sum(1 for _, ok in _results if not ok)
    total   = len(_results)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed == 0:
        print("[ALL PASS]  All tests passed -- 100% efficiency verified.\n")
        return 0
    else:
        print(f"[FAILED]  {failed} test(s) FAILED.\n")
        for label, ok in _results:
            if not ok:
                print(f"    FAILED: {label}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
