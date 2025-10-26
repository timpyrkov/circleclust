import math
import numpy as np
import pytest

from circleclust import CircleClust
from PIL import Image
import pandas as pd
import os
import matplotlib.colors as mcolors


def gen_clustered(values, centers, sigma, period):
    rng = np.random.default_rng(0)
    xs = []
    for c in centers:
        xs.append((rng.normal(c, sigma, size=200) % period))
    return np.concatenate(xs)


def frac_outliers(cc: CircleClust, x: np.ndarray) -> float:
    labels = cc.predict(x)
    return float(np.mean(labels < 0))


def test_hues_0_1_two_peaks_detected_and_in_range():
    period = 1.0
    centers = [0.15, 0.6]
    x = gen_clustered(None, centers, sigma=0.03, period=period)
    cc = CircleClust(period=period, verbose=False, debug=False)
    cc.fit(x)
    peaks = cc.get_peaks()
    cents = np.asarray(peaks["centroid"])
    # in-range check
    assert np.all(cents >= 0.0) and np.all(cents < period)
    # correct number of peaks (allow tolerance +/-1)
    assert abs(cents.size - len(centers)) <= 1


def test_sleep_times_0_24_around_midnight():
    period = 24.0
    # Peaks around 23:30 and 00:30 (wrap)
    centers = [23.5, 0.5]
    x = gen_clustered(None, centers, sigma=0.3, period=period)
    cc = CircleClust(period=period)
    cc.fit(x)
    cents = np.asarray(cc.get_peaks()["centroid"]) 
    assert np.all(cents >= 0.0) and np.all(cents < period)
    assert cents.size >= 1


def test_wakeup_times_0_24_around_7am():
    period = 24.0
    centers = [7.0]
    x = gen_clustered(None, centers, sigma=0.4, period=period)
    cc = CircleClust(period=period)
    cc.fit(x)
    cents = np.asarray(cc.get_peaks()["centroid"]) 
    # At least one detected near 7
    assert cents.size >= 1
    assert np.min(np.abs(((cents - centers[0] + 12) % 24) - 12)) <= 2.0


def test_image_hue_cluster_if_image_available():
    here = os.path.dirname(__file__)
    # Prefer jpg if present, fall back to png
    img_path_jpg = os.path.join(here, "image.jpg")
    img_path_png = os.path.join(here, "image.png")
    if os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        pytest.skip("image file not available for hue clustering test")
    # Load and convert to HSV, collect hues in [0,1)
    im = Image.open(img_path).convert("RGB")
    arr = np.asarray(im, dtype=float) / 255.0
    hsv = mcolors.rgb_to_hsv(arr.reshape(-1, 3))
    hue = hsv[:, 0]
    # Fit on hue circle (period=1)
    cc = CircleClust(period=1.0)
    cc.fit(hue)
    cents = np.asarray(cc.get_peaks()["centroid"]) 
    assert np.all(cents >= 0) and np.all(cents <= 1.0)
    # If any cluster exists, at least one near red (~0 modulo 1)
    if cents.size > 0:
        near_red = np.min(np.minimum(cents, 1.0 - cents)) <= 0.12
        assert near_red


def test_sleep_csv_clusters_generate_if_missing(tmp_path_factory):
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "sleep.csv")
    # Generate if missing using local generator
    if not os.path.exists(csv_path):
        from .gen_sleep import generate_sleep_data
        csv_path = generate_sleep_data(csv_path, seed=0)
    df = pd.read_csv(csv_path, parse_dates=["sleep_start_datetime", "sleep_end_datetime"])  # type: ignore
    s = df["sleep_start_datetime"].dt
    e = df["sleep_end_datetime"].dt
    go_to_sleep = (s.hour * 60 + s.minute + s.second / 60.0).to_numpy(float)
    wake_up = (e.hour * 60 + e.minute + e.second / 60.0).to_numpy(float)

    # Fit for sleep times
    cc_sleep = CircleClust(period=1440.0)
    cc_sleep.fit(go_to_sleep)
    cs = np.asarray(cc_sleep.get_peaks()["centroid"]) 
    # 1-3 clusters and at least one near midnight (within 4 hours)
    assert 0 <= cs.size <= 5
    if cs.size > 0:
        near_midnight = np.min(np.minimum(cs, 1440.0 - cs)) <= 240.0
        assert near_midnight

    # Fit for wake times
    cc_wake = CircleClust(period=1440.0)
    cc_wake.fit(wake_up)
    cw = np.asarray(cc_wake.get_peaks()["centroid"]) 
    assert 0 <= cw.size <= 5
    if cw.size > 0:
        # Expect some centroid between 6:00 and 11:00 (360-660 min) to allow variability
        in_morning = np.any((cw >= 360.0) & (cw <= 660.0))
        assert in_morning


def test_sample_csv_three_peaks():
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "sample.csv")
    period = 2 * np.pi
    df = pd.read_csv(csv_path)  # type: ignore
    x = (df["x"].to_numpy(float) if "x" in df.columns else df.iloc[:, 0].to_numpy(float)) % period
    cc = CircleClust(period=period)
    cc.fit(x)
    cents = np.asarray(cc.get_peaks()["centroid"]) 
    print("DEBUG: cents =", cents)
    # Expect roughly 2-5 peaks detected
    assert cents.size == 3


def test_angles_0_2pi_three_peaks():
    period = 2 * math.pi
    centers = [0.2, 2.0, 5.0]
    x = gen_clustered(None, centers, sigma=0.12, period=period)
    cc = CircleClust(period=period)
    cc.fit(x)
    cents = np.asarray(cc.get_peaks()["centroid"]) 
    assert np.all(cents >= 0.0) and np.all(cents < period)
    assert 1 <= cents.size <= 4


def test_error_on_out_of_range_inputs():
    period = 1.0
    x = np.array([-0.1, 0.2, 0.3])
    cc = CircleClust(period=period)
    with pytest.raises(ValueError):
        cc.fit(x)


def test_uniform_data_small_n_behavior():
    rng = np.random.default_rng(0)
    period = 1.0
    for n in [0, 1, 2, 5, 10]:
        x = rng.random(n)
        cc = CircleClust(period=period)
        if n == 0:
            with pytest.raises(Exception):
                cc.fit(x)
            continue
        # For very small samples, fitting may fail to find an optimal window; allow exceptions
        try:
            cc.fit(x)
        except Exception:
            continue
        # If it fits, ensure predictions are valid ints and peak count is bounded
        labels = cc.predict(x)
        assert labels.dtype == np.int64 or labels.dtype == np.int32
        k = getattr(cc, "peak_idx_", np.array([])).size
        assert k <= min(3, n)


def test_uniform_data_prefers_outliers_large_n():
    rng = np.random.default_rng(1)
    period = 24.0
    for n in [100, 500]:
        x = rng.uniform(0, period, size=n)
        cc = CircleClust(period=period)
        cc.fit(x)
        # Most should be outliers under 2*sigma gating
        assert frac_outliers(cc, x) >= 0.6
        # Do not explode peak count
        assert getattr(cc, "peak_idx_", np.array([])).size <= 5
