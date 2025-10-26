import numpy as np
from kneed import KneeLocator
from typing import Iterable, Optional
from scipy.optimize import curve_fit as _curve_fit
from .utils import *


class CircleClust:
    """
    Circular clustering with automatic detection of centroids as distribution peaks.

    This class clusters circular data (e.g., times of day, hues) by:
    - Binning data on a circle with periodic boundaries.
    - Applying a fixed Hann smoothing window of 9 bins with circular padding.
    - Screening bin counts via k = 1..max_screen_divisor where B = 9*k and
      window width (in radians of the unit circle) is 2π/k.
    - Selecting the binning that minimizes test RMS between histogram and
      smoothed envelope, averaged over `max_screen_iter` random splits.
    - Detecting circular local maxima on the smoothed histogram as cluster centers.

    Parameters
    ----------
    data : Iterable[float] | None
        Optional data to fit immediately upon construction.
    period : float, default 2π
        Period of the input values; inputs are wrapped into [0, range).
        Important to provide valid data range, for example:
        - 1 if your data period is 1 (e.g. color hues in HLS or HSV space)
        - 24 if your data is measured in hours (e.g. times of day)
        - 360 if your data is measured in degrees
        - 2π if your data is measured in radians (e.g. angles)
    window : float | None
        Manual override for smoothing window width (in radians on unit circle).
        If provided, the screener will be skipped and nearest k≈2π/window used.
    max_screen_divisor : int, default 32
        Maximum divisor k in screening; B = 9*k.
    max_screen_iter : int, default 2
        Number of screening repetitions with different train/test splits; RMS is
        averaged across repetitions before selecting k.
    train_frac : float, default 0.7
        Fraction of samples for training during screening.
    random_seed : int, default 0
        Base random seed for reproducible splits.
    verbose : bool, default False
        Enable informational prints via `verbose_print`.
    """
    def __init__(
        self,
        data: Optional[Iterable[float]] = None,
        period: float = None,
        window: float | None = None,
        max_screen_divisor: int = 32,
        max_screen_iter: int = 2,
        train_frac: float = 0.7,
        random_seed: int = 0,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.period = period 
        self.train_frac = float(train_frac)
        self.random_seed = int(random_seed)
        self.max_screen_divisor = int(max_screen_divisor)
        self.max_screen_iter = int(max_screen_iter)
        self.window = window
        self.window_optimal = None
        self.debug = bool(debug)
        # If debug is on, force verbose on for richer context
        self.verbose = True if self.debug else bool(verbose)
        self.debug_dir = 'debug'

        self.bins_per_window_ = 9
        self.peak_idx_ = None
        self.peak_sigma_ = None
        self.centroid_ = None
        self.centroid_radius_ = None
        self.h_all_ = None
        self.s_all_ = None

        if data is not None:
            self.fit(data, period)


    def set_window(self, window: float):
        """
        Sets manually specified smoothing window.

        Parameters
        ----------
        window : float
            Smoothing window width (in radians on unit circle).
        """
        w = float(window)
        if w <= 0:
            raise ValueError("Window must be positive.")
        if self.period is not None and w >= self.period:
            raise ValueError("Window must be less than period.")
        self.window = w


    def _set_period(self, period: float | None = None):
        """
        Sets period for the input values.

        Parameters
        ----------
        period : float | None, default None
            Period of the input values. If None, uses default 2π.
        """
        if period is None or period <= 0:
            if self.period is None:
                warning_print("Valid period not provided, using default 2π.")
                self.period = 2 * np.pi
        else:
            self.period = period


    def _screen_smoothing_windows_rmsd(self, x_train: np.ndarray, x_test: np.ndarray, iteration: int):
        """
        Screens bin counts for fixed smoothing window and computes RMS.

        For each k in 1..max_screen_divisor, compute B=9*k, histogram and smooth
        on train, and compute RMS(train) and RMS(test) against the (scaled) smooth
        envelope. Returns arrays aligned to k values.

        Parameters
        ----------
        x_train : np.ndarray
            Wrapped train split in [0, range).
        x_test : np.ndarray
            Wrapped test split in [0, range).
        iteration : int
            Iteration number (1..max_screen_iter).

        Returns
        -------
        (rmsd_tr, rmsd_ts) : Tuple[np.ndarray, np.ndarray]
            RMS arrays per each screened window.
        """
        # Initialize array of N windows per period
        nwindows = np.arange(1, self.max_screen_divisor + 1)

        # Initialize arrays for RMS
        rmsd_tr = np.zeros_like(nwindows, dtype=float)
        rmsd_ts = np.zeros_like(nwindows, dtype=float)

        for i, nwindow in enumerate(nwindows):
            # Compute histogram bin edges
            nbins = int(nwindow) * self.bins_per_window_
            edges = np.linspace(0.0, self.period, nbins + 1, endpoint=True)

            # Compute histogram and smooth on train
            h_train, _ = np.histogram(x_train, bins=edges)
            s_train = _periodic_smooth(h_train, self.bins_per_window_)

            # Compute histogram and smooth on test
            h_test, _ = np.histogram(x_test, bins=edges)
            scale = (h_test.sum() / max(1, h_train.sum())) if h_train.sum() else 1.0
            s_test = s_train * scale

            # Compute masked RMSD
            mask =( h_train > 0) | (h_test > 0)
            tr_rmsd = _rmsd(h_train, s_train, mask=mask)
            ts_rmsd = _rmsd(h_test, s_test, mask=mask)

            rmsd_tr[i] = tr_rmsd
            rmsd_ts[i] = ts_rmsd

            if self.debug:
                debug_plot_screen_iteration(h_train, s_train, h_test, s_test, iteration, i, outdir=self.debug_dir)

        return rmsd_tr, rmsd_ts


    def _find_optimal_window(self, data: Iterable[float], period: float):
        """
        Finds optimal smoothing window by screening.

        Parameters
        ----------
        data : Iterable[float]
            Input data to fit.
        period : float
            Period of the input values; inputs are wrapped into [0, period).
            Important to provide valid data range, for example:
            - 1 if your data period is 1 (e.g. color hues in HLS or HSV space)
            - 24 if your data is measured in hours (e.g. times of day)
            - 360 if your data is measured in degrees
            - 2π if your data is measured in radians (e.g. angles)
        
        Returns
        -------
        float
            Optimal smoothing window width in radians on the unit circle.
        """
        # Check if there is enough data
        x = np.asarray(list(data), dtype=float)
        if x.size <= 1:
            warning_print("Not enough data points to fit optimal window.")
            return self

        # Initialize random number generator
        rng = np.random.default_rng(self.random_seed)
        
        # Initialize number of train and test samples
        n_train = max(1, int(self.train_frac * x.size))
        n_train = min(n_train, x.size - 1)
        n_test = x.size - n_train

        # Check if there is enough data
        if n_train < 2 or n_test < 2:
            warning_print("Not enough data points to fit optimal window.")
            return self
        
        # Initialize accumulator of train and test RMS
        n_screened_windows = self.max_screen_divisor
        nwindows = np.arange(1, n_screened_windows + 1)
        rmsd_tr_acc = np.zeros(n_screened_windows, dtype=float)
        rmsd_ts_acc = np.zeros(n_screened_windows, dtype=float)

        for i in range(self.max_screen_iter):
            # Shuffle data and split into train and test
            idx = np.arange(x.size)
            rng.shuffle(idx)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            x_train = x[train_idx]
            x_test = x[test_idx]

            rmsd_tr, rmsd_ts = self._screen_smoothing_windows_rmsd(x_train, x_test, i)
            rmsd_tr_acc += rmsd_tr
            rmsd_ts_acc += rmsd_ts

        # Normalize RMS across iterations
        rmsd_tr_mean = rmsd_tr_acc / max(1, self.max_screen_iter)
        rmsd_ts_mean = rmsd_ts_acc / max(1, self.max_screen_iter)

        # Find optimal window at minimum of the test RMSD
        i0 = int(np.argmin(rmsd_ts_mean))
        confidence = 1.0
        if i0 <= 1 or i0 >= n_screened_windows - 2:
            confidence = 0.0
        else:
            r = np.concatenate([rmsd_ts_mean[i0-2:i0], rmsd_ts_mean[i0+1:i0+3]])
            if rmsd_ts_mean[i0] >= 0.5 * np.mean(r):
                confidence = 0.0

        # Try knee (elbow) on decreasing convex curve of test RMS vs k (nwindows)
        if confidence < 0.5:
            try:
                knee = KneeLocator(nwindows, rmsd_ts_mean, curve='convex', direction='decreasing')
                if knee.knee is not None:
                    # map knee x-value (k) to nearest index
                    i0 = int(np.argmin(np.abs(nwindows - float(knee.knee))))
            except Exception:
                pass

        # Avoid edges if possible (clamp to interior) to reduce over/under-smoothing
        if n_screened_windows >= 3:
            if i0 == 0:
                i0 = 1
            elif i0 == n_screened_windows - 1:
                i0 = n_screened_windows - 2

        self.window_optimal = self.period / nwindows[i0]
        if i0 == 0 or i0 == (n_screened_windows - 1):
            warning_print("Optimal smoothing window at edge of search range; screening may not have converged.")

        if self.verbose:
            print()
            print("Selected optimal window via knee (fallback: min) on test RMS:")
            print("idx | window | tr_rmsd | ts_rmsd")
            for i in range(n_screened_windows):
                label = "*" if i == i0 else ""
                print(f"{int(nwindows[i]):3d} | {self.period / nwindows[i]:.4f} | {rmsd_tr_mean[i]:.4f} | {rmsd_ts_mean[i]:.4f} {label}")
            print()
            
        if self.debug:
            debug_plot_window_selection(rmsd_tr_mean, rmsd_ts_mean, i0=i0, outdir=self.debug_dir)

        return self.window_optimal


    def fit(self, data: Iterable[float], period: float | None = None):
        """
        Fits the model by selecting smoothing binning and finding centers.

        Parameters
        ----------
        data : Iterable[float]
            Input data to fit.
        period : float, default 2π
            Period of the input values; inputs are wrapped into [0, period).
            Important to provide valid data range, for example:
            - 1 if your data period is 1 (e.g. color hues in HLS or HSV space)
            - 24 if your data is measured in hours (e.g. times of day)
            - 360 if your data is measured in degrees
            - 2π if your data is measured in radians (e.g. angles)
        
        Returns
        -------
        self : CircleClust
            The fitted instance with detected peak_idx, peak_sigma, centroid and centroid_radius arrays.
        """
        # Set period
        self._set_period(period)
        
        # Check if data are within the period range
        x = np.asarray(list(data), dtype=float)
        if np.any(x < 0) or np.any(x >= self.period):
            raise ValueError(f"Data must be within [0, {self.period}) or provide a valid data period.")
        
        # Set window
        window = None
        if self.window is not None:
            # Use manually set window if provided
            window = float(self.window)
        else:
            # Automatically detect optimal window
            self._find_optimal_window(data, period)
            window = self.window_optimal
        if window is None:
            raise ValueError("Failed to detect optimal window.")

        # Calculate histogram and binning
        nwindows = max(1, int(np.round(self.period / window)))
        nbins = 9 * nwindows
        edges = np.linspace(0.0, self.period, nbins + 1, endpoint=True)
        self.h_all_, _ = np.histogram(x, bins=edges)
        self.s_all_ = _periodic_smooth(self.h_all_, self.bins_per_window_)

        # Find and fill peak index and centroid coordinate arrays
        self.peak_idx_ = _find_idxs_of_periodic_peaks(self.s_all_, self.bins_per_window_)
        self.centroid_ = self.peak_idx_ * self.period / nbins

        # Move to determine peak sigma and centroid radius

        # Compute periodic midpoints between adjacent peaks to define disjoint segments
        nbins = len(self.h_all_)
        self.peak_midpoints_ = _find_idxs_of_midpoints_between_peaks(self.peak_idx_, nbins)

        # Fit Gaussian per peak to estimate sigma/width using only its segment
        means: list[float] = []
        centroids: list[float] = []
        sigmas: list[float] = []
        amplitudes: list[float] = []
        baselines: list[float] = []
        for k, peak_id in enumerate(self.peak_idx_):
            left_b = int(self.peak_midpoints_[k - 1])
            right_b = int(self.peak_midpoints_[k])
            # Build segment values from smoothed signal and compute center index within the segment
            curve_segment = _slice_periodic_segment(self.s_all_, left_b, right_b)
            sigma = 0.0
            mean = np.mod(peak_id - left_b, nbins)
            centroid = self.centroid_[k]
            sigma = len(curve_segment) / 6
            baseline = np.mean(curve_segment)
            amplitude = np.max(curve_segment) - baseline
            if len(curve_segment) >= 2:
                popt = fit_gaussian_shifted(curve_segment)
                popt_ = fit_gaussian_shifted(curve_segment, mu=mean)
                if popt is not None and abs(popt[1] - mean) <= 1:
                    amplitude, mean, sigma, baseline = popt
                elif popt_ is not None:
                    amplitude, mean, sigma, baseline = popt_
            mean = np.mod(mean + left_b, nbins)
            centroid = mean * self.period / nbins
            means.append(int(mean))
            centroids.append(float(centroid))
            sigma = min(sigma, len(curve_segment) / 4)
            sigmas.append(float(sigma))
            amplitudes.append(float(amplitude))
            baselines.append(float(baseline))

        # Filter out peaks with too small amplitude (likely coming from noise)
        means = np.asarray(means, dtype=int)
        centroids = np.asarray(centroids, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)
        baselines = np.asarray(baselines, dtype=float)
        mask = amplitudes > baselines
        self.peak_idx_ = means[mask]
        self.centroid_ = centroids[mask]
        self.peak_sigma_ = sigmas[mask]
        self.centroid_radius_ = 2.0 * self.peak_sigma_ * (self.period / nbins)

        if self.verbose:
            print()
            print("Identified peaks:")
            print("idx | centroid | radius")
            for i in range(len(self.centroid_)):
                print(f"{i:3d} | {self.centroid_[i]:.4f} | {self.centroid_radius_[i]:.4f}")
            print()

        return self


    def predict(self, x: Iterable[float]):
        """
        Predicts cluster labels by nearest circular distance to centers.

        Parameters
        ----------
        x : Iterable[float]
            Input values (wrapped to [0, range)). Units must match `range` used
            during fitting (e.g., radians, minutes, or fraction of circle).

        Returns
        -------
        np.ndarray
            Integer labels in [0, n_centers-1], or -1 for outliers.
        """
        # Check if model is fitted
        if self.centroid_ is None or self.centroid_radius_ is None:
            raise RuntimeError("CircleClust is not fitted.")
        centroid_array = np.asarray(self.centroid_, dtype=float)
        radius_array = np.asarray(self.centroid_radius_, dtype=float)
        # Initialize all points as outliers and return if no centroids
        labels = np.full(x.shape[0], -1, dtype=int)
        if centroid_array.size == 0:
            return labels
        # For each centroid, assign all points within 2*sigma to that centroid id
        x_ = np.asarray(x, dtype=float)
        for i, centroid in enumerate(centroid_array):
            diff = np.abs(x_ - centroid)
            diff = np.minimum(diff, self.period - diff)
            mask = diff <= radius_array[i]
            labels[mask] = i
        return labels


    def get_peaks(self, as_list: bool = False):
        """
        Returns detected peaks.

        Parameters
        ----------
        as_list : bool, default False
            If True, returns a list of dicts [{'centroid': c, 'radius': r}, ...].
            If False, returns a dict with numpy arrays {'centroid': array, 'radius': array}.

        Returns
        -------
        dict | list
            Peaks information as a dict of arrays or a list of per-peak dicts.
        """
        if getattr(self, "centroid_", None) is None or getattr(self, "centroid_radius_", None) is None:
            raise RuntimeError("CircleClust is not fitted.")
        centroids = np.asarray(self.centroid_, dtype=float)
        radii = np.asarray(self.centroid_radius_, dtype=float)
        if as_list:
            return [{"centroid": float(c), "radius": float(r)} for c, r in zip(centroids, radii)]
        return {"centroid": centroids, "radius": radii}


    def get_centroids(self, as_list: bool = False):
        """
        Returns detected centroids.

        Parameters
        ----------
        as_list : bool, default False
            If True, returns a list of dicts [{'centroid': c, 'radius': r}, ...].
            If False, returns a dict with numpy arrays {'centroid': array, 'radius': array}.

        Returns
        -------
        dict | list
            Centroids information as a dict of arrays or a list of per-centroid dicts.
        """
        data = self.get_peaks(as_list=as_list)
        return data


    def get_clusters(self, as_list: bool = False):
        """
        Returns detected clusters.

        Parameters
        ----------
        as_list : bool, default False
            If True, returns a list of dicts [{'centroid': c, 'radius': r}, ...].
            If False, returns a dict with numpy arrays {'centroid': array, 'radius': array}.

        Returns
        -------
        dict | list
            Clusters information as a dict of arrays or a list of per-cluster dicts.
        """
        data = self.get_peaks(as_list=as_list)
        return data


    def show_peaks(self, color: Optional[str] = "#2980BA", title: str = "CircleClust peaks", output: Optional[str] = None):
        """
        Shows peaks in the histogram.

        Parameters
        ----------
        color : Optional[str], default "#2980BA"
            Color of the peaks.
        title : str, default "CircleClust peaks"
            Title of the plot.
        output : Optional[str], default None
            Output file name. If None, shows plot in a window.
        """
        # Check if model is fitted
        if getattr(self, "h_all_", None) is None or getattr(self, "s_all_", None) is None:
            raise RuntimeError("Call fit() before show_peaks().")
        # Get histogram and smoothed signal
        h = np.asarray(self.h_all_, dtype=float)
        s = np.asarray(self.s_all_, dtype=float)
        n = h.size
        x = np.arange(n)
        # Plot histogram and smoothed signal
        plt.figure(figsize=(12, 6))
        plt.bar(x, h, color=color, alpha=0.3, width=0.9, align="center")
        plt.plot(x, s, color=color, linewidth=2.0)
        if getattr(self, "peak_idx_", None) is not None and self.centroid_.size:
            px = self.centroid_ * n / self.period
            ix = self.peak_idx_
            py = s[ix] + 0.1 * h.max()
            plt.scatter(px, py, marker='*', s=200, color=color, zorder=4)
        # Add +/- sigma whiskers for each peak by filling a NaN array between i0..i1 (with wrap)
        if getattr(self, "peak_sigma_", None) is not None and self.peak_sigma_.size:
            lw = 2.0
            wy = py - 0.05 * h.max()
            y_cap = 0.2
            for i, c in enumerate(px):
                x0 = c - 2 * self.peak_sigma_[i]
                x1 = c + 2 * self.peak_sigma_[i]
                _draw_horizontal_whisker(x0, x1, wy[i], y_cap=y_cap, linewidth=lw, color=color)
                if x0 < 0:
                    _draw_horizontal_whisker(x0 + n, x1 + n, wy[i], y_cap=y_cap, linewidth=lw, color=color)
                if x1 > n:
                    _draw_horizontal_whisker(x0 - n, x1 - n, wy[i], y_cap=y_cap, linewidth=lw, color=color)
        plt.xlim(-0.5, n - 0.5)
        # Add xticks covering the full period at 0, 0.25, 0.5, 0.75, 1.0 * period
        ticks_frac = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        tick_pos = ticks_frac * n
        tick_pos = np.minimum(tick_pos, n - 1)  # keep within axis range
        tick_labels = [f"{(f * self.period):.2f}" for f in ticks_frac]
        plt.xticks(tick_pos, tick_labels, fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=72)
            plt.close()
        else:
            plt.show()
        return

    def show_centroids(self, output: Optional[str] = None):
        """
        Shows centroids in the histogram.

        Parameters
        ----------
        output : Optional[str], default None
            Output file name. If None, shows plot in a window.
        """
        self.show_peaks(title="CircleClust centroids", output=output)
        return


    def show_clusters(self, output: Optional[str] = None):
        """
        Shows clusters in the histogram.

        Parameters
        ----------
        output : Optional[str], default None
            Output file name. If None, shows plot in a window.
        """
        self.show_peaks(title="CircleClust clusters", output=output)
        return

def _draw_horizontal_whisker(
                            x_start: float, 
                            x_end: float, 
                            y: float, 
                            y_cap: float = 0.2,
                            cap_start: bool = True, 
                            cap_end: bool = True, 
                            linewidth: float = 1.0, 
                            color: str = "black",
                            **kwargs):
    """
    Draws a horizontal whisker line on a given matplotlib Axes.

    Parameters
    ----------
    x_start : float
        Start x-coordinate of whisker
    x_end : float
        End x-coordinate of whisker
    y : float
        Y-coordinate of the horizontal line
    y_cap : float
        Half-height of vertical caps (extends ±y_height/2 from y)
    cap_start : bool
        Whether to draw a vertical cap at x_start
    cap_end : bool
        Whether to draw a vertical cap at x_end
    linewidth : float
        Line width for all parts
    color : str
        Color of the line and caps
    kwargs : dict
        Additional kwargs passed to `ax.plot` (e.g., color)
    """
    # Draw horizontal line
    plt.plot([x_start, x_end], [y, y], linewidth=linewidth, color=color, **kwargs)
    # Draw start cap
    if cap_start:
        plt.plot([x_start, x_start], [y - y_cap, y + y_cap], linewidth=linewidth, color=color, **kwargs)
    # Draw end cap
    if cap_end:
        plt.plot([x_end, x_end], [y - y_cap, y + y_cap], linewidth=linewidth, color=color, **kwargs)
    return

def _hann_window(n: int) -> np.ndarray:
    """
    Generates a normalized Hann window of given length.

    Parameters
    ----------
    n : int
        Length of the window.

    Returns
    -------
    np.ndarray
        Hann window of length n.
    """
    window = np.hanning(n)
    s = window.sum()
    window = window / s if s != 0 else window
    return window


def _periodic_smooth(x: np.ndarray, bins_per_window: int) -> np.ndarray:
    """
    Smoothes periodically a 1D sequence with a Hann window of given bin width.

    Parameters
    ----------
    x : np.ndarray
        Sequence to smooth (e.g., histogram counts).
    bins_per_window : int
        Odd window length in bins (e.g., 9). Must be >= 3.

    Returns
    -------
    np.ndarray
        Smoothed sequence of the same length as input.
    """
    if bins_per_window is None:
        bins_per_window = 9
    win = int(bins_per_window)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    pad = win // 2
    w = _hann_window(win)
    x_ = np.asarray(x, dtype=float)
    x_padded = np.pad(x_, (pad, pad), mode='wrap')
    smoothed = np.convolve(x_padded, w, mode='valid')
    return smoothed


def _rmsd(x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Computes root mean square deviation between two 1D arrays.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.

    Returns
    -------
    float
        Root mean square deviation of the input arrays.
    """
    x_ = np.asarray(x, dtype=float)
    y_ = np.asarray(y, dtype=float)
    if mask is not None and np.any(mask):
        x_ = x_[mask]
        y_ = y_[mask]
    rmsd = float(np.sqrt(np.mean((x_ - y_) * (x_ - y_))))
    return rmsd


def _find_idxs_of_periodic_peaks(x_smooth: np.ndarray, 
                                 bins_per_window: int | None = None,
                                 epsilon: float = 1e-6) -> np.ndarray:
    """
    Finds true local peaks in a sequence of binned and smoothed periodic data.

    Algorithm
    ---------
    1) Pad by half-window in wrap mode.
    2) Find local maxima by comparing to rolls of ±1.
    3) Merge flat maxima into plateaus and take their centers.
    4) For each candidate, ensure no higher value exists within ±window around it.
    5) Map back to original indices (mod n) and deduplicate.

    Parameters
    ----------
    x_smooth : np.ndarray
        Already smoothed 1D sequence.
    bins_per_window : int
        Odd window length in bins (e.g., 9). Must be >= 3. Used to define binning of data.
    epsilon : float, default 1e-6
        Epsilon value for neighborhood check.

    Returns
    -------
    np.ndarray
        Unique indices of detected peaks in the original (un-padded) array. Sorted in ascending order.
    """
    x_ = np.asarray(x_smooth, dtype=float)
    n = x_.size
    if n == 0:
        return np.array([], dtype=int)

    win = int(bins_per_window)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    half = win // 2

    x_padded = np.pad(x_, (half, half), mode='wrap')
    left = np.roll(x_padded, 1)
    right = np.roll(x_padded, -1)
    is_plateau = (x_padded >= left) & (x_padded >= right) & (x_padded > epsilon)

    # Use find_intervals on the boolean mask (converted to int) over padded domain
    intervals = find_intervals(is_plateau.astype(int))  # shape (K,2), half-open [start,end)
    peaks_idx = []
    for i0, i1 in intervals:
        # center index in [a,b): map to inclusive representation by b-1
        c = (int(i0) + int(i1) - 1) // 2
        c_val = x_padded[c]
        # Neighborhood check within ±win
        neighborhood = x_padded[i0 - half : i1 + half]
        if c_val > epsilon and np.all(neighborhood <= c_val + epsilon):
            peaks_idx.append(c)

    # Map back to original indices (remove padding and wrap)
    peaks_orig = [((idx - half) % n) for idx in peaks_idx]
    if not peaks_orig:
        return np.array([], dtype=int)
    peaks_idxs = np.unique(np.asarray(peaks_orig, dtype=int))

    # Periodic non-maximum suppression: keep strongest peaks separated by >= half-window
    if peaks_idxs.size <= 1:
        return peaks_idxs
    peak_strengths = x_smooth[peaks_idxs]
    idxs = np.argsort(peak_strengths)[::-1]  # descending by strength
    peaks_retained: list[int] = []
    for idx in idxs:
        candidate_id = int(peaks_idxs[idx])
        if not peaks_retained:
            peaks_retained.append(candidate_id)
            continue
        retain = True
        for peak_id in peaks_retained:
            d = abs(candidate_id - peak_id)
            d = min(d, n - d)  # circular distance in bins
            if d < half:
                retain = False
                break
        if retain:
            peaks_retained.append(candidate_id)
    peak_idxs = np.sort(np.array(peaks_retained, dtype=int))
    return peak_idxs


def _find_idxs_of_midpoints_between_peaks(peak_idxs: np.ndarray, nbins: int) -> np.ndarray:
    """
    Finds midpoints between consecutive peaks of periodic data.

    Parameters
    ----------
    peak_idxs : np.ndarray
        Array of peak indices. Important: must be sorted in ascending order.
    nbins : int
        Number of bins in the binned periodic data.

    Returns
    -------
    np.ndarray
        Array of midpoint indices. Sorted in ascending order.

    """
    # Check that peak_idxs is not empty
    if peak_idxs.size == 0:
        return np.array([], dtype=int)
    # If only one peak, return midpoint on the opposite side of the circle
    if peak_idxs.size == 1:
        midpoint = peak_idxs[0] + nbins // 2
        return np.array([midpoint % nbins], dtype=int)
    # Check that peak_idxs is sorted in ascending order
    if not np.all(np.diff(peak_idxs) >= 0):
        raise ValueError("peak_idxs must be sorted in ascending order")
    # Forward midpoints between consecutive peaks on circle
    midpoints = []
    for i in range(peak_idxs.size):
        i0 = int(peak_idxs[i])
        i1 = int(peak_idxs[(i + 1) % peak_idxs.size])
        d = (i1 - i0) % nbins
        # Handle periodic wrap-around
        if d < 0:
            d += nbins
        midpoints.append(i0 + d // 2)
    midpoints = np.asarray(midpoints, dtype=int)
    return midpoints

def _slice_periodic_segment(x: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """
    Slices a periodic segment from an array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    i0 : int
        Start index.
    i1 : int
        End index.

    Returns
    -------
    np.ndarray
        Sliced array.
    """
    x_ = np.asarray(x)
    n = x_.size
    if n == 0:
        return x_.copy()
    i0 = int(i0) % n
    i1 = int(i1) % n
    if i0 < i1:
        return x_[i0 : i1 + 1]
    return np.concatenate([x_[i0:], x_[: i1 + 1]])

