import os
import sys
import colorsys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from typing import Tuple, Optional


def verbose_print(message: str, verbose: bool = False):
    """
    Print a message if verbose is True.
    
    Args:
        message: Message to print
        verbose: Whether to print the message
    """
    if verbose:
        print(message)


def warning_print(message: str):
    """Print a warning message to stderr without raising.

    The message is prefixed with "WARNING:".

    Parameters
    ----------
    message : str
        The message to write to stderr.
    """
    print(f"WARNING: {message}", file=sys.stderr)


def find_intervals(x: np.ndarray) -> np.ndarray:
    """
    Finds continuous positive intervals in 1d-array.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-negative numeric values

    Returns
    -------
    np.ndarray
        2D array - N intervals x 2 indices (start, end)

    """
    assert x.ndim == 1 and not any(x < 0)
    x_ = (x > 0).astype(int)
    x_ = np.pad(x_, (1, 1), mode='constant', constant_values=0)
    x_ = np.diff(x_)
    idx = np.arange(len(x_))
    i0 = idx[x_ == 1]
    i1 = idx[x_ == -1]
    idx = np.stack([i0,i1]).T
    return idx


def debug_plot_screen_iteration(h_train: np.ndarray, s_train: np.ndarray, h_test: np.ndarray, s_test: np.ndarray, iteration: int, i_window: int, outdir: str = "debug"):
    t = np.arange(len(h_train))
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Iteration {iteration+1} {i_window+1}', fontsize=22)
    plt.subplot(2, 1, 1)
    plt.ylabel('Train')
    plt.scatter(t, h_train, color='red')
    plt.plot(t, h_train, color='red', label='Train histogram')
    plt.plot(t, s_train, color='blue', label='Smoothed train')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.ylabel('Test')
    plt.scatter(t, h_test, color='red')
    plt.plot(t, h_test, color='red', label='Test histogram')
    plt.plot(t, s_test, color='blue', label='Smoothed test')
    plt.legend()
    os.makedirs(outdir or ".", exist_ok=True)
    plt.savefig(os.path.join(outdir, f'screen_{iteration+1}_{i_window+1}.jpg'))
    plt.close()


def debug_plot_window_selection(rms_tr_mean: np.ndarray, rms_ts_mean: np.ndarray, i0: int, outdir: str = 'debug'):
    plt.figure(figsize=(12, 6))
    t = np.arange(len(rms_tr_mean))
    plt.suptitle('Window selection based on model complexity', fontsize=22)
    plt.scatter(t, rms_tr_mean, color='green')
    plt.plot(t, rms_tr_mean, color='green', label='Train RMS')
    plt.scatter(t, rms_ts_mean, color='blue')
    plt.plot(t, rms_ts_mean, color='blue', label='Test RMS')
    # Mark selected knee/minimum with an asterisk at the higher of the two series
    y_sel = max(float(rms_tr_mean[i0]), float(rms_ts_mean[i0]))
    plt.scatter([i0], [y_sel], marker='*', s=180, color='black', zorder=5, label='selected')
    plt.legend()
    os.makedirs(outdir or ".", exist_ok=True)
    plt.savefig(os.path.join(outdir, 'window.jpg'))
    plt.close()


def gaussian_shifted(x: np.ndarray, A: float, mu: float, sigma: float, C: float) -> np.ndarray:
    """
    Computes gaussian normal distribution with a vertical shift.

    Parameters:
        x : array-like
            Input x values
        A : float
            Amplitude
        mu : float
            Mean
        sigma : float
            Standard deviation
        C : float
            Vertical shift constant
    
    Returns:
        y : array-like
            Gaussian values
    """
    y = np.abs(A) * np.exp(- (x - np.abs(mu))**2 / (2 * sigma**2)) + np.abs(C)
    return y


def fit_gaussian_shifted(y: np.ndarray, mu: float | None = None):
    """
    Fits a vertically-shifted Gaussian to data.
    
    Parameters:
        y : array-like
            Input data values on an equispaced grid
        mu : float or None
            If float, uses fixed center; if None, fits center

    Returns:
        popt : array
            Fitted parameters A, mu, sigma, C or None
    """
    # Generate x values
    x = np.arange(len(y))
    # Fit curve
    if mu is None:
        # Fit A, mu, sigma, C
        try:
            p0 = [np.max(y), float(len(x) / 2), float(len(x) / 4), np.mean(y)]
            popt, pcov = curve_fit(gaussian_shifted, x, y, p0=p0)
            popt = [np.abs(p) for p in popt]
        except Exception:
            warning_print("Failed to fit Gaussian to data. Return None.")
            return None 
        return popt
    else:
        # Define wrapper with fixed mu
        def gaussian_with_fixed_mu(x, A, sigma, C):
            return gaussian_shifted(x, A, mu, sigma, C)
        # Fit only A, sigma, C
        try:
            p0 = [np.max(y), float(len(x) / 4), np.mean(y)]
            popt, pcov = curve_fit(gaussian_with_fixed_mu, x, y, p0=p0)
            popt = [np.abs(p) for p in popt]
        except Exception:
            warning_print("Failed to fit Gaussian to data. Return None.")
            return None 
        # Reconstruct full parameter list with fixed mu
        popt = [popt[0], mu, popt[1], popt[2]]
        return popt
