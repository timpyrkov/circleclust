#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

TWOPI = 2.0 * math.pi


def periodic_gaussian(theta: np.ndarray, mu: float, sigma: float, kwrap: int = 3) -> np.ndarray:
    vals = np.zeros_like(theta, dtype=float)
    for m in range(-kwrap, kwrap + 1):
        d = theta - (mu + m * TWOPI)
        vals += np.exp(-0.5 * (d / sigma) ** 2)
    return vals


def draw_logo(size_px: int = 512, dpi: int = 72, cmap_name: str = "YlGnBu", output: str = "logo.png"):
    # Compute figure size in inches for the requested pixel size at the given DPI
    figsize = (size_px / dpi, size_px / dpi)

    # Prepare angle grid
    n = 2048
    theta = np.linspace(0.0, TWOPI, n, endpoint=False)

    # Three peak components on the circle (angles in radians)
    peaks = [
        {"a": 0.2, "mu": 0.15 * TWOPI, "sigma": 0.3},
        {"a": 0.3, "mu": 0.85 * TWOPI, "sigma": 0.4},
        {"a": 0.4, "mu": 0.45 * TWOPI, "sigma": 0.4},
    ]

    # Colors from the colormap, avoid the exact ends
    cmap = plt.colormaps.get_cmap(cmap_name)
    color_positions = np.linspace(0.15, 0.85, len(peaks))
    color = [cmap(p) for p in color_positions][::-1]
    color = ['#CCFF66', '#6BB0DE', '#2980BA', 'white']
    #color = ['#CC241D', '#D79921', '#689D6A', '#FBF1C7']

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#00000000')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect('equal')
    ax.set_xlim(-.95, .95)
    ax.set_ylim(-.95, .95)
    ax.axis('off')
    fig.patch.set_facecolor('#00000000')

    # Create base circle
    r0 = 0.3  # leave margins
    x0 = r0 * np.cos(theta)
    y0 = r0 * np.sin(theta)
    #ax.plot(x0, y0, color=(0.6, 0.6, 0.6), linewidth=2.0, solid_capstyle='round')

    # Draw backgrounds
    n = len(peaks)
    linewidth = 60
    for i, peak in enumerate(peaks):
        amp = float(peak["a"])  # radial modulation amplitude
        mu = float(peak["mu"]) % TWOPI
        sigma = max(1e-3, float(peak["sigma"]))
        profile = periodic_gaussian(theta, mu, sigma)
        r = r0 + 0.12 * (n - i) + amp * profile
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color=color[3], linewidth=linewidth, solid_capstyle='round')

    # Fill inner area (solid white) inside each "peak"
    for i, peak in enumerate(peaks):
        amp = float(peak["a"])
        mu = float(peak["mu"]) % TWOPI
        sigma = max(1e-3, float(peak["sigma"]))
        profile = periodic_gaussian(theta, mu, sigma)
        r = r0 + 0.12 * (n - i) + amp * profile
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Create a polygon that fills from center to the profile
        x_fill = np.concatenate([[0], x])
        y_fill = np.concatenate([[0], y])
        ax.fill(x_fill, y_fill, color=color[3], zorder=1)

    # Draw peaks
    n = len(peaks)
    linewidth = 33
    for i, peak in enumerate(peaks):
        amp = float(peak["a"])  # radial modulation amplitude
        mu = float(peak["mu"]) % TWOPI
        sigma = max(1e-3, float(peak["sigma"]))
        profile = periodic_gaussian(theta, mu, sigma)
        r = r0 + 0.12 * (n - i) + amp * profile
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color=color[i], linewidth=linewidth, solid_capstyle='round')

    # Save PNG
    fig.savefig(output, dpi=dpi, transparent=True)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate CircleClust logo images.")
    parser.add_argument("-o", "--output", type=str, default="logo.png", help="Output file path")
    parser.add_argument("-s", "--size", type=int, default=512, help="Output image size in pixels")
    parser.add_argument("-c", "--cmap", type=str, default="YlGnBu", help="Matplotlib colormap name")
    parser.add_argument("-d", "--dpi", type=int, default=72, help="DPI for saved images")
    args = parser.parse_args()

    # Generate requested sizes
    output = draw_logo(size_px=args.size, dpi=args.dpi, cmap_name=args.cmap, output=args.output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
