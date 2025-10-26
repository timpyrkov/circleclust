#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt


def generate_sleep_data(out_path: str, seed: int = 0):
    """Generate synthetic sleep data for the year 2025 and save to CSV.

    Parameters
    ----------
    out_path : str
        Output CSV filepath. If a directory is included, it will be created.
    seed : int, default 0
        Seed for NumPy's Generator to ensure reproducibility.

    Notes
    -----
    - Simulates daily go-to-sleep and wake-up times with weekday/weekend and
      illness-week rules.
    - Handles minutes > 1440 by rolling into the next day.
    - CSV columns: `sleep_start_datetime`, `sleep_end_datetime`, `wake_reason`.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    rng = np.random.default_rng(seed)

    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)

    # Choose fixed illness weeks (Mon-Sun) for reproducibility
    illness_weeks = [
        (date(2025, 3, 23), date(2025, 3, 29)),  # March 23-29, 2025
        (date(2025, 10, 13), date(2025, 10, 19)),  # Oct 13-19, 2025
    ]

    def is_in_illness_week(d: date) -> bool:
        return any(start <= d <= end for start, end in illness_weeks)

    rows = []

    current = start_date
    while current <= end_date:
        next_day = current + timedelta(days=1)
        next_wd = next_day.weekday()  # Mon=0 .. Sun=6

        ill = is_in_illness_week(current)

        # Go-to-sleep minutes past previous midnight
        if ill:
            mean_sleep = 1560  # 2:00 AM next day
            std_sleep = 120
        elif next_wd in (5, 6):  # day before weekend (Fri or Sat night)
            mean_sleep = 1470  # 1:30 AM next day
            std_sleep = 90
        else:  # day before weekday (Sun-Thu night)
            mean_sleep = 1410  # 11:30 PM same day
            std_sleep = 60

        sleep_start_min = rng.normal(mean_sleep, std_sleep)
        # keep within a sensible range [0, 2000] without hard clipping too strong
        sleep_start_min = max(0, min(float(sleep_start_min), 2000))

        base_dt = datetime.combine(current, datetime.min.time())
        sleep_dt = base_dt + timedelta(minutes=sleep_start_min)

        # Wake-up time
        if ill:
            duration_mean = 11 * 60 # Sleep longer during illness
            duration_std = 60
            duration = max(4 * 60, min(16 * 60, float(rng.normal(duration_mean, duration_std))))
            wake_dt = sleep_dt + timedelta(minutes=duration)
            wake_reason = "Illness: Sleep longer"
        elif next_wd in (5, 6):  # Sleep longer on day before weekend (Fri or Sat night)
            duration_mean = 8 * 60
            duration_std = 60
            duration = max(4 * 60, min(16 * 60, float(rng.normal(duration_mean, duration_std))))
            wake_dt = sleep_dt + timedelta(minutes=duration)
            wake_reason = "Weekend: Recover from sleep debt"
        else:  # Mon-Fri wake at ~06:30
            wake_mean = 6 * 60 + 30  # 390
            wake_std = 10
            wake_base = datetime.combine(next_day, datetime.min.time())
            wake_dt = wake_base + timedelta(minutes=float(rng.normal(wake_mean, wake_std)))
            wake_reason = "Weekday: Alarm clock"

        rows.append(
            {
                "sleep_start_datetime": sleep_dt.isoformat(),
                "sleep_end_datetime": wake_dt.isoformat(),
                "wake_reason": wake_reason,
            }
        )

        current += timedelta(days=1)

    df = pd.DataFrame(rows)

    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def visualize_actogram(csv_path: str):
    """Visualize a 2025 actogram heatmap from a CSV of sleep intervals.

    Builds a per-minute array for 365 days where 0 = asleep and 1 = awake,
    reshapes to 365 x 1440, and displays five subplots (73 days each) using a
    heatmap. No images are saved.

    Parameters
    ----------
    csv_path : str
        Path to a CSV with `sleep_start_datetime` and `sleep_end_datetime`.
    """
    df = pd.read_csv(csv_path, parse_dates=["sleep_start_datetime", "sleep_end_datetime"])

    year_start = datetime(2025, 1, 1)
    year_end = datetime(2025, 12, 31, 23, 59)
    total_minutes = 365 * 1440

    arr = np.ones(total_minutes, dtype=np.uint8)

    for _, row in df.iterrows():
        s = row["sleep_start_datetime"]
        e = row["sleep_end_datetime"]
        if pd.isna(s) or pd.isna(e):
            continue
        s = max(s.to_pydatetime(), year_start)
        e = min(e.to_pydatetime(), year_end + timedelta(minutes=1))
        if e <= year_start or s >= year_end + timedelta(minutes=1):
            continue
        s_idx = int((s - year_start).total_seconds() // 60)
        e_idx = int(np.ceil((e - year_start).total_seconds() / 60.0))
        s_idx = max(0, min(total_minutes, s_idx))
        e_idx = max(0, min(total_minutes, e_idx))
        if e_idx > s_idx:
            arr[s_idx:e_idx] = 0

    mat = arr.reshape((365, 1440))

    fig, axes = plt.subplots(1, 5, figsize=(18, 6), constrained_layout=True)
    days_per_col = 73
    for i in range(5):
        start_day = i * days_per_col
        end_day = (i + 1) * days_per_col
        block = mat[start_day:end_day, :]
        ax = axes[i]
        im = ax.imshow(block, aspect='auto', interpolation='nearest', cmap='cividis', vmin=0, vmax=1)
        # Set x-ticks every six hours: 0, 360, 720, 1080, 1440 (last tick set at 1439 to stay within bounds)
        xticks = [0, 360, 720, 1080, 1439]
        ax.set_xlim(0, 1439)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"])
        start_dt = (year_start + timedelta(days=start_day)).date().isoformat()
        end_dt = (year_start + timedelta(days=end_day - 1)).date().isoformat()
        ax.set_title(f"{start_dt} - {end_dt}")
        ax.set_xlabel("Time of day")
        if i == 0:
            ax.set_ylabel("Days")
    fig.suptitle("2025 Actogram", fontsize=22)
    plt.show()


def main():
    """CLI entry point for generating and/or visualizing 2025 sleep data.

    Behavior
    --------
    - If `-o/--out` points to an existing file and `-v/--visualize` is given,
      plot from the existing file; otherwise print an error and exit.
    - If the file does not exist, generate the CSV; also plot if `-v` is set.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic 2025 sleep records CSV.")
    parser.add_argument("-o", "--out", type=str, default="sleep.csv", help="Output CSV path (default: sleep.csv)")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize sleep patterns")
    args = parser.parse_args()

    if os.path.exists(args.out):
        if args.visualize:
            visualize_actogram(args.out)
            return
        else:
            print(f"Error: output file exists: {args.out}. Use -v/--visualize to plot.")
            sys.exit(1)

    out_path = generate_sleep_data(args.out, seed=args.seed)
    print(f"Saved: {out_path}")
    if args.visualize:
        visualize_actogram(out_path)


if __name__ == "__main__":
    main()
