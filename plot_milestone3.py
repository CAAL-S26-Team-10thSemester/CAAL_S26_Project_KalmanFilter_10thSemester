#!/usr/bin/env python3
"""
plot_milestone3.py  —  Milestone-3 Plotting Script
====================================================
Generates all required plots:

  1. Time-series (x, y, z) for a chosen joint:
       - Position
       - Velocity
       - Acceleration
       - Jerk
     → For BOTH LKF and EKF assembly outputs (4 plots × 2 filters = 8 files)

  2. True vs Noisy vs Estimated position (x, y, z)
       → For LKF and EKF separately (2 files)

  3. LKF vs EKF comparison plots for the chosen joint
       → Position, Velocity, Acceleration, Jerk (4 files)

  4. Milestone-2 vs Milestone-3 comparison plots
       → Position comparison per axis (3 files)

Usage:
  python3 plot_milestone3.py \
      --noisy   "...Noisy Values....csv"  \
      --true    "...True Values....csv"   \
      --lkf_m3  lkf_asm_results.csv       \
      --ekf_m3  ekf_asm_results.csv       \
      [--lkf_m2  lkf_m2_results.csv]      \   # optional Milestone-2 LKF
      [--ekf_m2  ekf_m2_results.csv]      \   # optional Milestone-2 EKF
      [--joint   0]                            # 0-22, default 0 (pelvis)

Outputs:   saved to ./plots/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────────────────────
NUM_JOINTS  = 23
STATE_DIM   = 12    # px vx ax jx  py vy ay jy  pz vz az jz
MEAS_DIM    = 3
DT          = 1.0 / 30.0

JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]

# State layout per joint (12-element block):
#   [0] px  [1] vx  [2] ax  [3] jx
#   [4] py  [5] vy  [6] ay  [7] jy
#   [8] pz  [9] vz [10] az [11] jz
STATE_IDX = {
    "px": 0,  "vx": 1,  "ax": 2,  "jx": 3,
    "py": 4,  "vy": 5,  "ay": 6,  "jy": 7,
    "pz": 8,  "vz": 9,  "az": 10, "jz": 11,
}

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def load_filter_csv(path):
    """Load lkf_asm_results.csv / ekf_asm_results.csv.
    First column is frame index; remaining 276 columns are state.
    Returns numpy array (N_frames, N_STATE=276).
    """
    df = pd.read_csv(path)
    arr = df.iloc[:, 1:].values   # drop frame column
    assert arr.shape[1] == NUM_JOINTS * STATE_DIM, \
        f"Expected {NUM_JOINTS*STATE_DIM} state columns, got {arr.shape[1]}"
    return arr                    # (N_frames, 276)


def load_noisy_csv(path):
    """Load noisy/true CSV: N_frames × (NUM_JOINTS*MEAS_DIM).
    May or may not have a header row.
    Returns (N_frames, NUM_JOINTS, 3).
    """
    try:
        df = pd.read_csv(path, header=None)
        float(df.iloc[0, 0])      # first row is data
    except (ValueError, IndexError):
        df = pd.read_csv(path)    # first row is header

    data = df.values.astype(float)
    if data.shape[1] > NUM_JOINTS * MEAS_DIM:
        data = data[:, :NUM_JOINTS * MEAS_DIM]
    N = data.shape[0]
    return data.reshape(N, NUM_JOINTS, MEAS_DIM)


def get_joint_state(state_arr, joint_idx):
    """Extract the 12-element sub-state for one joint from (N, 276) array.
    Returns (N, 12).
    """
    base = joint_idx * STATE_DIM
    return state_arr[:, base:base + STATE_DIM]


def time_axis(N):
    return np.arange(N) * DT


def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Time-series for one filter ─────────────────────────────────────

def plot_timeseries(state_arr, joint_idx, joint_name, label, tag):
    """4 figures: position, velocity, acceleration, jerk (each 3 axes)."""
    js  = get_joint_state(state_arr, joint_idx)
    t   = time_axis(len(js))
    dims = ("x", "y", "z")

    quantities = [
        ("Position",     ["px","py","pz"], "m",       "pos"),
        ("Velocity",     ["vx","vy","vz"], "m/s",     "vel"),
        ("Acceleration", ["ax","ay","az"], "m/s²",    "acc"),
        ("Jerk",         ["jx","jy","jz"], "m/s³",    "jrk"),
    ]

    for qty_name, keys, unit, qtag in quantities:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            f"{label} — Joint: {joint_name} — {qty_name}", fontsize=13)
        colours = ["#e41a1c", "#377eb8", "#4daf4a"]
        for i, (ax, key, dim) in enumerate(zip(axes, keys, dims)):
            ax.plot(t, js[:, STATE_IDX[key]],
                    color=colours[i], linewidth=1.2, label=f"{dim}-axis")
            ax.set_ylabel(f"{dim} ({unit})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[-1].set_xlabel("Time (s)")
        fname = f"{tag}_{joint_name}_{qtag}.png"
        savefig(fig, fname)


# ── Plot 2: True vs Noisy vs Estimated position ────────────────────────────

def plot_true_noisy_estimated(true_arr, noisy_arr, state_arr,
                              joint_idx, joint_name, label, tag):
    """3-panel figure: px, py, pz."""
    js    = get_joint_state(state_arr, joint_idx)
    true_j  = true_arr [:, joint_idx, :]   # (N, 3)
    noisy_j = noisy_arr[:, joint_idx, :]

    N   = min(len(js), len(true_j), len(noisy_j))
    t   = time_axis(N)
    pos_keys = ["px", "py", "pz"]
    axes_lbl = ("x", "y", "z")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f"{label} — Joint: {joint_name} — True vs Noisy vs Estimated",
        fontsize=13)

    for i, (ax, key, al) in enumerate(zip(axes, pos_keys, axes_lbl)):
        ax.plot(t, true_j [:N, i],
                "k-",   linewidth=1.4, label="True",      zorder=3)
        ax.plot(t, noisy_j[:N, i],
                "gray",  linewidth=0.7, alpha=0.7,
                label="Noisy",     zorder=1)
        ax.plot(t, js     [:N, STATE_IDX[key]],
                "r--",  linewidth=1.2, label="Estimated", zorder=2)
        ax.set_ylabel(f"p_{al} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fname = f"{tag}_{joint_name}_true_noisy_est.png"
    savefig(fig, fname)


# ── Plot 3: LKF vs EKF comparison ──────────────────────────────────────────

def plot_lkf_vs_ekf(lkf_arr, ekf_arr, joint_idx, joint_name):
    """4 figures: position, velocity, acceleration, jerk."""
    lkf_js = get_joint_state(lkf_arr, joint_idx)
    ekf_js = get_joint_state(ekf_arr, joint_idx)
    N  = min(len(lkf_js), len(ekf_js))
    t  = time_axis(N)

    quantities = [
        ("Position",     ["px","py","pz"], "m",    "pos"),
        ("Velocity",     ["vx","vy","vz"], "m/s",  "vel"),
        ("Acceleration", ["ax","ay","az"], "m/s²", "acc"),
        ("Jerk",         ["jx","jy","jz"], "m/s³", "jrk"),
    ]
    dims = ("x", "y", "z")

    for qty_name, keys, unit, qtag in quantities:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig.suptitle(
            f"LKF vs EKF — Joint: {joint_name} — {qty_name}", fontsize=13)
        colours = ["#e41a1c", "#377eb8", "#4daf4a"]
        for i, (ax, key, dim) in enumerate(zip(axes, keys, dims)):
            ax.plot(t, lkf_js[:N, STATE_IDX[key]],
                    color=colours[i], linewidth=1.3,
                    linestyle="-",  label=f"LKF {dim}")
            ax.plot(t, ekf_js[:N, STATE_IDX[key]],
                    color=colours[i], linewidth=1.3,
                    linestyle="--", label=f"EKF {dim}")
            ax.set_ylabel(f"{dim} ({unit})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        axes[-1].set_xlabel("Time (s)")
        fname = f"lkf_vs_ekf_{joint_name}_{qtag}.png"
        savefig(fig, fname)


# ── Plot 4: Milestone-2 vs Milestone-3 comparison ──────────────────────────

def plot_m2_vs_m3(m2_arr, m3_arr, joint_idx, joint_name,
                  filter_name, tag):
    """Position comparison: Milestone-2 (Python) vs Milestone-3 (assembly)."""
    m2_js = get_joint_state(m2_arr, joint_idx)
    m3_js = get_joint_state(m3_arr, joint_idx)
    N  = min(len(m2_js), len(m3_js))
    t  = time_axis(N)

    dims = ("x", "y", "z")
    pos_keys = ["px", "py", "pz"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f"{filter_name} — Milestone-2 (C/C++ ref) vs Milestone-3 (ASM)\n"
        f"Joint: {joint_name} — Position",
        fontsize=12)
    colours = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, (ax, key, dim) in enumerate(zip(axes, pos_keys, dims)):
        ax.plot(t, m2_js[:N, STATE_IDX[key]],
                color=colours[i], linewidth=1.5,
                linestyle="-",  label=f"M2 p_{dim}")
        ax.plot(t, m3_js[:N, STATE_IDX[key]],
                color=colours[i], linewidth=1.2,
                linestyle="--", label=f"M3-ASM p_{dim}")
        ax.set_ylabel(f"p_{dim} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fname = f"{tag}_{joint_name}_m2_vs_m3.png"
    savefig(fig, fname)


# ── Plot 5: Error histogram ─────────────────────────────────────────────────

def plot_error_histogram(ref_arr, asm_arr, filter_name, tag):
    """Histogram of |asm - ref| across all frames × all state elements."""
    N = min(len(ref_arr), len(asm_arr))
    errors = np.abs(asm_arr[:N] - ref_arr[:N]).flatten()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(errors + 1e-20), bins=60,
            color="#4daf4a", edgecolor="k", linewidth=0.4)
    ax.axvline(-9, color="red", linestyle="--", linewidth=1.5,
               label="ε_tol = 1e-9")
    ax.set_xlabel("log₁₀(|error|)")
    ax.set_ylabel("Count")
    ax.set_title(f"{filter_name} — Error distribution (ASM vs reference)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fname = f"{tag}_error_histogram.png"
    savefig(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--noisy",  required=True,  help="Noisy measurements CSV")
    ap.add_argument("--true",   required=False, default=None,
                    help="True (ground-truth) measurements CSV")
    ap.add_argument("--lkf_m3", required=True,  help="LKF assembly results CSV")
    ap.add_argument("--ekf_m3", required=True,  help="EKF assembly results CSV")
    ap.add_argument("--lkf_m2", required=False, default=None,
                    help="Milestone-2 LKF reference CSV (optional)")
    ap.add_argument("--ekf_m2", required=False, default=None,
                    help="Milestone-2 EKF reference CSV (optional)")
    ap.add_argument("--joint",  type=int, default=0,
                    help="Joint index to plot (0-22, default 0 = pelvis)")
    args = ap.parse_args()

    joint_idx  = args.joint
    joint_name = JOINT_NAMES[joint_idx]
    print(f"\n[INFO] Plotting joint: {joint_idx} ({joint_name})")
    print(f"[INFO] Output directory: {PLOT_DIR}/\n")

    # Load data
    print("[LOAD] Noisy CSV ...")
    noisy_arr = load_noisy_csv(args.noisy)

    true_arr = None
    if args.true:
        print("[LOAD] True CSV ...")
        true_arr = load_noisy_csv(args.true)

    print("[LOAD] LKF M3 ...")
    lkf_m3 = load_filter_csv(args.lkf_m3)
    print("[LOAD] EKF M3 ...")
    ekf_m3 = load_filter_csv(args.ekf_m3)

    lkf_m2 = None
    if args.lkf_m2:
        print("[LOAD] LKF M2 ...")
        lkf_m2 = load_filter_csv(args.lkf_m2)

    ekf_m2 = None
    if args.ekf_m2:
        print("[LOAD] EKF M2 ...")
        ekf_m2 = load_filter_csv(args.ekf_m2)

    # ── 1. Time-series plots ─────────────────────────────────────────────
    print("\n[PLOT] Time-series: LKF ...")
    plot_timeseries(lkf_m3, joint_idx, joint_name,
                    "LKF Assembly (M3)", "lkf_m3")

    print("[PLOT] Time-series: EKF ...")
    plot_timeseries(ekf_m3, joint_idx, joint_name,
                    "EKF Assembly (M3)", "ekf_m3")

    # ── 2. True vs Noisy vs Estimated ───────────────────────────────────
    if true_arr is not None:
        print("[PLOT] True vs Noisy vs Estimated: LKF ...")
        plot_true_noisy_estimated(
            true_arr, noisy_arr, lkf_m3,
            joint_idx, joint_name, "LKF Assembly (M3)", "lkf_m3")

        print("[PLOT] True vs Noisy vs Estimated: EKF ...")
        plot_true_noisy_estimated(
            true_arr, noisy_arr, ekf_m3,
            joint_idx, joint_name, "EKF Assembly (M3)", "ekf_m3")
    else:
        print("[SKIP] True CSV not provided — skipping true/noisy/est plots")

    # ── 3. LKF vs EKF comparison ─────────────────────────────────────────
    print("[PLOT] LKF vs EKF comparison ...")
    plot_lkf_vs_ekf(lkf_m3, ekf_m3, joint_idx, joint_name)

    # ── 4. Milestone-2 vs Milestone-3 ────────────────────────────────────
    if lkf_m2 is not None:
        print("[PLOT] LKF M2 vs M3 ...")
        plot_m2_vs_m3(lkf_m2, lkf_m3, joint_idx, joint_name, "LKF", "lkf")
        print("[PLOT] LKF error histogram ...")
        plot_error_histogram(lkf_m2, lkf_m3, "LKF", "lkf")
    else:
        print("[SKIP] LKF M2 not provided — skipping LKF M2 vs M3 plots")

    if ekf_m2 is not None:
        print("[PLOT] EKF M2 vs M3 ...")
        plot_m2_vs_m3(ekf_m2, ekf_m3, joint_idx, joint_name, "EKF", "ekf")
        print("[PLOT] EKF error histogram ...")
        plot_error_histogram(ekf_m2, ekf_m3, "EKF", "ekf")
    else:
        print("[SKIP] EKF M2 not provided — skipping EKF M2 vs M3 plots")

    print(f"\n[DONE] All plots saved to ./{PLOT_DIR}/")


if __name__ == "__main__":
    main()
