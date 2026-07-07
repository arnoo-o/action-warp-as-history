#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT.parent / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ANALYSIS_DIR = REPO_ROOT / "data" / "nuke-000000" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CS2 action analysis outputs.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def load_inputs(analysis_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    button_df = pd.read_csv(analysis_dir / "button_summary.csv")
    combo_df = pd.read_csv(analysis_dir / "top_action_combos.csv")
    fire_df = pd.read_csv(analysis_dir / "primary_fire_segments.csv")
    timeline_df = pd.read_csv(analysis_dir / "primary_fire_timeline_bins.csv")
    return summary, button_df, combo_df, fire_df, timeline_df


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def save_button_bar(button_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = button_df.sort_values("frame_ratio_pct", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    colors = ["#d95f02" if label == "primary_fire" else "#1f77b4" for label in plot_df["label"]]
    ax.bar(plot_df["label"], plot_df["frame_ratio_pct"], color=colors)
    style_axis(ax)
    ax.set_title("CS2 Action Share by Button", fontsize=14, weight="bold")
    ax.set_ylabel("Frame Ratio (%)")
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(plot_df["frame_ratio_pct"]):
        ax.text(idx, value + 0.45, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_combo_bar(combo_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = combo_df.head(12).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.barh(plot_df["action_combo"], plot_df["frame_ratio_pct"], color="#4c78a8")
    style_axis(ax)
    ax.set_title("Top Action Combos", fontsize=14, weight="bold")
    ax.set_xlabel("Frame Ratio (%)")
    for y, value in enumerate(plot_df["frame_ratio_pct"]):
        ax.text(value + 0.18, y, f"{value:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_fire_timeline(timeline_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(timeline_df["normalized_window"], timeline_df["segment_ratio_pct"], color="#d95f02")
    style_axis(ax)
    ax.set_title("Primary Fire Segment Start Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Normalized Video Time Window")
    ax.set_ylabel("Share of Fire Segments (%)")
    ax.tick_params(axis="x", rotation=0)
    for idx, value in enumerate(timeline_df["segment_ratio_pct"]):
        ax.text(idx, value + 0.35, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_fire_duration(fire_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    axes[0].hist(fire_df["duration_s"], bins=20, color="#e45756", edgecolor="white")
    style_axis(axes[0])
    axes[0].set_title("Primary Fire Duration Histogram", fontsize=13, weight="bold")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Segment Count")

    axes[1].scatter(
        fire_df["start_norm"],
        fire_df["duration_s"],
        s=26,
        alpha=0.75,
        color="#f58518",
        edgecolors="none",
    )
    style_axis(axes[1])
    axes[1].set_title("Fire Start Position vs Duration", fontsize=13, weight="bold")
    axes[1].set_xlabel("Normalized Start Time")
    axes[1].set_ylabel("Duration (s)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_dashboard(summary: dict, button_df: pd.DataFrame, combo_df: pd.DataFrame, timeline_df: pd.DataFrame, fire_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.patch.set_facecolor("white")
    fig.suptitle("Warp-as-History CS2 Nuke Action Overview", fontsize=18, weight="bold", y=0.98)

    top_buttons = button_df.sort_values("frame_ratio_pct", ascending=False).head(8)
    colors = ["#d95f02" if label == "primary_fire" else "#4c78a8" for label in top_buttons["label"]]
    axes[0, 0].bar(top_buttons["label"], top_buttons["frame_ratio_pct"], color=colors)
    style_axis(axes[0, 0])
    axes[0, 0].set_title("Top Button Share")
    axes[0, 0].set_ylabel("Frame Ratio (%)")
    axes[0, 0].tick_params(axis="x", rotation=30)

    top_combos = combo_df.head(10).iloc[::-1]
    axes[0, 1].barh(top_combos["action_combo"], top_combos["frame_ratio_pct"], color="#72b7b2")
    style_axis(axes[0, 1])
    axes[0, 1].set_title("Top Action Combos")
    axes[0, 1].set_xlabel("Frame Ratio (%)")

    axes[1, 0].bar(timeline_df["normalized_window"], timeline_df["segment_ratio_pct"], color="#e45756")
    style_axis(axes[1, 0])
    axes[1, 0].set_title("Primary Fire Timeline")
    axes[1, 0].set_xlabel("Normalized Time")
    axes[1, 0].set_ylabel("Segment Share (%)")

    axes[1, 1].hist(fire_df["duration_s"], bins=20, color="#f58518", edgecolor="white")
    style_axis(axes[1, 1])
    axes[1, 1].set_title("Primary Fire Duration")
    axes[1, 1].set_xlabel("Duration (s)")
    axes[1, 1].set_ylabel("Count")

    fire_summary = summary["primary_fire_summary"]
    summary_text = (
        f"Videos: {summary['total_videos']}\n"
        f"Frames: {summary['total_frames']:,}\n"
        f"Duration: {summary['total_duration_s']:.1f}s\n"
        f"Primary fire frames: {fire_summary['frame_count']:,} ({fire_summary['frame_ratio_pct']:.2f}%)\n"
        f"Fire segments: {fire_summary['segment_count']}\n"
        f"Median fire duration: {fire_summary['duration_stats_s']['median']:.3f}s\n"
        f"P90 fire duration: {fire_summary['duration_stats_s']['p90']:.3f}s\n"
        f"Median fire start: {fire_summary['normalized_start_stats']['median']:.3f} of clip"
    )
    fig.text(
        0.735,
        0.23,
        summary_text,
        fontsize=10.5,
        va="top",
        ha="left",
        bbox={"facecolor": "#f8f8f8", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.5"},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    summary, button_df, combo_df, fire_df, timeline_df = load_inputs(analysis_dir)

    save_button_bar(button_df, analysis_dir / "viz_button_share.png")
    save_combo_bar(combo_df, analysis_dir / "viz_top_combos.png")
    save_fire_timeline(timeline_df, analysis_dir / "viz_primary_fire_timeline.png")
    save_fire_duration(fire_df, analysis_dir / "viz_primary_fire_duration.png")
    save_dashboard(summary, button_df, combo_df, timeline_df, fire_df, analysis_dir / "viz_dashboard.png")

    print(f"saved visualizations to {analysis_dir}")


if __name__ == "__main__":
    main()
