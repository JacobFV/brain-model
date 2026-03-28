#!/usr/bin/env python3
"""
Render 3D brain surface visualizations for UX conditions.
Produces:
- Side-by-side brain heatmaps (good vs bad UX)
- Rotating brain GIFs
- Diff brains showing what regions change between conditions
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
sys.path.insert(0, str(_root))

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("clean_ui", "cluttered_ui", "Clean vs Cluttered UI"),
    ("flow_enabling", "flow_breaking", "Flow State vs Interrupted"),
    ("honest_checkout", "dark_checkout", "Honest vs Dark Checkout"),
    ("helpful_error", "cryptic_error", "Helpful vs Cryptic Error"),
    ("graceful_failure", "catastrophic_failure", "Graceful vs Catastrophic Failure"),
    ("meaningful_progress", "manipulative_gamification", "Progress vs Manipulative Gamification"),
    ("good_hierarchy", "bad_hierarchy", "Good vs Bad Information Architecture"),
    ("search_works", "search_fails", "Search Works vs Fails"),
]


def load_predictions():
    """Load pre-computed UX predictions."""
    npz_path = OUTPUT / "ux_predictions.npz"
    if not npz_path.exists():
        # Generate them
        from core.model import load_model, text_to_predictions
        # Import stimuli from the main run script
        print("No cached predictions found, generating...")
        return None
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def render_pair_brains(results, good, bad, title, output_path):
    """Render a 2x3 grid: good (left, right, top) and bad (left, right, top)."""
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")

    good_act = results[good].mean(axis=0)
    bad_act = results[bad].mean(axis=0)

    # Shared color scale
    vmax = max(np.percentile(good_act, 97), np.percentile(bad_act, 97))
    vmin = 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={"projection": "3d"})

    views = ["left", "right", "top"]
    for j, view in enumerate(views):
        plotter.plot_surf(good_act, axes=[axes[0, j]], views=[view], cmap="hot", vmin=vmin, vmax=vmax)
        plotter.plot_surf(bad_act, axes=[axes[1, j]], views=[view], cmap="hot", vmin=vmin, vmax=vmax)

    good_label = good.replace("_", " ").title()
    bad_label = bad.replace("_", " ").title()
    axes[0, 0].set_title(f"{good_label}\n(left)", fontsize=13, fontweight="bold")
    axes[0, 1].set_title(f"{good_label}\n(right)", fontsize=13, fontweight="bold")
    axes[0, 2].set_title(f"{good_label}\n(top)", fontsize=13, fontweight="bold")
    axes[1, 0].set_title(f"{bad_label}\n(left)", fontsize=13, fontweight="bold")
    axes[1, 1].set_title(f"{bad_label}\n(right)", fontsize=13, fontweight="bold")
    axes[1, 2].set_title(f"{bad_label}\n(top)", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def render_diff_brain(results, good, bad, title, output_path):
    """Render the difference brain: what lights up MORE in bad UX (red) vs good (blue)."""
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")

    good_act = results[good].mean(axis=0)
    bad_act = results[bad].mean(axis=0)
    diff = bad_act - good_act

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": "3d"})

    views = ["left", "right", "top"]
    for j, view in enumerate(views):
        plotter.plot_surf(diff, axes=[axes[j]], views=[view], cmap="bwr",
                          norm_percentile=97, symmetric_cbar=True)
        axes[j].set_title(view.title(), fontsize=13, fontweight="bold")

    fig.suptitle(f"{title}\nRed = bad UX more active | Blue = good UX more active",
                 fontsize=16, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def render_rotating_gif(results, condition, output_path, n_frames=36, cmap="hot"):
    """Render a rotating brain GIF for a single condition."""
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    act = results[condition].mean(axis=0)
    vmax = np.percentile(act, 97)

    frames = []
    for i in range(n_frames):
        azim = i * (360 / n_frames)
        elev = 15

        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "3d"})
        plotter.plot_surf(act, axes=[ax], views=[(elev, azim)], cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(condition.replace("_", " ").title(), fontsize=14, fontweight="bold")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    # Save as GIF
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                 duration=100, loop=0, optimize=True)
    print(f"Saved GIF: {output_path} ({len(frames)} frames)")


def render_rotating_diff_gif(results, good, bad, title, output_path, n_frames=36):
    """Render a rotating diff brain GIF."""
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    diff = results[bad].mean(axis=0) - results[good].mean(axis=0)

    frames = []
    for i in range(n_frames):
        azim = i * (360 / n_frames)
        elev = 15

        fig, ax = plt.subplots(figsize=(7, 5.5), subplot_kw={"projection": "3d"})
        plotter.plot_surf(diff, axes=[ax], views=[(elev, azim)], cmap="bwr",
                          norm_percentile=97, symmetric_cbar=True)
        ax.set_title(f"{title}\nRed=bad UX | Blue=good UX", fontsize=12, fontweight="bold")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                 duration=100, loop=0, optimize=True)
    print(f"Saved diff GIF: {output_path} ({len(frames)} frames)")


def render_hero_image(results, output_path):
    """
    Render a big hero image: 4 key conditions side by side with brains.
    """
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")

    conditions = ["clean_ui", "cluttered_ui", "flow_enabling", "dark_checkout"]
    labels = ["Clean UI", "Cluttered UI", "Flow State", "Dark Checkout"]

    all_act = [results[c].mean(axis=0) for c in conditions]
    vmax = max(np.percentile(a, 97) for a in all_act)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), subplot_kw={"projection": "3d"})

    for i, (act, label) in enumerate(zip(all_act, labels)):
        plotter.plot_surf(act, axes=[axes[0, i]], views=["left"], cmap="hot", vmin=0, vmax=vmax)
        plotter.plot_surf(act, axes=[axes[1, i]], views=["right"], cmap="hot", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"{label}\n(left)", fontsize=14, fontweight="bold")
        axes[1, i].set_title(f"(right)", fontsize=12)

    fig.suptitle("Your Brain on UX: How Different Interfaces Light Up the Cortex",
                 fontsize=20, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved hero: {output_path}")


def main():
    results = load_predictions()
    if results is None:
        print("ERROR: No predictions found. Run the brain_ux experiment first.")
        return

    print(f"Loaded {len(results)} conditions")
    print(f"Available: {list(results.keys())}")

    # Hero image
    print("\n=== Rendering hero image ===")
    render_hero_image(results, OUTPUT / "brain_ux_hero.png")

    # Paired comparison brains
    print("\n=== Rendering paired brain comparisons ===")
    for good, bad, title in PAIRS:
        if good in results and bad in results:
            safe_name = f"brain_{good}_vs_{bad}"
            render_pair_brains(results, good, bad, title, OUTPUT / f"{safe_name}.png")
            render_diff_brain(results, good, bad, title, OUTPUT / f"{safe_name}_diff.png")

    # Rotating GIFs for key conditions
    print("\n=== Rendering rotating brain GIFs ===")
    for condition in ["clean_ui", "cluttered_ui", "flow_enabling", "flow_breaking", "dark_checkout"]:
        if condition in results:
            render_rotating_gif(results, condition, OUTPUT / f"rotating_{condition}.gif", n_frames=36)

    # Rotating diff GIFs for the most interesting pairs
    print("\n=== Rendering rotating diff GIFs ===")
    key_pairs = [
        ("clean_ui", "cluttered_ui", "Clean vs Cluttered"),
        ("flow_enabling", "flow_breaking", "Flow vs Interrupted"),
        ("honest_checkout", "dark_checkout", "Honest vs Dark"),
    ]
    for good, bad, title in key_pairs:
        if good in results and bad in results:
            render_rotating_diff_gif(results, good, bad, title,
                                     OUTPUT / f"rotating_diff_{good}_vs_{bad}.gif", n_frames=36)

    print(f"\nAll visualizations saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
