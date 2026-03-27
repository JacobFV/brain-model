#!/usr/bin/env python3
"""
Experiment: Learning what we don't know about brain UX.

Probes TRIBE v2 with descriptions of different UI/UX patterns to discover
how different interface paradigms affect predicted brain activity. Tests:
- Cognitive load: cluttered vs clean UI
- Dark patterns vs honest UX
- Gamification effects on reward circuitry
- Information hierarchy: well-structured vs flat
- Error states and frustration patterns
- Accessibility: screen reader narration vs visual description
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
from core.model import load_model, text_to_predictions, CACHE_FOLDER
from core.roi import get_roi_indices, roi_means

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

STIMULI = {
    # --- Cognitive load ---
    "clean_ui": "The screen shows a single search box centered on a white page. Below it, two buttons: Search and I'm Feeling Lucky. Nothing else competes for your attention. You know exactly what to do.",
    "cluttered_ui": "The page is covered in flashing banners, pop-ups asking you to subscribe, a chat widget bouncing in the corner, auto-playing video with sound, cookie consent covering half the screen, and somewhere buried under all of it is the content you came for.",
    "progressive_disclosure": "The form shows just your name and email at first. After you fill those in, it smoothly reveals the next section: your address. Each step is simple and clear. You never see more than you need.",
    "wall_of_options": "The settings page has two hundred and forty toggles arranged in a flat list with no categories, no search, no defaults highlighted. You need to change your notification preferences but you can't find them anywhere in this ocean of switches.",

    # --- Dark patterns ---
    "honest_unsubscribe": "You click Unsubscribe. A page confirms: You've been unsubscribed. No more emails. There's a link to resubscribe if you change your mind. Done.",
    "dark_unsubscribe": "You click Unsubscribe. The page asks: Are you sure? Your options are Stay Subscribed in bright green, or a tiny gray link that says No, I don't want to receive exclusive deals, savings, and personalized recommendations. Clicking it leads to another page where you must select a reason and then wait fifteen business days.",
    "honest_checkout": "Your cart shows three items, their prices, a clear total, and a single Pay Now button. Tax and shipping are already included in the displayed price.",
    "dark_checkout": "You reach checkout and discover three items you didn't add: insurance, premium support, and a donation to a charity. All pre-checked. The prices changed since you saw them. The total is higher than expected and the Remove buttons are disguised as decorative elements.",

    # --- Information architecture ---
    "good_hierarchy": "The documentation page has a clear table of contents on the left. Each section has a descriptive heading. Code examples are syntax-highlighted. Related topics are linked at the bottom. You find what you need in ten seconds.",
    "bad_hierarchy": "Everything is in one enormous page with no headings, no table of contents, no visual distinction between topics. Code is inline with the prose. Links go to other equally unstructured pages. You scroll endlessly searching for the one paragraph you need.",
    "search_works": "You type your question into the search bar and the first result is exactly what you were looking for. The answer is highlighted in the preview. You click and it scrolls directly to the relevant section.",
    "search_fails": "You type your question and get back hundreds of results, none of which match. You rephrase it four times. Each result page takes three seconds to load. The fifth attempt returns No results found for a query you know should have answers.",

    # --- Error states ---
    "helpful_error": "Something went wrong, but the message explains exactly what happened: Your file is too large. Maximum size is ten megabytes. Your file is twelve megabytes. It offers a Compress and Retry button right there.",
    "cryptic_error": "Error code 0xE8000015. An unknown error occurred. There is no additional information, no suggestion, no link, no way to understand what went wrong or how to fix it. Just OK to dismiss.",
    "graceful_failure": "The network dropped during your upload but the progress bar pauses and a gentle message says: Connection lost. Your upload will resume automatically when you're back online. Your work is saved.",
    "catastrophic_failure": "The screen flashes white and everything is gone. The document you spent three hours writing has vanished. There is no autosave, no recovery, no undo. The app shows a fresh empty page as if nothing happened.",

    # --- Gamification ---
    "meaningful_progress": "You've completed three of seven modules. A progress bar shows your journey and each completed section has a brief summary of what you learned. You can see how far you've come and what's ahead.",
    "manipulative_gamification": "You've earned seventy-three points and unlocked the Bronze Engagement Badge. A countdown timer shows you'll lose your daily streak in two hours and fourteen minutes. A push notification reminds you that your friends have more points than you.",

    # --- Accessibility ---
    "visual_experience": "The dashboard displays a colorful heat map of user activity, with red hotspots showing high engagement and blue areas showing drop-off. A animated line chart trends upward in the corner. Icons represent each metric.",
    "screenreader_experience": "Dashboard. User activity summary. Heading level two. Most active region: North America, eight thousand four hundred sessions. Least active: Antarctica, zero sessions. Trend: up twelve percent month over month. Table with five rows and three columns follows.",

    # --- Flow state ---
    "flow_enabling": "The code editor fills the entire screen. Autocomplete suggestions appear instantly as you type. The cursor moves smoothly. There are no notifications, no distractions. Your test results update in real time in a slim panel at the bottom.",
    "flow_breaking": "Every two minutes a dialog interrupts your work to ask if you want to save. The autocomplete takes three seconds to appear and covers the code you're reading. A red notification badge shows forty-seven unread messages in the corner of your eye.",
}


def analyze_ux_patterns(results: dict[str, np.ndarray], roi_idx: dict):
    """Analyze brain response patterns across UX conditions."""
    profiles = {}
    for name, preds in results.items():
        profiles[name] = roi_means(preds, roi_idx)

    emotions = list(profiles.keys())
    rois = list(roi_idx.keys())
    matrix = np.array([[profiles[e][r] for r in rois] for e in emotions])

    analysis = {}

    # Paired comparisons
    pairs = [
        ("clean_ui", "cluttered_ui", "Cognitive Load"),
        ("honest_unsubscribe", "dark_unsubscribe", "Dark Pattern (Unsubscribe)"),
        ("honest_checkout", "dark_checkout", "Dark Pattern (Checkout)"),
        ("good_hierarchy", "bad_hierarchy", "Information Architecture"),
        ("search_works", "search_fails", "Search Experience"),
        ("helpful_error", "cryptic_error", "Error Messaging"),
        ("graceful_failure", "catastrophic_failure", "Failure Handling"),
        ("meaningful_progress", "manipulative_gamification", "Gamification"),
        ("visual_experience", "screenreader_experience", "Accessibility Mode"),
        ("flow_enabling", "flow_breaking", "Flow State"),
        ("progressive_disclosure", "wall_of_options", "Complexity Management"),
    ]

    paired_results = {}
    for good, bad, label in pairs:
        if good in profiles and bad in profiles:
            good_vals = np.array([profiles[good][r] for r in rois])
            bad_vals = np.array([profiles[bad][r] for r in rois])
            diff = bad_vals - good_vals
            paired_results[label] = {
                "roi_diffs": {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))},
                "overall_activation_good": float(good_vals.mean()),
                "overall_activation_bad": float(bad_vals.mean()),
                "activation_increase_pct": round(float((bad_vals.mean() - good_vals.mean()) / (abs(good_vals.mean()) + 1e-10) * 100), 2),
                "most_affected_roi": rois[int(np.argmax(np.abs(diff)))],
                "max_diff": float(diff[np.argmax(np.abs(diff))]),
            }

    analysis["paired_comparisons"] = paired_results

    # Correlation between all UX conditions
    corr = np.corrcoef(matrix)
    analysis["condition_correlations"] = {
        emotions[i]: {emotions[j]: round(float(corr[i, j]), 4) for j in range(len(emotions))}
        for i in range(len(emotions))
    }

    # Dark patterns: do they consistently activate specific regions?
    dark = ["dark_unsubscribe", "dark_checkout", "manipulative_gamification"]
    honest = ["honest_unsubscribe", "honest_checkout", "meaningful_progress"]
    if all(e in profiles for e in dark + honest):
        dark_mean = np.mean([matrix[emotions.index(e)] for e in dark], axis=0)
        honest_mean = np.mean([matrix[emotions.index(e)] for e in honest], axis=0)
        diff = dark_mean - honest_mean
        analysis["dark_pattern_signature"] = {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))}

    # Frustration signature: cryptic error + search fails + catastrophic failure + cluttered
    frustrating = ["cryptic_error", "search_fails", "catastrophic_failure", "cluttered_ui", "wall_of_options"]
    pleasant = ["helpful_error", "search_works", "graceful_failure", "clean_ui", "progressive_disclosure"]
    if all(e in profiles for e in frustrating + pleasant):
        frust_mean = np.mean([matrix[emotions.index(e)] for e in frustrating], axis=0)
        pleas_mean = np.mean([matrix[emotions.index(e)] for e in pleasant], axis=0)
        diff = frust_mean - pleas_mean
        analysis["frustration_signature"] = {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))}

    return analysis, matrix, emotions, rois


# ── HIX Proxy Signals ─────────────────────────────────────────────────────
# Human Interface eXperience proxy signals derived from brain ROI activations.
# These can serve as reward/cost functions for closed-loop autonomous design.

HIX_SIGNALS = {
    "cognitive_load": {
        "description": "Mental effort required to process the interface",
        "positive_rois": ["Frontal sup.", "Frontal mid.", "Broca's area", "Angular/TPJ"],
        "negative_rois": ["Precuneus", "Cingulate post."],
        "interpretation": "high = hard to process, low = effortless",
    },
    "frustration": {
        "description": "Negative affect from blocked goals or confusion",
        "positive_rois": ["Insula", "Cingulate ant.", "Frontal mid."],
        "negative_rois": ["Precuneus", "Cingulate post."],
        "interpretation": "high = frustrated/confused, low = satisfied",
    },
    "engagement": {
        "description": "Sustained attention and interest",
        "positive_rois": ["Wernicke's area", "Temporal mid.", "Frontal sup.", "Fusiform (FFA)"],
        "negative_rois": [],
        "interpretation": "high = engaged, low = disengaged/bored",
    },
    "reward": {
        "description": "Positive affect, satisfaction, delight",
        "positive_rois": ["Orbital frontal", "Cingulate ant.", "Temporal pole"],
        "negative_rois": ["Insula"],
        "interpretation": "high = delighted, low = indifferent",
    },
    "confusion": {
        "description": "Mismatch between expectation and experience",
        "positive_rois": ["Cingulate ant.", "Frontal mid.", "Insula"],
        "negative_rois": ["Wernicke's area", "Temporal mid."],
        "interpretation": "high = confused/lost, low = clear understanding",
    },
    "flow": {
        "description": "Optimal state of effortless focused engagement",
        "positive_rois": ["Frontal sup.", "Broca's area", "Motor", "Somatosensory"],
        "negative_rois": ["Angular/TPJ", "Precuneus", "Cingulate ant."],
        "interpretation": "high = in flow, low = distracted or self-conscious",
    },
    "trust": {
        "description": "Sense of safety and reliability of the interface",
        "positive_rois": ["Temporal pole", "Cingulate post.", "Precuneus"],
        "negative_rois": ["Insula", "Cingulate ant."],
        "interpretation": "high = trusting, low = suspicious/wary",
    },
    "spatial_clarity": {
        "description": "Ease of navigating and orienting within the interface",
        "positive_rois": ["Parahipp. (PPA)", "Occipital", "Angular/TPJ"],
        "negative_rois": ["Frontal mid.", "Insula"],
        "interpretation": "high = clear spatial model, low = disoriented",
    },
}


def compute_hix_signals(preds: np.ndarray, roi_idx: dict) -> dict[str, float]:
    """Compute all HIX proxy signals from brain predictions."""
    activation = roi_means(preds, roi_idx)
    signals = {}
    for sig_name, sig_def in HIX_SIGNALS.items():
        pos = sum(activation.get(r, 0) for r in sig_def["positive_rois"])
        neg = sum(activation.get(r, 0) for r in sig_def["negative_rois"])
        signals[sig_name] = float(pos - neg)
    return signals


def compute_hix_for_all(results: dict[str, np.ndarray], roi_idx: dict) -> dict[str, dict[str, float]]:
    """Compute HIX signals for every UX condition."""
    return {name: compute_hix_signals(preds, roi_idx) for name, preds in results.items()}


def plot_hix_dashboard(hix_all: dict, output_path):
    """Plot HIX signal dashboard: heatmap of all signals × conditions."""
    conditions = sorted(hix_all.keys())
    signals = sorted(HIX_SIGNALS.keys())
    matrix = np.array([[hix_all[c][s] for s in signals] for c in conditions])

    # Normalize per-signal for visual comparison
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        rng = col.max() - col.min()
        if rng > 0:
            matrix[:, j] = (col - col.min()) / rng

    fig, ax = plt.subplots(figsize=(14, max(10, len(conditions) * 0.4)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(signals)))
    ax.set_xticklabels([s.replace("_", "\n") for s in signals], fontsize=9, rotation=0)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c.replace("_", " ") for c in conditions], fontsize=8)
    plt.colorbar(im, ax=ax, label="Normalized signal (0=best, 1=worst for neg signals)")
    ax.set_title("HIX Proxy Signals: Human Interface Experience Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_hix_paired(hix_all: dict, output_path):
    """Show HIX signal differences for good vs bad UX pairs."""
    pairs = [
        ("clean_ui", "cluttered_ui", "Clean vs Cluttered"),
        ("honest_checkout", "dark_checkout", "Honest vs Dark Checkout"),
        ("helpful_error", "cryptic_error", "Helpful vs Cryptic Error"),
        ("flow_enabling", "flow_breaking", "Flow vs Interrupted"),
        ("good_hierarchy", "bad_hierarchy", "Good vs Bad IA"),
        ("search_works", "search_fails", "Search Works vs Fails"),
        ("meaningful_progress", "manipulative_gamification", "Meaningful vs Manipulative"),
        ("graceful_failure", "catastrophic_failure", "Graceful vs Catastrophic"),
    ]

    signals = sorted(HIX_SIGNALS.keys())
    valid_pairs = [(g, b, l) for g, b, l in pairs if g in hix_all and b in hix_all]
    if not valid_pairs:
        return

    fig, axes = plt.subplots(len(valid_pairs), 1, figsize=(14, 3.5 * len(valid_pairs)))
    if len(valid_pairs) == 1:
        axes = [axes]

    for ax, (good, bad, label) in zip(axes, valid_pairs):
        good_vals = [hix_all[good][s] for s in signals]
        bad_vals = [hix_all[bad][s] for s in signals]
        x = np.arange(len(signals))
        w = 0.35
        ax.bar(x - w / 2, good_vals, w, label="Good UX", color="#4daf4a", alpha=0.8)
        ax.bar(x + w / 2, bad_vals, w, label="Bad UX", color="#e41a1c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in signals], fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.3)

    fig.suptitle("HIX Signals: Good vs Bad UX\n(reward signals for autonomous design)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_paired_comparisons(analysis, output_path):
    """Plot ROI differences for each paired UX comparison."""
    pairs = analysis.get("paired_comparisons", {})
    if not pairs:
        return

    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, pairs.items()):
        diffs = data["roi_diffs"]
        rois = list(diffs.keys())
        vals = [diffs[r] for r in rois]
        colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
        ax.barh(range(len(rois)), vals, color=colors)
        ax.set_yticks(range(len(rois)))
        ax.set_yticklabels(rois, fontsize=7)
        ax.set_title(f"{label} (red=bad UX higher, blue=good UX higher)", fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle("Brain Response: Bad UX vs Good UX by Region", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    model = load_model()
    roi_idx = get_roi_indices()

    results = {}
    for name, text in STIMULI.items():
        preds, _ = text_to_predictions(model, text, label=name)
        results[name] = preds

    print("\n=== Analyzing UX patterns ===")
    analysis, matrix, conditions, rois = analyze_ux_patterns(results, roi_idx)

    with open(OUTPUT / "ux_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    np.savez(OUTPUT / "ux_predictions.npz", **{k: v for k, v in results.items()})

    plot_paired_comparisons(analysis, OUTPUT / "ux_paired_comparisons.png")

    # === HIX Proxy Signals ===
    print("\n=== Computing HIX Proxy Signals ===")
    hix_all = compute_hix_for_all(results, roi_idx)

    with open(OUTPUT / "hix_signals.json", "w") as f:
        json.dump({
            "signals": hix_all,
            "signal_definitions": {k: {kk: vv for kk, vv in v.items()}
                                   for k, v in HIX_SIGNALS.items()},
        }, f, indent=2, default=str)

    plot_hix_dashboard(hix_all, OUTPUT / "hix_dashboard.png")
    plot_hix_paired(hix_all, OUTPUT / "hix_paired.png")

    # Print HIX summary
    print("\nHIX Signal Summary (selected conditions):")
    for condition in ["clean_ui", "cluttered_ui", "flow_enabling", "flow_breaking",
                      "dark_checkout", "honest_checkout"]:
        if condition in hix_all:
            sigs = hix_all[condition]
            print(f"\n  {condition}:")
            for sig, val in sorted(sigs.items()):
                interp = HIX_SIGNALS[sig]["interpretation"]
                print(f"    {sig:20s} = {val:+.4f}  ({interp})")

    # === Original findings ===
    print("\n=== KEY FINDINGS ===")
    for label, data in analysis.get("paired_comparisons", {}).items():
        pct = data["activation_increase_pct"]
        roi = data["most_affected_roi"]
        direction = "more" if data["max_diff"] > 0 else "less"
        print(f"  {label}: bad UX {pct:+.1f}% overall activation, {roi} most affected ({direction} active)")

    if "dark_pattern_signature" in analysis:
        print("\nDark pattern signature (regions more active):")
        items = sorted(analysis["dark_pattern_signature"].items(), key=lambda x: -x[1])
        for roi, val in items[:5]:
            print(f"  +{val:.6f}  {roi}")

    if "frustration_signature" in analysis:
        print("\nFrustration signature (regions more active):")
        items = sorted(analysis["frustration_signature"].items(), key=lambda x: -x[1])
        for roi, val in items[:5]:
            print(f"  +{val:.6f}  {roi}")


if __name__ == "__main__":
    main()
