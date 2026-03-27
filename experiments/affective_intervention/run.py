#!/usr/bin/env python3
"""
Experiment: Audiovisuosemantic Affective Intervention

Phase 1 — Entrainment: Find stimulus trajectories that REDUCE variance in
          brain state across diverse initial conditions ("funnel" the brain
          into a known state without needing to measure it)

Phase 2 — Intervention: Given a known brain state, optimize stimulus
          to drive the brain toward a target affective signature

Phase 3 — End-to-end: Concatenate entrainment + intervention into a single
          stimulus trajectory that produces desired affect regardless of
          initial conditions

Key insight: We don't need an EEG to know the brain's state if we can
design stimuli that collapse the state space first.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
from core.model import load_model, text_to_predictions, CACHE_FOLDER
from core.roi import get_roi_indices, roi_means

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Affective targets: desired brain region activation profiles ───────────

# Defined as relative weights on ROIs (higher = we want more activation there)
AFFECTIVE_TARGETS = {
    "calm": {
        "Cingulate post.": 1.0, "Precuneus": 0.8, "Frontal sup.": 0.3,
        "Temporal mid.": 0.2, "Insula": -0.5, "Motor": -0.3,
    },
    "focused": {
        "Frontal sup.": 1.0, "Frontal mid.": 0.8, "Broca's area": 0.6,
        "Angular/TPJ": 0.4, "Precuneus": -0.3, "Temporal pole": -0.2,
    },
    "creative": {
        "Angular/TPJ": 1.0, "Precuneus": 0.8, "Temporal mid.": 0.7,
        "Frontal mid.": 0.5, "Cingulate ant.": 0.4,
    },
    "empathetic": {
        "Angular/TPJ": 1.0, "Insula": 0.9, "Cingulate ant.": 0.8,
        "Temporal pole": 0.7, "Frontal sup.": 0.4,
    },
}

# ── Diverse initial conditions (different starting mental states) ─────────

INITIAL_CONDITIONS = {
    "resting": "You are sitting quietly in a comfortable chair with your eyes closed, breathing slowly, thinking of nothing in particular.",
    "anxious": "Your heart is racing. The exam is in one hour and you haven't studied enough. Your mind keeps jumping between topics, unable to focus.",
    "angry": "Someone just cut in front of you in line after you waited forty minutes. They pretended not to see you. Your jaw is clenched.",
    "sad": "You just watched the last person leave the funeral. The house is empty. Everything is too quiet.",
    "excited": "You just got the call. You got the job. You want to tell everyone. Your body is buzzing with energy.",
    "bored": "The meeting has been going on for two hours. Someone is reading from a spreadsheet. You've counted the ceiling tiles three times.",
    "focused_work": "You are deep in a coding problem, the solution almost within reach. Your attention is narrow and intense.",
    "daydreaming": "You're staring out the window at clouds drifting past. Your mind wanders freely between half-formed thoughts and memories.",
}

# ── Candidate entrainment stimuli ─────────────────────────────────────────

ENTRAINMENT_CANDIDATES = {
    "breath_focus": "Breathe in slowly through your nose for four counts. Hold for four counts. Breathe out through your mouth for six counts. Feel your chest rise and fall. Focus only on the breath. Nothing else matters right now. In. Hold. Out. In. Hold. Out.",
    "body_scan": "Notice your feet on the ground. Feel their weight. Now move your attention to your ankles. Your calves. Feel the chair beneath your thighs. Your hands resting in your lap. Your shoulders. Let each part of your body relax as you notice it.",
    "counting_anchor": "Count backwards from ten. Ten. Nine. Eight. With each number, let your thoughts settle. Seven. Six. Five. Feel your mind becoming clearer. Four. Three. Two. One. Now you are here, fully present, ready.",
    "sensory_ground": "Notice five things you can see. Four things you can touch. Three things you can hear. Two things you can smell. One thing you can taste. You are anchored in the present moment through your senses.",
    "story_hook": "Imagine a door. It is old and wooden, painted blue, with a brass handle worn smooth by a thousand hands. Behind it is a room you have never seen but somehow remember. You reach for the handle.",
    "rhythm_entrain": "Tap. Tap. Tap. A steady rhythm, like a heartbeat. One beat per second. Let your breathing synchronize. Let your thoughts synchronize. The rhythm is all there is. Tap. Tap. Tap. Everything else fades.",
}


def measure_state_variance(model, stimuli: dict[str, str]) -> tuple[np.ndarray, dict]:
    """
    Run diverse initial conditions through the model and measure
    the variance in resulting brain states.
    """
    states = {}
    for name, text in stimuli.items():
        preds, _ = text_to_predictions(model, text, label=name)
        states[name] = preds.mean(axis=0)  # mean activation (n_vertices,)

    state_matrix = np.stack(list(states.values()))  # (N, n_vertices)
    variance = state_matrix.var(axis=0)  # per-vertex variance across conditions
    return variance, states


def evaluate_entrainment(model, initial_conditions: dict, entrainment_stimuli: dict):
    """
    For each entrainment stimulus, measure how much it reduces
    variance in brain state across diverse initial conditions.

    We simulate: initial_condition_text + entrainment_text
    and measure variance in the FINAL brain state.
    """
    results = {}

    # Baseline: variance from initial conditions alone
    baseline_var, _ = measure_state_variance(model, initial_conditions)
    baseline_total = float(baseline_var.sum())
    print(f"Baseline state variance (no entrainment): {baseline_total:.4f}")

    for ent_name, ent_text in entrainment_stimuli.items():
        # Combine each initial condition with the entrainment stimulus
        combined = {}
        for ic_name, ic_text in initial_conditions.items():
            combined_text = f"{ic_text} {ent_text}"
            combined[ic_name] = combined_text

        post_var, post_states = measure_state_variance(model, combined)
        post_total = float(post_var.sum())
        reduction = (1 - post_total / baseline_total) * 100

        # Per-ROI variance reduction
        roi_idx = get_roi_indices()
        roi_reduction = {}
        for roi_name, idx in roi_idx.items():
            roi_base = float(baseline_var[idx].mean())
            roi_post = float(post_var[idx].mean())
            roi_reduction[roi_name] = {
                "before": roi_base,
                "after": roi_post,
                "reduction_pct": round((1 - roi_post / (roi_base + 1e-10)) * 100, 2),
            }

        results[ent_name] = {
            "total_variance": post_total,
            "variance_reduction_pct": round(reduction, 2),
            "roi_reduction": roi_reduction,
        }
        print(f"  {ent_name}: variance={post_total:.4f} (reduction: {reduction:.1f}%)")

    return results, baseline_total


def evaluate_affective_steering(model, target_name: str, target_weights: dict, candidate_texts: dict):
    """
    Score how well each candidate text moves the brain toward
    the target affective profile.
    """
    roi_idx = get_roi_indices()

    # Normalize target weights
    target_rois = {k: v for k, v in target_weights.items() if k in roi_idx}
    if not target_rois:
        return {}

    scores = {}
    for name, text in candidate_texts.items():
        preds, _ = text_to_predictions(model, text, label=f"{target_name}:{name}")
        activation = roi_means(preds, roi_idx)

        # Score: weighted sum of activations matching target
        score = sum(activation.get(roi, 0) * weight for roi, weight in target_rois.items())
        scores[name] = {
            "affective_score": float(score),
            "roi_activations": {k: round(v, 6) for k, v in activation.items()},
        }

    return scores


def plot_entrainment_results(results, baseline_var, output_dir):
    """Visualize entrainment effectiveness."""
    names = list(results.keys())
    reductions = [results[n]["variance_reduction_pct"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in reductions]
    bars = ax.bar(range(len(names)), reductions, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Variance reduction (%)")
    ax.set_title("Entrainment Effectiveness:\nHow much does each stimulus reduce brain state uncertainty?",
                 fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)

    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "entrainment_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'entrainment_effectiveness.png'}")


def plot_affective_scores(all_scores, output_dir):
    """Plot affective steering scores for each target."""
    fig, axes = plt.subplots(1, len(all_scores), figsize=(6 * len(all_scores), 8))
    if len(all_scores) == 1:
        axes = [axes]

    for ax, (target, scores) in zip(axes, all_scores.items()):
        names = list(scores.keys())
        vals = [scores[n]["affective_score"] for n in names]
        sorted_idx = np.argsort(vals)[::-1]
        names = [names[i] for i in sorted_idx]
        vals = [vals[i] for i in sorted_idx]

        ax.barh(range(len(names)), vals, color="#377eb8")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
        ax.set_title(f"Target: {target}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Affective alignment score")

    fig.suptitle("Which stimuli best steer the brain toward desired states?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "affective_steering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'affective_steering.png'}")


def main():
    model = load_model()

    # === Phase 1: Entrainment ===
    print("=== PHASE 1: ENTRAINMENT ===")
    print("Testing which stimuli best collapse brain state variance...\n")

    ent_results, baseline_var = evaluate_entrainment(
        model, INITIAL_CONDITIONS, ENTRAINMENT_CANDIDATES
    )

    # Rank by effectiveness
    ranked = sorted(ent_results.items(), key=lambda x: -x[1]["variance_reduction_pct"])
    print(f"\nEntrainment ranking:")
    for name, data in ranked:
        print(f"  {data['variance_reduction_pct']:+6.1f}%  {name}")

    best_entrainment = ranked[0][0]
    print(f"\nBest entrainment stimulus: {best_entrainment}")

    plot_entrainment_results(ent_results, baseline_var, OUTPUT)

    # === Phase 2: Affective Steering ===
    print("\n=== PHASE 2: AFFECTIVE STEERING ===")

    # Generate candidate intervention texts for each target
    INTERVENTION_CANDIDATES = {
        "guided_relaxation": "Your muscles soften. Your breath deepens. A warm wave of calm washes through your body from head to toe. Everything slows down. You are safe. You are at peace.",
        "focus_prompt": "Clear your mind. There is one task in front of you. It deserves your full attention. Block out everything else. The world narrows to just this moment, this thought, this action.",
        "creative_spark": "What if the sky were green and the grass were blue? What if music could be tasted and colors could be heard? Let your mind wander to impossible places. There are no wrong answers here.",
        "empathy_bridge": "Imagine being someone else for a moment. Feel what they feel. See through their eyes. They have fears and hopes just like you. Their heart beats the same rhythm as yours.",
        "nature_calm": "A still mountain lake reflects the sky perfectly. Pine trees line the shore. The only sound is birdsong echoing across the water. Time moves slowly here.",
        "energize": "Stand up. Feel the energy in your legs. Take a deep breath and clap your hands together. You are alive. You are capable. Today is yours to shape.",
        "wonder": "Look up at the night sky. Each star is a sun, many with planets of their own. The light reaching your eyes has been traveling for thousands of years. You are seeing the past.",
        "gratitude": "Think of one person who made your life better. Remember their face. Remember something kind they did. Let the warmth of that memory fill your chest.",
    }

    all_affective_scores = {}
    for target_name, target_weights in AFFECTIVE_TARGETS.items():
        print(f"\nEvaluating interventions for target: {target_name}")
        scores = evaluate_affective_steering(model, target_name, target_weights, INTERVENTION_CANDIDATES)
        all_affective_scores[target_name] = scores

        ranked = sorted(scores.items(), key=lambda x: -x[1]["affective_score"])
        for name, data in ranked[:3]:
            print(f"  {data['affective_score']:.4f}  {name}")

    plot_affective_scores(all_affective_scores, OUTPUT)

    # === Phase 3: End-to-end trajectory ===
    print("\n=== PHASE 3: END-TO-END TRAJECTORY ===")

    # For each affective target, combine best entrainment + best intervention
    for target_name, target_weights in AFFECTIVE_TARGETS.items():
        scores = all_affective_scores[target_name]
        best_intervention = max(scores.items(), key=lambda x: x[1]["affective_score"])[0]

        ent_text = ENTRAINMENT_CANDIDATES[best_entrainment]
        int_text = INTERVENTION_CANDIDATES[best_intervention]

        combined = f"{ent_text} {int_text}"
        preds, _ = text_to_predictions(model, combined, label=f"e2e_{target_name}")
        activation = roi_means(preds)

        target_rois = {k: v for k, v in target_weights.items() if k in activation}
        score = sum(activation.get(roi, 0) * weight for roi, weight in target_rois.items())

        print(f"  {target_name}: entrainment='{best_entrainment}' + intervention='{best_intervention}'")
        print(f"    End-to-end affective score: {score:.4f}")

    # Save everything
    summary = {
        "entrainment_results": {k: {kk: vv for kk, vv in v.items() if kk != "roi_reduction"}
                                for k, v in ent_results.items()},
        "best_entrainment": best_entrainment,
        "affective_targets": AFFECTIVE_TARGETS,
        "affective_scores": {t: {n: d["affective_score"] for n, d in scores.items()}
                             for t, scores in all_affective_scores.items()},
    }
    with open(OUTPUT / "affective_intervention_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary -> {OUTPUT / 'affective_intervention_summary.json'}")


if __name__ == "__main__":
    main()
