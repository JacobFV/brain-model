#!/usr/bin/env python3
"""
Experiment: Virtual EEG Headset (v2 — physically grounded)

Simulates EEG by sampling TRIBE v2 cortical predictions at locations
corresponding to real 10-20 system electrode placements. Each electrode
reads the average activation of a ~10mm radius cortical patch beneath it.

This gives us a (n_timesteps, n_electrodes) signal per stimulus —
a physically meaningful simulated EEG at fMRI temporal resolution.

Key questions:
- How much semantic information survives electrode-level spatial sampling?
- Which electrodes carry the most category-discriminative information?
- How does electrode count (full 10-20 vs subsets) affect classification?
- Does temporal structure help vs just using mean activation?
- Baseline: does the brain add anything beyond raw text embeddings?
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
from core.model import load_model, text_to_predictions, CACHE_FOLDER

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Standard 10-20 electrode positions ────────────────────────────────────

ELECTRODES_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2", "Oz",
]

# Which brain region each electrode roughly corresponds to
ELECTRODE_REGIONS = {
    "Fp1": "prefrontal L", "Fp2": "prefrontal R",
    "F7": "frontal-temporal L", "F3": "frontal L", "Fz": "frontal midline",
    "F4": "frontal R", "F8": "frontal-temporal R",
    "T3": "temporal L", "C3": "central L", "Cz": "central midline",
    "C4": "central R", "T4": "temporal R",
    "T5": "posterior temporal L", "P3": "parietal L", "Pz": "parietal midline",
    "P4": "parietal R", "T6": "posterior temporal R",
    "O1": "occipital L", "O2": "occipital R", "Oz": "occipital midline",
}

PATCH_RADIUS_MM = 10.0  # ~1cm radius per electrode patch


def build_electrode_patches(radius_mm: float = PATCH_RADIUS_MM) -> dict[str, np.ndarray]:
    """Map each 10-20 electrode to a cortical surface patch on fsaverage5."""
    import mne
    from nilearn import datasets
    import nibabel as nib

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    coords_lh = nib.load(fsaverage["pial_left"]).agg_data()[0]
    coords_rh = nib.load(fsaverage["pial_right"]).agg_data()[0]
    all_coords = np.vstack([coords_lh, coords_rh])

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]

    patches = {}
    for name in ELECTRODES_1020:
        if name not in ch_pos:
            continue
        scalp_pos = ch_pos[name] * 1000  # m → mm

        # Project scalp position to nearest cortical vertex
        dists = np.linalg.norm(all_coords - scalp_pos, axis=1)
        nearest = np.argmin(dists)

        # Get patch: all vertices within radius of nearest
        cortex_pos = all_coords[nearest]
        patch_dists = np.linalg.norm(all_coords - cortex_pos, axis=1)
        patch = np.where(patch_dists <= radius_mm)[0]
        patches[name] = patch

    return patches


def sample_eeg(preds: np.ndarray, patches: dict[str, np.ndarray]) -> np.ndarray:
    """
    Sample virtual EEG from brain predictions.
    preds: (n_timesteps, n_vertices) from TRIBE v2
    Returns: (n_timesteps, n_electrodes) — the simulated EEG signal
    """
    electrode_names = sorted(patches.keys())
    n_t = preds.shape[0]
    n_e = len(electrode_names)
    eeg = np.zeros((n_t, n_e))

    for j, name in enumerate(electrode_names):
        patch_idx = patches[name]
        eeg[:, j] = preds[:, patch_idx].mean(axis=1)

    return eeg


# ── Stimulus corpus ───────────────────────────────────────────────────────
# More stimuli per category for proper cross-validation

CATEGORIES = {
    "animal": [
        "A golden retriever bounding across a field, ears flapping, tongue out, chasing a frisbee through the grass",
        "An eagle circling high above a mountain valley, wings barely moving, riding the thermals in silence",
        "A cat curled up on a warm windowsill, purring, eyes half closed, watching rain slide down the glass",
        "A whale breaching the surface in a spray of white water, crashing back down with a thunderous splash",
        "A spider spinning a web between two branches, each thread catching the morning dew like tiny diamonds",
        "A horse galloping along a beach at sunset, hooves splashing through shallow waves, mane flying",
        "A swarm of fireflies blinking in a dark meadow, each one a tiny beacon drifting through the warm night air",
        "Two wolves howling together on a ridge under a full moon, their voices carrying across the frozen valley",
    ],
    "music": [
        "A solo violin playing a haunting melody in an empty cathedral, each note echoing off ancient stone walls",
        "Heavy drums pounding a tribal rhythm, the beat so deep you feel it vibrating in your chest and teeth",
        "A jazz piano improvising over a walking bass line, notes tumbling out in unexpected cascading runs",
        "An orchestra building toward a massive crescendo, every instrument adding to a wall of overwhelming sound",
        "A guitar playing fingerpicked arpeggios by a campfire, gentle and warm, the wood crackling between phrases",
        "Electronic music with a deep bass drop, the sub-frequencies shaking the floor of a dark nightclub",
        "A choir singing in perfect harmony, voices layered so tightly they seem to become a single instrument",
        "A music box playing a simple lullaby, the tiny metallic notes pinging in a quiet dark room",
    ],
    "danger": [
        "A rattlesnake coiled on the path, its rattle buzzing as you freeze mid-step, heart suddenly pounding",
        "The car ahead swerves and you slam the brakes, tires screaming on wet asphalt, everything in slow motion",
        "Smoke pouring under the hotel room door at three in the morning, the fire alarm shrieking overhead",
        "A crack in the ice shooting across the frozen lake beneath your feet, the groan of something about to give",
        "Lightning striking a tree twenty feet away with a deafening crack, the air tasting of ozone and fear",
        "The elevator cable snapping, the sudden lurch downward, the lights flickering, a moment of weightlessness",
        "A massive wave rising behind the boat, dark green and curling, blocking out the entire horizon",
        "Walking through a dark alley and hearing footsteps behind you that match your pace exactly",
    ],
    "spatial": [
        "Standing at the edge of the Grand Canyon, looking down a thousand feet of layered red stone to the river",
        "A cathedral ceiling soaring a hundred feet overhead, light filtering through stained glass in colored beams",
        "Floating in the middle of a dark ocean at night, no land visible in any direction, stars reflected below",
        "A narrow tunnel deep underground, barely wide enough to crawl through, stone pressing in on all sides",
        "The view from the top of a skyscraper, the city spread out below like a circuit board of lights and streets",
        "An enormous empty warehouse, footsteps echoing, your voice bouncing back from walls you can barely see",
        "A dense forest where the canopy blocks all sunlight, the trunks forming a maze with no visible path",
        "Standing on the wing of a plane on the tarmac, looking out at the flat expanse of runway stretching to the horizon",
    ],
    "social": [
        "Your best friend calling at midnight because they need to talk, and you sit on the kitchen floor and listen",
        "Standing at a podium giving a speech to five hundred people, every single pair of eyes fixed on you",
        "A baby gripping your finger for the first time, tiny hand wrapped tight, looking up at your face",
        "A stranger on the bus smiling at you for no reason, and that smile lifting your whole morning",
        "Two old men on a park bench playing chess in complete silence, decades of friendship needing no words",
        "Walking into a surprise party and thirty people shouting your name, faces you love all in one room",
        "Holding someone while they cry, not saying anything, just being there, their shoulders shaking against yours",
        "A job interview where the panel stares at you in silence for ten seconds after your answer",
    ],
    "language": [
        "Reading a poem where every word is common but the arrangement unlocks a meaning you never had words for before",
        "The word petrichor, meaning the smell of rain on dry earth, a word for something you always knew but could never name",
        "Someone speaking a language you have never heard, melodic and rhythmic, and understanding their meaning from gesture alone",
        "The frustration of knowing exactly what you mean but the right word hovering just beyond your reach, refusing to come",
        "Reading a letter from a hundred years ago and feeling the writer's personality come alive through their handwriting",
        "Two people finishing each other's sentences, their thoughts synchronized, a conversation that flows like one mind",
        "A child making up a word for something that doesn't have one, and the word being somehow perfect",
        "Translating a joke into another language and watching the humor evaporate completely despite accurate translation",
    ],
}


def generate_dataset(model, patches):
    """Generate virtual EEG dataset: (n_samples, n_timesteps, n_electrodes)."""
    category_names = sorted(CATEGORIES.keys())
    all_eeg_mean = []     # mean EEG per stimulus (n_electrodes,)
    all_eeg_temporal = [] # full temporal EEG (n_timesteps, n_electrodes)
    all_labels = []
    all_texts = []
    all_raw_preds = []    # full vertex predictions for baseline

    for cat_idx, cat_name in enumerate(category_names):
        for text in CATEGORIES[cat_name]:
            preds, _ = text_to_predictions(model, text, label=cat_name)
            eeg = sample_eeg(preds, patches)

            all_eeg_mean.append(eeg.mean(axis=0))       # (n_electrodes,)
            all_eeg_temporal.append(eeg)                  # (T, n_electrodes)
            all_raw_preds.append(preds.mean(axis=0))      # (n_vertices,)
            all_labels.append(cat_idx)
            all_texts.append(text)

    X_mean = np.stack(all_eeg_mean)       # (N, n_electrodes)
    X_raw = np.stack(all_raw_preds)       # (N, n_vertices)
    y = np.array(all_labels)

    return X_mean, all_eeg_temporal, X_raw, y, all_texts, category_names


def evaluate_with_loocv(X, y, name=""):
    """Leave-one-out cross-validation with SVM."""
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")
    acc = scores.mean()
    print(f"  {name}: LOOCV accuracy = {acc:.1%} ({scores.sum():.0f}/{len(scores)} correct)")
    return acc


def evaluate_electrode_importance(X_mean, y, electrode_names):
    """Test each electrode individually and in ablation."""
    single_scores = {}
    for j, name in enumerate(electrode_names):
        X_single = X_mean[:, j:j+1]
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
        loo = LeaveOneOut()
        scores = cross_val_score(clf, X_single, y, cv=loo, scoring="accuracy")
        single_scores[name] = float(scores.mean())

    return dict(sorted(single_scores.items(), key=lambda x: -x[1]))


def evaluate_temporal_features(eeg_temporal, y):
    """Test if temporal features (std, slope, peak time) help beyond mean."""
    features = []
    for eeg in eeg_temporal:
        # eeg: (T, n_electrodes)
        mean = eeg.mean(axis=0)
        std = eeg.std(axis=0)
        peak_time = eeg.argmax(axis=0) / max(eeg.shape[0], 1)
        slope = np.polyfit(range(eeg.shape[0]), eeg.mean(axis=1), 1)[0] if eeg.shape[0] > 1 else 0
        features.append(np.concatenate([mean, std, peak_time, [slope]]))

    X_temporal = np.stack(features)
    return evaluate_with_loocv(X_temporal, y, name="EEG mean+std+peak+slope")


def plot_electrode_importance(single_scores, output_path):
    """Plot which electrodes carry the most information."""
    names = list(single_scores.keys())
    scores = [single_scores[n] for n in names]
    chance = 1 / 6  # 6 categories

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#2ecc71" if s > chance * 1.5 else "#95a5a6" for s in scores]
    bars = ax.barh(range(len(names)), scores, color=colors)
    ax.axvline(chance, color="red", linestyle="--", linewidth=1, label=f"Chance ({chance:.1%})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({ELECTRODE_REGIONS.get(n, '')})" for n in names], fontsize=9)
    ax.set_xlabel("LOOCV Accuracy (single electrode)")
    ax.set_title("Which EEG electrodes carry the most semantic information?", fontsize=14, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_eeg_traces(eeg_temporal, labels, category_names, output_path):
    """Plot example EEG traces for a few stimuli."""
    electrode_names = sorted(ELECTRODES_1020)
    fig, axes = plt.subplots(len(category_names), 1, figsize=(14, 3 * len(category_names)))

    for i, cat in enumerate(category_names):
        cat_idx = i
        # Find first stimulus of this category
        idx = labels.tolist().index(cat_idx)
        eeg = eeg_temporal[idx]  # (T, n_electrodes)

        for j in range(eeg.shape[1]):
            offset = j * 0.15
            axes[i].plot(eeg[:, j] + offset, linewidth=0.8, alpha=0.7)

        axes[i].set_ylabel(cat)
        axes[i].set_yticks([])
        if i == len(category_names) - 1:
            axes[i].set_xlabel("Timestep (TR)")

    fig.suptitle("Virtual EEG Traces by Category (20 channels, 10-20 system)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    model = load_model()

    # Build electrode→cortex mapping
    print("=== Building electrode-to-cortex mapping ===")
    patches = build_electrode_patches(PATCH_RADIUS_MM)
    electrode_names = sorted(patches.keys())
    print(f"Mapped {len(patches)} electrodes, {sum(len(v) for v in patches.values())} total vertices")

    # Generate dataset
    print("\n=== Generating virtual EEG dataset ===")
    X_mean, eeg_temporal, X_raw, y, texts, category_names = generate_dataset(model, patches)
    n_samples = len(y)
    n_categories = len(category_names)
    print(f"Dataset: {n_samples} samples, {n_categories} categories, {X_mean.shape[1]} electrodes")

    # Save dataset
    np.savez(OUTPUT / "virtual_eeg_dataset.npz", X_mean=X_mean, X_raw=X_raw, y=y,
             categories=category_names, electrodes=electrode_names)

    # === Baselines ===
    print("\n=== Classification (Leave-One-Out CV) ===")

    # Chance
    chance = 1.0 / n_categories
    print(f"  Chance: {chance:.1%}")

    # Dummy classifier
    dummy = DummyClassifier(strategy="most_frequent")
    loo = LeaveOneOut()
    dummy_scores = cross_val_score(dummy, X_mean, y, cv=loo, scoring="accuracy")
    print(f"  Dummy (most frequent): {dummy_scores.mean():.1%}")

    # Full cortex (all 20484 vertices) — upper bound
    acc_full = evaluate_with_loocv(X_raw, y, name="Full cortex (20484 vertices)")

    # 20-channel EEG (mean activation)
    acc_eeg20 = evaluate_with_loocv(X_mean, y, name="20-ch EEG (mean activation)")

    # Temporal features
    acc_temporal = evaluate_temporal_features(eeg_temporal, y)

    # Electrode subsets
    print("\n=== Electrode subset analysis ===")
    # Frontal only
    frontal = [e for e in electrode_names if e.startswith(("F", "Fp"))]
    frontal_idx = [electrode_names.index(e) for e in frontal]
    acc_frontal = evaluate_with_loocv(X_mean[:, frontal_idx], y, name=f"Frontal only ({len(frontal)} ch)")

    # Temporal+parietal
    tp = [e for e in electrode_names if e.startswith(("T", "P"))]
    tp_idx = [electrode_names.index(e) for e in tp]
    acc_tp = evaluate_with_loocv(X_mean[:, tp_idx], y, name=f"Temporal+Parietal ({len(tp)} ch)")

    # Occipital only
    occ = [e for e in electrode_names if e.startswith("O")]
    occ_idx = [electrode_names.index(e) for e in occ]
    acc_occ = evaluate_with_loocv(X_mean[:, occ_idx], y, name=f"Occipital only ({len(occ)} ch)")

    # === Single electrode importance ===
    print("\n=== Single electrode importance ===")
    single_scores = evaluate_electrode_importance(X_mean, y, electrode_names)
    for name, score in list(single_scores.items())[:10]:
        region = ELECTRODE_REGIONS.get(name, "")
        above_chance = "***" if score > chance * 2 else "**" if score > chance * 1.5 else "*" if score > chance else ""
        print(f"  {name:>4s} ({region:>20s}): {score:.1%} {above_chance}")

    # === Plots ===
    print("\n=== Generating plots ===")
    plot_electrode_importance(single_scores, OUTPUT / "electrode_importance.png")
    plot_eeg_traces(eeg_temporal, y, category_names, OUTPUT / "eeg_traces.png")

    # === Summary ===
    summary = {
        "n_samples": n_samples,
        "n_categories": n_categories,
        "categories": category_names,
        "n_electrodes": len(electrode_names),
        "electrode_names": electrode_names,
        "patch_radius_mm": PATCH_RADIUS_MM,
        "chance_accuracy": chance,
        "results": {
            "full_cortex_20484v": acc_full,
            "eeg_20ch_mean": acc_eeg20,
            "eeg_20ch_temporal": acc_temporal,
            "frontal_only": acc_frontal,
            "temporal_parietal": acc_tp,
            "occipital_only": acc_occ,
        },
        "single_electrode_scores": single_scores,
    }
    with open(OUTPUT / "virtual_eeg_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Chance: {chance:.1%}")
    print(f"Full cortex:          {acc_full:.1%}")
    print(f"20-ch EEG (mean):     {acc_eeg20:.1%}")
    print(f"20-ch EEG (temporal): {acc_temporal:.1%}")
    print(f"Frontal subset:       {acc_frontal:.1%}")
    print(f"Temporal+Parietal:    {acc_tp:.1%}")
    print(f"Occipital:            {acc_occ:.1%}")
    print(f"\nBest single electrode: {list(single_scores.keys())[0]} ({list(single_scores.values())[0]:.1%})")


if __name__ == "__main__":
    main()
