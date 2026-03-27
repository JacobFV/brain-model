#!/usr/bin/env python3
"""
Brain Radio: See how different text lights up the brain.

Uses Meta's TRIBE v2 to predict fMRI brain responses to different text stimuli,
then generates comparative brain surface visualizations.
"""

import argparse
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE_FOLDER = Path("./cache")

# ── Built-in stimulus presets ──────────────────────────────────────────────

PRESETS = {
    "poetry": textwrap.dedent("""\
        Do not go gentle into that good night,
        Old age should burn and rave at close of day;
        Rage, rage against the dying of the light.
        Though wise men at their end know dark is right,
        Because their words had forked no lightning they
        Do not go gentle into that good night.
    """),
    "code": textwrap.dedent("""\
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + middle + quicksort(right)
    """),
    "love": textwrap.dedent("""\
        I have loved you for a thousand years, and I will love you
        for a thousand more. Every moment I spend with you feels like
        the first time we met. Your smile is the sunrise that wakes
        my world. I carry your heart with me, I carry it in my heart.
    """),
    "fear": textwrap.dedent("""\
        Something was watching from the darkness. I could feel its eyes
        on the back of my neck, cold and patient. The floorboards creaked
        behind me, but when I turned, nothing was there. My heart pounded
        so hard I could hear it. Then the lights went out.
    """),
    "math": textwrap.dedent("""\
        Consider the Riemann zeta function, defined as the sum of one over
        n to the power s, for all natural numbers n. The non-trivial zeros
        of this function all lie on the critical line where the real part
        of s equals one half. This remains unproven after over a century.
    """),
    "music": textwrap.dedent("""\
        The cello began with a low, mournful note that swelled into a
        cascade of harmonics. The melody twisted through minor keys,
        each phrase breathing tension and release. A sudden pizzicato
        broke the spell, and the orchestra erupted in fortissimo triumph.
    """),
}


def load_model():
    """Load TRIBE v2 from HuggingFace."""
    from tribev2 import TribeModel

    print("Loading TRIBE v2 model...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)

    # Force local computation mode for all feature extractors
    # (the default exca caching can hang on first run)
    for attr in ["text_feature", "audio_feature", "video_feature"]:
        feat = getattr(model.data, attr, None)
        if feat is not None and hasattr(feat, "infra"):
            feat.infra.mode = "force"

    print("Model loaded.")
    return model


def _build_events_for_text(text: str, cache_folder: Path) -> "pd.DataFrame":
    """
    Build a TRIBE v2 events dataframe from raw text, bypassing the broken
    whisperx subprocess. We use gTTS to generate audio, then construct
    word-level events with evenly-distributed timing.
    """
    import hashlib
    import pandas as pd
    import soundfile as sf
    from gtts import gTTS
    from langdetect import detect
    from neuralset.events.transforms import (
        AddContextToWords,
        AddSentenceToWords,
        AddText,
        ChunkEvents,
        RemoveMissing,
        standardize_events,
    )

    # Generate audio via gTTS
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    audio_dir = cache_folder / f"brain_radio_{text_hash}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio.mp3"

    if not audio_path.exists():
        lang = detect(text)
        tts = gTTS(text, lang=lang)
        tts.save(str(audio_path))

    # Get audio duration
    info = sf.info(str(audio_path))
    duration = info.duration

    # Build Audio event
    audio_event = {
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0.0,
        "duration": duration,
        "timeline": "default",
        "subject": "default",
    }

    # Build Word events with evenly-distributed timing
    words = text.split()
    n_words = len(words)
    word_duration = duration / max(n_words, 1)

    # Split text into sentences for sentence assignment
    import re
    sentences = re.split(r'(?<=[.!?;,])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    word_events = []
    for i, word in enumerate(words):
        # Find which sentence this word belongs to
        sentence_text = ""
        cum_words = 0
        for sent in sentences:
            sent_words = sent.split()
            if i < cum_words + len(sent_words):
                sentence_text = sent
                break
            cum_words += len(sent_words)
        if not sentence_text:
            sentence_text = sentences[-1]

        word_events.append({
            "type": "Word",
            "text": word.strip(".,;:!?\"'()-"),
            "start": i * word_duration,
            "duration": word_duration * 0.8,
            "timeline": "default",
            "subject": "default",
            "language": "english",
            "sequence_id": 0,
            "sentence": sentence_text,
        })

    df = pd.DataFrame([audio_event] + word_events)

    # Run remaining transforms (skip ExtractWordsFromAudio)
    transforms = [
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.99),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    df = standardize_events(df)
    for transform in transforms:
        df = transform(df)
    return standardize_events(df, auto_fill=False)


def text_to_predictions(model, text: str, label: str = ""):
    """Run a text string through TRIBE v2 and return brain predictions."""
    tag = f" [{label}]" if label else ""
    print(f"Predicting brain response{tag}...")

    df = _build_events_for_text(text, CACHE_FOLDER)
    preds, segments = model.predict(events=df)
    print(f"  -> {preds.shape[0]} timesteps x {preds.shape[1]} vertices")
    return preds, segments


def mean_activation(preds: np.ndarray) -> np.ndarray:
    """Average brain activation across all timesteps -> (n_vertices,)."""
    return preds.mean(axis=0)


def peak_activation(preds: np.ndarray) -> np.ndarray:
    """Peak brain activation across all timesteps -> (n_vertices,)."""
    return preds.max(axis=0)


# ── Visualization ──────────────────────────────────────────────────────────


def plot_comparison(results: dict[str, np.ndarray], mode: str = "mean", output: str = "brain_comparison.png"):
    """
    Render side-by-side brain heatmaps for multiple stimuli.

    results: {label: preds} where preds is (n_timesteps, n_vertices)
    mode: "mean" or "peak"
    """
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    agg_fn = mean_activation if mode == "mean" else peak_activation
    labels = list(results.keys())
    n = len(labels)
    views = ["left", "right"]

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), subplot_kw={"projection": "3d"})
    if n == 1:
        axes = axes[np.newaxis, :]

    # Compute global vmin/vmax for consistent color scale across stimuli
    all_signals = [agg_fn(results[l]) for l in labels]
    vmax = np.percentile(np.concatenate(all_signals), 97)
    vmin = 0

    for i, label in enumerate(labels):
        signals = all_signals[i]
        for j, view in enumerate(views):
            plotter.plot_surf(signals, axes=[axes[i, j]], views=[view],
                              cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f"{label} ({view})", fontsize=14, fontweight="bold")

    fig.suptitle(f"Brain Radio: {mode.title()} Activation Comparison", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved comparison -> {output}")
    plt.close(fig)


def plot_diff(results: dict[str, np.ndarray], output: str = "brain_diff.png"):
    """
    For each pair of stimuli, show which brain regions are MORE activated
    by one vs the other (signed difference of mean activations).
    """
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    labels = list(results.keys())
    if len(labels) < 2:
        print("Need at least 2 stimuli for diff view.")
        return

    base_label = labels[0]
    base_act = mean_activation(results[base_label])
    others = labels[1:]

    fig, axes = plt.subplots(len(others), 2, figsize=(14, 4 * len(others)), subplot_kw={"projection": "3d"})
    if len(others) == 1:
        axes = axes[np.newaxis, :]

    for i, other_label in enumerate(others):
        diff = mean_activation(results[other_label]) - base_act
        for j, view in enumerate(["left", "right"]):
            plotter.plot_surf(diff, axes=[axes[i, j]], views=[view],
                              cmap="bwr", norm_percentile=97, symmetric_cbar=True)
            title = f"{other_label} vs {base_label}"
            axes[i, j].set_title(f"{title} ({view})", fontsize=12, fontweight="bold")

    fig.suptitle("Brain Diff: Red = more active, Blue = less active", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved diff view -> {output}")
    plt.close(fig)


def _get_roi_indices() -> dict[str, np.ndarray]:
    """
    Build a mapping from human-readable brain region names to vertex indices
    in fsaverage5 (20484 vertices total: 10242 left + 10242 right).
    Uses the Destrieux cortical atlas from nilearn.
    """
    from nilearn import datasets
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = datasets.fetch_atlas_surf_destrieux()

    lh = np.array(atlas["map_left"])   # (10242,)
    rh = np.array(atlas["map_right"])  # (10242,)
    # TRIBE v2 concatenates left then right hemisphere
    full_map = np.concatenate([lh, rh])  # (20484,)

    labels = [str(l) for l in atlas["labels"]]

    # Map friendly names -> Destrieux label indices
    roi_label_map = {
        "Visual (V1/V2)":     ["S_calcarine", "G_cuneus"],
        "Occipital":          ["G_occipital_sup", "G_occipital_middle", "Pole_occipital"],
        "Auditory (A1)":      ["G_temp_sup-G_T_transv"],
        "Broca's area":       ["G_front_inf-Opercular", "G_front_inf-Triangul"],
        "Wernicke's area":    ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
        "Fusiform (FFA)":     ["G_oc-temp_lat-fusifor"],
        "Parahipp. (PPA)":    ["G_oc-temp_med-Parahip"],
        "Frontal sup.":       ["G_front_sup"],
        "Angular/TPJ":        ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
        "Precuneus":          ["G_precuneus"],
        "Motor":              ["G_precentral"],
        "Temporal mid.":      ["G_temporal_middle"],
    }

    roi_indices = {}
    for friendly_name, atlas_names in roi_label_map.items():
        indices = []
        for aname in atlas_names:
            if aname in labels:
                label_idx = labels.index(aname)
                indices.append(np.where(full_map == label_idx)[0])
        if indices:
            roi_indices[friendly_name] = np.concatenate(indices)

    return roi_indices


def plot_fingerprint(results: dict[str, np.ndarray], output: str = "brain_fingerprint.png"):
    """
    Radar chart of mean activation in key brain regions for each stimulus.
    Uses ROI-level averages to create a 'brain fingerprint'.
    """
    roi_indices = _get_roi_indices()
    valid_rois = list(roi_indices.keys())

    if len(valid_rois) < 3:
        print("Not enough ROIs found for fingerprint plot, skipping.")
        return

    n_rois = len(valid_rois)
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # Compute all ROI means first so we can normalize across stimuli
    all_values = {}
    for label, preds in results.items():
        act = mean_activation(preds)
        all_values[label] = [float(act[roi_indices[roi]].mean()) for roi in valid_rois]

    # Normalize each ROI to [0, 1] across all stimuli for fair visual comparison
    values_array = np.array(list(all_values.values()))  # (n_stimuli, n_rois)
    roi_mins = values_array.min(axis=0)
    roi_maxs = values_array.max(axis=0)
    roi_range = roi_maxs - roi_mins
    roi_range[roi_range == 0] = 1  # avoid division by zero

    for (label, _), color in zip(results.items(), colors):
        raw = np.array(all_values[label])
        normed = (raw - roi_mins) / roi_range
        values = normed.tolist() + [normed[0]]
        ax.plot(angles, values, "o-", linewidth=2, markersize=6, label=label, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_rois, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Brain Fingerprint by Region\n(normalized per-region)", fontsize=15, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved fingerprint -> {output}")
    plt.close(fig)


def render_video(results: dict[str, np.ndarray], segments_map: dict, output_dir: str = "."):
    """Render an MP4 of brain activity over time for each stimulus."""
    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh="fsaverage5")

    for label, preds in results.items():
        filepath = Path(output_dir) / f"brain_{label}.mp4"
        print(f"Rendering video for '{label}' -> {filepath}")
        plotter.plot_timesteps_mp4(
            preds,
            filepath=str(filepath),
            segments=segments_map.get(label),
            norm_percentile=97,
        )
        print(f"  Done: {filepath}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Brain Radio: Compare how different text lights up the brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Built-in presets: {', '.join(PRESETS.keys())}

            Examples:
              brain-radio poetry love fear          # compare 3 presets
              brain-radio --custom "Hello world"    # use custom text
              brain-radio poetry --custom "To be or not to be" --video
        """),
    )
    parser.add_argument("presets", nargs="*", help="Preset names to compare")
    parser.add_argument("--custom", action="append", default=[], metavar="TEXT",
                        help="Custom text stimulus (can be repeated)")
    parser.add_argument("--video", action="store_true", help="Also render MP4 videos")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--mode", choices=["mean", "peak"], default="mean",
                        help="Aggregation mode for comparison plot")
    args = parser.parse_args()

    # Default to a fun set if nothing specified
    if not args.presets and not args.custom:
        args.presets = ["poetry", "code", "love", "fear"]

    # Validate presets
    for p in args.presets:
        if p not in PRESETS:
            parser.error(f"Unknown preset '{p}'. Available: {', '.join(PRESETS.keys())}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Run predictions
    results = {}
    segments_map = {}

    for name in args.presets:
        preds, segs = text_to_predictions(model, PRESETS[name], label=name)
        results[name] = preds
        segments_map[name] = segs

    for i, text in enumerate(args.custom):
        label = f"custom_{i+1}"
        preds, segs = text_to_predictions(model, text, label=label)
        results[label] = preds
        segments_map[label] = segs

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_comparison(results, mode=args.mode, output=str(out / "brain_comparison.png"))
    plot_diff(results, output=str(out / "brain_diff.png"))
    plot_fingerprint(results, output=str(out / "brain_fingerprint.png"))

    if args.video:
        render_video(results, segments_map, output_dir=str(out))

    print(f"\nAll outputs saved to {out}/")
    print("Open brain_comparison.png to see how each text lights up the brain!")


if __name__ == "__main__":
    main()
