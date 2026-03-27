#!/usr/bin/env python3
"""
Experiment: Learning what we don't know about brain emotion stimuli.

Systematically probes TRIBE v2 with diverse emotional stimuli to discover
non-obvious patterns in cortical responses. Focuses on:
- Compound/complex emotions (bittersweet, schadenfreude, nostalgia)
- Emotional transitions (fear→relief, joy→grief)
- Emotion expressed through different narrative modes
- Asymmetries in valence, arousal, and social vs non-social emotion
- Temporal dynamics: how does emotion "build" over timesteps?
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.model import load_model, text_to_predictions, CACHE_FOLDER
from core.roi import get_roi_indices, roi_means

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Stimulus battery ──────────────────────────────────────────────────────
# Organized by what we want to discover, not by known categories

STIMULI = {
    # --- Primary emotions (baseline) ---
    "joy_pure": "The letter arrived and it said I got in. I screamed and jumped and hugged everyone in the room. I couldn't stop smiling. This was the happiest moment of my life.",
    "fear_pure": "Something moved in the darkness behind me. I could hear breathing that wasn't mine. My heart hammered as I reached for the door handle and found it locked.",
    "anger_pure": "They lied to my face and stole everything I had worked for. My fists clenched. My jaw tightened. I wanted to scream until the walls shook.",
    "sadness_pure": "She was gone. The house was empty now, her chair still warm, her tea still steaming. I sat on the floor and the silence pressed in from every direction.",
    "disgust_pure": "I opened the container and the smell hit me first, a wave of rot and decay. Inside was a mass of writhing larvae feeding on blackened meat.",

    # --- Compound emotions (what patterns emerge?) ---
    "bittersweet": "My daughter walked across the stage to receive her diploma. I was so proud my chest ached, but I couldn't stop thinking about how she was leaving home tomorrow and nothing would ever be the same.",
    "nostalgic_joy": "I found my grandfather's old watch in a drawer. It still ticked. I held it to my ear and suddenly I was eight years old again, sitting on his lap while he told me stories about the sea.",
    "schadenfreude": "The bully who tormented me for years just tripped and fell flat on his face in front of the whole school. Everyone laughed. I knew I shouldn't enjoy it, but god it felt good.",
    "melancholic_beauty": "The last light of autumn filtered through the dying leaves, painting everything gold. It was the most beautiful thing I had ever seen, and it made me want to cry because I knew it would be gone by morning.",
    "anxious_excitement": "Tomorrow is the big day. I've prepared for months but my stomach is churning. I can't tell if I'm terrified or thrilled. Maybe both. I keep rehearsing in my head.",
    "tender_grief": "I found his favorite old sweater and pressed it to my face. It still smelled like him. I smiled through tears, remembering how he always wore it inside out.",
    "righteous_fury": "They poisoned the river and the children got sick. The company knew for years and covered it up. Something inside me crystallized into cold, purposeful determination. This ends now.",
    "grateful_humility": "A stranger paid for my groceries when my card was declined. She just smiled and walked away. I stood there holding my bags, overwhelmed by the kindness of someone I would never see again.",
    "wistful_regret": "I drove past our old house today. Someone had cut down the oak tree we planted together. I thought about all the things I never said and all the moments I let slip through my fingers.",

    # --- Emotional transitions (how does the brain handle shifts?) ---
    "fear_to_relief": "The ground shook and the ceiling cracked. I thought the building was collapsing. I ran for the door, heart pounding. Then it stopped. Silence. Just a small earthquake. I laughed, shaking, alive.",
    "joy_to_grief": "We were all laughing at the dinner table, telling old stories, when the phone rang. Mom answered. Her face changed. She set down the phone and said Dad's plane never landed.",
    "anger_to_compassion": "I stormed into his office ready to quit. Then I saw him hunched at his desk, crying. His wife was in the hospital. All my rage dissolved. I sat down and asked if he needed anything.",
    "disgust_to_wonder": "I recoiled from the slimy mass on the tide pool rock. Then it moved, unfolding into the most intricate creature I had ever seen, its translucent body pulsing with bioluminescent blue light.",

    # --- Social vs non-social emotion ---
    "social_shame": "Everyone in the room went quiet when they saw my mistake on the projector. Two hundred people staring at me. My face burned. I wanted the floor to swallow me whole.",
    "social_pride": "My team looked at me with genuine respect after I presented our results. The room applauded. My mentor nodded from the back row. Years of work, finally recognized.",
    "nonsocial_awe": "I stood at the rim of the Grand Canyon at sunrise. The scale of it made my knees weak. Layers of stone a billion years old, carved by water and time. I felt very small and very lucky.",
    "nonsocial_serenity": "The lake was perfectly still at dawn. Not a breath of wind. The reflection of the mountains was so perfect I couldn't tell which was real. I sat and watched and thought of nothing.",

    # --- Emotion through abstraction vs narrative ---
    "abstract_dread": "Something is wrong. Not here, not now, but somewhere deep and fundamental. A fracture in the architecture of things. The certainty that what holds the world together is failing.",
    "abstract_love": "There is a warmth that exists between two people who have chosen each other again and again across decades. Not passion, not habit, but a quiet gravitational pull that neither questions.",
    "narrative_dread": "The doctor paused too long after reading the results. She took off her glasses and rubbed her eyes. When she looked at me, I knew before she spoke. The walls of the room seemed to close in.",
    "narrative_love": "He made her coffee every morning for thirty years. The same way. Two sugars, a little cream. He never asked if she wanted it. He just brought it. Even on the morning she didn't wake up.",

    # --- Moral emotions ---
    "moral_outrage": "The whistleblower who exposed the fraud was fired and blacklisted while the executives who stole billions received bonuses and promotions. The system protected the guilty and destroyed the innocent.",
    "moral_elevation": "The firefighter went back into the burning building three times. The third time, the roof collapsed. They found her body shielding two children, both alive. She had known she wouldn't make it out.",
    "guilt": "I promised I would visit him in the hospital. Every day I told myself I would go tomorrow. Then one morning my phone buzzed. I never got to say goodbye.",
    "contempt": "He sat across from the committee with that smug grin, lying effortlessly about everything. He knew they couldn't prove it. He enjoyed watching them try. I felt my respect for humanity shrink.",
}


def analyze_emotion_patterns(results: dict[str, np.ndarray], roi_idx: dict):
    """Compute ROI activation profiles for all emotions and find non-obvious patterns."""
    profiles = {}
    for name, preds in results.items():
        profiles[name] = roi_means(preds, roi_idx)

    # Convert to matrix for analysis
    emotions = list(profiles.keys())
    rois = list(roi_idx.keys())
    matrix = np.array([[profiles[e][r] for r in rois] for e in emotions])

    analysis = {}

    # 1. Correlation matrix between emotions in cortical space
    corr = np.corrcoef(matrix)
    analysis["emotion_correlations"] = {
        emotions[i]: {emotions[j]: round(float(corr[i, j]), 4) for j in range(len(emotions))}
        for i in range(len(emotions))
    }

    # 2. Find surprising similarities (high correlation between seemingly different emotions)
    surprising_pairs = []
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            c = corr[i, j]
            pair_name = f"{emotions[i]} <-> {emotions[j]}"
            surprising_pairs.append((pair_name, float(c)))
    surprising_pairs.sort(key=lambda x: x[1], reverse=True)
    analysis["most_similar_pairs"] = surprising_pairs[:20]
    analysis["most_dissimilar_pairs"] = surprising_pairs[-20:]

    # 3. ROI selectivity: which ROIs best discriminate between emotions?
    roi_variance = {}
    for j, roi in enumerate(rois):
        vals = matrix[:, j]
        roi_variance[roi] = float(np.std(vals))
    analysis["roi_discriminability"] = dict(sorted(roi_variance.items(), key=lambda x: -x[1]))

    # 4. Compound vs pure emotions: do compounds use different regions?
    pure = ["joy_pure", "fear_pure", "anger_pure", "sadness_pure", "disgust_pure"]
    compound = ["bittersweet", "nostalgic_joy", "schadenfreude", "melancholic_beauty",
                 "anxious_excitement", "tender_grief", "righteous_fury"]
    if all(e in profiles for e in pure + compound):
        pure_mean = np.mean([matrix[emotions.index(e)] for e in pure], axis=0)
        compound_mean = np.mean([matrix[emotions.index(e)] for e in compound], axis=0)
        diff = compound_mean - pure_mean
        compound_vs_pure = {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))}
        compound_vs_pure = dict(sorted(compound_vs_pure.items(), key=lambda x: -abs(x[1])))
        analysis["compound_vs_pure_rois"] = compound_vs_pure

    # 5. Transitions: how much does the brain state shift during emotion transitions?
    transitions = ["fear_to_relief", "joy_to_grief", "anger_to_compassion", "disgust_to_wonder"]
    transition_dynamics = {}
    for t in transitions:
        if t in results:
            preds = results[t]
            if preds.shape[0] > 2:
                # Compute timestep-to-timestep change magnitude
                diffs = np.diff(preds, axis=0)
                change_per_step = np.linalg.norm(diffs, axis=1)
                transition_dynamics[t] = {
                    "mean_change": float(change_per_step.mean()),
                    "max_change": float(change_per_step.max()),
                    "max_change_timestep": int(np.argmax(change_per_step)),
                    "total_timesteps": int(preds.shape[0]),
                    "change_trajectory": change_per_step.tolist(),
                }
    analysis["transition_dynamics"] = transition_dynamics

    # 6. Social vs non-social: lateralization and region differences
    social = ["social_shame", "social_pride"]
    nonsocial = ["nonsocial_awe", "nonsocial_serenity"]
    if all(e in profiles for e in social + nonsocial):
        social_mean = np.mean([matrix[emotions.index(e)] for e in social], axis=0)
        nonsocial_mean = np.mean([matrix[emotions.index(e)] for e in nonsocial], axis=0)
        diff = social_mean - nonsocial_mean
        social_vs_nonsocial = {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))}
        social_vs_nonsocial = dict(sorted(social_vs_nonsocial.items(), key=lambda x: -abs(x[1])))
        analysis["social_vs_nonsocial_rois"] = social_vs_nonsocial

    # 7. Abstract vs narrative: does abstraction activate different areas?
    abstract = ["abstract_dread", "abstract_love"]
    narrative = ["narrative_dread", "narrative_love"]
    if all(e in profiles for e in abstract + narrative):
        abstract_mean = np.mean([matrix[emotions.index(e)] for e in abstract], axis=0)
        narrative_mean = np.mean([matrix[emotions.index(e)] for e in narrative], axis=0)
        diff = abstract_mean - narrative_mean
        abstract_vs_narrative = {rois[j]: round(float(diff[j]), 6) for j in range(len(rois))}
        abstract_vs_narrative = dict(sorted(abstract_vs_narrative.items(), key=lambda x: -abs(x[1])))
        analysis["abstract_vs_narrative_rois"] = abstract_vs_narrative

    # 8. Temporal dynamics per emotion: how does activation evolve?
    temporal = {}
    for name, preds in results.items():
        roi_time = {}
        for roi_name, idx in roi_idx.items():
            timeseries = preds[:, idx].mean(axis=1)  # mean activation in ROI per timestep
            roi_time[roi_name] = {
                "peak_timestep": int(np.argmax(timeseries)),
                "onset_value": float(timeseries[0]),
                "peak_value": float(timeseries.max()),
                "final_value": float(timeseries[-1]),
                "ramp_up": float(timeseries.max() - timeseries[0]),
            }
        temporal[name] = roi_time
    analysis["temporal_dynamics"] = temporal

    return analysis, matrix, emotions, rois


def plot_emotion_similarity_matrix(corr_dict, emotions, output_path):
    """Plot correlation matrix between all emotions."""
    n = len(emotions)
    matrix = np.array([[corr_dict[emotions[i]][emotions[j]] for j in range(n)] for i in range(n)])

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(emotions, rotation=90, fontsize=7)
    ax.set_yticklabels(emotions, fontsize=7)
    plt.colorbar(im, ax=ax, label="Cortical pattern correlation")
    ax.set_title("Emotion Similarity in Cortical Space", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_compound_vs_pure(analysis, output_path):
    """Plot which ROIs differentiate compound from pure emotions."""
    data = analysis.get("compound_vs_pure_rois", {})
    if not data:
        return
    rois = list(data.keys())
    vals = [data[r] for r in rois]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(rois)), vals, color=colors)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=9)
    ax.set_xlabel("Activation difference (compound - pure)")
    ax.set_title("Which brain regions distinguish compound from pure emotions?", fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_transition_dynamics(analysis, output_path):
    """Plot how brain state changes during emotional transitions."""
    transitions = analysis.get("transition_dynamics", {})
    if not transitions:
        return

    fig, axes = plt.subplots(len(transitions), 1, figsize=(12, 3 * len(transitions)), sharex=False)
    if len(transitions) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, transitions.items()):
        trajectory = data["change_trajectory"]
        ax.plot(trajectory, "o-", linewidth=2, markersize=4, color="#e41a1c")
        ax.fill_between(range(len(trajectory)), trajectory, alpha=0.2, color="#e41a1c")
        ax.axvline(data["max_change_timestep"], color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Neural state change")
        ax.set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Brain State Transitions During Emotional Shifts", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    model = load_model()
    roi_idx = get_roi_indices()

    # Run all stimuli
    results = {}
    for name, text in STIMULI.items():
        preds, _ = text_to_predictions(model, text, label=name)
        results[name] = preds

    # Analyze
    print("\n=== Analyzing emotion patterns ===")
    analysis, matrix, emotions, rois = analyze_emotion_patterns(results, roi_idx)

    # Save raw analysis
    # Filter out non-serializable items from temporal_dynamics
    serializable = {k: v for k, v in analysis.items() if k != "temporal_dynamics"}
    with open(OUTPUT / "emotion_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved analysis -> {OUTPUT / 'emotion_analysis.json'}")

    # Save predictions
    np.savez(OUTPUT / "emotion_predictions.npz", **{k: v for k, v in results.items()})

    # Generate plots
    plot_emotion_similarity_matrix(analysis["emotion_correlations"], emotions, OUTPUT / "emotion_similarity.png")
    plot_compound_vs_pure(analysis, OUTPUT / "compound_vs_pure.png")
    plot_transition_dynamics(analysis, OUTPUT / "transition_dynamics.png")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print("\nMost similar emotion pairs (cortical pattern):")
    for pair, corr in analysis["most_similar_pairs"][:10]:
        print(f"  {corr:.3f}  {pair}")
    print("\nMost dissimilar emotion pairs:")
    for pair, corr in analysis["most_dissimilar_pairs"][:10]:
        print(f"  {corr:.3f}  {pair}")
    print("\nROIs that best discriminate between emotions (highest variance):")
    for roi, var in list(analysis["roi_discriminability"].items())[:8]:
        print(f"  {var:.6f}  {roi}")
    if "compound_vs_pure_rois" in analysis:
        print("\nROIs more active in compound vs pure emotions:")
        items = list(analysis["compound_vs_pure_rois"].items())
        for roi, diff in items[:5]:
            print(f"  +{diff:.6f}  {roi}")
        print("ROIs less active:")
        for roi, diff in items[-5:]:
            print(f"  {diff:.6f}  {roi}")


if __name__ == "__main__":
    main()
