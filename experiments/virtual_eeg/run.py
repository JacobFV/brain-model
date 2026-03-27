#!/usr/bin/env python3
"""
Experiment: Virtual EEG Headset

Uses TRIBE v2 as a synthetic fMRI oracle to generate training data for:
1. Forward model: stimulus → low-dimensional "virtual EEG" signals
2. Inverse model: "virtual EEG" signals → semantic embedding

Pipeline:
  text/audio → TRIBE v2 → 20K vertex fMRI → PCA/autoencoder → N-channel "EEG"
  N-channel "EEG" → decoder → CLIP/sentence-transformer embedding space

The "virtual EEG" is a compressed representation of cortical activity that
could theoretically be measured with real EEG electrodes positioned optimally
on the scalp. We learn what the minimal measurement is that still carries
semantic information.
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
from torch.utils.data import DataLoader, TensorDataset

_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
from core.model import load_model, text_to_predictions, CACHE_FOLDER

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Stimulus corpus for training data generation ──────────────────────────

CATEGORIES = {
    "animal": [
        "A golden retriever running across a sunlit field, tongue out, ears flapping in the wind",
        "A cat curled up on a warm windowsill, purring softly as rain taps against the glass",
        "An eagle soaring above a mountain valley, its wings barely moving in the thermal currents",
        "A whale breaching the surface of the ocean, crashing back down in an explosion of white spray",
        "Tiny ants carrying a leaf fragment ten times their size along a narrow trail in the dirt",
    ],
    "music": [
        "A solo violin playing a haunting melody in an empty cathedral, each note echoing off stone walls",
        "Heavy drums pounding a tribal rhythm that makes your whole body vibrate with the beat",
        "A jazz piano improvising over a walking bass line in a smoky underground club",
        "An orchestra building to a massive crescendo, every instrument contributing to a wall of sound",
        "A child plinking out Twinkle Twinkle Little Star on a toy piano, missing every other note",
    ],
    "food": [
        "The smell of fresh bread baking in the oven, warm and yeasty, filling the entire house",
        "Biting into a perfectly ripe peach, the juice running down your chin, sweet and fragrant",
        "Sizzling garlic and onions hitting a hot pan with olive oil, the kitchen filling with aroma",
        "A bowl of steaming ramen with soft-boiled eggs, green onions, and slices of tender pork belly",
        "Bitter dark chocolate melting slowly on your tongue, rich and complex with hints of cherry",
    ],
    "danger": [
        "A snake coiled on the path ahead, its rattle buzzing as you freeze mid-step",
        "The car in front of you swerves suddenly and you slam the brakes, tires screaming",
        "A crack of thunder directly overhead so loud it shakes your ribcage",
        "The ice beneath your feet groans and a dark crack shoots across the frozen lake",
        "Smoke pouring under the door of your hotel room at three in the morning",
    ],
    "abstract_thought": [
        "Infinity stretches in all directions, not as space but as possibility, each moment branching",
        "The number zero: nothing, and yet the most important something, the pivot of all mathematics",
        "Time flows differently in memory, some hours lasting forever and whole years compressed to a feeling",
        "Consciousness observing itself, the strange loop of a mind trying to understand its own existence",
        "The idea that every person you pass on the street has a life as vivid and complex as your own",
    ],
    "social": [
        "Your best friend calls you at midnight because they need someone to talk to, and you listen",
        "A stranger on the bus smiles at you and it lifts your entire morning without reason",
        "Standing at a podium giving a speech to five hundred people, every eye fixed on you",
        "Two old men on a park bench playing chess, neither speaking, perfectly content in silence",
        "A baby reaches up and grabs your finger for the first time, holding on with surprising strength",
    ],
    "spatial": [
        "Standing at the edge of a cliff looking down a thousand feet to the valley floor below",
        "Navigating through a narrow alleyway in a foreign city, tall buildings blocking the sky on both sides",
        "Floating weightless in the middle of a vast dark ocean at night, no land visible in any direction",
        "The inside of a cathedral, looking up at vaulted ceilings soaring a hundred feet overhead",
        "A tiny room barely larger than a closet, the walls pressing in close, the ceiling low",
    ],
    "language": [
        "She spoke in a language I had never heard, melodic and rhythmic, and yet I understood her meaning perfectly from her gestures and expression",
        "The word petrichor, meaning the smell of rain on dry earth, a word that captures something you always knew but never named",
        "Reading a poem where every word is common but the arrangement creates meaning that transcends the individual parts",
        "A conversation where two people keep finishing each other's sentences, their thoughts synchronized",
        "The frustration of knowing exactly what you want to say but being unable to find the right word for it",
    ],
}


class BrainEncoder(nn.Module):
    """Compress 20K vertex fMRI to N-channel virtual EEG."""

    def __init__(self, n_vertices: int = 20484, n_channels: int = 32, hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_vertices, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_channels),
        )

    def forward(self, x):
        return self.encoder(x)


class SemanticDecoder(nn.Module):
    """Decode virtual EEG back to semantic embedding space."""

    def __init__(self, n_channels: int = 32, embed_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class CategoryClassifier(nn.Module):
    """Classify semantic category from virtual EEG."""

    def __init__(self, n_channels: int = 32, n_categories: int = 8, hidden: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_categories),
        )

    def forward(self, x):
        return self.classifier(x)


def generate_dataset(model):
    """Generate synthetic fMRI dataset from TRIBE v2."""
    all_preds = []
    all_labels = []
    all_texts = []
    category_names = sorted(CATEGORIES.keys())

    for cat_idx, cat_name in enumerate(category_names):
        for text in CATEGORIES[cat_name]:
            preds, _ = text_to_predictions(model, text, label=f"{cat_name}")
            mean_activation = preds.mean(axis=0)  # (20484,)
            all_preds.append(mean_activation)
            all_labels.append(cat_idx)
            all_texts.append(text)

    X = np.stack(all_preds)  # (N, 20484)
    y = np.array(all_labels)  # (N,)
    return X, y, all_texts, category_names


def train_virtual_eeg(X, y, n_channels=32, n_epochs=200):
    """Train the virtual EEG encoder + category classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_categories = len(np.unique(y))

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    # Split train/test
    n = len(X_t)
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    encoder = BrainEncoder(n_vertices=X.shape[1], n_channels=n_channels).to(device)
    classifier = CategoryClassifier(n_channels=n_channels, n_categories=n_categories).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    # Also add reconstruction loss to ensure EEG preserves info
    decoder_recon = nn.Linear(n_channels, X.shape[1]).to(device)
    optimizer_recon = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder_recon.parameters()),
        lr=1e-3,
    )

    history = {"train_loss": [], "test_acc": [], "recon_loss": []}

    for epoch in range(n_epochs):
        encoder.train()
        classifier.train()

        # Classification loss
        eeg = encoder(X_t[train_idx])
        logits = classifier(eeg)
        cls_loss = criterion(logits, y_t[train_idx])

        # Reconstruction loss
        recon = decoder_recon(eeg)
        recon_loss = nn.functional.mse_loss(recon, X_t[train_idx])

        loss = cls_loss + 0.01 * recon_loss
        optimizer.zero_grad()
        optimizer_recon.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_recon.step()

        # Test accuracy
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            test_eeg = encoder(X_t[test_idx])
            test_logits = classifier(test_eeg)
            test_preds = test_logits.argmax(dim=1)
            acc = (test_preds == y_t[test_idx]).float().mean().item()

        history["train_loss"].append(float(loss))
        history["test_acc"].append(acc)
        history["recon_loss"].append(float(recon_loss))

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: loss={loss:.4f} recon={recon_loss:.4f} test_acc={acc:.2%}")

    return encoder, classifier, history


def analyze_channel_importance(encoder, X, roi_indices):
    """Analyze what each virtual EEG channel represents anatomically."""
    device = next(encoder.parameters()).device
    encoder.eval()

    # Get encoder weights from first layer
    W = encoder.encoder[0].weight.detach().cpu().numpy()  # (hidden, n_vertices)

    # For each channel in the final output, trace which vertices matter most
    # Use gradient-based attribution
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(device)

    channel_roi_map = {}
    n_channels = encoder.encoder[-1].out_features

    for ch in range(n_channels):
        X_input = X_t.clone().requires_grad_(True)
        eeg = encoder(X_input)
        eeg[:, ch].sum().backward()
        grad = X_input.grad.abs().mean(dim=0).cpu().numpy()  # (n_vertices,)

        # Map to ROIs
        roi_importance = {}
        for roi_name, idx in roi_indices.items():
            roi_importance[roi_name] = float(grad[idx].mean())
        roi_importance = dict(sorted(roi_importance.items(), key=lambda x: -x[1]))
        channel_roi_map[f"channel_{ch}"] = {k: round(v, 6) for k, v in list(roi_importance.items())[:5]}

    return channel_roi_map


def plot_results(history, channel_map, category_names, output_dir):
    """Generate visualization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train loss", color="#e41a1c")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(history["test_acc"], label="Test accuracy", color="#377eb8")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Category Classification ({len(category_names)} classes)")
    ax2.axhline(1.0 / len(category_names), color="gray", linestyle="--", label="Chance")
    ax2.legend()

    fig.suptitle("Virtual EEG Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "virtual_eeg_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'virtual_eeg_training.png'}")

    # Channel importance heatmap
    rois = set()
    for ch_data in channel_map.values():
        rois.update(ch_data.keys())
    rois = sorted(rois)
    channels = sorted(channel_map.keys())

    matrix = np.zeros((len(channels), len(rois)))
    for i, ch in enumerate(channels):
        for j, roi in enumerate(rois):
            matrix[i, j] = channel_map[ch].get(roi, 0)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=90, fontsize=8)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=8)
    ax.set_title("Virtual EEG Channel → Brain Region Mapping", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Gradient importance")
    fig.tight_layout()
    fig.savefig(output_dir / "channel_roi_mapping.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'channel_roi_mapping.png'}")


def main():
    from core.roi import get_roi_indices

    model = load_model()

    print("=== Generating synthetic fMRI dataset ===")
    X, y, texts, category_names = generate_dataset(model)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} vertices, {len(category_names)} categories")

    # Save dataset
    np.savez(OUTPUT / "virtual_eeg_dataset.npz", X=X, y=y, categories=category_names)

    # PCA baseline: how many components to explain 95% variance?
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(X.shape[0] - 1, 100))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cumvar >= 0.95)) + 1
    print(f"PCA: {n95} components explain 95% of variance")

    # Train virtual EEG models at different channel counts
    results = {}
    for n_ch in [4, 8, 16, 32]:
        print(f"\n=== Training {n_ch}-channel virtual EEG ===")
        encoder, classifier, history = train_virtual_eeg(X, y, n_channels=n_ch, n_epochs=300)
        final_acc = history["test_acc"][-1]
        results[n_ch] = {"final_accuracy": final_acc, "history": history}
        print(f"  Final test accuracy: {final_acc:.2%}")

        if n_ch == 32:
            roi_idx = get_roi_indices()
            channel_map = analyze_channel_importance(encoder, X, roi_idx)
            plot_results(history, channel_map, category_names, OUTPUT)

            # Save channel mapping
            with open(OUTPUT / "channel_roi_mapping.json", "w") as f:
                json.dump(channel_map, f, indent=2)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"PCA components for 95% variance: {n95}")
    print(f"Chance accuracy: {1/len(category_names):.2%}")
    for n_ch, data in results.items():
        print(f"  {n_ch:2d}-channel EEG: {data['final_accuracy']:.2%} accuracy")

    summary = {
        "pca_95pct_components": n95,
        "chance_accuracy": 1 / len(category_names),
        "channel_results": {str(k): v["final_accuracy"] for k, v in results.items()},
        "category_names": category_names,
        "n_samples": int(X.shape[0]),
    }
    with open(OUTPUT / "virtual_eeg_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
