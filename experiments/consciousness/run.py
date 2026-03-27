#!/usr/bin/env python3
"""
Experiment: Forward Dynamics of Brain Activity (Consciousness Prediction)

Learn the brain's transition function: given cortical state at time t,
predict cortical state at time t+1. TRIBE v2 gives us time-series of
brain states during stimulus processing. We train a model to predict
the next brain state from the current one.

Key questions:
- How predictable are brain dynamics? (prediction loss as a function of horizon)
- Are some brain regions more "autonomous" (predictable from own past)?
- Does the brain have attractor states? (convergence of different trajectories)
- Can we identify the "surprise" moments where brain dynamics deviate from prediction?
- What is the effective dimensionality of brain dynamics?
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.model import load_model, text_to_predictions, CACHE_FOLDER

OUTPUT = Path("/work/output") if Path("/work").exists() else Path("./output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Long-form stimuli that produce multi-timestep trajectories
TRAJECTORIES = {
    "story_arc": """
        It was a quiet morning when the letter arrived. At first she thought it was junk mail.
        But the envelope was heavy, cream-colored, with her name in calligraphy. Inside was
        an invitation to a place she had only dreamed about. Her hands trembled as she read
        the details. She would need to leave everything behind. The choice was impossible and
        inevitable. She packed a single bag and walked out the door without looking back.
        The train carried her through landscapes she had never imagined. Mountains gave way
        to desert gave way to ocean. When she finally arrived, she understood why she had
        been chosen. The place was not a destination but a mirror, and in it she saw the person
        she had always been but never had the courage to become.
    """,
    "escalating_tension": """
        The footsteps started on the floor above. Slow, measured, deliberate. Then they stopped.
        Silence. The lights flickered once. The footsteps resumed, closer now, on the stairs.
        Each step creaked under weight. I held my breath. The hallway light went dark. The
        footsteps reached the landing. A long pause. Then the doorknob turned, slowly, testing.
        I pressed my back against the wall. The door opened an inch. Two inches. A shadow
        spilled across the floor. And then a voice, soft and familiar, said my name.
    """,
    "philosophical_journey": """
        Consider this: you have never experienced the present moment. By the time your neurons
        fire and your consciousness registers what is happening, the moment has already passed.
        You live perpetually in the recent past, a reconstruction assembled by your brain from
        fragmentary signals. And yet this reconstruction feels seamless, immediate, real. Your
        brain is the most sophisticated storyteller in the universe, narrating reality to you
        in real time, and you have never once questioned the narrator. Until now.
        What if you could step outside the story? What if you could see the raw data before
        your brain interprets it? Would it look like anything at all?
    """,
    "sensory_sequence": """
        First, the smell of pine needles warmed by afternoon sun. Then the sound of a stream
        tumbling over smooth stones. The feel of cool water on bare feet. A flash of blue as
        a kingfisher dives. The taste of wild blackberries, tart and sweet, staining your
        fingers purple. The crunch of dry leaves underfoot as the trail climbs. Wind carrying
        the scent of rain from the west. Thunder, still far away. The first cold drops on
        warm skin. And then the downpour, all senses overwhelmed at once.
    """,
    "mathematical_beauty": """
        Euler's identity connects five fundamental constants in a single equation of impossible
        elegance: e to the power of i times pi, plus one, equals zero. The base of natural
        logarithms, the imaginary unit, the ratio of a circle to its diameter, the multiplicative
        identity, and the additive identity, all woven together in a statement that is not just
        true but somehow beautiful, as if mathematics itself has an aesthetic dimension that
        transcends human invention. Mathematicians have called it the most beautiful equation
        ever written. It suggests a deep structure to reality that we can glimpse but not
        fully comprehend.
    """,
    "mundane_routine": """
        Monday morning. The alarm goes off at six thirty. Snooze. It goes off again at six
        thirty-nine. Get up. Shower. Brush teeth. Coffee. Check email. Nothing urgent. Toast
        with butter. Put on shoes. Lock the door. Walk to the station. Same platform, same
        spot. Train arrives. Stand for twenty minutes. Get off. Walk four blocks. Badge in.
        Elevator to the third floor. Sit down. Turn on computer. Wait for it to boot. Open
        the same five tabs as yesterday.
    """,
}


class BrainDynamicsModel(nn.Module):
    """Predict brain state at t+1 from brain state at t (and optionally t-1, t-2...)."""

    def __init__(self, n_vertices: int = 20484, hidden: int = 512, context_steps: int = 3):
        super().__init__()
        self.context_steps = context_steps
        self.encoder = nn.Sequential(
            nn.Linear(n_vertices * context_steps, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(hidden, n_vertices)

    def forward(self, x_context):
        """x_context: (batch, context_steps * n_vertices)"""
        h = self.encoder(x_context)
        return self.predictor(h)


def prepare_dynamics_data(trajectories: dict[str, np.ndarray], context_steps: int = 3):
    """Convert trajectory predictions into (context, target) pairs for training."""
    X_list, y_list, traj_labels = [], [], []

    for name, preds in trajectories.items():
        T = preds.shape[0]
        for t in range(context_steps, T):
            context = preds[t - context_steps:t].flatten()  # (context_steps * n_vertices,)
            target = preds[t]  # (n_vertices,)
            X_list.append(context)
            y_list.append(target)
            traj_labels.append(name)

    return np.stack(X_list), np.stack(y_list), traj_labels


def train_dynamics_model(X, y, n_vertices, context_steps=3, n_epochs=500):
    """Train brain dynamics prediction model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    n = len(X_t)
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    model = BrainDynamicsModel(n_vertices=n_vertices, context_steps=context_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {"train_loss": [], "test_loss": []}

    for epoch in range(n_epochs):
        model.train()
        pred = model(X_t[train_idx])
        loss = nn.functional.mse_loss(pred, y_t[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_t[test_idx])
            test_loss = nn.functional.mse_loss(test_pred, y_t[test_idx])

        history["train_loss"].append(float(loss))
        history["test_loss"].append(float(test_loss))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: train={loss:.6f} test={test_loss:.6f}")

    return model, history


def analyze_predictability(trajectories: dict[str, np.ndarray]):
    """Analyze how predictable each trajectory and brain region is."""
    from core.roi import get_roi_indices
    roi_idx = get_roi_indices()

    analysis = {}

    for name, preds in trajectories.items():
        T = preds.shape[0]
        if T < 3:
            continue

        # Autocorrelation: how similar is t+1 to t?
        autocorr = []
        for t in range(T - 1):
            corr = np.corrcoef(preds[t], preds[t + 1])[0, 1]
            autocorr.append(float(corr))

        # Per-ROI temporal variability
        roi_dynamics = {}
        for roi_name, idx in roi_idx.items():
            timeseries = preds[:, idx].mean(axis=1)
            roi_dynamics[roi_name] = {
                "temporal_std": float(np.std(timeseries)),
                "autocorrelation": float(np.corrcoef(timeseries[:-1], timeseries[1:])[0, 1]) if T > 2 else 0,
                "range": float(timeseries.max() - timeseries.min()),
            }

        # Effective dimensionality of dynamics
        from sklearn.decomposition import PCA
        if T > 3:
            pca = PCA(n_components=min(T - 1, 20))
            pca.fit(preds)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            eff_dim = int(np.argmax(cumvar >= 0.95)) + 1
        else:
            eff_dim = T

        # State velocity (how fast brain state changes)
        velocities = np.linalg.norm(np.diff(preds, axis=0), axis=1)

        analysis[name] = {
            "n_timesteps": T,
            "mean_autocorrelation": float(np.mean(autocorr)),
            "min_autocorrelation": float(np.min(autocorr)),
            "effective_dimensionality": eff_dim,
            "mean_velocity": float(velocities.mean()),
            "max_velocity": float(velocities.max()),
            "max_velocity_timestep": int(np.argmax(velocities)),
            "roi_dynamics": roi_dynamics,
        }

    return analysis


def plot_dynamics(history, analysis, trajectories, output_dir):
    """Generate dynamics visualizations."""
    # Training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train", alpha=0.7)
    ax.plot(history["test_loss"], label="Test", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Brain Dynamics Prediction: Can We Predict the Next Brain State?", fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(output_dir / "dynamics_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # State velocity per trajectory
    names = list(trajectories.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        preds = trajectories[name]
        velocities = np.linalg.norm(np.diff(preds, axis=0), axis=1)
        ax.plot(velocities, "o-", linewidth=2, markersize=4, color="#e41a1c")
        ax.fill_between(range(len(velocities)), velocities, alpha=0.2, color="#e41a1c")
        ax.set_ylabel("State velocity")
        ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="bold")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Brain State Velocity: How Fast Does the Brain Change?", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "state_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Predictability comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    traj_names = [n for n in analysis.keys()]
    autocorrs = [analysis[n]["mean_autocorrelation"] for n in traj_names]
    dims = [analysis[n]["effective_dimensionality"] for n in traj_names]

    x = range(len(traj_names))
    ax.bar(x, autocorrs, color="#377eb8")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in traj_names], fontsize=9)
    ax.set_ylabel("Mean autocorrelation (t → t+1)")
    ax.set_title("How Predictable Are Brain Dynamics for Each Stimulus?", fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, dims, "ro-", markersize=8, label="Effective dimensionality")
    ax2.set_ylabel("Effective dimensionality (95% var)", color="red")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "predictability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for path in ["dynamics_training.png", "state_velocity.png", "predictability.png"]:
        print(f"Saved: {output_dir / path}")


def main():
    model = load_model()

    # Generate trajectories
    print("=== Generating brain trajectories ===")
    trajectories = {}
    for name, text in TRAJECTORIES.items():
        preds, _ = text_to_predictions(model, text, label=name)
        trajectories[name] = preds

    # Analyze predictability
    print("\n=== Analyzing brain dynamics predictability ===")
    analysis = analyze_predictability(trajectories)

    # Train dynamics model
    print("\n=== Training forward dynamics model ===")
    context_steps = 3
    n_vertices = next(iter(trajectories.values())).shape[1]
    X, y, labels = prepare_dynamics_data(trajectories, context_steps=context_steps)
    print(f"Training data: {X.shape[0]} samples, context={context_steps} steps")

    dyn_model, history = train_dynamics_model(X, y, n_vertices, context_steps=context_steps)

    # Results
    final_test_loss = history["test_loss"][-1]
    baseline_loss = float(np.mean((y[1:] - y[:-1]) ** 2))  # predict no change
    print(f"\nFinal test MSE: {final_test_loss:.6f}")
    print(f"Baseline (predict no change): {baseline_loss:.6f}")
    print(f"Improvement over baseline: {(1 - final_test_loss / baseline_loss) * 100:.1f}%")

    # Save
    np.savez(OUTPUT / "brain_trajectories.npz", **{k: v for k, v in trajectories.items()})
    plot_dynamics(history, analysis, trajectories, OUTPUT)

    summary = {
        "final_test_mse": final_test_loss,
        "baseline_mse": baseline_loss,
        "improvement_pct": (1 - final_test_loss / baseline_loss) * 100,
        "context_steps": context_steps,
        "n_training_samples": int(X.shape[0]),
        "trajectory_analysis": {k: {kk: vv for kk, vv in v.items() if kk != "roi_dynamics"}
                                for k, v in analysis.items()},
    }
    with open(OUTPUT / "consciousness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== KEY FINDINGS ===")
    for name, data in analysis.items():
        print(f"  {name}: autocorr={data['mean_autocorrelation']:.3f}, "
              f"dim={data['effective_dimensionality']}, "
              f"velocity={data['mean_velocity']:.4f}")


if __name__ == "__main__":
    main()
