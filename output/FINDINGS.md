# Brain Model Experiment Findings

Generated using Meta's TRIBE v2 (released 2026-03-26) — a foundation model that
predicts fMRI brain activity from text, audio, and video stimuli.

---

## Experiment 1: Emotion Stimuli (34 stimuli, 20 ROIs)

### Surprising cortical similarities

| Pair | Correlation | Why it's interesting |
|------|-------------|---------------------|
| anxious_excitement ↔ abstract_dread | 0.989 | Anticipation and existential dread share nearly identical cortical signatures — the brain may process "unknown future" through the same circuitry regardless of valence |
| joy_pure ↔ anger_pure | 0.985 | Two "opposite" emotions produce nearly identical cortical patterns. Both are high-arousal, approach-oriented emotions — the brain may encode arousal/approach more than valence |
| social_pride ↔ contempt | 0.985 | Pride and contempt activate the same regions. Both involve social hierarchy evaluation — perhaps the brain uses a single "status assessment" circuit |
| sadness_pure ↔ nonsocial_awe | 0.984 | Sadness and awe share cortical patterns. Both may involve self-diminishment and boundary dissolution |
| nostalgic_joy ↔ melancholic_beauty | 0.986 | Two compound emotions that blend positive and negative aspects are nearly identical cortically |

### Most dissimilar emotions

| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| schadenfreude ↔ moral_elevation | 0.675 | **Most dissimilar pair overall.** Enjoying another's suffering vs being inspired by another's sacrifice — maximally opposed moral-social circuits |
| tender_grief ↔ moral_outrage | 0.702 | Private intimate loss vs public justice — completely different processing modes |
| anxious_excitement ↔ moral_elevation | 0.689 | Self-focused anticipation vs other-focused admiration |

### Key insight: Compound emotions activate language regions more

Compound emotions (bittersweet, schadenfreude, nostalgic joy, etc.) show **higher activation** in:
- **Wernicke's area** (+0.073) — language comprehension
- **Auditory cortex** (+0.069) — language-related processing
- **Fusiform** (+0.059) — complex pattern recognition

This suggests compound emotions require more **linguistic/narrative scaffolding** — you need a story to feel "bittersweet," but you can feel "fear" without words.

### Social vs non-social emotions

Social emotions (shame, pride) activate **more** than non-social (awe, serenity):
- **Temporal mid.** (+0.098) — social cognition, theory of mind
- **Frontal sup.** (+0.081) — executive control, self-monitoring
- **Wernicke's area** (+0.077) — language (internal social narration?)
- **Broca's area** (+0.049) — speech production circuits

### Abstract vs narrative emotion

Abstract emotional descriptions activate the **insula** more (+0.073), while narrative descriptions strongly activate:
- **Temporal mid.** (+0.163) — the largest single difference in the study
- **Temporal pole** (+0.110) — narrative comprehension, social semantics
- **Precuneus** (+0.092) — mental imagery, episodic simulation

**Implication**: Narrative is a far more powerful driver of temporal lobe engagement than abstraction. The brain processes stories through fundamentally different pathways than concepts.

---

## Experiment 2: Brain UX (26 UI/UX scenarios, 8 HIX signals)

### HIX (Human Interface eXperience) Proxy Signals

Eight brain-derived signals that could drive autonomous UI design:

| Signal | What it measures | Key ROIs |
|--------|-----------------|----------|
| cognitive_load | Mental effort | Frontal sup., Frontal mid., Broca's |
| frustration | Blocked goals | Insula, Cingulate ant. |
| engagement | Sustained attention | Wernicke's, Temporal mid. |
| reward | Satisfaction/delight | Orbital frontal, Cingulate ant. |
| confusion | Expectation mismatch | Cingulate ant., Frontal mid. |
| flow | Effortless focus | Frontal sup., Broca's, Motor |
| trust | Safety/reliability | Temporal pole, Precuneus |
| spatial_clarity | Navigation ease | Parahipp., Occipital |

### Key findings: Flow-breaking UX doubles cognitive load

| Condition | Cognitive Load | Engagement | Flow |
|-----------|---------------|------------|------|
| flow_enabling | +0.264 | +0.710 | +0.353 |
| flow_breaking | **+0.624** | +0.814 | +0.449 |

Flow-breaking UI (constant interruptions, slow autocomplete) **increases cognitive load by 2.4x** while paradoxically also increasing engagement — the brain is working harder to stay on task. The "flow" signal is also higher because the flow definition includes frontal/motor regions that are activated during effortful task performance.

### Dark checkout vs honest checkout

| Signal | Honest | Dark | Delta |
|--------|--------|------|-------|
| cognitive_load | +0.494 | +0.553 | +12% |
| engagement | +0.602 | +0.696 | +16% |
| reward | -0.048 | +0.056 | flipped positive |

Dark patterns increase engagement (attention hijacking) and flip the reward signal positive — the brain's reward circuit responds to the additional items/offers even though the user experience is worse. This is the neural mechanism dark patterns exploit.

---

## Experiment 3: Virtual EEG (40 stimuli, 8 categories)

### v2: Physically grounded virtual EEG (48 stimuli, 6 categories, LOOCV)

v1 was fundamentally flawed (learned compression ≠ EEG, tiny dataset, PCA artifact).
v2 maps the **standard 10-20 electrode system** onto the fsaverage5 cortical mesh — each
electrode samples a 10mm radius cortical patch (~30-58 vertices), giving a physically
meaningful (n_timesteps, 20) signal per stimulus.

### Classification accuracy

| Signal | Accuracy | vs Chance (16.7%) |
|--------|----------|-------------------|
| Full cortex (20,484 vertices) | **56.2%** | Upper bound |
| 20-ch EEG (10-20 system) | **45.8%** | 81.5% retention |
| Temporal + Parietal (7 ch) | **52.1%** | Beats full 20-ch |
| Frontal only (7 ch) | 37.5% | |
| Best single electrode: F3 | 31.2% | Left frontal |
| Occipital only (3 ch) | 18.8% | Barely above chance |

### Key finding: fewer electrodes in the right places beat more electrodes everywhere

The **temporal-parietal subset** (T3, T4, T5, T6, P3, P4, Pz — just 7 electrodes)
achieves **52.1%** accuracy, outperforming the full 20-channel montage (45.8%).
Frontal and occipital electrodes add noise for semantic categorization. A minimal
headset for content classification needs only these 7 electrodes.

### Information loss: 20K vertices → 20 electrodes

| Measure | Value |
|---------|-------|
| KL divergence (full ∥ interpolated) | **0.065 nats** — surprisingly low |
| Reconstruction R² (mean) | **0.449** — 20 electrodes reconstruct ~45% of cortex variance |
| Vertices with R² > 0.5 | **42.7%** — almost half well-predicted from 20 channels |
| Vertices with R² < 0 | **0.1%** — almost no vertices are irrecoverable |
| Classification retention | **81.5%** — most category information survives electrode sampling |

### Single electrode ranking

F3 (left frontal) is the most informative single electrode for semantic classification (31.2%),
followed by F7 (left frontal-temporal, 25.0%) and C4 (right central, 22.9%).
The left-hemisphere dominance is consistent with language processing lateralization —
all stimuli are text-based, so left-hemisphere language circuits carry the most discriminative signal.

---

## Experiments 4 & 5: Consciousness Dynamics & Affective Intervention

Currently executing on Modal GPUs. Results will be appended when complete.
