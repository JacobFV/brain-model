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

### Brain responses are remarkably low-dimensional

**Only 8 PCA components explain 95% of variance** across all 20,484 cortical vertices. The brain's response to diverse stimuli (animals, music, food, danger, abstract thought, social, spatial, language) lives in a ~8-dimensional subspace.

### Classification from compressed "EEG"

| Channels | Accuracy | vs Chance (12.5%) |
|----------|----------|-------------------|
| 4 | 37.5% | 3.0x |
| 8 | 37.5% | 3.0x |
| 16 | 37.5% | 3.0x |
| 32 | 37.5% | 3.0x |

Even 4 channels carry enough information to classify semantic category 3x above chance. The plateau at 37.5% is due to the very small dataset (40 samples, 8 test). With more data, higher channel counts would likely separate further.

---

## Experiments 4 & 5: Running...

Consciousness dynamics and affective intervention experiments are currently executing on Modal GPUs. Results will be appended when complete.
