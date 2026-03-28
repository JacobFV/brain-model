# Your Brain on Bad UX: What a Neural Model Reveals About Interface Design

*Using Meta's TRIBE v2 to derive brain-based signals for autonomous UI optimization*

---

## The Premise

What if you could measure how a user's brain responds to your interface — without putting them in a brain scanner? Meta's TRIBE v2 model, released March 26, 2026, predicts fMRI brain activity from text, audio, and video with 70x higher resolution than its predecessor. We used it to simulate what happens in the cortex when users encounter 26 different UI/UX scenarios, ranging from clean minimalist design to dark-pattern-laden checkout flows.

The result: a set of eight **HIX (Human Interface eXperience) proxy signals** — brain-derived metrics that could serve as reward functions for closed-loop autonomous interface design.

---

## The Experiment

We described 26 UI/UX scenarios to TRIBE v2 as naturalistic text — the way a user might narrate their experience. Each description was converted to speech, processed through the model's tri-modal architecture (LLaMA 3.2 for text, Wav2Vec-BERT for audio), and mapped onto 20,484 cortical vertices on the fsaverage5 brain mesh.

We then extracted activation levels from 20 named brain regions and computed weighted signal composites.

### Scenarios tested

| Category | Good UX | Bad UX |
|----------|---------|--------|
| Cognitive load | Clean search page | Flashing banners, popups, auto-play |
| Dark patterns | Honest unsubscribe | 15-day wait, guilt-trip copy |
| Checkout | Clear prices, one button | Pre-checked insurance, hidden fees |
| Errors | Explains problem + fix | `Error 0xE8000015` |
| Failure | Auto-saves, resumes | Three hours of work, gone |
| Gamification | Meaningful progress bar | Streak timers, FOMO badges |
| Navigation | Good headings + TOC | One giant unstructured page |
| Search | First result is correct | Hundreds of irrelevant results |
| Flow | Fullscreen editor, instant feedback | Interruptions every 2 minutes |
| Accessibility | Visual dashboard | Screen reader narration |
| Complexity | Progressive disclosure | 240 toggles, flat list |

---

## The Eight HIX Signals

We define eight proxy signals as weighted combinations of brain region activations. Each maps to a measurable aspect of user experience:

```
HIX Signal          Key Brain Regions (positive)         Interpretation
─────────────────────────────────────────────────────────────────────────
cognitive_load      Frontal sup, Frontal mid, Broca's    high = hard to process
frustration         Insula, Cingulate ant                high = blocked/confused
engagement          Wernicke's, Temporal mid              high = sustained attention
reward              Orbital frontal, Cingulate ant        high = delighted
confusion           Cingulate ant, Frontal mid            high = lost/disoriented
flow                Frontal sup, Broca's, Motor           high = effortless focus
trust               Temporal pole, Precuneus              high = feels safe
spatial_clarity     Parahippocampal, Occipital            high = easy to navigate
```

### HIX Dashboard

![HIX Dashboard](brain_ux/hix_dashboard.png)

*Heatmap of all 8 HIX signals across 26 UX conditions. Green = desirable, Red = problematic. Each column is normalized independently.*

---

## Key Findings

### 1. Flow-breaking UI doubles cognitive load

| | Cognitive Load | Engagement | Flow |
|---|---|---|---|
| **Flow-enabling** (fullscreen editor, instant feedback) | 0.264 | 0.710 | 0.353 |
| **Flow-breaking** (interruptions, slow autocomplete) | **0.624** | 0.814 | 0.449 |

The flow-breaking interface drives cognitive load up **2.4x** while paradoxically *also* increasing engagement. The brain is working harder to maintain task focus despite constant interruptions. This is not productive engagement — it's the neural signature of fighting your tools.

**Design implication**: High engagement is not always good. If cognitive load rises with it, the user is struggling, not flowing.

### 2. Dark patterns hijack the reward circuit

| | Reward Signal | Cognitive Load |
|---|---|---|
| **Honest checkout** | -0.048 (negative) | 0.494 |
| **Dark checkout** | **+0.056** (positive) | 0.553 |

The dark checkout *flips the reward signal positive*. Pre-checked add-ons and "exclusive deals" trigger the brain's reward circuitry even as the user is being manipulated. This is the neural mechanism dark patterns exploit: they create a reward response that masks the user's actual best interest.

**Design implication**: Reward signal alone is insufficient for ethical UX optimization. A system optimizing purely for neural reward would converge on dark patterns. You need the *ratio* of reward to frustration, or explicit trust/confusion constraints.

### 3. Cryptic errors suppress spatial clarity

| | Spatial Clarity | Frustration |
|---|---|---|
| **Helpful error** ("file too large, compress?") | 0.148 | 0.075 |
| **Cryptic error** (`0xE8000015`) | 0.059 | 0.091 |

When users encounter a cryptic error, their spatial clarity signal drops — the brain's navigational system *disengages*. The user literally loses their sense of where they are in the interface. Helpful errors with actionable next steps preserve spatial orientation.

### 4. Manipulative gamification vs meaningful progress

| | Engagement | Trust | Reward |
|---|---|---|---|
| **Meaningful progress** | 0.589 | -0.068 | -0.017 |
| **Manipulative gamification** | **0.735** | **-0.199** | **+0.098** |

Manipulative gamification (streak counters, FOMO, social comparison) drives higher engagement (+25%) and positive reward, but **craters trust** (-193%). The brain responds to the game mechanics, but the trust circuit recognizes the manipulation.

**Design implication**: Gamification that optimizes for engagement at the expense of trust will eventually lose users. The trust signal is a leading indicator of churn.

---

## Paired Comparisons: Good vs Bad UX

![HIX Paired](brain_ux/hix_paired.png)

*Side-by-side HIX signal comparison for 8 paired UX scenarios. Green = good UX, Red = bad UX.*

---

## Brain Region Responses

![Brain ROI Comparisons](brain_ux/ux_paired_comparisons.png)

*Per-region activation differences for 11 paired comparisons. Red bars = bad UX activates region more. Blue bars = good UX activates region more.*

Notable patterns:
- **Insula** (emotional salience/disgust) consistently activates more for bad UX
- **Temporal mid.** (narrative comprehension) activates more for engaging, well-structured content
- **Cingulate ant.** (conflict monitoring) spikes for ambiguous or deceptive patterns

---

## Toward Autonomous Design

These eight signals form a differentiable reward function. A system that generates UI layouts and copy could optimize against TRIBE v2 predictions:

```
loss = (
    α₁ * cognitive_load      # minimize
  + α₂ * frustration          # minimize
  - α₃ * engagement           # maximize
  - α₄ * reward               # maximize (with constraint)
  + α₅ * confusion            # minimize
  - α₆ * flow                 # maximize
  - α₇ * trust                # maximize (hard constraint)
  - α₈ * spatial_clarity      # maximize
)
```

With trust as a hard constraint (never decrease), this prevents convergence toward dark patterns while still optimizing for user satisfaction.

### The closed loop

```
┌─────────────┐     text/image      ┌───────────┐     fMRI prediction     ┌─────────────┐
│  UI Design  │ ──────────────────→ │  TRIBE v2  │ ──────────────────────→ │ HIX Signals │
│  Generator  │                     │            │                         │  Extractor  │
└──────┬──────┘                     └────────────┘                         └──────┬──────┘
       │                                                                          │
       │                        gradient / reward signal                          │
       └──────────────────────────────────────────────────────────────────────────┘
```

The generator proposes UI descriptions. TRIBE v2 predicts brain responses. HIX signals score the experience. Gradients flow back to improve the design. No real users needed in the loop — the brain model acts as a virtual user-testing oracle.

---

## Limitations

1. **Text-mediated**: We describe UIs in words, not render them visually. TRIBE v2 also accepts video input, which could enable testing actual screenshots or prototypes.

2. **Average brain**: TRIBE v2 predicts the *average* subject's response. Individual differences in UI preferences (expert vs novice, accessibility needs) are averaged out.

3. **Simulated, not measured**: These are model predictions, not real fMRI data. The model was trained on movies/podcasts, not UI interactions. Transfer to the UX domain needs validation.

4. **Ethical boundary**: A system that can predict and optimize brain responses to interfaces could be used for manipulation as easily as for good design. The trust constraint is a start, but governance matters.

---

## Conclusion

Bad UX doesn't just frustrate users — it measurably disrupts spatial reasoning, hijacks reward circuits, and erodes trust at the neural level. The eight HIX proxy signals provide a vocabulary for these effects and a foundation for building design systems that optimize for genuine human experience rather than engagement theater.

The code and data are at [github.com/jacob/brain-model](https://github.com/jacob/brain-model).

---

*Built with Meta TRIBE v2 and Modal GPU infrastructure. Brain visualizations rendered with nilearn on fsaverage5 cortical mesh.*
