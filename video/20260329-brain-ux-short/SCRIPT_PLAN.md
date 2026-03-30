# Video Script: "I Built a System That Reads Your Brain's Response to UI Design"

**Format:** 60s YouTube Short, vertical 1080x1920
**Tone:** Independent researcher sharing a discovery. Not corporate, not lecture-y. Like texting a friend something wild you found.
**POV:** First person. "I built...", "I found...", "I was shocked when..."

---

## Structure

### 1. HOOK (0-3s)
**[A-roll: face to camera, excited]**

> "I built a system that can see what bad UI does to your brain."

*Visual: Quick flash of brain heatmap behind you*

---

### 2. WHAT I DID (3-10s)
**[B-roll: screen recording of code / terminal running]**

> "Meta just open-sourced a brain model called TRIBE v2. It predicts fMRI brain scans from any stimulus. So I fed it descriptions of different UI experiences — clean designs, dark patterns, error messages — and mapped what lights up."

*Visual: Terminal running experiments, brain images generating*

---

### 3. THE CLEAN VS CLUTTERED FINDING (10-18s)
**[B-roll: side-by-side UI mockups → brain heatmaps]**

> "Clean UI versus cluttered UI — same task, completely different brain. The cluttered version forces your frontal cortex into overdrive. Your brain is literally fighting the interface."

*Visual: Clean search page | Cluttered nightmare → paired brain heatmaps*

---

### 4. THE DARK PATTERN BOMBSHELL (18-28s)
**[A-roll: face to camera, leaning in]**

> "But here's what blew my mind."

**[B-roll: honest checkout vs dark checkout mockup → brain diff]**

> "Dark checkout patterns — the pre-checked insurance, the hidden fees — they actually flip your brain's reward signal positive. Your reward circuit lights up even while you're being manipulated. That's the neural mechanism. That's WHY dark patterns work."

*Visual: Checkout mockups side by side → rotating brain diff GIF*

---

### 5. THE FLOW STATE FINDING (28-36s)
**[B-roll: code editor mockups → brain comparison]**

> "Flow-breaking UI — constant save dialogs, slow autocomplete — doubles your cognitive load. And here's the trap: engagement goes UP too. So metrics say users are engaged, but their brains are struggling, not flowing."

*Visual: Clean editor vs interrupted editor → brain heatmaps*

---

### 6. WHAT I BUILT FROM THIS (36-46s)
**[A-roll: face to camera, matter of fact]**

> "So I derived eight brain signals from this — cognitive load, frustration, flow, trust, reward — things you can actually measure from the model's predictions."

**[B-roll: HIX dashboard heatmap]**

> "I'm calling them HIX signals. Human Interface Experience. They could be the reward function for an AI that designs interfaces autonomously. No user testing needed — the brain model IS the user."

*Visual: HIX dashboard heatmap, then the closed-loop diagram*

---

### 7. CTA (46-53s)
**[A-roll: face to camera]**

> "Code's open source. Link in bio. Bad UX doesn't just annoy you — it measurably disrupts your brain. Now we can see it."

*Visual: GitHub repo, brain hero image*

---

## A-roll segments (where you appear on camera)
- 0-3s: Hook
- 18-20s: "But here's what blew my mind" (2s transition)
- 36-42s: "So I derived eight brain signals..."
- 46-53s: CTA

**Total face time: ~15s out of 53s**

## B-roll assets we already have
- `brain_ux_hero.png` — 4 brains side by side
- `brain_clean_ui_vs_cluttered_ui.png` — paired heatmaps
- `brain_honest_checkout_vs_dark_checkout.png` — checkout comparison
- `brain_flow_enabling_vs_flow_breaking.png` — flow comparison
- `rotating_diff_honest_checkout_vs_dark_checkout.gif` — rotating diff
- `hix_dashboard.png` — HIX signal heatmap
- `hix_paired.png` — good vs bad signal bars
- UI mockups: clean, cluttered, honest checkout, dark checkout, flow, flow-breaking, helpful error, cryptic error

## Assets still needed
- [ ] Your face recordings (4 clips, ~15s total)
- [ ] Screen recording of terminal running experiments (can be faked with a scrolling terminal component)
- [ ] Background music (we have the generated synth track)
- [ ] Optional: better TTS voice once OpenAI key is topped up (currently using gTTS)

## Notes
- Keep energy conversational, not polished. This is a researcher sharing findings, not a brand video.
- The "blew my mind" beat at 18s is the emotional peak — lean into it.
- End strong with the open source angle. Independent researcher energy.
