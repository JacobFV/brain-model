# I Built a Brain Scanner for UI Design

## visual language

- **chair** — main talking position, relaxed, slightly off-center
- **close** — up-close face, for weight and emphasis
- **screen** — screen recording, terminal, code
- **broll** — full-frame generated visual (brain heatmap, mockup, chart)
- **split** — side-by-side comparison (mockup|mockup or mockup|brain)
- **overlay** — face in corner over broll (picture-in-picture)
- **brain3d** — rotating 3D brain animation, full or partial frame
- **ux-anim** — animated UI mockup (scrolling, clicking, popups appearing)
- **black** — end card

---

## the script

<!-- INTENT: the viewer should feel like they stumbled into a late-night
     conversation with someone who found something genuinely unsettling.
     not a lecture. not a pitch. the energy of "dude, look at this."
     the thesis-level compression: bad design is not an inconvenience,
     it is a measurable assault on the dynamics of cognition. and we
     just got the instruments to see it. -->

[chair]
So Meta recently released v2 of their fMRI scan prediction neural network TRIBE.

[close]
fMRIs are these scans that show where in the brain blood is moving around and the TRIBEv2 is a neural network that predicts how 20,000 different points on your brain will be activated by visual, audio, and text stimuli.

<!-- VIEWER FEELING: "ok, that's a wild capability. what did he do with it?" -->

[overlay: face bottom-right, screen filling rest — terminal running, brain images rendering in real-time, progress bars, heatmaps appearing]
so obviously i immediately started feeding it descriptions of user interfaces. like, what does a clean google-style search page do to your brain versus a page covered in popups, auto-playing video, flashing banners, cookie consent walls — the whole nightmare.

[split: left=ux-anim of clean search page (cursor typing, results appearing cleanly) | right=ux-anim of cluttered page (popups spawning, banners flashing, chat widget bouncing, cookie wall sliding up)]
and the answer is: completely different brains.

[split → brain3d: the split mockups shrink to top corners, two rotating brain heatmaps fade in below each one — clean brain on left (moderate warm glow), cluttered brain on right (frontal cortex blazing)]
the cluttered version JACKS your frontal cortex. your brain is working overtime just to find the content.

[brain3d: diff brain rotating, red regions pulsing gently — text label "frontal overload" fading in]
it's not engagement. it's struggle.

<!-- VIEWER FEELING: shift. this isn't a tech demo anymore. something is wrong. -->

[close — lean in, slower]
but here's where it gets actually disturbing.

[chair]
i tested dark checkout patterns. you know — the pre-checked insurance, the "you're about to miss out" copy, hidden fees that show up at the last step. and i compared them to an honest checkout. three items, clear price, one button, pay now.

[split: left=ux-anim honest checkout (clean items, total, green "Pay Now" button) | right=ux-anim dark checkout (items appearing, then sneaky checkboxes fading in pre-checked, price climbing, small gray remove links)]

the dark version FLIPS your brain's reward signal positive.

<!-- VIEWER FEELING: wait, what? the BAD one feels GOOD? -->

[close — dead serious]
your orbital frontal cortex — the part that goes "ooh nice" — it lights up MORE for the manipulative version.

[brain3d: slow rotation, reward regions glowing red, label "orbital frontal cortex" with arrow — then the honest checkout brain fades in as ghost overlay for comparison, reward regions dim]
your brain is literally rewarding you for being tricked. that's the mechanism. that's WHY dark patterns work even when you KNOW they're dark patterns.

<!-- VIEWER FEELING: genuine unease. the interface isn't just annoying —
     it's exploiting a circuit you can't override by knowing about it.
     this is the gut punch. let it breathe. -->

[chair — slight pause before speaking, reset energy from disturbed → analytical]
and then i tested flow. a clean code editor — fullscreen, instant autocomplete, no distractions.

[ux-anim: code editor, cursor typing, autocomplete popping up instantly, test bar at bottom showing green checks — fluid, satisfying]

versus one that hits you with a save dialog every two minutes and has a notification badge in the corner.

[ux-anim: same editor but dialog pops up mid-typing "Save your work?", red "47" badge pulses in corner, code fades to half-opacity behind dialog]

[split: both editors top half | both brains bottom half — flow brain (moderate, even) vs interrupted brain (frontal blazing, widespread activation)]
the interrupted version doubles your cognitive load.

[overlay: face in corner, brain3d filling frame — diff brain rotating, all red]
but here's the trap — engagement metrics go UP too.

[close]
so if you're a company optimizing for engagement, you converge on the thing that makes brains struggle. the metric lies.

<!-- VIEWER FEELING: the system is broken. engagement ≠ wellbeing.
     measuring the wrong thing produces the wrong outcome. now what? -->

[chair — energy shifts. no longer reporting. building.]
so i did something about it. i derived eight brain-based signals from the data.

[broll: HIX dashboard heatmap, each signal name animating in one by one with its row lighting up — cognitive load, frustration, flow, trust, reward, confusion, engagement, spatial clarity]
cognitive load. frustration. flow. trust. reward. confusion. engagement. spatial clarity. i'm calling them HIX signals — human interface experience.

[broll: closed-loop diagram animating — "UI Generator" box → arrow → "TRIBE v2" box → arrow → "HIX Signals" box → gradient arrow curving back to generator. labels fade in sequentially]
the idea is: these become the reward function for autonomous UI design. not "did users click" but "did users' brains THRIVE." and you don't need real users in the loop — the brain model IS the virtual user.

<!-- VIEWER FEELING: oh. this isn't just analysis. he built the feedback loop.
     this could actually change how interfaces get designed. -->

[close — weight]
but you need trust as a HARD constraint.

[overlay: face corner, brain3d of dark pattern reward response rotating behind]
because if you optimize purely for reward, you literally converge on dark patterns. the brain rewards manipulation. so you have to build the ethics into the math.

<!-- VIEWER FEELING: the dark pattern finding from earlier
     just became an engineering constraint. the disturbing thing
     became the design principle. that's satisfying. -->

[chair — open, direct, no performance]
all the code is open source. every experiment, every visualization, the whole pipeline.

[close]
because i think being able to see what design does to brains should be a public capability. not a competitive moat.

<!-- VIEWER FEELING: this person is serious. this isn't content. this is a tool
     they're giving away because they think it matters. -->

[chair — direct to camera. no smile. slow.]
bad UX isn't just annoying.

[brain3d: slow dissolve from clean-UI brain (warm, moderate) to cluttered-UI brain (frontal blazing, overloaded) — the transformation happening in real time on a rotating brain]
it is measurably violent to cognition.

[hold on face, 2 beats of silence]

and now we can see it.

[black: github link centered, handle below, brain heatmap ghost-fading behind]

---

## structure

| time | visual | audio | viewer feeling |
|------|--------|-------|----------------|
| 0-3s | chair | hook: TRIBE v2 | curiosity |
| 3-6s | close | fMRI explainer | "ok that's wild" |
| 6-12s | overlay (face+screen) → split (mockups) | feeding it UIs | anticipation |
| 12-16s | split→brain3d (rotating diff) | "not engagement — struggle" | concern |
| 16-17s | close | "actually disturbing" | tension |
| 17-21s | chair → split (checkout mockups animating) | dark pattern setup | dread |
| 21-28s | close → brain3d (reward regions) | reward flip | gut punch |
| 28-34s | ux-anim → split (editors+brains) | flow / cognitive load | recognition |
| 34-36s | overlay → close | "the metric lies" | anger |
| 36-40s | chair | "did something about it" | relief, builder energy |
| 40-46s | broll (HIX dashboard + loop diagram) | HIX signals | excitement |
| 46-50s | close → overlay (face+dark brain) | trust constraint | resolution |
| 50-54s | chair → close | open source | respect |
| 54-58s | chair + brain3d dissolve | "violent to cognition" | weight |
| 58-60s | black | silence → link | stillness |

## face time: ~22s of 60s
## 3d brain time: ~18s
## ux animation time: ~12s
## split-screen time: ~14s

## recording notes
- the whole thing should feel like ONE continuous thought, not segments
- chair shots: you're talking to ONE person across a table
- close shots: you've stopped talking to a room and started talking to THEM
- the "reward flip" at 21-28s is the emotional center — play it like you're still processing it yourself
- "so i did something about it" is the turn — shift from witness to engineer
- the close: no smile, no wrap-up energy, no "thanks for watching." just the claim and silence
- the 2 beats of silence before "and now we can see it" is the most important moment in the video
