# E02: Introduction to Hyperscanning — Copilot Directive

## Metadata

- **Notebook ID**: E02
- **Title**: Introduction to Hyperscanning
- **Branch**: `feature/E02_introduction_hyperscanning`
- **Duration**: 70 minutes
- **Prerequisites**: All of Block A-D, E01 (Graph Theory Basics)
- **Next**: F01 (Spectral Coherence)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Define hyperscanning and explain its scientific importance
2. Describe common hyperscanning paradigms and research questions
3. Understand the unique methodological challenges of inter-brain analysis
4. Structure hyperscanning data correctly (within vs between connectivity)
5. Apply appropriate preprocessing for dual-brain recordings
6. Select appropriate connectivity metrics for different research questions
7. Interpret inter-brain synchrony in the context of social neuroscience

---

## Required Imports
```
numpy, matplotlib, scipy.signal
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/connectivity.py, src/graph.py
New: src/hyperscanning.py
```

---

## Section 1: What is Hyperscanning?

**Purpose**: Define the field and its importance.

**Content to cover**:
- **Definition**: Simultaneous recording of brain activity from two or more interacting individuals
- **Etymology**: "hyper" (beyond) + "scanning" (brain imaging)
- **The insight**: Social interaction is a TWO-brain phenomenon
  - Traditional neuroscience: one brain at a time
  - But social cognition happens BETWEEN people!
  - "It takes two to tango" 🧠↔️🧠
- **Historical context**:
  - Montague et al., 2002: first fMRI hyperscanning
  - EEG hyperscanning: more practical, better temporal resolution
  - Now: EEG, fNIRS, fMRI, MEG hyperscanning studies
- **Key question**: Do our brains synchronize during social interaction?
  - And if so: what does it mean? What drives it? What are consequences?

**Key message**: "Hyperscanning lets us study the social brain IN social context."

**Visualization 1**:
- Conceptual illustration:
  - Two heads facing each other
  - EEG caps on both
  - Wavy lines representing brain signals
  - Arrows showing potential synchronization
  - Beautiful, colorful, welcoming
- Caption: "Hyperscanning: studying two brains in interaction"

**Length**: 5-6 paragraphs

---

## Section 2: Why Hyperscanning Matters

**Purpose**: Motivate the approach scientifically.

**Content to cover**:
- **The social brain hypothesis**: 
  - Human brains evolved FOR social interaction
  - Social cognition is our "killer app"
  - But we've mostly studied it in isolated individuals!
- **Limitations of single-brain studies**:
  - Person watches videos of faces → not real interaction
  - Person imagines social scenarios → artificial
  - Missing: real-time, bidirectional, dynamic exchange
- **What hyperscanning reveals**:
  - Inter-brain synchrony correlates with rapport
  - Synchrony predicts cooperation success
  - Leader-follower dynamics visible in brain signals
  - Shared attention creates shared neural patterns
- **Applications**:
  - Education: teacher-student synchrony predicts learning
  - Clinical: therapist-patient rapport assessment
  - Development: parent-child bonding
  - Performance: team coordination in sports, music
  - Human-AI interaction: future frontier!

**Visualization 2**:
- Research domains wheel:
  - Center: "Hyperscanning"
  - Spokes to: Education, Clinical, Development, Music, Sports, Communication
  - Each with small icon
  - Beautiful pastel colors
- Caption: "Hyperscanning applications across domains"

---

## Section 3: Common Hyperscanning Paradigms

**Purpose**: Survey experimental designs.

**Content to cover**:
- **Cooperation tasks**:
  - Joint button pressing (synchronization)
  - Cooperative games
  - Puzzle solving together
  - Question: Does synchrony support cooperation?
- **Communication tasks**:
  - Conversation (structured or free)
  - Storytelling (speaker-listener)
  - Teaching-learning
  - Question: How does information flow between brains?
- **Joint attention tasks**:
  - Looking at same object
  - Shared visual attention
  - Question: Does shared attention synchronize brains?
- **Imitation/synchrony tasks**:
  - Mirror movements
  - Musical coordination (drumming, singing)
  - Dance
  - Question: Does behavioral synchrony drive neural synchrony?
- **Competitive tasks**:
  - Economic games (prisoner's dilemma)
  - Sports competition
  - Question: Does competition also synchronize (or desynchronize)?
- **Naturalistic paradigms**:
  - Free conversation
  - Real-world interactions
  - Most ecologically valid but hardest to analyze

**Visualization 3**:
- Grid of paradigm illustrations:
  - 2×3 grid showing different tasks
  - Cooperation: two people building together
  - Communication: speech bubbles between two heads
  - Joint attention: two people looking at same object
  - Imitation: mirrored movements
  - Competition: game/contest icon
  - Naturalistic: conversation scene
- Caption: "Common hyperscanning experimental paradigms"

---

## Section 4: Inter-Brain Synchrony — The Core Concept

**Purpose**: Define what we're measuring.

**Content to cover**:
- **Inter-brain synchrony (IBS)**:
  - Correlation/coupling between brain signals of two people
  - Same metrics as single-brain connectivity, applied BETWEEN brains
  - PLV, coherence, correlation, etc. — all applicable!
- **What synchrony means**:
  - NOT telepathy! 
  - Shared processing of common stimuli
  - Coupled dynamics from interaction
  - Common physiological rhythms
- **Types of coupling**:
  - **Phase synchrony**: phases align across brains
  - **Amplitude coupling**: power fluctuations correlate
  - **Frequency coupling**: different frequencies can couple
  - **Directed coupling**: information flow (transfer entropy)
- **Key insight from foundations**:
  - Everything we learned applies!
  - Phase (B02) → inter-brain phase synchrony
  - Amplitude (B03) → inter-brain amplitude correlation
  - Coherence (F01, F02) → inter-brain coherence
  - PLV (G01) → inter-brain PLV
  - TE (D03) → inter-brain directed influence

**Visualization 4**:
- Two EEG signals (one from each participant)
- Highlight moments of synchrony:
  - Aligned phases
  - Correlated amplitudes
- Show metric computation between them
- Caption: "Inter-brain synchrony: coupling between two brains"

---

## Section 5: Unique Challenges of Hyperscanning

**Purpose**: Honest assessment of methodological issues.

**Content to cover**:

**Challenge 1: No volume conduction between brains!** ✅
- Actually an ADVANTAGE!
- Single-brain: volume conduction creates spurious connectivity
- Hyperscanning: separate heads, no shared conduction
- Inter-brain synchrony cannot be artifact of volume conduction
- HOWEVER: still need to worry about:
  - Common external stimuli
  - Shared environment artifacts (EMG, movement)

**Challenge 2: Behavioral confounds**
- People synchronize behavior when interacting
- Synchronized movements → synchronized muscle artifacts
- Shared gaze → similar visual processing
- Same speech → similar auditory processing
- Must consider: is neural synchrony BEYOND behavioral synchrony?

**Challenge 3: Stimulus-driven vs interaction-driven synchrony**
- Both participants see same screen → correlated evoked responses
- Not "social" synchrony, just shared input
- Solutions:
  - Pseudo-pair analysis (pair non-interacting participants)
  - Control conditions
  - Focus on frequencies without strong evoked responses

**Challenge 4: Data structure complexity**
- Single brain: n×n connectivity matrix
- Two brains: within-P1, within-P2, AND between
- Need careful organization
- 2n × 2n combined matrix or separate blocks

**Challenge 5: Statistical considerations**
- What's the null hypothesis?
- Pseudo-pairs / shuffled pairs as control
- Multiple comparisons across channel pairs AND participant pairs
- Random effects at participant-pair level

**Visualization 5**:
- Challenge illustration:
  - Two participants watching same screen
  - Both brains respond similarly to stimulus
  - Is this "social" synchrony or just shared input?
  - Diagram showing the confound
- Caption: "Stimulus-driven confound in hyperscanning"

**Visualization 6**:
- Comparison diagram:
  - Left: Single-brain analysis (volume conduction problem)
  - Right: Hyperscanning analysis (no volume conduction, but behavioral confounds)
  - Pros and cons of each
- Caption: "Single-brain vs hyperscanning challenges"

---

## Section 6: Data Structure for Hyperscanning

**Purpose**: Practical data organization.

**Content to cover**:
- **Recording setup**:
  - Two EEG systems (ideally synchronized)
  - Or one system with dual cap
  - Synchronization crucial! (trigger signals, timestamps)
- **Data structure**:
  - Participant 1: (n_channels_P1, n_samples)
  - Participant 2: (n_channels_P2, n_samples)
  - Must align in time
- **Connectivity blocks**:
  - **Within-P1**: connectivity among P1's channels
  - **Within-P2**: connectivity among P2's channels
  - **Between P1→P2**: connectivity from P1 channels to P2 channels
  - **Between P2→P1**: same as above (symmetric for most metrics)
- **Combined matrix structure**:
```
            P1 channels    P2 channels
P1 channels [  Within-P1  |  Between   ]
P2 channels [  Between.T  |  Within-P2 ]
```
- **Channel naming convention**:
  - P1_Fz, P1_Cz, P1_Pz, ...
  - P2_Fz, P2_Cz, P2_Pz, ...
  - Clear labeling prevents confusion!

**Code to produce**:

Function 1: `create_hyperscanning_data_structure`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    channel_names_p1: list[str]
    channel_names_p2: list[str]
Returns:
    dict[str, Any]
    {
        "data_combined": NDArray,  # (n_channels_total, n_samples)
        "channel_names": list[str],  # with P1_, P2_ prefixes
        "n_channels_p1": int,
        "n_channels_p2": int,
        "participant_labels": NDArray  # 1 for P1 channels, 2 for P2 channels
    }
Create unified data structure for hyperscanning analysis
```

Function 2: `extract_connectivity_blocks`
```
Parameters:
    full_matrix: NDArray[np.float64]  # (n_total, n_total)
    n_channels_p1: int
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,  # (n_p1, n_p1)
        "within_p2": NDArray,  # (n_p2, n_p2)
        "between": NDArray     # (n_p1, n_p2)
    }
Extract connectivity blocks from combined matrix
```

Function 3: `combine_connectivity_blocks`
```
Parameters:
    within_p1: NDArray[np.float64]
    within_p2: NDArray[np.float64]
    between: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Combine blocks into full (n_p1 + n_p2) × (n_p1 + n_p2) matrix
```

**Visualization 7**:
- Data structure diagram:
  - Two data arrays (P1, P2)
  - Arrow to combined array
  - Connectivity matrix with labeled blocks
  - Color-coded: within-P1, within-P2, between
- Caption: "Hyperscanning data structure"

---

## Section 7: Preprocessing for Hyperscanning

**Purpose**: Specific preprocessing considerations.

**Content to cover**:
- **Standard EEG preprocessing applies**:
  - Filtering (as in A04)
  - Artifact rejection
  - Bad channel interpolation
  - Re-referencing
- **Hyperscanning-specific considerations**:

**Time synchronization**:
- CRITICAL: both recordings must be temporally aligned
- Check trigger signals match
- Interpolate if sampling rates differ
- Verify alignment before analysis!

**Common reference**:
- Ideally: same reference scheme for both
- Average reference popular for hyperscanning
- Different references can affect between-brain metrics

**Artifact rejection**:
- Reject epochs where EITHER participant has artifacts
- Otherwise comparing clean vs noisy epochs
- May lose more data than single-subject studies

**Movement artifacts**:
- Social interaction involves movement
- Synchronized movements → synchronized artifacts
- Consider: high-pass filter, ICA, motion tracking

**Code to produce**:

Function 4: `synchronize_recordings`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    timestamps_p1: NDArray[np.float64]
    timestamps_p2: NDArray[np.float64]
    target_fs: float
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Interpolate both recordings to common time base
Returns synchronized data arrays
```

Function 5: `reject_epochs_both_participants`
```
Parameters:
    epochs_p1: NDArray[np.float64]  # (n_epochs, n_channels, n_samples)
    epochs_p2: NDArray[np.float64]
    threshold_p1: float
    threshold_p2: float
Returns:
    Tuple[NDArray, NDArray, NDArray[np.bool_]]
    (clean_epochs_p1, clean_epochs_p2, keep_mask)
Reject epochs where EITHER participant exceeds threshold
```

Function 6: `check_synchronization`
```
Parameters:
    trigger_p1: NDArray[np.float64]
    trigger_p2: NDArray[np.float64]
    tolerance_samples: int = 5
Returns:
    dict[str, Any]
    {"is_synchronized": bool, "max_offset": int, "mean_offset": float, "n_triggers_matched": int}
Verify recordings are properly synchronized via trigger channel
```

**Visualization 8**:
- Preprocessing pipeline diagram:
  - Two parallel streams (P1 and P2)
  - Converging at synchronization check
  - Joint artifact rejection
  - Output: synchronized, clean data for both
- Caption: "Hyperscanning preprocessing pipeline"

---

## Section 8: Choosing Connectivity Metrics

**Purpose**: Guide metric selection for research questions.

**Content to cover**:
- **We'll cover many metrics** — how to choose?
- **Decision framework**:

| Question | Metric Type | Examples |
|----------|-------------|----------|
| Are phases aligned? | Phase-based | PLV, PLI, wPLI |
| Do amplitudes co-vary? | Amplitude-based | CCorr, PowCorr |
| Linear coupling in frequency? | Coherence-based | COH, ImCoh |
| Any statistical dependency? | Information-theoretic | MI |
| Who influences whom? | Directed | TE, GC |

- **For hyperscanning specifically**:
  - **PLV**: Most common, intuitive, but sensitive to volume conduction (less issue for between-brain)
  - **Coherence**: Classic, includes both phase and amplitude
  - **ImCoh**: Zero-lag robust (good practice even between brains)
  - **PLI/wPLI**: Robust to volume conduction (less critical between brains but good habit)
  - **Amplitude correlation**: Different mechanism than phase
  - **Transfer entropy**: For directed questions (who leads?)

- **Frequency bands**:
  - Different bands may show different effects
  - Common findings: alpha, theta, gamma all implicated in different tasks
  - Report multiple bands or have hypothesis for specific band

**Visualization 9**:
- Decision tree flowchart:
  - "Do you care about direction?" → Yes: TE/GC, No: continue
  - "Phase or amplitude?" → Phase: PLV/PLI, Amplitude: CCorr
  - "Worried about artifacts?" → Yes: robust metrics
  - Beautiful, clear, helpful
- Caption: "Choosing a connectivity metric"

**Visualization 10**:
- Metric summary table as visual:
  - Rows: PLV, COH, ImCoh, PLI, wPLI, CCorr, PowCorr, MI, TE
  - Columns: Phase-based?, Amplitude-based?, Directed?, Robust to volume conduction?
  - Check marks in appropriate cells
  - Color-coded by metric family
- Caption: "Connectivity metrics at a glance"

---

## Section 9: Pseudo-Pair Analysis — The Key Control

**Purpose**: Essential control for hyperscanning.

**Content to cover**:
- **The fundamental question**: Is synchrony due to INTERACTION?
- **Pseudo-pair analysis**:
  - Take P1 from pair A, P2 from pair B (never actually interacted)
  - Compute "synchrony" between them
  - This is the NULL: synchrony expected by chance
- **If real pairs > pseudo-pairs**:
  - Synchrony is interaction-specific!
  - Not just common stimulus response
  - Not just similar brain structure
- **Implementation**:
  - Create all possible pseudo-pairs (or subsample)
  - Compute connectivity for each pseudo-pair
  - Build null distribution
  - Compare real pairs to null

**Code to produce**:

Function 7: `create_pseudo_pairs`
```
Parameters:
    participant_ids: list[str]  # list of all participant IDs
    real_pairs: list[tuple[str, str]]  # actual interaction pairs
    n_pseudo: int | None = None  # number of pseudo-pairs (None = all possible)
    seed: int | None = None
Returns:
    list[tuple[str, str]]
Generate pseudo-pairs (participants who never interacted)
Excludes actual pairs
```

Function 8: `compute_pseudo_pair_null`
```
Parameters:
    connectivity_dict: dict[tuple[str, str], NDArray]  # pair_id -> connectivity matrix
    real_pairs: list[tuple[str, str]]
    metric_func: Callable  # function to compute summary metric
Returns:
    dict[str, Any]
    {"null_distribution": NDArray, "null_mean": float, "null_std": float}
Build null distribution from pseudo-pairs
```

Function 9: `test_against_pseudo_pairs`
```
Parameters:
    real_value: float
    null_distribution: NDArray[np.float64]
Returns:
    dict[str, float]
    {"pvalue": float, "zscore": float, "percentile": float}
Test real pair value against pseudo-pair null
```

**Visualization 11**:
- Pseudo-pair concept:
  - Top: Real pairs (connected by interaction arrow)
  - Bottom: Pseudo-pairs (no interaction arrow, crossed out)
  - Show that pseudo-pairs share same stimuli but no interaction
- Caption: "Real pairs vs pseudo-pairs"

**Visualization 12**:
- Null distribution from pseudo-pairs
- Real pairs marked as points
- Some significantly above null, some not
- Caption: "Testing real pairs against pseudo-pair null"

---

## Section 10: Interpreting Inter-Brain Synchrony

**Purpose**: What does synchrony actually mean?

**Content to cover**:
- **Caution**: synchrony ≠ communication ≠ understanding
- **Possible mechanisms**:
  
1. **Shared stimulus processing**
   - Both see/hear same thing → similar brain responses
   - Not "social" per se
   - Control with pseudo-pairs

2. **Behavioral coordination**
   - Synchronized actions → synchronized motor/sensory areas
   - Movement artifacts a concern
   - May be genuine coordination signature

3. **Predictive coupling**
   - Brain A predicts brain B's output
   - Brain B adjusts based on A's behavior
   - Interactive loop creates synchrony

4. **Common physiological rhythms**
   - Shared arousal state
   - Breathing, heart rate coupling
   - Less "cognitive" but still social

- **What synchrony correlates with**:
  - Task performance (better coordination = more synchrony)
  - Subjective rapport (feeling "in sync")
  - Learning outcomes (teacher-student)
  - Clinical outcomes (therapist-patient)
- **Direction of causality?**
  - Does synchrony CAUSE better interaction?
  - Or does good interaction CAUSE synchrony?
  - Or both? (bidirectional)
  - Mostly correlational evidence so far

**Visualization 13**:
- Mechanisms diagram:
  - Four panels showing different mechanisms
  - Shared stimulus, behavioral coordination, predictive coupling, physiological
  - Each with distinct visual representation
- Caption: "Mechanisms underlying inter-brain synchrony"

---

## Section 11: HyPyP and Other Tools

**Purpose**: Introduce the production tools.

**Content to cover**:
- **HyPyP** (Hyperscanning Python Pipeline):
  - Developed by our lab!
  - Built on MNE-Python
  - Implements many metrics we'll cover
  - Will compare our implementations to HyPyP
- **Key HyPyP features**:
  - Data preprocessing functions
  - Connectivity computation (PLV, coherence, etc.)
  - Statistical analysis (surrogates, pseudo-pairs)
  - Visualization tools
- **Other tools**:
  - MNE-Python: foundation for EEG analysis
  - mne-connectivity: single-brain connectivity
  - BCT (Brain Connectivity Toolbox): graph metrics
  - EEGLAB (MATLAB): hyperscanning plugins
- **Our approach in this workshop**:
  - Build from scratch to UNDERSTAND
  - Then see how HyPyP does it
  - Prepares you to use AND extend the tools

**Code to produce**:

Function 10: `demonstrate_hypyp_comparison`
```
Parameters:
    None (uses simulated data internally)
Returns:
    dict[str, Any]
    {"our_result": ..., "hypyp_result": ..., "difference": ...}
Template function showing how to compare our implementation to HyPyP
(Actual comparison will be in metric-specific notebooks)
```

**Visualization 14**:
- Tool ecosystem diagram:
  - Center: Your analysis
  - Connected to: MNE-Python, HyPyP, this workshop's code
  - Shows how pieces fit together
- Caption: "Hyperscanning tool ecosystem"

---

## Section 12: A Complete Hyperscanning Analysis Pipeline

**Purpose**: End-to-end workflow.

**Content to cover**:
- **Step 1**: Data loading and synchronization
- **Step 2**: Preprocessing (filter, artifact rejection, re-reference)
- **Step 3**: Create hyperscanning data structure
- **Step 4**: Compute connectivity (within and between)
- **Step 5**: Statistical testing (surrogates, pseudo-pairs)
- **Step 6**: Graph analysis (optional)
- **Step 7**: Visualization and interpretation

**Code to produce**:

Function 11: `hyperscanning_analysis_pipeline`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    channel_names: list[str]
    freq_band: tuple[float, float]
    metric: str = "plv"  # or "coherence", "correlation"
    n_surrogates: int = 500
Returns:
    dict[str, Any]
    {
        "connectivity_within_p1": NDArray,
        "connectivity_within_p2": NDArray,
        "connectivity_between": NDArray,
        "connectivity_full": NDArray,
        "pvalues": NDArray,
        "significant_mask": NDArray,
        "summary_stats": dict
    }
Complete hyperscanning analysis in one function
(For demonstration - in practice, run steps separately)
```

**Visualization 15**:
- Pipeline flowchart:
  - Visual flow from raw data to results
  - Each step as a box
  - Arrows showing data flow
  - Color-coded by stage
- Caption: "Complete hyperscanning analysis pipeline"

**Visualization 16**:
- Example results dashboard (4-6 panels):
  - A: Both participants' signals (time series)
  - B: Power spectra comparison
  - C: Connectivity matrix (full 2n×2n)
  - D: Between-brain connectivity detail
  - E: Significant connections (circular plot)
  - F: Summary metrics bar chart
- Caption: "Hyperscanning analysis results dashboard"

---

## Section 13: What's Coming Next

**Purpose**: Preview the metrics notebooks.

**Content to cover**:
- **Now you have the foundation!**
- **Block F: Coherence-based metrics**
  - F01: Spectral Coherence (COH)
  - F02: Imaginary Coherence (ImCoh)
- **Block G: Phase-based metrics**
  - G01: Phase Locking Value (PLV)
  - G02: Phase Lag Index (PLI)
  - G03: Weighted Phase Lag Index (wPLI)
- **Block H: Amplitude-based metrics**
  - H01: Envelope Correlation (CCorr)
  - H02: Power Correlation (PowCorr)
- **Block I: Information-theoretic metrics** (future)
  - MI-based connectivity
  - TE-based connectivity
- **Each notebook will**:
  - Explain the metric intuitively
  - Implement from scratch
  - Apply to hyperscanning
  - Compare to HyPyP
  - Discuss when to use

**Visualization 17**:
- Metric roadmap:
  - Visual "journey" through upcoming notebooks
  - Grouped by type (coherence, phase, amplitude)
  - Show connections between metrics
  - Exciting preview feeling!
- Caption: "Your journey through connectivity metrics"

---

## Section 14: Hands-On Exercises

**Purpose**: Practice hyperscanning concepts.

**Exercise 1**: Data Structure
- Create simulated data for two "participants" (8 channels each)
- Use `create_hyperscanning_data_structure` to combine
- Verify channel labeling is correct
- Extract and visualize each connectivity block

**Exercise 2**: Synchronization Check
- Create two signals with slight time offset
- Use cross-correlation to detect the offset
- Apply correction
- Verify synchronization

**Exercise 3**: Pseudo-Pair Analysis
- Generate 6 "participants" in 3 real pairs
- Create all possible pseudo-pairs
- Compute simple correlation between each pair
- Compare real vs pseudo distributions
- Are real pairs significantly more synchronized?

**Exercise 4**: Connectivity Pipeline
- Use simulated hyperscanning data
- Run full pipeline:
  - Filter to alpha band
  - Compute correlation-based connectivity
  - Extract between-brain block
  - Test significance with surrogates
- Visualize results

**Exercise 5**: Metric Selection
- Given three scenarios:
  - A: "We want to know if phases align during conversation"
  - B: "We want to know who influences whom during teaching"
  - C: "We want robust synchrony measure for clinical application"
- Choose appropriate metric for each
- Justify your choice

**Exercise 6**: Interpretation Challenge
- You find high inter-brain synchrony during joint task
- But it disappears when you use pseudo-pair control
- Wait — it's LOWER than pseudo-pairs!?
- What might explain this? (Discussion exercise)

---

## Section 15: Functions for src/hyperscanning.py

**Purpose**: Create hyperscanning utilities module.

**Create new file src/hyperscanning.py with**:
- `create_hyperscanning_data_structure`
- `extract_connectivity_blocks`
- `combine_connectivity_blocks`
- `synchronize_recordings`
- `reject_epochs_both_participants`
- `check_synchronization`
- `create_pseudo_pairs`
- `compute_pseudo_pair_null`
- `test_against_pseudo_pairs`
- `hyperscanning_analysis_pipeline`

---

## Summary

**Key takeaways to list**:
- **Hyperscanning**: simultaneous recording from interacting individuals
- Studies the social brain IN social context
- Measures inter-brain synchrony (IBS): coupling between two brains
- No volume conduction between brains (advantage!) but other confounds exist
- Data structure: within-P1, within-P2, and between blocks
- **Preprocessing**: synchronization, joint artifact rejection crucial
- **Metric choice** depends on research question:
  - Phase alignment → PLV, PLI, wPLI
  - Amplitude coupling → CCorr, PowCorr
  - Linear frequency coupling → Coherence
  - Direction of influence → Transfer Entropy
- **Pseudo-pair analysis**: essential control for interaction specificity
- Synchrony correlates with rapport, performance, learning, clinical outcomes
- Interpretation requires caution: correlation ≠ causation ≠ communication
- HyPyP: our lab's tool built on MNE-Python
- You're now ready to learn the specific metrics!

---

## Discussion Questions

1. You want to study parent-child attachment through hyperscanning. What paradigm would you design? What frequencies might be most relevant? What confounds would you worry about?

2. A reviewer criticizes your hyperscanning study: "This synchrony is just because both participants see the same screen." How do you respond? What analyses would address this concern?

3. Inter-brain synchrony during therapy predicts treatment outcome. Does this mean we should try to INCREASE synchrony to improve treatment? What cautions would you raise?

4. You find theta synchrony during cooperation but alpha synchrony during competition. What might explain frequency-specific effects in different social contexts?

5. Your hyperscanning study involves video calls (remote interaction). What additional challenges does this introduce compared to face-to-face interaction?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Conceptual | Two brains interacting | subject_1, subject_2 |
| 2 | Wheel diagram | Applications | pastel palette |
| 3 | Grid | Paradigms | varied colors |
| 4 | Time series | Inter-brain synchrony example | subject_1, subject_2 |
| 5 | Diagram | Stimulus-driven confound | warning colors |
| 6 | Comparison | Single-brain vs hyperscanning | signal_1, signal_2 |
| 7 | Structure | Data organization | subject_1, subject_2 |
| 8 | Flowchart | Preprocessing pipeline | pastel blues |
| 9 | Decision tree | Metric selection | varied |
| 10 | Table | Metrics summary | metric families |
| 11 | Concept | Pseudo-pairs | subject colors |
| 12 | Distribution | Pseudo-pair null | signal_1, highlight |
| 13 | Multi-panel | Synchrony mechanisms | varied |
| 14 | Ecosystem | Tool relationships | pastel colors |
| 15 | Pipeline | Full analysis flow | stages colors |
| 16 | Dashboard | Results example | subject_1, subject_2 |
| 17 | Roadmap | Upcoming metrics | colorful journey |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Data structure functions handle different channel counts
- [ ] Synchronization handles edge cases
- [ ] Pseudo-pair generation is properly randomized
- [ ] Pipeline function modular and well-documented
- [ ] Clear distinction between P1 and P2 throughout
- [ ] Consistent naming convention (P1_, P2_ prefixes)