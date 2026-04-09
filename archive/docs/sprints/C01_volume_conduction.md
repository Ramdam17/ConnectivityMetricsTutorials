# C01: The Volume Conduction Problem — Copilot Directive

## Metadata

- **Notebook ID**: C01
- **Title**: The Volume Conduction Problem
- **Branch**: `feature/C01_volume_conduction`
- **Duration**: 60 minutes
- **Prerequisites**: B01 (Hilbert Transform), B02 (Working with Phase)
- **Next**: C02 (Connectivity Matrices)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand what volume conduction is physically
2. Explain how volume conduction creates spurious connectivity
3. Recognize zero-lag correlation as a signature of volume conduction
4. Understand why raw EEG connectivity is problematic
5. Describe solutions: imaginary coherence, PLI, source localization
6. Simulate volume conduction effects
7. Appreciate why this matters critically for hyperscanning

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.stats, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/filtering.py, src/hilbert.py, src/phase.py
```

---

## Section 1: Introduction — The Elephant in the Room

**Purpose**: Frame volume conduction as THE central problem in EEG connectivity.

**Content to cover**:
- EEG measures electrical activity at the scalp
- The skull and tissues between brain and electrodes conduct electricity
- This creates a fundamental problem for connectivity analysis
- **The issue**: a single brain source appears at MULTIPLE electrodes
- If two electrodes see the same source, they will appear "connected"
- But this isn't real connectivity — it's an artifact!
- This problem is even MORE critical in hyperscanning
- Before learning ANY connectivity metric, we must understand this

**Key message**: "If you don't understand volume conduction, your connectivity results may be meaningless."

**Length**: 4-5 paragraphs, set serious tone

---

## Section 2: What is Volume Conduction?

**Purpose**: Explain the physics simply.

**Content to cover**:
- Brain activity = electrical currents from neurons
- These currents create electrical fields
- Fields spread through brain tissue, CSF, skull, scalp
- All these tissues are conductive (like a resistor network)
- Result: activity from ONE brain source reaches MANY electrodes
- The signal at each electrode is a MIXTURE of many sources
- Analogy: listening to an orchestra from outside the concert hall
  - You hear all instruments mixed together
  - Hard to isolate individual players
- This is called "field spread" or "volume conduction"

**Visualization 1**:
- Schematic diagram (can be simple/conceptual)
- Show: brain source → field spreading → multiple electrodes
- Concentric circles emanating from a point source
- Multiple electrodes at different distances receiving signal
- Caption: "A single source contributes to many electrodes"

---

## Section 3: The Problem for Connectivity

**Purpose**: Show how volume conduction creates false connectivity.

**Content to cover**:
- Scenario: ONE oscillating source in the brain
- Two nearby electrodes both pick up this source
- Signal at electrode A ≈ signal at electrode B (scaled differently)
- If we compute correlation: very high! (same signal)
- If we compute phase synchronization: perfect! (same signal)
- **But there's no actual CONNECTION between brain regions**
- It's one source appearing twice, not two sources communicating
- This is called **spurious connectivity** or **artificial coupling**

**Visualization 2**:
- Simulation of volume conduction scenario
- One source signal (10 Hz oscillation)
- Two "electrodes" that receive weighted versions:
  - Electrode A: 0.8 × source + noise
  - Electrode B: 0.5 × source + noise
- Three subplots:
  - Source signal
  - Electrode A signal
  - Electrode B signal
- Visually obvious they look similar (same phase!)

**Visualization 3**:
- Compute correlation and PLV between electrodes A and B
- Show very high values (correlation > 0.8, PLV > 0.9)
- Bar chart or annotated values
- Big red warning: "THIS IS NOT REAL CONNECTIVITY!"

---

## Section 4: The Zero-Lag Signature

**Purpose**: Introduce key diagnostic for volume conduction.

**Content to cover**:
- Volume conduction is (nearly) instantaneous
- Electrical fields propagate at ~speed of light
- For EEG purposes: effectively instantaneous (no measurable delay)
- **Key insight**: volume conduction creates ZERO-LAG correlation
- The signals are correlated with NO time delay
- Equivalently: phase difference ≈ 0 (or π for inverted polarity)
- Real neural communication has delays:
  - Axonal conduction: ~1-10 m/s
  - Synaptic delays: ~0.5-2 ms
  - Results in non-zero phase lag
- **Diagnostic**: if phase difference is always 0 or π → suspicious!

**Code to produce**:

Function 1: `simulate_volume_conduction`
```
Parameters:
    source_signal: NDArray[np.float64]
    weights: list[float]  # mixing weights for each electrode
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    NDArray[np.float64]
    Shape: (n_electrodes, n_samples)
Simulate volume conduction: each electrode gets weighted source + noise
```

Function 2: `compute_cross_correlation`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    max_lag_samples: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (lags, correlation_values)
Cross-correlation function to identify time lag of maximum correlation
```

**Visualization 4**:
- Cross-correlation function between the two electrodes
- X-axis: lag (in ms)
- Y-axis: correlation
- Peak should be at LAG = 0 (zero-lag)
- Annotate the peak location
- Caption: "Peak at zero lag is signature of volume conduction"

**Visualization 5**:
- Phase difference distribution (from B02)
- Histogram of instantaneous phase differences
- Should be concentrated at 0 (or π)
- Very narrow distribution
- Caption: "Phase difference locked near zero → volume conduction"

---

## Section 5: Why This Matters for Hyperscanning

**Purpose**: Connect to the workshop's main topic.

**Content to cover**:
- In single-brain EEG: volume conduction between nearby electrodes
- In hyperscanning: we compare DIFFERENT BRAINS
- Good news: no volume conduction BETWEEN participants!
- Electrical fields don't jump from one head to another
- Bad news: volume conduction still affects WITHIN-participant analysis
- Also bad: if we're sloppy with electrode positions across participants...
- ...we might compare different brain regions unknowingly
- **Key advantage of hyperscanning**: inter-brain connectivity is "cleaner"
- But intra-brain connectivity still has the problem

**Visualization 6**:
- Schematic of two-brain hyperscanning setup
- Show: within-brain volume conduction (problematic)
- Show: between-brain (no volume conduction possible)
- Use subject_1 and subject_2 colors
- Clear visual distinction

---

## Section 6: Demonstrating the Problem — Simulation

**Purpose**: Quantitative demonstration with realistic parameters.

**Content to cover**:
- Create a more realistic simulation
- Multiple sources, multiple electrodes
- Each electrode receives mixture of sources (mixing matrix)
- Compute connectivity between all electrode pairs
- Show that nearby electrodes have spuriously high connectivity
- Electrodes seeing same sources → false positive connections

**Code to produce**:

Function 3: `create_mixing_matrix`
```
Parameters:
    n_sources: int
    n_electrodes: int
    spread: float = 0.5  # controls how much sources spread
    seed: int | None = None
Returns:
    NDArray[np.float64]
    Shape: (n_electrodes, n_sources)
Create a mixing matrix simulating field spread
Nearby electrodes share more sources
```

Function 4: `apply_mixing`
```
Parameters:
    sources: NDArray[np.float64]  # (n_sources, n_samples)
    mixing_matrix: NDArray[np.float64]  # (n_electrodes, n_sources)
    noise_level: float = 0.1
Returns:
    NDArray[np.float64]  # (n_electrodes, n_samples)
Apply mixing matrix to simulate scalp EEG from sources
```

**Visualization 7**:
- Simulation with 3 sources, 6 electrodes
- Show:
  - Mixing matrix as heatmap (which electrodes see which sources)
  - Source signals (3 independent oscillations)
  - Resulting electrode signals
- Multi-panel figure

**Visualization 8**:
- Connectivity matrix (e.g., simple correlation) between all electrode pairs
- Heatmap format
- Show high values between electrodes sharing sources
- Annotate: "High connectivity = shared sources, not communication"

---

## Section 7: Solutions Overview

**Purpose**: Preview methods that address volume conduction.

**Content to cover**:
- Several approaches have been developed:

1. **Imaginary part of coherence (ImCoh)**
   - Volume conduction contributes to REAL part only
   - Imaginary part reflects lagged connectivity
   - Discard real part → remove volume conduction
   - Covered in F02

2. **Phase Lag Index (PLI)**
   - Ignore phase differences of exactly 0 or π
   - Only count consistent non-zero phase lags
   - Covered in G02

3. **Weighted PLI (wPLI)**
   - Weight by magnitude of imaginary component
   - More robust version of PLI
   - Covered in G03

4. **Source localization**
   - Estimate activity at brain source level
   - Analyze connectivity between sources, not electrodes
   - Reduces (doesn't eliminate) mixing
   - Beyond this workshop's scope but mentioned

5. **Laplacian reference (Surface Laplacian)**
   - Spatial filtering that sharpens topography
   - Reduces field spread
   - Simple preprocessing step

**Visualization 9**:
- Flowchart or diagram showing different solutions
- Problem (volume conduction) → multiple solution paths
- Color-code by approach type
- Reference future notebooks

---

## Section 8: Imaginary Coherence — Intuition

**Purpose**: Deep dive on the most elegant solution.

**Content to cover**:
- Coherence is complex-valued (has real and imaginary parts)
- Real part: captures zero-lag (in-phase) connectivity
- Imaginary part: captures lagged connectivity
- Volume conduction → zero phase lag → purely REAL
- True connectivity → non-zero phase lag → has IMAGINARY component
- **Solution**: use only the imaginary part!
- Trade-off: we lose sensitivity to true zero-lag connectivity
- But: true zero-lag between separate brain regions is rare anyway

**Visualization 10**:
- Complex plane diagram
- Show coherence as a vector
- Volume conduction: vector along real axis
- True connectivity: vector with imaginary component
- Illustrate that |imaginary part| captures true connectivity

**Visualization 11**:
- Simulation comparison:
  - Case 1: pure volume conduction
  - Case 2: true connectivity with phase lag
- For each, show:
  - Coherence (magnitude)
  - Imaginary coherence
- Volume conduction: high coherence, near-zero imag coherence
- True connectivity: both are substantial

---

## Section 9: Phase Lag Index — Intuition

**Purpose**: Explain PLI approach conceptually.

**Content to cover**:
- Different approach than imaginary coherence
- Focus on the SIGN of phase difference
- Phase difference < 0: signal 1 leads
- Phase difference > 0: signal 2 leads
- Volume conduction: phase difference ≈ 0 (or π)
  - Sign flips randomly due to noise
  - No consistent lead/lag
- True connectivity: consistent non-zero phase lag
  - One signal consistently leads
  - Sign is consistent over time
- PLI = |mean(sign(Δφ))|
- Range: 0 (no consistent lag) to 1 (perfectly consistent lag)

**Code to produce**:

Function 5: `compute_plv_simple`
```
Parameters:
    phase1: NDArray[np.float64]
    phase2: NDArray[np.float64]
Returns:
    float
Simple PLV computation (preview, will be in G01)
```

Function 6: `compute_pli_simple`
```
Parameters:
    phase1: NDArray[np.float64]
    phase2: NDArray[np.float64]
Returns:
    float
Simple PLI computation (preview, will be in G02)
PLI = |mean(sign(phase_diff))|
Exclude phase differences very close to 0 or π
```

**Visualization 12**:
- Phase difference time series for two scenarios:
  - Volume conduction: phase diff ≈ 0, sign flips randomly
  - True connectivity: phase diff ≈ +0.3 rad, sign consistently positive
- Two rows, each showing:
  - Phase difference over time
  - Sign of phase difference over time
  - Running PLI value
- Clear contrast between scenarios

---

## Section 10: Practical Demonstration — PLV vs PLI

**Purpose**: Show PLI robustness to volume conduction.

**Content to cover**:
- Compare PLV and PLI on simulated data
- Scenario 1: Volume conduction (same source, scaled)
- Scenario 2: True connectivity (independent sources with consistent lag)
- PLV fails to distinguish (high for both)
- PLI correctly identifies true connectivity

**Code to produce**:

Function 7: `simulate_true_connectivity`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    phase_lag: float  # radians
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (signal1, signal2)
Two signals with specified phase lag (true connectivity)
```

**Visualization 13**:
- 2×2 comparison grid:
  - Row 1: Volume conduction scenario
  - Row 2: True connectivity scenario
  - Column 1: PLV value (bar)
  - Column 2: PLI value (bar)
- Show: PLV high for both, PLI only high for true connectivity
- Key figure demonstrating PLI advantage

---

## Section 11: The Laplacian Reference

**Purpose**: Introduce surface Laplacian as preprocessing solution.

**Content to cover**:
- Spatial filter applied to electrode data
- Estimates second spatial derivative of potential
- Effect: sharpens the topography, reduces spread
- Each electrode becomes more "local"
- Simple to apply: weighted combination of neighbors
- Doesn't eliminate volume conduction but reduces it
- Common preprocessing before connectivity analysis
- Trade-off: amplifies high-frequency spatial noise

**Note**: Full implementation requires electrode positions; show concept only

**Visualization 14**:
- Conceptual illustration
- Topographic map before Laplacian (broad, smooth)
- Topographic map after Laplacian (sharp, localized)
- Can be schematic if electrode positions unavailable

---

## Section 12: Summary of Approaches

**Purpose**: Consolidate the different solutions.

**Table**:
| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Imaginary Coherence** | Use only imaginary part | Elegant, mathematically clean | Loses true zero-lag connectivity |
| **PLI** | Ignore zero/π phase diffs | Robust to volume conduction | Less sensitive than PLV |
| **wPLI** | Weight by |imag| | More robust than PLI | Slightly more complex |
| **Surface Laplacian** | Spatial filtering | Simple preprocessing | Doesn't eliminate problem |
| **Source localization** | Estimate brain sources | Most principled | Complex, requires head model |

**Recommendation for hyperscanning**:
- Inter-brain connectivity: PLV is fine (no volume conduction between heads)
- Intra-brain connectivity: use PLI/wPLI or imaginary coherence
- Always consider your scientific question

---

## Section 13: Common Pitfalls

**Purpose**: Warn about mistakes practitioners make.

**Content to cover**:

**Pitfall 1**: Ignoring volume conduction entirely
- "We found high coherence between frontal and parietal regions!"
- Could be one source seen by both areas

**Pitfall 2**: Assuming PLI/wPLI solves everything
- These metrics are less sensitive
- May miss real connectivity that happens to have near-zero lag

**Pitfall 3**: Over-interpreting zero-lag connectivity
- If you use PLV and find high values at neighboring electrodes...
- ...it's almost certainly volume conduction

**Pitfall 4**: Forgetting about polarity inversions
- Phase difference of π also indicates volume conduction
- (source between two electrodes → opposite polarity)

**Pitfall 5**: Not validating with simulation
- Always test your analysis pipeline on simulated data
- Know what your metrics do with known ground truth

---

## Section 14: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Volume Conduction Simulation
- Create a single 10 Hz source
- Simulate 4 electrodes with different mixing weights
- Compute PLV between all pairs
- Verify: electrodes with similar weights have higher PLV
- This is spurious!

**Exercise 2**: Zero-Lag Detection
- Generate two scenarios: volume conduction vs true connectivity
- Compute cross-correlation for both
- Identify which has peak at zero lag
- Use this as a diagnostic

**Exercise 3**: PLV vs PLI Comparison
- Volume conduction simulation (same source)
- True connectivity simulation (lagged sources)
- Compute both PLV and PLI
- Create comparison figure
- Interpret the results

**Exercise 4**: Varying Phase Lag
- Generate signals with phase lags: 0, π/8, π/4, π/2
- Compute PLV and PLI for each
- Plot both metrics vs phase lag
- At what lag does PLI become substantial?

**Exercise 5**: Mixing Matrix Effects
- Create different mixing matrices (more/less spread)
- Same sources, different mixing
- Analyze how connectivity patterns change
- Understand the impact of field spread

---

## Section 15: Functions for src/connectivity_utils.py

**Purpose**: Identify functions for new module.

**New file**: src/connectivity_utils.py

Functions to extract:
- `simulate_volume_conduction`
- `compute_cross_correlation`
- `create_mixing_matrix`
- `apply_mixing`
- `simulate_true_connectivity`
- `compute_plv_simple` (preview, will move to src/metrics/)
- `compute_pli_simple` (preview, will move to src/metrics/)

---

## Summary

**Key takeaways to list**:
- Volume conduction: single source appears at multiple electrodes
- Creates spurious connectivity that is NOT real brain communication
- Signature: zero-lag correlation, phase difference ≈ 0 or π
- Major solutions: imaginary coherence, PLI, wPLI, source localization
- For hyperscanning: inter-brain connectivity is cleaner (no shared volume)
- Intra-brain still affected — use robust metrics
- Always simulate and validate before trusting results
- This is THE most important concept for EEG connectivity analysis

---

## Discussion Questions

1. You analyze EEG from two participants during conversation and find high PLV at electrode pairs Fz-Fz (one Fz per participant). Should you be concerned about volume conduction? Why or why not?

2. A paper reports "significant coherence between left motor and right motor cortex during bimanual coordination." What questions would you ask about their analysis to evaluate whether this reflects true connectivity?

3. Your PLI analysis shows very low inter-brain connectivity, but your PLV analysis shows high values. What might explain this discrepancy? Which result should you trust more?

4. In hyperscanning, why is it still important to consider volume conduction even though the two participants have separate heads?

5. A colleague suggests: "Let's just use very distant electrode pairs to avoid volume conduction." What are the pros and cons of this approach?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Schematic | Field spread from source to electrodes | conceptual |
| 2 | 3 subplots | Source and two electrode signals | signal_1, signal_2, signal_3 |
| 3 | Bar chart | Spurious PLV/correlation values | warning red |
| 4 | Single plot | Cross-correlation with zero-lag peak | signal_1 |
| 5 | Histogram | Phase difference distribution | signal_2 |
| 6 | Schematic | Two-brain hyperscanning setup | subject_1, subject_2 |
| 7 | Multi-panel | Mixing matrix + sources + electrodes | signal colors |
| 8 | Heatmap | Spurious connectivity matrix | sequential cmap |
| 9 | Flowchart | Solutions overview | various |
| 10 | Complex plane | Coherence vector illustration | signal_1, signal_2 |
| 11 | 2×2 comparison | Coherence vs ImCoh for two scenarios | signal colors |
| 12 | 2 rows | Phase diff and PLI for two scenarios | signal_1, signal_2 |
| 13 | 2×2 grid | PLV vs PLI comparison | primary, secondary |
| 14 | 2 topomaps | Before/after Laplacian (schematic) | sequential cmap |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Simulations use reproducible random seeds
- [ ] Clear distinction between preview functions and final implementations
- [ ] Warning messages included where appropriate
- [ ] Cross-references to future notebooks (F02, G02, G03)