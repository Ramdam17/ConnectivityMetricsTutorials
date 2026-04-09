# G01: Phase Locking Value (PLV) — Copilot Directive

## Metadata

- **Notebook ID**: G01
- **Title**: Phase Locking Value (PLV)
- **Branch**: `feature/G01_phase_locking_value`
- **Duration**: 65 minutes
- **Prerequisites**: B01 (Hilbert Transform), B02 (Working with Phase), E02 (Introduction to Hyperscanning)
- **Next**: G02 (Phase Lag Index)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Define phase locking value as consistency of phase difference
2. Implement PLV computation from instantaneous phases
3. Interpret PLV values (0 = no locking, 1 = perfect locking)
4. Understand PLV's sensitivity to volume conduction
5. Compute time-resolved PLV for dynamic connectivity
6. Build PLV matrices for multi-channel analysis
7. Apply PLV to hyperscanning inter-brain synchrony

---

## Required Imports
```
numpy, matplotlib, scipy.signal
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/phase.py, src/hyperscanning.py
New: src/plv.py
```

---

## Section 1: Introduction — Are the Phases Locked?

**Purpose**: Introduce PLV conceptually.

**Content to cover**:
- **Recall phase** (B02): instantaneous position in oscillation cycle
- **Question**: Do two signals oscillate "together"?
- **Phase locking**: consistent phase relationship over time
  - Not necessarily zero phase difference!
  - Could be 90° apart, but CONSISTENTLY 90° apart
- **PLV measures**: How consistent is the phase difference?
- **Intuition**: Two dancers
  - Phase locked: always in sync (or always offset by same amount)
  - Not phase locked: sometimes together, sometimes apart
- **Why PLV for neuroscience**:
  - Neural communication via oscillations
  - Coherent oscillations enable information transfer
  - PLV captures functional connectivity

**Key message**: "PLV measures how consistently two signals maintain their phase relationship."

**Visualization 1**:
- Two scenarios side by side:
  - Left: Phase-locked signals (constant phase difference)
  - Right: Not phase-locked (variable phase difference)
- Show phase difference over time below each
- PLV value annotated
- Caption: "Phase locking: consistent vs variable phase difference"

**Length**: 4-5 paragraphs

---

## Section 2: The Mathematics of PLV

**Purpose**: Formal definition.

**Content to cover**:
- **Phase difference**: Δφ(t) = φ_x(t) - φ_y(t)
- **Phase difference as unit vector**: e^(iΔφ(t))
  - Lives on unit circle in complex plane
  - Magnitude always 1, angle = phase difference
- **PLV definition**:
  $$PLV = \left| \frac{1}{N} \sum_{t=1}^{N} e^{i(\phi_x(t) - \phi_y(t))} \right|$$
- **Interpretation**:
  - Average the unit vectors over time
  - If phase difference consistent → vectors point same direction → large resultant
  - If phase difference random → vectors point everywhere → small resultant
- **Properties**:
  - Range: 0 to 1
  - PLV = 1: perfect phase locking (constant Δφ)
  - PLV = 0: no phase locking (uniform random Δφ)
  - Independent of actual phase difference value (measures consistency)

**Visualization 2**:
- Unit circle diagrams:
  - Left: Phase-locked — vectors clustered → long resultant → high PLV
  - Right: Not locked — vectors spread → short resultant → low PLV
- Caption: "PLV as resultant vector length on unit circle"

**Visualization 3**:
- Mathematical diagram:
  - Show e^(iΔφ) as vectors
  - Sum of vectors
  - Magnitude of sum = PLV
- Caption: "PLV computation visualized"

---

## Section 3: Implementing PLV

**Purpose**: Practical computation.

**Content to cover**:
- **Pipeline**:
  1. Band-pass filter both signals (focus on frequency of interest)
  2. Extract instantaneous phase using Hilbert transform
  3. Compute phase difference Δφ(t) = φ_x(t) - φ_y(t)
  4. Convert to unit vectors: e^(iΔφ(t))
  5. Average across time
  6. Take magnitude: PLV = |average|
- **Key points**:
  - Must filter first! (PLV on broadband signal is meaningless)
  - Phase from Hilbert transform (as in B01)
  - Use np.exp(1j * phase_diff) for complex exponential

**Code to produce**:

Function 1: `compute_plv`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    float
Phase Locking Value between x and y in specified frequency band
Includes filtering and Hilbert transform
```

Function 2: `compute_plv_from_phases`
```
Parameters:
    phase_x: NDArray[np.float64]  # already extracted phases (radians)
    phase_y: NDArray[np.float64]
Returns:
    float
PLV from pre-computed phases (no filtering)
Use when phases already extracted
```

Function 3: `compute_phase_difference`
```
Parameters:
    phase_x: NDArray[np.float64]
    phase_y: NDArray[np.float64]
    wrap: bool = True  # wrap to [-π, π]
Returns:
    NDArray[np.float64]
Phase difference time series
```

**Visualization 4**:
- PLV computation pipeline:
  - Two signals → filter → Hilbert → phases → difference → unit vectors → average → magnitude
  - Visual flowchart with example data at each step
- Caption: "PLV computation pipeline"

---

## Section 4: PLV with Controlled Examples

**Purpose**: Build intuition with synthetic signals.

**Content to cover**:
- **Example 1**: Identical signals
  - PLV = 1 (perfect locking, Δφ = 0)
- **Example 2**: Constant phase offset
  - Y lags X by 90°
  - PLV still = 1 (consistent offset!)
- **Example 3**: Varying phase relationship
  - Phase offset changes slowly over time
  - PLV < 1 (inconsistent)
- **Example 4**: Random phase relationship
  - Independent signals
  - PLV ≈ 0 (random phase difference)
- **Example 5**: Noisy phase locking
  - Some locking + noise
  - PLV between 0 and 1

**Code to produce**:

Function 4: `generate_phase_locked_signals`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    phase_offset: float = 0.0  # constant offset in radians
    plv_target: float = 1.0  # desired PLV (adds noise to reduce)
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate two signals with specified phase locking
plv_target < 1 adds phase noise to reduce PLV
```

**Visualization 5**:
- Grid of examples (3×2 or 2×3):
  - Each panel: signals + phase difference + PLV value
  - Show range from PLV≈1 to PLV≈0
- Caption: "PLV for different phase relationships"

**Visualization 6**:
- PLV vs noise level:
  - X-axis: phase noise (variance)
  - Y-axis: PLV
  - Show smooth decrease from 1 to 0
- Caption: "PLV decreases with phase noise"

---

## Section 5: PLV and Volume Conduction

**Purpose**: Critical limitation of PLV.

**Content to cover**:
- **The problem**: Volume conduction creates zero-lag relationships
- **Impact on PLV**:
  - Same signal at two electrodes → Δφ ≈ 0 → PLV = 1
  - Spurious high PLV!
- **Why PLV fails**:
  - PLV measures consistency of ANY phase difference
  - Doesn't distinguish Δφ = 0 (suspicious) from Δφ = 30° (believable)
- **Solutions** (previewed):
  - **PLI** (G02): ignores zero-phase contributions
  - **wPLI** (G03): weights by imaginary component
- **For hyperscanning**:
  - Between-brain PLV is safe (no volume conduction)
  - Within-brain PLV should be interpreted cautiously

**Visualization 7**:
- Volume conduction scenario:
  - Single source, two electrodes
  - Both see same signal → Δφ ≈ 0
  - Phase difference histogram: peak at 0
  - PLV = very high
- Caption: "Volume conduction inflates PLV"

**Visualization 8**:
- Comparison:
  - True connectivity: phase difference distributed but consistent
  - Volume conduction: phase difference tightly at zero
  - Both have high PLV!
- Caption: "PLV can't distinguish true connectivity from volume conduction"

---

## Section 6: Time-Resolved PLV

**Purpose**: Track PLV dynamics over time.

**Content to cover**:
- Static PLV: single value for entire recording
- **Dynamic/time-resolved PLV**: how does locking change over time?
- **Sliding window approach**:
  - Compute PLV in short windows
  - Slide window across time
  - Get PLV time series
- **Parameters**:
  - Window size: trade-off (longer = more stable, shorter = better time resolution)
  - Overlap: typically 50-90%
- **Applications**:
  - Task-related PLV changes
  - Event-related synchronization
  - Dynamic functional connectivity
- **For hyperscanning**: Track synchrony changes during interaction

**Code to produce**:

Function 5: `compute_plv_timeseries`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    window_sec: float = 1.0
    overlap: float = 0.5
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (time_centers, plv_values)
Time-resolved PLV using sliding windows
```

Function 6: `compute_plv_epochs`
```
Parameters:
    epochs_x: NDArray[np.float64]  # (n_epochs, n_samples)
    epochs_y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    NDArray[np.float64]  # (n_epochs,)
PLV for each epoch separately
Useful for trial-by-trial analysis
```

**Visualization 9**:
- Time-resolved PLV example:
  - Top: two signals
  - Middle: phase difference over time
  - Bottom: PLV time series from sliding window
  - Show periods of high and low synchrony
- Caption: "Time-resolved PLV reveals dynamic connectivity"

---

## Section 7: PLV Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- Compute PLV for all channel pairs → PLV matrix
- **Properties**:
  - Symmetric: PLV(X,Y) = PLV(Y,X)
  - Diagonal = 1 (self-PLV)
  - Values 0 to 1
- **Efficiency considerations**:
  - n channels → n(n-1)/2 unique pairs
  - Filter all channels once, then compute pairwise PLV
- **Visualization**: same as coherence matrix (C02)

**Code to produce**:

Function 7: `compute_plv_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
PLV matrix for all channel pairs
Efficient: extracts phases once, then computes all pairs
```

Function 8: `compute_plv_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
Returns:
    dict[str, NDArray[np.float64]]
    {"delta": matrix, "theta": matrix, ...}
PLV matrices for all standard frequency bands
```

**Visualization 10**:
- PLV matrix example:
  - Simulated 16-channel data
  - Clear cluster structure if designed
  - Proper colormap (0-1 scale)
- Caption: "PLV connectivity matrix"

**Visualization 11**:
- Band comparison (2×3 grid):
  - PLV matrices for each frequency band
  - Different patterns across bands
- Caption: "Frequency-specific PLV patterns"

---

## Section 8: PLV for Hyperscanning

**Purpose**: The key application!

**Content to cover**:
- **Inter-brain PLV**:
  - Compute PLV between P1 and P2 channels
  - Measures phase synchrony between brains
- **Structure** (as in previous notebooks):
  - Within-P1 PLV matrix
  - Within-P2 PLV matrix
  - Between-brain PLV matrix
- **Interpretation**:
  - High between-brain PLV → synchronized neural oscillations
  - May reflect shared attention, communication, coordination
- **Research findings**:
  - Inter-brain PLV increases during cooperation
  - Predicts task success, rapport
  - Different frequencies for different social processes
- **PLV is the most common hyperscanning metric!**

**Code to produce**:

Function 9: `compute_plv_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,
        "within_p2": NDArray,
        "between": NDArray,
        "full": NDArray
    }
PLV matrices for hyperscanning
```

Function 10: `compute_global_plv_hyperscanning`
```
Parameters:
    plv_dict: dict[str, NDArray[np.float64]]
Returns:
    dict[str, float]
    {
        "mean_within_p1": float,
        "mean_within_p2": float,
        "mean_between": float,
        "ratio_between_within": float
    }
Summary statistics for hyperscanning PLV
```

**Visualization 12**:
- Hyperscanning PLV visualization:
  - Full 2n×2n matrix with blocks labeled
  - Subject colors distinguishing participants
- Caption: "Inter-brain phase locking in hyperscanning"

**Visualization 13**:
- Circular hyperscanning plot:
  - Two semicircles for P1 and P2
  - Lines for between-brain PLV connections
  - Line thickness/color by PLV strength
- Caption: "Inter-brain PLV network"

---

## Section 9: Statistical Significance for PLV

**Purpose**: When is PLV "real"?

**Content to cover**:
- **Chance-level PLV**: Even random signals have non-zero PLV
- **Theoretical expectation** for independent signals:
  - PLV ~ 1/√N (where N = number of samples)
  - More data → lower expected noise PLV
- **Surrogate testing**:
  - Shuffle one signal's phase (preserve spectrum)
  - Compute surrogate PLV
  - Repeat many times → null distribution
- **Rayleigh test** for uniformity:
  - Test if phase differences are uniformly distributed
  - Related to PLV significance

**Code to produce**:

Function 11: `plv_surrogate_test`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    n_surrogates: int = 500
    seed: int | None = None
Returns:
    dict[str, Any]
    {
        "plv_observed": float,
        "null_mean": float,
        "null_std": float,
        "pvalue": float,
        "threshold_95": float
    }
Surrogate testing for PLV significance
```

Function 12: `rayleigh_test`
```
Parameters:
    phase_diff: NDArray[np.float64]
Returns:
    dict[str, float]
    {"z_statistic": float, "pvalue": float}
Rayleigh test for non-uniformity of phase differences
High z / low p = significantly non-uniform = phase locked
```

**Visualization 14**:
- Significance testing:
  - Null distribution from surrogates
  - Observed PLV marked
  - Significance threshold
- Caption: "Testing PLV significance"

---

## Section 10: Comparison with Coherence

**Purpose**: When to use PLV vs coherence.

**Content to cover**:
- **Similarities**:
  - Both measure frequency-specific connectivity
  - Both sensitive to volume conduction
  - Both range 0-1 (for magnitude)
- **Differences**:
  - **Coherence**: includes amplitude information (cross-spectrum magnitude)
  - **PLV**: pure phase consistency (ignores amplitude)
  - Coherence can be affected by power fluctuations
  - PLV is power-normalized (unit vectors)
- **When they differ**:
  - High coherence, low PLV: amplitude coupling without phase locking
  - High PLV, lower coherence: phase locking with amplitude variability
- **Recommendations**:
  - PLV: when pure phase synchrony is the question
  - Coherence: when overall coupling matters
  - Often compute both!

**Code to produce**:

Function 13: `compare_plv_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, float]
    {"plv": float, "coherence": float, "difference": float}
Compare PLV and coherence for same signal pair
```

**Visualization 15**:
- PLV vs Coherence scatter:
  - Multiple signal pairs
  - X-axis: coherence, Y-axis: PLV
  - Color by type of relationship
  - Usually correlated but not identical
- Caption: "PLV vs coherence comparison"

---

## Section 11: Comparison with HyPyP

**Purpose**: Connect to production tool.

**Content to cover**:
- HyPyP implements PLV in `hypyp.analyses.connectivity`
- Based on MNE-Python's spectral connectivity
- **Comparison**:
  - Our implementation vs HyPyP
  - Should match closely
  - Note any parameter differences

**Code to produce**:

Function 14: `compare_with_hypyp_plv`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, Any]
    {
        "our_plv": NDArray,
        "hypyp_plv": NDArray,  # if available
        "max_difference": float,
        "correlation": float
    }
Compare our PLV with HyPyP implementation
```

**Visualization 16**:
- Validation plot:
  - Our PLV matrix vs HyPyP PLV matrix
  - Scatter or difference plot
  - Should be nearly identical
- Caption: "Validating PLV against HyPyP"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Basic PLV
- Generate two 10 Hz signals with known phase offset
- Compute PLV
- Verify PLV ≈ 1 (regardless of offset value)
- Add phase noise, observe PLV decrease

**Exercise 2**: Unit Circle Visualization
- Generate phase-locked signals
- Extract phases, compute differences
- Plot phase differences on unit circle
- Verify clustering for high PLV, spread for low PLV

**Exercise 3**: Volume Conduction Demonstration
- Simulate volume conduction (same signal at two "electrodes")
- Compute PLV → high!
- Note: this is SPURIOUS
- Save for comparison with PLI (G02)

**Exercise 4**: Time-Resolved PLV
- Create signals where PLV changes over time
- First half: phase locked (PLV high)
- Second half: not locked (PLV low)
- Compute sliding window PLV
- Verify you capture the transition

**Exercise 5**: PLV Matrix
- Create 8-channel simulated data
- Design cluster structure (channels 1-4 locked, 5-8 locked, between-cluster weak)
- Compute PLV matrix
- Does matrix reveal your designed structure?

**Exercise 6**: Hyperscanning PLV
- Create two-participant data (6 channels each)
- Add between-brain phase locking for specific pairs
- Compute hyperscanning PLV
- Identify the synchronized pairs
- Test against surrogates

**Exercise 7**: PLV vs Coherence
- Create scenario where PLV and coherence differ:
  - High amplitude correlation but variable phase
- Compute both metrics
- Discuss which better captures your scenario

---

## Section 13: Functions for src/plv.py

**Purpose**: Create PLV analysis module.

**Create new file src/plv.py with**:
- `compute_plv`
- `compute_plv_from_phases`
- `compute_phase_difference`
- `generate_phase_locked_signals`
- `compute_plv_timeseries`
- `compute_plv_epochs`
- `compute_plv_matrix`
- `compute_plv_matrix_bands`
- `compute_plv_hyperscanning`
- `compute_global_plv_hyperscanning`
- `plv_surrogate_test`
- `rayleigh_test`
- `compare_plv_coherence`
- `compare_with_hypyp_plv`

---

## Summary

**Key takeaways to list**:
- **PLV**: measures consistency of phase difference over time
- Formula: PLV = |mean(e^(i×Δφ))|
- Range: 0 (no locking) to 1 (perfect locking)
- **Measures phase consistency**, not actual phase value
- **Pipeline**: filter → Hilbert → phases → difference → unit vectors → average → magnitude
- **Sensitive to volume conduction** (like coherence) — use PLI/wPLI if concerned
- **Time-resolved PLV**: sliding window approach for dynamic connectivity
- **PLV matrix**: symmetric, diagonal = 1
- **For hyperscanning**: most common metric for inter-brain synchrony
- Inter-brain PLV safe from volume conduction (separate heads)
- **Statistical testing**: surrogates or Rayleigh test
- PLV ignores amplitude (unlike coherence)
- **Next**: PLI addresses volume conduction problem (G02)

---

## Discussion Questions

1. Two colleagues claim PLV = 0.6 between frontal and occipital electrodes. What questions would you ask before believing this reflects true connectivity?

2. You're studying conversation and find high inter-brain PLV at the speech rate (4-5 Hz theta). Is this "neural synchrony" or just both people responding to the same acoustic rhythm? How would you investigate?

3. PLV = 0.3. Is this "low" or "moderate"? What additional information would you need to interpret this value?

4. Your time-resolved PLV shows spikes exactly when both participants press buttons (behavioral synchrony task). Is this neural PLV or movement artifact? How would you tell?

5. A reviewer asks "Why PLV instead of coherence?" for your hyperscanning study. How do you justify your metric choice?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Side-by-side | Locked vs not locked | signal_1, signal_2 |
| 2 | Unit circles | Clustering vs spread | signal_1 |
| 3 | Mathematical | PLV as vector sum | signal_1 |
| 4 | Pipeline | Computation steps | process colors |
| 5 | Grid | Example scenarios | signal_1, signal_2 |
| 6 | Line plot | PLV vs noise | signal_1 |
| 7 | Simulation | Volume conduction | signal_1, warning |
| 8 | Comparison | True vs spurious | signal_1, signal_2 |
| 9 | 3-panel | Time-resolved PLV | signal_1, signal_2 |
| 10 | Heatmap | PLV matrix | viridis (0-1) |
| 11 | 2×3 grid | Band matrices | viridis |
| 12 | Heatmap | Hyperscanning full | subject_1, subject_2 |
| 13 | Circular | Hyperscanning network | subject_1, subject_2 |
| 14 | Distribution | Significance test | signal_1, highlight |
| 15 | Scatter | PLV vs coherence | signal_1 |
| 16 | Validation | Our vs HyPyP | signal_1, signal_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Phase extraction handles edge effects
- [ ] Filter designed appropriately for band
- [ ] Complex exponential computed correctly (np.exp(1j * ...))
- [ ] PLV clipped to [0, 1] if numerical issues
- [ ] Matrix diagonal handled (set to 1)
- [ ] Hyperscanning functions handle different channel counts
- [ ] Sliding window handles boundaries