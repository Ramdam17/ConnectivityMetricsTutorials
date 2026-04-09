# H01: Envelope Correlation (CCorr) — Copilot Directive

## Metadata

- **Notebook ID**: H01
- **Title**: Envelope Correlation (CCorr)
- **Branch**: `feature/H01_envelope_correlation`
- **Duration**: 55 minutes
- **Prerequisites**: B03 (Amplitude Envelope), E02 (Introduction to Hyperscanning)
- **Next**: H02 (Power Correlation)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Distinguish amplitude coupling from phase coupling
2. Define envelope correlation as Pearson correlation of amplitude envelopes
3. Implement envelope correlation computation
4. Interpret envelope correlation values (-1 to +1)
5. Understand orthogonalization to address volume conduction
6. Apply envelope correlation to hyperscanning analysis
7. Compare amplitude-based and phase-based connectivity

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.stats
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/phase.py, src/plv.py
New: src/amplitude.py
```

---

## Section 1: Introduction — A Different Kind of Connectivity

**Purpose**: Motivate amplitude-based connectivity.

**Content to cover**:
- **So far**: Phase-based metrics (PLV, PLI, wPLI, coherence)
  - Ask: "Are oscillations aligned in time?"
  - Focus on WHEN peaks occur
- **Now**: Amplitude-based metrics
  - Ask: "Do oscillation strengths rise and fall together?"
  - Focus on HOW STRONG oscillations are
- **Different neural mechanisms**:
  - Phase synchrony: precise timing coordination
  - Amplitude correlation: co-modulation of activity levels
- **Example intuition**:
  - Two brain regions both become "more alpha-y" during relaxation
  - Their alpha power increases together
  - But phases might not be locked!
- **Why both matter**:
  - Phase: moment-to-moment coordination
  - Amplitude: slower co-modulation of "engagement"

**Key message**: "Envelope correlation measures whether signal strengths fluctuate together, independent of phase."

**Visualization 1**:
- Two-panel comparison:
  - Left: Phase-locked signals (peaks aligned)
  - Right: Amplitude-correlated signals (envelopes rise/fall together, phases independent)
- Caption: "Phase coupling vs amplitude coupling: different phenomena"

**Length**: 4-5 paragraphs

---

## Section 2: Recall — The Amplitude Envelope

**Purpose**: Connect to B03 foundations.

**Content to cover**:
- **Recall from B03**: Amplitude envelope via Hilbert transform
- Envelope A(t) = |analytic signal| = √(x² + x̂²)
- Represents instantaneous oscillation strength
- Varies more slowly than the oscillation itself
- **For connectivity**: Compare envelopes of two signals

**Visualization 2**:
- Single signal with envelope overlaid
- Highlighting: envelope captures "how strong" at each moment
- Caption: "Amplitude envelope: instantaneous oscillation strength"

---

## Section 3: Envelope Correlation Definition

**Purpose**: Formal definition.

**Content to cover**:
- **Envelope correlation** (also called amplitude envelope correlation, AEC):
  $$CCorr = \frac{Cov(A_x, A_y)}{\sigma_{A_x} \cdot \sigma_{A_y}} = Pearson(A_x, A_y)$$
- Simply the Pearson correlation between the two amplitude envelopes
- **Properties**:
  - Range: -1 to +1 (unlike phase metrics which are 0 to 1)
  - CCorr = +1: envelopes perfectly positively correlated (rise/fall together)
  - CCorr = -1: envelopes perfectly anti-correlated (one up when other down)
  - CCorr = 0: no linear relationship between envelopes
- **Interpretation**:
  - Positive: regions "activate" together
  - Negative: one activates when other deactivates (less common)
  - Zero: independent amplitude fluctuations

**Visualization 3**:
- Three scenarios:
  - CCorr ≈ +0.8: envelopes rise and fall together
  - CCorr ≈ 0: envelopes independent
  - CCorr ≈ -0.6: envelopes anti-correlated
- Caption: "Envelope correlation captures amplitude co-modulation"

---

## Section 4: Implementing Envelope Correlation

**Purpose**: Practical computation.

**Content to cover**:
- **Pipeline**:
  1. Band-pass filter both signals (frequency of interest)
  2. Extract amplitude envelopes (Hilbert transform)
  3. (Optional) Low-pass filter envelopes to focus on slow fluctuations
  4. Compute Pearson correlation between envelopes
- **Key considerations**:
  - Must filter first! (Envelope of broadband signal not meaningful)
  - Optional envelope smoothing (removes fast fluctuations)
  - Use scipy.stats.pearsonr for correlation

**Code to produce**:

Function 1: `compute_envelope_correlation`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
    envelope_lowpass: float | None = None  # optional smoothing cutoff
Returns:
    float
Envelope correlation between x and y in specified band
```

Function 2: `compute_envelope_correlation_from_envelopes`
```
Parameters:
    env_x: NDArray[np.float64]  # pre-computed envelope
    env_y: NDArray[np.float64]
Returns:
    float
Correlation from pre-computed envelopes
```

Function 3: `smooth_envelope`
```
Parameters:
    envelope: NDArray[np.float64]
    fs: float
    cutoff: float = 1.0  # Hz
Returns:
    NDArray[np.float64]
Low-pass filter envelope for slow fluctuations only
```

**Visualization 4**:
- Computation pipeline:
  - Two signals → filter → Hilbert → envelopes → (smooth) → correlation
  - Visual flow with example data
- Caption: "Envelope correlation computation pipeline"

---

## Section 5: Envelope Correlation with Controlled Examples

**Purpose**: Build intuition with synthetic signals.

**Content to cover**:
- **Example 1**: Same amplitude modulation
  - Two signals share same envelope (different phases)
  - CCorr = +1
- **Example 2**: Independent modulation
  - Each signal has own random envelope
  - CCorr ≈ 0
- **Example 3**: Anti-correlated modulation
  - When one envelope up, other down
  - CCorr < 0
- **Example 4**: Partial correlation
  - Shared component + independent noise
  - 0 < CCorr < 1

**Code to produce**:

Function 4: `generate_amplitude_correlated_signals`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    correlation_target: float  # desired envelope correlation
    modulation_freq: float = 0.5  # how fast envelopes change
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate two signals with specified envelope correlation
```

**Visualization 5**:
- Grid of examples (2×2):
  - Each panel: two signals + envelopes + CCorr value
  - Shows range from +1 to -1
- Caption: "Envelope correlation examples"

**Visualization 6**:
- CCorr vs actual shared modulation:
  - X-axis: proportion of shared modulation
  - Y-axis: measured CCorr
  - Should be linear relationship
- Caption: "Envelope correlation tracks shared modulation"

---

## Section 6: The Volume Conduction Problem (Again!)

**Purpose**: Critical issue for amplitude correlation.

**Content to cover**:
- **The problem**: Volume conduction affects amplitude too!
  - Same source at two electrodes → same envelope
  - CCorr artificially inflated
- **Even worse than for phase**:
  - PLI/wPLI can ignore zero-lag phase
  - For amplitude: zero-lag IS the signal (no lag to exploit)
- **Solution**: Orthogonalization (Hipp et al., 2012)

**Visualization 7**:
- Volume conduction scenario:
  - Single source, two electrodes
  - Nearly identical envelopes
  - CCorr ≈ 1 (spurious!)
- Caption: "Volume conduction inflates envelope correlation"

---

## Section 7: Orthogonalized Envelope Correlation

**Purpose**: The solution to volume conduction.

**Content to cover**:
- **Key insight**: Volume conduction creates instantaneous (zero-lag) correlation
- **Orthogonalization**: Remove the zero-lag component!
- **Method** (Hipp et al., 2012):
  1. Compute analytic signals z_x(t) and z_y(t)
  2. Orthogonalize: remove component of z_y that's in phase with z_x
     $$z_{y⊥}(t) = Im(z_y(t) \cdot e^{-i\phi_x(t)})$$
  3. Compute envelopes from orthogonalized signals
  4. Correlate the orthogonalized envelopes
- **Effect**: Removes zero-lag (volume conduction) contribution
- **Properties**:
  - Still ranges -1 to +1
  - Symmetric version: average both directions (y⊥x and x⊥y)
  - More conservative than standard CCorr

**Code to produce**:

Function 5: `orthogonalize_signals`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (y_orthogonalized_to_x, x_orthogonalized_to_y)
Orthogonalize signals to remove zero-lag component
```

Function 6: `compute_orthogonalized_envelope_correlation`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    symmetric: bool = True  # average both directions
Returns:
    float
Envelope correlation with orthogonalization (AEC-c)
```

**Visualization 8**:
- Orthogonalization effect:
  - Original envelopes (highly correlated from VC)
  - Orthogonalized envelopes (correlation reduced)
  - Show the "cleaned" result
- Caption: "Orthogonalization removes zero-lag contribution"

**Visualization 9**:
- Comparison bar chart:
  - Volume conduction: CCorr high, CCorr-orth low
  - True connection: CCorr high, CCorr-orth still high
- Caption: "Orthogonalization distinguishes true from spurious"

---

## Section 8: Time-Resolved Envelope Correlation

**Purpose**: Dynamic amplitude connectivity.

**Content to cover**:
- Static CCorr: single value for entire recording
- **Dynamic CCorr**: How does amplitude coupling change over time?
- **Sliding window approach**:
  - Extract envelopes for full signal
  - Compute correlation in short windows
  - Get CCorr time series
- **Parameters**: window size, overlap
- **Applications**: Task-related amplitude coupling changes

**Code to produce**:

Function 7: `compute_envelope_correlation_timeseries`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    window_sec: float = 2.0
    overlap: float = 0.5
    orthogonalize: bool = True
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (time_centers, ccorr_values)
Time-resolved envelope correlation
```

**Visualization 10**:
- Time-resolved CCorr:
  - Top: two signals
  - Middle: envelopes
  - Bottom: sliding window CCorr
- Caption: "Dynamic envelope correlation"

---

## Section 9: Envelope Correlation Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- Compute CCorr for all channel pairs
- **Properties**:
  - Symmetric: CCorr(X,Y) = CCorr(Y,X)
  - Diagonal = 1 (self-correlation)
  - Range: -1 to +1 (can be negative!)
- **Visualization**: Diverging colormap (blue-white-red) centered at 0
- **Orthogonalized matrix**: More conservative, recommended for EEG

**Code to produce**:

Function 8: `compute_envelope_correlation_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    orthogonalize: bool = True
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
Envelope correlation matrix
```

Function 9: `compute_envelope_correlation_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
    orthogonalize: bool = True
Returns:
    dict[str, NDArray[np.float64]]
Envelope correlation matrices for all bands
```

**Visualization 11**:
- Two matrices side by side:
  - Standard CCorr matrix
  - Orthogonalized CCorr matrix
  - Use diverging colormap
- Caption: "Envelope correlation matrix: standard vs orthogonalized"

**Visualization 12**:
- Band comparison (2×3 grid):
  - Orthogonalized CCorr for each frequency band
- Caption: "Band-specific amplitude coupling"

---

## Section 10: Envelope Correlation for Hyperscanning

**Purpose**: The key application!

**Content to cover**:
- **Inter-brain envelope correlation**:
  - Do participants' brain activity levels co-fluctuate?
  - Different question from phase synchrony!
- **No volume conduction between brains**:
  - Orthogonalization less critical for between-brain
  - But still useful for within-brain blocks
- **Research findings**:
  - Inter-brain amplitude correlation in emotional sharing
  - Co-activation patterns during joint attention
  - May reflect shared arousal/engagement

**Code to produce**:

Function 10: `compute_envelope_correlation_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    orthogonalize_within: bool = True  # for within-brain
    orthogonalize_between: bool = False  # not needed between
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,
        "within_p2": NDArray,
        "between": NDArray,
        "full": NDArray
    }
Envelope correlation matrices for hyperscanning
```

Function 11: `compute_global_envelope_correlation_hyperscanning`
```
Parameters:
    ccorr_dict: dict[str, NDArray[np.float64]]
Returns:
    dict[str, float]
    {
        "mean_within_p1": float,
        "mean_within_p2": float,
        "mean_between": float
    }
Summary statistics
```

**Visualization 13**:
- Hyperscanning envelope correlation:
  - Full matrix with blocks labeled
  - Diverging colormap
- Caption: "Inter-brain amplitude coupling in hyperscanning"

---

## Section 11: Envelope Correlation vs Phase Metrics

**Purpose**: When to use which.

**Content to cover**:
- **Different phenomena**:
  - Phase metrics: oscillation timing alignment
  - Amplitude metrics: oscillation strength co-modulation
- **Different timescales**:
  - Phase: fast (within-cycle)
  - Amplitude: slow (envelope fluctuations over seconds)
- **When they agree**: Strong coupling on both dimensions
- **When they diverge**:
  - High phase, low amplitude: timing aligned but strengths independent
  - High amplitude, low phase: strengths coupled but timing independent
- **Recommendation**: Compute both, report both, interpret differently

**Code to produce**:

Function 12: `compare_phase_amplitude_connectivity`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, float]
    {
        "plv": float,
        "wpli": float,
        "ccorr": float,
        "ccorr_orth": float
    }
Compare phase and amplitude connectivity
```

**Visualization 14**:
- Scatter plot:
  - X-axis: wPLI (phase)
  - Y-axis: CCorr-orth (amplitude)
  - Multiple channel pairs
  - Note: not necessarily correlated!
- Caption: "Phase vs amplitude connectivity: different dimensions"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Basic Envelope Correlation
- Generate two signals with shared amplitude modulation
- Compute CCorr
- Verify high positive correlation

**Exercise 2**: Phase Independence
- Generate signals with correlated envelopes but random phases
- Compute CCorr → high
- Compute wPLI → low
- Demonstrates amplitude coupling ≠ phase coupling

**Exercise 3**: Volume Conduction Demonstration
- Simulate volume conduction (same signal at two electrodes)
- Compute CCorr → very high (spurious)
- Compute orthogonalized CCorr → much lower
- Key demonstration of orthogonalization value

**Exercise 4**: Time-Resolved Analysis
- Create signals where envelope correlation changes over time
- Compute sliding window CCorr
- Track the dynamics

**Exercise 5**: CCorr Matrix
- Create 8-channel data with amplitude coupling structure
- Compute standard and orthogonalized matrices
- Compare — which pairs show biggest difference?

**Exercise 6**: Phase vs Amplitude
- Create signals with:
  - Pair A: high phase coupling, low amplitude coupling
  - Pair B: low phase coupling, high amplitude coupling
- Compute wPLI and CCorr for each
- Verify the dissociation

**Exercise 7**: Hyperscanning Application
- Create two-participant data with between-brain amplitude coupling
- Compute hyperscanning CCorr
- Compare with wPLI hyperscanning results

---

## Section 13: Functions for src/amplitude.py

**Purpose**: Create amplitude connectivity module.

**Create new file src/amplitude.py with**:
- `compute_envelope_correlation`
- `compute_envelope_correlation_from_envelopes`
- `smooth_envelope`
- `generate_amplitude_correlated_signals`
- `orthogonalize_signals`
- `compute_orthogonalized_envelope_correlation`
- `compute_envelope_correlation_timeseries`
- `compute_envelope_correlation_matrix`
- `compute_envelope_correlation_matrix_bands`
- `compute_envelope_correlation_hyperscanning`
- `compute_global_envelope_correlation_hyperscanning`
- `compare_phase_amplitude_connectivity`

---

## Summary

**Key takeaways to list**:
- **Envelope correlation** measures amplitude co-modulation
- Different from phase coupling! (strength vs timing)
- Formula: Pearson correlation of amplitude envelopes
- Range: -1 to +1 (can be negative)
- **Pipeline**: filter → Hilbert → envelopes → correlate
- **Volume conduction problem**: inflates CCorr (same envelope)
- **Solution**: Orthogonalization removes zero-lag component
- Orthogonalized CCorr (AEC-c) is more conservative
- **For hyperscanning**: captures shared activation/engagement
- Between-brain: orthogonalization less critical
- Within-brain: orthogonalization recommended
- **Report both** phase and amplitude metrics — different information!
- **Next**: Power correlation (H02) — another amplitude approach

---

## Discussion Questions

1. A colleague says "we found high envelope correlation between frontal and parietal regions." What follow-up questions would you ask about their analysis?

2. You find high wPLI but low envelope correlation between two regions. What might this mean neurophysiologically? What about the opposite pattern?

3. Orthogonalization removes zero-lag components. But couldn't there be true instantaneous amplitude coupling? Is orthogonalization always appropriate?

4. Envelope correlation operates on slow fluctuations (envelope changes). What timescale of neural processing might this capture compared to phase synchrony?

5. For hyperscanning, you find strong between-brain envelope correlation in alpha band during a conversation task. Is this "neural coupling" or might there be alternative explanations (e.g., shared arousal from the task)?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 2-panel | Phase vs amplitude coupling | signal_1, signal_2 |
| 2 | Single signal | Signal with envelope | signal_1 |
| 3 | 3 scenarios | CCorr range examples | signal_1, signal_2 |
| 4 | Pipeline | Computation flow | process colors |
| 5 | 2×2 grid | Example scenarios | signal_1, signal_2 |
| 6 | Line plot | CCorr vs shared modulation | signal_1 |
| 7 | Simulation | Volume conduction | signal_1, warning |
| 8 | Before/after | Orthogonalization effect | signal_1, signal_2 |
| 9 | Bar chart | Standard vs orth comparison | signal_1, signal_2 |
| 10 | 3-panel | Time-resolved CCorr | signal_1, signal_2 |
| 11 | 2 matrices | Standard vs orth | diverging (RdBu) |
| 12 | 2×3 grid | Band matrices | diverging |
| 13 | Heatmap | Hyperscanning full | diverging |
| 14 | Scatter | Phase vs amplitude | signal_1 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Envelope extraction handles edge effects
- [ ] Orthogonalization computed correctly (imaginary part formula)
- [ ] Symmetric orthogonalization averages both directions
- [ ] Diverging colormap used for CCorr visualization
- [ ] Matrix diagonal set to 1 (self-correlation)
- [ ] Sliding window handles boundaries
- [ ] Pearson correlation handles edge cases (constant signal)