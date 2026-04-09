# F02: Imaginary Coherence — Copilot Directive

## Metadata

- **Notebook ID**: F02
- **Title**: Imaginary Coherence (ImCoh)
- **Branch**: `feature/F02_imaginary_coherence`
- **Duration**: 55 minutes
- **Prerequisites**: F01 (Spectral Coherence), C01 (Volume Conduction Problem)
- **Next**: G01 (Phase Locking Value)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain why standard coherence fails with volume conduction
2. Understand imaginary coherence as a robust alternative
3. Implement imaginary coherence computation
4. Interpret imaginary coherence values and their sign
5. Recognize the trade-off (sensitivity vs specificity)
6. Apply imaginary coherence to hyperscanning analysis
7. Choose appropriately between coherence and imaginary coherence

---

## Required Imports
```
numpy, matplotlib, scipy.signal
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/coherence.py, src/hyperscanning.py
```

---

## Section 1: The Problem with Standard Coherence

**Purpose**: Motivate imaginary coherence.

**Content to cover**:
- **Recall from F01**: Coherence measures frequency-specific correlation
- **Recall from C01**: Volume conduction spreads signals across electrodes
- **The problem**:
  - Single source → appears at multiple electrodes with ZERO phase lag
  - Zero-lag signal → high coherence (signals are identical!)
  - Standard coherence can't tell: real connection vs volume conduction
- **Why magnitude is the problem**:
  - Coherence uses |S_xy|² — only magnitude of cross-spectrum
  - Zero-lag connections have purely REAL cross-spectrum
  - True lagged connections have IMAGINARY component
- **The solution**: Use only the imaginary part!

**Key message**: "Imaginary coherence ignores zero-lag contributions, eliminating volume conduction artifacts."

**Visualization 1**:
- Complex plane diagram:
  - Cross-spectrum as vector
  - Zero-lag: purely real (on x-axis)
  - Lagged connection: has imaginary component (off x-axis)
  - Standard coherence: uses whole vector length
  - Imaginary coherence: uses only y-component
- Caption: "Zero-lag signals contribute only to real part"

**Length**: 4-5 paragraphs

---

## Section 2: Cross-Spectrum in the Complex Plane

**Purpose**: Build intuition for imaginary part.

**Content to cover**:
- **Cross-spectral density** S_xy(f) is COMPLEX:
  - S_xy = |S_xy| × e^(iφ)
  - |S_xy|: magnitude (coupling strength)
  - φ: phase difference between X and Y
- **Real and imaginary parts**:
  - Real(S_xy) = |S_xy| × cos(φ)
  - Imag(S_xy) = |S_xy| × sin(φ)
- **Phase relationships**:
  - φ = 0 (zero lag): purely REAL (Imag = 0)
  - φ = 90°: purely IMAGINARY (Real = 0)
  - φ = 180°: purely REAL, negative
  - Other phases: both Real and Imag non-zero
- **Key insight**:
  - Volume conduction → φ ≈ 0 → contributes to REAL only
  - True connection → φ ≠ 0 → contributes to IMAGINARY

**Visualization 2**:
- Complex plane with multiple vectors:
  - Vector at 0° (purely real) — labeled "Volume conduction"
  - Vector at 45° (real + imaginary) — labeled "True connection"
  - Vector at 90° (purely imaginary)
  - Show projections onto real and imaginary axes
- Caption: "Cross-spectrum components for different phase relationships"

**Visualization 3**:
- Phase vs cross-spectrum components:
  - X-axis: phase (0° to 360°)
  - Y-axis: Real(S_xy) and Imag(S_xy)
  - Show sinusoidal relationships
  - Highlight: at 0° and 180°, Imag = 0
- Caption: "How phase affects real and imaginary components"

---

## Section 3: Defining Imaginary Coherence

**Purpose**: Formal definition.

**Content to cover**:
- **Standard coherence**:
  $$C_{xy}(f) = \frac{|S_{xy}(f)|}{\sqrt{S_{xx}(f) \cdot S_{yy}(f)}}$$
  (often squared, but let's use magnitude form here)
- **Imaginary coherence** (Nolte et al., 2004):
  $$ImCoh_{xy}(f) = \frac{Im(S_{xy}(f))}{\sqrt{S_{xx}(f) \cdot S_{yy}(f)}}$$
- **Properties**:
  - Range: -1 to +1 (can be negative!)
  - ImCoh = 0: either no connection OR zero-lag connection
  - ImCoh > 0: Y leads X in phase
  - ImCoh < 0: X leads Y in phase
  - |ImCoh| ≤ |Coherence| always
- **Absolute imaginary coherence**:
  - Sometimes use |ImCoh| for magnitude only
  - Loses phase direction information
- **Why this works**:
  - Volume conduction: Imag(S_xy) ≈ 0 → ImCoh ≈ 0
  - True connection: Imag(S_xy) ≠ 0 → ImCoh ≠ 0

**Visualization 4**:
- Formula comparison:
  - Standard coherence formula with magnitude highlighted
  - Imaginary coherence formula with Im() highlighted
  - Side by side, clear visual
- Caption: "Standard coherence vs imaginary coherence formulas"

---

## Section 4: Implementing Imaginary Coherence

**Purpose**: Practical computation.

**Content to cover**:
- Uses same cross-spectrum and power spectra as coherence
- Only difference: extract imaginary part before normalizing
- **Steps**:
  1. Compute cross-spectrum S_xy (complex)
  2. Compute power spectra S_xx, S_yy
  3. Extract Im(S_xy)
  4. Normalize: ImCoh = Im(S_xy) / √(S_xx × S_yy)
- **Implementation note**:
  - Cross-spectrum from Welch's method is already averaged
  - Take imaginary part of averaged cross-spectrum

**Code to produce**:

Function 1: `compute_imaginary_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    noverlap: int | None = None
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, imaginary_coherence)
Compute imaginary coherence (signed, -1 to +1)
```

Function 2: `compute_abs_imaginary_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    noverlap: int | None = None
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, abs_imaginary_coherence)
Absolute imaginary coherence (0 to 1)
```

**Visualization 5**:
- Computation pipeline:
  - Cross-spectrum (complex) → take imaginary part → normalize
  - Visual flowchart
- Caption: "Imaginary coherence computation"

---

## Section 5: Imaginary Coherence vs Volume Conduction

**Purpose**: Demonstrate the key advantage.

**Content to cover**:
- **Test scenario**:
  - Single source picked up by two electrodes (volume conduction)
  - Standard coherence: HIGH
  - Imaginary coherence: ~ZERO
- **True connectivity scenario**:
  - Two genuinely connected sources with lag
  - Standard coherence: moderate to high
  - Imaginary coherence: non-zero (detects the lag)
- **Comparison**:
  - Volume conduction: Coh high, ImCoh ≈ 0
  - True connection: Coh high, ImCoh ≠ 0
  - ImCoh distinguishes them!

**Code to produce**:

Function 3: `simulate_volume_conduction_imcoh`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Simulate two electrodes seeing same source (volume conduction)
Zero phase lag between outputs
```

Function 4: `simulate_lagged_connectivity`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    lag_samples: int
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Simulate true connectivity with time lag
Non-zero phase difference between outputs
```

**Visualization 6**:
- Side-by-side comparison:
  - Left: Volume conduction scenario
    - Two signals (nearly identical)
    - Coherence spectrum (high)
    - ImCoh spectrum (≈ 0)
  - Right: True lagged connectivity
    - Two signals (similar but lagged)
    - Coherence spectrum (high)
    - ImCoh spectrum (non-zero)
- Caption: "Imaginary coherence distinguishes true connectivity from volume conduction"

**Visualization 7**:
- Bar chart comparison:
  - Three scenarios: Volume conduction, True connection (small lag), True connection (large lag)
  - For each: standard coherence bar and ImCoh bar
  - Shows ImCoh is low only for volume conduction
- Caption: "Coherence vs imaginary coherence across scenarios"

---

## Section 6: The Sign of Imaginary Coherence

**Purpose**: Interpret positive vs negative values.

**Content to cover**:
- **Unlike standard coherence, ImCoh can be negative**
- **Interpretation**:
  - ImCoh > 0: Y's phase leads X's phase
  - ImCoh < 0: X's phase leads Y's phase
  - Convention depends on cross-spectrum definition
- **What does "lead" mean?**
  - If Y leads: Y's oscillation peaks before X's
  - Information flow interpretation (cautiously!)
- **Absolute value |ImCoh|**:
  - Ignores direction
  - Use when only coupling strength matters
  - Commonly used for connectivity matrices

**Visualization 8**:
- Phase lead/lag illustration:
  - Two sinusoids, one leading the other
  - Show how this translates to ImCoh sign
  - Clear visual of "Y leads X" vs "X leads Y"
- Caption: "Imaginary coherence sign reflects phase lead/lag"

---

## Section 7: Band-Averaged Imaginary Coherence

**Purpose**: Practical summarization.

**Content to cover**:
- Same logic as band coherence (F01)
- Average ImCoh across frequencies in band
- **Caution with signed values**:
  - If some frequencies positive, some negative, they may cancel!
  - Often use |ImCoh| for band averaging
  - Or average magnitude separately from phase information

**Code to produce**:

Function 5: `compute_band_imaginary_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    nperseg: int = 256
    absolute: bool = True  # use absolute value before averaging
Returns:
    float
Band-averaged (absolute) imaginary coherence
```

Function 6: `compute_all_band_imaginary_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
    nperseg: int = 256
    absolute: bool = True
Returns:
    dict[str, float]
Imaginary coherence for all standard bands
```

**Visualization 9**:
- Full spectrum with bands:
  - ImCoh spectrum (can be negative)
  - Shaded band regions
  - Bar chart of band |ImCoh| below
- Caption: "Band-averaged absolute imaginary coherence"

---

## Section 8: Imaginary Coherence Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- **For |ImCoh|**: matrix is symmetric (like coherence)
- **For signed ImCoh**: matrix is antisymmetric!
  - ImCoh(X,Y) = -ImCoh(Y,X)
  - Diagonal = 0
- Usually use |ImCoh| for connectivity matrices
- **Interpretation**:
  - Same as coherence but robust to volume conduction
  - Lower values overall (excludes zero-lag contributions)
  - But more trustworthy!

**Code to produce**:

Function 7: `compute_imaginary_coherence_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    nperseg: int = 256
    absolute: bool = True
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
Imaginary coherence matrix for all channel pairs
If absolute=True, matrix is symmetric
If absolute=False, matrix is antisymmetric
```

**Visualization 10**:
- Two matrices side by side:
  - Standard coherence matrix
  - Imaginary coherence matrix (|ImCoh|)
  - Same data, same scale
  - Show ImCoh is generally lower but pattern preserved
- Caption: "Coherence vs imaginary coherence matrices"

**Visualization 11**:
- Difference matrix:
  - Coherence - |ImCoh|
  - Highlights pairs where volume conduction was inflating coherence
  - Largest differences = most suspicious connections
- Caption: "Identifying potential volume conduction (Coh - ImCoh)"

---

## Section 9: Imaginary Coherence for Hyperscanning

**Purpose**: Apply to inter-brain connectivity.

**Content to cover**:
- **Between-brain connectivity**:
  - Volume conduction not an issue between separate heads
  - Standard coherence is "safe" for between-brain
  - ImCoh still valid, just less necessary
- **Within-brain connectivity**:
  - Volume conduction IS an issue
  - ImCoh recommended for within-brain blocks
- **Practical approach**:
  - Compute both coherence and ImCoh
  - Trust ImCoh for within-brain
  - Either works for between-brain
- **Consistency**:
  - Some prefer ImCoh everywhere for consistency
  - Same metric within and between → easier comparison

**Code to produce**:

Function 8: `compute_imaginary_coherence_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    nperseg: int = 256
    absolute: bool = True
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,
        "within_p2": NDArray,
        "between": NDArray,
        "full": NDArray
    }
Imaginary coherence for hyperscanning
```

**Visualization 12**:
- Hyperscanning comparison:
  - 2×2 grid
  - Row 1: Coherence (within-P1, between)
  - Row 2: ImCoh (within-P1, between)
  - Show within-brain differs more than between-brain
- Caption: "Coherence vs ImCoh in hyperscanning: within-brain shows bigger differences"

---

## Section 10: Trade-offs and When to Use

**Purpose**: Practical guidance.

**Content to cover**:
- **Advantages of ImCoh**:
  - Robust to volume conduction
  - Conservative (fewer false positives)
  - Can't be fooled by zero-lag artifacts
  - Sign gives phase direction
- **Disadvantages of ImCoh**:
  - Lower sensitivity (misses zero-lag true connections if they exist)
  - True instantaneous coupling would be invisible
  - Generally lower values (harder to detect weak effects)
- **When to use standard coherence**:
  - Source-level analysis (volume conduction addressed)
  - Between-brain hyperscanning (no volume conduction)
  - When you've validated no volume conduction concern
- **When to use ImCoh**:
  - Sensor-level EEG (default choice)
  - Within-brain connectivity
  - When you want to be conservative
  - Can't do source localization
- **Good practice**: Report both, discuss differences

**Visualization 13**:
- Decision flowchart:
  - "Analysis level?" → Source: use coherence; Sensor: continue
  - "Can you verify no volume conduction?" → Yes: coherence OK; No: use ImCoh
  - "Within-brain or between-brain?" → Within: ImCoh; Between: either
- Caption: "Choosing between coherence and imaginary coherence"

---

## Section 11: Comparison with HyPyP

**Purpose**: Connect to production tool.

**Content to cover**:
- HyPyP implements both coherence and imaginary coherence
- Located in `hypyp.analyses.connectivity` module
- Uses MNE-Python spectral functions
- **Comparison**:
  - Run our implementation and HyPyP on same data
  - Verify results match
  - Note any parameter differences

**Code to produce**:

Function 9: `compare_with_hypyp_coherence`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, Any]
    {
        "our_coh": NDArray,
        "our_imcoh": NDArray,
        "hypyp_coh": NDArray,  # if HyPyP available
        "hypyp_imcoh": NDArray,
        "match": bool
    }
Compare our implementation with HyPyP (if installed)
```

**Visualization 14**:
- Validation comparison:
  - Scatter plot: our ImCoh vs HyPyP ImCoh
  - Should fall on diagonal line
  - Correlation coefficient reported
- Caption: "Validating against HyPyP implementation"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Complex Plane Understanding
- Generate two signals with known phase relationship
- Compute cross-spectrum manually
- Extract real and imaginary parts
- Verify: phase = atan2(imag, real)
- Change phase, observe how real/imag change

**Exercise 2**: Volume Conduction Test
- Use `simulate_volume_conduction_imcoh`
- Compute coherence and ImCoh
- Verify: coherence high, ImCoh ≈ 0
- This is the key demonstration!

**Exercise 3**: True Connectivity Test
- Use `simulate_lagged_connectivity` with lag = 5 samples
- Compute coherence and ImCoh
- Both should be non-zero
- Compare with Exercise 2 results

**Exercise 4**: Sign Interpretation
- Create signals where Y leads X
- Compute signed ImCoh
- Verify positive
- Swap signals (X becomes Y)
- Verify sign flips

**Exercise 5**: Band Comparison
- Generate signals with:
  - True connectivity in alpha (with lag)
  - Volume conduction artifact in beta (zero lag)
- Compute band coherence and band ImCoh
- Alpha: both high
- Beta: coherence high, ImCoh low
- ImCoh correctly identifies true connectivity!

**Exercise 6**: Matrix Comparison
- Create 8-channel data with known connectivity structure
- Include some "volume conduction" pairs (zero lag)
- Compute coherence and ImCoh matrices
- Which pairs differ most?
- Those are the suspicious ones!

**Exercise 7**: Hyperscanning Application
- Create two-participant data
- Add within-brain volume conduction artifacts
- Add genuine between-brain connectivity
- Compute coherence and ImCoh for all blocks
- Verify ImCoh "cleans up" within-brain
- Between-brain similar for both

---

## Section 13: Add Functions to src/coherence.py

**Purpose**: Extend coherence module.

**Add to src/coherence.py**:
- `compute_imaginary_coherence`
- `compute_abs_imaginary_coherence`
- `simulate_volume_conduction_imcoh`
- `simulate_lagged_connectivity`
- `compute_band_imaginary_coherence`
- `compute_all_band_imaginary_coherence`
- `compute_imaginary_coherence_matrix`
- `compute_imaginary_coherence_hyperscanning`
- `compare_with_hypyp_coherence`

---

## Summary

**Key takeaways to list**:
- **Standard coherence** uses magnitude of cross-spectrum
- **Problem**: volume conduction creates zero-lag signals with high coherence
- **Imaginary coherence**: ImCoh = Im(S_xy) / √(S_xx × S_yy)
- Uses only the imaginary part of cross-spectrum
- **Volume conduction → zero lag → zero imaginary part → ImCoh ≈ 0**
- **True connectivity → non-zero lag → non-zero imaginary → ImCoh ≠ 0**
- Range: -1 to +1 (signed) or 0 to 1 (absolute)
- Sign indicates phase lead/lag direction
- **Trade-off**: more conservative, may miss true instantaneous coupling
- |ImCoh| matrix is symmetric; signed ImCoh is antisymmetric
- **For hyperscanning**: ImCoh useful for within-brain; either metric for between-brain
- **Best practice**: report both, note differences, interpret cautiously

---

## Discussion Questions

1. You find high coherence but low ImCoh between two nearby electrodes. What's the most likely explanation? What additional tests would you run?

2. A neuroscience paper reports high ImCoh between frontal and occipital sites during visual attention. Why is this more convincing than if they had reported standard coherence?

3. ImCoh = 0 could mean (a) no connection, or (b) true zero-lag connection. Is there any way to distinguish these cases? Does it matter for interpretation?

4. You're comparing two conditions and find coherence differs but ImCoh doesn't. What might be happening? Which result is more trustworthy?

5. For hyperscanning between-brain analysis, some argue ImCoh is unnecessary (no volume conduction). Others prefer using it for consistency. What's your position and why?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Complex plane | Zero-lag vs lagged vectors | signal_1, signal_2 |
| 2 | Complex plane | Multiple phase angles | varied |
| 3 | Line plot | Phase vs components | signal_1, signal_2 |
| 4 | Formula | Coherence vs ImCoh | highlight |
| 5 | Flowchart | ImCoh computation | process colors |
| 6 | Side-by-side | Volume conduction vs true | signal_1, signal_2 |
| 7 | Bar chart | Scenario comparison | signal_1, signal_2 |
| 8 | Waveforms | Phase lead/lag | signal_1, signal_2 |
| 9 | Spectrum + bars | Band ImCoh | band colors |
| 10 | 2 matrices | Coherence vs ImCoh | viridis |
| 11 | Matrix | Difference (Coh - ImCoh) | diverging |
| 12 | 2×2 grid | Hyperscanning comparison | subject_1, subject_2 |
| 13 | Decision tree | When to use which | varied |
| 14 | Scatter | Validation vs HyPyP | signal_1 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Handle complex number operations correctly
- [ ] Signed vs absolute ImCoh clearly distinguished
- [ ] Matrix symmetry/antisymmetry handled correctly
- [ ] Volume conduction simulation realistic
- [ ] HyPyP comparison gracefully handles missing package
- [ ] Clear documentation of sign conventions