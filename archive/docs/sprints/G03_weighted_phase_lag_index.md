# G03: Weighted Phase Lag Index (wPLI) — Copilot Directive

## Metadata

- **Notebook ID**: G03
- **Title**: Weighted Phase Lag Index (wPLI)
- **Branch**: `feature/G03_weighted_phase_lag_index`
- **Duration**: 50 minutes
- **Prerequisites**: G02 (Phase Lag Index)
- **Next**: H01 (Envelope Correlation)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain PLI's noise sensitivity problem
2. Define wPLI as a weighted improvement over PLI
3. Implement wPLI computation
4. Interpret wPLI values
5. Compare PLI, wPLI, and PLV across scenarios
6. Apply wPLI to hyperscanning analysis
7. Choose appropriately among phase-based metrics

---

## Required Imports
```
numpy, matplotlib, scipy.signal
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/phase.py, src/plv.py
```

---

## Section 1: The Problem with PLI

**Purpose**: Motivate wPLI.

**Content to cover**:
- **PLI advantage**: Ignores zero-lag (volume conduction robust)
- **PLI disadvantage**: Sign function is discontinuous!
- **The issue**:
  - Small noise fluctuations around Δφ = 0 cause sign flips
  - Phase difference slightly positive → +1
  - Same difference slightly negative → -1
  - These add noise to PLI estimate
- **Particularly problematic** when true phase difference is small
  - Genuine connection with small lag
  - Noise makes it look like no connection (signs balance)
- **What we want**: Weight contributions by how far from zero

**Key message**: "wPLI weights phase differences by their magnitude, reducing noise sensitivity near zero."

**Visualization 1**:
- PLI noise sensitivity illustration:
  - Phase differences fluctuating around small positive value
  - Some cross zero due to noise
  - Sign flips shown
  - PLI artificially reduced
- Caption: "Small phase differences are unstable in PLI"

**Length**: 3-4 paragraphs

---

## Section 2: The wPLI Solution — Weighting by Magnitude

**Purpose**: Introduce the weighting concept.

**Content to cover**:
- **Key idea**: Don't give equal weight to all phase differences
- **Phase differences near zero**: 
  - More likely to be volume conduction OR noise
  - Give them LESS weight
- **Phase differences far from zero**:
  - More likely to be real connectivity
  - Give them MORE weight
- **How to weight?**: Use |sin(Δφ)| as weight
  - sin(Δφ) = 0 when Δφ = 0 or π → weight = 0
  - |sin(Δφ)| = 1 when Δφ = ±π/2 → maximum weight
- **Effect**: Phase differences near zero contribute less

**Visualization 2**:
- Weighting function:
  - X-axis: phase difference (-π to π)
  - Y-axis: weight = |sin(Δφ)|
  - Zero weight at 0 and ±π
  - Maximum at ±π/2
- Caption: "wPLI weighting function: zero weight at 0 and π"

**Visualization 3**:
- Comparison:
  - PLI: all non-zero phases weighted equally (sign only)
  - wPLI: phases near zero weighted less
  - Visual of "contribution" from each phase value
- Caption: "PLI vs wPLI weighting scheme"

---

## Section 3: The wPLI Formula

**Purpose**: Formal definition.

**Content to cover**:
- **wPLI definition** (Vinck et al., 2011):
  $$wPLI = \frac{|\langle |Im(S_{xy})| \cdot sign(Im(S_{xy})) \rangle|}{\langle |Im(S_{xy})| \rangle}$$
  Or equivalently in terms of phase differences:
  $$wPLI = \frac{|\sum_t |sin(\Delta\phi_t)| \cdot sign(sin(\Delta\phi_t))|}{\sum_t |sin(\Delta\phi_t)|}$$
- **Simplification**:
  $$wPLI = \frac{|\sum_t sin(\Delta\phi_t)|}{\\sum_t |sin(\Delta\phi_t)|}$$
  Numerator: sum of sin(Δφ) (signed)
  Denominator: sum of |sin(Δφ)| (unsigned) — normalization
- **Properties**:
  - Range: 0 to 1
  - wPLI = 0: symmetric or near-zero phase differences
  - wPLI = 1: all phases at exactly ±π/2
  - Less sensitive to noise near zero than PLI

**Visualization 4**:
- Formula breakdown:
  - Numerator: signed sum (can cancel)
  - Denominator: absolute sum (always positive)
  - Ratio gives the asymmetry, weighted by magnitude
- Caption: "wPLI formula components"

---

## Section 4: Implementing wPLI

**Purpose**: Practical computation.

**Content to cover**:
- **Pipeline**:
  1. Band-pass filter both signals
  2. Extract phases (Hilbert)
  3. Compute phase differences Δφ(t)
  4. Compute sin(Δφ(t)) for each time point
  5. Numerator: |sum of sin values|
  6. Denominator: sum of |sin values|
  7. wPLI = numerator / denominator

**Code to produce**:

Function 1: `compute_wpli`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    float
Weighted Phase Lag Index between x and y
```

Function 2: `compute_wpli_from_phases`
```
Parameters:
    phase_x: NDArray[np.float64]
    phase_y: NDArray[np.float64]
Returns:
    float
wPLI from pre-computed phases
```

Function 3: `compute_wpli_from_cross_spectrum`
```
Parameters:
    cross_spectrum: NDArray[np.complex128]
Returns:
    float
wPLI from cross-spectral density (complex)
Uses imaginary part directly
```

**Visualization 5**:
- Computation pipeline:
  - Phase diff → sin → numerator (sum) vs denominator (sum of abs)
  - Clear visual flow
- Caption: "wPLI computation pipeline"

---

## Section 5: wPLI vs PLI — Noise Robustness

**Purpose**: Demonstrate the improvement.

**Content to cover**:
- **Test scenario**: 
  - True connection with SMALL phase lag
  - Add varying noise levels
- **Expected behavior**:
  - PLI: drops rapidly as noise increases (sign flips)
  - wPLI: more stable (small contributions weighted less)
- **Demonstration**: Run simulation, compare robustness

**Code to produce**:

Function 4: `compare_pli_wpli_noise`
```
Parameters:
    n_samples: int = 10000
    fs: float = 500.0
    frequency: float = 10.0
    true_phase_lag: float = 0.2  # small lag (radians)
    noise_levels: list[float] = [0.0, 0.1, 0.2, 0.5, 1.0]
    seed: int | None = None
Returns:
    dict[str, Any]
    {
        "noise_levels": list,
        "pli_values": list,
        "wpli_values": list
    }
Compare PLI and wPLI robustness to noise
```

**Visualization 6**:
- Noise robustness comparison:
  - X-axis: noise level
  - Y-axis: connectivity value
  - PLI line: drops quickly
  - wPLI line: more stable
- Caption: "wPLI is more robust to noise than PLI"

**Visualization 7**:
- Phase difference distributions at different noise levels:
  - Low noise: clear peak
  - High noise: spread, crosses zero often
  - Show how PLI is affected by zero-crossing, wPLI less so
- Caption: "Noise causes zero-crossing, destabilizing PLI"

---

## Section 6: wPLI vs Volume Conduction

**Purpose**: Verify wPLI maintains robustness to volume conduction.

**Content to cover**:
- wPLI inherits PLI's volume conduction robustness
- Volume conduction → Δφ ≈ 0 → sin(Δφ) ≈ 0 → numerator ≈ 0 → wPLI ≈ 0
- **Demonstration**: Volume conduction gives low wPLI (like PLI)

**Code to produce**:

Function 5: `demonstrate_wpli_volume_conduction`
```
Parameters:
    n_samples: int = 10000
    fs: float = 500.0
    frequency: float = 10.0
    seed: int | None = None
Returns:
    dict[str, Any]
    {
        "plv_vc": float, "pli_vc": float, "wpli_vc": float,
        "plv_tc": float, "pli_tc": float, "wpli_tc": float
    }
Compare PLV, PLI, wPLI for volume conduction vs true connectivity
```

**Visualization 8**:
- Three-metric comparison:
  - Bar chart for volume conduction: PLV high, PLI low, wPLI low
  - Bar chart for true connection: PLV high, PLI high, wPLI high
- Caption: "wPLI maintains robustness to volume conduction"

---

## Section 7: Debiased wPLI (Optional Advanced)

**Purpose**: Address small bias in wPLI.

**Content to cover**:
- Standard wPLI has small positive bias for finite samples
- **Debiased wPLI** (dwPLI) corrects this
- Formula adjustment removes bias term
- Not always necessary but good to know
- Some software packages use dwPLI by default

**Code to produce**:

Function 6: `compute_debiased_wpli`
```
Parameters:
    phase_x: NDArray[np.float64]
    phase_y: NDArray[np.float64]
Returns:
    float
Debiased wPLI with bias correction
```

**Visualization 9**:
- wPLI vs dwPLI comparison:
  - For independent signals
  - wPLI: small positive bias
  - dwPLI: closer to zero
- Caption: "Debiased wPLI reduces positive bias"

---

## Section 8: wPLI Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- Compute wPLI for all channel pairs
- Properties same as PLI matrix: symmetric, 0-1 range
- Expected: wPLI slightly different from PLI, especially for noisy pairs

**Code to produce**:

Function 7: `compute_wpli_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
wPLI matrix for all channel pairs
```

Function 8: `compute_wpli_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
Returns:
    dict[str, NDArray[np.float64]]
wPLI matrices for all frequency bands
```

**Visualization 10**:
- Three matrices side by side:
  - PLV matrix
  - PLI matrix
  - wPLI matrix
  - Same data, same scale
- Caption: "PLV vs PLI vs wPLI matrices"

---

## Section 9: wPLI for Hyperscanning

**Purpose**: Apply to inter-brain connectivity.

**Content to cover**:
- wPLI works well for hyperscanning
- For within-brain: provides volume conduction robustness with better stability than PLI
- For between-brain: comparable to PLV but more conservative
- **Recommendation**: wPLI is often the "safest" phase-based metric
  - Robust to volume conduction (like PLI)
  - Robust to noise (better than PLI)

**Code to produce**:

Function 9: `compute_wpli_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]
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
wPLI matrices for hyperscanning
```

**Visualization 11**:
- Hyperscanning wPLI:
  - Full matrix with participant blocks
  - Circular plot of between-brain connections
- Caption: "Inter-brain wPLI in hyperscanning"

---

## Section 10: Choosing Among Phase Metrics

**Purpose**: Practical guidance.

**Content to cover**:
- **Summary comparison**:

| Metric | Volume Conduction | Noise Robustness | Sensitivity |
|--------|-------------------|------------------|-------------|
| PLV | ❌ High false positives | Good | High |
| PLI | ✅ Robust | Poor (sign flips) | Moderate |
| wPLI | ✅ Robust | Good | Moderate |

- **Recommendations**:
  - **PLV**: Between-brain (no volume conduction) or source level
  - **PLI**: When simplicity matters, for comparison with older studies
  - **wPLI**: Default choice for sensor-level EEG (best balance)
- **Always report** which metric and why

**Visualization 12**:
- Decision flowchart:
  - "Source or sensor level?" → Source: PLV ok, Sensor: continue
  - "Volume conduction concern?" → Yes: PLI or wPLI
  - "Noisy data / small effects?" → Yes: wPLI preferred
- Caption: "Choosing a phase-based connectivity metric"

**Visualization 13**:
- Summary bar chart:
  - Three scenarios (volume conduction, true with noise, true clean)
  - Three metrics (PLV, PLI, wPLI)
  - Shows when each succeeds/fails
- Caption: "Phase metrics across scenarios"

---

## Section 11: Comparison with HyPyP

**Purpose**: Connect to production tool.

**Content to cover**:
- HyPyP implements PLV, PLI, and wPLI
- Compare our wPLI with HyPyP implementation
- Should match closely

**Code to produce**:

Function 10: `compare_with_hypyp_wpli`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, Any]
    {
        "our_wpli": NDArray,
        "hypyp_wpli": NDArray,
        "max_difference": float,
        "correlation": float
    }
Compare our wPLI with HyPyP
```

**Visualization 14**:
- Validation scatter plot:
  - Our wPLI vs HyPyP wPLI
  - Should fall on diagonal
- Caption: "Validating wPLI against HyPyP"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: wPLI Basics
- Generate phase-locked signals with moderate lag (π/4)
- Compute PLV, PLI, wPLI
- All should be reasonably high
- Add noise, observe stability differences

**Exercise 2**: Small Phase Lag Challenge
- Generate signals with SMALL phase lag (π/20)
- Compute PLI → may be unstable
- Compute wPLI → more stable
- This is wPLI's advantage

**Exercise 3**: Volume Conduction Check
- Simulate volume conduction
- Verify: PLV high, PLI low, wPLI low
- wPLI maintains robustness

**Exercise 4**: Noise Robustness Experiment
- Systematically vary noise level
- Plot PLI and wPLI vs noise
- Confirm wPLI degrades more slowly

**Exercise 5**: Three-Metric Comparison Matrix
- Create 8-channel data with varied connectivity types
- Compute PLV, PLI, wPLI matrices
- Compare — which pairs differ most across metrics?
- Interpret the differences

**Exercise 6**: Hyperscanning Application
- Create two-participant data
- Compute wPLI hyperscanning analysis
- Compare with PLV results
- Within-brain: larger differences
- Between-brain: smaller differences

---

## Section 13: Add Functions to src/plv.py

**Purpose**: Extend module with wPLI.

**Add to src/plv.py**:
- `compute_wpli`
- `compute_wpli_from_phases`
- `compute_wpli_from_cross_spectrum`
- `compare_pli_wpli_noise`
- `demonstrate_wpli_volume_conduction`
- `compute_debiased_wpli`
- `compute_wpli_matrix`
- `compute_wpli_matrix_bands`
- `compute_wpli_hyperscanning`
- `compare_with_hypyp_wpli`

---

## Summary

**Key takeaways to list**:
- **PLI problem**: Sign function sensitive to noise near zero
- **wPLI solution**: Weight contributions by |sin(Δφ)|
- Phases near zero get less weight → noise robustness
- Formula: wPLI = |Σ sin(Δφ)| / Σ |sin(Δφ)|
- Range: 0 to 1
- **Maintains volume conduction robustness** (like PLI)
- **Improved noise robustness** (better than PLI)
- **Debiased wPLI**: corrects small positive bias
- **For hyperscanning**: often the best default phase metric
- **Decision guide**:
  - PLV: simple, but vulnerable to volume conduction
  - PLI: VC robust, but noise sensitive
  - wPLI: VC robust AND noise robust — best of both!
- HyPyP implements all three

---

## Discussion Questions

1. You're analyzing EEG from a clinical population with noisier data than typical. Would this affect your metric choice? How?

2. wPLI weights phases near 0 and π less. But what if the true connectivity happens to have exactly π/2 phase lag? Is wPLI "unfairly" advantaged in detecting such connections?

3. Some researchers argue wPLI is now the "gold standard" for EEG phase connectivity. Do you agree? What might still justify using PLV in some contexts?

4. The progression PLV → PLI → wPLI shows increasing robustness but decreasing simplicity. When does simplicity matter? When should you prefer robustness?

5. You need to compare your hyperscanning results with a 2010 paper that used PLV. Can you use wPLI and still make valid comparisons? How would you handle this?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Illustration | PLI noise sensitivity | signal_1, warning |
| 2 | Function plot | Weighting function | signal_1 |
| 3 | Comparison | PLI vs wPLI weighting | signal_1, signal_2 |
| 4 | Formula | wPLI components | highlight |
| 5 | Pipeline | wPLI computation | process colors |
| 6 | Line plot | Noise robustness | signal_1, signal_2 |
| 7 | Histograms | Phase distributions + noise | signal_1 |
| 8 | Bar chart | Three metrics comparison | signal_1, signal_2, accent |
| 9 | Comparison | wPLI vs dwPLI | signal_1, signal_2 |
| 10 | 3 matrices | PLV/PLI/wPLI | viridis |
| 11 | Heatmap + circular | Hyperscanning wPLI | subject_1, subject_2 |
| 12 | Decision tree | Metric selection | varied |
| 13 | Bar chart | Metrics across scenarios | signal_1, signal_2, accent |
| 14 | Scatter | Validation vs HyPyP | signal_1 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Division by zero handled (denominator = 0)
- [ ] wPLI clipped to [0, 1] if numerical issues
- [ ] Debiased formula implemented correctly
- [ ] Cross-spectrum based computation matches phase-based
- [ ] Noise robustness clearly demonstrated
- [ ] Comparison with HyPyP gracefully handles missing package