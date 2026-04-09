# H02: Power Correlation (PowCorr) — Copilot Directive

## Metadata

- **Notebook ID**: H02
- **Title**: Power Correlation (PowCorr)
- **Branch**: `feature/H02_power_correlation`
- **Duration**: 45 minutes
- **Prerequisites**: H01 (Envelope Correlation), A03 (Power Spectrum)
- **Next**: Workshop completion / Advanced topics

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Distinguish power correlation from envelope correlation
2. Define power correlation as correlation of squared envelopes / band power
3. Implement power correlation computation
4. Understand time-windowed vs instantaneous approaches
5. Apply power correlation to hyperscanning analysis
6. Choose between envelope correlation and power correlation
7. Complete the amplitude connectivity toolkit

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.stats
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/phase.py, src/amplitude.py
```

---

## Section 1: Introduction — Power vs Amplitude

**Purpose**: Clarify the distinction and motivation.

**Content to cover**:
- **Recall H01**: Envelope correlation uses amplitude A(t)
- **Power**: P(t) = A(t)² (amplitude squared)
- **Why power?**:
  - Power is what we typically compute in spectral analysis
  - More directly related to "energy" in the signal
  - Emphasizes larger amplitudes more (squared)
- **Power correlation**: Correlation of power time series
- **Relationship**: If amplitudes are correlated, powers will be too (but not identical)
  - Squaring is monotonic → preserves correlation sign
  - But changes the distribution (emphasizes extremes)

**Key message**: "Power correlation captures co-modulation of signal energy, closely related to envelope correlation but with emphasis on larger fluctuations."

**Visualization 1**:
- Three-panel comparison:
  - Signal x(t)
  - Amplitude envelope A(t)
  - Power A(t)² 
- Show how power emphasizes peaks more
- Caption: "Signal → Amplitude → Power"

**Length**: 3-4 paragraphs

---

## Section 2: Power Correlation Definition

**Purpose**: Formal definition.

**Content to cover**:
- **Power correlation**:
  $$PowCorr = Pearson(P_x, P_y) = Pearson(A_x^2, A_y^2)$$
- **Instantaneous power**: P(t) = A(t)² = |z(t)|² where z is analytic signal
- **Properties**:
  - Range: -1 to +1 (like envelope correlation)
  - PowCorr = +1: powers perfectly correlated
  - PowCorr = 0: no linear power relationship
  - PowCorr < 0: anti-correlated (rare)
- **Relationship to envelope correlation**:
  - Usually highly correlated with CCorr
  - PowCorr typically slightly higher (squaring reduces noise relative to signal)
  - Not identical — different emphasis

**Visualization 2**:
- Scatter: envelope correlation vs power correlation for many pairs
- Typically on diagonal but not exactly
- Caption: "Envelope vs power correlation: related but distinct"

---

## Section 3: Two Approaches to Power Correlation

**Purpose**: Different computation methods.

**Content to cover**:
- **Approach 1**: Instantaneous power correlation
  - Compute P(t) = A(t)² at each time point
  - Correlate power time series
  - High temporal resolution
- **Approach 2**: Windowed band power correlation
  - Divide signal into epochs/windows
  - Compute band power for each window
  - Correlate power values across windows
  - Lower temporal resolution, more robust
- **When to use**:
  - Instantaneous: dynamic connectivity, short recordings
  - Windowed: more stable estimates, trial-based designs

**Visualization 3**:
- Side-by-side comparison:
  - Instantaneous: continuous power time series
  - Windowed: power values for discrete windows
- Caption: "Instantaneous vs windowed power"

---

## Section 4: Implementing Power Correlation

**Purpose**: Practical computation.

**Content to cover**:
- **Instantaneous power correlation pipeline**:
  1. Band-pass filter
  2. Hilbert transform → analytic signal
  3. Compute power: P(t) = |z(t)|² = A(t)²
  4. Pearson correlation of power time series
- **Windowed power correlation pipeline**:
  1. Divide into windows/epochs
  2. For each window: compute band power (FFT or filter+Hilbert)
  3. Get power vector for each signal
  4. Pearson correlation of power vectors

**Code to produce**:

Function 1: `compute_power_correlation`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    method: str = "instantaneous"  # or "windowed"
    window_sec: float = 1.0  # for windowed method
Returns:
    float
Power correlation between x and y
```

Function 2: `compute_instantaneous_power`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    NDArray[np.float64]
Instantaneous power time series
```

Function 3: `compute_windowed_band_power`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    window_sec: float = 1.0
    overlap: float = 0.5
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (time_centers, power_values)
Band power computed in sliding windows
```

**Visualization 4**:
- Computation pipeline for both methods:
  - Parallel flowcharts
  - Highlight the branching point
- Caption: "Power correlation computation methods"

---

## Section 5: Power Correlation with Examples

**Purpose**: Build intuition.

**Content to cover**:
- **Example 1**: Signals with correlated power modulation
  - Both signals get "stronger" and "weaker" together
  - PowCorr high
- **Example 2**: Independent power fluctuations
  - PowCorr ≈ 0
- **Example 3**: Compare with envelope correlation
  - Similar results, not identical

**Code to produce**:

Function 4: `generate_power_correlated_signals`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    power_correlation_target: float
    modulation_freq: float = 0.5
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate signals with specified power correlation
```

**Visualization 5**:
- Example signals with power overlaid:
  - Two signals
  - Power time series
  - Scatter of power values
  - PowCorr annotated
- Caption: "Power correlation example"

**Visualization 6**:
- CCorr vs PowCorr comparison:
  - Same signal pairs
  - Both metrics computed
  - Usually similar, sometimes different
- Caption: "When envelope and power correlation diverge"

---

## Section 6: Volume Conduction and Orthogonalization

**Purpose**: Apply lessons from H01.

**Content to cover**:
- **Same issue**: Volume conduction inflates power correlation
- **Same solution**: Orthogonalization
- **Orthogonalized power correlation**:
  - Orthogonalize signals (as in H01)
  - Compute power from orthogonalized envelopes
  - Correlate orthogonalized powers
- **Implementation**: Can use same orthogonalization, just square after

**Code to produce**:

Function 5: `compute_orthogonalized_power_correlation`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    symmetric: bool = True
Returns:
    float
Power correlation with orthogonalization
```

**Visualization 7**:
- Volume conduction test:
  - Standard PowCorr: high
  - Orthogonalized PowCorr: low
  - For true connection: both high
- Caption: "Orthogonalized power correlation is robust to volume conduction"

---

## Section 7: Power Correlation Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- Compute PowCorr for all channel pairs
- Properties: symmetric, -1 to +1, diagonal = 1
- Compare with envelope correlation matrix

**Code to produce**:

Function 6: `compute_power_correlation_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    orthogonalize: bool = True
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
Power correlation matrix
```

Function 7: `compute_power_correlation_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
    orthogonalize: bool = True
Returns:
    dict[str, NDArray[np.float64]]
Power correlation matrices for all bands
```

**Visualization 8**:
- Three matrices:
  - Envelope correlation
  - Power correlation
  - Difference (PowCorr - CCorr)
- Caption: "Comparing envelope and power correlation matrices"

---

## Section 8: Power Correlation for Hyperscanning

**Purpose**: Apply to inter-brain connectivity.

**Content to cover**:
- **Inter-brain power correlation**:
  - Do participants' power levels co-fluctuate?
  - Shared engagement, arousal, effort
- **Similar to envelope correlation** but with power emphasis
- **Research context**:
  - Less common than envelope correlation in literature
  - But conceptually similar
  - May be more robust to noise (squaring reduces relative noise)

**Code to produce**:

Function 8: `compute_power_correlation_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    orthogonalize_within: bool = True
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,
        "within_p2": NDArray,
        "between": NDArray,
        "full": NDArray
    }
Power correlation matrices for hyperscanning
```

**Visualization 9**:
- Hyperscanning power correlation:
  - Full matrix
  - Diverging colormap
- Caption: "Inter-brain power correlation"

---

## Section 9: Choosing Between Amplitude Metrics

**Purpose**: Practical guidance for metric selection.

**Content to cover**:
- **Envelope correlation (CCorr)**:
  - Most commonly used
  - More literature for comparison
  - Linear in amplitude
- **Power correlation (PowCorr)**:
  - Emphasizes larger fluctuations
  - May be more robust to low-amplitude noise
  - Matches spectral analysis (power, not amplitude)
- **When they differ**:
  - Large fluctuations dominant: similar
  - Small fluctuations important: may diverge
- **Recommendation**:
  - CCorr is the standard choice
  - PowCorr is alternative for robustness
  - Report which you use, consider computing both

**Visualization 10**:
- Decision guidance:
  - Simple flowchart for amplitude metric choice
  - Factors: literature comparison, noise level, fluctuation range
- Caption: "Choosing amplitude connectivity metrics"

---

## Section 10: Complete Amplitude Connectivity Comparison

**Purpose**: Wrap up amplitude-based metrics.

**Content to cover**:
- **Summary of amplitude metrics**:
  - Envelope correlation (CCorr)
  - Orthogonalized envelope correlation (CCorr-orth)
  - Power correlation (PowCorr)
  - Orthogonalized power correlation (PowCorr-orth)
- **Comparison with phase metrics**:
  - Phase: timing alignment (PLV, PLI, wPLI)
  - Amplitude: strength co-modulation (CCorr, PowCorr)
  - Different neural mechanisms, complementary information

**Code to produce**:

Function 9: `compute_all_amplitude_metrics`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, float]
    {
        "ccorr": float,
        "ccorr_orth": float,
        "powcorr": float,
        "powcorr_orth": float
    }
All amplitude connectivity metrics
```

Function 10: `compute_all_connectivity_metrics`
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
        "pli": float,
        "wpli": float,
        "coherence": float,
        "imaginary_coherence": float,
        "ccorr": float,
        "ccorr_orth": float,
        "powcorr": float,
        "powcorr_orth": float
    }
Complete connectivity profile
```

**Visualization 11**:
- Comprehensive comparison:
  - Bar chart of all metrics for one channel pair
  - Phase metrics grouped
  - Amplitude metrics grouped
- Caption: "Complete connectivity profile"

**Visualization 12**:
- Correlation matrix of metrics:
  - How do different metrics correlate with each other?
  - Phase metrics cluster
  - Amplitude metrics cluster
  - Some cross-correlation
- Caption: "Relationships between connectivity metrics"

---

## Section 11: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Power Correlation Basics
- Generate signals with correlated power modulation
- Compute instantaneous power correlation
- Verify high correlation

**Exercise 2**: Instantaneous vs Windowed
- Same signals
- Compute both instantaneous and windowed power correlation
- Compare values
- Discuss trade-offs

**Exercise 3**: CCorr vs PowCorr
- Generate signals where amplitude and power correlations differ
- Hint: add low-amplitude noise
- Compute both
- Discuss when they diverge

**Exercise 4**: Orthogonalization Check
- Simulate volume conduction
- Compute standard vs orthogonalized power correlation
- Verify orthogonalized version is robust

**Exercise 5**: Complete Metric Comparison
- Create channel pairs with different connectivity types
- Compute ALL metrics (phase + amplitude)
- Create comprehensive comparison table
- Interpret the patterns

**Exercise 6**: Hyperscanning Application
- Create two-participant data
- Compute both CCorr and PowCorr hyperscanning matrices
- Compare between-brain coupling estimates
- Which metric would you report and why?

---

## Section 12: Add Functions to src/amplitude.py

**Purpose**: Complete amplitude module.

**Add to src/amplitude.py**:
- `compute_power_correlation`
- `compute_instantaneous_power`
- `compute_windowed_band_power`
- `generate_power_correlated_signals`
- `compute_orthogonalized_power_correlation`
- `compute_power_correlation_matrix`
- `compute_power_correlation_matrix_bands`
- `compute_power_correlation_hyperscanning`
- `compute_all_amplitude_metrics`
- `compute_all_connectivity_metrics`

---

## Summary

**Key takeaways to list**:
- **Power correlation**: correlation of squared amplitude (power) time series
- Power = A² emphasizes larger fluctuations
- **Two approaches**: instantaneous (continuous) or windowed (epoched)
- Range: -1 to +1 (like envelope correlation)
- **Highly related to envelope correlation** — usually similar values
- PowCorr may be more robust to low-amplitude noise
- **Orthogonalization** addresses volume conduction (like CCorr)
- **For hyperscanning**: captures shared power fluctuations
- **Choice**: CCorr is standard, PowCorr is alternative
- Report which you use!
- **Complete toolkit now**: Phase (PLV, PLI, wPLI, COH, ImCOH) + Amplitude (CCorr, PowCorr)
- Different metrics capture different aspects of connectivity

---

## Discussion Questions

1. In what scenarios would you expect envelope correlation and power correlation to give notably different results? What signal characteristics would drive this difference?

2. You're designing a hyperscanning study and need to choose connectivity metrics. Would you compute both phase and amplitude metrics? How would you interpret results if they conflict?

3. The literature mostly reports envelope correlation (AEC). Is there value in also reporting power correlation, or does it add unnecessary complexity?

4. Power is A². What about using log-power (dB) for correlation? Would this be more or less sensitive to large fluctuations? When might log-power correlation be useful?

5. Looking across all the metrics covered in this workshop (PLV, PLI, wPLI, COH, ImCOH, CCorr, PowCorr), which would you recommend as a "default" set for a new hyperscanning study? Justify your choices.

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 3-panel | Signal → Amplitude → Power | signal_1 |
| 2 | Scatter | CCorr vs PowCorr | signal_1 |
| 3 | Side-by-side | Instantaneous vs windowed | signal_1, signal_2 |
| 4 | Pipeline | Two computation methods | process colors |
| 5 | Multi-panel | Example with power overlay | signal_1, signal_2 |
| 6 | Comparison | CCorr vs PowCorr divergence | signal_1, signal_2 |
| 7 | Bar chart | Volume conduction test | signal_1, signal_2 |
| 8 | 3 matrices | CCorr, PowCorr, difference | diverging |
| 9 | Heatmap | Hyperscanning power corr | diverging |
| 10 | Decision tree | Metric selection | varied |
| 11 | Bar chart | All metrics comparison | varied |
| 12 | Correlation matrix | Metrics relationships | diverging |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Power computed correctly (amplitude squared)
- [ ] Windowed method handles edge windows
- [ ] Orthogonalization reuses H01 functions
- [ ] Diverging colormap for correlation matrices
- [ ] All connectivity functions consistent interface
- [ ] compute_all_connectivity_metrics comprehensive