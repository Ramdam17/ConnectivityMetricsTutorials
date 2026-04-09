# F01: Spectral Coherence — Copilot Directive

## Metadata

- **Notebook ID**: F01
- **Title**: Spectral Coherence (COH)
- **Branch**: `feature/F01_spectral_coherence`
- **Duration**: 65 minutes
- **Prerequisites**: A02 (Frequency Domain), A03 (Power Spectrum), E02 (Introduction to Hyperscanning)
- **Next**: F02 (Imaginary Coherence)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Define coherence as correlation in the frequency domain
2. Understand the relationship between coherence, cross-spectrum, and power spectra
3. Implement coherence computation using Welch's method
4. Interpret coherence values (0 to 1 scale)
5. Recognize coherence's sensitivity to volume conduction
6. Compute coherence matrices for multi-channel data
7. Apply coherence to hyperscanning inter-brain connectivity

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.fft
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/spectral.py, src/hyperscanning.py
New: src/coherence.py
```

---

## Section 1: Introduction — Correlation in the Frequency Domain

**Purpose**: Introduce coherence conceptually.

**Content to cover**:
- **Recall correlation**: measures linear relationship between two signals
- **Problem**: time-domain correlation mixes all frequencies
- **Question**: Are signals related at SPECIFIC frequencies?
- **Coherence**: frequency-resolved correlation
  - "How correlated are X and Y at 10 Hz?"
  - "What about at 20 Hz?"
- **Intuition**: 
  - Two musicians playing together
  - May be synchronized at tempo (4 Hz) but not at higher harmonics
  - Coherence reveals frequency-specific coupling
- **Historical importance**:
  - Classic EEG connectivity measure
  - Used since 1960s-70s
  - Foundation for modern metrics

**Key message**: "Coherence tells us how correlated two signals are at each frequency."

**Visualization 1**:
- Two signals with frequency-specific relationship:
  - Correlated at 10 Hz (alpha)
  - Uncorrelated at 20 Hz (beta)
- Time series on top
- Coherence spectrum below showing peak at 10 Hz
- Caption: "Coherence reveals frequency-specific correlation"

**Length**: 4-5 paragraphs

---

## Section 2: The Mathematics of Coherence

**Purpose**: Formal definition.

**Content to cover**:
- **Cross-spectral density (CSD)**:
  - S_xy(f) = expected value of X(f) × Y*(f)
  - Complex-valued: magnitude and phase
  - Measures co-variation at each frequency
- **Power spectral density (PSD)**:
  - S_xx(f) = |X(f)|² (power of X at frequency f)
  - S_yy(f) = |Y(f)|² (power of Y at frequency f)
- **Coherence definition**:
  $$C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) \cdot S_{yy}(f)}$$
- **Properties**:
  - Range: 0 ≤ C_xy(f) ≤ 1
  - C = 1: perfect linear relationship at frequency f
  - C = 0: no linear relationship at frequency f
  - Normalized: independent of signal amplitudes
- **Interpretation**:
  - Squared correlation coefficient at each frequency
  - What fraction of Y's power at f is linearly related to X's power at f

**Visualization 2**:
- Mathematical diagram:
  - X(f) and Y(f) as vectors in complex plane
  - Cross-spectrum as their product
  - Normalization by magnitudes
- Caption: "Coherence formula components"

---

## Section 3: Estimating Coherence with Welch's Method

**Purpose**: Practical computation.

**Content to cover**:
- **Challenge**: raw periodogram is noisy
- **Solution**: Welch's method (as used for PSD in A03)
- **Steps**:
  1. Divide signals into overlapping segments
  2. Window each segment (Hanning, Hamming, etc.)
  3. Compute FFT of each segment
  4. Average cross-spectra and power spectra across segments
  5. Compute coherence from averaged spectra
- **Why averaging helps**:
  - Single segment: coherence always = 1 (no averaging)
  - Multiple segments: noise averages out, true relationship remains
- **Trade-off**:
  - More segments → better estimate, worse frequency resolution
  - Fewer segments → better frequency resolution, noisier estimate
- **Key insight**: Need multiple segments for meaningful coherence!

**Code to produce**:

Function 1: `compute_cross_spectrum`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    noverlap: int | None = None  # default: nperseg // 2
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.complex128]]
    (frequencies, cross_spectral_density)
Compute cross-spectral density using Welch's method
Returns complex CSD (magnitude and phase information)
```

Function 2: `compute_coherence`
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
    (frequencies, coherence)
Compute magnitude-squared coherence using Welch's method
Returns real-valued coherence (0 to 1)
```

**Visualization 3**:
- Welch's method illustration for coherence:
  - Two signals divided into segments
  - Each segment pair → cross-spectrum
  - Average → smooth coherence estimate
- Caption: "Welch's method for coherence estimation"

---

## Section 4: Coherence of Simple Signals

**Purpose**: Build intuition with controlled examples.

**Content to cover**:
- **Example 1**: Identical signals
  - Y = X
  - Coherence = 1 at all frequencies
- **Example 2**: Identical + noise
  - Y = X + noise
  - Coherence < 1, depends on SNR
- **Example 3**: Phase-shifted signals
  - Y = X delayed by τ
  - Coherence still = 1! (phase doesn't affect coherence magnitude)
- **Example 4**: Scaled signals
  - Y = α × X
  - Coherence = 1 (normalization removes amplitude effects)
- **Example 5**: Uncorrelated signals
  - X and Y independent
  - Coherence ≈ 0 (with noise from finite data)
- **Example 6**: Frequency-specific relationship
  - Y has same 10 Hz component as X, different 20 Hz component
  - Coherence high at 10 Hz, low at 20 Hz

**Code to produce**:

Function 3: `generate_coherent_signals`
```
Parameters:
    n_samples: int
    fs: float
    frequency: float
    coherence_level: float  # 0 to 1
    snr_db: float = 20.0
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate two signals with specified coherence at given frequency
Uses shared component + independent noise to control coherence
```

**Visualization 4**:
- 3×2 grid of examples:
  - Each subplot: time series (small) + coherence spectrum
  - Identical, Identical+noise, Phase-shifted
  - Scaled, Uncorrelated, Frequency-specific
- Caption: "Coherence behavior in different scenarios"

**Visualization 5**:
- Coherence vs SNR:
  - X-axis: SNR (dB)
  - Y-axis: Coherence at target frequency
  - Shows how noise reduces coherence
- Caption: "Coherence decreases with noise"

---

## Section 5: Coherence Interpretation

**Purpose**: What do coherence values mean?

**Content to cover**:
- **Coherence = 1**: 
  - Perfect linear relationship at that frequency
  - Could be: same source, strong coupling, common input
- **Coherence = 0**:
  - No linear relationship
  - Signals may still be nonlinearly related!
  - Or just independent
- **Coherence between 0 and 1**:
  - Partial linear relationship
  - Noise contamination
  - Variable relationship over time
- **Squared coherence interpretation**:
  - C² = fraction of variance explained
  - C = 0.7 means C² = 0.49 → 49% of Y explained by X
- **Beware**:
  - High coherence ≠ causation
  - Could be common input (both driven by third source)
  - Volume conduction creates spurious coherence!

**Visualization 6**:
- Coherence interpretation scale:
  - Vertical bar from 0 to 1
  - Labeled regions: "No relationship" (0-0.2), "Weak" (0.2-0.4), "Moderate" (0.4-0.6), "Strong" (0.6-0.8), "Very strong" (0.8-1.0)
  - Example signals at each level
- Caption: "Interpreting coherence values"

---

## Section 6: The Volume Conduction Problem for Coherence

**Purpose**: Critical limitation of coherence.

**Content to cover**:
- **Recall from C01**: Volume conduction spreads signals
- **Impact on coherence**:
  - Single source appears at multiple electrodes
  - These electrodes show HIGH coherence (same signal!)
  - But it's not TRUE connectivity
- **Signature of volume conduction**:
  - Zero phase lag (signals are identical)
  - Coherence captures this as strong relationship
- **Why coherence fails here**:
  - Coherence uses MAGNITUDE of cross-spectrum
  - Ignores phase information
  - Can't distinguish zero-lag from true lagged connectivity
- **Solutions**:
  - **Imaginary coherence** (F02): uses only imaginary part
  - **PLI/wPLI** (G02, G03): ignores zero-phase contributions
  - **Source localization**: analyze at source level
- **For hyperscanning**: Less critical for BETWEEN-brain connectivity (separate heads), but still matters for WITHIN-brain

**Visualization 7**:
- Volume conduction demonstration:
  - Single simulated source
  - Two "electrodes" picking it up
  - High coherence between them
  - Phase difference ≈ 0
- Caption: "Volume conduction creates spurious coherence"

**Visualization 8**:
- Comparison diagram:
  - Left: True connectivity (lagged, lower coherence)
  - Right: Volume conduction (zero-lag, high coherence)
  - Same coherence value, different meanings!
- Caption: "Coherence can't distinguish true connectivity from volume conduction"

---

## Section 7: Band-Averaged Coherence

**Purpose**: Summarize coherence across frequency bands.

**Content to cover**:
- Full coherence spectrum can be complex to interpret
- Often want single value per frequency band:
  - Delta (1-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-100 Hz)
- **Band coherence**: average coherence across frequencies in band
- **Weighted average**: weight by power in band (optional)
- **Caution**: averaging may obscure narrow-band effects

**Code to produce**:

Function 4: `compute_band_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    nperseg: int = 256
    method: str = "mean"  # or "weighted"
Returns:
    float
Average coherence in specified frequency band
```

Function 5: `compute_all_band_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
    nperseg: int = 256
Returns:
    dict[str, float]
    {"delta": ..., "theta": ..., "alpha": ..., "beta": ..., "gamma": ...}
Coherence for all standard EEG frequency bands
Default bands if None provided
```

**Visualization 9**:
- Two-panel figure:
  - Top: Full coherence spectrum with band regions shaded
  - Bottom: Bar chart of band-averaged coherence
- Caption: "Band-averaged coherence summarizes frequency-specific coupling"

---

## Section 8: Coherence Matrix

**Purpose**: Scale up to multi-channel analysis.

**Content to cover**:
- With n channels, compute coherence for all pairs
- Result: n × n coherence matrix (per frequency or band)
- **Properties**:
  - Symmetric: C_xy = C_yx
  - Diagonal = 1 (self-coherence)
  - Values 0 to 1
- **For EEG**: 64 channels → 64×64 matrix
- **Storage**: only need upper triangle (+ diagonal)
- **Visualization**: heatmap, circular plot (as in C02)

**Code to produce**:

Function 6: `compute_coherence_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float] | None = None  # if None, return full spectrum
    nperseg: int = 256
Returns:
    NDArray[np.float64]  # (n_channels, n_channels) or (n_channels, n_channels, n_freqs)
Coherence matrix for all channel pairs
If band specified, returns single matrix of band-averaged coherence
If band is None, returns 3D array with full spectrum for each pair
```

Function 7: `compute_coherence_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
    nperseg: int = 256
Returns:
    dict[str, NDArray[np.float64]]
    {"delta": matrix, "theta": matrix, ...}
Coherence matrices for all frequency bands
```

**Visualization 10**:
- Coherence matrix example:
  - Simulated 16-channel data
  - Heatmap with proper colormap (0-1 scale)
  - Channel labels
  - Colorbar
- Caption: "Coherence matrix across EEG channels"

**Visualization 11**:
- Band comparison:
  - 2×3 grid of coherence matrices (one per band)
  - Same color scale across all
  - Show band-specific connectivity patterns
- Caption: "Coherence patterns across frequency bands"

---

## Section 9: Coherence for Hyperscanning

**Purpose**: Apply coherence to inter-brain connectivity.

**Content to cover**:
- **Inter-brain coherence**:
  - Compute coherence between P1 and P2 channels
  - Reveals frequency-specific inter-brain coupling
- **Structure**:
  - Within-P1 coherence matrix
  - Within-P2 coherence matrix
  - Between-brain coherence matrix (P1 channels × P2 channels)
- **Advantages for hyperscanning**:
  - Volume conduction NOT an issue between brains
  - Can trust inter-brain coherence more than intra-brain
- **Still useful within-brain**:
  - But interpret with caution (volume conduction)
  - Or use imaginary coherence (F02)
- **Research findings**:
  - Inter-brain coherence increases during cooperation
  - Frequency bands differ by task (theta for memory, alpha for attention)
  - Coherence predicts task performance

**Code to produce**:

Function 8: `compute_coherence_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    nperseg: int = 256
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,  # (n_ch_p1, n_ch_p1)
        "within_p2": NDArray,  # (n_ch_p2, n_ch_p2)
        "between": NDArray,    # (n_ch_p1, n_ch_p2)
        "full": NDArray        # (n_total, n_total)
    }
Compute coherence matrices for hyperscanning
```

Function 9: `compute_global_coherence_hyperscanning`
```
Parameters:
    coherence_dict: dict[str, NDArray[np.float64]]
Returns:
    dict[str, float]
    {
        "mean_within_p1": float,
        "mean_within_p2": float,
        "mean_between": float,
        "ratio_between_within": float
    }
Summary statistics for hyperscanning coherence
```

**Visualization 12**:
- Hyperscanning coherence:
  - Full 2n×2n matrix with blocks labeled
  - Or separate panels for within-P1, within-P2, between
  - Subject-specific colors for clarity
- Caption: "Inter-brain coherence in hyperscanning"

**Visualization 13**:
- Circular hyperscanning plot:
  - Two semicircles (P1 left, P2 right)
  - Lines colored by coherence strength
  - Between-brain connections highlighted
- Caption: "Visualizing inter-brain coherence"

---

## Section 10: Statistical Significance for Coherence

**Purpose**: Determine what coherence is "real."

**Content to cover**:
- **Random signals have non-zero coherence** due to chance
- **Theoretical threshold** (for independent signals):
  - Depends on number of segments (K)
  - Threshold ≈ 1 - α^(1/(K-1)) for significance level α
- **Surrogate testing** (more robust):
  - Shuffle one signal (destroy relationship)
  - Compute surrogate coherence
  - Repeat many times → null distribution
  - Compare observed to null
- **Multiple comparisons**:
  - Many frequency bins, many channel pairs
  - Apply FDR or Bonferroni correction

**Code to produce**:

Function 10: `coherence_significance_threshold`
```
Parameters:
    n_segments: int
    alpha: float = 0.05
Returns:
    float
Theoretical significance threshold for coherence
Based on formula: 1 - alpha^(1/(n_segments-1))
```

Function 11: `coherence_surrogate_test`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    n_surrogates: int = 500
    nperseg: int = 256
    seed: int | None = None
Returns:
    dict[str, Any]
    {"observed": float, "null_mean": float, "null_std": float, "pvalue": float, "threshold_95": float}
Surrogate testing for coherence significance
```

**Visualization 14**:
- Significance testing:
  - Coherence spectrum with significance threshold line
  - Significant frequencies highlighted
  - Null distribution inset
- Caption: "Testing coherence significance"

---

## Section 11: Comparison with scipy.signal.coherence

**Purpose**: Validate against standard library.

**Content to cover**:
- scipy.signal.coherence is the standard implementation
- Our implementation should match (or very close)
- **Comparison**:
  - Same input signals
  - Same parameters (nperseg, noverlap, window)
  - Compare outputs
- **Why implement ourselves?**
  - Understanding!
  - Flexibility for extensions
  - Foundation for imaginary coherence, etc.
- **When to use scipy**:
  - Production code
  - Need speed
  - Trust validated implementation

**Code to produce**:

Function 12: `compare_with_scipy_coherence`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    nperseg: int = 256
Returns:
    dict[str, Any]
    {"our_coherence": NDArray, "scipy_coherence": NDArray, "max_difference": float, "correlation": float}
Compare our implementation with scipy.signal.coherence
```

**Visualization 15**:
- Comparison plot:
  - Our coherence vs scipy coherence
  - Overlay or difference plot
  - Should be nearly identical
- Caption: "Validating our implementation against scipy"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Basic Coherence
- Generate two signals with shared 10 Hz component
- Add independent noise to each
- Compute coherence spectrum
- Identify the peak at 10 Hz
- Vary noise level and observe effect on coherence

**Exercise 2**: Phase Effects
- Create two identical 10 Hz signals
- Add phase shifts (0°, 45°, 90°, 180°)
- Compute coherence for each
- Verify coherence is unchanged by phase shift

**Exercise 3**: Volume Conduction Simulation
- Simulate single source appearing at two electrodes
- Compute coherence (should be very high)
- Note: this is SPURIOUS connectivity
- Save this example for comparison with imaginary coherence (F02)

**Exercise 4**: Band Coherence
- Generate multi-frequency signals
- High coupling at alpha, low at beta, medium at theta
- Compute band-averaged coherence
- Verify pattern matches your design

**Exercise 5**: Coherence Matrix
- Create 8-channel simulated data
- Channels 1-4 share some activity (cluster 1)
- Channels 5-8 share different activity (cluster 2)
- Little connection between clusters
- Compute and visualize coherence matrix
- Do you see the cluster structure?

**Exercise 6**: Hyperscanning Coherence
- Simulate two "participants" (6 channels each)
- Add inter-brain coherence between specific channel pairs
- Compute full hyperscanning coherence
- Extract and visualize between-brain matrix
- Test significance against surrogates

**Exercise 7**: Scipy Comparison
- Choose random signals
- Compute coherence with your function
- Compute coherence with scipy.signal.coherence
- Verify they match (within numerical precision)

---

## Section 13: Functions for src/coherence.py

**Purpose**: Create coherence analysis module.

**Create new file src/coherence.py with**:
- `compute_cross_spectrum`
- `compute_coherence`
- `generate_coherent_signals`
- `compute_band_coherence`
- `compute_all_band_coherence`
- `compute_coherence_matrix`
- `compute_coherence_matrix_bands`
- `compute_coherence_hyperscanning`
- `compute_global_coherence_hyperscanning`
- `coherence_significance_threshold`
- `coherence_surrogate_test`
- `compare_with_scipy_coherence`

---

## Summary

**Key takeaways to list**:
- **Coherence**: correlation in the frequency domain
- Formula: C_xy(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))
- Range: 0 (no relationship) to 1 (perfect linear relationship)
- Computed using Welch's method (averaged segments)
- **Interpretation**: fraction of variance explained at each frequency
- Independent of phase — captures magnitude of relationship only
- **Volume conduction problem**: creates spurious high coherence at zero lag
- Coherence can't distinguish true coupling from volume conduction
- **Band coherence**: summarize across frequency ranges
- **Coherence matrix**: all pairwise coherences (symmetric, diagonal = 1)
- **For hyperscanning**: inter-brain coherence safe from volume conduction
- **Significance**: theoretical threshold or surrogate testing
- Next: Imaginary Coherence (F02) addresses volume conduction issue

---

## Discussion Questions

1. Two EEG channels show coherence = 0.8 at 10 Hz. What are at least three different explanations for this finding? How would you distinguish between them?

2. You compute coherence between a motor cortex electrode and a parietal electrode during movement. High coherence in beta band. A colleague says "they're both picking up muscle artifact." How do you investigate this concern?

3. Inter-brain coherence during conversation is higher in theta band for real pairs vs pseudo-pairs. What does this tell you? What doesn't it tell you?

4. Coherence is normalized by power, so it's "independent of amplitude." But in what ways might amplitude still affect your coherence results?

5. You want to compare coherence across subjects with different data lengths. Subject A has 10 minutes of data, Subject B has 2 minutes. What concerns arise? How might you address them?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Time series + spectrum | Frequency-specific coherence | signal_1, signal_2 |
| 2 | Diagram | Formula components | signal_1, signal_2 |
| 3 | Process illustration | Welch's method | signal_1 |
| 4 | 3×2 grid | Example scenarios | signal_1, signal_2 |
| 5 | Line plot | Coherence vs SNR | signal_1 |
| 6 | Scale bar | Coherence interpretation | gradient |
| 7 | Simulation | Volume conduction | signal_1, signal_2 |
| 8 | Comparison | True vs spurious | signal_1, warning |
| 9 | Dual panel | Spectrum + band bars | band colors |
| 10 | Heatmap | Coherence matrix | viridis (0-1) |
| 11 | 2×3 grid | Band matrices | viridis |
| 12 | Heatmap | Hyperscanning full matrix | subject_1, subject_2 |
| 13 | Circular | Hyperscanning connections | subject_1, subject_2 |
| 14 | Spectrum + threshold | Significance testing | signal_1, highlight |
| 15 | Comparison | Our vs scipy | signal_1, signal_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Welch parameters properly handled
- [ ] Edge cases for short signals handled
- [ ] Coherence values clipped to [0, 1] if numerical issues
- [ ] Matrix diagonal handled appropriately (set to 1 or NaN)
- [ ] Hyperscanning functions handle different channel counts
- [ ] Comparison with scipy validates our implementation