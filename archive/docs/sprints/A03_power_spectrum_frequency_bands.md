# A03: Power Spectrum and Frequency Bands — Copilot Directive

## Metadata

- **Notebook ID**: A03
- **Title**: Power Spectrum and Frequency Bands
- **Branch**: `feature/A03_power_spectrum_frequency_bands`
- **Duration**: 60 minutes
- **Prerequisites**: A02 (The Frequency Domain)
- **Next**: A04 (Filtering Signals)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain the difference between amplitude spectrum and power spectrum
2. Compute Power Spectral Density (PSD) using different methods
3. Apply Welch's method for robust spectral estimation
4. Define and extract EEG frequency bands (delta, theta, alpha, beta, gamma)
5. Compute and compare band power across signals or conditions

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.fft, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/spectral.py (from A02)
```

---

## Section 1: Introduction

**Purpose**: Motivate why power spectrum is the standard in EEG analysis.

**Content to cover**:
- Amplitude spectrum shows magnitude, but power is more meaningful for energy analysis
- Power Spectral Density (PSD) quantifies how signal energy is distributed across frequencies
- EEG research typically reports power, not amplitude
- Band power analysis is foundational: alpha power, beta power, etc.
- Connectivity metrics often use cross-spectral density (related concept)

**Length**: 3-4 paragraphs

---

## Section 2: From Amplitude to Power

**Purpose**: Clarify the relationship between amplitude and power.

**Content to cover**:
- Power is amplitude squared: P(f) = |X(f)|²
- Why squared? Power relates to energy (physical interpretation)
- Units: if signal is in µV, power is in µV²
- Often reported in decibels (dB): 10 * log10(P)
- Power spectrum vs Power Spectral Density (normalization by frequency resolution)

**Key equations**:
- Power: P[k] = |X[k]|²
- PSD: S[k] = |X[k]|² / (fs * N) — normalized by frequency bin width

**Visualization 1**:
- Same signal analyzed two ways
- Two subplots side by side:
  - Left: amplitude spectrum
  - Right: power spectrum
- Show how small amplitude differences become larger in power (squaring effect)
- Use same signal from A02 for continuity

---

## Section 3: Power Spectral Density (PSD)

**Purpose**: Introduce PSD as the standard representation.

**Content to cover**:
- PSD normalizes power by frequency resolution
- Units: µV²/Hz (power per unit frequency)
- Allows comparison across recordings with different durations or sampling rates
- Total power in a band = integral of PSD over that band (area under curve)
- PSD is what scipy.signal.welch() computes

**Code to produce**:

Function 1: `compute_psd_fft`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, psd_values)
Simple periodogram using FFT
One-sided spectrum
Proper normalization to µV²/Hz
```

**Visualization 2**:
- EEG-like synthetic signal (from A01)
- PSD plot with proper units on y-axis (µV²/Hz)
- Log scale on y-axis (common convention)
- Highlight the 1/f trend typical of EEG

---

## Section 4: The Periodogram and Its Limitations

**Purpose**: Understand why simple FFT-based PSD is problematic.

**Content to cover**:
- Periodogram: PSD estimated from single FFT
- Problem 1: high variance — noisy estimate, especially for short signals
- Problem 2: spectral leakage (as introduced in A02)
- Problem 3: single realization — no averaging
- Need for more robust estimation methods

**Visualization 3**:
- Generate same underlying signal multiple times with different noise realizations
- Compute periodogram for each
- Overlay 5-10 periodograms on same plot
- Show high variability between estimates
- Conclude: periodogram is inconsistent estimator

---

## Section 5: Welch's Method

**Purpose**: Introduce the standard robust PSD estimation technique.

**Content to cover**:
- Welch's method: divide signal into overlapping segments, compute periodogram of each, average
- Parameters:
  - Segment length (nperseg): controls frequency resolution
  - Overlap (noverlap): typically 50%
  - Window function: Hann is common default
- Trade-off: longer segments = better frequency resolution, fewer segments = more variance
- This is the de facto standard in EEG analysis

**Code to produce**:

Function 2: `compute_psd_welch`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    nperseg: Optional[int] = None (default to fs*2 for 0.5 Hz resolution)
    noverlap: Optional[int] = None (default to nperseg//2)
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, psd_values)
Wrapper around scipy.signal.welch with sensible defaults
```

**Visualization 4**:
- Same signal as Visualization 3
- Compare periodogram (single estimate) vs Welch (averaged)
- Two subplots:
  - Top: periodogram (noisy)
  - Bottom: Welch estimate (smooth)
- Same y-axis scale for fair comparison

**Visualization 5**:
- Effect of segment length on Welch estimate
- Three subplots with different nperseg values:
  - Short segments: smooth but poor frequency resolution
  - Medium segments: balanced
  - Long segments: better resolution but noisier
- Annotate frequency resolution for each

---

## Section 6: EEG Frequency Bands

**Purpose**: Define the canonical frequency bands used in EEG research.

**Content to cover**:
- Historical origin of band definitions (Berger, clinical EEG)
- Standard bands (note: boundaries vary slightly in literature):
  - Delta: 1-4 Hz (deep sleep, pathology)
  - Theta: 4-8 Hz (drowsiness, memory)
  - Alpha: 8-13 Hz (relaxed wakefulness, eyes closed)
  - Beta: 13-30 Hz (active thinking, attention)
  - Gamma: 30-100 Hz (cognitive processing, binding)
- Bands are somewhat arbitrary but widely adopted
- Sub-bands exist (low alpha, high alpha, etc.)

**Create constant/configuration**:
```python
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}
```

**Visualization 6**:
- PSD of EEG-like signal
- Shade each frequency band with its designated color (from STYLE_GUIDE.md)
- Legend showing band names
- Classic EEG-style PSD plot

---

## Section 7: Computing Band Power

**Purpose**: Extract power in specific frequency bands.

**Content to cover**:
- Band power: integral of PSD over frequency range
- Discrete approximation: sum of PSD values × frequency resolution
- Absolute band power: in µV²
- Relative band power: band power / total power (proportion)
- Log band power: 10 * log10(band power) — often more normally distributed

**Code to produce**:

Function 3: `compute_band_power`
```
Parameters:
    psd_frequencies: NDArray[np.float64]
    psd_values: NDArray[np.float64]
    band: tuple[float, float]
Returns:
    float (absolute band power)
Use trapezoidal integration (np.trapz) for accuracy
```

Function 4: `compute_all_band_powers`
```
Parameters:
    psd_frequencies: NDArray[np.float64]
    psd_values: NDArray[np.float64]
    bands: dict[str, tuple[float, float]] = EEG_BANDS
Returns:
    dict[str, float]
Compute power for all standard bands
```

Function 5: `compute_relative_band_power`
```
Parameters:
    psd_frequencies: NDArray[np.float64]
    psd_values: NDArray[np.float64]
    band: tuple[float, float]
    total_range: tuple[float, float] = (1.0, 100.0)
Returns:
    float (relative power, 0 to 1)
```

**Visualization 7**:
- Bar chart of absolute band powers
- X-axis: band names (delta, theta, alpha, beta, gamma)
- Y-axis: power (µV²)
- Use band colors from STYLE_GUIDE.md
- Show for synthetic EEG signal with known components

**Visualization 8**:
- Pie chart or stacked bar of relative band powers
- Shows proportion of total power in each band
- Alternative: horizontal stacked bar (easier to compare across conditions)

---

## Section 8: Comparing Spectra Across Conditions

**Purpose**: Demonstrate practical application of PSD comparison.

**Content to cover**:
- Common analysis: compare band power between conditions or groups
- Example: eyes open vs eyes closed (alpha increases with eyes closed)
- Visualization strategies for comparison
- Statistical considerations (brief mention, detailed in C03)

**Code to produce**:
- Generate two synthetic signals representing different "conditions"
- Condition 1: lower alpha power
- Condition 2: higher alpha power (simulating eyes closed)

**Visualization 9**:
- Overlay PSD of two conditions on same plot
- Use different colors for each condition
- Shade the alpha band region
- Show clear difference in alpha range

**Visualization 10**:
- Grouped bar chart comparing band powers
- Two groups (condition 1 vs condition 2)
- Five bands each
- Clear visual comparison

---

## Section 9: Decibel Scale

**Purpose**: Introduce logarithmic representation of power.

**Content to cover**:
- Decibels: dB = 10 * log10(P)
- Why use dB? Power spans many orders of magnitude
- Easier to visualize small differences
- Common in EEG literature
- Converting between linear and dB scales

**Code to produce**:

Function 6: `power_to_db`
```
Parameters:
    power: NDArray[np.float64] | float
    reference: float = 1.0
Returns:
    NDArray[np.float64] | float
Convert power to decibels: 10 * log10(power / reference)
```

**Visualization 11**:
- Same PSD plotted two ways:
  - Linear scale (µV²/Hz)
  - Decibel scale (dB)
- Show how dB scale reveals details in low-power regions

---

## Section 10: Practical Example — Multi-Component EEG Analysis

**Purpose**: Comprehensive analysis workflow.

**Content to cover**:
- Create realistic EEG-like signal with:
  - Strong alpha component (10 Hz)
  - Moderate beta component (20 Hz)
  - Weak theta component (6 Hz)
  - 1/f background noise
- Complete analysis pipeline

**Visualization 12**:
- Comprehensive figure with 4 panels:
  - Top left: time series (2-3 seconds)
  - Top right: PSD with band shading
  - Bottom left: absolute band power bar chart
  - Bottom right: relative band power bar chart
- Publication-quality figure

---

## Section 11: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Provide a synthetic signal with hidden frequency components
- Ask learner to:
  - Compute PSD using Welch's method
  - Identify dominant frequency band
  - Compute relative alpha power

**Exercise 2**:
- Investigate the effect of Welch parameters
- Same signal, vary nperseg: 128, 256, 512, 1024
- Plot all four PSDs overlaid
- Discuss trade-offs observed

**Exercise 3**:
- Compare two "subjects":
  - Subject A: high theta, low beta
  - Subject B: low theta, high beta
- Compute and visualize band power differences
- Which bands show the largest difference?

**Exercise 4**:
- Real-world scenario: alpha blocking
- Create signal where alpha power decreases halfway through
- Compute PSD for first half vs second half
- Visualize and quantify the alpha power change

---

## Section 12: Functions for src/spectral.py

**Purpose**: Identify functions to add to the spectral module.

**Functions to add to src/spectral.py**:
- `compute_psd_fft`
- `compute_psd_welch`
- `compute_band_power`
- `compute_all_band_powers`
- `compute_relative_band_power`
- `power_to_db`

**Constants to add to src/constants.py (new file)**:
- `EEG_BANDS`

---

## Summary

**Key takeaways to list**:
- Power spectrum is amplitude squared; PSD normalizes by frequency resolution
- Periodogram is a noisy estimator; Welch's method provides robust PSD estimation
- Welch parameters trade off frequency resolution against variance
- EEG frequency bands (delta, theta, alpha, beta, gamma) are standard analysis targets
- Band power can be absolute (µV²) or relative (proportion of total)
- Decibel scale is useful for visualizing power across orders of magnitude
- PSD comparison across conditions is a fundamental EEG analysis technique

---

## Discussion Questions

1. You are analyzing a 10-second EEG recording. What frequency resolution can you achieve with Welch's method using 2-second segments? What if you use 1-second segments?

2. A study reports increased "alpha power" in meditators vs controls. What additional information would you need to properly interpret this finding? (Hint: absolute vs relative, eyes open/closed, etc.)

3. Why might relative band power be more appropriate than absolute band power when comparing across participants?

4. The 1/f characteristic of EEG means lower frequencies have more power. How might this affect interpretation of band power comparisons between low-frequency (delta, theta) and high-frequency (beta, gamma) bands?

5. In hyperscanning, would you expect two interacting participants to have similar PSDs? Why or why not? What aspects might be similar or different?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 2 subplots (side-by-side) | Amplitude vs power spectrum | signal_1 |
| 2 | Single plot | PSD with log y-axis, 1/f trend | signal_1 |
| 3 | Single plot (overlay) | Multiple periodograms showing variance | signal_1 (varying alpha) |
| 4 | 2 subplots (column) | Periodogram vs Welch comparison | signal_1, signal_2 |
| 5 | 3 subplots (column) | Effect of segment length | signal_1 |
| 6 | Single plot | PSD with shaded frequency bands | band colors |
| 7 | Bar chart | Absolute band powers | band colors |
| 8 | Stacked bar / pie | Relative band powers | band colors |
| 9 | Single plot (overlay) | Two conditions PSD comparison | signal_1, signal_2 |
| 10 | Grouped bar chart | Band power comparison across conditions | signal_1, signal_2 |
| 11 | 2 subplots (column) | Linear vs dB scale | signal_1 |
| 12 | 2×2 grid | Comprehensive analysis figure | band colors |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants, especially EEG_BANDS)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses functions from A01 and A02 where appropriate
- [ ] Welch defaults are sensible for EEG analysis