# B03: Amplitude Envelope — Copilot Directive

## Metadata

- **Notebook ID**: B03
- **Title**: Amplitude Envelope
- **Branch**: `feature/B03_amplitude_envelope`
- **Duration**: 60 minutes
- **Prerequisites**: B01 (The Hilbert Transform)
- **Next**: B04 (Wavelets and Time-Frequency Analysis)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand the amplitude envelope as instantaneous signal strength
2. Extract and interpret envelopes from band-limited signals
3. Analyze slow fluctuations in oscillatory power
4. Distinguish between amplitude, power, and envelope
5. Apply envelope smoothing techniques
6. Recognize envelope correlation as a connectivity mechanism

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.ndimage, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/filtering.py, src/hilbert.py
```

---

## Section 1: Introduction

**Purpose**: Motivate amplitude envelope analysis for connectivity.

**Content to cover**:
- Phase-based metrics (PLV, PLI) capture synchronization of oscillation timing
- Amplitude-based metrics capture co-fluctuation of oscillation strength
- Different neural mechanism: amplitude coupling vs phase coupling
- Envelope correlation (CCorr) and power correlation (PowCorr) are key metrics
- Amplitude fluctuations are slower than the oscillation itself
- This enables analysis of neural communication at different timescales

**Length**: 3-4 paragraphs

---

## Section 2: What is the Amplitude Envelope?

**Purpose**: Define and visualize the envelope concept.

**Content to cover**:
- Envelope: the curve that traces the peaks of an oscillation
- For a narrowband signal, envelope represents instantaneous amplitude
- Computed as magnitude of analytic signal: A(t) = |z(t)|
- Envelope is always positive (magnitude)
- Envelope changes slowly compared to the carrier oscillation
- Analogy: AM radio — envelope carries information, carrier is high frequency

**Visualization 1**:
- Amplitude-modulated signal (10 Hz carrier, 1 Hz modulation)
- Single plot showing:
  - Original oscillating signal
  - Upper envelope (positive)
  - Lower envelope (negative, mirrored)
- Clear visual of how envelope "hugs" the oscillation

**Visualization 2**:
- EEG-like alpha signal with natural amplitude fluctuations
- Overlay the envelope
- Show how envelope captures "bursts" of alpha activity

---

## Section 3: Envelope vs Amplitude vs Power

**Purpose**: Clarify terminology that is often confused.

**Content to cover**:
- **Instantaneous amplitude**: |z(t)| — the envelope (always positive)
- **Signal amplitude**: the value x(t) — can be positive or negative
- **Power**: amplitude squared — A(t)² or |z(t)|²
- **Band power**: integral of PSD over frequency range (from A03)
- In practice, "envelope" and "instantaneous amplitude" are synonymous
- Power emphasizes large amplitudes more (squaring effect)
- Literature sometimes uses these terms loosely

**Table**:
| Term | Definition | Range | Units (if signal in µV) |
|------|------------|-------|-------------------------|
| Signal amplitude | x(t) | (-∞, +∞) | µV |
| Instantaneous amplitude | \|z(t)\| | [0, +∞) | µV |
| Envelope | \|z(t)\| | [0, +∞) | µV |
| Instantaneous power | \|z(t)\|² | [0, +∞) | µV² |

**Visualization 3**:
- Same signal shown three ways:
  - Original signal x(t)
  - Envelope A(t) = |z(t)|
  - Instantaneous power A(t)²
- Three subplots, aligned in time
- Note how power exaggerates peaks more than envelope

---

## Section 4: Extracting the Envelope — Complete Pipeline

**Purpose**: Reinforce the standard workflow.

**Content to cover**:
- Recap from B01: filter → Hilbert → magnitude
- Importance of filtering: envelope only meaningful for narrowband signals
- Choice of frequency band determines what oscillation we track
- Different bands = different envelopes = different dynamics

**Code to produce**:

Function 1: `extract_envelope`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    NDArray[np.float64]
Complete pipeline: bandpass filter, then Hilbert, then magnitude
Wrapper using functions from src/filtering.py and src/hilbert.py
```

**Visualization 4**:
- Raw EEG-like signal (broadband)
- Extract envelope for multiple bands:
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
- Four subplots (raw + 3 envelopes)
- Show that different bands have different envelope dynamics

---

## Section 5: Temporal Dynamics of the Envelope

**Purpose**: Analyze how envelopes evolve over time.

**Content to cover**:
- Envelopes fluctuate on slower timescales than the oscillation
- A 10 Hz oscillation might have envelope fluctuations at 0.5-2 Hz
- These slow fluctuations reflect neural modulation processes
- Envelope dynamics can reveal cognitive states, attention, arousal
- Important for understanding when oscillations are "active"

**Visualization 5**:
- Alpha-band filtered signal and its envelope
- Also show the envelope's own "frequency content"
- Two rows:
  - Row 1: signal and envelope (time domain)
  - Row 2: PSD of envelope (showing it has low-frequency content)
- Demonstrate that envelope is a slow signal

**Code to produce**:

Function 2: `compute_envelope_psd`
```
Parameters:
    envelope: NDArray[np.float64]
    fs: float
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, psd)
Compute PSD of the envelope signal
Useful for analyzing envelope dynamics
```

---

## Section 6: Envelope Smoothing

**Purpose**: Techniques to reduce noise in envelope estimates.

**Content to cover**:
- Raw envelope can be noisy, especially with short data or low SNR
- Smoothing reduces rapid fluctuations
- Methods:
  - Moving average (simple but distorts edges)
  - Gaussian smoothing (better edge behavior)
  - Low-pass filtering the envelope
- Trade-off: more smoothing = less temporal resolution
- Choice depends on analysis goals

**Code to produce**:

Function 3: `smooth_envelope_moving_average`
```
Parameters:
    envelope: NDArray[np.float64]
    window_samples: int
Returns:
    NDArray[np.float64]
Simple moving average smoothing
Handle edges with 'same' mode
```

Function 4: `smooth_envelope_gaussian`
```
Parameters:
    envelope: NDArray[np.float64]
    sigma_samples: float
Returns:
    NDArray[np.float64]
Gaussian smoothing using scipy.ndimage.gaussian_filter1d
```

Function 5: `smooth_envelope_lowpass`
```
Parameters:
    envelope: NDArray[np.float64]
    fs: float
    cutoff: float = 2.0
Returns:
    NDArray[np.float64]
Low-pass filter the envelope (remove fast fluctuations)
```

**Visualization 6**:
- Noisy envelope (from low-SNR signal)
- Compare smoothing methods:
  - Raw envelope
  - Moving average (window = 50 ms)
  - Gaussian (sigma = 20 ms)
  - Low-pass at 2 Hz
- Four traces overlaid or in subplots
- Discuss trade-offs visible in the plot

---

## Section 7: Envelope Correlation — Preview of Connectivity

**Purpose**: Introduce envelope correlation as amplitude-based connectivity.

**Content to cover**:
- Two signals can have correlated amplitude fluctuations
- Even if phases are unrelated, envelopes may co-vary
- Envelope correlation: Pearson correlation of two envelopes
- Interpretation: when signal 1 is "strong", is signal 2 also "strong"?
- Different from phase synchronization: amplitude coupling
- Used in CCorr connectivity metric

**Code to produce**:

Function 6: `compute_envelope_correlation`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    float (Pearson correlation coefficient)
Extract envelopes from both signals, compute correlation
```

**Visualization 7**:
- Two signals with correlated envelopes (amplitude co-fluctuation)
- Four subplots:
  - Signal 1 (alpha filtered) with envelope
  - Signal 2 (alpha filtered) with envelope
  - Both envelopes overlaid (show similarity)
  - Scatter plot: envelope 1 vs envelope 2 (show correlation)
- Annotate correlation coefficient

**Visualization 8**:
- Two signals with uncorrelated envelopes
- Same four subplots
- Scatter plot shows no relationship
- Correlation ≈ 0

---

## Section 8: The Volume Conduction Problem for Amplitude

**Purpose**: Discuss spurious amplitude correlations.

**Content to cover**:
- Volume conduction affects amplitude metrics too
- If two channels pick up the same source, their envelopes will be correlated
- This is spurious correlation, not true connectivity
- Solutions:
  - Orthogonalization: remove shared signal component
  - Source localization: analyze in source space
  - Laplacian reference: spatial filtering
- Brief mention here; detailed treatment in C01

**Visualization 9**:
- Simulated volume conduction scenario
- One source signal, two channels with different mixing weights
- Show that envelopes are highly correlated (spuriously)
- This motivates orthogonalization (covered in H01)

---

## Section 9: Envelope in Different Frequency Bands

**Purpose**: Compare envelope dynamics across bands.

**Content to cover**:
- Different frequency bands have different typical envelope dynamics
- Alpha: often shows clear bursts (8-13 Hz, envelope changes over ~0.5-2 s)
- Beta: faster fluctuations (13-30 Hz, envelope changes over ~0.2-0.5 s)
- Theta: slower, sustained fluctuations
- Gamma: very fast carrier, envelope can change rapidly
- Cross-frequency coupling: envelope of one band may correlate with phase/amplitude of another

**Visualization 10**:
- Same EEG-like signal
- Extract envelopes for all standard bands
- Five subplots (one per band)
- Normalize envelopes for visual comparison
- Show different temporal dynamics across bands

**Visualization 11**:
- Correlation matrix of envelopes across bands
- Compute envelope for each band, correlate all pairs
- Heatmap showing which bands have correlated amplitude fluctuations
- Discuss what correlations might mean

---

## Section 10: Envelope Statistics

**Purpose**: Quantify envelope properties.

**Content to cover**:
- Mean envelope: average amplitude over time
- Envelope variance: variability of amplitude
- Coefficient of variation: std/mean — normalized variability
- Percentiles: characterize envelope distribution
- These statistics describe oscillation "behavior" over a recording

**Code to produce**:

Function 7: `compute_envelope_statistics`
```
Parameters:
    envelope: NDArray[np.float64]
Returns:
    dict[str, float]
    {"mean": ..., "std": ..., "cv": ..., "median": ..., "p25": ..., "p75": ...}
Comprehensive envelope statistics
```

**Visualization 12**:
- Envelope time series
- Overlay horizontal lines for mean, median
- Shade region between 25th and 75th percentile
- Histogram of envelope values on the side (rotated)
- Rich visualization of envelope distribution

---

## Section 11: Practical Application — Alpha Blocking

**Purpose**: Classic EEG phenomenon demonstrating envelope utility.

**Content to cover**:
- Alpha blocking: alpha amplitude decreases when eyes open
- Classic demonstration of envelope analysis
- Simulate: eyes closed (high alpha) → eyes open (low alpha)
- Quantify the change using envelope statistics

**Code to produce**:
- Generate synthetic data simulating alpha blocking:
  - First 5 seconds: strong alpha
  - Last 5 seconds: weak alpha

**Visualization 13**:
- Time series showing the full recording
- Alpha envelope clearly showing the decrease
- Mark the transition point
- Bar chart comparing mean envelope before vs after

---

## Section 12: Hyperscanning Application — Amplitude Coupling

**Purpose**: Apply envelope analysis to two-brain scenario.

**Content to cover**:
- In hyperscanning, we can analyze envelope correlation between participants
- Question: do the two brains show co-fluctuation of oscillatory amplitude?
- Example: during joint attention, do both participants' alpha envelopes co-vary?
- This is a different kind of coupling than phase synchronization
- May reflect shared arousal, attention, or cognitive state

**Visualization 14**:
- Simulated hyperscanning with amplitude coupling
- Two participants with:
  - Independent phases (no phase synchronization)
  - Correlated envelopes (amplitude coupling)
- Show:
  - Both alpha-filtered signals
  - Both envelopes overlaid
  - Envelope correlation value
  - Note: PLV would be low, but envelope correlation is high

**Visualization 15**:
- Comparison of three scenarios:
  - Phase coupling only (high PLV, low envelope correlation)
  - Amplitude coupling only (low PLV, high envelope correlation)
  - Both (high PLV, high envelope correlation)
- 3×3 grid or summary figure
- Illustrate that these are independent connectivity mechanisms

---

## Section 13: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Create an amplitude-modulated signal (known modulation pattern)
- Extract envelope
- Verify envelope matches the modulation function
- Compute correlation between extracted envelope and true modulation

**Exercise 2**:
- Generate EEG-like signal with multiple bands
- Extract envelopes for theta, alpha, and beta
- Compute cross-correlation between band envelopes
- Identify which bands have most correlated amplitude dynamics

**Exercise 3**:
- Smoothing comparison
- Use a noisy synthetic signal
- Apply three smoothing methods with different parameters
- Measure: noise reduction vs temporal smearing
- Which settings preserve transients while reducing noise?

**Exercise 4**:
- Envelope correlation exercise
- Create two signals with known envelope correlation (shared modulation)
- Add independent noise to each
- Measure how envelope correlation degrades with noise level
- Plot: SNR vs envelope correlation

**Exercise 5**:
- Simulate alpha blocking experiment
- Generate data with transition from high to low alpha
- Compute envelope and detect the transition time automatically
- Use envelope threshold crossing or change-point detection

---

## Section 14: Functions for src/envelope.py

**Purpose**: Identify functions for new envelope module.

**Functions to add to src/envelope.py (new file)**:
- `extract_envelope`
- `compute_envelope_psd`
- `smooth_envelope_moving_average`
- `smooth_envelope_gaussian`
- `smooth_envelope_lowpass`
- `compute_envelope_correlation`
- `compute_envelope_statistics`

---

## Summary

**Key takeaways to list**:
- Amplitude envelope is the instantaneous magnitude of the analytic signal
- Envelope represents slow fluctuations of oscillatory power
- Power = envelope squared; both are used in connectivity analysis
- Smoothing reduces noise but trades off temporal resolution
- Envelope correlation measures amplitude coupling between signals
- Different from phase synchronization: independent connectivity mechanism
- Envelope dynamics differ across frequency bands
- Volume conduction also affects amplitude metrics (orthogonalization needed)
- In hyperscanning, envelope correlation reveals shared amplitude dynamics

---

## Discussion Questions

1. A study reports high envelope correlation between two participants' alpha activity during a cooperative task. What might this suggest about their cognitive states? What alternative explanations should be considered?

2. You analyze the same EEG data using PLV and envelope correlation. PLV is low but envelope correlation is high. What does this pattern suggest about the relationship between the two signals?

3. Why might smoothing the envelope be important before computing envelope correlation between two signals? What are the risks of over-smoothing?

4. In hyperscanning, participant A shows high alpha amplitude while participant B shows low alpha amplitude, but their envelopes are highly correlated (both go up and down together, just at different levels). How would you interpret this?

5. Cross-frequency coupling refers to relationships between different frequency bands. How might envelope analysis be used to study cross-frequency amplitude-amplitude coupling?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Single plot (overlay) | AM signal with upper/lower envelope | signal_1, signal_2 |
| 2 | Single plot (overlay) | EEG alpha with envelope | alpha color, signal_2 |
| 3 | 3 subplots (column) | Signal vs envelope vs power | signal_1, signal_2, signal_3 |
| 4 | 4 subplots (column) | Raw + envelopes for theta/alpha/beta | band colors |
| 5 | 2×1 grid | Envelope and its PSD | signal_1, signal_2 |
| 6 | 4 subplots or overlay | Smoothing methods comparison | signal_1-4 |
| 7 | 4 subplots | Correlated envelopes + scatter | signal_1, signal_2 |
| 8 | 4 subplots | Uncorrelated envelopes + scatter | signal_1, signal_2 |
| 9 | 3 subplots | Volume conduction effect on envelope | signal_1, signal_2 |
| 10 | 5 subplots | Envelope dynamics across bands | band colors |
| 11 | Heatmap | Cross-band envelope correlation matrix | gradient |
| 12 | Complex plot | Envelope statistics visualization | signal_1, reference |
| 13 | 3 subplots | Alpha blocking demonstration | alpha color |
| 14 | 4 subplots | Hyperscanning amplitude coupling | subject_1, subject_2 |
| 15 | 3×2 grid | Phase vs amplitude coupling scenarios | subject_1, subject_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses functions from A04 (filtering) and B01 (Hilbert)
- [ ] Smoothing functions handle edge effects appropriately
- [ ] Statistics function returns well-documented dict structure