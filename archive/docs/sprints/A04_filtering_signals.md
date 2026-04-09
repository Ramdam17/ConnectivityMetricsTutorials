# A04: Filtering Signals — Copilot Directive

## Metadata

- **Notebook ID**: A04
- **Title**: Filtering Signals
- **Branch**: `feature/A04_filtering_signals`
- **Duration**: 60 minutes
- **Prerequisites**: A02 (The Frequency Domain)
- **Next**: B01 (The Hilbert Transform)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain why filtering is essential for EEG analysis
2. Distinguish between filter types (low-pass, high-pass, band-pass, band-stop)
3. Understand key filter parameters (order, cutoff frequency, transition bandwidth)
4. Compare FIR and IIR filters and their trade-offs
5. Apply zero-phase filtering to avoid temporal distortion
6. Design and apply filters using scipy and MNE

---

## Required Imports
```
numpy, matplotlib, scipy.signal, mne, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/spectral.py
```

---

## Section 1: Introduction

**Purpose**: Motivate why filtering is a fundamental preprocessing step.

**Content to cover**:
- Raw EEG contains many sources: brain activity, muscle artifacts, line noise, drift
- We often want to isolate specific frequency ranges (e.g., alpha band only)
- Filtering removes unwanted frequency components while preserving others
- Critical for connectivity analysis: must isolate band of interest before computing PLV, coherence, etc.
- Caution: improper filtering can introduce artifacts or distort results

**Length**: 3-4 paragraphs

---

## Section 2: Filter Types

**Purpose**: Introduce the four basic filter types.

**Content to cover**:
- **Low-pass filter**: passes frequencies below cutoff, attenuates above
  - Use case: remove high-frequency noise, muscle artifacts
- **High-pass filter**: passes frequencies above cutoff, attenuates below
  - Use case: remove slow drifts, baseline wander
- **Band-pass filter**: passes frequencies within a range, attenuates outside
  - Use case: isolate specific frequency band (e.g., alpha 8-13 Hz)
- **Band-stop (notch) filter**: attenuates frequencies within a range, passes outside
  - Use case: remove line noise (50 Hz or 60 Hz)

**Visualization 1**:
- 2×2 grid showing idealized frequency response of each filter type
- X-axis: frequency, Y-axis: gain (0 to 1)
- Shade the passband region
- Clear labels for each filter type

---

## Section 3: Filter Characteristics

**Purpose**: Understand the key parameters that define a filter.

**Content to cover**:
- **Cutoff frequency (fc)**: the frequency where attenuation begins (typically -3 dB point)
- **Transition bandwidth**: how quickly filter transitions from pass to stop (sharp vs gradual)
- **Passband ripple**: small variations in gain within passband
- **Stopband attenuation**: how much unwanted frequencies are reduced (in dB)
- **Filter order**: higher order = sharper transition but more computation and potential artifacts

**Visualization 2**:
- Single filter frequency response with annotations:
  - Passband region
  - Stopband region
  - Transition band
  - Cutoff frequency (-3 dB point)
  - Stopband attenuation level
- Educational diagram style

**Visualization 3**:
- Effect of filter order on frequency response
- Same cutoff frequency, varying orders (e.g., 2, 4, 8, 16)
- Show how higher order = sharper cutoff
- Overlay on same plot with different line styles

---

## Section 4: FIR vs IIR Filters

**Purpose**: Compare the two main filter design approaches.

**Content to cover**:
- **FIR (Finite Impulse Response)**:
  - Output depends only on current and past inputs
  - Always stable
  - Linear phase (no phase distortion) when symmetric
  - Requires higher order for sharp cutoffs
  - Preferred for EEG analysis
- **IIR (Infinite Impulse Response)**:
  - Output depends on inputs AND past outputs (feedback)
  - Can be unstable if poorly designed
  - Non-linear phase (distorts signal shape)
  - More efficient: sharp cutoff with lower order
  - Common types: Butterworth, Chebyshev, Elliptic
- **Recommendation**: Use FIR for EEG, or IIR with zero-phase filtering

**Table**:
| Property | FIR | IIR |
|----------|-----|-----|
| Stability | Always stable | Can be unstable |
| Phase | Linear (symmetric) | Non-linear |
| Order for sharp cutoff | High | Low |
| Computational cost | Higher | Lower |
| EEG recommendation | Preferred | Use with zero-phase |

---

## Section 5: Designing Filters in Python

**Purpose**: Practical filter design using scipy.

**Content to cover**:
- scipy.signal for filter design
- FIR design: firwin, firwin2
- IIR design: butter, cheby1, cheby2, ellip
- Key parameters: order, cutoff (normalized or in Hz), filter type

**Code to produce**:

Function 1: `design_fir_filter`
```
Parameters:
    fs: float
    cutoff: float | tuple[float, float]
    filter_type: str ("lowpass", "highpass", "bandpass", "bandstop")
    order: int = 101
Returns:
    NDArray[np.float64] (filter coefficients)
Use scipy.signal.firwin
Cutoff in Hz, internally normalized
Order should be odd for type I FIR
```

Function 2: `design_iir_filter`
```
Parameters:
    fs: float
    cutoff: float | tuple[float, float]
    filter_type: str
    order: int = 4
    design: str = "butter"
Returns:
    tuple[NDArray[np.float64], NDArray[np.float64]] (b, a coefficients)
Use scipy.signal.butter (or iirfilter for other designs)
```

**Visualization 4**:
- Compare FIR and IIR filter with same cutoff (e.g., 30 Hz low-pass)
- Two subplots:
  - Frequency response (magnitude)
  - Phase response
- Show FIR has linear phase, IIR has non-linear phase

---

## Section 6: Applying Filters

**Purpose**: Apply designed filters to signals.

**Content to cover**:
- scipy.signal.lfilter: causal filtering (introduces phase delay)
- scipy.signal.filtfilt: zero-phase filtering (forward-backward)
- Why zero-phase matters: preserves temporal relationships (critical for connectivity!)
- filtfilt effectively doubles the filter order

**Code to produce**:

Function 3: `apply_filter`
```
Parameters:
    signal: NDArray[np.float64]
    b: NDArray[np.float64]
    a: NDArray[np.float64] = np.array([1.0])  # FIR default
    zero_phase: bool = True
Returns:
    NDArray[np.float64]
Use filtfilt if zero_phase, else lfilter
Handle edge effects with padding method
```

**Visualization 5**:
- Original signal with sharp transient (e.g., step or spike)
- Three versions:
  - Original
  - Causal filter (lfilter): shows delay/distortion
  - Zero-phase filter (filtfilt): preserves timing
- Highlight the temporal shift in causal filtering

---

## Section 7: Phase Distortion and Why It Matters

**Purpose**: Deep dive into phase distortion implications.

**Content to cover**:
- IIR filters have frequency-dependent delay (group delay)
- Different frequencies arrive at different times after filtering
- This distorts signal shape and timing
- For connectivity analysis: phase relationships are the measurement!
- Phase distortion would corrupt PLV, PLI, coherence phase
- Zero-phase filtering eliminates this problem

**Visualization 6**:
- Composite signal with two frequency components (e.g., 5 Hz and 20 Hz)
- Apply IIR low-pass (causal) that passes both
- Show how the relative phase between components changes
- Compare with zero-phase version

### [Optional Deep Dive] Group Delay

**Content**:
- Group delay definition: τ(f) = -dφ/df
- Constant group delay = linear phase = no distortion
- FIR symmetric filters have constant group delay
- IIR filters have frequency-dependent group delay
- scipy.signal.group_delay() for visualization

---

## Section 8: Common EEG Filtering Operations

**Purpose**: Practical examples relevant to EEG preprocessing.

**Content to cover**:
- High-pass at 0.1-1 Hz: remove slow drifts
- Low-pass at 40-100 Hz: remove muscle artifacts, prepare for analysis
- Band-pass for specific bands: e.g., 8-13 Hz for alpha
- Notch at 50/60 Hz: remove line noise

**Code to produce**:

Function 4: `highpass_filter`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    cutoff: float
    order: int = 4
Returns:
    NDArray[np.float64]
Convenience wrapper using IIR Butterworth + zero-phase
```

Function 5: `lowpass_filter`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    cutoff: float
    order: int = 4
Returns:
    NDArray[np.float64]
```

Function 6: `bandpass_filter`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    low_cutoff: float
    high_cutoff: float
    order: int = 4
Returns:
    NDArray[np.float64]
```

Function 7: `notch_filter`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    notch_freq: float
    quality_factor: float = 30.0
Returns:
    NDArray[np.float64]
Use scipy.signal.iirnotch
```

**Visualization 7**:
- Synthetic EEG with:
  - Slow drift (0.1 Hz)
  - Alpha oscillation (10 Hz)
  - Line noise (50 Hz)
  - High-frequency noise (>100 Hz)
- Four subplots showing progressive filtering:
  - Raw signal
  - After high-pass (drift removed)
  - After low-pass (HF noise removed)
  - After notch (line noise removed)
- PSD subplot alongside each to show spectral effect

---

## Section 9: Filtering for Band Extraction

**Purpose**: Demonstrate filtering to isolate frequency bands (prerequisite for Hilbert).

**Content to cover**:
- To analyze a specific band: band-pass filter first
- Band-pass creates a "narrowband" signal
- This is required before applying Hilbert transform (B01)
- Filter bandwidth affects temporal resolution (narrower band = smoother envelope)

**Visualization 8**:
- EEG-like signal with multiple frequency components
- Extract alpha band (8-13 Hz) using band-pass
- Three subplots:
  - Original signal
  - Alpha-filtered signal
  - PSD comparison (original vs filtered)
- Show only alpha frequencies remain

**Visualization 9**:
- Extract all standard EEG bands from same signal
- Five subplots (delta, theta, alpha, beta, gamma)
- Each shows filtered time series
- Use band colors from STYLE_GUIDE.md

---

## Section 10: Filter Edge Effects

**Purpose**: Understand and mitigate artifacts at signal boundaries.

**Content to cover**:
- Filters need "warm-up" time at signal start/end
- Edge effects: transient artifacts at boundaries
- Mitigation strategies:
  - Padding (reflect, constant, wrap)
  - Discard edge samples after filtering
  - MNE's approach: automatic padding
- Longer filters = longer edge effects

**Visualization 10**:
- Short signal filtered with different padding strategies
- Show edge artifacts with no padding
- Show improvement with reflection padding
- Indicate samples that should be discarded

---

## Section 11: Using MNE for Filtering

**Purpose**: Introduce MNE's filtering functions (standard in EEG research).

**Content to cover**:
- MNE provides well-tested filtering with sensible defaults
- mne.filter.filter_data() for raw arrays
- Automatic FIR design with optimal parameters
- Handles edge effects automatically
- Recommended for production EEG analysis

**Code to produce**:

Function 8: `mne_bandpass_filter`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    low_cutoff: float
    high_cutoff: float
Returns:
    NDArray[np.float64]
Wrapper around mne.filter.filter_data
```

**Visualization 11**:
- Compare our implementation vs MNE on same signal
- Two subplots:
  - Our bandpass_filter result
  - MNE filter result
- Show they produce similar results
- Discuss any differences (filter design choices)

---

## Section 12: Filter Design Considerations Summary

**Purpose**: Practical guidelines for EEG filtering.

**Content to cover (as a reference table/list)**:
- High-pass: 0.1-1 Hz, removes drift, use low order to avoid artifacts
- Low-pass: typically 40-100 Hz depending on analysis
- Band-pass: for band-specific analysis, width affects temporal resolution
- Notch: only if line noise present, narrow Q factor
- Always use zero-phase (filtfilt) for EEG
- Check filter response before applying
- Consider edge effects, discard or pad appropriately
- Document filter settings in methods section

---

## Section 13: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Design a band-pass filter for theta band (4-8 Hz)
- Apply to synthetic signal containing theta + alpha + beta
- Verify with PSD that only theta remains
- Compare FIR (order 101) vs IIR (order 4) results

**Exercise 2**:
- Create signal with 50 Hz line noise contamination
- Design and apply notch filter
- Visualize before/after in time and frequency domains
- Experiment with different Q factors (10, 30, 50)

**Exercise 3**:
- Investigate phase distortion
- Create two synchronized sine waves (same phase)
- Filter with causal IIR filter
- Measure phase difference after filtering
- Repeat with zero-phase filter and compare

**Exercise 4**:
- Filter order exploration
- Same band-pass (8-13 Hz), varying orders: 2, 4, 6, 8
- Plot frequency responses
- Apply to signal and compare edge effects
- What order would you choose and why?

---

## Section 14: Functions for src/filtering.py

**Purpose**: Identify functions for new filtering module.

**Functions to add to src/filtering.py (new file)**:
- `design_fir_filter`
- `design_iir_filter`
- `apply_filter`
- `highpass_filter`
- `lowpass_filter`
- `bandpass_filter`
- `notch_filter`
- `mne_bandpass_filter`

---

## Summary

**Key takeaways to list**:
- Filters selectively pass or attenuate frequency components
- Four types: low-pass, high-pass, band-pass, band-stop (notch)
- Key parameters: cutoff frequency, order, transition bandwidth
- FIR filters have linear phase; IIR filters are more efficient but have non-linear phase
- Always use zero-phase filtering (filtfilt) for EEG to preserve temporal relationships
- Phase distortion from causal filtering would corrupt connectivity measurements
- Band-pass filtering is required before Hilbert transform for envelope/phase extraction
- MNE provides well-tested filtering functions for production use
- Always check filter frequency response and watch for edge effects

---

## Discussion Questions

1. You want to study gamma oscillations (30-100 Hz) but your signal has strong muscle artifact in the same range. What are your options, and what are the trade-offs?

2. A colleague filtered their EEG at 1-40 Hz before computing phase-based connectivity. They used a causal Butterworth filter. What problem might this cause for their connectivity estimates?

3. Why might using a very narrow band-pass filter (e.g., 9-11 Hz) create problems for analyzing rapid changes in alpha activity?

4. In hyperscanning, two participants' data are recorded on different systems that might have slightly different hardware filters. How could this affect inter-brain connectivity analysis?

5. You notice edge artifacts in your filtered signal. What strategies could you use to mitigate this, and which would you recommend for a 5-minute EEG recording?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 2×2 grid | Four filter types (idealized response) | signal_1 (passband shaded) |
| 2 | Single plot (annotated) | Filter characteristics diagram | signal_1, reference gray |
| 3 | Single plot (overlay) | Filter order comparison | signal_1-4 varying alpha |
| 4 | 2 subplots | FIR vs IIR magnitude and phase | signal_1, signal_2 |
| 5 | 3 subplots (column) | Causal vs zero-phase filtering | signal_1, signal_2, signal_3 |
| 6 | 2 subplots | Phase distortion demonstration | signal_1, signal_2 |
| 7 | 4×2 grid | Progressive filtering (time + PSD) | signal_1 |
| 8 | 3 subplots | Alpha band extraction | signal_1, alpha color |
| 9 | 5 subplots | All EEG bands extracted | band colors |
| 10 | 3 subplots | Edge effects with different padding | signal_1 |
| 11 | 2 subplots | Our filter vs MNE comparison | signal_1, signal_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses functions from previous notebooks where appropriate
- [ ] Zero-phase filtering is the default
- [ ] Filter design validated by plotting frequency response