# B02: Working with Phase — Copilot Directive

## Metadata

- **Notebook ID**: B02
- **Title**: Working with Phase
- **Branch**: `feature/B02_working_with_phase`
- **Duration**: 60 minutes
- **Prerequisites**: B01 (The Hilbert Transform)
- **Next**: B03 (Amplitude Envelope)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand phase as a circular (angular) variable
2. Compute and interpret phase differences between two signals
3. Handle phase wrapping and unwrapping correctly
4. Apply circular statistics (mean, variance, concentration)
5. Visualize phase distributions using polar plots
6. Recognize phase synchronization patterns

---

## Required Imports
```
numpy, matplotlib, scipy.stats, scipy.signal, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/filtering.py, src/hilbert.py
```

---

## Section 1: Introduction

**Purpose**: Motivate the importance of phase for connectivity analysis.

**Content to cover**:
- Phase synchronization is a primary mechanism for neural communication
- "Communication through coherence" hypothesis: neurons communicate via phase alignment
- Connectivity metrics PLV, PLI, wPLI are all based on phase relationships
- Phase is fundamentally different from ordinary numbers — it is circular
- Standard statistics (mean, variance) do not apply directly
- This notebook builds the tools for proper phase analysis

**Length**: 3-4 paragraphs

---

## Section 2: Phase as a Circular Variable

**Purpose**: Build intuition about the circular nature of phase.

**Content to cover**:
- Phase is an angle, typically in radians: [-π, π] or [0, 2π]
- Phase "wraps around": π + 0.1 ≈ -π + 0.1 (they are close!)
- Analogy: clock times — 11:55 and 00:05 are 10 minutes apart, not 11h50m
- Linear arithmetic fails: mean of -π and π is 0, but they are the same phase!
- Need circular (directional) statistics
- Visualization: phase as a point on the unit circle

**Visualization 1**:
- Unit circle diagram
- Show several phase values as points on the circle
- Illustrate that -π and π are the same point
- Show that phases near π and -π are actually close

**Visualization 2**:
- The "averaging problem"
- Two phase values: -0.9π and +0.9π
- Linear mean: 0 (completely wrong)
- Circular mean: ≈ π (correct — they are both near π)
- Illustrate on unit circle with vectors

---

## Section 3: Phase Wrapping and Unwrapping

**Purpose**: Handle the discontinuity in phase representation.

**Content to cover**:
- Phase wrapping: phase jumps from π to -π (or vice versa) as it crosses the boundary
- This is an artifact of representation, not a real discontinuity
- Unwrapping: removes artificial jumps to create continuous phase
- Useful for: computing instantaneous frequency, visualizing phase evolution
- np.unwrap() adds ±2π to remove jumps
- Caution: unwrapping can fail with noisy or undersampled signals

**Code to produce**:

Function 1: `wrap_phase`
```
Parameters:
    phase: NDArray[np.float64]
Returns:
    NDArray[np.float64] (wrapped to [-π, π])
Use np.angle(np.exp(1j * phase)) or modular arithmetic
```

Function 2: `unwrap_phase`
```
Parameters:
    phase: NDArray[np.float64]
Returns:
    NDArray[np.float64] (unwrapped, continuous)
Wrapper around np.unwrap with clear docstring
```

**Visualization 3**:
- Long sine wave (several seconds, many cycles)
- Two subplots:
  - Wrapped phase: sawtooth pattern with jumps
  - Unwrapped phase: continuous linear increase
- Annotate the jumps and how unwrapping resolves them

**Visualization 4**:
- Unwrapping failure case
- Very noisy narrowband signal
- Show that unwrapping can accumulate errors
- Indicate when unwrapping is reliable vs unreliable

---

## Section 4: Phase Difference Between Two Signals

**Purpose**: Compute and interpret phase relationships.

**Content to cover**:
- Phase difference: Δφ = φ₁ - φ₂
- Result should be wrapped to [-π, π]
- Interpretation:
  - Δφ ≈ 0: signals are in-phase (synchronized)
  - Δφ ≈ π: signals are anti-phase
  - Δφ ≈ π/2: signals are in quadrature
- Constant phase difference = phase-locked signals
- Varying phase difference = not synchronized

**Code to produce**:

Function 3: `compute_phase_difference`
```
Parameters:
    phase1: NDArray[np.float64]
    phase2: NDArray[np.float64]
Returns:
    NDArray[np.float64] (wrapped to [-π, π])
Compute difference and wrap result
```

**Visualization 5**:
- Two synchronized signals (same frequency, constant phase offset)
- Three subplots:
  - Both signals overlaid
  - Both phases overlaid
  - Phase difference over time (should be constant)

**Visualization 6**:
- Two signals with drifting phase relationship
- Same three subplots
- Phase difference should vary over time
- This represents non-synchronized signals

---

## Section 5: Circular Mean

**Purpose**: Compute the average direction of phase values.

**Content to cover**:
- Cannot simply average angles: mean([-π, π]) = 0 is wrong
- Circular mean algorithm:
  1. Convert each phase to unit vector: (cos φ, sin φ)
  2. Average the x and y components separately
  3. Convert back to angle: atan2(mean_y, mean_x)
- Result is the "preferred direction" of the phase distribution
- The magnitude of the resultant vector indicates concentration

**Code to produce**:

Function 4: `circular_mean`
```
Parameters:
    phases: NDArray[np.float64]
Returns:
    float (mean phase in radians)
Implement using cos/sin averaging
```

**Visualization 7**:
- Set of phase values clustered around a direction
- Unit circle plot:
  - Individual phases as points on circle
  - Mean vector from origin
  - Circular mean as the angle of mean vector
- Show for both concentrated and dispersed distributions

---

## Section 6: Circular Variance and Concentration

**Purpose**: Quantify the spread of phase distributions.

**Content to cover**:
- Circular variance: measure of phase dispersion
- Resultant vector length R: mean of unit vectors' magnitude
  - R = |mean(e^(iφ))|
  - R = 1: all phases identical (perfect concentration)
  - R = 0: phases uniformly distributed (no preferred direction)
- Circular variance: V = 1 - R
- Circular standard deviation: σ = √(-2 ln R)
- R is essentially what PLV measures!

**Code to produce**:

Function 5: `resultant_vector_length`
```
Parameters:
    phases: NDArray[np.float64]
Returns:
    float (R, range 0 to 1)
Compute |mean(exp(1j * phases))|
```

Function 6: `circular_variance`
```
Parameters:
    phases: NDArray[np.float64]
Returns:
    float (V = 1 - R, range 0 to 1)
```

Function 7: `circular_std`
```
Parameters:
    phases: NDArray[np.float64]
Returns:
    float (circular standard deviation)
√(-2 ln R), handle R=0 case
```

**Visualization 8**:
- Three phase distributions with different concentrations:
  - Highly concentrated (R ≈ 0.95)
  - Moderately concentrated (R ≈ 0.5)
  - Uniform/dispersed (R ≈ 0.1)
- Three polar plots side by side
- Annotate R value for each

---

## Section 7: Visualizing Phase Distributions

**Purpose**: Techniques for displaying phase data.

**Content to cover**:
- Phase histogram (on linear axis): can be misleading at boundaries
- Polar histogram: natural representation for circular data
- Rose plot: histogram in polar coordinates
- Phase scatter on unit circle
- Color-coded time series: phase as color

**Code to produce**:

Function 8: `plot_phase_polar_histogram`
```
Parameters:
    phases: NDArray[np.float64]
    n_bins: int = 36
    ax: Optional[plt.Axes] = None
Returns:
    plt.Axes
Polar histogram / rose plot
```

Function 9: `plot_phase_on_circle`
```
Parameters:
    phases: NDArray[np.float64]
    ax: Optional[plt.Axes] = None
    show_mean: bool = True
Returns:
    plt.Axes
Scatter plot on unit circle with optional mean vector
```

**Visualization 9**:
- Same phase distribution shown four ways:
  - Linear histogram (problematic at edges)
  - Polar histogram (rose plot)
  - Scatter on unit circle
  - Mean vector overlaid
- 2×2 grid comparing representations

---

## Section 8: Phase Difference Distribution

**Purpose**: Analyze the distribution of phase differences over time.

**Content to cover**:
- For two signals, compute phase difference at each time point
- The distribution of phase differences reveals synchronization
- Concentrated distribution → signals are phase-locked
- Uniform distribution → signals are independent
- This is the foundation of PLV: R of the phase difference distribution

**Visualization 10**:
- Two synchronized signals (same frequency, constant offset)
- Compute phase difference time series
- Show polar histogram: concentrated around one value
- Compute R (this is PLV!)

**Visualization 11**:
- Two independent signals (same frequency, random phase drift)
- Compute phase difference time series
- Show polar histogram: uniform distribution
- R ≈ 0 (no synchronization)

**Visualization 12**:
- Intermediate case: partial synchronization
- Signals are synchronized for part of the recording, not for other parts
- Show time-varying phase difference
- Show histogram: somewhat concentrated but not perfectly
- R is intermediate

---

## Section 9: Connection to Connectivity Metrics

**Purpose**: Bridge to the connectivity metrics we will study.

**Content to cover**:
- PLV (Phase Locking Value) = resultant vector length of phase differences
  - PLV = |mean(e^(i·Δφ))|
  - Exactly what we computed with `resultant_vector_length` on phase differences!
- Preview of PLI: considers only the sign of phase difference
- Preview of wPLI: weights by magnitude of imaginary component
- All these metrics are built on the tools from this notebook

**Code to produce**:

Function 10: `compute_plv_simple`
```
Parameters:
    phase1: NDArray[np.float64]
    phase2: NDArray[np.float64]
Returns:
    float (PLV value, 0 to 1)
Simple PLV implementation using circular statistics
Note: this is a preview, full implementation in G01
```

**Visualization 13**:
- Show the relationship between phase difference distribution and PLV
- Three examples:
  - High PLV (concentrated distribution)
  - Medium PLV
  - Low PLV (uniform distribution)
- Side by side: distribution and corresponding PLV value

---

## Section 10: Practical Considerations

**Purpose**: Address practical issues in phase analysis.

**Content to cover**:
- Sampling rate: need enough samples per cycle for accurate phase
- Filter bandwidth: too narrow = poor temporal resolution, too wide = phase less meaningful
- Noise: corrupts phase estimates, especially at low amplitudes
- Edge effects: phase estimates unreliable at signal boundaries
- Amplitude threshold: ignore phase when amplitude is very low (noisy phase)

**Code to produce**:

Function 11: `mask_low_amplitude_phase`
```
Parameters:
    phase: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    threshold_percentile: float = 10.0
Returns:
    NDArray[np.float64] (phase with NaN where amplitude is low)
Mask out unreliable phase estimates
```

**Visualization 14**:
- Signal with varying amplitude (some very low sections)
- Two subplots:
  - Phase (raw): noisy during low-amplitude periods
  - Phase (masked): NaN where amplitude below threshold
- Illustrate why masking is important

---

## Section 11: Phase in Hyperscanning Context

**Purpose**: Apply phase concepts to two-brain analysis.

**Content to cover**:
- In hyperscanning, we compare phases between two participants
- Each participant's signal: filter → Hilbert → phase
- Phase difference between participants at each time point
- Distribution of inter-brain phase differences
- Interpretation: are the two brains phase-synchronized?

**Visualization 15**:
- Simulated hyperscanning scenario
- Two "brains" with:
  - Alpha oscillations
  - Some periods of synchronization
  - Some periods of independence
- Show:
  - Both signals (alpha-filtered)
  - Both phases
  - Phase difference over time
  - Polar histogram of phase differences
  - Computed PLV

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Generate 100 random phases uniformly distributed on [-π, π]
- Compute linear mean (should be ≈ 0)
- Compute circular mean
- Compute R (should be ≈ 0 for uniform)
- Visualize on unit circle

**Exercise 2**:
- Generate phases clustered around π/4 with some spread (von Mises distribution)
- Compute circular mean (should be ≈ π/4)
- Compute R (should be high)
- Vary the concentration parameter and observe effect on R

**Exercise 3**:
- Create two 10 Hz signals with known constant phase difference (e.g., π/3)
- Extract phases using Hilbert transform
- Compute phase difference
- Verify it equals π/3 throughout
- Compute PLV (should be ≈ 1)

**Exercise 4**:
- Create two 10 Hz signals where phase difference slowly drifts
- Compute phase difference over time
- Plot the polar histogram
- Compute PLV (should be < 1)
- Experiment: how fast must phase drift before PLV drops significantly?

**Exercise 5**:
- Hyperscanning simulation
- Generate two "EEG" signals with alpha oscillations
- First half: signals synchronized (constant phase difference)
- Second half: signals independent (random phase relationship)
- Compute PLV for each half separately
- Visualize and interpret the results

---

## Section 13: Functions for src/phase.py

**Purpose**: Identify functions for new phase module.

**Functions to add to src/phase.py (new file)**:
- `wrap_phase`
- `unwrap_phase`
- `compute_phase_difference`
- `circular_mean`
- `resultant_vector_length`
- `circular_variance`
- `circular_std`
- `plot_phase_polar_histogram`
- `plot_phase_on_circle`
- `mask_low_amplitude_phase`
- `compute_plv_simple`

---

## Summary

**Key takeaways to list**:
- Phase is a circular variable; standard statistics do not apply
- Phase wrapping is a representation artifact; unwrapping creates continuity
- Phase difference Δφ = φ₁ - φ₂ reveals synchronization between signals
- Circular mean uses vector averaging: mean angle of (cos φ, sin φ)
- Resultant vector length R measures phase concentration (0 = uniform, 1 = identical)
- R of phase differences is exactly PLV (Phase Locking Value)
- Polar histograms and unit circle plots are natural phase visualizations
- Mask phase estimates when amplitude is low (unreliable)
- In hyperscanning, inter-brain phase synchronization indicates neural coupling

---

## Discussion Questions

1. Two EEG channels show a constant phase difference of 0. Does this necessarily mean they are measuring the same neural source? What else could explain this?

2. You compute PLV between two participants and get a value of 0.3. Is this "good" synchronization? What additional information would help you interpret this value?

3. Why is it important to band-pass filter signals before computing phase differences? What would happen if you compared phases from broadband signals?

4. In a hyperscanning study, you find high inter-brain PLV in the alpha band during eye contact, but low PLV during independent tasks. What might this suggest about neural mechanisms of social interaction?

5. The circular mean of a set of phases is undefined when R = 0 (uniform distribution). Why does this make sense intuitively?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Unit circle diagram | Phase as point on circle | signal_1, reference |
| 2 | Unit circle + vectors | Circular mean problem | signal_1, signal_2 |
| 3 | 2 subplots (column) | Wrapped vs unwrapped phase | phase_positive |
| 4 | 2 subplots | Unwrapping failure case | phase_positive, reference |
| 5 | 3 subplots | Synchronized signals phase difference | signal_1, signal_2 |
| 6 | 3 subplots | Non-synchronized signals | signal_1, signal_2 |
| 7 | Unit circle | Circular mean visualization | signal_1, highlight |
| 8 | 3 polar plots | Different concentration levels | signal_1 |
| 9 | 2×2 grid | Phase visualization methods | signal_1 |
| 10 | 2 subplots | Synchronized: polar histogram + R | signal_1, high_sync |
| 11 | 2 subplots | Independent: polar histogram + R | signal_1, low_sync |
| 12 | 3 subplots | Partial synchronization | signal_1, signal_2 |
| 13 | 3×2 grid | PLV and distribution relationship | gradient |
| 14 | 2 subplots | Amplitude masking of phase | signal_1, reference |
| 15 | 5 subplots | Hyperscanning phase analysis | subject_1, subject_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses functions from B01 (Hilbert) where appropriate
- [ ] Circular statistics implementations numerically stable
- [ ] Edge cases handled (R=0, empty arrays)
- [ ] Polar plot functions use matplotlib projection='polar'