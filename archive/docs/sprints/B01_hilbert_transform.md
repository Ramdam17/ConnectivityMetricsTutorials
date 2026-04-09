# B01: The Hilbert Transform — Copilot Directive

## Metadata

- **Notebook ID**: B01
- **Title**: The Hilbert Transform
- **Branch**: `feature/B01_hilbert_transform`
- **Duration**: 60 minutes
- **Prerequisites**: A04 (Filtering Signals)
- **Next**: B02 (Working with Phase)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain the concept of the analytic signal
2. Understand what the Hilbert transform computes (90° phase shift)
3. Extract instantaneous amplitude (envelope) from a signal
4. Extract instantaneous phase from a signal
5. Recognize when the Hilbert transform is valid (narrowband requirement)
6. Apply the Hilbert transform using scipy

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.fft, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/spectral.py, src/filtering.py
```

---

## Section 1: Introduction

**Purpose**: Motivate why we need instantaneous amplitude and phase.

**Content to cover**:
- Many connectivity metrics require phase information (PLV, PLI, wPLI)
- Amplitude-based metrics require envelope extraction (CCorr, PowCorr)
- Problem: a raw signal has no single "phase" — it contains multiple frequencies
- Solution: filter to a narrow band, then extract instantaneous phase and amplitude
- The Hilbert transform is the mathematical tool that enables this extraction
- Foundation for all phase-based and many amplitude-based connectivity analyses

**Length**: 3-4 paragraphs

---

## Section 2: The Problem — What is "Phase" of a Complex Signal?

**Purpose**: Illustrate why we cannot directly measure phase of broadband signals.

**Content to cover**:
- A pure sine wave has well-defined phase at each time point
- A composite signal (sum of sines) has no single phase
- Example: 5 Hz + 10 Hz signal — which phase do we report?
- Need to isolate a single frequency band first
- Even then, how do we extract the phase? This is where Hilbert helps.

**Visualization 1**:
- Three subplots:
  - Pure 10 Hz sine wave (phase is clear)
  - Composite signal (5 Hz + 10 Hz + 20 Hz) — phase is ambiguous
  - Band-pass filtered version (10 Hz only) — phase becomes meaningful
- Annotate that middle signal has no single "phase"

---

## Section 3: The Analytic Signal Concept

**Purpose**: Introduce the analytic signal as the key construction.

**Content to cover**:
- Analytic signal: z(t) = x(t) + i·H{x(t)}
- x(t) is the original real signal
- H{x(t)} is the Hilbert transform of x(t)
- The analytic signal is complex-valued
- It encodes both amplitude and phase information
- Geometric interpretation: a rotating vector (phasor) in the complex plane

**Key equations**:
- Analytic signal: z(t) = x(t) + i·x̂(t) where x̂(t) = H{x(t)}
- Instantaneous amplitude: A(t) = |z(t)| = √(x² + x̂²)
- Instantaneous phase: φ(t) = arg(z(t)) = atan2(x̂, x)

**Visualization 2**:
- Phasor diagram concept
- Show a rotating vector in the complex plane
- Real part = original signal
- Imaginary part = Hilbert transform
- Magnitude = envelope, Angle = phase
- Animated concept (or multiple snapshots in time)

---

## Section 4: What the Hilbert Transform Does

**Purpose**: Explain the Hilbert transform operation.

**Content to cover**:
- The Hilbert transform shifts all frequency components by -90° (π/2)
- For a sine wave: H{sin(ωt)} = -cos(ωt)
- For a cosine: H{cos(ωt)} = sin(ωt)
- In frequency domain: multiply positive frequencies by -i, negative by +i
- The result is a signal "in quadrature" with the original
- Together, original + i·Hilbert form the analytic signal

**Visualization 3**:
- Pure sine wave and its Hilbert transform
- Two subplots:
  - Top: original sine wave
  - Bottom: Hilbert transform (90° shifted version)
- Show they are in quadrature (one is at peak when other crosses zero)

**Visualization 4**:
- Frequency domain view
- Show how Hilbert transform affects the spectrum
- Original: symmetric spectrum
- Analytic signal: only positive frequencies (doubled amplitude)

---

## Section 5: Computing the Hilbert Transform in Python

**Purpose**: Practical implementation using scipy.

**Content to cover**:
- scipy.signal.hilbert() computes the analytic signal (not just the Hilbert transform)
- Returns complex array: real part is original, imaginary part is Hilbert transform
- Common confusion: function returns analytic signal, not the transform itself

**Code to produce**:

Function 1: `compute_analytic_signal`
```
Parameters:
    signal: NDArray[np.float64]
Returns:
    NDArray[np.complex128]
Wrapper around scipy.signal.hilbert
Clear docstring explaining output
```

Function 2: `compute_hilbert_transform`
```
Parameters:
    signal: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Returns only the imaginary part (the actual Hilbert transform)
```

**Visualization 5**:
- Verify on simple sine wave
- Four subplots:
  - Original signal x(t)
  - Hilbert transform x̂(t)
  - Analytic signal real part (should equal original)
  - Analytic signal imaginary part (should equal Hilbert transform)

---

## Section 6: Extracting Instantaneous Amplitude (Envelope)

**Purpose**: Show how to extract the signal envelope.

**Content to cover**:
- Instantaneous amplitude = magnitude of analytic signal
- A(t) = |z(t)| = √(real² + imag²)
- Also called the "envelope" — it traces the peaks of the oscillation
- Represents the slowly-varying amplitude of the oscillation
- Used in envelope correlation (CCorr) connectivity metric

**Code to produce**:

Function 3: `compute_envelope`
```
Parameters:
    signal: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Compute magnitude of analytic signal
```

**Visualization 6**:
- Amplitude-modulated signal (e.g., 10 Hz carrier with 1 Hz modulation)
- Three elements on same plot:
  - Original signal (oscillating)
  - Envelope (upper bound, smooth curve)
  - Negative envelope (lower bound, for symmetry)
- Show how envelope captures the amplitude fluctuations

**Visualization 7**:
- EEG-like signal (alpha band filtered)
- Overlay envelope on the filtered signal
- Show how envelope tracks the "power" over time

---

## Section 7: Extracting Instantaneous Phase

**Purpose**: Show how to extract the signal phase.

**Content to cover**:
- Instantaneous phase = argument (angle) of analytic signal
- φ(t) = atan2(imag, real)
- Phase is in radians, range [-π, π] or [0, 2π]
- Phase increases linearly for a pure oscillation
- Phase "wraps" when it reaches ±π
- Used in PLV, PLI, wPLI connectivity metrics

**Code to produce**:

Function 4: `compute_instantaneous_phase`
```
Parameters:
    signal: NDArray[np.float64]
Returns:
    NDArray[np.float64] (phase in radians)
Compute angle of analytic signal using np.angle
```

**Visualization 8**:
- Pure sine wave and its instantaneous phase
- Two subplots:
  - Top: sine wave
  - Bottom: phase (sawtooth pattern from -π to π, wrapping)
- Show the linear increase and wrapping behavior

**Visualization 9**:
- Phase of amplitude-modulated signal
- Show that phase continues to increase linearly despite amplitude changes
- Demonstrate that amplitude and phase are independent

---

## Section 8: The Narrowband Requirement

**Purpose**: Critical understanding of when Hilbert transform is valid.

**Content to cover**:
- The Hilbert transform gives meaningful results ONLY for narrowband signals
- Narrowband: signal contains frequencies in a limited range
- For broadband signals, instantaneous frequency can become negative (physically meaningless)
- Rule of thumb: bandwidth should be less than center frequency
- **Critical**: always band-pass filter before applying Hilbert transform
- This is why A04 (filtering) is a prerequisite

**Visualization 10**:
- Broadband signal (no filtering)
- Compute envelope and phase
- Show that:
  - Envelope has strange artifacts
  - Phase jumps erratically
- Label as "INVALID — do not use without filtering"

**Visualization 11**:
- Same signal, now band-pass filtered
- Compute envelope and phase
- Show that:
  - Envelope is smooth and meaningful
  - Phase increases smoothly with occasional wraps
- Label as "VALID — narrowband signal"

---

## Section 9: Complete Workflow — From Raw Signal to Phase/Amplitude

**Purpose**: Demonstrate the full analysis pipeline.

**Content to cover**:
- Step 1: Start with raw (broadband) signal
- Step 2: Band-pass filter to frequency band of interest
- Step 3: Apply Hilbert transform to get analytic signal
- Step 4: Extract envelope and/or phase
- This workflow is the foundation for connectivity analysis

**Code to produce**:

Function 5: `extract_band_amplitude`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    NDArray[np.float64] (envelope)
Complete pipeline: filter then envelope
```

Function 6: `extract_band_phase`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    NDArray[np.float64] (phase in radians)
Complete pipeline: filter then phase
```

**Visualization 12**:
- Complete workflow figure (4-5 subplots in column):
  - Raw EEG-like signal
  - Band-pass filtered (alpha 8-13 Hz)
  - Envelope overlaid on filtered signal
  - Instantaneous phase
  - (Optional) Phase shown as color on signal

---

## Section 10: Instantaneous Frequency

**Purpose**: Briefly introduce instantaneous frequency as derivative of phase.

**Content to cover**:
- Instantaneous frequency: f(t) = (1/2π) · dφ/dt
- For a pure sine wave at f₀, instantaneous frequency = f₀ (constant)
- For narrowband signal, fluctuates around center frequency
- Can become negative for broadband signals (meaningless)
- Less commonly used in connectivity, but good to understand

**Code to produce**:

Function 7: `compute_instantaneous_frequency`
```
Parameters:
    phase: NDArray[np.float64]
    fs: float
Returns:
    NDArray[np.float64] (frequency in Hz)
Use np.diff on unwrapped phase, handle edge
```

**Visualization 13**:
- Narrowband signal with varying frequency (e.g., chirp within alpha band)
- Two subplots:
  - Signal
  - Instantaneous frequency over time
- Show how it tracks the changing frequency

---

## Section 11: Comparison with Alternative Methods

**Purpose**: Briefly mention other approaches for context.

**Content to cover**:
- Hilbert transform is not the only way to get envelope/phase
- Alternatives:
  - Wavelet transform: provides time-frequency decomposition
  - Short-time Fourier transform: similar but windowed
  - Peak detection: crude envelope estimation
- Hilbert is preferred for single-band analysis due to simplicity
- Wavelets preferred when analyzing multiple bands simultaneously
- For connectivity metrics, Hilbert after band-pass is standard

---

## Section 12: Edge Effects and Practical Considerations

**Purpose**: Address practical issues.

**Content to cover**:
- Edge effects: Hilbert transform has transients at signal boundaries
- Caused by: filtering and Hilbert transform both have edge issues
- Mitigation:
  - Pad signal before processing, trim after
  - Use longer recordings, discard edges
  - MNE handles this automatically
- Phase unwrapping: sometimes needed for continuous phase analysis
- Numerical precision: very small amplitudes can cause noisy phase estimates

**Visualization 14**:
- Show edge effects in envelope and phase
- Indicate regions that should be discarded
- Show effect of padding

---

## Section 13: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Create a 10 Hz sine wave with time-varying amplitude (amplitude modulation)
- Extract envelope using Hilbert transform
- Verify that envelope matches the modulation pattern
- Plot original, envelope, and known modulation function

**Exercise 2**:
- Create a signal with two components: 8 Hz and 12 Hz
- Apply Hilbert directly (no filtering) — observe invalid results
- Band-pass filter to 7-9 Hz, then apply Hilbert — observe valid envelope
- Band-pass filter to 11-13 Hz, then apply Hilbert — observe different envelope
- Conclude: filtering determines which component we analyze

**Exercise 3**:
- Phase analysis exercise
- Create two sine waves: same frequency (10 Hz), different phases (0 and π/4)
- Extract phase of each using Hilbert
- Compute phase difference
- Verify it equals π/4 throughout the signal

**Exercise 4**:
- Real workflow exercise
- Generate EEG-like signal with multiple components + noise
- Extract alpha band envelope and phase
- Plot the results
- Identify time points where alpha is strongest (highest envelope)

---

## Section 14: Functions for src/hilbert.py

**Purpose**: Identify functions for new Hilbert module.

**Functions to add to src/hilbert.py (new file)**:
- `compute_analytic_signal`
- `compute_hilbert_transform`
- `compute_envelope`
- `compute_instantaneous_phase`
- `compute_instantaneous_frequency`
- `extract_band_amplitude`
- `extract_band_phase`

---

## Summary

**Key takeaways to list**:
- The analytic signal z(t) = x(t) + i·H{x(t)} encodes amplitude and phase
- Hilbert transform shifts all frequencies by 90°
- Instantaneous amplitude (envelope) = |z(t)|
- Instantaneous phase = arg(z(t))
- **Critical**: Hilbert transform only valid for narrowband signals
- Always band-pass filter before applying Hilbert transform
- Standard workflow: filter → Hilbert → extract envelope/phase
- This is the foundation for PLV, PLI, wPLI, and envelope correlation

---

## Discussion Questions

1. Why must we band-pass filter before applying the Hilbert transform? What would happen if we applied it to raw, broadband EEG?

2. You want to analyze phase synchronization in the alpha band between two people. Describe the complete preprocessing pipeline you would use before comparing their phases.

3. The envelope of a signal changes slowly compared to the oscillation itself. Why is this, and what implications does it have for the temporal resolution of amplitude-based connectivity measures?

4. Two signals have identical envelopes but different phases. Can the Hilbert transform distinguish between them? How?

5. In hyperscanning, we often compute phase differences between two brains. If both signals are filtered to the alpha band, what does a constant phase difference of 0 mean? What about a constant difference of π?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 3 subplots (column) | Phase ambiguity in composite signal | signal_1, signal_2, signal_3 |
| 2 | Single plot (diagram) | Phasor/rotating vector concept | signal_1, reference |
| 3 | 2 subplots (column) | Sine and its Hilbert transform | signal_1, signal_2 |
| 4 | 2 subplots | Frequency domain view of analytic signal | signal_1, signal_2 |
| 5 | 4 subplots (grid) | Verify Hilbert computation | signal_1 |
| 6 | Single plot (overlay) | AM signal with envelope | signal_1, signal_2 |
| 7 | Single plot (overlay) | EEG alpha with envelope | alpha color, signal_2 |
| 8 | 2 subplots (column) | Sine and its phase (sawtooth) | signal_1, phase_positive |
| 9 | 2 subplots | AM signal: amplitude changes, phase steady | signal_1, phase_positive |
| 10 | 3 subplots | Invalid: broadband Hilbert | signal_1 (with warning) |
| 11 | 3 subplots | Valid: narrowband Hilbert | signal_1 |
| 12 | 4-5 subplots (column) | Complete workflow visualization | signal_1, alpha color |
| 13 | 2 subplots | Instantaneous frequency tracking | signal_1, signal_2 |
| 14 | 2 subplots | Edge effects demonstration | signal_1, reference |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses filtering functions from A04
- [ ] Clear warnings about narrowband requirement in docstrings
- [ ] Edge effect handling documented