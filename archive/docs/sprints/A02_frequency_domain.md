# A02: The Frequency Domain — Copilot Directive

## Metadata

- **Notebook ID**: A02
- **Title**: The Frequency Domain
- **Branch**: `feature/A02_frequency_domain`
- **Duration**: 60 minutes
- **Prerequisites**: A01 (Signals and Sampling)
- **Next**: A03 (Power Spectrum and Frequency Bands)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain the intuition behind the Fourier transform (decomposition into oscillations)
2. Distinguish between amplitude spectrum and phase spectrum
3. Compute the Discrete Fourier Transform (DFT) using FFT
4. Interpret frequency domain representations of signals
5. Understand frequency resolution and its relationship to signal duration

---

## Required Imports
```
numpy, matplotlib, scipy.fft, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py (from A01)
```

---

## Section 1: Introduction

**Purpose**: Motivate why frequency domain analysis is essential for EEG.

**Content to cover**:
- Time domain shows amplitude over time, but EEG is fundamentally oscillatory
- Different brain states are characterized by different frequency content (alpha during relaxation, beta during attention)
- Frequency domain reveals hidden structure: a complex signal is the sum of simple oscillations
- The Fourier transform is the bridge between time and frequency domains
- Mention that most connectivity metrics operate in the frequency domain

**Length**: 3-4 paragraphs

---

## Section 2: Intuition — Signals as Sums of Oscillations

**Purpose**: Build intuition before introducing mathematics.

**Content to cover**:
- Any signal can be represented as a sum of sine waves (Fourier's insight)
- Each sine wave has: frequency, amplitude, phase
- Simple analogy: a chord in music is multiple notes (frequencies) played together
- The Fourier transform tells us which frequencies are present and how strong they are

**Visualization 1**:
- Create a composite signal from 3 sine waves (e.g., 3 Hz, 7 Hz, 12 Hz)
- Show 4 subplots in a column:
  - Individual sine wave 1 (3 Hz)
  - Individual sine wave 2 (7 Hz)
  - Individual sine wave 3 (12 Hz)
  - Sum of all three
- Use COLORS["signal_1"], COLORS["signal_2"], COLORS["signal_3"] for components
- Use COLORS["signal_4"] for the sum

**Visualization 2**:
- Reverse demonstration: start with a complex-looking signal
- Show it can be decomposed into simple components
- Teaser for what Fourier transform does automatically

---

## Section 3: The Fourier Transform

**Purpose**: Introduce the Fourier transform conceptually.

**Content to cover**:
- The Fourier transform converts a time-domain signal into frequency-domain representation
- Input: signal x(t), Output: X(f) — a function of frequency
- X(f) is complex-valued: contains both amplitude and phase information
- For digital signals, we use the Discrete Fourier Transform (DFT)
- FFT (Fast Fourier Transform) is an efficient algorithm to compute DFT

**Key equations to present** (in markdown/LaTeX):
- Continuous Fourier Transform (conceptual): X(f) = ∫ x(t) e^(-2πift) dt
- Discrete Fourier Transform: X[k] = Σ x[n] e^(-2πikn/N)
- No need to derive, just present and explain terms

### [Optional Deep Dive] Mathematical Foundation

**Content**:
- Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
- Why complex exponentials? They are eigenfunctions of linear time-invariant systems
- Orthogonality of Fourier basis
- DFT as matrix multiplication (conceptual)

---

## Section 4: Computing the FFT in Python

**Purpose**: Practical implementation of Fourier transform.

**Content to cover**:
- Using scipy.fft.fft() or numpy.fft.fft()
- Understanding the output: array of complex numbers
- Length of output equals length of input
- Frequency axis construction: which index corresponds to which frequency

**Code to produce**:

Function 1: `compute_fft`
```
Parameters: 
    signal: NDArray[np.float64]
    fs: float
Returns: 
    Tuple[NDArray[np.float64], NDArray[np.complex128]]
    (frequencies, fft_values)
Full type hints and docstring
Use scipy.fft.fft and scipy.fft.fftfreq
```

**Visualization 3**:
- Simple sine wave (e.g., 10 Hz)
- Two subplots side by side:
  - Left: time domain signal
  - Right: magnitude of FFT (show spike at 10 Hz)
- Clear axis labels: Time (s) vs Frequency (Hz)

**Visualization 4**:
- Composite signal from Visualization 1 (3 Hz + 7 Hz + 12 Hz)
- Two subplots:
  - Left: time domain (complex looking)
  - Right: frequency domain (three clear peaks at 3, 7, 12 Hz)
- Demonstrate how Fourier "reveals" the hidden components

---

## Section 5: Amplitude Spectrum

**Purpose**: Extract and interpret amplitude information.

**Content to cover**:
- The FFT output is complex: X[k] = a + bi
- Amplitude (magnitude): |X[k]| = √(a² + b²)
- Amplitude spectrum: |X[k]| as a function of frequency
- Interpretation: how much of each frequency is present in the signal
- Normalization considerations (divide by N, or N/2 for one-sided)

**Code to produce**:

Function 2: `compute_amplitude_spectrum`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    normalize: bool = True
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, amplitudes)
Only return positive frequencies (one-sided spectrum)
```

**Visualization 5**:
- Composite signal with known amplitudes (e.g., 3 Hz amp=1.0, 7 Hz amp=0.5, 12 Hz amp=0.3)
- Amplitude spectrum showing peaks with correct relative heights
- Annotate the peaks with their frequencies

---

## Section 6: Phase Spectrum

**Purpose**: Understand the often-neglected phase information.

**Content to cover**:
- Phase: angle of the complex number, φ = atan2(b, a)
- Phase spectrum: φ[k] as a function of frequency
- Interpretation: the starting point (offset) of each oscillation
- Phase is crucial for connectivity analysis (PLV, PLI, etc.)
- Phase is often ignored in power analysis but essential for synchronization

**Code to produce**:

Function 3: `compute_phase_spectrum`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (frequencies, phases_in_radians)
Only return positive frequencies
```

**Visualization 6**:
- Two sine waves at same frequency (10 Hz) but different phases (0 and π/2)
- Grid of 2 rows × 2 columns:
  - Row 1: time domain of signal 1, time domain of signal 2
  - Row 2: phase spectrum of signal 1, phase spectrum of signal 2
- Show that amplitude spectra are identical but phase spectra differ

**Visualization 7**:
- Demonstration of phase importance
- Create two composite signals with same frequencies and amplitudes but different phases
- Show they look completely different in time domain
- Show their amplitude spectra are identical
- Conclude: phase carries essential information about signal structure

---

## Section 7: Frequency Resolution

**Purpose**: Understand the limits of frequency analysis.

**Content to cover**:
- Frequency resolution: Δf = fs / N = 1 / T (where T is signal duration)
- Longer signals → better frequency resolution
- Trade-off: temporal resolution vs frequency resolution
- Cannot distinguish frequencies closer than Δf
- Practical implications: need sufficient data length for fine frequency analysis

**Visualization 8**:
- Same signal (e.g., 10 Hz sine wave) with different durations
- Three subplots showing amplitude spectrum:
  - 0.5 seconds: broad peak, poor resolution
  - 2 seconds: sharper peak
  - 5 seconds: very sharp peak
- Annotate each with its frequency resolution Δf

**Code to produce**:

Function 4: `compute_frequency_resolution`
```
Parameters:
    fs: float
    n_samples: int
Returns:
    float (frequency resolution in Hz)
```

---

## Section 8: Symmetry and One-Sided Spectrum

**Purpose**: Explain why we often only show positive frequencies.

**Content to cover**:
- For real-valued signals, FFT output is symmetric around 0 Hz
- Negative frequencies are redundant (contain same information as positive)
- Convention: show one-sided spectrum (0 to fs/2)
- The Nyquist frequency appears at the center of the full spectrum
- Doubling amplitude for one-sided spectrum (energy conservation)

**Visualization 9**:
- Full FFT output (both negative and positive frequencies)
- Highlight the symmetry
- Show one-sided version below it
- Explain this is the standard representation

---

## Section 9: Windowing (Brief Introduction)

**Purpose**: Introduce the concept of spectral leakage and windowing.

**Content to cover**:
- FFT assumes signal is periodic — but real signals are not
- Discontinuity at edges causes "spectral leakage" (energy spreads to nearby frequencies)
- Windowing: multiply signal by a tapered window before FFT
- Common windows: Hann, Hamming, Blackman
- Trade-off: reduces leakage but broadens peaks
- More detail in A03 (Power Spectrum)

**Visualization 10**:
- Sine wave that does not complete full cycles in the window
- Two subplots:
  - Without windowing: leakage visible (smeared peak)
  - With Hann window: cleaner peak
- Brief demonstration, not exhaustive treatment

---

## Section 10: Practical Example — Composite EEG-like Signal

**Purpose**: Apply all concepts to a realistic signal.

**Content to cover**:
- Create a synthetic "EEG-like" signal with multiple frequency components
- Add some noise
- Compute and visualize the full analysis

**Code to produce**:
- Use functions from A01 (generate_composite_signal, generate_pink_noise)
- Combine to create realistic test signal

**Visualization 11**:
- Figure with 3 subplots:
  - Top: time domain signal (2-3 seconds)
  - Middle: amplitude spectrum (identify frequency components)
  - Bottom: phase spectrum
- Annotate the frequency peaks corresponding to "brain rhythms"

---

## Section 11: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**:
- Create a signal with unknown frequencies (provide the signal, hide the code that made it)
- Ask learner to use FFT to identify the component frequencies
- Solution should reveal 5 Hz, 11 Hz, and 23 Hz components

**Exercise 2**:
- Two signals are provided, they look similar in time domain
- Use amplitude and phase spectra to determine how they differ
- One has same frequencies but different phases

**Exercise 3**:
- Investigate frequency resolution
- Create a signal with two close frequencies (e.g., 10 Hz and 11 Hz)
- Determine the minimum signal duration needed to resolve them as separate peaks
- Verify experimentally

---

## Section 12: Functions for src/signals.py and src/spectral.py

**Purpose**: Identify which functions should be moved to src modules.

**Functions to add to src/spectral.py (new file)**:
- `compute_fft`
- `compute_amplitude_spectrum`
- `compute_phase_spectrum`
- `compute_frequency_resolution`

**Note**: Mention at the end of notebook that these functions are available in `src/spectral.py` for future notebooks.

---

## Summary

**Key takeaways to list**:
- The Fourier transform decomposes signals into constituent oscillations
- FFT is the efficient algorithm to compute the Discrete Fourier Transform
- Amplitude spectrum shows the strength of each frequency component
- Phase spectrum shows the timing offset of each component
- Frequency resolution depends on signal duration: Δf = 1/T
- For real signals, the spectrum is symmetric; we typically show positive frequencies only
- Windowing reduces spectral leakage from finite-length signals

---

## Discussion Questions

1. You record 1 second of EEG data. Can you distinguish between 10 Hz and 10.5 Hz oscillations? What about with 3 seconds of data?

2. Two EEG channels show identical amplitude spectra but different phase spectra. What might this indicate about the relationship between these brain regions?

3. Why do you think most EEG connectivity metrics (coherence, PLV, etc.) operate in the frequency domain rather than the time domain?

4. A colleague shows you an amplitude spectrum with a very sharp, narrow peak. What does this tell you about the underlying signal?

5. In hyperscanning, we analyze two brains simultaneously. How might phase relationships between two participants' EEG signals at the same frequency be meaningful?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 4 subplots (column) | Component sine waves + sum | signal_1, signal_2, signal_3, signal_4 |
| 2 | 2 subplots | Complex signal decomposition | signal_1, gradient |
| 3 | 2 subplots (side-by-side) | Simple sine: time vs frequency | signal_1 |
| 4 | 2 subplots (side-by-side) | Composite: time vs frequency peaks | signal_4 |
| 5 | Single plot | Amplitude spectrum with annotated peaks | signal_1 |
| 6 | 2×2 grid | Phase comparison of two signals | signal_1, signal_2 |
| 7 | 2×2 grid | Same amplitude, different phase demonstration | signal_1, signal_2 |
| 8 | 3 subplots (column) | Frequency resolution vs duration | signal_1 |
| 9 | 2 subplots | Full spectrum vs one-sided | signal_1, signal_2 |
| 10 | 2 subplots | Spectral leakage and windowing | signal_1, signal_3 |
| 11 | 3 subplots (column) | Complete EEG-like analysis | signal_1, band colors |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Uses functions from A01 (src/signals.py) where appropriate