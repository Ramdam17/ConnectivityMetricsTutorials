# B04: Wavelets and Time-Frequency Analysis — Copilot Directive

## Metadata

- **Notebook ID**: B04
- **Title**: Wavelets and Time-Frequency Analysis
- **Branch**: `feature/B04_wavelets_time_frequency`
- **Duration**: 75 minutes
- **Prerequisites**: A02 (Frequency Domain), A04 (Filtering), B01 (Hilbert Transform)
- **Next**: C01 (Volume Conduction Problem)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand limitations of FFT for non-stationary signals
2. Explain the time-frequency trade-off (Heisenberg uncertainty)
3. Understand Short-Time Fourier Transform (STFT) as a bridge concept
4. Define and visualize Morlet wavelets
5. Perform wavelet convolution to extract time-frequency representations
6. Extract instantaneous power and phase from wavelet coefficients
7. Choose appropriate wavelet parameters for EEG analysis
8. Compare wavelet approach vs Hilbert approach for connectivity

---

## Required Imports
```
numpy, matplotlib, scipy.signal, scipy.fft, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/filtering.py, src/hilbert.py
```

---

## Section 1: Introduction — The Problem with FFT

**Purpose**: Motivate time-frequency analysis.

**Content to cover**:
- FFT tells us WHICH frequencies are present but not WHEN
- FFT assumes signal is stationary (same statistics throughout)
- EEG is highly non-stationary: oscillations come and go
- Example: alpha bursts appear and disappear over seconds
- We need a method that provides BOTH time AND frequency information
- Solution: time-frequency analysis (spectrograms, wavelets)

**Visualization 1**:
- Non-stationary signal: concatenate three segments
  - 0-3s: 5 Hz oscillation
  - 3-6s: 15 Hz oscillation  
  - 6-9s: 10 Hz oscillation
- Two subplots:
  - Top: time-domain signal (clearly shows changes)
  - Bottom: FFT magnitude spectrum (shows all three frequencies but no timing!)
- Caption: "FFT loses temporal information"

---

## Section 2: Intuition — Windows in Time

**Purpose**: Build intuition for time-frequency analysis.

**Content to cover**:
- Idea: compute FFT on SHORT windows of the signal
- Slide window through time → frequency content at each time point
- Trade-off: 
  - Short window → good time resolution, poor frequency resolution
  - Long window → poor time resolution, good frequency resolution
- This is the **Heisenberg uncertainty principle** for signals!
- Cannot have perfect resolution in both time AND frequency simultaneously
- Δt × Δf ≥ constant (fundamental limit)

**Visualization 2**:
- Same non-stationary signal
- Show different window sizes:
  - Very short window (0.2s): good time resolution, fuzzy frequency
  - Medium window (1s): balanced
  - Long window (3s): good frequency resolution, fuzzy time
- Side-by-side spectrograms showing the trade-off
- Annotate which features are resolved in each

---

## Section 3: Short-Time Fourier Transform (STFT)

**Purpose**: Introduce STFT as the simplest time-frequency method.

**Content to cover**:
- STFT: FFT computed on sliding windows
- Window function (Hanning, Hamming) reduces edge artifacts
- Parameters:
  - Window length (nperseg): determines frequency resolution
  - Overlap (noverlap): determines time sampling density
  - Window type: controls spectral leakage
- Output: 2D array (frequency × time) of complex coefficients
- Magnitude squared = spectrogram (power over time-frequency)
- STFT uses FIXED window size → fixed resolution at all frequencies

**Code to produce**:

Function 1: `compute_stft`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    noverlap: int | None = None  # defaults to nperseg // 2
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]
    (frequencies, times, stft_matrix)
Wrapper around scipy.signal.stft
```

Function 2: `compute_spectrogram`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    noverlap: int | None = None
    window: str = "hann"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    (frequencies, times, power_matrix)
Power spectrogram: |STFT|²
```

**Visualization 3**:
- Non-stationary signal from Section 1
- Spectrogram (time × frequency heatmap)
- Clearly shows 5 Hz → 15 Hz → 10 Hz transition
- Colorbar for power
- Both axes labeled with units

---

## Section 4: Limitations of STFT

**Purpose**: Motivate wavelets as an improvement.

**Content to cover**:
- STFT uses same window size for ALL frequencies
- Problem: optimal window size depends on frequency!
  - Low frequencies (e.g., 2 Hz): need long window to capture full cycle
  - High frequencies (e.g., 40 Hz): can use short window
- STFT is a compromise that's suboptimal for both
- Solution: use ADAPTIVE window size that scales with frequency
- This is what wavelets do!

**Visualization 4**:
- STFT spectrogram of signal with both low (3 Hz) and high (40 Hz) components
- Show that:
  - With short window: high freq well-resolved, low freq smeared
  - With long window: low freq well-resolved, high freq smeared
- Two spectrograms side by side
- Annotate the resolution problems

---

## Section 5: Introduction to Wavelets

**Purpose**: Define wavelets conceptually.

**Content to cover**:
- Wavelet: a small, localized oscillation (wave + let = "small wave")
- Key properties:
  - Oscillates (has frequency content)
  - Localized in time (starts and ends, unlike infinite sine)
  - Zero mean (integrates to zero)
- Many wavelet families: Morlet, Mexican hat, Daubechies, etc.
- For EEG/neural signals: **Morlet wavelet** is standard
- Morlet = sine wave × Gaussian envelope
- Natural for oscillatory neural signals

**Visualization 5**:
- Gallery of different wavelet types:
  - Morlet wavelet
  - Mexican hat (Ricker) wavelet
  - Haar wavelet
  - Daubechies wavelet
- 2×2 grid, each showing time-domain shape
- Caption: "Morlet is most common for neural oscillations"

---

## Section 6: The Morlet Wavelet

**Purpose**: Deep dive into Morlet wavelet properties.

**Content to cover**:
- Morlet wavelet formula:
  $$\psi(t) = e^{i 2\pi f_0 t} \cdot e^{-t^2 / (2\sigma^2)}$$
- Two components:
  - Complex exponential: oscillation at frequency f₀
  - Gaussian: envelope that localizes in time
- Parameter σ (sigma) controls width → trade-off
- Often parameterized by number of cycles (n_cycles)
  - n_cycles = f₀ × σ × √(2)
  - More cycles → better frequency resolution, worse time resolution
- Standard choice for EEG: 5-7 cycles
- Wavelet is complex → provides both amplitude AND phase

**Code to produce**:

Function 3: `create_morlet_wavelet`
```
Parameters:
    fs: float
    frequency: float
    n_cycles: float = 7.0
Returns:
    NDArray[np.complex128]
Create a Morlet wavelet at specified frequency
Length determined automatically (±3 standard deviations)
Returns complex wavelet centered at t=0
```

**Visualization 6**:
- Single Morlet wavelet (e.g., 10 Hz, 7 cycles)
- Three subplots:
  - Real part
  - Imaginary part
  - Magnitude (envelope)
- Show the Gaussian envelope shape
- Annotate: "7 cycles of 10 Hz oscillation"

**Visualization 7**:
- Effect of n_cycles parameter
- Same frequency (10 Hz), different n_cycles: 3, 5, 7, 10
- 4 subplots showing wavelet shapes
- Annotate time extent and frequency specificity
- Caption: "More cycles = longer wavelet = better frequency resolution"

---

## Section 7: Wavelet Convolution

**Purpose**: Explain how wavelets extract time-frequency information.

**Content to cover**:
- To analyze signal at frequency f: convolve signal with wavelet at f
- Convolution: slide wavelet along signal, compute overlap at each point
- Result: complex time series of wavelet coefficients
- |coefficient|² = power at that frequency and time
- angle(coefficient) = phase at that frequency and time
- Repeat for multiple frequencies → full time-frequency representation

**Code to produce**:

Function 4: `wavelet_convolution`
```
Parameters:
    signal: NDArray[np.float64]
    wavelet: NDArray[np.complex128]
Returns:
    NDArray[np.complex128]
Convolve signal with wavelet using FFT-based convolution
Return complex coefficients (same length as signal)
```

Function 5: `compute_wavelet_transform`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    frequencies: NDArray[np.float64]
    n_cycles: float | NDArray[np.float64] = 7.0
Returns:
    NDArray[np.complex128]
    Shape: (n_frequencies, n_times)
Full wavelet transform across specified frequencies
If n_cycles is array, use different cycles per frequency
```

**Visualization 8**:
- Step-by-step wavelet convolution illustration
- Signal with a burst of 10 Hz activity
- Show wavelet sliding along signal
- Show resulting coefficient magnitude peaking at burst location
- Animated concept (or 4-panel showing different time points)

---

## Section 8: Time-Frequency Representation

**Purpose**: Create and interpret wavelet spectrograms.

**Content to cover**:
- Apply wavelet transform across many frequencies
- Result: 2D matrix (frequency × time) of complex coefficients
- Power = |coefficients|² → wavelet spectrogram
- Unlike STFT, resolution adapts to frequency:
  - Low frequencies: good frequency resolution, poorer time resolution
  - High frequencies: good time resolution, poorer frequency resolution
- This matches how neural oscillations behave!

**Code to produce**:

Function 6: `compute_wavelet_power`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    frequencies: NDArray[np.float64]
    n_cycles: float | NDArray[np.float64] = 7.0
Returns:
    NDArray[np.float64]
    Shape: (n_frequencies, n_times)
Wavelet power (squared magnitude of coefficients)
```

Function 7: `plot_time_frequency`
```
Parameters:
    times: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    power: NDArray[np.float64]
    ax: plt.Axes | None = None
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    log_scale: bool = False
Returns:
    plt.Axes
Plot time-frequency representation with proper axes
Optional log scaling for power
```

**Visualization 9**:
- Non-stationary signal from Section 1
- Wavelet spectrogram
- Compare side-by-side with STFT spectrogram
- Show that wavelet has adaptive resolution
- Both using same color scale for fair comparison

---

## Section 9: Choosing Wavelet Parameters

**Purpose**: Practical guidance for parameter selection.

**Content to cover**:
- **Frequency range**: typically 1-100 Hz for EEG
- **Frequency spacing**: linear or logarithmic
  - Linear: equal spacing (1, 2, 3, ... Hz)
  - Logarithmic: more resolution at low frequencies
- **Number of cycles**: the key parameter
  - Low n_cycles (3-4): good time resolution, poor frequency resolution
  - High n_cycles (7-10): good frequency resolution, poor time resolution
  - Common default: 7 cycles
- **Adaptive cycles**: vary n_cycles with frequency
  - e.g., n_cycles = frequency / 2 (minimum 3)
  - Keeps constant number of cycles as proportion of period
- For connectivity: match parameters to your scientific question

**Code to produce**:

Function 8: `create_frequency_vector`
```
Parameters:
    fmin: float
    fmax: float
    n_frequencies: int
    spacing: str = "linear"  # or "log"
Returns:
    NDArray[np.float64]
Create frequency vector for wavelet analysis
```

Function 9: `compute_adaptive_cycles`
```
Parameters:
    frequencies: NDArray[np.float64]
    min_cycles: float = 3.0
    max_cycles: float = 10.0
Returns:
    NDArray[np.float64]
Compute adaptive n_cycles for each frequency
Scales with frequency, bounded by min/max
```

**Visualization 10**:
- Effect of n_cycles on time-frequency resolution
- Same signal, three spectrograms:
  - n_cycles = 3 (time-focused)
  - n_cycles = 7 (balanced)
  - n_cycles = 12 (frequency-focused)
- 1×3 grid with shared colorbar
- Annotate trade-offs visible in each

---

## Section 10: Extracting Phase from Wavelets

**Purpose**: Show wavelets provide instantaneous phase.

**Content to cover**:
- Wavelet coefficients are complex: W(t, f) = A(t, f) × e^(iφ(t, f))
- Phase: φ(t, f) = angle(W(t, f))
- Unlike Hilbert (which gives phase for ONE frequency band)...
- Wavelets give phase at MULTIPLE frequencies simultaneously
- Useful for:
  - Phase-based connectivity at specific frequencies
  - Cross-frequency phase analysis
  - Phase alignment across trials (event-related analysis)

**Code to produce**:

Function 10: `compute_wavelet_phase`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    frequencies: NDArray[np.float64]
    n_cycles: float | NDArray[np.float64] = 7.0
Returns:
    NDArray[np.float64]
    Shape: (n_frequencies, n_times)
Instantaneous phase at each frequency and time
Returns values in [-π, π]
```

**Visualization 11**:
- Signal with clear oscillation
- Extract phase at that oscillation's frequency
- Three subplots:
  - Original signal (filtered to band)
  - Wavelet power at that frequency
  - Wavelet phase at that frequency
- Show that phase increases linearly during oscillation

---

## Section 11: Wavelet vs Hilbert Approach

**Purpose**: Compare the two methods for extracting amplitude/phase.

**Content to cover**:
- **Hilbert approach** (from B01):
  - Band-pass filter → Hilbert transform → envelope/phase
  - One frequency band at a time
  - Computationally efficient
  - Standard in many connectivity pipelines
- **Wavelet approach**:
  - Direct convolution with wavelets at multiple frequencies
  - All frequencies simultaneously
  - More parameters to tune
  - Better for exploring time-frequency patterns
- **When to use which**:
  - Known frequency band of interest → Hilbert is simpler
  - Exploratory analysis → Wavelets show full picture
  - Cross-frequency analysis → Wavelets necessary
  - Connectivity metrics (PLV, etc.) → Often use Hilbert, but wavelet is valid

**Visualization 12**:
- Same signal analyzed both ways
- 2×2 grid:
  - Top-left: Hilbert envelope (alpha band)
  - Top-right: Wavelet power at 10 Hz
  - Bottom-left: Hilbert phase (alpha band)
  - Bottom-right: Wavelet phase at 10 Hz
- Show high correlation between methods
- Caption: "Both methods yield similar results for narrowband analysis"

---

## Section 12: Edge Effects in Wavelet Analysis

**Purpose**: Warn about artifacts at signal boundaries.

**Content to cover**:
- Wavelets extend beyond signal edges → edge artifacts
- Longer wavelets (more cycles) = more edge artifact
- Low frequencies = longer wavelets = more edge artifact
- Solutions:
  - Pad signal (zeros, reflection, extrapolation)
  - Crop edges after analysis
  - Use longer recordings than needed
- Rule of thumb: discard n_cycles / (2 × frequency) at each edge

**Code to produce**:

Function 11: `compute_edge_samples`
```
Parameters:
    frequency: float
    fs: float
    n_cycles: float
Returns:
    int
Number of samples affected by edge artifacts at given frequency
```

**Visualization 13**:
- Signal with clear start/end
- Wavelet spectrogram showing full signal
- Highlight edge regions with semi-transparent overlay
- Show that low frequencies have larger affected regions
- Annotate: "Crop these regions before analysis"

---

## Section 13: Practical Application — Event-Related Time-Frequency

**Purpose**: Apply wavelets to event-related analysis.

**Content to cover**:
- Common use: analyze oscillatory activity around events
- Example: stimulus onset, response, social cue
- Epoch data around event (e.g., -1s to +2s)
- Compute wavelet power for each trial
- Average across trials → event-related spectral perturbation (ERSP)
- Reveals: which frequencies change when, relative to event

**Code to produce**:
- Generate synthetic event-related data:
  - Baseline period: mixed frequencies
  - Post-event: alpha suppression, gamma increase

**Visualization 14**:
- Event-related time-frequency plot
- Time on x-axis (with 0 = event onset marked)
- Frequency on y-axis
- Power (color) showing changes
- Clear visual of alpha decrease, gamma increase post-event
- Typical ERSP visualization style

---

## Section 14: Hyperscanning Application — Time-Resolved Connectivity

**Purpose**: Connect wavelets to hyperscanning analysis.

**Content to cover**:
- Wavelet approach enables TIME-RESOLVED connectivity
- Instead of one PLV value per recording...
- Get PLV(t) at each time point!
- Useful for:
  - Tracking synchronization dynamics during interaction
  - Identifying moments of high/low connectivity
  - Correlating connectivity with behavior
- Brief preview: compute phase at each time, then PLV in sliding window

**Visualization 15**:
- Simulated hyperscanning with dynamic coupling
- Two participants, coupling varies over time:
  - 0-3s: low coupling
  - 3-6s: high coupling
  - 6-9s: low coupling
- Four subplots:
  - Participant 1 spectrogram
  - Participant 2 spectrogram
  - Phase difference at 10 Hz over time
  - Sliding-window PLV over time (showing the dynamics)
- Caption: "Wavelet enables time-resolved connectivity analysis"

---

## Section 15: Comparison with MNE-Python

**Purpose**: Show professional implementation.

**Content to cover**:
- MNE-Python has robust time-frequency functions
- `mne.time_frequency.tfr_morlet`: standard wavelet analysis
- `mne.time_frequency.tfr_multitaper`: alternative method
- These handle:
  - Edge effects properly
  - Baseline correction
  - Statistical comparisons
  - Multi-channel analysis
- For production hyperscanning: use MNE, not custom code
- Our implementations are for UNDERSTANDING

**Code to produce**:
- Brief demonstration of MNE's tfr_morlet interface (conceptual, may not run without MNE data structure)

---

## Section 16: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: STFT Parameter Exploration
- Create signal with known time-varying frequency
- Compute STFT with different window sizes
- Find the setting that best resolves both time AND frequency
- Document the trade-off you observe

**Exercise 2**: Morlet Wavelet Creation
- Create Morlet wavelets at 5, 10, 20, 40 Hz (all with 7 cycles)
- Plot all four and compare their temporal extent
- Verify: higher frequency = shorter wavelet
- Compute duration in seconds for each

**Exercise 3**: Wavelet vs Hilbert Comparison
- Generate alpha-band signal with amplitude modulation
- Extract envelope using:
  - Hilbert method (filter → Hilbert → magnitude)
  - Wavelet method (wavelet at 10 Hz → magnitude)
- Compute correlation between the two envelopes
- Which method is more robust to noise?

**Exercise 4**: Edge Effect Quantification
- Create clean 10 Hz oscillation
- Compute wavelet power with n_cycles = 3, 7, 12
- Identify where edge artifacts occur for each
- Verify against theoretical prediction from `compute_edge_samples`

**Exercise 5**: Event-Related Analysis
- Generate synthetic data: 20 trials of event-related response
- Compute single-trial wavelet power
- Average across trials
- Create ERSP plot showing time-frequency changes

**Exercise 6**: Time-Resolved PLV (Preview)
- Two signals with coupling that varies over time
- Extract phase at 10 Hz using wavelets
- Compute PLV in sliding 1-second windows
- Plot PLV over time and verify it tracks the coupling changes

---

## Section 17: Summary

**Key takeaways to list**:
- FFT loses temporal information; time-frequency methods preserve it
- Heisenberg uncertainty: cannot have perfect time AND frequency resolution
- STFT: fixed window size, fixed resolution at all frequencies
- Wavelets: adaptive resolution (better time at high freq, better freq at low freq)
- Morlet wavelet: Gaussian-windowed sinusoid, standard for EEG
- n_cycles parameter controls time-frequency trade-off
- Wavelet coefficients provide both power AND phase
- Wavelet approach complements Hilbert approach
- Edge effects are significant, especially at low frequencies
- For hyperscanning: wavelets enable time-resolved connectivity analysis

---

## Discussion Questions

1. You're analyzing EEG during a 500 ms stimulus presentation. You want to see alpha suppression (8-13 Hz) and gamma increase (30-50 Hz). How would you set your wavelet parameters to optimize resolution for both?

2. A colleague argues that Hilbert transform is "better" than wavelets because it's simpler. How would you explain when each method is more appropriate?

3. Your wavelet spectrogram shows interesting activity in the first 200 ms of your 3-second trials. How can you verify this is real activity and not edge artifact?

4. In hyperscanning, you want to track moment-by-moment synchronization between two participants. What are the trade-offs in choosing your time-frequency analysis parameters?

5. You analyze the same data with n_cycles=3 and n_cycles=10 and get different results. The n_cycles=3 analysis shows a brief gamma burst that's not visible with n_cycles=10. Which result is "correct"? How would you resolve this?

---

## Functions for src/wavelets.py

**New file**: src/wavelets.py

Functions to extract:
- `create_morlet_wavelet`
- `wavelet_convolution`
- `compute_wavelet_transform`
- `compute_wavelet_power`
- `compute_wavelet_phase`
- `create_frequency_vector`
- `compute_adaptive_cycles`
- `compute_edge_samples`
- `plot_time_frequency`

Also add to src/spectral.py:
- `compute_stft`
- `compute_spectrogram`

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 2 subplots | Non-stationary signal + FFT spectrum | signal_1, highlight |
| 2 | 3 spectrograms | Window size trade-off | viridis colormap |
| 3 | 2 subplots | Signal + STFT spectrogram | signal_1, viridis |
| 4 | 2 spectrograms | STFT resolution problem | viridis |
| 5 | 2×2 grid | Wavelet gallery | signal_1-4 |
| 6 | 3 subplots | Morlet wavelet components | signal_1, signal_2 |
| 7 | 4 subplots | Effect of n_cycles on wavelet shape | signal_1-4 |
| 8 | 4 panels | Wavelet convolution illustration | signal_1, signal_2 |
| 9 | 2 spectrograms | Wavelet vs STFT comparison | viridis |
| 10 | 1×3 grid | n_cycles effect on spectrogram | viridis |
| 11 | 3 subplots | Signal, power, phase from wavelet | band color |
| 12 | 2×2 grid | Wavelet vs Hilbert comparison | signal_1, signal_2 |
| 13 | Spectrogram + overlay | Edge effects visualization | viridis, red overlay |
| 14 | ERSP plot | Event-related time-frequency | RdBu colormap |
| 15 | 4 subplots | Hyperscanning time-resolved connectivity | subject_1, subject_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings  
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] FFT-based convolution used for efficiency
- [ ] Edge effects documented and handled
- [ ] Frequency vectors support both linear and log spacing
- [ ] Complex wavelet coefficients handled correctly