# A01: Signals and Sampling — Copilot Directive

## Metadata

- **Notebook ID**: A01
- **Title**: Signals and Sampling
- **Branch**: `feature/A01_signals_and_sampling`
- **Duration**: 60 minutes
- **Prerequisites**: None
- **Next**: A02 (The Frequency Domain)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain the difference between continuous and discrete signals
2. Define sampling rate and its relationship to temporal resolution
3. State the Nyquist theorem and predict when aliasing occurs
4. Generate synthetic signals (sine waves, composite signals, noise) in Python
5. Visualize the effects of undersampling

---

## Required Imports
```
numpy, matplotlib, typing, numpy.typing
Local: src/colors.py (COLORS dict), src/plotting.py (configure_plots function)
```

---

## Section 1: Introduction

**Purpose**: Motivate why understanding signals and sampling matters for EEG analysis.

**Content to cover**:
- EEG measures electrical activity as a continuous phenomenon
- Computers store discrete samples
- The quality of our analysis depends on how well discrete samples represent the continuous reality
- Mention typical EEG sampling rates (256 Hz, 512 Hz, 1024 Hz)

**Length**: 3-4 paragraphs

---

## Section 2: Continuous vs Discrete Signals

**Purpose**: Build intuition about the discretization process.

**Content to cover**:
- Continuous signal: defined at every point in time (mathematical ideal)
- Discrete signal: values only at specific time points
- Sampling: the process of converting continuous to discrete
- Sampling interval (Δt) and sampling rate (fs = 1/Δt)

**Visualization 1**: 
- Plot a continuous sine wave (use many points to appear smooth)
- Overlay discrete samples as scatter points
- Show different sampling densities (many samples vs few samples)
- Use COLORS["signal_1"] for continuous, COLORS["signal_2"] for samples

**Code to produce**:
- Function `generate_time_vector(duration: float, fs: float) -> NDArray[np.float64]`
- Function `generate_sine_wave(t: NDArray[np.float64], frequency: float, amplitude: float, phase: float) -> NDArray[np.float64]`

---

## Section 3: Sampling Rate and Temporal Resolution

**Purpose**: Understand the relationship between sampling rate and what we can capture.

**Content to cover**:
- Higher sampling rate = more samples per second = finer temporal detail
- Sampling rate determines the highest frequency we can accurately represent
- EEG context: why 256 Hz is often sufficient (gamma < 100 Hz)

**Visualization 2**:
- Same signal sampled at different rates (e.g., 1000 Hz, 100 Hz, 20 Hz)
- Three subplots stacked vertically
- Show how low sampling rate loses waveform shape

**Interactive element**:
- Create signals with different fs values
- Visually compare the reconstructed waveforms

---

## Section 4: The Nyquist Theorem

**Purpose**: Introduce the fundamental limit of digital sampling.

**Content to cover**:
- Nyquist theorem statement: to accurately represent a frequency f, we need fs > 2f
- Nyquist frequency: fN = fs / 2 (highest representable frequency)
- Intuition: need at least 2 samples per cycle to capture oscillation
- Examples with numbers: fs = 256 Hz → can capture up to 128 Hz

**Visualization 3**:
- 10 Hz sine wave sampled at different rates
- Show: fs = 100 Hz (good), fs = 25 Hz (borderline), fs = 15 Hz (bad)
- Demonstrate that 25 Hz barely captures the 10 Hz wave

### [Optional Deep Dive] Mathematical Basis

**Content**:
- Brief mention of Shannon-Nyquist sampling theorem
- Formula: fs > 2 * fmax
- Reference to interpolation and perfect reconstruction (conceptual only)

---

## Section 5: Aliasing

**Purpose**: Demonstrate what happens when Nyquist is violated.

**Content to cover**:
- Aliasing definition: high frequencies masquerading as low frequencies
- When fs < 2f, the signal "folds back" into lower frequencies
- Aliased frequency formula: f_aliased = |f - k*fs| where k makes result < fN
- Why this is dangerous: fake low-frequency oscillations appear

**Visualization 4**:
- High-frequency sine wave (e.g., 40 Hz) sampled at low rate (e.g., 50 Hz)
- Show the original signal and the aliased reconstruction
- The 40 Hz wave appears as 10 Hz when sampled at 50 Hz
- Use different colors to distinguish true vs aliased

**Visualization 5**:
- Interactive demonstration: vary sampling rate with a slider concept
- Show multiple frequencies and how they alias
- Create a composite figure showing aliasing zones

**Code to produce**:
- Function `compute_aliased_frequency(true_freq: float, fs: float) -> float`

---

## Section 6: Creating Synthetic Signals

**Purpose**: Build a toolkit for generating test signals used throughout the workshop.

**Content to cover**:
- Why synthetic signals: controlled properties, known ground truth
- Building blocks: sine waves, noise, composite signals

**Code to produce**:

Function 1: `generate_sine_wave` (already defined above, ensure complete)
```
Parameters: t, frequency, amplitude (default=1.0), phase (default=0.0)
Returns: NDArray[np.float64]
Full type hints and docstring
```

Function 2: `generate_white_noise`
```
Parameters: n_samples: int, amplitude: float, seed: Optional[int]
Returns: NDArray[np.float64]
Use numpy random generator with optional seed for reproducibility
```

Function 3: `generate_pink_noise`
```
Parameters: n_samples: int, amplitude: float, seed: Optional[int]
Returns: NDArray[np.float64]
1/f spectrum - more realistic for EEG
```

Function 4: `generate_composite_signal`
```
Parameters: t: NDArray, frequencies: list[float], amplitudes: list[float], phases: Optional[list[float]]
Returns: NDArray[np.float64]
Sum of multiple sine waves
```

**Visualization 6**:
- Grid of 4 subplots showing each signal type
- Time domain plots
- Clear labels indicating signal type and parameters

---

## Section 7: Practical Considerations for EEG

**Purpose**: Connect theory to real EEG recording practice.

**Content to cover**:
- Typical EEG sampling rates and why they are chosen
- Anti-aliasing filters in EEG amplifiers (low-pass before digitization)
- Oversampling: why some systems use higher rates than strictly necessary
- Downsampling: when and how to reduce sampling rate safely

**No code required**: Conceptual discussion only

**Table to include**:
| EEG System | Typical fs | Nyquist | Usable Range |
|------------|-----------|---------|--------------|
| Clinical | 256 Hz | 128 Hz | < 100 Hz |
| Research | 512 Hz | 256 Hz | < 200 Hz |
| High-density | 1024 Hz | 512 Hz | < 400 Hz |

---

## Section 8: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: 
- Create a 5 Hz sine wave, sample at 100 Hz, plot it
- Then sample the same wave at 8 Hz and observe aliasing
- Ask: what frequency does the aliased signal appear to have?

**Exercise 2**:
- Create a composite signal with 3 Hz, 7 Hz, and 15 Hz components
- Sample at 40 Hz (safe) and at 25 Hz (aliasing for 15 Hz component)
- Compare the two results

**Exercise 3**:
- Generate 2 seconds of signal combining:
  - 10 Hz sine wave (amplitude 1.0) - "alpha"
  - 25 Hz sine wave (amplitude 0.5) - "beta"
  - Pink noise (amplitude 0.2)
- Plot the result, this will be our "fake EEG" for future notebooks

---

## Section 9: Functions for src/signals.py

**Purpose**: Identify which functions should be moved to the src module.

**Functions to extract**:
- `generate_time_vector`
- `generate_sine_wave`
- `generate_white_noise`
- `generate_pink_noise`
- `generate_composite_signal`
- `compute_aliased_frequency`

**Note**: In the notebook, first implement inline, then at the end mention these are available in `src/signals.py` for future notebooks.

---

## Summary

**Key takeaways to list**:
- Digital signals are discrete samples of continuous phenomena
- Sampling rate determines temporal resolution and frequency limits
- Nyquist theorem: fs > 2 * fmax to avoid aliasing
- Aliasing creates false low-frequency components from undersampled high frequencies
- EEG systems use anti-aliasing filters and appropriate sampling rates
- Synthetic signals allow controlled experimentation

---

## Discussion Questions

1. An EEG system samples at 256 Hz. A participant has strong muscle artifact at 150 Hz. What will happen to this artifact in the recorded data if no anti-aliasing filter is applied?

2. You want to study gamma oscillations (30-100 Hz). What is the minimum sampling rate you would need? What sampling rate would you actually recommend, and why?

3. Why might a researcher choose to record at 1024 Hz even if they only care about frequencies below 50 Hz?

4. Two signals look identical after sampling. Does this mean they were the same signal before sampling? Why or why not?

5. In hyperscanning, two EEG systems must be synchronized. How might small differences in actual sampling rate (e.g., 255.8 Hz vs 256.2 Hz) affect connectivity analysis over a long recording?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Line + scatter | Continuous vs discrete signal | signal_1, signal_2 |
| 2 | 3 subplots | Same signal, different sampling rates | signal_1 |
| 3 | 3 subplots | Nyquist demonstration | signal_1, signal_2, signal_3 |
| 4 | 2 subplots | Aliasing example (true vs aliased) | signal_1, signal_3 |
| 5 | Complex figure | Aliasing zones visualization | gradient |
| 6 | 2x2 grid | Signal types (sine, white, pink, composite) | signal_1-4 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures use STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction