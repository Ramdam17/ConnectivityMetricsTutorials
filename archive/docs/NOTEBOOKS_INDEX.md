# Notebooks Index — Hyperscanning Workshop

Complete index of all notebooks with objectives, key concepts, and outputs.

> **Last updated**: March 2026 — reflects actual notebook structure including splits (A04a/b, B02a/b) and full G/H/I scope.

---

# Part 1: Foundations

## Block A: Signal Fundamentals

### A01: Signals and Sampling

**Objective**: Understand how continuous signals become digital data.

**Key Concepts**:
- Continuous vs discrete signals
- Sampling rate and temporal resolution
- Nyquist theorem and aliasing
- Creating synthetic signals (sine waves, noise)

**Outputs**:
- Functions to generate basic signals
- Visualizations of sampling effects
- Understanding of aliasing artifacts

**Prerequisites**: None

**Estimated Duration**: 60 minutes

---

### A02: The Frequency Domain

**Objective**: Understand how signals can be represented as sums of oscillations.

**Key Concepts**:
- Fourier transform intuition (decomposition into frequencies)
- Amplitude spectrum and phase spectrum
- Discrete Fourier Transform (DFT) and FFT
- Frequency resolution and windowing

**Outputs**:
- Function to compute and plot frequency spectrum
- Understanding of amplitude vs phase information
- Ability to interpret spectral plots

**Prerequisites**: A01

**Estimated Duration**: 60 minutes

---

### A03: Power Spectrum and Frequency Bands

**Objective**: Quantify oscillatory activity in specific frequency ranges.

**Key Concepts**:
- Power Spectral Density (PSD)
- EEG frequency bands (delta, theta, alpha, beta, gamma)
- Band power extraction
- Welch's method for robust estimation

**Outputs**:
- Function to compute PSD
- Function to extract band power
- Visualizations comparing spectra across conditions

**Prerequisites**: A02

**Estimated Duration**: 60 minutes

---

### A04a: Filter Fundamentals ✅

**Objective**: Understand the theory and design of digital filters.

**Key Concepts**:
- Why we filter (noise removal, band isolation)
- Filter types: low-pass, high-pass, band-pass, notch
- FIR vs IIR filters
- Filter design parameters (order, cutoff, transition band)
- Frequency response and impulse response

**Outputs**:
- Functions to design filters
- Understanding of filter types and trade-offs

**Prerequisites**: A02

**Estimated Duration**: 45 minutes

---

### A04b: Applied Filtering ✅

**Objective**: Apply filters correctly to EEG-like signals and avoid common artifacts.

**Key Concepts**:
- Applying filters in practice
- Edge effects and padding
- Phase distortion and zero-phase filtering
- Filter artifacts and validation

**Outputs**:
- Functions to apply filters
- Understanding of filter artifacts
- Clean band-limited signals for further analysis

**Prerequisites**: A04a

**Estimated Duration**: 45 minutes

---

## Block B: Phase and Amplitude

### B01: The Hilbert Transform

**Objective**: Extract instantaneous phase and amplitude from a signal.

**Key Concepts**:
- The analytic signal concept
- Hilbert transform as 90° phase shift
- Instantaneous amplitude (envelope)
- Instantaneous phase
- Requirements: narrowband signals

**Outputs**:
- Function to compute analytic signal
- Functions to extract envelope and phase
- Understanding of when Hilbert transform is valid

**Prerequisites**: A04

**Estimated Duration**: 60 minutes

---

### B02a: Circular Statistics ✅

**Objective**: Understand and apply circular statistics to phase data.

**Key Concepts**:
- Phase as an angle (0 to 2π)
- Phase wrapping and unwrapping
- Circular mean, circular variance, concentration (κ)
- Phase distribution visualizations (polar plots, rose diagrams)

**Outputs**:
- Functions for circular statistics
- Understanding of the circular nature of phase

**Prerequisites**: B01

**Estimated Duration**: 45 minutes

---

### B02b: Phase Relationships ✅

**Objective**: Analyze phase differences and understand phase synchronization.

**Key Concepts**:
- Phase difference between two signals
- Phase locking concept (preview of G01)
- Phase difference distributions
- Interpretation of phase lead/lag

**Outputs**:
- Functions for phase difference computation
- Understanding of phase synchronization concept

**Prerequisites**: B02a

**Estimated Duration**: 45 minutes

---

### B03: Amplitude Envelope

**Objective**: Analyze slow fluctuations in signal amplitude.

**Key Concepts**:
- Envelope as instantaneous amplitude
- Low-frequency fluctuations of band power
- Amplitude vs power (squared amplitude)
- Envelope smoothing techniques
- Amplitude coupling concept

**Outputs**:
- Functions to extract and process envelopes
- Visualizations of envelope dynamics
- Understanding of amplitude-based connectivity

**Prerequisites**: B01

**Estimated Duration**: 60 minutes

---

### B04: Wavelets and Time-Frequency Analysis

**Objective**: Analyze how frequency content changes over time.

**Key Concepts**:
- Limitation of Fourier: no temporal information
- Time-frequency trade-off (Heisenberg uncertainty)
- Wavelet transform intuition
- Morlet wavelets for EEG
- Scalograms and time-frequency plots

**Outputs**:
- Function to compute wavelet transform
- Time-frequency visualizations
- Understanding of when to use wavelets vs Fourier

**Prerequisites**: A02, B01

**Estimated Duration**: 60 minutes

---

## Block C: Connectivity Concepts

### C01: The Volume Conduction Problem

**Objective**: Understand why EEG connectivity analysis is challenging.

**Key Concepts**:
- How electrical signals spread through tissue
- Volume conduction creates spurious correlations
- Zero-lag artifact: instant signal mixing
- Field spread vs true connectivity
- Source localization (brief mention)

**Outputs**:
- Simulations demonstrating volume conduction
- Understanding of why some metrics fail
- Motivation for robust connectivity measures

**Prerequisites**: B02

**Estimated Duration**: 60 minutes

---

### C02: Connectivity Matrices

**Objective**: Represent and visualize brain connectivity patterns.

**Key Concepts**:
- Connectivity matrix structure (electrodes × electrodes)
- Symmetric vs asymmetric (directed) matrices
- Intra-brain vs inter-brain matrices
- Visualization: heatmaps, chord diagrams, topographic plots
- Thresholding and binarization

**Outputs**:
- Functions to create and manipulate connectivity matrices
- Multiple visualization techniques
- Understanding of matrix interpretation

**Prerequisites**: A03

**Estimated Duration**: 60 minutes

---

### C03: Statistical Inference for Connectivity

**Objective**: Determine whether connectivity values are statistically significant.

**Key Concepts**:
- Why statistical testing is necessary
- The multiple comparisons problem
- Surrogate data: shuffling, phase randomization
- Permutation testing
- False Discovery Rate (FDR) correction
- Cluster-based corrections (brief)

**Outputs**:
- Functions to generate surrogate data
- Functions for permutation testing
- Understanding of corrected significance thresholds

**Prerequisites**: C02

**Estimated Duration**: 60 minutes

---

## Block D: Information Theory

### D01: Entropy Basics

**Objective**: Quantify uncertainty and information content in signals.

**Key Concepts**:
- Probability distributions from data
- Shannon entropy: measuring uncertainty
- Joint entropy and conditional entropy
- Discretization strategies (binning, symbolic)
- Entropy estimation challenges

**Outputs**:
- Functions to estimate entropy
- Understanding of entropy interpretation
- Awareness of estimation biases

**Prerequisites**: A01

**Estimated Duration**: 60 minutes

---

### D02: Mutual Information

**Objective**: Quantify shared information between two signals.

**Key Concepts**:
- Mutual information definition
- Relationship to entropy: MI = H(X) + H(Y) - H(X,Y)
- MI vs correlation: captures nonlinear dependencies
- Estimation methods: binning, KSG estimator
- Normalized mutual information

**Outputs**:
- Functions to compute mutual information
- Comparison with correlation
- Understanding of MI as connectivity metric

**Prerequisites**: D01

**Estimated Duration**: 60 minutes

---

### D03: Transfer Entropy and Causality

**Objective**: Detect directed information flow between signals.

**Key Concepts**:
- Correlation vs causation
- Granger causality concept
- Transfer entropy: information-theoretic Granger
- Time-lagged dependencies
- Limitations and confounds

**Outputs**:
- Functions to compute transfer entropy
- Understanding of directed connectivity
- Awareness of causality interpretation limits

**Prerequisites**: D02

**Estimated Duration**: 60 minutes

---

## Block E: Hyperscanning Context

### E01: Graph Theory Basics

**Objective**: Analyze connectivity patterns as networks.

**Key Concepts**:
- Graphs: nodes (electrodes) and edges (connectivity)
- Weighted vs binary graphs
- Basic metrics: degree, strength, clustering coefficient
- Path length and efficiency
- Network visualization techniques

**Outputs**:
- Functions to convert matrices to graphs
- Functions for basic graph metrics
- Network visualizations

**Prerequisites**: C02

**Estimated Duration**: 60 minutes

---

### E02: Introduction to Hyperscanning

**Objective**: Understand the framework for multi-brain analysis.

**Key Concepts**:
- What is hyperscanning?
- History and methodological evolution
- Paradigms: cooperation, competition, communication
- Intra-brain vs inter-brain connectivity
- The dyad as unit of analysis
- Challenges specific to hyperscanning

**Outputs**:
- Conceptual framework for inter-brain analysis
- Overview of metrics to be covered
- Understanding of experimental design considerations

**Prerequisites**: C01, E01

**Estimated Duration**: 60 minutes

---

# Part 2: Connectivity Metrics

## Block F: Coherence-Based

### F01: Spectral Coherence (COH)

**Objective**: Measure frequency-specific linear coupling between signals.

**Key Concepts**:
- Coherence definition: normalized cross-spectrum
- Interpretation: shared oscillatory activity
- Magnitude-squared coherence
- Coherence vs correlation
- Sensitivity to volume conduction

**Outputs**:
- Function to compute coherence from scratch
- Comparison with HyPyP implementation
- Hyperscanning application example

**Prerequisites**: A03, C02

**Estimated Duration**: 60 minutes

---

### F02: Imaginary Coherence (ImCoh)

**Objective**: Measure connectivity while reducing volume conduction artifacts.

**Key Concepts**:
- Complex coherence: real and imaginary parts
- Why imaginary part rejects zero-lag
- Interpretation of imaginary coherence
- Limitations: loses some true connectivity

**Outputs**:
- Function to compute imaginary coherence
- Comparison with standard coherence
- Hyperscanning application example

**Prerequisites**: F01, C01

**Estimated Duration**: 60 minutes

---

## Block G: Phase-Based

### G01: Phase Locking Value (PLV)

**Objective**: Measure consistency of phase difference between signals.

**Key Concepts**:
- Phase synchronization concept
- PLV definition: circular mean of phase differences
- Range: 0 (no sync) to 1 (perfect sync)
- Relationship to coherence
- Sensitivity to volume conduction

**Outputs**:
- Function to compute PLV from scratch
- Comparison with HyPyP implementation
- Hyperscanning application example

**Prerequisites**: B02, C02

**Estimated Duration**: 60 minutes

---

### G02: Phase Lag Index (PLI)

**Objective**: Measure phase synchronization robust to volume conduction.

**Key Concepts**:
- PLI definition: asymmetry of phase difference distribution
- Why it rejects zero-lag (volume conduction)
- Range: 0 to 1
- Limitation: sensitive to noise near 0 and π

**Outputs**:
- Function to compute PLI from scratch
- Comparison with PLV on same data
- Hyperscanning application example

**Prerequisites**: G01, C01

**Estimated Duration**: 60 minutes

---

### G03: Weighted Phase Lag Index (wPLI)

**Objective**: Improve PLI robustness to noise.

**Key Concepts**:
- wPLI definition: weighting by magnitude of imaginary component
- Why weighting helps near zero-lag
- Comparison with PLI
- Debiased wPLI variant

**Outputs**:
- Function to compute wPLI from scratch
- Comparison with PLI
- Hyperscanning application example

**Prerequisites**: G02

**Estimated Duration**: 60 minutes

---

## Block H: Amplitude-Based

### H01: Envelope Correlation (CCorr)

**Objective**: Measure co-fluctuation of signal amplitudes.

**Key Concepts**:
- Amplitude envelope correlation
- Orthogonalization to reduce leakage
- Interpretation: amplitude coupling
- Relationship to coherence

**Outputs**:
- Function to compute envelope correlation
- Comparison with HyPyP implementation
- Hyperscanning application example

**Prerequisites**: B03, C02

**Estimated Duration**: 60 minutes

---

### H02: Power Correlation (PowCorr)

**Objective**: Measure correlation of signal power over time.

**Key Concepts**:
- Power as squared amplitude
- Power correlation vs envelope correlation
- When to use each
- Temporal resolution considerations

**Outputs**:
- Function to compute power correlation
- Comparison with envelope correlation
- Hyperscanning application example

**Prerequisites**: H01

**Estimated Duration**: 60 minutes

---

## Block G: Phase-Based

### G01: Phase Locking Value (PLV)

**Objective**: Measure consistency of phase difference between two signals.

**Key Concepts**:
- Phase synchronization concept
- PLV definition: magnitude of the mean complex phase difference
- Range: 0 (no sync) to 1 (perfect sync)
- Relationship to coherence
- Sensitivity to volume conduction (zero-lag problem)

**Outputs**:
- Function to compute PLV from scratch
- Comparison with HyPyP implementation
- Inter-brain PLV connectivity matrix

**Prerequisites**: B02b, C02

**Estimated Duration**: 60 minutes

---

### G02: Phase Lag Index (PLI)

**Objective**: Measure phase synchronization robust to volume conduction artifacts.

**Key Concepts**:
- Why PLV fails: zero-lag volume conduction
- PLI definition: asymmetry of phase difference distribution
- Why it rejects zero-lag
- Range: 0 to 1
- Limitation: sensitive to noise near 0 and π

**Outputs**:
- Function to compute PLI from scratch
- Comparison with PLV on same data (with/without simulated volume conduction)
- Inter-brain PLI connectivity matrix

**Prerequisites**: G01, C01

**Estimated Duration**: 60 minutes

---

### G03: Weighted Phase Lag Index (wPLI)

**Objective**: Improve PLI robustness to noise.

**Key Concepts**:
- Why PLI is noisy near zero phase difference
- wPLI definition: weighting by magnitude of imaginary component
- Debiased wPLI variant
- G01 → G02 → G03: a progression of refinements

**Outputs**:
- Function to compute wPLI and debiased wPLI
- Comparison G01 → G02 → G03 on same dataset
- Inter-brain wPLI connectivity matrix

**Prerequisites**: G02

**Estimated Duration**: 60 minutes

---

## Block H: Amplitude-Based

### H01: Envelope Correlation (CCorr)

**Objective**: Measure co-fluctuation of signal amplitudes between two brains.

**Key Concepts**:
- Amplitude envelope correlation
- Orthogonalization to reduce signal leakage
- Interpretation: amplitude coupling
- Relationship to coherence

**Outputs**:
- Function to compute envelope correlation (with orthogonalization)
- Comparison with HyPyP implementation
- Inter-brain CCorr connectivity matrix

**Prerequisites**: B03a, C02

**Estimated Duration**: 60 minutes

---

### H02: Power Correlation (PowCorr)

**Objective**: Measure correlation of signal power over time.

**Key Concepts**:
- Power as squared amplitude
- Power correlation vs envelope correlation
- When to use each
- Temporal resolution considerations (windowing)

**Outputs**:
- Function to compute power correlation
- Comparison with envelope correlation (CCorr)
- Inter-brain PowCorr connectivity matrix

**Prerequisites**: H01

**Estimated Duration**: 60 minutes

---

## Block I: Information-Theoretic

### I01: Mutual Information Connectivity

**Objective**: Apply MI as a non-linear connectivity metric.

**Key Concepts**:
- MI as connectivity: captures non-linear dependencies
- Comparison with PLV (linear phase synchrony)
- Estimation methods: KSG estimator for continuous signals
- Normalized MI
- Limitations: non-directional, estimation-sensitive

**Outputs**:
- Function to compute MI connectivity matrix
- Comparison MI vs PLV on same data
- Inter-brain MI connectivity matrix

**Prerequisites**: D02, C02

**Estimated Duration**: 60 minutes

---

### I02: Transfer Entropy Connectivity

**Objective**: Detect directed information flow between two brains.

**Key Concepts**:
- Directed connectivity: who influences whom?
- TE as information-theoretic Granger causality
- Asymmetric connectivity matrices
- TE vs PLI/wPLI: different aspects of coupling
- Limitations: TE ≠ causality in a strong sense, delay-sensitive

**Outputs**:
- Function to compute TE connectivity
- Demonstration of directional asymmetry (A → B ≠ B → A)
- Inter-brain asymmetric TE matrix

**Prerequisites**: D03, I01

**Estimated Duration**: 60 minutes

---

# Quick Reference: Notebook Flow

```
A01 → A02 → A03 → A04a → A04b → B01 → B02a → B02b → B03a → B03b
                                                ↓              ↓
                                               C01            H01 → H02
                                                ↓
                            A03 → C02 → C03 → E01
                                   ↓            ↓
                                  E02a          E02b → E02c
                                   ↓
                              Metrics (Part 2)

D01 → D02 → D03  (parallel track, independent)

B01 → B04a → B04b  (wavelet track, parallel to B02)

Connectivity Metrics (after E02):
  Coherence:     F01a → F01b → F02
  Phase:         G01  → G02  → G03
  Amplitude:     H01  → H02
  Info-theoretic: I01 → I02     (requires D02/D03)
```