---
title: "Block A — Signal Fundamentals"
---

# Block A — Signal Fundamentals

Before measuring connectivity between two brains, you need to understand what a signal is and how to work with it mathematically. Block A builds that foundation: from the raw voltage time-series recorded at an EEG electrode, to filtered frequency-band signals ready for connectivity analysis.

---

## Notebooks in This Block

| Notebook | Topic | Duration |
|----------|-------|----------|
| {doc}`A01_signals_and_sampling` | What is a digital signal? Sampling rate, Nyquist theorem, aliasing | ~60 min |
| {doc}`A02_frequency_domain` | Fourier transform, FFT, amplitude and phase spectra | ~60 min |
| {doc}`A03_power_spectrum_frequency_bands` | Power spectral density, Welch's method, EEG band power | ~60 min |
| {doc}`A04_filter_fundamentals` | Filter types (IIR/FIR), frequency response, filter order | ~45 min |
| {doc}`A05_applied_filtering` | Zero-phase filtering, notch filter, bandpass extraction | ~45 min |

---

## Why This Block Matters

Every EEG connectivity metric in Part 2 operates on signals that have been **sampled**, **transformed into the frequency domain**, and **bandpass filtered**. A misunderstanding at this level propagates through every subsequent analysis:

- An aliased signal at 256 Hz sampling may look like a real 10 Hz alpha oscillation while actually being muscle artifact at 246 Hz.
- Power spectrum estimation with too few data points produces a noisy, unreliable spectrum — Welch's method exists specifically to solve this.
- Filtering with a causal IIR filter introduces a phase shift that will corrupt phase-based connectivity metrics (PLV, PLI) computed in Block G.

:::warning
A04 introduces a concept with major downstream consequences: **zero-phase filtering**. Always use zero-phase (forward-backward) filtering for offline EEG analysis. The implications are revisited in {doc}`../B_phase_and_amplitude/B01_hilbert_transform`.
:::

---

## Key Concepts Introduced

**From A01:**
- Sampling rate $f_s$ and the Nyquist limit $f_{Nyquist} = f_s / 2$
- Aliasing: what happens when $f > f_{Nyquist}$
- Synthetic signal generation (used in every subsequent notebook)

**From A02:**
- The Discrete Fourier Transform (DFT) and its fast implementation (FFT)
- Frequency resolution $\Delta f = f_s / N$
- Amplitude spectrum, phase spectrum, complex spectrum

**From A03:**
- Power Spectral Density (PSD) via Welch's method
- EEG frequency bands: $\delta$ (1–4 Hz), $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma$ (30+ Hz)
- Absolute vs relative band power

**From A04–A05:**
- IIR vs FIR filter design trade-offs
- Zero-phase filtering with `filtfilt`
- Notch filtering for powerline noise (50/60 Hz)

---

## Dependencies

A01 is the entry point — no prerequisites. Each subsequent notebook builds on the previous:

```
A01 → A02 → A03
              └→ A04 → A05
```

After completing A05, continue to {doc}`../B_phase_and_amplitude/intro`.
