---
title: "Part 1: Foundations"
---

# Part 1 — Foundations

Before computing connectivity between two EEG signals, you need to understand the signals themselves. Part 1 builds that foundation block by block, from raw time-series to the conceptual tools needed for inter-brain analysis.

---

## Blocks at a Glance

| Block | Topic | Notebooks | Key skill |
|-------|-------|-----------|-----------|
| **A** | Signal Fundamentals | A01–A04b | Sampling, FFT, filtering |
| **B** | Phase & Amplitude | B01–B04b | Hilbert transform, PLV intuition |
| **C** | Connectivity Concepts | C01–C03 | Volume conduction, matrices, statistics |
| **D** | Information Theory | D01–D03 | Entropy, mutual information, TE |
| **E** | Hyperscanning Context | E01–E02c | Graph theory, dual-brain structure |

---

## Why This Order?

**Block A** gives you the vocabulary: what a signal is, how sampling works, what the FFT reveals, and how filtering shapes a signal's spectrum.

**Block B** goes deeper into the temporal structure of oscillations. The Hilbert transform extracts instantaneous phase — the backbone of most connectivity metrics.

**Block C** introduces the unique challenges of EEG connectivity: volume conduction (signals bleeding across the scalp), how to represent pairwise relationships as matrices, and how to test for significance.

**Block D** covers a different statistical language — information theory — that describes dependencies without assuming linearity.

**Block E** puts everything in the hyperscanning frame: what it means to record two brains simultaneously, how to structure dual-brain data, and how graph theory helps interpret connectivity patterns.

---

## Before You Start

:::note
Make sure you have completed the setup described in {doc}`../how_to_use`. Start with {doc}`A_signal_fundamentals/A01_signals_and_sampling` if you are new to the series.
:::
