---
title: EEG Connectivity Metrics — A Hyperscanning Tutorial Series
---

# EEG Connectivity Metrics
## A Hyperscanning Tutorial Series

This is a comprehensive workshop series on **EEG connectivity analysis for hyperscanning** — from signal processing fundamentals to advanced inter-brain connectivity metrics.

Designed for graduate students and researchers in neuroscience, psychology, and biomedical engineering at the [PPSP Lab, CHU Sainte-Justine](https://ppsp-lab.com). The series assumes basic familiarity with EEG and Python. No prior signal processing expertise required.

---

## What You Will Learn

By the end of this series you will be able to:

- Decompose EEG signals into frequency components and extract band power
- Apply filtering, compute the Hilbert transform, and work with instantaneous phase
- Understand why **volume conduction** is a fundamental challenge in EEG connectivity
- Implement from scratch: **PLV, PLI, wPLI, coherence, imaginary coherence, envelope correlation, mutual information, and transfer entropy**
- Apply these metrics in a **hyperscanning** (dual-brain) context
- Validate results with the [HyPyP](https://github.com/ppsp-team/HyPyP) library

---

## Structure

The series is divided into two parts:

::::{grid} 1 1 2 2
:::{card} Part 1 — Foundations
:link: 01_foundations/intro
:link-type: doc
Blocks A through E: signal processing, phase analysis, connectivity concepts, information theory, and hyperscanning context.
:::
:::{card} Part 2 — Connectivity Metrics
:link: 02_connectivity_metrics/intro
:link-type: doc
Blocks F through I: coherence, PLV/PLI/wPLI, amplitude coupling, and information-theoretic connectivity.
:::
::::

---

## Pedagogical Approach

- **Intuition first, formalism second.** Every concept starts with a concrete synthetic example before introducing equations.
- **Implement from scratch.** You write the metric before using the library version.
- **Progressive complexity.** Each block builds on the previous one — see the dependency map in {doc}`how_to_use`.
- **Optional deep dives.** Sections marked `[Optional Deep Dive]` go deeper into mathematics. They can be skipped without losing the thread.
- **Discussion questions.** Every notebook ends with questions for the live session.

---

## Prerequisites

| Topic | Level |
|-------|-------|
| Python (numpy, matplotlib) | Intermediate |
| EEG basics (electrodes, frequency bands) | Introductory |
| Signal processing | None required |
| Statistics | Introductory |

---

## How to Use This Book

See {doc}`how_to_use` for navigation tips, local setup, and the notebook dependency map.
