---
title: "Part 2: Connectivity Metrics"
---

# Part 2 — Connectivity Metrics

With the foundations in place, Part 2 implements the actual inter-brain connectivity metrics used in hyperscanning research. Each block covers a family of metrics sharing a common theoretical basis.

---

## Blocks at a Glance

| Block | Topic | Notebooks | Key metric(s) |
|-------|-------|-----------|---------------|
| **F** | Coherence-Based | F01a–F02 | Spectral coherence, imaginary coherence |
| **G** | Phase-Based | G01–G03 | PLV, PLI, wPLI |
| **H** | Amplitude-Based | H01–H02 | Envelope correlation, power correlation |
| **I** | Information-Theoretic | I01–I02 | MI connectivity, transfer entropy |

---

## Choosing a Metric

No single metric is universally best. The choice depends on your question:

| If you want to measure… | Use |
|-------------------------|-----|
| Broadband spectral coupling | Coherence (F01) |
| Phase coupling robust to volume conduction | PLI or wPLI (G02–G03) |
| Amplitude coupling | Envelope correlation (H01) |
| Non-linear dependency | Mutual information (I01) |
| Directed influence (who drives whom) | Transfer entropy (I02) |

:::warning
Volume conduction inflates connectivity estimates for all metrics. Always read {doc}`../01_foundations/C_connectivity_concepts/C01_volume_conduction` before interpreting results.
:::

---

## Validation Strategy

Every notebook in Part 2 validates the from-scratch implementation against the `src/` library equivalent, and where applicable, against [HyPyP](https://github.com/ppsp-team/HyPyP). Results should match to numerical precision.

---

## Prerequisites

All of Part 1, especially:
- {doc}`../01_foundations/B_phase_and_amplitude/B02a_circular_statistics` (phase statistics)
- {doc}`../01_foundations/C_connectivity_concepts/C01_volume_conduction` (why some metrics exist)
- {doc}`../01_foundations/C_connectivity_concepts/C02a_connectivity_matrices` (matrix representation)
