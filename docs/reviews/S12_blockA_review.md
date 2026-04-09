# S12 — Block A Review Report

**Date**: 2026-04-09  
**Notebooks**: A01–A04b  
**Reviewers**: Scientific agent + Science mediation agent

---

## Scientific Review — Summary

### HIGH Priority (addressed in S12)

| ID | Notebook | Issue |
|----|----------|-------|
| H1 | A01 | Aliasing formula `\|f - k*f_s\|` inconsistent with code; misleads on manual application |
| H2 | A02 | Normalization text "divide by N or N/2" ambiguous; code uses 2/N correctly |
| H3 | A03 | Delta band 1.0 Hz stated as canonical; common convention is 0.5 Hz |
| H4 | A01 | "256 Hz is sufficient for gamma" lacks anti-aliasing margin context |

### MEDIUM Priority (to address in content sprints)

| ID | Notebook | Issue |
|----|----------|-------|
| M1 | A02 | Δf = 1/T is bin spacing, not resolving power; windowing effect missing |
| M2 | A03 | nperseg > signal length causes silent zero-padding; not flagged |
| M3 | A04b | filtfilt -3 dB cutoff shift not mentioned |
| M4 | A03 | Minimum K segments for variance reduction not stated (K ≥ 8 rule) |
| M5 | A04a | IIR feedback/FIR feedforward structural distinction missing from narrative |

### LOW Priority

| ID | Notebook | Issue |
|----|----------|-------|
| L1 | A03 | Welch (1967) citation absent |
| L2 | A02 | Cooley-Tukey (1965) citation absent |
| L3 | A01/A03 | Gamma upper bound 100 Hz; high-gamma not mentioned |

---

## Science Mediation Review — Summary

### BLOCKING (addressed in S12)

| ID | Notebook | Issue |
|----|----------|-------|
| B1 | A02 | Complex numbers lack visual scaffolding (Argand plane); blocks all phase-based connectivity understanding |
| B2 | A04a | Opens with filter plots before plain-language definition; dB used before defined |
| B3 | A04a | Phase response consequences for connectivity not explained before A04b |
| B4 | A03 | Stationarity assumption underlying Welch never stated |

### IMPROVEMENT (to address in content sprints)

| ID | Notebook | Issue |
|----|----------|-------|
| I1 | A01 | "Anti-aliasing filter / low-pass filter" used without definition |
| I2 | A02 | Windowing mechanism not explained (only effect shown) |
| I3 | A03 | dB reference choices: different refs produce incomparable values — not warned |
| I4 | A03 | Exercises lack difficulty labels (🟢/🟡/🔴) |
| I5 | A02 | Visualization 7 (sync vs desync) has no follow-up prose cell |
| I6 | A04b | Only 2 exercises; suggest adding PLV phase-difference exercise |

### Strengths (to preserve)

- A01/A02: Intuition before formalism — well-structured
- A01: Aliasing "frequency folding" diagram is genuinely strong
- A02: Phase and connectivity motivated early (section 6, Viz 7)
- A03: Eyes-open vs eyes-closed alpha example ideal for audience
- A04b: Hyperscanning rationale explicit at each decision point
- All: Collapsible solution cells well-chosen

---

## Fixes Applied in S12

- H1: Added folding-rule clarification to A01 aliasing section
- H2: Fixed FFT normalization text in A02 to match code (2/N)
- H3: Added 0.5 Hz convention caveat to A03 delta band definition
- H4: Added anti-aliasing margin context to A01
- B1: Added Argand plane explanation paragraph to A02 complex numbers section
- B3: Added phase-response consequences note to A04a
- B4: Added stationarity sentence to A03 Welch section
