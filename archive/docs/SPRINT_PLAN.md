# Notebook Improvement Sprint Plan

> **Version**: 2.0
> **Created**: December 2024 | **Updated**: March 2026
> **Scope**: Notebooks A01 to I02

---

## Overview

This document outlines the sprint plan for standardizing and improving all notebooks from A01 to I02. Each notebook gets its own sprint to ensure focused, quality improvements.

> **March 2026 update**: Sprints A01–B02b completed. Scope extended to G, H, I series. Sprints F–H reclassified as **Refonte** (base existante, contenu non fonctionnel). Sprint I reclassified as **Création complète** (notebooks non existants).

### Global Tasks (Prerequisites)

Before starting individual notebook sprints:

- [x] **Fix `constants.py`**: Align `BAND_COLORS` with `colors.py` and `STYLE_GUIDE.md`
- [x] ~~**Create `scripts/generate_quick_versions.py`**~~: Deprecated — Quick versions created manually for quality
- [x] **Update `STYLE_GUIDE.md`**: Reference new `NOTEBOOK_WRITING_SPEC.md`
- [x] **Update `NOTEBOOK_WRITING_SPEC.md`**: Document cell order (Header → TOC → Imports), External Resources format with NotebookLM, manual Quick version workflow

---

## Sprint Status Legend

| Status | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔄 | In progress |
| ✅ | Completed |
| 🔀 | Needs splitting |

---

## Series A: Signal Fundamentals

### Sprint A01: Signals and Sampling ✅

**Notebook**: `A01_signals_and_sampling.ipynb` + `A01_signals_and_sampling_quick.ipynb`
**Lines**: ~955 | **Status**: ✅ Completed + Quick version done
**Needs Split**: No

#### Tasks

- [x] Add proper markdown header (title, duration, prerequisites, objectives)
- [x] Add Table of Contents
- [x] Standardize imports cell (use Path pattern)
- [x] Verify all colors use `COLORS` dict
- [x] Reformat exercises to standard template (🎯, difficulty level, collapsible)
- [x] Add External Resources section (NotebookLM video, quiz, flashcards + external videos)
- [x] Verify 5 discussion questions present
- [x] Create `A01_signals_and_sampling_quick.ipynb` manually
- [x] Final review and test all cells

#### External Resources Added

- [x] NotebookLM: Video Overview, Quiz, Flashcards
- [x] 3Blue1Brown: "But what is the Fourier Transform?"
- [x] Sampling, Aliasing & Nyquist Theorem video
- [x] Wikipedia: Nyquist-Shannon sampling theorem
- [x] SciPy signal processing tutorial

---

### Sprint A02: Frequency Domain ✅

**Notebook**: `A02_frequency_domain.ipynb` + `A02_frequency_domain_quick.ipynb`
**Lines**: ~1312 | **Status**: ✅ Completed + Quick version done
**Needs Split**: No

#### Tasks

- [x] Add proper markdown header (moved to top, before TOC)
- [x] Add Table of Contents (moved after header, before imports)
- [x] Standardize imports cell
- [x] Verify all colors use `COLORS` dict
- [x] Reformat 3 exercises to standard template (🎯 + difficulty)
- [x] Add External Resources section (NotebookLM + unique videos/reading)
- [x] Verify 5 discussion questions
- [x] Create `A02_frequency_domain_quick.ipynb` manually
- [x] Final review

#### External Resources Added

- [x] NotebookLM: Video Overview, Quiz, Flashcards
- [x] Veritasium: The Remarkable Story Behind FFT
- [x] Steve Brunton: The Fast Fourier Transform (FFT)
- [x] Wikipedia: Fast Fourier Transform
- [x] BetterExplained: Interactive Guide to Fourier Transform

---

### Sprint A03: Power Spectrum and Frequency Bands ✅

**Notebook**: `A03_power_spectrum_frequency_bands.ipynb` + `A03_power_spectrum_frequency_bands_quick.ipynb`
**Lines**: ~1845 | **Status**: ✅ Completed + Quick version done
**Needs Split**: No (but close to threshold)

#### Tasks

- [x] Verify/update markdown header
- [x] Add/update Table of Contents
- [x] Standardize imports cell
- [x] **FIX**: Removed inline `EEG_BANDS` — now imported from `src.constants`
- [x] **FIX**: Removed inline `BAND_COLORS` — now imported from `src.constants`
- [x] Reformat exercises to standard template
- [x] Add External Resources section
- [x] Verify 5 discussion questions
- [x] Generate quick version
- [x] Final review

---

### Sprint A04: Filtering Signals ✅

**Notebooks**: `A04a_filter_fundamentals.ipynb` + quick | `A04b_applied_filtering.ipynb` + quick
**Lines**: split en deux | **Status**: ✅ Split + Quick versions done
**Needs Split**: Done

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| A04a_filter_design.ipynb | Filter types, FIR vs IIR, design principles | ~1300 |
| A04b_filter_application.ipynb | Applying filters, edge effects, practical considerations | ~1300 |

#### Tasks for A04a (Filter Design)

- [ ] Create new notebook with proper header "(Part 1 of 2)"
- [ ] Add Table of Contents
- [ ] Move sections: Filter types, FIR design, IIR design
- [ ] Keep exercises 1-2 (filter design focused)
- [ ] Add transition to Part 2
- [ ] Add External Resources section
- [ ] Add Summary specific to Part 1
- [ ] Generate quick version

#### Tasks for A04b (Filter Application)

- [ ] Create new notebook with proper header "(Part 2 of 2)"
- [ ] Reference Part 1 in intro
- [ ] Add Table of Contents
- [ ] Move sections: Applying filters, edge effects, frequency response analysis
- [ ] Keep exercises 3-4 (application focused)
- [ ] Add External Resources section
- [ ] Add Summary and Discussion questions
- [ ] Generate quick version

#### External Resources to Add

- [ ] Scipy signal filtering documentation
- [ ] MNE-Python: Filtering tutorial
- [ ] "Don't filter when you can avoid it" (paper on filter artifacts)

---

## Series B: Phase and Amplitude

### Sprint B01: Hilbert Transform ✅

**Notebooks**: `B01_hilbert_transform.ipynb` + `B01_hilbert_transform_quick.ipynb`
**Lines**: ~1932 | **Status**: ✅ Completed + Quick version done
**Needs Split**: No

#### Tasks

- [x] Verify/update markdown header
- [x] Add/update Table of Contents
- [x] Standardize imports
- [x] Verify all colors use `COLORS` dict
- [x] Reformat exercises to standard template
- [x] Add External Resources section + NotebookLM links
- [x] Verify 5 discussion questions
- [x] Generate quick version
- [x] Final review

---

### Sprint B02: Working with Phase ✅

**Notebooks**: `B02a_circular_statistics.ipynb` + quick | `B02b_phase_relationships.ipynb` + quick
**Lines**: split en deux | **Status**: ✅ Split + Quick versions done
**Needs Split**: Done

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| B02a_circular_statistics.ipynb | Phase representation, circular mean, circular variance | ~1200 |
| B02b_phase_relationships.ipynb | Phase difference, synchronization measures intro | ~1200 |

#### Tasks for B02a (Circular Statistics)

- [ ] Create new notebook "(Part 1 of 2)"
- [ ] Table of Contents
- [ ] Focus: Phase as circular variable, circular stats
- [ ] Keep relevant exercises
- [ ] External Resources section
- [ ] Generate quick version

#### Tasks for B02b (Phase Relationships)

- [ ] Create new notebook "(Part 2 of 2)"
- [ ] Table of Contents
- [ ] Focus: Phase differences, intro to synchronization
- [ ] Keep relevant exercises
- [ ] External Resources section
- [ ] Generate quick version

---

### Sprint B03: Amplitude Envelope 🔀

**Notebook**: `B03_amplitude_envelope.ipynb`  
**Lines**: ~2318 | **Status**: Needs splitting  
**Needs Split**: **YES** (>2200 lines)

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| B03a_envelope_extraction.ipynb | Hilbert envelope, filtering for envelope | ~1150 |
| B03b_envelope_correlation.ipynb | Envelope correlations, amplitude coupling | ~1150 |

#### Tasks

- [ ] Split according to plan
- [ ] Standardize both parts
- [ ] Remove any inline color modifications
- [ ] Generate quick versions

---

### Sprint B04: Wavelets and Time-Frequency 🔀

**Notebook**: `B04_wavelets_time_frequency.ipynb`  
**Lines**: ~2456 | **Status**: Has TOC, needs splitting  
**Needs Split**: **YES** (>2200 lines)

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| B04a_stft_spectrograms.ipynb | STFT, spectrograms, time-frequency tradeoff | ~1200 |
| B04b_wavelet_analysis.ipynb | Morlet wavelets, CWT, scalograms | ~1200 |

#### Tasks

- [ ] Split according to plan
- [ ] Already has good TOC structure (preserve)
- [ ] Standardize both parts
- [ ] Generate quick versions

---

## Series C: Connectivity Concepts

### Sprint C01: Volume Conduction ⬜

**Notebook**: `C01_volume_conduction.ipynb`  
**Lines**: ~2064 | **Status**: Complete, within limits  
**Needs Split**: No (close to threshold, review during sprint)

#### Tasks

- [ ] Verify/update header
- [ ] Add/update Table of Contents
- [ ] Remove color aliases (use `COLORS` directly)
- [ ] Reformat exercises to standard template
- [ ] Add External Resources section
- [ ] Verify 5 discussion questions
- [ ] Add `[QUICK_VERSION: ...]` markers
- [ ] Generate quick version
- [ ] Final review

#### External Resources to Add

- [ ] Nunez & Srinivasan: EEG source imaging
- [ ] FieldTrip: Volume conduction tutorial
- [ ] Paper: "Volume conduction in EEG"

---

### Sprint C02: Connectivity Matrices 🔀

**Notebook**: `C02_connectivity_matrices.ipynb`  
**Lines**: ~2567 | **Status**: Needs splitting  
**Needs Split**: **YES** (>2200 lines)

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| C02a_matrix_computation.ipynb | Building connectivity matrices, different metrics | ~1300 |
| C02b_matrix_visualization.ipynb | Visualization, interpretation, hyperscanning matrices | ~1300 |

#### Tasks

- [ ] Split according to plan
- [ ] Standardize both parts
- [ ] Generate quick versions

---

### Sprint C03: Statistical Significance ⬜

**Notebook**: `C03_statistical_significance.ipynb`  
**Lines**: ~2180 | **Status**: Within limits (barely)  
**Needs Split**: No

#### Tasks

- [ ] Verify/update header
- [ ] Add/update Table of Contents
- [ ] Standardize imports
- [ ] Verify colors
- [ ] Reformat exercises
- [ ] Add External Resources section
- [ ] Add `[QUICK_VERSION: ...]` markers
- [ ] Generate quick version
- [ ] Final review

#### External Resources to Add

- [ ] "Statistical approaches for EEG-based connectivity"
- [ ] Maris & Oostenveld: Cluster permutation testing
- [ ] MNE-Python: Statistical analysis tutorial

---

## Series D: Information Theory

### Sprint D01: Entropy and Information ⬜

**Notebook**: `D01_entropy_information.ipynb`  
**Lines**: ~1543 | **Status**: Complete, good  
**Needs Split**: No

#### Tasks

- [ ] Verify/update header
- [ ] Add Table of Contents
- [ ] Standardize imports
- [ ] Verify colors
- [ ] Reformat exercises
- [ ] Add External Resources section
- [ ] Generate quick version
- [ ] Final review

#### External Resources to Add

- [ ] Shannon: "A Mathematical Theory of Communication"
- [ ] 3Blue1Brown: Information theory basics
- [ ] Wikipedia: Entropy (information theory)

---

### Sprint D02: Mutual Information ⬜

**Notebook**: `D02_mutual_information.ipynb`  
**Lines**: ~1906 | **Status**: Complete  
**Needs Split**: No

#### Tasks

- [ ] Verify/update header
- [ ] Add Table of Contents
- [ ] Standardize imports
- [ ] Verify colors
- [ ] Reformat exercises
- [ ] Add External Resources section
- [ ] Generate quick version
- [ ] Final review

---

### Sprint D03: Transfer Entropy ⬜

**Notebook**: `D03_transfer_entropy.ipynb`  
**Lines**: ~1819 | **Status**: Complete  
**Needs Split**: No

#### Tasks

- [ ] Verify/update header
- [ ] Add Table of Contents
- [ ] Standardize imports
- [ ] Verify colors
- [ ] Reformat exercises
- [ ] Add External Resources section
- [ ] Generate quick version
- [ ] Final review

---

## Series E: Hyperscanning Context

### Sprint E01: Graph Theory Basics ⬜

**Notebook**: `E01_graph_theory_basics.ipynb`  
**Lines**: ~2272 | **Status**: At threshold, evaluate during sprint  
**Needs Split**: Evaluate (exactly at 2200 threshold)

#### Tasks

- [ ] **First**: Evaluate if split is needed
- [ ] If split: E01a (Graph concepts) / E01b (Graph metrics)
- [ ] If not: Complete standard tasks
- [ ] Add Table of Contents
- [ ] Standardize all formatting
- [ ] Generate quick version

---

### Sprint E02: Introduction to Hyperscanning 🔀

**Notebook**: `E02_introduction_hyperscanning.ipynb`  
**Lines**: ~2854 | **Status**: **LONGEST** - Needs splitting  
**Needs Split**: **YES** (highest priority for split)

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| E02a_hyperscanning_concepts.ipynb | What is hyperscanning, paradigms, history | ~1000 |
| E02b_hyperscanning_data.ipynb | Data structures, preprocessing, analysis approaches | ~1000 |
| E02c_hyperscanning_analysis.ipynb | Analysis pipelines, common metrics overview | ~800 |

#### Tasks

- [ ] Split into 3 parts (given length)
- [ ] Each part gets proper header "(Part X of 3)"
- [ ] Maintain pedagogical flow across parts
- [ ] Generate quick versions for all

---

## Series F: Coherence-Based Metrics

### Sprint F01: Spectral Coherence ⬜

**Notebook**: `F01_spectral_coherence.ipynb`  
**Lines**: ~2311 | **Status**: Needs splitting  
**Needs Split**: **YES** (>2200 lines)

#### Split Plan

| New Notebook | Content | Estimated Lines |
|--------------|---------|-----------------|
| F01a_coherence_theory.ipynb | What is coherence, mathematical foundation | ~1150 |
| F01b_coherence_practice.ipynb | Computing coherence, interpretation, hyperscanning | ~1150 |

#### Tasks

- [ ] Split according to plan
- [ ] Already uses imports from src/coherence.py (good!)
- [ ] Reformat 7 exercises across both parts
- [ ] Generate quick versions

---

### Sprint F02: Imaginary Coherence ⬜

**Notebook**: `F02_imaginary_coherence.ipynb`  
**Lines**: ~2072 | **Status**: Within limits  
**Needs Split**: No

#### Tasks

- [ ] Verify/update header
- [ ] Add Table of Contents
- [ ] Verify imports and colors
- [ ] Reformat 7 exercises (uses commented solutions - convert to collapsible)
- [ ] Add External Resources section
- [ ] Generate quick version
- [ ] Final review

---

## Sprint Execution Order

### Phase 1: Foundation (Sprints 1-4)

| Sprint # | Notebook | Priority | Reason |
|----------|----------|----------|--------|
| 0 | Global fixes | 🔴 Critical | Fix constants.py, create script |
| 1 | A01 | 🟢 Easy | Short, foundation for all |
| 2 | A02 | 🟢 Easy | Short, frequently referenced |
| 3 | A03 | 🟡 Medium | Fix color issues |
| 4 | A04 | 🔴 Complex | First split |

### Phase 2: Phase & Amplitude (Sprints 5-8)

| Sprint # | Notebook | Priority | Reason |
|----------|----------|----------|--------|
| 5 | B01 | 🟢 Easy | Standard formatting only |
| 6 | B02 | 🔴 Complex | Split required |
| 7 | B03 | 🔴 Complex | Split required |
| 8 | B04 | 🔴 Complex | Split required |

### Phase 3: Connectivity (Sprints 9-11)

| Sprint # | Notebook | Priority | Reason |
|----------|----------|----------|--------|
| 9 | C01 | 🟢 Easy | Near threshold but OK |
| 10 | C02 | 🔴 Complex | Split required |
| 11 | C03 | 🟢 Easy | Standard formatting |

### Phase 4: Information Theory (Sprints 12-14)

| Sprint # | Notebook | Priority | Reason |
|----------|----------|----------|--------|
| 12 | D01 | 🟢 Easy | Standard formatting |
| 13 | D02 | 🟢 Easy | Standard formatting |
| 14 | D03 | 🟢 Easy | Standard formatting |

### Phase 5: Hyperscanning & Coherence (Sprints 15-18)

| Sprint # | Notebook | Priority | Reason |
|----------|----------|----------|--------|
| 15 | E01 | 🟡 Medium | Evaluate split need |
| 16 | E02 | 🔴 Complex | **3-way split** (longest) |
| 17 | F01 | 🔴 Complex | Split required |
| 18 | F02 | 🟢 Easy | Standard formatting |

---

## Estimated Effort

| Category | Count | Est. Time per Sprint |
|----------|-------|---------------------|
| 🟢 Easy (no split) | 10 | 1-2 hours |
| 🟡 Medium | 2 | 2-3 hours |
| 🔴 Complex (split) | 7 | 3-5 hours |

**Total Notebooks**: 18 original → ~28 after splits  
**Estimated Total Time**: 40-60 hours

---

---

## Series G: Phase-Based Metrics

> **Mode** : Refonte — base existante mais contenu non fonctionnel, à réécrire.

### Sprint G01: Phase Locking Value (PLV) ⬜

**Notebook**: `G01_phase_locking_value.ipynb`
**Status**: ⬜ Refonte requise (ne compile pas, contenu incorrect)

#### Tasks

- [ ] Audit base existante — identifier sections récupérables
- [ ] Réécrire implémentation from scratch : `np.abs(np.mean(np.exp(1j * phase_diff)))`
- [ ] Comparaison HyPyP `hypyp.analyses.pair_connectivity()`
- [ ] Application hyperscanning : matrice inter-brain PLV
- [ ] Relier à C01 : PLV est sensible au volume conduction zero-lag
- [ ] Exercices standardisés (🎯 + difficulté + solution cachée)
- [ ] External Resources section
- [ ] Quick version

---

### Sprint G02: Phase Lag Index (PLI) ⬜

**Notebook**: `G02_phase_lag_index.ipynb`
**Status**: ⬜ Refonte requise

#### Tasks

- [ ] Build on G01 : pourquoi PLV échoue face au volume conduction
- [ ] Réécrire : `np.abs(np.mean(np.sign(np.imag(cross_spectrum))))`
- [ ] Comparaison PLV vs PLI sur données simulées avec volume conduction
- [ ] Exercices, External Resources, quick version

---

### Sprint G03: Weighted Phase Lag Index (wPLI) ⬜

**Notebook**: `G03_weighted_phase_lag_index.ipynb`
**Status**: ⬜ Refonte requise

#### Tasks

- [ ] Build on G02 : pourquoi PLI est bruitée près de 0 et π
- [ ] Réécrire debiased wPLI
- [ ] Comparaison G01 → G02 → G03 sur même dataset (progression logique)
- [ ] Exercices, External Resources, quick version

---

## Series H: Amplitude-Based Metrics

> **Mode** : Refonte — base existante mais contenu non fonctionnel, à réécrire.

### Sprint H01: Envelope Correlation (CCorr) ⬜

**Notebook**: `H01_envelope_correlation.ipynb`
**Status**: ⬜ Refonte requise

#### Tasks

- [ ] Build on B03 : enveloppe extraite → corrélation inter-brain
- [ ] Orthogonalisation pour réduire le leakage (signal source amplifié)
- [ ] Comparaison HyPyP `hypyp.analyses.pair_connectivity()`
- [ ] Application hyperscanning : matrice inter-brain CCorr
- [ ] Exercices, External Resources, quick version

---

### Sprint H02: Power Correlation (PowCorr) ⬜

**Notebook**: `H02_power_correlation.ipynb`
**Status**: ⬜ Refonte requise

#### Tasks

- [ ] Distinction power vs amplitude (squared envelope)
- [ ] Comparaison CCorr vs PowCorr : quand utiliser lequel
- [ ] Considérations de résolution temporelle (fenêtrage)
- [ ] Exercices, External Resources, quick version

---

## Series I: Information-Theoretic Connectivity

> **Mode** : Création complète — notebooks non existants, à créer de zéro.

### Sprint I01: Mutual Information Connectivity ⬜

**Notebook**: `I01_mutual_information_connectivity.ipynb` (à créer)
**Prérequis**: D02 (MI) + C02 (matrices)
**Status**: ⬜ Création complète

#### Tasks

- [ ] Créer notebook selon NOTEBOOK_TEMPLATE.md
- [ ] Application MI comme métrique de connectivité non-linéaire
- [ ] Comparaison MI vs PLV (linéaire vs non-linéaire)
- [ ] Estimation via KSG estimator (plus robuste que binning pour signaux continus)
- [ ] Application hyperscanning : MI inter-brain
- [ ] Avantages/limites : non-linéaire, non-directionnel, sensible à l'estimation
- [ ] Exercices, External Resources, quick version

---

### Sprint I02: Transfer Entropy Connectivity ⬜

**Notebook**: `I02_transfer_entropy_connectivity.ipynb` (à créer)
**Prérequis**: D03 (TE) + I01 (MI connectivity)
**Status**: ⬜ Création complète

#### Tasks

- [ ] Créer notebook selon NOTEBOOK_TEMPLATE.md
- [ ] TE pour connectivité **dirigée** : qui influence qui ?
- [ ] Comparaison avec PLI/wPLI (phase) : TE capture des structures différentes
- [ ] Démonstration asymétrie directionnelle sur données simulées (signal A → B)
- [ ] Application hyperscanning : matrice asymétrique inter-brain TE
- [ ] Mise en garde : TE ≠ causalité au sens fort, sensible aux délais
- [ ] Exercices, External Resources, quick version

---

## Final Deliverables

After all sprints complete:

1. **~44 standardized tutorial notebooks** (A01 to I02, with all splits)
2. **~44 quick version notebooks** (created manually per spec)
3. **Updated `NOTEBOOKS_INDEX.md`** reflecting full structure A01→I02
4. **Updated `README.md`** landing page
5. All notebooks execute without errors (restart & run all verified)

---

## Appendix: New Notebook Structure After Splits

```
notebooks/
├── 01_foundations/
│   ├── A_signal_fundamentals/
│   │   ├── A01_signals_and_sampling.ipynb
│   │   ├── A01_signals_and_sampling_quick.ipynb
│   │   ├── A02_frequency_domain.ipynb
│   │   ├── A02_frequency_domain_quick.ipynb
│   │   ├── A03_power_spectrum_frequency_bands.ipynb
│   │   ├── A03_power_spectrum_frequency_bands_quick.ipynb
│   │   ├── A04a_filter_design.ipynb
│   │   ├── A04a_filter_design_quick.ipynb
│   │   ├── A04b_filter_application.ipynb
│   │   └── A04b_filter_application_quick.ipynb
│   ├── B_phase_and_amplitude/
│   │   ├── B01_hilbert_transform.ipynb
│   │   ├── B01_hilbert_transform_quick.ipynb
│   │   ├── B02a_circular_statistics.ipynb
│   │   ├── B02a_circular_statistics_quick.ipynb
│   │   ├── B02b_phase_relationships.ipynb
│   │   ├── B02b_phase_relationships_quick.ipynb
│   │   ├── B03a_envelope_extraction.ipynb
│   │   ├── B03a_envelope_extraction_quick.ipynb
│   │   ├── B03b_envelope_correlation.ipynb
│   │   ├── B03b_envelope_correlation_quick.ipynb
│   │   ├── B04a_stft_spectrograms.ipynb
│   │   ├── B04a_stft_spectrograms_quick.ipynb
│   │   ├── B04b_wavelet_analysis.ipynb
│   │   └── B04b_wavelet_analysis_quick.ipynb
│   ├── C_connectivity_concepts/
│   │   ├── C01_volume_conduction.ipynb
│   │   ├── C01_volume_conduction_quick.ipynb
│   │   ├── C02a_matrix_computation.ipynb
│   │   ├── C02a_matrix_computation_quick.ipynb
│   │   ├── C02b_matrix_visualization.ipynb
│   │   ├── C02b_matrix_visualization_quick.ipynb
│   │   ├── C03_statistical_significance.ipynb
│   │   └── C03_statistical_significance_quick.ipynb
│   ├── D_information_theory/
│   │   ├── D01_entropy_information.ipynb
│   │   ├── D01_entropy_information_quick.ipynb
│   │   ├── D02_mutual_information.ipynb
│   │   ├── D02_mutual_information_quick.ipynb
│   │   ├── D03_transfer_entropy.ipynb
│   │   └── D03_transfer_entropy_quick.ipynb
│   └── E_hyperscanning_context/
│       ├── E01_graph_theory_basics.ipynb (or split)
│       ├── E01_graph_theory_basics_quick.ipynb
│       ├── E02a_hyperscanning_concepts.ipynb
│       ├── E02a_hyperscanning_concepts_quick.ipynb
│       ├── E02b_hyperscanning_data.ipynb
│       ├── E02b_hyperscanning_data_quick.ipynb
│       ├── E02c_hyperscanning_analysis.ipynb
│       └── E02c_hyperscanning_analysis_quick.ipynb
└── 02_connectivity_metrics/
    └── F_coherence_based/
        ├── F01a_coherence_theory.ipynb
        ├── F01a_coherence_theory_quick.ipynb
        ├── F01b_coherence_practice.ipynb
        ├── F01b_coherence_practice_quick.ipynb
        ├── F02_imaginary_coherence.ipynb
        └── F02_imaginary_coherence_quick.ipynb
```

---

*End of Sprint Plan*
