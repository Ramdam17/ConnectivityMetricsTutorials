# Workshop Context — Hyperscanning Educational Series

## Project Overview
Educational workshop series for the Social Neuro AI Lab, specializing in hyperscanning.
Target: weekly peer-learning sessions (1 hour), notebooks shared 2-3 days before for pre-reading.

## Project Structure

```
hyperscanning_workshop/
│
├── notebooks/
│   │
│   ├── 01_foundations/
│   │   │
│   │   ├── A_signal_fundamentals/
│   │   │   ├── A01_signals_and_sampling.ipynb
│   │   │   ├── A02_frequency_domain.ipynb
│   │   │   ├── A03_power_spectrum_frequency_bands.ipynb
│   │   │   └── A04_filtering_signals.ipynb
│   │   │
│   │   ├── B_phase_and_amplitude/
│   │   │   ├── B01_hilbert_transform.ipynb
│   │   │   ├── B02_working_with_phase.ipynb
│   │   │   ├── B03_amplitude_envelope.ipynb
│   │   │   └── B04_wavelets_time_frequency.ipynb
│   │   │
│   │   ├── C_connectivity_concepts/
│   │   │   ├── C01_volume_conduction_problem.ipynb
│   │   │   ├── C02_connectivity_matrices.ipynb
│   │   │   └── C03_statistical_inference.ipynb
│   │   │
│   │   ├── D_information_theory/
│   │   │   ├── D01_entropy_basics.ipynb
│   │   │   ├── D02_mutual_information.ipynb
│   │   │   └── D03_transfer_entropy_and_causality.ipynb
│   │   │
│   │   └── E_hyperscanning_context/
│   │       ├── E01_graph_theory_basics.ipynb
│   │       └── E02_introduction_to_hyperscanning.ipynb
│   │
│   └── 02_connectivity_metrics/
│       │
│       ├── F_coherence_based/
│       │   ├── F01_spectral_coherence.ipynb
│       │   └── F02_imaginary_coherence.ipynb
│       │
│       ├── G_phase_based/
│       │   ├── G01_phase_locking_value.ipynb
│       │   ├── G02_phase_lag_index.ipynb
│       │   └── G03_weighted_phase_lag_index.ipynb
│       │
│       ├── H_amplitude_based/
│       │   ├── H01_envelope_correlation.ipynb
│       │   └── H02_power_correlation.ipynb
│       │
│       └── I_information_theoretic/
│           └── (future notebooks)
│
├── src/
│   ├── __init__.py
│   ├── colors.py
│   ├── plotting.py
│   └── signals.py
│
├── docs/
│   ├── CONTEXT.md
│   ├── NOTEBOOK_TEMPLATE.md
│   ├── NOTEBOOKS_INDEX.md
│   ├── STYLE_GUIDE.md
│   └── sprints/
│       ├── A01_signals_and_sampling.md
│       ├── A02_frequency_domain.md
│       └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── copilot-instructions.md
└── pyproject.toml
```

## Audience
- International team (English only)
- Mixed backgrounds: psychology, mathematics, engineering, neuroscience
- Assumed knowledge: signals basics, EEG fundamentals (electrodes, frequency bands)
- No assumed knowledge: signal processing, connectivity metrics, information theory

## Pedagogical Approach
- Build intuition first, formalism second
- Start with synthetic signals, progress to real hyperscanning data
- Implement from scratch before using HyPyP
- Optional deep dives for mathematical details
- End with discussion questions for live session

## Data Structure
- Dyad: 2 EEG recordings from interacting participants
- Control: 1 EEG recording from unrelated participant (surrogate comparison)
- Data provided later; use synthetic signals until then

## Technical Stack
- Python 3.11+ with type hints
- Poetry for dependency management
- Core: numpy, scipy, matplotlib, mne, hypyp
- Style: see STYLE_GUIDE.md
- Linting: ruff, mypy (pre-commit hooks)

## Workshop Structure

### Part 1: Foundations (16 notebooks)
| Block | Topic | Notebooks |
|-------|-------|-----------|
| A | Signal Fundamentals | A01-A04 |
| B | Phase and Amplitude | B01-B04 |
| C | Connectivity Concepts | C01-C03 |
| D | Information Theory | D01-D03 |
| E | Hyperscanning Context | E01-E02 |

### Part 2: Connectivity Metrics (7+ notebooks)
| Block | Topic | Notebooks |
|-------|-------|-----------|
| F | Coherence-based | F01-F02 |
| G | Phase-based | G01-G03 |
| H | Amplitude-based | H01-H02 |
| I | Information-theoretic | Future |

## Notebook Template Structure
1. Header: title, duration (~1h), learning objectives, prerequisites
2. Imports
3. Sections with pattern: Concept (markdown) → Implementation (code) → Interpretation (markdown)
4. Optional deep dives marked explicitly
5. Comparison with HyPyP implementation
6. Hyperscanning application example
7. Summary: key takeaways
8. Discussion: 3-5 questions for live session

## Dependencies Between Notebooks

### Block A (Signal Fundamentals)
- A01: standalone (entry point)
- A02: requires A01
- A03: requires A02
- A04: requires A02

### Block B (Phase and Amplitude)
- B01: requires A04
- B02: requires B01
- B03: requires B01
- B04: requires A02, B01

### Block C (Connectivity Concepts)
- C01: requires B02
- C02: requires A03
- C03: requires C02

### Block D (Information Theory)
- D01: requires A01
- D02: requires D01
- D03: requires D02

### Block E (Hyperscanning Context)
- E01: requires C02
- E02: requires C01, E01

### Metrics (Part 2)
- F01 (COH): requires A03, C02
- F02 (ImCoh): requires F01, C01
- G01 (PLV): requires B02, C02
- G02 (PLI): requires G01, C01
- G03 (wPLI): requires G02
- H01 (CCorr): requires B03, C02
- H02 (PowCorr): requires H01

## Conventions
- No emojis in notebooks
- Minimal metaphors (only when necessary for pedagogy)
- Figures: soft pastel palette (see STYLE_GUIDE.md)
- Code: typed Python, NumPy-style docstrings
- One git branch per notebook: `feature/<notebook_id>`