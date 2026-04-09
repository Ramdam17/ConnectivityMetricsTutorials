# Connectivity Metrics Tutorials

A comprehensive workshop series on **EEG connectivity analysis for hyperscanning** — from signal processing fundamentals to advanced inter-brain connectivity metrics.

Designed for graduate students and researchers in neuroscience, psychology, and biomedical engineering. Assumes basic familiarity with EEG and Python. No prior signal processing expertise required.

---

## What You Will Learn

By the end of this workshop, you will be able to:

- Decompose EEG signals into frequency components and extract band power
- Apply filtering, compute the Hilbert transform, and work with instantaneous phase
- Understand why volume conduction is a fundamental challenge in EEG connectivity
- Implement from scratch: PLV, PLI, wPLI, coherence, imaginary coherence, envelope correlation, mutual information, and transfer entropy
- Apply these metrics in a hyperscanning (dual-brain) context
- Validate results with the [HyPyP](https://github.com/ppsp-team/HyPyP) library

---

## Notebook Structure

Each topic has two versions:
- **Tutorial** (`*.ipynb`): full implementation with inline code, explanations, and exercises
- **Quick** (`*_quick.ipynb`): streamlined version importing from `src/`, for reference or review

### Part 1: Foundations

| Block | Topic | Notebooks |
|-------|-------|-----------|
| A | Signal Fundamentals | A01 Signals & Sampling · A02 Frequency Domain · A03 Power Spectrum · A04a Filter Design · A04b Applied Filtering |
| B | Phase & Amplitude | B01 Hilbert Transform · B02a Circular Statistics · B02b Phase Relationships · B03a Envelope Extraction · B03b Envelope Correlation · B04a STFT & Spectrograms · B04b Wavelet Analysis |
| C | Connectivity Concepts | C01 Volume Conduction · C02a Matrix Computation · C02b Matrix Visualization · C03 Statistical Significance |
| D | Information Theory | D01 Entropy · D02 Mutual Information · D03 Transfer Entropy |
| E | Hyperscanning Context | E01 Graph Theory · E02a Hyperscanning Concepts · E02b Hyperscanning Data · E02c Hyperscanning Analysis |

### Part 2: Connectivity Metrics

| Block | Topic | Notebooks |
|-------|-------|-----------|
| F | Coherence-Based | F01a Coherence Theory · F01b Coherence Practice · F02 Imaginary Coherence |
| G | Phase-Based | G01 PLV · G02 PLI · G03 wPLI |
| H | Amplitude-Based | H01 Envelope Correlation · H02 Power Correlation |
| I | Information-Theoretic | I01 MI Connectivity · I02 Transfer Entropy Connectivity |

---

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management

---

## Setup

```bash
git clone https://github.com/ppsp-team/ConnectivityMetricsTutorials.git
cd ConnectivityMetricsTutorials
poetry install
poetry shell
jupyter-book build book/
open book/_build/html/index.html
```

Or run notebooks locally with `jupyter lab` from the `book/` directory.

---

## Project Layout

```
book/                        # Jupyter Book source (published on GitHub Pages)
├── 01_foundations/          # Blocks A–E: signal processing foundations
└── 02_connectivity_metrics/ # Blocks F–I: connectivity metric implementations

src/                         # Reusable Python modules (imported by all notebooks)
archive/                     # Original notebooks (reference implementations)
docs/                        # Style guide, writing spec, sprint plan, notebook index
data/                        # Raw and processed signals
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/CONTEXT.md](docs/CONTEXT.md) | Project context, audience, and pedagogical approach |
| [docs/NOTEBOOKS_INDEX.md](docs/NOTEBOOKS_INDEX.md) | Full index of all notebooks |
| [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) | Visual and code style standards |
| [docs/NOTEBOOK_WRITING_SPEC.md](docs/NOTEBOOK_WRITING_SPEC.md) | Specification for notebook structure |
| [docs/SPRINT_PLAN.md](docs/SPRINT_PLAN.md) | Development sprint plan and progress |

---

## License

MIT — see [LICENSE](LICENSE).
