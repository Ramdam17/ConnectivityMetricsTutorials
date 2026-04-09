# EEG Connectivity Metrics — A Hyperscanning Tutorial Series

A comprehensive workshop series on **EEG connectivity analysis for hyperscanning** — from signal processing fundamentals to advanced inter-brain connectivity metrics, published as a [Jupyter Book](https://ppsp-team.github.io/ConnectivityMetricsTutorials).

Designed for graduate students and researchers in neuroscience, psychology, and biomedical engineering. Assumes basic familiarity with EEG and Python. No prior signal processing expertise required.

---

## What You Will Learn

By the end of this series, you will be able to:

- Decompose EEG signals into frequency components and extract band power
- Apply filtering, compute the Hilbert transform, and work with instantaneous phase
- Understand why volume conduction is a fundamental challenge in EEG connectivity
- Implement from scratch: **PLV, PLI, wPLI, coherence, imaginary coherence, envelope correlation, mutual information, and transfer entropy**
- Apply these metrics in a hyperscanning (dual-brain) context
- Validate results with the [HyPyP](https://github.com/ppsp-team/HyPyP) library

---

## Book Structure

### Part 1 — Foundations

| Block | Topic | Notebooks |
|-------|-------|-----------|
| A | Signal Fundamentals | A01 Signals & Sampling · A02 Frequency Domain · A03 Power Spectrum · A04a Filter Fundamentals · A04b Applied Filtering |
| B | Phase & Amplitude | B01 Hilbert Transform · B02a Circular Statistics · B02b Phase Relationships · B03a Amplitude Envelope · B03b Envelope Correlation · B04a STFT & Spectrograms · B04b Wavelet Analysis |
| C | Connectivity Concepts | C01 Volume Conduction · C02a Matrix Computation · C02b Matrix Visualization · C03 Statistical Significance |
| D | Information Theory | D01 Entropy · D02 Mutual Information · D03 Transfer Entropy |
| E | Hyperscanning Context | E01 Graph Theory · E02a Hyperscanning Concepts · E02b Hyperscanning Data · E02c Hyperscanning Analysis |

### Part 2 — Connectivity Metrics

| Block | Topic | Notebooks |
|-------|-------|-----------|
| F | Coherence-Based | F01a Coherence Theory · F01b Coherence Practice · F02 Imaginary Coherence |
| G | Phase-Based | G01 PLV · G02 PLI · G03 wPLI |
| H | Amplitude-Based | H01 Envelope Correlation · H02 Power Correlation |
| I | Information-Theoretic | I01 MI Connectivity · I02 Transfer Entropy Connectivity |

---

## Setup

### Read online

The book is published at **[ppsp-team.github.io/ConnectivityMetricsTutorials](https://ppsp-team.github.io/ConnectivityMetricsTutorials)** — no installation required.

### Run locally

```bash
git clone https://github.com/ppsp-team/ConnectivityMetricsTutorials.git
cd ConnectivityMetricsTutorials
uv sync
uv run jupyter lab book/
```

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

To build the full HTML book locally:

```bash
uv run jupyter-book build
open _build/html/index.html
```

---

## Project Layout

```
book/                  # Jupyter Book content (published on GitHub Pages)
├── 01_foundations/    # Blocks A–E: signal processing foundations
└── 02_connectivity_metrics/  # Blocks F–I: connectivity metrics
myst.yml               # Jupyter Book configuration (mystmd)
src/                   # Reusable Python modules (imported by all notebooks)
archive/               # Original notebook drafts (reference only)
docs/                  # Style guides, review reports, JB template
data/                  # Raw and processed EEG data
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/CONTEXT.md](docs/CONTEXT.md) | Project context, audience, and pedagogical approach |
| [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) | Visual and code style standards |
| [docs/JB_STYLE_GUIDE.md](docs/JB_STYLE_GUIDE.md) | MyST/Jupyter Book formatting guide |
| [docs/JB_NOTEBOOK_TEMPLATE.ipynb](docs/JB_NOTEBOOK_TEMPLATE.ipynb) | Template for new notebooks |
| [docs/reviews/](docs/reviews/) | Sprint review reports |

---

## License

MIT — see [LICENSE](LICENSE).
