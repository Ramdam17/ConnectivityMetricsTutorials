---
title: How to Use This Book
---

# How to Use This Book

## Navigating the Book

Use the **sidebar** on the left to browse the two parts and their blocks. Each block is a group of 2–7 notebooks on a focused topic. Notebooks within a block build on each other sequentially.

The **search bar** (top right) lets you find any concept, function name, or equation across the entire series.

---

## Running Notebooks Locally

```bash
git clone https://github.com/ppsp-team/ConnectivityMetricsTutorials.git
cd ConnectivityMetricsTutorials
uv sync
uv run jupyter lab book/
```

All notebooks import from the `src/` library at the project root. No installation of the library itself is required — `uv sync` handles all dependencies.

---

## Notebook Dependency Map

Read this as: *"Block X requires Block Y to have been completed first."*

```
A01 (entry point)
 └─ A02 ─ A03
       └─ A04a ─ A04b
                    └─ B01 ─ B02a ─ B02b ─┐
                         └─ B03a ─ B03b   │
                         └─ B04a ─ B04b   │
                                          │
            A03 ──────────────────────────┼─ C02a ─ C02b ─ C03
            B02b ─ C01 ───────────────────┘
                                          │
                                          └─ E01 ─ E02a ─ E02b ─ E02c
            A01 ── D01 ─ D02 ─ D03

Part 2 (requires Part 1):
  C02a + A03 ─── F01a ─ F01b ─ F02
  B02b + C02a ── G01 ─ G02 ─ G03
  B03b + C02a ── H01 ─ H02
  D02 ─────────── I01 ─ I02
```

:::note
If you are new to the series, start with {doc}`01_foundations/A_signal_fundamentals/A01_signals_and_sampling`.
If you have signal processing background, you may start at {doc}`01_foundations/B_phase_and_amplitude/B01_hilbert_transform`.
:::

---

## Session Format

Notebooks are designed for **1-hour weekly sessions**. The recommended workflow:

1. **Before the session (30 min):** Read the notebook, run the cells, attempt the exercises.
2. **During the session (60 min):** Present key concepts, compare implementations, discuss the discussion questions at the end.
3. **After the session:** Review the optional deep dive sections if interested.

---

## Code Conventions

All notebooks follow the same conventions:

- **Imports:** Standard library → third-party → `src/` project modules
- **Colours:** Always from `src.colors.COLORS` — never hardcoded hex values
- **Figures:** `figsize=(8, 5)` at 150 DPI (single plots); see {doc}`../docs/STYLE_GUIDE` for other layouts
- **Functions:** NumPy-style docstrings, full type hints, literature references in the docstring

---

## Contributing

Found an error or want to improve a notebook? See `CONTRIBUTING.md` at the project root.
Bugs and suggestions: [open an issue on GitHub](https://github.com/ppsp-team/ConnectivityMetricsTutorials/issues).
