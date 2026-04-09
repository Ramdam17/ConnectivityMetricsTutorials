# Copilot Instructions — Connectivity Metrics Tutorial Book

## Language
All code, comments, docstrings, and documentation must be written in English.

## Project Structure
- Book content: `book/` (Jupyter Book / mystmd)
- Reusable functions: `src/` directory
- Archive (reference only, do not modify): `archive/`
- Naming: `A01_`, `A02_`, `B01_`, etc. No `_quick` suffix.

## Git Workflow
- One branch per sprint: `feat/SXX-short-description`
- Commit scope: `feat(SXX): ...`, `fix(SXX): ...`, `docs(SXX): ...`
- Merge to main after review

## Dependencies
Managed with **uv** (`pyproject.toml` + `uv.lock`). Core: numpy, scipy, matplotlib, mne, hypyp, jupyter-book.

## Typed Python
- Complete type hints on all functions (params + return)
- Use `numpy.typing.NDArray` for arrays
- Use `typing` module (Optional, Union, Tuple, Callable)
- mypy strict compatible

## Coding Standards
- PEP 8 compliant
- NumPy-style docstrings
- Functions over 20 lines go to `src/`
- No hardcoded values — import constants from `src.constants` and `src.colors`

## Notebook Structure (MyST / Jupyter Book)
See `docs/JB_STYLE_GUIDE.md` for the full spec.

Required sections in order:
1. YAML frontmatter (raw cell)
2. Introduction
3. Intuition
4. Implementation
5. Application
6. Exercises (minimum 3, with collapsible solutions)
7. Summary
8. Resources
9. Discussion Questions

**No quick notebook variants.** Each topic has exactly one definitive notebook.
**No interactive widgets.** Static figures only.

## Colors (CRITICAL)
- Always import: `from src.colors import COLORS`
- Signal keys: `signal_1` … `signal_6`
- Band keys: `delta`, `theta`, `alpha`, `beta`, `gamma`
- NEVER hardcode hex colours in notebooks

## Visualization
- Follow `docs/STYLE_GUIDE.md` for dimensions and font sizes
- Save figures explicitly with `fig.savefig(...)` before `plt.show()`
- Use the `{figure}` MyST directive in markdown cells for captioned figures
