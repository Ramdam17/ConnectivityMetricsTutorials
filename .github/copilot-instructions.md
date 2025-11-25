# Copilot Instructions — Hyperscanning Workshop

## Language
All code, comments, docstrings, and documentation must be written in English.

## File Management
Never create, delete, or edit any file without explicit user consent.

## Project Structure
- Notebooks: `01_foundations/` and `02_connectivity_metrics/`, by category (A, B, C...)
- Reusable functions: `src/` directory
- Naming: `A01_`, `A02_`, `B01_`, etc.

## Git Workflow
- One branch per notebook: `feature/<notebook_id>`
- Clear commit messages
- Merge to main after review

## Dependencies
Managed with Poetry. Core: numpy, scipy, matplotlib, mne, hypyp.

## Typed Python
- Complete type hints on all functions (params + return)
- Use `numpy.typing.NDArray` for arrays
- Use `typing` module (Optional, Union, Tuple, Callable)
- mypy strict compatible

## Coding Standards
- PEP 8 compliant
- NumPy-style docstrings
- Functions over 20 lines go to `src/`
- No hardcoded values

## Notebook Structure
Introduction → Intuition → Implementation → Visualization → HyPyP comparison → Application → Summary → Discussion

## Visualization
Follow STYLE_GUIDE.md for colors, dimensions, fonts.