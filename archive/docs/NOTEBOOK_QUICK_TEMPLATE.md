# Quick Notebook Template — Hyperscanning Workshop

This template defines the **exact structure** for quick reference notebooks (`*_quick.ipynb`).

Quick notebooks import functions from `src/` instead of defining them inline.
They must follow the same structure as full notebooks but with condensed content.

---

## Cell Structure (MANDATORY ORDER)

| Cell # | Type | Content |
|--------|------|---------|
| 1 | Markdown | Header (Title, Duration, Prerequisites, Learning Objectives) |
| 2 | Markdown | Table of Contents |
| 3 | Code | Imports |
| 4+ | Markdown/Code | Sections (following full notebook structure) |
| N-3 | Markdown | Summary |
| N-2 | Markdown | External Resources |
| N-1 | Markdown | Discussion Questions |

---

## Cell 1: Header (Markdown) — EXACT FORMAT

```markdown
# [ID]: [Title] (Quick Version)

**Duration**: ~30 minutes  
**Prerequisites**: [List notebook IDs, e.g., A01, A02]

> 💡 **Quick Version**: This notebook imports pre-built functions from `src/[module].py` instead of defining them inline. For the full tutorial with step-by-step function implementations, see [[ID]_[title].ipynb]([ID]_[title].ipynb).

## Learning Objectives

By the end of this notebook, you will be able to:
- [Objective 1: action verb + concept]
- [Objective 2: action verb + concept]
- [Objective 3: action verb + concept]
- Use the functions from `src/[module].py`

---
```

### Examples:
- `A01`: `# A01: Signals and Sampling (Quick Version)`
- `A02`: `# A02: Frequency Domain Analysis (Quick Version)`
- `A03`: `# A03: Power Spectrum and Frequency Bands (Quick Version)`
- `A04a`: `# A04a: Filter Fundamentals (Quick Version)`
- `A04b`: `# A04b: Applied Filtering (Quick Version)`

---

## Cell 2: Table of Contents (Markdown) — EXACT FORMAT

```markdown
## Table of Contents

1. [Introduction](#section-1-introduction)
2. [Section Title](#section-2-section-title)
3. [Section Title](#section-3-section-title)
...
N-2. [Exercises](#section-n-2-exercises)
N-1. [Summary](#summary)
N. [External Resources](#external-resources)
N+1. [Discussion Questions](#discussion-questions)

---
```

### Anchor Format Rules:
- Use `#section-X-title-with-dashes` format (NOT `#1.-Title`)
- All lowercase
- Replace spaces with dashes
- Summary, External Resources, Discussion Questions use simple anchors

---

## Cell 3: Imports (Code) — EXACT FORMAT

```python
# =============================================================================
# Imports
# =============================================================================

# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Local imports
src_path = Path.cwd().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from [module] import [function1], [function2], [function3]
from colors import COLORS

# Matplotlib configuration
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

print(f"NumPy version: {np.__version__}")
print(f"Source path: {src_path}")
```

### Import Rules:
- Use `# =====...=====` comment style for section headers
- Path to src: `Path.cwd().parent.parent.parent / "src"` (3 parents for A_signal_fundamentals/)
- Always import COLORS from colors module
- Always set matplotlib configuration

---

## Section Cells — FORMAT

### Section Header (Markdown)

```markdown
## Section N: [Title]

[2-3 paragraphs explaining the concept - SAME as full notebook but condensed]

Key points:
- Point 1
- Point 2
- Point 3
```

### Section Code (Code)

```python
# =============================================================================
# Section N: [Title]
# =============================================================================

# Use imported function (not defined inline)
result = imported_function(data, param1=value1, param2=value2)

# Visualization using COLORS
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y, color=COLORS["signal_1"], linewidth=1.5, label="Signal 1")
ax.plot(x, z, color=COLORS["signal_2"], linewidth=1.5, label="Signal 2")
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_title("Descriptive Title")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Color Usage — MANDATORY

Always use colors from `COLORS` dictionary. Available keys:

```python
# Signal colors (for multiple signals)
COLORS["signal_1"]  # Sky Blue - Subject 1 / Primary
COLORS["signal_2"]  # Rose Pink - Subject 2 / Secondary
COLORS["signal_3"]  # Sage Green - Reference / Tertiary
COLORS["signal_4"]  # Golden - Highlight / Quaternary
COLORS["signal_5"]  # Lavender
COLORS["signal_6"]  # Soft Teal

# Frequency bands (for EEG bands)
COLORS["delta"]     # 1-4 Hz
COLORS["theta"]     # 4-8 Hz
COLORS["alpha"]     # 8-13 Hz
COLORS["beta"]      # 13-30 Hz
COLORS["gamma"]     # 30+ Hz

# Connectivity (for matrices/heatmaps)
COLORS["low_sync"]  # Low connectivity
COLORS["high_sync"] # High connectivity

# Diverging (for correlations)
COLORS["negative"]  # Negative values
COLORS["zero"]      # Zero
COLORS["positive"]  # Positive values

# Utility
COLORS["grid"]      # Grid lines
COLORS["text"]      # Text color
COLORS["background"]# Background
```

### NEVER use:
- `COLORS["primary"]` ❌
- `COLORS["secondary"]` ❌
- `COLORS["accent1"]` ❌
- Hardcoded hex colors ❌

---

## Summary Cell (Markdown) — EXACT FORMAT

```markdown
## Summary

Key takeaways from this notebook:

- **[Concept 1]**: [One sentence explanation]
- **[Concept 2]**: [One sentence explanation]
- **[Concept 3]**: [One sentence explanation]

### Functions from `src/[module].py`

| Function | Purpose |
|----------|---------|
| `function_1()` | Brief description |
| `function_2()` | Brief description |
| `function_3()` | Brief description |

### Next Steps

In the next notebook ([ID]: [Title]), we will explore [preview of next topic].
```

---

## External Resources Cell (Markdown) — EXACT FORMAT

```markdown
## External Resources

### Documentation
- [Resource 1](URL) - Brief description
- [Resource 2](URL) - Brief description

### Scientific Papers
- [Author et al. (Year)](URL) - Brief description

### Videos
- [Video Title](URL) - Brief description
```

---

## Discussion Questions Cell (Markdown) — EXACT FORMAT

```markdown
## Discussion Questions

1. **[Topic 1]**: [Question about understanding]

2. **[Topic 2]**: [Question about application]

3. **[Topic 3]**: [Question about comparison]

4. **[Topic 4]**: [Question about limitations]

5. **[Topic 5]**: [Open-ended question]
```

---

## Checklist Before Committing

- [ ] Cell 1 is Header with Title, Duration, Prerequisites, Quick Version note, Learning Objectives
- [ ] Cell 2 is Table of Contents with `#section-X-...` anchors
- [ ] Cell 3 is Imports with `# =====` style, correct src path, COLORS imported
- [ ] All colors use `COLORS["signal_X"]` or other valid keys (NOT primary/secondary/accent)
- [ ] Summary includes function table from src/
- [ ] External Resources present
- [ ] Discussion Questions present
- [ ] All sections mirror the full notebook structure

---

## Quick Notebook vs Full Notebook Differences

| Aspect | Full Notebook | Quick Notebook |
|--------|---------------|----------------|
| Duration | ~60 minutes | ~30 minutes |
| Functions | Defined inline | Imported from src/ |
| Explanations | Detailed | Condensed |
| Deep Dives | Included | Omitted |
| Exercises | Full | Simplified |
| Visualizations | Same | Same |
| Structure | Complete | Same sections, less text |
