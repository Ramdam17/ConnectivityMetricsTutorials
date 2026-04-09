# Notebook Writing Specification — Hyperscanning Workshop

> **Version**: 1.0  
> **Last Updated**: December 2024  
> **Status**: Active

This document defines the complete specifications for writing and maintaining notebooks in the Hyperscanning Workshop. All contributors must follow these guidelines.

---

## Table of Contents

1. [General Principles](#1-general-principles)
2. [Notebook Structure](#2-notebook-structure)
3. [Color Palette Usage](#3-color-palette-usage)
4. [Exercise Format](#4-exercise-format)
5. [Code Style](#5-code-style)
6. [Splitting Long Notebooks](#6-splitting-long-notebooks)
7. [Dual Version System](#7-dual-version-system)
8. [External Resources](#8-external-resources)
9. [Checklist Before Commit](#9-checklist-before-commit)

---

## 1. General Principles

### 1.1 Language
- **All content in English**: code, comments, docstrings, markdown text, variable names
- Use clear, accessible language suitable for graduate students
- Avoid jargon without explanation

### 1.2 File Naming
- Format: `[Block][Number]_[snake_case_title].ipynb`
- Examples: `A01_signals_and_sampling.ipynb`, `B02a_circular_statistics.ipynb`
- Quick versions: `[Block][Number]_[title]_quick.ipynb`

### 1.3 Target Audience
- Graduate students in neuroscience/psychology
- Basic Python knowledge assumed
- No advanced math background required (explain when used)

---

## 2. Notebook Structure

### 2.1 Required Sections (in order)

> ⚠️ **Important**: The order Header → TOC → Imports must be respected in all notebooks.

| # | Section | Type | Required |
|---|---------|------|----------|
| 1 | Header | Markdown | ✅ |
| 2 | Table of Contents | Markdown | ✅ |
| 3 | Imports | Code | ✅ |
| 4 | Introduction | Markdown | ✅ |
| 5 | Intuition | Markdown + Code | ✅ |
| 6 | Implementation | Code + Markdown | ✅ |
| 7 | Visualization | Code | ✅ |
| 8 | HyPyP Comparison | Code + Markdown | When applicable |
| 9 | Application (Hyperscanning) | Code + Markdown | When applicable |
| 10 | Exercises | Markdown + Code | ✅ (minimum 3) |
| 11 | Summary | Markdown | ✅ |
| 12 | External Resources | Markdown | ✅ |
| 13 | Discussion Questions | Markdown | ✅ (5 questions) |

### 2.2 Header Format

```markdown
# [Block][Number]: [Full Title]

**Duration**: ~XX minutes  
**Prerequisites**: [Comma-separated notebook IDs]

## Learning Objectives

By the end of this notebook, you will be able to:
- [Action verb] + [concept]
- [Action verb] + [concept]
- [Action verb] + [concept]

---
```

### 2.3 Table of Contents Format

```markdown
## Table of Contents

1. [Introduction](#1-introduction)
2. [Intuition: Why This Matters](#2-intuition-why-this-matters)
3. [Implementation](#3-implementation)
   - 3.1 [Subsection Title](#31-subsection-title)
   - 3.2 [Subsection Title](#32-subsection-title)
4. [Visualization](#4-visualization)
5. [Application to Hyperscanning](#5-application-to-hyperscanning)
6. [Exercises](#6-exercises)
7. [Summary](#7-summary)
8. [External Resources](#8-external-resources)
9. [Discussion Questions](#9-discussion-questions)

---
```

### 2.4 Section Numbering

- Main sections: `## 1. Section Title`
- Subsections: `### 1.1 Subsection Title`
- Deep dives: `### 📚 [Optional] Mathematical Derivation`
- Use horizontal rules (`---`) between major sections

### 2.5 Imports Cell Template

```python
# =============================================================================
# Imports
# =============================================================================

# Standard library
import sys
from pathlib import Path
from typing import Optional, Tuple

# Third-party
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import signal

# Local imports
src_path = Path.cwd().parents[2]  # Adjust based on notebook depth
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.colors import COLORS
from src.constants import EEG_BANDS
from src.plotting import configure_plots

# Apply plot configuration
configure_plots()
```

### 2.6 Summary Section Format

```markdown
## 7. Summary

### Key Takeaways

- **[Concept 1]**: [One sentence explanation]
- **[Concept 2]**: [One sentence explanation]
- **[Concept 3]**: [One sentence explanation]

### What You Learned

1. [Skill 1]
2. [Skill 2]
3. [Skill 3]

### Next Steps

Continue to **[Next Notebook ID]: [Title]** to learn about [topic].
```

### 2.7 Discussion Questions Format

```markdown
## 9. Discussion Questions

Reflect on these questions to deepen your understanding:

1. **[Question 1]**
   
2. **[Question 2]**

3. **[Question 3]**

4. **[Question 4]**

5. **[Question 5]**
```

---

## 3. Color Palette Usage

### 3.1 Golden Rule

> **NEVER use hardcoded hex colors in notebooks.**  
> Always import from `src/colors.py`.

### 3.2 Standard Import

```python
from src.colors import COLORS
```

### 3.3 Color Keys Reference

| Key | Color | Usage |
|-----|-------|-------|
| `signal_1` | Sky Blue (#7EB8DA) | Subject 1, first signal |
| `signal_2` | Rose Pink (#F4A4B8) | Subject 2, second signal |
| `signal_3` | Sage Green (#B8D4A8) | Reference, third signal |
| `signal_4` | Golden (#E8C87A) | Highlight, emphasis |
| `signal_5` | Lavender (#C4A8D4) | Additional signal |
| `signal_6` | Soft Teal (#A8D4D0) | Additional signal |
| `negative` | Coral (#E17055) | Negative correlations |
| `positive` | Turquoise (#00CEC9) | Positive correlations |
| `delta` | Plum (#6C5B7B) | Delta band (1-4 Hz) |
| `theta` | Rose (#C06C84) | Theta band (4-8 Hz) |
| `alpha` | Gold (#F8B500) | Alpha band (8-13 Hz) |
| `beta` | Teal (#00CEC9) | Beta band (13-30 Hz) |
| `gamma` | Mint (#6DD47E) | Gamma band (30+ Hz) |

### 3.4 Usage Examples

```python
# Single signal
ax.plot(t, signal, color=COLORS["signal_1"], label="Subject 1")

# Two subjects comparison
ax.plot(t, s1, color=COLORS["signal_1"], label="Subject 1")
ax.plot(t, s2, color=COLORS["signal_2"], label="Subject 2")

# Frequency bands
for band, (low, high) in EEG_BANDS.items():
    ax.axvspan(low, high, alpha=0.3, color=COLORS[band], label=band.capitalize())

# Connectivity matrices (diverging)
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
    "connectivity",
    [COLORS["negative"], COLORS["zero"], COLORS["positive"]]
)
```

### 3.5 Forbidden Practices

❌ **Never do this:**
```python
# Hardcoded colors
ax.plot(x, y, color="#7EB8DA")
ax.plot(x, y, color="blue")

# Creating aliases in notebook
PRIMARY_BLUE = COLORS["signal_1"]  # Don't create aliases!

# Modifying COLORS dict
COLORS["my_color"] = "#123456"  # Never modify!
```

---

## 4. Exercise Format

### 4.1 Standard Exercise Template

```markdown
### 🎯 Exercise N: [Descriptive Title]

**Objective**: [One sentence describing what the student will practice]

**Context**: [Optional: 1-2 sentences of background if needed]

**Instructions**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected output**: [Describe what a correct solution looks like]
```

```python
# =============================================================================
# Exercise N: [Title]
# =============================================================================

# Your code here:



```

```markdown
<details>
<summary>💡 Click to reveal solution</summary>

```python
# Solution
[solution code]
```

**Explanation**: [2-3 sentences explaining the key aspects of the solution]

</details>
```

### 4.2 Exercise Guidelines

- **Minimum 3 exercises** per notebook
- **Progressive difficulty**: Easy → Medium → Challenging
- **Labels**: Add difficulty indicator
  - 🟢 **Easy**: Direct application of concepts shown
  - 🟡 **Medium**: Requires combining multiple concepts
  - 🔴 **Challenging**: Requires creative thinking or extension

### 4.3 Example Exercise

```markdown
### 🎯 Exercise 2: Detecting Aliasing 🟡

**Objective**: Identify when aliasing occurs by manipulating sampling frequency.

**Instructions**:
1. Generate a 50 Hz sine wave
2. Sample it at 80 Hz (below Nyquist) and 200 Hz (above Nyquist)
3. Plot both sampled versions and compare
4. Calculate the aliased frequency for the undersampled case

**Expected output**: Two plots showing correct vs aliased sampling
```

---

## 5. Code Style

### 5.1 Function Documentation

All functions must have NumPy-style docstrings:

```python
def compute_coherence(
    signal_1: NDArray[np.float64],
    signal_2: NDArray[np.float64],
    fs: float,
    nperseg: int = 256,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spectral coherence between two signals.

    Parameters
    ----------
    signal_1 : NDArray[np.float64]
        First input signal.
    signal_2 : NDArray[np.float64]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    frequencies : NDArray[np.float64]
        Array of sample frequencies.
    coherence : NDArray[np.float64]
        Coherence values between 0 and 1.

    Examples
    --------
    >>> freqs, coh = compute_coherence(s1, s2, fs=256)
    """
    # Implementation
```

### 5.2 Code Cell Headers

Use clear section headers in code cells:

```python
# =============================================================================
# Section: Generate Synthetic Signals
# =============================================================================

# Create test signals
...

# -----------------------------------------------------------------------------
# Subsection: Apply filter
# -----------------------------------------------------------------------------
```

### 5.3 Visualization Template

```python
# Create figure
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Plot data
ax.plot(t, signal, color=COLORS["signal_1"], linewidth=1.5, label="Signal")

# Configure axes
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Amplitude (µV)", fontsize=12)
ax.set_title("Descriptive Title", fontsize=14)
ax.set_xlim([0, 1])

# Add legend and grid
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, color=COLORS["grid"])

# Finalize
plt.tight_layout()
plt.show()
```

---

## 6. Splitting Long Notebooks

### 6.1 When to Split

| Lines | Action |
|-------|--------|
| < 1500 | Keep as single notebook |
| 1500-2200 | Consider splitting if natural break points exist |
| > 2200 | **Must split** |

### 6.2 How to Split

1. **Identify natural break points** at section boundaries
2. **Balance size**: Each part should be within ±20% of average
3. **Naming**: Use letter suffix: `D01a_`, `D01b_`, `D01c_`
4. **Each part must be self-contained**:
   - Own header with title including "(Part N of M)"
   - Own imports
   - Own summary
   - Reference to other parts

### 6.3 Split Notebook Header

```markdown
# D01a: Entropy and Information Theory (Part 1 of 2)

**Duration**: ~45 minutes  
**Prerequisites**: A01, A02, A03

> This is **Part 1** of a 2-part series on entropy.  
> - Part 1 (this notebook): Introduction to Entropy  
> - Part 2 ([D01b](D01b_entropy_applications.ipynb)): Applications

## Learning Objectives
...
```

### 6.4 Split Summary Template

```markdown
## Summary

[Standard summary content]

### Continue to Part 2

This notebook covered the fundamentals. In **Part 2**, you will learn:
- [Topic 1]
- [Topic 2]

➡️ Continue to [D01b: Entropy Applications](D01b_entropy_applications.ipynb)
```

---

## 7. Dual Version System

### 7.1 Overview

Each notebook exists in two versions:
- **Tutorial version** (`A01_*.ipynb`): Functions defined inline for learning
- **Quick version** (`A01_*_quick.ipynb`): Functions imported from `src/`

### 7.2 Tutorial Version (Default)

- Functions are defined inline with full implementation
- Educational comments explain each step
- After function definition, add note:

```markdown
> 💡 **Note**: This function is also available in `src/signals.py` as `generate_sine_wave()`.
```

### 7.3 Quick Version

The Quick version is a **full notebook** with the same structure as the Tutorial version:

- **Same structure**: Header → TOC → Imports → Sections → Exercises → Summary → External Resources → Discussion
- **Functions imported** from `src/` instead of defined inline
- **Condensed content**: Less explanatory text, same visualizations
- **Same exercises** with solutions

> ⚠️ **Important**: Quick versions are NOT reference cards. They are complete notebooks that can be followed independently.

### 7.4 Quick Version Header

```markdown
# A02: Frequency Domain Analysis (Quick Version)

**Duration**: ~30 minutes  
**Prerequisites**: A01 (Signals and Sampling), Basic Python, NumPy fundamentals

> 💡 **Quick Version**: This notebook imports pre-built functions from `src/spectral.py` instead of defining them inline. For the full tutorial with step-by-step function implementations, see [A02_frequency_domain.ipynb](A02_frequency_domain.ipynb).

## Learning Objectives
...
```

### 7.5 Quick Version Creation

Quick versions are created **manually** (not auto-generated) to ensure quality:
1. Copy the tutorial structure
2. Replace function definitions with imports from `src/`
3. Condense explanatory text while keeping key concepts
4. Keep all visualizations and exercises
5. Ensure the notebook runs independently

---

## 8. External Resources

### 8.1 Required Section

Every notebook must include an "External Resources" section before Discussion.

### 8.2 Standard Format

```markdown
## External Resources

### 🎥 Video Summary

- **[Notebook Title - Video Overview](NotebookLM_URL)** — AI-generated video summary of this notebook's key concepts

### 📝 Practice & Review

- **[Quiz: Test Your Understanding](NotebookLM_URL)** — Interactive quiz on [topic]
- **[Flashcards: Key Terms](NotebookLM_URL)** — Review flashcards for spaced repetition learning

### 🎥 Video Tutorials

- **[Video Title](YouTube_URL)** (Duration) — Brief description
- **[Video Title](YouTube_URL)** (Duration) — Brief description

### 📖 Further Reading

- **[Resource Title](URL)** — Brief description (1 sentence)
- **[Resource Title](URL)** — Brief description

### 📚 Academic References (optional)

- Author et al. (Year). *Title*. Journal. [DOI](URL)

---
```

### 8.3 NotebookLM Resources

Each notebook should have three NotebookLM-generated resources:
1. **Video Overview** (not "Audio Overview") — Generated video summary
2. **Quiz** — Interactive knowledge check
3. **Flashcards** — Key terms review

> ⚠️ **Important**: Each notebook must have **unique** external resources. Do not copy-paste the same YouTube links or articles across notebooks.

### 8.4 Resource Categories

Include at least one resource from each applicable category:
- NotebookLM resources (video, quiz, flashcards)
- Video content (lectures, demonstrations)
- Conceptual explanations (blogs, tutorials)
- Academic papers (foundational or review papers) — optional

---

## 9. Checklist Before Commit

### 9.1 Structure Check

- [ ] **Cell order is correct**: Header → TOC → Imports (verify with notebook summary)
- [ ] Header with title, duration, prerequisites, learning objectives
- [ ] Table of Contents with working links
- [ ] All required sections present
- [ ] Section numbers consistent
- [ ] Summary with key takeaways
- [ ] External resources section with unique NotebookLM + external links
- [ ] 5 discussion questions

> ⚠️ **Cell Order Verification**: After creating or editing a notebook, always verify the first 3 cells:
> 1. Cell 1 = Header (starts with `# [Block][Number]: Title`)
> 2. Cell 2 = TOC (starts with `## Table of Contents`)
> 3. Cell 3 = Imports (Python code cell with imports)

### 9.2 Code Check

- [ ] All cells execute without errors
- [ ] No hardcoded colors (using `COLORS` dict)
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Imports grouped correctly
- [ ] Path handling works from notebook location

### 9.3 Content Check

- [ ] Minimum 3 exercises with solutions
- [ ] Exercises use standard format
- [ ] All visualizations have labels, titles, legends
- [ ] No unexplained jargon
- [ ] Next steps point to correct notebook

### 9.4 Quick Version Check

- [ ] `_quick.ipynb` version generated
- [ ] Quick version executes correctly
- [ ] All functions available in `src/`

---

## Appendix A: Figure Dimensions Reference

| Type | figsize | dpi |
|------|---------|-----|
| Single plot | (10, 5) | 150 |
| Two plots side-by-side | (12, 5) | 150 |
| 2×2 grid | (12, 10) | 150 |
| Time series (wide) | (14, 4) | 150 |
| Connectivity matrix | (8, 7) | 150 |
| Comparison (3 plots) | (15, 4) | 150 |

## Appendix B: Common Mistakes

| Mistake | Correction |
|---------|------------|
| `color="blue"` | `color=COLORS["signal_1"]` |
| `sys.path.append("../../..")` | Use `Path.cwd().parents[N]` pattern |
| Missing type hints | Add `param: type` and `-> return_type` |
| Exercise without difficulty | Add 🟢/🟡/🔴 indicator |
| No TOC | Always include Table of Contents |
| Hardcoded paths | Use `Path` objects with relative paths |

---

*End of Notebook Writing Specification*
