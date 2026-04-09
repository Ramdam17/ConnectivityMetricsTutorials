# Notebook Template — Hyperscanning Workshop

Use this template as the starting structure for each notebook.

---

## Header Cell (Markdown)
```
# [Block][Number]: [Title]

**Duration**: ~60 minutes
**Prerequisites**: [List notebook IDs, e.g., A01, A02]

## Learning Objectives

By the end of this notebook, you will be able to:
- [Objective 1: action verb + concept]
- [Objective 2: action verb + concept]
- [Objective 3: action verb + concept]

---
```

---

## Imports Cell (Code)
```python
# Standard library
from typing import Optional, Tuple

# Third-party
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import signal

# Local
from src.colors import COLORS
from src.plotting import configure_plots
from src.signals import generate_sine_wave

# Configuration
configure_plots()
```

---

## Section Pattern

### Section N: [Title] (Markdown)
```
## Section N: [Title]

[2-4 paragraphs explaining the concept]

Key points:
- Point 1
- Point 2
- Point 3
```

### Implementation (Code)
```python
def function_name(
    param1: NDArray[np.float64],
    param2: float,
) -> NDArray[np.float64]:
    """
    Brief description.

    Parameters
    ----------
    param1 : NDArray[np.float64]
        Description of param1.
    param2 : float
        Description of param2.

    Returns
    -------
    NDArray[np.float64]
        Description of return value.
    """
    # Implementation
    result = param1 * param2
    return result
```

### Interpretation (Markdown)
```
The result shows that [interpretation of the output].

Notice how [key observation]. This is because [explanation].
```

---

## Visualization Cell Pattern
```python
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

ax.plot(x, y, color=COLORS["signal_1"], linewidth=1.5, label="Signal 1")

ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Amplitude (µV)", fontsize=12)
ax.set_title("Descriptive Title", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Optional Deep Dive Section (Markdown + Code)
```
### [Optional Deep Dive] Mathematical Derivation

For those interested in the mathematical foundations:

[LaTeX equations and detailed derivation]

$$
X(f) = \int_{-\infty}^{\infty} x(t) e^{-2\pi i f t} dt
$$

[Explanation of each term]
```

---

## HyPyP Comparison Section
```
## Section N: Comparison with HyPyP

Let us verify our implementation against HyPyP.
```
```python
# Our implementation
result_ours = our_function(data)

# HyPyP implementation
from hypyp.prep import ...
result_hypyp = hypyp_function(data)

# Compare
difference = np.abs(result_ours - result_hypyp).max()
print(f"Maximum difference: {difference:.2e}")
```
```
The results match within numerical precision, confirming our implementation is correct.
```

---

## Hyperscanning Application Section
```
## Section N: Application to Hyperscanning Data

Now let us apply this method to real dyadic EEG data.

### Context
[Brief description of the experimental paradigm]

### Analysis
```
```python
# Load dyad data
subject_1 = load_data("subject_1")
subject_2 = load_data("subject_2")
control = load_data("control")

# Compute metric for dyad
result_dyad = compute_metric(subject_1, subject_2)

# Compute metric for surrogate (subject_1 vs control)
result_surrogate = compute_metric(subject_1, control)

# Visualize comparison
```
```
### Interpretation

The dyad shows [observation], while the surrogate comparison shows [observation].
This suggests that [interpretation in social neuroscience context].
```

---

## Summary Cell (Markdown)
```
## Summary

Key takeaways from this notebook:

- **[Concept 1]**: [One sentence explanation]
- **[Concept 2]**: [One sentence explanation]
- **[Concept 3]**: [One sentence explanation]

### What we learned to do:
- [Skill 1]
- [Skill 2]
- [Skill 3]

### Next steps
In the next notebook ([ID]: [Title]), we will explore [preview of next topic].
```

---

## Discussion Cell (Markdown)
```
## Discussion Questions

For the live session, consider the following questions:

1. [Conceptual question about understanding]

2. [Application question: when would you use this?]

3. [Comparison question: how does this relate to X?]

4. [Critical thinking: what are the limitations?]

5. [Open question: what would happen if...?]
```

---

## Complete Notebook Skeleton
```
# [ID]: [Title]
**Duration**: ~60 minutes
**Prerequisites**: [...]

## Learning Objectives
- [...]

---

## Section 1: Introduction
[Context and motivation]

## Section 2: Building Intuition
[Synthetic signals demonstration]

## Section 3: Implementation from Scratch
[Step-by-step code]

### [Optional Deep Dive] Mathematical Foundation
[Formal derivation]

## Section 4: Visualization
[Beautiful plots]

## Section 5: Comparison with HyPyP
[Verification]

## Section 6: Hyperscanning Application
[Real data example]

## Summary
[Key takeaways]

## Discussion Questions
[5 questions]
```