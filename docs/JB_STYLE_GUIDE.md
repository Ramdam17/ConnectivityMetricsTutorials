# Jupyter Book Style Guide

Extension of [STYLE_GUIDE.md](STYLE_GUIDE.md) for the MyST/Jupyter Book format.
Colour palette, figure dimensions, and font sizes are defined in STYLE_GUIDE.md and remain unchanged.

---

## MyST Admonitions

Use admonitions to guide the reader's attention. Do not overuse them — one per section maximum.

| Type | MyST syntax | When to use |
|------|-------------|-------------|
| `note` | `:::note` | Background context, supplementary info |
| `tip` | `:::tip` | Practical advice, shortcuts |
| `warning` | `:::warning` | Common mistakes, numerical pitfalls |
| `seealso` | `:::seealso` | Cross-references to other notebooks |
| `important` | `:::important` | Critical concepts that gate understanding |

**Example:**
```markdown
:::note
The Hilbert transform assumes an analytic signal. Applying it to broadband EEG
without prior bandpass filtering produces unreliable instantaneous phase estimates.
:::
```

---

## Cross-References

Use MyST cross-references to link between notebooks in the book.

```markdown
{doc}`../A_signal_fundamentals/A02_frequency_domain`   ← link to another notebook
{ref}`my-label`                                         ← link to a labelled target
```

Place a label before any figure or section you want to reference:
```markdown
(sec-plv-definition)=
## Definition
```

---

## Figures

All figures must use the `{figure}` directive (not raw markdown `![](...)`) to enable
numbering and captions.

```markdown
:::{figure} figures/plv_example.png
:name: fig-plv-example
:width: 80%

**Figure 1.** Phase difference distribution for two coupled EEG signals.
:::
```

In code cells, save figures explicitly rather than relying on inline display:
```python
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
# ... plotting code ...
fig.savefig("figures/plv_example.png", bbox_inches="tight")
plt.show()
```

---

## Math

Use `$$` for display equations and `$...$` for inline math.
Label equations that are referenced in the text:

```markdown
$$
\text{PLV} = \left| \frac{1}{N} \sum_{t=1}^{N} e^{i \Delta\phi(t)} \right|
$$ (eq-plv)

As shown in {eq}`eq-plv`, PLV is the resultant vector length.
```

---

## Notebook Header (Frontmatter)

Every notebook must begin with a YAML frontmatter cell (raw cell, not markdown):

```yaml
---
title: "A01 — Signals & Sampling"
subtitle: "Block A: Signal Fundamentals"
authors:
  - name: Rémy Ramadour
kernelspec:
  name: python3
  display_name: Python 3
---
```

---

## Section Structure

Mirrors the existing STYLE_GUIDE.md cell organization, adapted to MyST headings:

```
# Title (from frontmatter — do not repeat)

## 1. Introduction
## 2. Intuition
## 3. Implementation
### 3.1 Step one
### [Optional Deep Dive] Mathematical derivation
## 4. Application
## 5. Exercises
## 6. Summary
## 7. Resources
## 8. Discussion Questions
```

---

## What NOT to do

- Do not use `ipywidgets` or any interactive widgets — use static figures only.
- Do not reference "quick" versions (they no longer exist).
- Do not hardcode hex colours in notebooks — always import from `src.colors.COLORS`.
- Do not use emojis in section titles or body text.
- Do not repeat the title from the frontmatter as an `# H1` heading in the notebook body.
