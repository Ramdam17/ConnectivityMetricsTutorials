# Style Guide — Hyperscanning Workshop

> **See also**: [NOTEBOOK_WRITING_SPEC.md](NOTEBOOK_WRITING_SPEC.md) for complete notebook writing specifications.

## Color Palette

### Signal Colors
| Name       | Hex       | Usage                    |
|------------|-----------|--------------------------|
| Sky Blue   | `#7EB8DA` | Signal 1 / Subject 1     |
| Rose Pink  | `#F4A4B8` | Signal 2 / Subject 2     |
| Sage Green | `#B8D4A8` | Signal 3 / Reference     |
| Golden     | `#E8C87A` | Signal 4 / Highlight     |
| Lavender   | `#C4A8D4` | Signal 5                 |
| Soft Teal  | `#A8D4D0` | Signal 6                 |

### Connectivity
| Name        | Hex       | Usage                    |
|-------------|-----------|--------------------------|
| Low Sync    | `#ECF0F1` | Low connectivity         |
| High Sync   | `#9B59B6` | High connectivity        |

### Diverging (correlations)
| Name     | Hex       | Usage          |
|----------|-----------|----------------|
| Negative | `#E17055` | Negative values|
| Zero     | `#FFFFFF` | Zero           |
| Positive | `#00CEC9` | Positive values|

### Frequency Bands
| Band  | Hex       | Range     |
|-------|-----------|-----------|
| Delta | `#6C5B7B` | 1-4 Hz    |
| Theta | `#C06C84` | 4-8 Hz    |
| Alpha | `#F8B500` | 8-13 Hz   |
| Beta  | `#00CEC9` | 13-30 Hz  |
| Gamma | `#6DD47E` | 30+ Hz    |

## Figure Dimensions

| Type              | Width | Height | DPI |
|-------------------|-------|--------|-----|
| Single plot       | 8     | 5      | 150 |
| Side-by-side (2)  | 12    | 5      | 150 |
| Grid (2x2)        | 12    | 10     | 150 |
| Connectivity matrix | 8   | 7      | 150 |
| Time series       | 12    | 4      | 150 |

## Font Sizes

| Element       | Size |
|---------------|------|
| Title         | 14   |
| Axis label    | 12   |
| Tick label    | 10   |
| Legend        | 10   |
| Annotation    | 10   |

## Notebook Cell Organization

Each notebook follows this structure:

1. **Header cell**: Title, duration, learning objectives
2. **Imports cell**: All imports grouped (standard lib, third-party, local)
3. **Section pattern**:
   - Markdown: concept explanation
   - Code: implementation
   - Markdown: interpretation
4. **Deep Dive sections**: Marked with `[Optional Deep Dive]` in title
5. **Summary cell**: Key takeaways as bullet points
6. **Discussion cell**: 3-5 questions for group discussion

### Section Numbering
- Main sections: `## Section 1: Title`
- Subsections: `### 1.1 Subtitle`
- Deep dives: `### [Optional Deep Dive] Mathematical Derivation`