# C02: Connectivity Matrices — Copilot Directive

## Metadata

- **Notebook ID**: C02
- **Title**: Connectivity Matrices
- **Branch**: `feature/C02_connectivity_matrices`
- **Duration**: 55 minutes
- **Prerequisites**: C01 (Volume Conduction Problem), B02 (Working with Phase)
- **Next**: C03 (Statistical Significance)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand connectivity matrices as organized pairwise measurements
2. Construct connectivity matrices from multi-channel data
3. Interpret matrix properties (symmetry, diagonal)
4. Visualize connectivity with heatmaps and circular plots
5. Extract network-level summary statistics
6. Handle hyperscanning matrices (within + between participants)
7. Organize and store connectivity results efficiently

---

## Required Imports
```
numpy, matplotlib, scipy.stats, seaborn, typing, numpy.typing
Optional: mne (for topographic plots), networkx (for graph visualization)
Local: src/colors.py, src/plotting.py, src/phase.py, src/connectivity_utils.py
```

---

## Section 1: Introduction — From Pairs to Networks

**Purpose**: Motivate the need for organized connectivity analysis.

**Content to cover**:
- So far: connectivity between TWO signals (one pair)
- Real EEG: 32, 64, 128+ electrodes
- Real hyperscanning: 2 × n_electrodes!
- We need to analyze ALL pairs systematically
- Connectivity matrix: organized storage of pairwise values
- Entry (i, j) = connectivity between channel i and channel j
- Foundation for network neuroscience and graph theory
- Essential for comparing conditions, participants, groups

**Length**: 3-4 paragraphs

---

## Section 2: What is a Connectivity Matrix?

**Purpose**: Define the structure formally.

**Content to cover**:
- Square matrix: n_channels × n_channels
- Entry M[i, j] = connectivity metric between channel i and channel j
- **Diagonal**: M[i, i] = connectivity of channel with itself
  - Often undefined or set to 0/1/NaN depending on metric
  - Self-connectivity not meaningful for most metrics
- **Symmetry**: for undirected metrics, M[i, j] = M[j, i]
  - PLV, coherence, correlation are symmetric
  - Some metrics are directed (Granger, transfer entropy) → asymmetric
- **Values**: depend on metric (0-1 for PLV, -1 to +1 for correlation, etc.)

**Visualization 1**:
- Schematic of a connectivity matrix
- 6×6 example with labeled channels (F3, F4, C3, C4, P3, P4)
- Color-coded cells
- Annotations: "diagonal", "symmetric entries M[i,j] = M[j,i]"
- Caption: "Structure of a connectivity matrix"

---

## Section 3: Computing a Connectivity Matrix

**Purpose**: Implement systematic pairwise computation.

**Content to cover**:
- Algorithm:
  1. Loop over all unique pairs (i, j) where i < j
  2. Compute connectivity metric for each pair
  3. Fill matrix: M[i, j] = M[j, i] = result
  4. Handle diagonal appropriately
- Efficiency considerations:
  - n channels → n(n-1)/2 unique pairs
  - 64 channels → 2016 pairs!
  - Vectorization helps but loop is often clearest
- For now: use simple PLV as example metric

**Code to produce**:

Function 1: `compute_connectivity_matrix`
```
Parameters:
    data: NDArray[np.float64]  # shape (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    metric: str = "plv"  # for now, only plv supported
Returns:
    NDArray[np.float64]  # shape (n_channels, n_channels)
Compute pairwise connectivity for all channel pairs
Diagonal set to NaN
Matrix is symmetric
```

Function 2: `get_n_pairs`
```
Parameters:
    n_channels: int
Returns:
    int
Number of unique pairs: n(n-1)/2
```

Function 3: `get_pair_indices`
```
Parameters:
    n_channels: int
Returns:
    list[tuple[int, int]]
List of all unique (i, j) pairs where i < j
```

**Visualization 2**:
- Generate synthetic 6-channel data
- Some channels share sources (volume conduction simulation)
- Compute PLV matrix
- Show the resulting 6×6 heatmap
- Annotate which pairs have high PLV (shared sources)

---

## Section 4: Visualizing Connectivity Matrices

**Purpose**: Effective visualization techniques.

**Content to cover**:
- **Heatmap**: most common visualization
  - Color represents connectivity strength
  - Channel labels on axes
  - Colorbar with clear range
- **Considerations**:
  - Color map choice: sequential (0 to max) or diverging (for signed metrics)
  - Mask diagonal (often NaN or uninformative)
  - Sort channels meaningfully (by region, hemisphere)
  - Consistent color scale for comparisons

**Code to produce**:

Function 4: `plot_connectivity_matrix`
```
Parameters:
    matrix: NDArray[np.float64]
    channel_names: list[str] | None = None
    ax: plt.Axes | None = None
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    mask_diagonal: bool = True
    title: str | None = None
Returns:
    plt.Axes
Plot connectivity matrix as heatmap
Proper axis labels, colorbar, optional diagonal masking
```

**Visualization 3**:
- Same matrix from Section 3
- Well-formatted heatmap with:
  - Channel labels
  - Colorbar
  - Masked diagonal (gray or white)
  - Title indicating metric and frequency band
- Follow STYLE_GUIDE.md

**Visualization 4**:
- Comparison: different color maps
- Same matrix, 4 subplots:
  - viridis (sequential)
  - plasma (sequential)
  - RdBu (diverging — for signed metrics)
  - Pastel version from our palette
- Discuss when to use each

---

## Section 5: Circular Connectivity Plots

**Purpose**: Alternative visualization for network structure.

**Content to cover**:
- Heatmaps show all values but hide network structure
- Circular (chord) plots:
  - Channels arranged in a circle
  - Lines connect pairs with significant connectivity
  - Line thickness/color = connectivity strength
- Better for showing:
  - Which channels are connected
  - Network hubs (many connections)
  - Hemispheric patterns
- Common in network neuroscience papers

**Code to produce**:

Function 5: `plot_circular_connectivity`
```
Parameters:
    matrix: NDArray[np.float64]
    channel_names: list[str]
    threshold: float | None = None  # only show connections above threshold
    ax: plt.Axes | None = None
    linewidth_scale: float = 2.0
    node_colors: list[str] | None = None
Returns:
    plt.Axes
Circular plot with channels as nodes, connections as arcs
Line properties scaled by connectivity value
```

**Visualization 5**:
- Circular plot of the 6-channel example
- Threshold to show only strong connections
- Color-code by hemisphere (left=blue, right=pink)
- Clear visualization of network structure

**Visualization 6**:
- Side-by-side: heatmap vs circular plot
- Same data, different views
- Caption: "Choose visualization based on your message"

---

## Section 6: Matrix Properties and Sanity Checks

**Purpose**: Validate connectivity matrix results.

**Content to cover**:
- **Symmetry check**: for undirected metrics, verify M = M.T
- **Value range**: PLV in [0, 1], correlation in [-1, 1]
- **Diagonal**: should be NaN or expected value (1 for self-correlation)
- **No NaN/Inf**: except diagonal, all values should be finite
- **Distribution**: examine histogram of values
  - All zeros? → problem with computation
  - All ones? → problem with data
  - Reasonable spread? → good

**Code to produce**:

Function 6: `validate_connectivity_matrix`
```
Parameters:
    matrix: NDArray[np.float64]
    metric: str = "plv"
    tolerance: float = 1e-10
Returns:
    dict[str, bool | str]
    {"is_symmetric": bool, "in_range": bool, "has_nan": bool, "issues": str}
Validate matrix properties, return diagnostics
```

Function 7: `get_matrix_statistics`
```
Parameters:
    matrix: NDArray[np.float64]
    exclude_diagonal: bool = True
Returns:
    dict[str, float]
    {"mean": ..., "std": ..., "min": ..., "max": ..., "median": ...}
Summary statistics of matrix values
```

**Visualization 7**:
- Histogram of connectivity values (off-diagonal)
- Annotate mean, std
- Check for reasonable distribution
- Example with good data and example with problematic data (all zeros)

---

## Section 7: Extracting Upper Triangle

**Purpose**: Efficient storage and analysis of unique values.

**Content to cover**:
- Symmetric matrix: half the values are redundant
- Upper triangle: entries where i < j
- Lower triangle: entries where i > j (same as upper for symmetric)
- For analysis: extract upper triangle only
- Saves memory, avoids double-counting in statistics
- NumPy functions: `np.triu_indices`, `np.triu`

**Code to produce**:

Function 8: `get_upper_triangle_values`
```
Parameters:
    matrix: NDArray[np.float64]
    k: int = 1  # offset (1 = exclude diagonal)
Returns:
    NDArray[np.float64]
    1D array of upper triangle values
```

Function 9: `upper_triangle_to_matrix`
```
Parameters:
    values: NDArray[np.float64]
    n_channels: int
    fill_diagonal: float = np.nan
Returns:
    NDArray[np.float64]
Reconstruct symmetric matrix from upper triangle values
```

**Visualization 8**:
- Show matrix with upper triangle highlighted
- Arrow pointing to extracted 1D array
- Arrow back to reconstructed matrix
- Demonstrate roundtrip

---

## Section 8: Channel Grouping and Region Averaging

**Purpose**: Aggregate connectivity by brain region.

**Content to cover**:
- Individual electrode pairs can be noisy
- Often want connectivity between REGIONS, not individual electrodes
- Example: "frontal-to-parietal" connectivity
- Approach:
  1. Define channel groups (frontal, central, parietal, etc.)
  2. Average connectivity within each region pair
  3. Result: smaller region × region matrix
- More robust, easier to interpret
- Common in clinical and cognitive neuroscience

**Code to produce**:

Function 10: `define_channel_groups`
```
Parameters:
    channel_names: list[str]
    group_definitions: dict[str, list[str]]
    # e.g., {"frontal": ["F3", "Fz", "F4"], "parietal": ["P3", "Pz", "P4"]}
Returns:
    dict[str, list[int]]
Map group names to channel indices
```

Function 11: `compute_region_connectivity`
```
Parameters:
    matrix: NDArray[np.float64]
    channel_groups: dict[str, list[int]]
Returns:
    Tuple[NDArray[np.float64], list[str]]
    (region_matrix, region_names)
Average connectivity between all region pairs
Result: n_regions × n_regions matrix
```

**Visualization 9**:
- Original 16-channel matrix
- Channel groups highlighted (colored boxes on matrix)
- Resulting 4×4 region matrix
- Side-by-side comparison

---

## Section 9: Hyperscanning Matrices — The Big Picture

**Purpose**: Extend to two-participant analysis.

**Content to cover**:
- Hyperscanning: 2 participants × n electrodes each
- Three types of connectivity:
  1. **Within-participant 1**: n × n matrix
  2. **Within-participant 2**: n × n matrix  
  3. **Between-participants**: n × n matrix (the interesting one!)
- Full hyperscanning matrix: 2n × 2n
- Structure:
```
  [  Within-P1  |  Between   ]
  [  Between.T  |  Within-P2 ]
```
- Between-participants block: P1 channels × P2 channels
- This is where inter-brain synchrony lives!

**Visualization 10**:
- Large 2n × 2n hyperscanning matrix (schematic)
- Four quadrants labeled:
  - Top-left: Within P1 (volume conduction concerns!)
  - Top-right: Between (no volume conduction! ✓)
  - Bottom-left: Between.T
  - Bottom-right: Within P2
- Color-code quadrants
- Subject colors from palette

---

## Section 10: Computing Hyperscanning Connectivity

**Purpose**: Implement hyperscanning-specific computation.

**Content to cover**:
- Input: data from both participants
  - P1: (n_channels, n_samples)
  - P2: (n_channels, n_samples)
- Output options:
  - Full matrix (2n × 2n)
  - Between-participant matrix only (n × n)
  - Separate within + between matrices
- Between-participant matrix: NOT symmetric in general
  - M[i, j] = connectivity from P1_channel_i to P2_channel_j
  - Could define symmetric version by averaging

**Code to produce**:

Function 12: `compute_hyperscanning_connectivity`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    metric: str = "plv"
Returns:
    dict[str, NDArray[np.float64]]
    {"within_p1": ..., "within_p2": ..., "between": ..., "full": ...}
Compute all connectivity components for hyperscanning
```

Function 13: `extract_between_participant_matrix`
```
Parameters:
    full_matrix: NDArray[np.float64]  # (2n, 2n)
    n_channels_per_participant: int
Returns:
    NDArray[np.float64]  # (n, n)
Extract between-participant block from full matrix
```

**Visualization 11**:
- Simulated hyperscanning data (2 participants, 8 channels each)
- Some inter-brain coupling at specific channel pairs
- Full 16×16 matrix with quadrants visible
- Annotate the between-participant block

---

## Section 11: Visualizing Hyperscanning Connectivity

**Purpose**: Specialized visualizations for two-brain data.

**Content to cover**:
- Standard heatmap works but needs clear labeling
- Better: separate visualizations for within and between
- Circular plot with two groups (one per participant)
- Color-code by participant
- Inter-brain connections cross between groups

**Code to produce**:

Function 14: `plot_hyperscanning_matrix`
```
Parameters:
    full_matrix: NDArray[np.float64]
    channel_names_p1: list[str]
    channel_names_p2: list[str]
    ax: plt.Axes | None = None
    highlight_between: bool = True
Returns:
    plt.Axes
Plot full hyperscanning matrix with quadrant annotations
Optional highlighting of between-participant block
```

Function 15: `plot_hyperscanning_circular`
```
Parameters:
    between_matrix: NDArray[np.float64]
    channel_names_p1: list[str]
    channel_names_p2: list[str]
    threshold: float | None = None
    ax: plt.Axes | None = None
Returns:
    plt.Axes
Circular plot with P1 channels on left, P2 on right
Inter-brain connections shown as arcs crossing the middle
```

**Visualization 12**:
- Hyperscanning circular plot
- Left semicircle: Participant 1 channels (sky blue)
- Right semicircle: Participant 2 channels (rose pink)
- Connections crossing between them
- Beautiful, publication-ready figure

**Visualization 13**:
- Between-participant matrix only (n × n)
- X-axis: P2 channels
- Y-axis: P1 channels
- Clear labeling of what each axis represents
- Highlight strongest inter-brain connections

---

## Section 12: Global Connectivity Metrics

**Purpose**: Summarize matrices with single values.

**Content to cover**:
- Sometimes need one number per participant/condition
- Global metrics:
  - **Mean connectivity**: average of all (off-diagonal) values
  - **Density**: proportion of connections above threshold
  - **Variance**: spread of connectivity values
- For hyperscanning:
  - Mean between-participant connectivity
  - Compare to mean within-participant
  - Ratio or difference can indicate inter-brain coupling strength

**Code to produce**:

Function 16: `compute_global_connectivity`
```
Parameters:
    matrix: NDArray[np.float64]
    exclude_diagonal: bool = True
Returns:
    float
Mean of connectivity matrix (global connectivity)
```

Function 17: `compute_connection_density`
```
Parameters:
    matrix: NDArray[np.float64]
    threshold: float
    exclude_diagonal: bool = True
Returns:
    float
Proportion of connections exceeding threshold
```

Function 18: `compute_hyperscanning_ratio`
```
Parameters:
    within_mean: float
    between_mean: float
Returns:
    float
Ratio of between to within connectivity
>1 indicates stronger inter-brain than intra-brain connectivity
```

**Visualization 14**:
- Bar chart comparing:
  - Within-P1 mean connectivity
  - Within-P2 mean connectivity
  - Between-participants mean connectivity
- Error bars if multiple time windows
- Show hyperscanning ratio

---

## Section 13: Saving and Loading Connectivity Data

**Purpose**: Practical data management.

**Content to cover**:
- Connectivity matrices can be large (especially time-resolved)
- Efficient storage formats:
  - NumPy .npy / .npz (simple, fast)
  - HDF5 (hierarchical, metadata support)
  - Pandas DataFrame (for tabular analysis)
- What to save:
  - Matrix values
  - Channel names
  - Parameters (metric, band, fs)
  - Participant/condition info
- Naming conventions for files

**Code to produce**:

Function 19: `save_connectivity_results`
```
Parameters:
    filepath: str
    matrix: NDArray[np.float64]
    channel_names: list[str]
    metadata: dict  # metric, band, fs, etc.
Returns:
    None
Save to .npz with matrix and metadata
```

Function 20: `load_connectivity_results`
```
Parameters:
    filepath: str
Returns:
    dict[str, Any]
    {"matrix": ..., "channel_names": ..., "metadata": ...}
Load connectivity results from .npz
```

---

## Section 14: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Build Your Own Matrix
- Generate 8-channel synthetic data
- 3 channels share one source, 3 share another, 2 are independent
- Compute PLV connectivity matrix
- Visualize as heatmap
- Interpret the structure (which clusters appear?)

**Exercise 2**: Visualization Comparison
- Take the matrix from Exercise 1
- Create: heatmap, circular plot, region-averaged matrix
- Which visualization best shows the cluster structure?
- Which would you use in a paper?

**Exercise 3**: Hyperscanning Matrix
- Generate data for 2 "participants" (6 channels each)
- Add inter-brain coupling at specific pairs (e.g., F3-F3, C3-C3)
- Compute full hyperscanning matrix
- Extract and visualize between-participant connectivity
- Can you identify the coupled pairs?

**Exercise 4**: Global Metrics
- Compute global connectivity for multiple "conditions":
  - Baseline (low coupling)
  - Task (high coupling)
- Compare mean connectivity, density, hyperscanning ratio
- Create bar chart showing condition differences

**Exercise 5**: Sanity Checks
- Intentionally create a "bad" matrix (add NaN, break symmetry)
- Run validation function
- Fix the issues
- Verify fixed matrix passes validation

---

## Section 15: Functions for src/connectivity.py

**Purpose**: Identify functions for connectivity module.

**New file**: src/connectivity.py

Functions to extract:
- `compute_connectivity_matrix`
- `get_n_pairs`
- `get_pair_indices`
- `plot_connectivity_matrix`
- `plot_circular_connectivity`
- `validate_connectivity_matrix`
- `get_matrix_statistics`
- `get_upper_triangle_values`
- `upper_triangle_to_matrix`
- `define_channel_groups`
- `compute_region_connectivity`
- `compute_hyperscanning_connectivity`
- `extract_between_participant_matrix`
- `plot_hyperscanning_matrix`
- `plot_hyperscanning_circular`
- `compute_global_connectivity`
- `compute_connection_density`
- `compute_hyperscanning_ratio`
- `save_connectivity_results`
- `load_connectivity_results`

---

## Summary

**Key takeaways to list**:
- Connectivity matrix: n×n array storing pairwise connectivity values
- Symmetric for undirected metrics (PLV, coherence, correlation)
- Diagonal is typically NaN or uninformative (self-connectivity)
- Visualizations: heatmaps for values, circular plots for network structure
- Upper triangle contains all unique information (for symmetric matrices)
- Region averaging reduces noise and aids interpretation
- Hyperscanning: 2n×2n matrix with within and between blocks
- Between-participant block is key for inter-brain synchrony
- Global metrics summarize matrix into single values
- Always validate matrix properties before analysis

---

## Discussion Questions

1. You have 64-channel EEG from two participants (128 channels total in hyperscanning). How many unique between-participant channel pairs are there? How long would computation take if each pair takes 0.1 seconds?

2. Your connectivity matrix shows high values everywhere (mean PLV = 0.85). What might cause this? Is it necessarily bad?

3. When would you prefer a circular connectivity plot over a heatmap? When would the heatmap be better?

4. In hyperscanning, why might we want to analyze the between-participant matrix separately from the within-participant matrices?

5. You're comparing connectivity between a "cooperation" and "competition" condition. Would you compare the full matrices, the between-participant matrices, or global metrics? What are the trade-offs?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Schematic | Matrix structure with annotations | conceptual |
| 2 | Heatmap | First connectivity matrix example | viridis |
| 3 | Heatmap | Well-formatted with all elements | viridis |
| 4 | 2×2 grid | Colormap comparison | various cmaps |
| 5 | Circular plot | 6-channel network | hemisphere colors |
| 6 | Side-by-side | Heatmap vs circular | viridis, pastel |
| 7 | Histogram | Distribution of connectivity values | signal_1 |
| 8 | Diagram | Upper triangle extraction | signal_1 |
| 9 | Side-by-side | Full matrix → region matrix | viridis |
| 10 | Schematic | 2n×2n hyperscanning structure | subject_1, subject_2 |
| 11 | Heatmap | Full hyperscanning matrix | viridis + quadrants |
| 12 | Circular | Hyperscanning with two groups | subject_1, subject_2 |
| 13 | Heatmap | Between-participant matrix only | viridis |
| 14 | Bar chart | Global connectivity comparison | condition colors |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Matrix symmetry handled correctly
- [ ] NaN/diagonal handling consistent
- [ ] Hyperscanning functions handle different-sized participant data
- [ ] File I/O includes proper error handling
- [ ] Channel name validation included