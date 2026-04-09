# C03: Statistical Significance — Copilot Directive

## Metadata

- **Notebook ID**: C03
- **Title**: Statistical Significance for Connectivity
- **Branch**: `feature/C03_statistical_significance`
- **Duration**: 70 minutes
- **Prerequisites**: C02 (Connectivity Matrices), basic statistics knowledge
- **Next**: D01 (Entropy and Information)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand why significance testing is essential for connectivity
2. Generate surrogate data to establish null distributions
3. Implement permutation testing for connectivity metrics
4. Apply multiple comparisons correction (FDR, Bonferroni)
5. Compute and interpret confidence intervals
6. Determine appropriate significance thresholds
7. Avoid common statistical pitfalls in connectivity analysis
8. Apply these methods to hyperscanning data

---

## Required Imports
```
numpy, matplotlib, scipy.stats, scipy.ndimage, typing, numpy.typing
Optional: statsmodels (for FDR), mne (for cluster correction)
Local: src/colors.py, src/plotting.py, src/phase.py, src/connectivity.py, src/connectivity_utils.py
```

---

## Section 1: Introduction — Why Significance Matters

**Purpose**: Motivate rigorous statistical testing.

**Content to cover**:
- Connectivity metrics always give you a NUMBER
- But is that number meaningful or just noise?
- Example: PLV = 0.35 between two channels
  - Is this "high"? "Low"? "Significant"?
  - Depends on what you'd expect by CHANCE
- Without proper statistics:
  - False positives: claiming connectivity that isn't there
  - False negatives: missing real connectivity
  - Non-reproducible results
- Scientific claims REQUIRE statistical validation
- This notebook: how to do it RIGHT

**Key message**: "A connectivity value without a p-value is just a number."

**Length**: 4-5 paragraphs

---

## Section 2: The Null Hypothesis for Connectivity

**Purpose**: Define what "chance" means for connectivity.

**Content to cover**:
- **Null hypothesis (H₀)**: no true connectivity between signals
- What does "no connectivity" look like?
  - Signals have similar spectral properties
  - But NO consistent phase/amplitude relationship
  - Any measured connectivity is due to chance
- **Alternative hypothesis (H₁)**: true connectivity exists
- We need to know the DISTRIBUTION of connectivity under H₀
- Then ask: is our observed value unlikely under H₀?
- If unlikely (p < α) → reject H₀ → claim significant connectivity

**Visualization 1**:
- Conceptual diagram
- Null distribution (histogram) with threshold marked
- Observed value shown
- Shaded area = p-value
- Caption: "Is our observation in the tail of the null distribution?"

---

## Section 3: Surrogate Data Methods

**Purpose**: Generate data that satisfies the null hypothesis.

**Content to cover**:
- **Surrogate data**: artificial data preserving some properties but destroying connectivity
- Key idea: keep spectral properties, destroy phase relationships
- Methods:
  1. **Phase shuffling**: randomize phase in frequency domain
  2. **Time shifting**: circular shift one signal relative to other
  3. **Trial shuffling**: mix trials between conditions/participants
  4. **Amplitude-adjusted phase shuffling (AAFT)**: preserves amplitude distribution too
- For connectivity: phase shuffling is most common
- Repeat many times → build null distribution

**Visualization 2**:
- Original signal pair (with connectivity)
- Surrogate pair (phase shuffled)
- Show: similar power spectra, but different phase relationship
- Two rows: original and surrogate
- Each row: signals + cross-correlation or phase difference

---

## Section 4: Phase Shuffling — Implementation

**Purpose**: Implement the most common surrogate method.

**Content to cover**:
- Algorithm:
  1. FFT of signal
  2. Keep amplitude spectrum unchanged
  3. Replace phase with random uniform [-π, π]
  4. Inverse FFT
- Result: same power spectrum, random phases
- Important: shuffle ONE signal only to destroy relationship
- Shuffling both independently is also valid

**Code to produce**:

Function 1: `shuffle_phase`
```
Parameters:
    signal: NDArray[np.float64]
    seed: int | None = None
Returns:
    NDArray[np.float64]
Phase-shuffle a signal while preserving power spectrum
Randomize phases uniformly in [-π, π]
```

Function 2: `generate_surrogate_pair`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    method: str = "phase_shuffle"  # or "time_shift"
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate surrogate pair with destroyed connectivity
Only shuffle signal2 (or both, depending on method)
```

**Visualization 3**:
- Original signal: show PSD
- Surrogate signal: show PSD
- Overlay both PSDs → nearly identical
- Caption: "Phase shuffling preserves spectral properties"

**Visualization 4**:
- Original pair: phase difference histogram (concentrated)
- Surrogate pair: phase difference histogram (uniform)
- Side-by-side comparison
- Caption: "Surrogates destroy phase relationship"

---

## Section 5: Time Shifting — Alternative Method

**Purpose**: Simpler surrogate method.

**Content to cover**:
- Circular shift one signal by random amount
- Destroys temporal alignment → destroys connectivity
- Simpler than phase shuffling
- May not preserve exact spectral properties for short signals
- Good for quick checks, permutation tests

**Code to produce**:

Function 3: `circular_shift`
```
Parameters:
    signal: NDArray[np.float64]
    shift_samples: int
Returns:
    NDArray[np.float64]
Circularly shift signal by specified samples
```

Function 4: `generate_shifted_surrogate`
```
Parameters:
    signal: NDArray[np.float64]
    min_shift_samples: int = 100  # minimum shift to ensure decorrelation
    seed: int | None = None
Returns:
    NDArray[np.float64]
Randomly shift signal by at least min_shift_samples
```

**Visualization 5**:
- Original aligned signals
- Shifted signals
- Cross-correlation showing peak shift
- Demonstrate that connectivity measure changes

---

## Section 6: Building the Null Distribution

**Purpose**: Generate many surrogates to estimate H₀ distribution.

**Content to cover**:
- One surrogate → one connectivity value under H₀
- Need MANY surrogates to estimate distribution
- Typical: 200-1000 surrogates (more = better estimate)
- For each surrogate:
  1. Generate surrogate data
  2. Compute connectivity metric
  3. Store value
- Result: distribution of connectivity under H₀
- Compare observed value to this distribution

**Code to produce**:

Function 5: `compute_null_distribution`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    metric: str = "plv"
    n_surrogates: int = 500
    surrogate_method: str = "phase_shuffle"
    seed: int | None = None
Returns:
    NDArray[np.float64]
    Array of connectivity values under null hypothesis
```

**Visualization 6**:
- Null distribution histogram (500 surrogates)
- Vertical line for observed value
- Shaded tail area (p-value region)
- Annotate: "Observed PLV = 0.42, p = 0.003"
- Clear visual of hypothesis test

---

## Section 7: Computing P-Values

**Purpose**: Quantify statistical significance.

**Content to cover**:
- **P-value**: probability of observing result at least as extreme under H₀
- For connectivity (typically one-tailed, high values = connectivity):
  - p = (# surrogates ≥ observed) / n_surrogates
- Edge case: if observed > all surrogates
  - p < 1/n_surrogates (report as p < 0.002 for 500 surrogates)
  - Can't say p = 0!
- **Two-tailed**: if interested in both high AND low extremes
  - p = (# surrogates at least as extreme) / n_surrogates
- Common thresholds: α = 0.05, 0.01, 0.001

**Code to produce**:

Function 6: `compute_pvalue`
```
Parameters:
    observed: float
    null_distribution: NDArray[np.float64]
    tail: str = "right"  # "right", "left", or "two"
Returns:
    float
Compute p-value from null distribution
Handle edge cases (observed beyond all surrogates)
```

Function 7: `compute_significance`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    metric: str = "plv"
    n_surrogates: int = 500
    alpha: float = 0.05
Returns:
    dict[str, float | bool]
    {"observed": ..., "pvalue": ..., "significant": ..., "threshold": ...}
Complete significance test for a signal pair
```

**Visualization 7**:
- Three examples:
  - Highly significant (p < 0.001): observed far in tail
  - Marginally significant (p ≈ 0.04): observed just in tail
  - Not significant (p = 0.23): observed within distribution
- Three panels with null distributions and observed values
- Color-coded by significance

---

## Section 8: The Multiple Comparisons Problem

**Purpose**: Address the critical issue of testing many pairs.

**Content to cover**:
- Connectivity matrix: MANY pairs tested simultaneously
- 64 channels → 2016 pairs!
- Problem: if α = 0.05, expect 5% false positives by chance
  - 2016 pairs × 0.05 = ~101 false positives!
- This is the **multiple comparisons problem**
- Solutions:
  1. **Bonferroni correction**: strictest, divide α by n_tests
  2. **False Discovery Rate (FDR)**: controls proportion of false positives
  3. **Cluster correction**: for spatially/temporally adjacent tests
  4. **Permutation testing on global statistics**
- Must apply correction whenever testing multiple pairs!

**Visualization 8**:
- Simulation: 1000 pairs, NO true connectivity
- Test each at α = 0.05
- Show ~50 "significant" results (all false positives!)
- Histogram of p-values (should be uniform under H₀)
- Caption: "Without correction, 5% false positives expected"

---

## Section 9: Bonferroni Correction

**Purpose**: Implement the simplest correction.

**Content to cover**:
- Bonferroni: divide α by number of tests
- α_corrected = α / n_tests
- Very conservative: controls **family-wise error rate (FWER)**
- FWER: probability of ANY false positive
- Example: 2016 pairs, α = 0.05
  - α_corrected = 0.05 / 2016 ≈ 0.000025
  - Need p < 0.000025 to be significant!
- Pros: simple, guaranteed control
- Cons: very strict, may miss real effects (low power)

**Code to produce**:

Function 8: `bonferroni_correction`
```
Parameters:
    pvalues: NDArray[np.float64]
    alpha: float = 0.05
Returns:
    Tuple[NDArray[np.bool_], float]
    (significant_mask, corrected_alpha)
Apply Bonferroni correction
Return boolean mask of significant tests
```

**Visualization 9**:
- P-values for all pairs (sorted)
- Horizontal line at α = 0.05 (uncorrected)
- Horizontal line at α_bonferroni (corrected)
- Show how many survive each threshold
- Caption: "Bonferroni is strict but controls false positives"

---

## Section 10: False Discovery Rate (FDR)

**Purpose**: More powerful correction method.

**Content to cover**:
- FDR: controls expected PROPORTION of false positives
- Less strict than Bonferroni, more powerful
- Benjamini-Hochberg procedure:
  1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₙ
  2. Find largest k where pₖ ≤ (k/n) × α
  3. Reject H₀ for all tests with p ≤ pₖ
- Controls FDR at level α (e.g., 5% of discoveries are false)
- Standard in neuroimaging (fMRI, EEG)
- Better balance of sensitivity and specificity

**Code to produce**:

Function 9: `fdr_correction`
```
Parameters:
    pvalues: NDArray[np.float64]
    alpha: float = 0.05
    method: str = "bh"  # Benjamini-Hochberg
Returns:
    Tuple[NDArray[np.bool_], NDArray[np.float64]]
    (significant_mask, adjusted_pvalues)
Apply FDR correction using Benjamini-Hochberg
Return mask and adjusted p-values
```

**Visualization 10**:
- Same p-values as Visualization 9
- Show BH procedure graphically:
  - Sorted p-values vs rank
  - BH threshold line: (k/n) × α
  - Intersection determines cutoff
- Compare number surviving Bonferroni vs FDR

**Visualization 11**:
- Venn diagram or bar chart
- Compare: uncorrected significant, Bonferroni significant, FDR significant
- Show FDR is intermediate between uncorrected and Bonferroni

---

## Section 11: Permutation Testing for Matrices

**Purpose**: Non-parametric approach for full matrices.

**Content to cover**:
- Alternative to surrogate-per-pair approach
- Permutation testing:
  1. Define test statistic (e.g., mean connectivity)
  2. Permute condition/group labels
  3. Recompute statistic
  4. Repeat many times → null distribution of STATISTIC
- Advantages:
  - Tests global effect, not individual pairs
  - Naturally handles multiple comparisons
  - No parametric assumptions
- For hyperscanning: permute participant pairing

**Code to produce**:

Function 10: `permutation_test_matrices`
```
Parameters:
    matrices_group1: list[NDArray[np.float64]]  # list of matrices
    matrices_group2: list[NDArray[np.float64]]
    n_permutations: int = 1000
    statistic: str = "mean_difference"
    seed: int | None = None
Returns:
    dict[str, Any]
    {"observed_statistic": ..., "null_distribution": ..., "pvalue": ...}
Permutation test comparing two groups of connectivity matrices
```

Function 11: `permutation_test_hyperscanning`
```
Parameters:
    connectivity_values: NDArray[np.float64]  # observed between-participant connectivity
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    n_permutations: int = 1000
    seed: int | None = None
Returns:
    dict[str, Any]
    {"observed": ..., "null_distribution": ..., "pvalue": ...}
Test if inter-brain connectivity exceeds chance
Permute time series between participants
```

**Visualization 12**:
- Permutation test result
- Null distribution of mean connectivity difference
- Observed difference marked
- P-value annotated
- Caption: "Permutation test for group difference"

---

## Section 12: Confidence Intervals

**Purpose**: Quantify uncertainty in connectivity estimates.

**Content to cover**:
- P-value tells if effect exists, not how big/reliable
- **Confidence interval (CI)**: range of plausible values
- 95% CI: if we repeated experiment, 95% of CIs would contain true value
- Methods for connectivity:
  1. **Bootstrap**: resample data, recompute metric, get distribution
  2. **Surrogate-based**: less common
- Bootstrap procedure:
  1. Resample time points with replacement
  2. Compute connectivity
  3. Repeat 1000+ times
  4. CI = [2.5th percentile, 97.5th percentile]

**Code to produce**:

Function 12: `bootstrap_connectivity`
```
Parameters:
    signal1: NDArray[np.float64]
    signal2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    metric: str = "plv"
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    seed: int | None = None
Returns:
    dict[str, float]
    {"observed": ..., "ci_low": ..., "ci_high": ..., "std": ...}
Bootstrap confidence interval for connectivity
```

**Visualization 13**:
- Bootstrap distribution of connectivity values
- Observed value marked
- CI bounds marked (vertical lines)
- Shaded CI region
- Caption: "95% CI shows reliability of estimate"

**Visualization 14**:
- Multiple channel pairs with CIs
- Forest plot style:
  - Point estimate (dot)
  - CI (horizontal line)
  - Reference line at 0 or threshold
- Easy comparison of effect sizes and uncertainty

---

## Section 13: Effect Size and Power

**Purpose**: Beyond p-values — practical significance.

**Content to cover**:
- **Effect size**: magnitude of connectivity, independent of sample size
- A tiny effect can be "significant" with enough data
- A large effect can be "not significant" with little data
- For connectivity:
  - The metric value itself is an effect size
  - Can standardize: (observed - mean_null) / std_null = z-score
- **Statistical power**: probability of detecting true effect
- Affected by:
  - Effect size (larger = more power)
  - Sample size / data length (more = more power)
  - Number of surrogates (more = better null estimate)
  - Alpha level (higher = more power but more false positives)

**Code to produce**:

Function 13: `compute_effect_size`
```
Parameters:
    observed: float
    null_distribution: NDArray[np.float64]
Returns:
    dict[str, float]
    {"zscore": ..., "percentile": ...}
Compute standardized effect size relative to null
```

**Visualization 15**:
- Scatter plot: effect size vs p-value for many pairs
- Color by significance
- Show that high effect + low p tend to go together
- But not always! Small effects can be significant with power

---

## Section 14: Practical Guidelines

**Purpose**: Actionable recommendations.

**Content to cover**:

**Guideline 1**: Always use surrogate testing
- Never trust raw connectivity values
- Minimum 500 surrogates, 1000 preferred

**Guideline 2**: Always correct for multiple comparisons
- FDR is good default (balance power/control)
- Bonferroni if very strict control needed
- Report which correction used

**Guideline 3**: Report more than p-values
- Include effect sizes (the metric values)
- Include confidence intervals
- Include number of surrogates/permutations

**Guideline 4**: Check your null distribution
- Should be reasonable (not all zeros, not identical to observed)
- Histogram should look plausible

**Guideline 5**: Be wary of "p-hacking"
- Don't try multiple tests until one works
- Pre-register analysis plan when possible
- Report all tests, not just significant ones

**Guideline 6**: For hyperscanning
- Inter-brain: permute participant pairings as null
- Real pair should exceed random pairings
- Within-brain: standard surrogate methods

**Table**:
| Situation | Recommended Approach |
|-----------|---------------------|
| Single pair, exploratory | Surrogate test, report p and CI |
| Multiple pairs, one comparison | Surrogate + FDR correction |
| Group comparison | Permutation test on global statistic |
| Hyperscanning | Permute participant pairings |
| Time-resolved | Cluster correction or FDR across time |

---

## Section 15: Common Pitfalls

**Purpose**: Warn about mistakes.

**Content to cover**:

**Pitfall 1**: Not enough surrogates
- 100 surrogates → p-value resolution only 0.01
- Need 1000+ for p < 0.001 claims

**Pitfall 2**: Forgetting multiple comparisons
- "We found significant connectivity at Fz-Pz"
- Did you test 2000 pairs and only report one?

**Pitfall 3**: Circular analysis
- Using same data to define ROI and test connectivity
- Inflates false positives

**Pitfall 4**: Confusing significance and importance
- p < 0.001 doesn't mean the effect matters
- Consider effect size and practical relevance

**Pitfall 5**: Ignoring non-significant results
- "No significant connectivity" is also a finding
- May indicate low power or true null effect

**Pitfall 6**: Wrong surrogate method
- Phase shuffling destroys phase connectivity
- But may not be appropriate for amplitude metrics
- Match surrogate to your null hypothesis

---

## Section 16: Complete Analysis Pipeline

**Purpose**: Put it all together.

**Content to cover**:
- Full workflow for significance testing:
  1. Compute observed connectivity matrix
  2. For each pair: generate surrogates, get null distribution
  3. Compute p-values for all pairs
  4. Apply FDR correction
  5. Create significance mask
  6. Report: significant pairs, effect sizes, corrected threshold
- Alternative: permutation test on global metric

**Code to produce**:

Function 14: `significance_analysis_pipeline`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
    metric: str = "plv"
    n_surrogates: int = 500
    alpha: float = 0.05
    correction: str = "fdr"
    seed: int | None = None
Returns:
    dict[str, Any]
    {
        "connectivity_matrix": ...,
        "pvalue_matrix": ...,
        "significant_mask": ...,
        "corrected_alpha": ...,
        "n_significant": ...
    }
Full pipeline: compute, test, correct, report
```

**Visualization 16**:
- Multi-panel summary figure:
  - Observed connectivity matrix
  - P-value matrix (log scale)
  - Significance mask (binary)
  - Significant connections only (masked matrix)
- Publication-ready summary

---

## Section 17: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Build a Null Distribution
- Two signals with known PLV = 0.5 (simulated)
- Generate 500 surrogates using phase shuffling
- Plot null distribution
- Compute p-value
- Is PLV = 0.5 significant?

**Exercise 2**: Multiple Comparisons Simulation
- Generate 100 independent signal pairs (NO true connectivity)
- Test each at α = 0.05 (uncorrected)
- Count "significant" results (should be ~5)
- Apply Bonferroni and FDR
- How many survive each correction?

**Exercise 3**: FDR vs Bonferroni Power
- Generate 100 pairs: 10 with true connectivity, 90 without
- Test all with surrogates
- Apply Bonferroni and FDR
- Compare: true positives and false positives for each
- Which method has better balance?

**Exercise 4**: Bootstrap Confidence Intervals
- Single pair with moderate connectivity
- Compute bootstrap CI
- Vary data length: 5s, 10s, 30s, 60s
- How does CI width change with data length?
- Plot CI width vs data length

**Exercise 5**: Full Pipeline
- 8-channel data with 3 truly connected pairs
- Run full significance pipeline
- Can you recover the true pairs?
- What happens if you use too few surrogates?

**Exercise 6**: Hyperscanning Permutation
- Data from 2 participants
- Compute between-participant connectivity
- Permutation test: shuffle time alignment
- Is inter-brain connectivity significant?

---

## Section 18: Functions for src/statistics.py

**Purpose**: Identify functions for statistics module.

**New file**: src/statistics.py

Functions to extract:
- `shuffle_phase`
- `generate_surrogate_pair`
- `circular_shift`
- `generate_shifted_surrogate`
- `compute_null_distribution`
- `compute_pvalue`
- `compute_significance`
- `bonferroni_correction`
- `fdr_correction`
- `permutation_test_matrices`
- `permutation_test_hyperscanning`
- `bootstrap_connectivity`
- `compute_effect_size`
- `significance_analysis_pipeline`

---

## Summary

**Key takeaways to list**:
- Connectivity values require statistical validation
- Surrogate data provides null distribution (no connectivity)
- P-value: probability of result under null hypothesis
- Multiple comparisons: testing many pairs inflates false positives
- Bonferroni: strict, controls family-wise error rate
- FDR: less strict, controls false discovery proportion
- Permutation tests: non-parametric, handles global statistics
- Confidence intervals: quantify uncertainty in estimates
- Effect size: magnitude independent of sample size
- Always report: metric values, p-values, correction method, CIs
- For hyperscanning: permute participant pairings as null

---

## Discussion Questions

1. You test 2016 electrode pairs and find 150 significant at p < 0.05 (uncorrected). After FDR correction, only 12 remain significant. What happened to the other 138? Were they "real"?

2. Your hyperscanning study shows inter-brain PLV = 0.28 with p = 0.03. A colleague's study shows PLV = 0.45 with p = 0.08. Which finding is "stronger"? How would you compare them?

3. Why might phase shuffling not be appropriate as a surrogate method for testing amplitude envelope correlation?

4. You want to claim that connectivity during cooperation exceeds connectivity during competition. Would you: (a) test each pair separately and count significant differences, or (b) use a permutation test on mean connectivity difference? Why?

5. A reviewer asks: "How did you determine your sample size / recording duration was sufficient?" How would you address this using the concepts from this notebook?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Conceptual | Null distribution with observed value | signal_1, highlight |
| 2 | 2×2 comparison | Original vs surrogate pairs | signal_1, signal_2 |
| 3 | Overlay | Original and surrogate PSDs | signal_1, signal_2 |
| 4 | Side-by-side histograms | Phase difference distributions | signal_1, signal_2 |
| 5 | 3 subplots | Time shift effect | signal_1 |
| 6 | Histogram + line | Null distribution with observed | signal_1, highlight |
| 7 | 1×3 grid | Significance examples | green/yellow/red |
| 8 | Histogram | False positives without correction | warning colors |
| 9 | P-values plot | Bonferroni threshold | signal_1, reference |
| 10 | BH procedure | Graphical FDR | signal_1, reference |
| 11 | Comparison | Uncorrected vs Bonferroni vs FDR | primary, secondary, accent |
| 12 | Histogram | Permutation test result | signal_1, highlight |
| 13 | Histogram + CI | Bootstrap distribution | signal_1 |
| 14 | Forest plot | Multiple pairs with CIs | signal_1 |
| 15 | Scatter | Effect size vs p-value | gradient |
| 16 | 2×2 grid | Full pipeline summary | viridis |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Random seeds propagated for reproducibility
- [ ] Edge cases handled (p=0, all significant, none significant)
- [ ] FDR implementation matches standard (Benjamini-Hochberg)
- [ ] Surrogate methods preserve appropriate properties
- [ ] Clear warnings about multiple comparisons
- [ ] Pipeline function integrates all steps cleanly