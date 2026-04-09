# D02: Mutual Information — Copilot Directive

## Metadata

- **Notebook ID**: D02
- **Title**: Mutual Information
- **Branch**: `feature/D02_mutual_information`
- **Duration**: 70 minutes
- **Prerequisites**: D01 (Entropy and Information)
- **Next**: D03 (Transfer Entropy)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Define mutual information as shared uncertainty reduction
2. Understand the relationship between MI and entropy (Venn diagram)
3. Compute joint entropy and conditional entropy
4. Implement MI estimation for continuous signals
5. Compare MI to correlation as a connectivity measure
6. Understand MI's advantages (captures nonlinear relationships)
7. Apply MI to neural signals and hyperscanning

---

## Required Imports
```
numpy, matplotlib, scipy.stats, scipy.special, typing, numpy.typing
Optional: sklearn.feature_selection (for mutual_info_regression)
Local: src/colors.py, src/plotting.py, src/signals.py, src/information.py
```

---

## Section 1: Introduction — Beyond Correlation

**Purpose**: Motivate MI as a connectivity measure.

**Content to cover**:
- So far: phase-based (PLV, PLI) and amplitude-based (envelope correlation) connectivity
- These capture LINEAR or PERIODIC relationships
- But neural relationships can be NONLINEAR!
- Example: signal Y increases when |X| is large (regardless of sign)
  - Correlation = 0 (no linear relationship)
  - But there IS a relationship!
- **Mutual Information**: captures ANY statistical dependency
- "How much does knowing X tell us about Y?"
- Powerful, general, but computationally harder
- Foundation for information-theoretic neuroscience

**Key message**: "MI detects relationships that correlation misses."

**Length**: 4-5 paragraphs

---

## Section 2: Intuition — Shared Information

**Purpose**: Build intuitive understanding.

**Content to cover**:
- **Thought experiment**: Two weather stations
  - Station A: records temperature in Paris
  - Station B: records temperature in Lyon
  - Knowing Paris temp reduces uncertainty about Lyon temp
  - They share information!
- **Mutual information** = shared uncertainty
- If X and Y are independent:
  - Knowing X tells nothing about Y
  - MI(X, Y) = 0
- If X determines Y completely:
  - Knowing X removes ALL uncertainty about Y
  - MI(X, Y) = H(Y) (maximum possible)
- MI is SYMMETRIC: MI(X, Y) = MI(Y, X)
- Measures "how much information is common to both"

**Visualization 1**:
- Scatter plots showing different relationships:
  - Independent (MI ≈ 0)
  - Linear correlation (MI > 0)
  - Nonlinear relationship (MI > 0, correlation ≈ 0)
- 1×3 grid with MI and correlation annotated on each
- Caption: "MI captures relationships correlation misses"

---

## Section 3: The Entropy Venn Diagram

**Purpose**: Visual framework for understanding MI.

**Content to cover**:
- **Venn diagram representation**:
  - Circle X: entropy H(X)
  - Circle Y: entropy H(Y)
  - Overlap: mutual information I(X; Y)
  - X only: H(X|Y) — uncertainty about X given Y
  - Y only: H(Y|X) — uncertainty about Y given X
  - Union: H(X, Y) — joint entropy
- **Key relationships**:
  - I(X; Y) = H(X) + H(Y) - H(X, Y)
  - I(X; Y) = H(X) - H(X|Y)
  - I(X; Y) = H(Y) - H(Y|X)
- MI = "what's shared" = "what we can predict about one from the other"

**Visualization 2**:
- Classic Venn diagram for entropy
- Two overlapping circles
- Regions labeled: H(X|Y), I(X;Y), H(Y|X)
- H(X,Y) = entire union
- Beautiful, clear, color-coded
- Caption: "The information Venn diagram"

**Visualization 3**:
- Three cases with Venn diagrams:
  - Independent: no overlap
  - Partially dependent: some overlap
  - Fully dependent: one circle inside other
- Show how MI changes in each case

---

## Section 4: Joint Entropy

**Purpose**: Define and compute joint entropy.

**Content to cover**:
- **Joint entropy** H(X, Y): uncertainty about the PAIR (X, Y)
- For discrete variables:
  $$H(X, Y) = -\sum_{x}\sum_{y} p(x, y) \log p(x, y)$$
- Where p(x, y) = joint probability distribution
- **Properties**:
  - H(X, Y) ≤ H(X) + H(Y) (subadditivity)
  - Equality when X and Y are independent
  - H(X, Y) ≥ max(H(X), H(Y))
- **For continuous signals**: need 2D binning
  - Create 2D histogram of (x, y) pairs
  - Normalize to get joint probability
  - Compute entropy

**Code to produce**:

Function 1: `compute_joint_histogram`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int | tuple[int, int] = 20
    method: str = "uniform"  # or "equiprobable"
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    (histogram_2d, x_edges, y_edges)
Compute 2D histogram for joint distribution
```

Function 2: `compute_joint_entropy`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int | tuple[int, int] = 20
    base: float = 2.0
Returns:
    float
Joint entropy H(X, Y) via 2D binning
```

**Visualization 4**:
- 2D histogram (heatmap) of joint distribution
- Example: two correlated Gaussian signals
- Colorbar showing probability density
- Marginal histograms on axes
- Caption: "Joint distribution visualized"

---

## Section 5: Conditional Entropy

**Purpose**: Define remaining uncertainty after observation.

**Content to cover**:
- **Conditional entropy** H(X|Y): uncertainty about X given we know Y
- Formula:
  $$H(X|Y) = H(X, Y) - H(Y)$$
- Alternative formula:
  $$H(X|Y) = -\sum_{x,y} p(x, y) \log p(x|y)$$
- **Interpretation**:
  - How much uncertainty remains about X after observing Y?
  - If Y tells us everything about X: H(X|Y) = 0
  - If Y tells us nothing about X: H(X|Y) = H(X)
- **Relationship to MI**:
  - I(X; Y) = H(X) - H(X|Y)
  - MI = initial uncertainty minus remaining uncertainty
  - MI = uncertainty REDUCED by knowing the other variable

**Code to produce**:

Function 3: `compute_conditional_entropy`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int | tuple[int, int] = 20
    base: float = 2.0
Returns:
    float
Conditional entropy H(X|Y)
Computed as H(X,Y) - H(Y)
```

**Visualization 5**:
- Bar chart showing entropy decomposition:
  - Total bar: H(X)
  - Split into: H(X|Y) + I(X;Y)
- Another bar: H(Y)
  - Split into: H(Y|X) + I(X;Y)
- Show I(X;Y) is the common part
- Caption: "MI is uncertainty reduction"

---

## Section 6: Mutual Information — The Formula

**Purpose**: Formal definition and computation.

**Content to cover**:
- **Definition 1** (via joint and marginal):
  $$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$
- **Definition 2** (via entropies):
  $$I(X; Y) = H(X) + H(Y) - H(X, Y)$$
- **Definition 3** (via conditional entropy):
  $$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$
- All three are equivalent!
- **Properties**:
  - I(X; Y) ≥ 0 always
  - I(X; Y) = 0 iff X and Y are independent
  - I(X; Y) = I(Y; X) (symmetric)
  - I(X; X) = H(X) (self-information)
- **For continuous signals**: use binning approach

**Code to produce**:

Function 4: `compute_mutual_information`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int | tuple[int, int] = 20
    base: float = 2.0
    method: str = "binning"  # or "knn" for future
Returns:
    float
Mutual information I(X; Y)
Compute via H(X) + H(Y) - H(X,Y)
```

**Visualization 6**:
- Compute MI for signal pairs with varying correlation
- X-axis: true correlation (0 to 1)
- Y-axis: MI value
- Show MI increases with dependency
- Caption: "MI increases with statistical dependency"

---

## Section 7: MI vs Correlation — The Key Difference

**Purpose**: Demonstrate MI's advantage.

**Content to cover**:
- **Correlation** measures LINEAR relationship
- **MI** measures ANY statistical dependency
- **Example 1**: Linear relationship
  - Y = aX + noise
  - Both correlation and MI are high
- **Example 2**: Nonlinear relationship
  - Y = X² + noise
  - Correlation ≈ 0 (no linear trend)
  - MI > 0 (there IS a dependency!)
- **Example 3**: Circular relationship
  - Y = sin(X)
  - Correlation depends on range
  - MI captures the functional relationship
- MI is more general but harder to estimate

**Code to produce**:

Function 5: `generate_nonlinear_relationship`
```
Parameters:
    n_samples: int
    relationship: str = "quadratic"  # or "sinusoidal", "circular", "xor"
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate X, Y with specified nonlinear relationship
```

**Visualization 7**:
- 2×3 grid showing different relationships:
  - Row 1: scatter plots
  - Row 2: bar chart comparing correlation vs MI
- Columns: linear, quadratic, sinusoidal
- Highlight: quadratic has low correlation but high MI
- Key demonstration figure!

**Visualization 8**:
- XOR-like relationship (extreme example)
- Four clusters in quadrants
- Correlation = 0, MI > 0
- Caption: "MI detects this pattern, correlation cannot"

---

## Section 8: Normalized Mutual Information

**Purpose**: Scale MI for comparison across different signals.

**Content to cover**:
- Raw MI depends on entropy of variables
- Hard to compare across different signals
- **Normalized MI** options:
  1. NMI = I(X;Y) / sqrt(H(X) × H(Y))
  2. NMI = I(X;Y) / max(H(X), H(Y))
  3. NMI = I(X;Y) / min(H(X), H(Y))
  4. NMI = 2 × I(X;Y) / (H(X) + H(Y))
- Range: 0 (independent) to 1 (perfectly dependent)
- Allows comparison across different signal types
- Similar to how we normalize other metrics

**Code to produce**:

Function 6: `compute_normalized_mi`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int = 20
    normalization: str = "geometric"  # or "max", "min", "arithmetic"
Returns:
    float
Normalized mutual information (range 0-1)
```

**Visualization 9**:
- Compare raw MI vs normalized MI
- Signals with different entropies
- Show NMI is more comparable
- Bar charts side by side

---

## Section 9: Estimation Challenges

**Purpose**: Address practical difficulties.

**Content to cover**:
- **Challenge 1**: Binning choice
  - Too few bins: underestimate MI
  - Too many bins: overestimate (sparse sampling bias)
  - Optimal depends on sample size and relationship
- **Challenge 2**: Bias
  - MI estimates are BIASED UPWARD
  - Even independent signals show positive MI due to finite sampling!
  - More bins = more bias
- **Challenge 3**: Computational cost
  - 2D histograms scale with bins²
  - For matrices: n² pairs to compute
- **Solutions**:
  - Adaptive binning
  - Bias correction (shuffled surrogates)
  - KNN-based estimators (Kraskov et al.)
  - Permutation testing for significance

**Code to produce**:

Function 7: `compute_mi_bias_corrected`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int = 20
    n_surrogates: int = 100
    seed: int | None = None
Returns:
    Tuple[float, float]
    (mi_corrected, mi_raw)
Bias-corrected MI using surrogate subtraction
MI_corrected = MI_observed - mean(MI_surrogates)
```

**Visualization 10**:
- Bias demonstration:
  - Two INDEPENDENT signals
  - Compute MI with different bin counts
  - Show MI > 0 due to bias
  - Show corrected MI ≈ 0
- Line plot: bins vs MI (raw and corrected)

---

## Section 10: Surrogate Testing for MI

**Purpose**: Statistical significance for MI.

**Content to cover**:
- Same principle as C03: compare to null distribution
- **Null hypothesis**: X and Y are independent
- **Surrogate generation**: 
  - Shuffle one signal (destroy temporal relationship)
  - Or phase shuffle (preserve spectrum)
- **Procedure**:
  1. Compute observed MI
  2. Generate N surrogates, compute MI for each
  3. P-value = proportion of surrogates ≥ observed
- **Bonus**: surrogate mean provides bias estimate
  - MI_corrected = MI_observed - mean(MI_surrogates)

**Code to produce**:

Function 8: `mi_significance_test`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_bins: int = 20
    n_surrogates: int = 500
    seed: int | None = None
Returns:
    dict[str, float]
    {"mi_observed": ..., "mi_corrected": ..., "pvalue": ..., "null_mean": ..., "null_std": ...}
Complete significance test for MI
```

**Visualization 11**:
- Null distribution from surrogates
- Observed MI marked
- P-value region shaded
- Bias (null mean) annotated
- Caption: "Testing MI significance with surrogates"

---

## Section 11: MI for Time Series — Considerations

**Purpose**: Apply MI to neural signals specifically.

**Content to cover**:
- EEG signals are TIME SERIES, not static samples
- **Option 1**: Treat each time point as a sample
  - Simple, but ignores temporal structure
  - Assumes stationarity
- **Option 2**: Sliding window MI
  - Compute MI in short windows
  - Get MI over time
  - Captures dynamic changes
- **Option 3**: Time-lagged MI
  - MI between X(t) and Y(t + τ)
  - Can reveal delayed relationships
  - Preview of transfer entropy!
- **Option 4**: MI of derived features
  - MI of phase time series
  - MI of amplitude envelopes
  - Can be more interpretable

**Code to produce**:

Function 9: `compute_mi_timeseries`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    window_samples: int
    step_samples: int
    n_bins: int = 20
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (time_points, mi_values)
Sliding window MI over time
```

Function 10: `compute_mi_lagged`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    max_lag_samples: int
    n_bins: int = 20
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (lags, mi_values)
MI as function of time lag
MI(X(t), Y(t+lag)) for various lags
```

**Visualization 12**:
- Time-resolved MI between two signals
- Signals have varying coupling over time
- Top: both signals
- Bottom: MI over time
- Show MI tracks coupling changes

**Visualization 13**:
- MI vs lag plot
- Peak at non-zero lag indicates delayed relationship
- Symmetric if relationship is bidirectional
- Caption: "Time-lagged MI reveals delayed dependencies"

---

## Section 12: MI Connectivity Matrix

**Purpose**: Scale up to multi-channel analysis.

**Content to cover**:
- Apply MI to all channel pairs → connectivity matrix
- Same structure as C02 but with MI values
- **Considerations**:
  - MI is symmetric: M[i,j] = M[j,i]
  - Diagonal: MI(X, X) = H(X) (often set to NaN for consistency)
  - Computational cost: n² pairs × binning
- **Comparison to other metrics**:
  - MI captures more than PLV or correlation
  - But harder to estimate reliably
  - May need more data

**Code to produce**:

Function 11: `compute_mi_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    n_bins: int = 20
    normalize: bool = True
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
MI connectivity matrix
If normalize=True, use normalized MI
```

**Visualization 14**:
- Compare connectivity matrices:
  - Correlation matrix
  - PLV matrix
  - MI matrix
- 1×3 grid with same data
- Highlight differences (especially for nonlinear relationships)
- Caption: "MI captures dependencies others miss"

---

## Section 13: MI for Hyperscanning

**Purpose**: Apply to two-brain analysis.

**Content to cover**:
- Between-brain MI: same as between-brain PLV structure
- No volume conduction concerns (separate heads)
- **Interpretation**:
  - High inter-brain MI: one participant's signal predicts the other's
  - Could indicate: shared attention, synchronization, communication
- **Advantages for hyperscanning**:
  - Captures nonlinear inter-brain coupling
  - Doesn't assume sinusoidal relationship
- **Practical considerations**:
  - Need sufficient data for reliable estimation
  - Bias correction important
  - Compare to shuffled participant pairings

**Code to produce**:

Function 12: `compute_mi_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    n_bins: int = 20
    normalize: bool = True
Returns:
    dict[str, NDArray[np.float64]]
    {"within_p1": ..., "within_p2": ..., "between": ...}
MI matrices for hyperscanning analysis
```

**Visualization 15**:
- Hyperscanning MI analysis
- Between-participant MI matrix
- Circular plot showing inter-brain MI connections
- Subject colors from palette

---

## Section 14: Advanced: KNN-Based MI Estimation

**Purpose**: Introduce more sophisticated estimation (optional deep dive).

**Content to cover**:
- Binning has limitations (bias, bin choice)
- **KNN estimator** (Kraskov-Stögbauer-Grassberger):
  - Uses k-nearest neighbors instead of bins
  - Adaptive "bins" based on local density
  - Less bias, works in high dimensions
  - Standard in neuroscience (e.g., JIDT, MNE)
- **Concept**:
  - For each point, find k nearest neighbors
  - Estimate local density from distances
  - Combine to estimate entropy/MI
- Implementation beyond this workshop, but important to know exists

**Note**: Reference libraries that implement KNN MI (JIDT, sklearn, npeet)

**Visualization 16** (optional):
- Conceptual illustration of KNN vs binning
- Point cloud with:
  - Binning grid overlay
  - KNN neighbor circles
- Caption: "KNN adapts to local density"

---

## Section 15: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Venn Diagram Verification
- Generate two correlated Gaussian signals
- Compute: H(X), H(Y), H(X,Y), H(X|Y), H(Y|X), I(X;Y)
- Verify: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- Verify: I(X;Y) = H(X) + H(Y) - H(X,Y)
- Draw Venn diagram with computed values

**Exercise 2**: MI vs Correlation Showdown
- Create three relationships:
  - Linear: Y = 2X + noise
  - Quadratic: Y = X² + noise
  - Sinusoidal: Y = sin(2πX) + noise
- Compute correlation and MI for each
- Which metric better captures each relationship?
- Create comparison figure

**Exercise 3**: Binning Sensitivity
- Two correlated signals (r = 0.5)
- Compute MI with bins: 5, 10, 20, 50, 100
- Plot MI vs bins
- At what point does bias become problematic?
- Apply surrogate correction — does it help uniformly?

**Exercise 4**: Surrogate Significance Test
- Generate independent signals
- Compute MI (should be ~0 but biased positive)
- Run surrogate test (500 surrogates)
- Is observed MI significant?
- Repeat with correlated signals — now significant?

**Exercise 5**: Time-Resolved MI
- Generate signals with time-varying coupling:
  - 0-5s: independent
  - 5-10s: coupled
  - 10-15s: independent
- Compute sliding window MI
- Can you detect the coupling period?
- What window size works best?

**Exercise 6**: MI Connectivity Matrix
- 6-channel synthetic data
- Some pairs linearly related, some nonlinearly related
- Compute correlation matrix and MI matrix
- Identify pairs where MI >> correlation
- These are the nonlinear relationships!

---

## Section 16: Functions for src/information.py

**Purpose**: Add functions to information module.

**Add to src/information.py**:
- `compute_joint_histogram`
- `compute_joint_entropy`
- `compute_conditional_entropy`
- `compute_mutual_information`
- `generate_nonlinear_relationship`
- `compute_normalized_mi`
- `compute_mi_bias_corrected`
- `mi_significance_test`
- `compute_mi_timeseries`
- `compute_mi_lagged`
- `compute_mi_matrix`
- `compute_mi_hyperscanning`

---

## Summary

**Key takeaways to list**:
- Mutual Information measures ANY statistical dependency (not just linear)
- MI = shared information = H(X) + H(Y) - H(X,Y)
- MI = uncertainty reduction when knowing the other variable
- MI ≥ 0, with MI = 0 iff independent
- MI is symmetric: I(X;Y) = I(Y;X)
- Key advantage: captures nonlinear relationships (correlation cannot!)
- Estimation requires binning — careful choice matters
- MI estimates are biased upward — use surrogate correction
- Significance testing via permutation/surrogates
- Normalized MI allows comparison across different signals
- For hyperscanning: inter-brain MI captures general dependencies
- Foundation for transfer entropy (next notebook!)

---

## Discussion Questions

1. Two brain regions show high correlation but low MI. Is this possible? What would it mean?

2. You find that MI between participants during a social task is 0.3 bits. How would you interpret this number? What context would you need?

3. A colleague argues "just use correlation, it's simpler." When would you push back and insist on MI? When might correlation be sufficient?

4. Time-lagged MI shows a peak at +200ms (signal X leads Y by 200ms). What does this suggest? How is this different from what transfer entropy would tell us?

5. Your MI estimate is 0.5 bits, but after surrogate correction it becomes 0.1 bits. What happened? Which value should you report?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 1×3 scatter plots | Different relationships | signal_1 |
| 2 | Venn diagram | Entropy relationships | subject_1, subject_2 |
| 3 | 3 Venn diagrams | Independence to full dependence | gradient |
| 4 | 2D histogram + marginals | Joint distribution | viridis |
| 5 | Bar chart | Entropy decomposition | signal_1, signal_2, overlap |
| 6 | Line plot | MI vs correlation strength | signal_1 |
| 7 | 2×3 grid | MI vs correlation comparison | signal colors |
| 8 | Scatter plot | XOR-like relationship | signal_1 |
| 9 | Bar comparison | Raw vs normalized MI | signal_1, signal_2 |
| 10 | Line plot | Bias vs bins | signal_1, corrected |
| 11 | Histogram | MI null distribution | signal_1, highlight |
| 12 | 2 subplots | Time-resolved MI | signal_1, signal_2 |
| 13 | Single plot | MI vs lag | signal_1 |
| 14 | 1×3 heatmaps | Correlation vs PLV vs MI | viridis |
| 15 | Heatmap + circular | Hyperscanning MI | subject_1, subject_2 |
| 16 | Conceptual | KNN vs binning | signal_1, signal_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Handle edge cases (empty bins, zero probabilities)
- [ ] Efficient 2D histogram computation
- [ ] Consistent with D01 entropy functions
- [ ] Surrogate methods reuse code from C03
- [ ] Clear documentation of bias issues
- [ ] Normalization options clearly documented