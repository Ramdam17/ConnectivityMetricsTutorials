# D03: Transfer Entropy — Copilot Directive

## Metadata

- **Notebook ID**: D03
- **Title**: Transfer Entropy
- **Branch**: `feature/D03_transfer_entropy`
- **Duration**: 75 minutes
- **Prerequisites**: D02 (Mutual Information), B04 (Wavelets — for time-lagged concepts)
- **Next**: E01 (Graph Theory Basics)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand the difference between correlation, MI, and directed information flow
2. Define transfer entropy as conditional mutual information
3. Understand the concept of Wiener-Granger causality
4. Implement transfer entropy estimation
5. Interpret TE in terms of prediction improvement
6. Address common pitfalls (bias, stationarity, lag selection)
7. Apply TE to hyperscanning for studying directed inter-brain influence

---

## Required Imports
```
numpy, matplotlib, scipy.stats, scipy.signal, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/information.py
```

---

## Section 1: Introduction — The Directionality Question

**Purpose**: Motivate the need for directed connectivity measures.

**Content to cover**:
- So far: symmetric connectivity measures
  - Correlation: r(X, Y) = r(Y, X)
  - PLV: PLV(X, Y) = PLV(Y, X)
  - MI: I(X; Y) = I(Y; X)
- But relationships can be DIRECTIONAL!
  - Teacher speaks → student listens
  - Leader acts → follower responds
  - Brain region A drives → brain region B responds
- **The question**: Does X INFLUENCE Y, or does Y INFLUENCE X, or both?
- **Transfer entropy**: measures directed information flow
- "How much does knowing X's PAST help predict Y's FUTURE?"
- Critical for hyperscanning: who influences whom during interaction?

**Key message**: "Transfer entropy answers: Does information FLOW from X to Y?"

**Length**: 4-5 paragraphs

---

## Section 2: Intuition — Prediction and Causality

**Purpose**: Build conceptual understanding.

**Content to cover**:
- **Thought experiment**: Predicting tomorrow's weather
  - Using only past weather here → some accuracy
  - Adding past weather from neighboring city → better accuracy?
  - If yes: neighboring city provides INFORMATION about our weather
  - Information FLOWS from neighbor to here
- **Wiener-Granger causality principle** (1956, 1969):
  - "X Granger-causes Y if past X helps predict future Y, beyond past Y alone"
  - Not true causality! But predictive influence
  - Foundation of transfer entropy
- **Transfer entropy** = information-theoretic version of Granger causality
  - Uses MI instead of linear regression
  - Captures nonlinear influences
  - More general, fewer assumptions

**Visualization 1**:
- Conceptual diagram of prediction:
  - Predict Y_future from Y_past only → uncertainty
  - Predict Y_future from Y_past AND X_past → less uncertainty?
  - Arrow showing TE = reduction in uncertainty
- Caption: "TE measures prediction improvement from X's past"

---

## Section 3: From Mutual Information to Transfer Entropy

**Purpose**: Mathematical development from MI to TE.

**Content to cover**:
- **Recall MI**: I(X; Y) = uncertainty reduction about Y from knowing X
- **Problem**: MI is about CURRENT values, ignores time
- **Solution**: Consider PAST and FUTURE
- **Notation**:
  - X_past = X(t-τ), X(t-2τ), ... (history of X)
  - Y_past = Y(t-τ), Y(t-2τ), ... (history of Y)
  - Y_future = Y(t) (what we want to predict)
- **Transfer entropy** from X to Y:
  $$TE_{X \to Y} = I(Y_{future}; X_{past} | Y_{past})$$
- Read as: "Information that X's past provides about Y's future, given Y's past"
- **Conditional MI**: information gained BEYOND what Y's past already tells us

**Visualization 2**:
- Timeline diagram:
  - X signal with "past" region highlighted
  - Y signal with "past" and "future" points marked
  - Arrows showing: Y_past → Y_future (baseline prediction)
  - Additional arrow: X_past → Y_future (extra information = TE)
- Caption: "TE quantifies the X_past → Y_future information flow"

---

## Section 4: The Transfer Entropy Formula

**Purpose**: Formal mathematical definition.

**Content to cover**:
- **Conditional mutual information**:
  $$I(A; B | C) = H(A | C) - H(A | B, C)$$
  - Information A provides about B, given we already know C
- **Transfer entropy** X → Y:
  $$TE_{X \to Y} = H(Y_t | Y_{past}) - H(Y_t | Y_{past}, X_{past})$$
- **Interpretation**:
  - H(Y_t | Y_past) = uncertainty about Y's future given only Y's past
  - H(Y_t | Y_past, X_past) = uncertainty given BOTH pasts
  - TE = reduction in uncertainty from adding X's information
- **Equivalently**:
  $$TE_{X \to Y} = I(Y_t; X_{past} | Y_{past})$$
- **Properties**:
  - TE ≥ 0 (adding information can't hurt prediction)
  - TE = 0 if X provides no additional predictive power
  - TE_{X→Y} ≠ TE_{Y→X} in general (asymmetric!)
- **Units**: bits (if using log₂)

**Visualization 3**:
- Venn diagram extension for conditional MI
- Three circles: Y_future, Y_past, X_past
- TE = specific intersection region
- Caption: "TE is conditional mutual information"

---

## Section 5: Implementing Transfer Entropy

**Purpose**: Practical computation.

**Content to cover**:
- **Step 1**: Define embedding parameters
  - k = history length for Y (how many past samples)
  - l = history length for X
  - τ = time delay between samples
- **Step 2**: Create state vectors
  - Y_past vector: [Y(t-τ), Y(t-2τ), ..., Y(t-kτ)]
  - X_past vector: [X(t-τ), X(t-2τ), ..., X(t-lτ)]
  - Y_future: Y(t)
- **Step 3**: Estimate entropies
  - H(Y_t | Y_past) via binning or KNN
  - H(Y_t | Y_past, X_past) via higher-dimensional binning
- **Step 4**: Compute TE = difference
- **Challenge**: High-dimensional binning with limited data!

**Code to produce**:

Function 1: `create_embedding_vectors`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k: int = 1  # history length for target (Y)
    l: int = 1  # history length for source (X)
    tau: int = 1  # delay in samples
Returns:
    Tuple[NDArray, NDArray, NDArray]
    (y_future, y_past, x_past)
Create embedded state vectors for TE computation
y_past shape: (n_valid, k)
x_past shape: (n_valid, l)
y_future shape: (n_valid,)
```

Function 2: `compute_transfer_entropy`
```
Parameters:
    x: NDArray[np.float64]  # source signal
    y: NDArray[np.float64]  # target signal
    k: int = 1  # Y history length
    l: int = 1  # X history length
    tau: int = 1  # embedding delay
    n_bins: int = 8  # bins per dimension
    base: float = 2.0
Returns:
    float
Transfer entropy from X to Y: TE_{X→Y}
```

**Visualization 4**:
- Step-by-step illustration of embedding:
  - Original time series X and Y
  - Show how vectors are constructed
  - Highlight which points form each embedded vector
- Caption: "Constructing state vectors for TE"

---

## Section 6: Simple Example — Coupled Oscillators

**Purpose**: Demonstrate TE on clear example.

**Content to cover**:
- **Scenario**: Two coupled oscillators
  - X drives Y with some delay
  - Y = f(Y_past) + coupling × X_past + noise
- **Expected result**:
  - TE_{X→Y} > 0 (X influences Y)
  - TE_{Y→X} ≈ 0 (Y doesn't influence X)
- **Demonstration**:
  - Create synthetic coupled signals
  - Compute TE in both directions
  - Verify asymmetry matches ground truth

**Code to produce**:

Function 3: `generate_coupled_signals`
```
Parameters:
    n_samples: int
    coupling_x_to_y: float = 0.5
    coupling_y_to_x: float = 0.0
    delay_samples: int = 5
    noise_level: float = 0.1
    seed: int | None = None
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
Generate coupled AR-like signals with specified directional coupling
```

**Visualization 5**:
- Two coupled signals (time series)
- Bar chart: TE_{X→Y} vs TE_{Y→X}
- Clear asymmetry showing X drives Y
- Caption: "TE captures directional coupling"

**Visualization 6**:
- Vary coupling strength from 0 to 1
- Plot TE vs coupling strength
- Show TE increases with coupling
- Caption: "TE scales with coupling strength"

---

## Section 7: Bidirectional Coupling

**Purpose**: Show TE with mutual influence.

**Content to cover**:
- Real systems often have BIDIRECTIONAL coupling
- X influences Y AND Y influences X
- TE can detect both:
  - TE_{X→Y} > 0 AND TE_{Y→X} > 0
- **Net information flow**:
  - Net_{X→Y} = TE_{X→Y} - TE_{Y→X}
  - Positive: X dominates
  - Negative: Y dominates
  - Zero: balanced bidirectional coupling
- Important for hyperscanning: often both participants influence each other!

**Code to produce**:

Function 4: `compute_net_transfer_entropy`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
Returns:
    dict[str, float]
    {"te_x_to_y": ..., "te_y_to_x": ..., "net_te": ..., "dominant_direction": ...}
Compute TE in both directions and net flow
```

**Visualization 7**:
- Bidirectionally coupled signals
- Bar chart showing both TEs and net TE
- Arrow diagram indicating dominant direction
- Caption: "Net TE reveals dominant influence direction"

**Visualization 8**:
- Vary coupling asymmetry
- X-axis: coupling_y_to_x (with coupling_x_to_y fixed)
- Y-axis: Net TE
- Show transition from X-dominant to balanced to Y-dominant
- Caption: "Net TE tracks coupling asymmetry"

---

## Section 8: Parameter Selection

**Purpose**: Guide on choosing embedding parameters.

**Content to cover**:
- **Critical parameters**:
  - k (Y history length): too short = miss dynamics, too long = sparse estimation
  - l (X history length): same trade-off
  - τ (embedding delay): should capture relevant timescales
  - n_bins: same issues as MI (bias vs resolution)
- **Guidelines**:
  - Start simple: k = l = 1, τ = relevant lag
  - τ often set by expected delay (e.g., 50-200ms for neural)
  - Increase k, l if system has longer memory
  - More parameters = need more data!
- **Advanced**: use auto-correlation or mutual information to guide τ
- **Rule of thumb**: total dimensions (k + l + 1) × bins should be << n_samples

**Code to produce**:

Function 5: `suggest_embedding_delay`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    method: str = "autocorr"  # or "mi_minimum"
Returns:
    int
Suggest embedding delay based on signal properties
autocorr: first zero crossing of autocorrelation
mi_minimum: first minimum of auto-MI
```

Function 6: `compute_te_scan_parameters`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k_range: tuple[int, int] = (1, 5)
    l_range: tuple[int, int] = (1, 5)
    tau_range: tuple[int, int] = (1, 20)
    n_bins: int = 8
Returns:
    dict[str, NDArray]
    {"k_values": ..., "l_values": ..., "tau_values": ..., "te_values": ...}
Scan TE across parameter ranges to find optimal
```

**Visualization 9**:
- Heatmap: TE as function of k and l (fixed τ)
- Show how TE changes with history lengths
- Identify "plateau" region for robust estimation
- Caption: "TE sensitivity to history length"

**Visualization 10**:
- Line plot: TE as function of τ
- Peak at true coupling delay
- Can reveal the timescale of influence
- Caption: "TE peaks at the true coupling delay"

---

## Section 9: Bias and Significance Testing

**Purpose**: Address estimation challenges.

**Content to cover**:
- **TE bias**: even more severe than MI bias!
  - Higher dimensional estimation
  - Positive bias even for independent signals
- **Bias correction methods**:
  1. Surrogate subtraction (as with MI)
  2. Effective transfer entropy: TE_eff = TE - TE_shuffled
  3. Statistical significance testing
- **Surrogate generation for TE**:
  - Shuffle source signal (destroy X→Y relationship)
  - Keep target intact (preserve Y dynamics)
  - Compute TE on surrogate
  - Repeat many times → null distribution
- **Significance**: observed TE vs null distribution

**Code to produce**:

Function 7: `compute_te_surrogate`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
    shuffle_method: str = "random"  # or "circular", "block"
    seed: int | None = None
Returns:
    float
TE computed with shuffled source signal
```

Function 8: `te_significance_test`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
    n_surrogates: int = 500
    seed: int | None = None
Returns:
    dict[str, float]
    {"te_observed": ..., "te_effective": ..., "pvalue": ..., "null_mean": ..., "null_std": ...}
Complete significance test for transfer entropy
```

**Visualization 11**:
- Null distribution from surrogates
- Observed TE marked
- Effective TE (after bias subtraction) annotated
- P-value shaded region
- Caption: "Testing TE significance"

---

## Section 10: TE vs Granger Causality

**Purpose**: Compare to classical approach.

**Content to cover**:
- **Granger Causality (GC)**:
  - Linear autoregressive models
  - Compare prediction error with/without X
  - F-test for significance
  - Assumes linear relationships!
- **Transfer Entropy**:
  - Information-theoretic (nonparametric)
  - Captures nonlinear influences
  - No linearity assumption
  - More general but harder to estimate
- **Relationship**:
  - For Gaussian signals: GC and TE are equivalent (up to scaling)
  - For non-Gaussian/nonlinear: TE captures more
- **When to use which**:
  - Linear systems, want interpretability: GC
  - Potentially nonlinear, want generality: TE
  - Often good to compare both!

**Visualization 12**:
- Two scenarios:
  - Linear coupling: GC and TE both detect
  - Nonlinear coupling: only TE detects
- 2×2 comparison showing GC vs TE for each case
- Caption: "TE captures nonlinear influences"

---

## Section 11: TE for Neural Signals

**Purpose**: Application to EEG-like data.

**Content to cover**:
- **Preprocessing considerations**:
  - Band-pass filter to frequency of interest
  - TE on raw broadband can be noisy
  - Often compute TE on phase or amplitude time series
- **Typical parameters for EEG**:
  - τ corresponding to ~10-50ms (depends on expected delay)
  - k, l = 1-3 (short memory for oscillations)
  - n_bins = 6-10 (balance bias and resolution)
- **Frequency-specific TE**:
  - Filter to band → compute TE
  - Different bands may show different information flow
- **Time-resolved TE**:
  - Sliding window approach
  - Track changes in information flow over time

**Code to produce**:

Function 9: `compute_te_bandlimited`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    k: int = 1
    l: int = 1
    tau_ms: float = 20.0  # delay in milliseconds
    n_bins: int = 8
Returns:
    float
TE computed on band-limited signals
Converts tau_ms to samples internally
```

Function 10: `compute_te_timeseries`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    window_samples: int
    step_samples: int
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (time_centers, te_values)
Time-resolved transfer entropy
```

**Visualization 13**:
- Band-specific TE analysis
- Compute TE in theta, alpha, beta, gamma bands
- Bar chart comparing TE across bands
- Different bands may show different information flow patterns
- Caption: "Information flow can be frequency-specific"

**Visualization 14**:
- Time-resolved TE between two signals
- Top: both signals
- Bottom: TE over time (both directions)
- Show how information flow changes during task
- Caption: "Dynamic information flow"

---

## Section 12: TE Connectivity Matrix

**Purpose**: Scale up to multi-channel analysis.

**Content to cover**:
- Apply TE to all directed pairs → ASYMMETRIC matrix
- M[i, j] = TE from channel i to channel j
- M[i, j] ≠ M[j, i] in general!
- **Interpretation**:
  - Rows: sources (information senders)
  - Columns: targets (information receivers)
  - High row sum: channel broadcasts to many
  - High column sum: channel receives from many
- **Net flow matrix**:
  - Net[i, j] = TE[i, j] - TE[j, i]
  - Antisymmetric: Net[i, j] = -Net[j, i]
  - Shows dominant direction for each pair

**Code to produce**:

Function 11: `compute_te_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
Directed TE matrix: entry [i,j] = TE from i to j
Diagonal set to NaN
NOT symmetric!
```

Function 12: `compute_net_te_matrix`
```
Parameters:
    te_matrix: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Net TE matrix: Net[i,j] = TE[i,j] - TE[j,i]
Antisymmetric matrix
```

**Visualization 15**:
- TE matrix (asymmetric heatmap)
- Note asymmetry explicitly
- Identify strongest directed connections
- Caption: "Directed TE matrix — note asymmetry"

**Visualization 16**:
- Net TE matrix
- Diverging colormap (red = i→j dominates, blue = j→i dominates)
- Antisymmetric pattern visible
- Caption: "Net information flow direction"

---

## Section 13: TE for Hyperscanning

**Purpose**: The key application for this workshop!

**Content to cover**:
- **Hyperscanning question**: Who leads the interaction?
- **Inter-brain TE**:
  - TE_{P1→P2}: information flow from participant 1 to participant 2
  - TE_{P2→P1}: information flow from participant 2 to participant 1
  - Net TE: reveals dominant direction
- **Applications**:
  - Leader-follower dynamics in joint action
  - Teacher-student knowledge transfer
  - Therapist-patient rapport
  - Parent-child interaction
- **Advantages**:
  - Directed (unlike PLV, MI)
  - Captures nonlinear influences
  - Time-resolved analysis possible
- **Challenges**:
  - Need sufficient data
  - Parameter selection critical
  - Computational cost

**Code to produce**:

Function 13: `compute_te_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]  # (n_channels, n_samples)
    data_p2: NDArray[np.float64]
    k: int = 1
    l: int = 1
    tau: int = 1
    n_bins: int = 8
Returns:
    dict[str, NDArray[np.float64]]
    {"te_p1_to_p2": ..., "te_p2_to_p1": ..., "net_te": ...}
Inter-brain transfer entropy analysis
Each matrix is (n_channels_p1, n_channels_p2) or similar
```

Function 14: `compute_global_te_flow`
```
Parameters:
    te_p1_to_p2: NDArray[np.float64]
    te_p2_to_p1: NDArray[np.float64]
Returns:
    dict[str, float]
    {"mean_p1_to_p2": ..., "mean_p2_to_p1": ..., "net_flow": ..., "leader": ...}
Global summary of inter-brain information flow
```

**Visualization 17**:
- Hyperscanning TE analysis
- Two matrices side by side:
  - TE: P1 → P2
  - TE: P2 → P1
- Net flow summary below
- Arrow indicating dominant direction
- Subject colors from palette

**Visualization 18**:
- Time-resolved hyperscanning TE
- Track leadership changes over interaction
- Top: task markers (e.g., conversation turns)
- Bottom: TE in both directions over time
- Net TE showing who leads when
- Caption: "Dynamic leadership in social interaction"

---

## Section 14: Limitations and Cautions

**Purpose**: Honest assessment of TE challenges.

**Content to cover**:

**Limitation 1**: Data requirements
- TE needs MORE data than MI (higher dimensions)
- Short recordings may give unreliable estimates
- Rule of thumb: thousands of samples minimum

**Limitation 2**: Parameter sensitivity
- Results depend on k, l, τ, n_bins choices
- Should report sensitivity analysis
- Results robust across parameters = more trustworthy

**Limitation 3**: Not true causality!
- TE measures PREDICTIVE information, not causation
- Confounds (common driver) can create apparent TE
- Cannot replace experimental manipulation

**Limitation 4**: Stationarity assumption
- Standard TE assumes stationary statistics
- Neural signals are often non-stationary
- Time-resolved TE helps but has own issues

**Limitation 5**: Computational cost
- O(n²) for matrix, expensive per pair
- High-dimensional histograms
- May need approximations for large datasets

**Visualization 19**:
- Common confound illustration:
  - Hidden variable Z drives both X and Y
  - Creates apparent TE even without direct connection
- Diagram showing confound scenario
- Caption: "TE is not causation — beware confounds"

---

## Section 15: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Basic TE Computation
- Create unidirectionally coupled signals (X → Y)
- Compute TE in both directions
- Verify: TE_{X→Y} >> TE_{Y→X}
- Vary coupling strength, plot TE vs coupling

**Exercise 2**: Bidirectional Coupling
- Create bidirectionally coupled signals with asymmetry
- Compute net TE
- Verify net TE sign matches dominant coupling direction
- What happens when coupling is balanced?

**Exercise 3**: Delay Detection
- Create coupled signals with specific delay (e.g., 10 samples)
- Compute TE as function of τ
- Can you recover the true delay from TE peak?
- How sharp is the peak?

**Exercise 4**: Parameter Sensitivity
- Fixed coupled signals
- Vary k from 1 to 5, compute TE each time
- Vary n_bins from 4 to 20, compute TE each time
- Which parameters most affect the result?
- What combination seems most robust?

**Exercise 5**: Significance Testing
- Generate independent signals (no coupling)
- Compute TE (will be biased positive)
- Run surrogate test — is it significant?
- Repeat with coupled signals — now significant?
- Compare effective TE (bias-corrected) across both

**Exercise 6**: Hyperscanning Scenario
- Generate two "participants":
  - P1 leads with 80% coupling to P2
  - P2 follows with 20% coupling to P1
- Compute inter-brain TE in both directions
- Compute net TE and verify P1 leads
- Time-resolve: can you see consistent leadership?

**Exercise 7**: Comparison to Correlation
- Create nonlinearly coupled signals
- Compute correlation, MI, and TE
- Which best captures the relationship?
- Which captures the directionality?

---

## Section 16: Functions for src/information.py

**Purpose**: Add TE functions to information module.

**Add to src/information.py**:
- `create_embedding_vectors`
- `compute_transfer_entropy`
- `generate_coupled_signals`
- `compute_net_transfer_entropy`
- `suggest_embedding_delay`
- `compute_te_scan_parameters`
- `compute_te_surrogate`
- `te_significance_test`
- `compute_te_bandlimited`
- `compute_te_timeseries`
- `compute_te_matrix`
- `compute_net_te_matrix`
- `compute_te_hyperscanning`
- `compute_global_te_flow`

---

## Summary

**Key takeaways to list**:
- Transfer Entropy measures DIRECTED information flow
- TE_{X→Y} = "How much does X's past help predict Y's future, beyond Y's past?"
- TE is asymmetric: TE_{X→Y} ≠ TE_{Y→X}
- Based on conditional mutual information
- Information-theoretic Granger causality (captures nonlinear)
- Requires embedding: history lengths k, l and delay τ
- Bias is significant — always use surrogate testing
- Effective TE = observed TE - surrogate mean
- Net TE = TE_{X→Y} - TE_{Y→X} indicates dominant direction
- For hyperscanning: reveals who leads the interaction
- NOT true causality — confounds can create apparent TE
- More data-hungry than MI; parameter selection matters
- Time-resolved TE tracks dynamic leadership

---

## Discussion Questions

1. You find that TE_{P1→P2} = 0.15 bits and TE_{P2→P1} = 0.08 bits during a cooperative task. What does this tell you about the interaction? What additional analyses might help interpret this?

2. TE computed at τ = 50ms is higher than at τ = 100ms. What might this indicate about the neural communication? How would you verify?

3. A critic argues "TE is just fancy correlation — you're not proving causation." How would you respond? What would it take to establish actual causality?

4. You compute TE between two EEG channels and find high TE in both directions (bidirectional). Does this mean the channels are mutually influencing each other? What alternative explanations exist?

5. Your hyperscanning TE analysis shows P1→P2 dominance in alpha band but P2→P1 dominance in theta band. What might this frequency-specific pattern mean for social interaction?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Conceptual diagram | Prediction improvement from X | signal_1, signal_2 |
| 2 | Timeline diagram | Embedding for TE | signal_1, signal_2 |
| 3 | Venn diagram | Conditional MI for TE | three-way overlap |
| 4 | Step-by-step | Vector construction | signal_1, signal_2 |
| 5 | Time series + bars | Unidirectional coupling | signal_1, signal_2 |
| 6 | Line plot | TE vs coupling strength | signal_1 |
| 7 | Bar chart | Bidirectional TE comparison | signal_1, signal_2 |
| 8 | Line plot | Net TE vs asymmetry | signal_1 |
| 9 | Heatmap | TE vs k and l | viridis |
| 10 | Line plot | TE vs τ (delay) | signal_1 |
| 11 | Histogram | TE null distribution | signal_1, highlight |
| 12 | 2×2 comparison | GC vs TE | signal_1, signal_2 |
| 13 | Bar chart | Band-specific TE | band colors |
| 14 | 2 subplots | Time-resolved TE | signal_1, signal_2 |
| 15 | Heatmap | Directed TE matrix | viridis |
| 16 | Heatmap | Net TE matrix | diverging RdBu |
| 17 | 2 heatmaps | Hyperscanning TE | subject_1, subject_2 |
| 18 | Time series | Dynamic leadership | subject_1, subject_2 |
| 19 | Diagram | Confound scenario | warning colors |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Embedding handles edge cases (short signals)
- [ ] High-dimensional histograms computed efficiently
- [ ] Surrogate methods properly shuffle only source
- [ ] Net TE correctly antisymmetric
- [ ] Clear warnings about causality interpretation
- [ ] Hyperscanning functions handle different channel counts
- [ ] Time conversion (ms to samples) handled correctly