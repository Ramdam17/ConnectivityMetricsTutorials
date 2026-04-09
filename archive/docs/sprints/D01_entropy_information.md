# D01: Entropy and Information — Copilot Directive

## Metadata

- **Notebook ID**: D01
- **Title**: Entropy and Information
- **Branch**: `feature/D01_entropy_information`
- **Duration**: 65 minutes
- **Prerequisites**: A03 (Power Spectrum), basic probability concepts
- **Next**: D02 (Mutual Information)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Understand information as reduction of uncertainty
2. Define and compute Shannon entropy
3. Interpret entropy in bits and nats
4. Compute entropy for discrete and continuous signals
5. Understand binning and its effects on entropy estimation
6. Apply entropy to neural signals
7. Recognize entropy as foundation for mutual information and transfer entropy

---

## Required Imports
```
numpy, matplotlib, scipy.stats, scipy.special, typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py
```

---

## Section 1: Introduction — What is Information?

**Purpose**: Build intuition for information as a mathematical concept.

**Content to cover**:
- Everyday meaning: "information" = facts, data, knowledge
- Mathematical meaning: "information" = reduction of UNCERTAINTY
- Key insight from Claude Shannon (1948):
  - Information can be QUANTIFIED
  - Measured in BITS
  - Foundation of all digital communication
- Why this matters for neuroscience:
  - How much information does a brain signal carry?
  - How much information is SHARED between signals? (→ connectivity!)
  - How much information FLOWS from one signal to another? (→ causality!)
- This notebook: the building blocks

**Key message**: "Information = Surprise. The less expected something is, the more information it carries."

**Length**: 4-5 paragraphs

---

## Section 2: Intuition — Surprise and Uncertainty

**Purpose**: Develop intuitive understanding before math.

**Content to cover**:
- **Thought experiment 1**: Weather forecast
  - "Tomorrow will be sunny" in Sahara desert → low information (expected)
  - "Tomorrow will be sunny" in London → more information (less expected)
  - "Tomorrow will snow" in Sahara → very high information (surprising!)
- **Key principle**: rare events carry more information
- **Thought experiment 2**: Coin flips
  - Fair coin: outcome is maximally uncertain → high information per flip
  - Biased coin (99% heads): outcome is predictable → low information
- **Uncertainty** = how spread out possibilities are
- **Information** = how much uncertainty is reduced by an observation
- Next: how to QUANTIFY this precisely

**Visualization 1**:
- Three probability distributions:
  - Uniform (maximum uncertainty)
  - Slightly peaked (moderate uncertainty)
  - Highly peaked (low uncertainty)
- Bar charts side by side
- Annotate: "Which has most uncertainty?"

---

## Section 3: Shannon Entropy — The Formula

**Purpose**: Introduce the mathematical definition.

**Content to cover**:
- **Shannon entropy** for discrete random variable X:
  $$H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)$$
- Where:
  - p(xᵢ) = probability of outcome xᵢ
  - log = logarithm (base 2 for bits, base e for nats)
  - Sum over all possible outcomes
- **Convention**: 0 × log(0) = 0 (limit as p→0)
- **Units**:
  - Base 2 (log₂): bits
  - Base e (ln): nats
  - 1 nat ≈ 1.44 bits
- **Interpretation**:
  - H(X) = average "surprise" of outcomes
  - H(X) = minimum bits needed to encode outcomes
  - H(X) = uncertainty about X before observing it

**Visualization 2**:
- Plot: -p × log₂(p) vs p
- Shows contribution to entropy from each probability
- Peak at p = 1/e ≈ 0.37
- Caption: "Each probability contributes this much to total entropy"

---

## Section 4: Computing Entropy — Examples

**Purpose**: Hands-on computation for simple cases.

**Content to cover**:
- **Example 1**: Fair coin
  - P(heads) = P(tails) = 0.5
  - H = -0.5 × log₂(0.5) - 0.5 × log₂(0.5) = 1 bit
  - Maximum entropy for 2 outcomes!
- **Example 2**: Biased coin (90% heads)
  - H = -0.9 × log₂(0.9) - 0.1 × log₂(0.1) ≈ 0.47 bits
  - Less than fair coin (more predictable)
- **Example 3**: Fair die (6 sides)
  - H = -6 × (1/6) × log₂(1/6) = log₂(6) ≈ 2.58 bits
- **Example 4**: Loaded die
  - Some sides more likely → lower entropy
- **General rule**: uniform distribution has MAXIMUM entropy

**Code to produce**:

Function 1: `compute_entropy_discrete`
```
Parameters:
    probabilities: NDArray[np.float64]  # must sum to 1
    base: float = 2.0  # 2 for bits, np.e for nats
Returns:
    float
Shannon entropy of discrete distribution
Handle p=0 case correctly
```

Function 2: `compute_entropy_from_counts`
```
Parameters:
    counts: NDArray[np.int64]
    base: float = 2.0
Returns:
    float
Estimate entropy from observed counts
Convert counts to probabilities first
```

**Visualization 3**:
- Bar chart of probabilities for each example
- Annotate entropy value on each
- Visual comparison of "spread" vs entropy

---

## Section 5: Maximum Entropy

**Purpose**: Understand the upper bound on entropy.

**Content to cover**:
- For n possible outcomes, maximum entropy is:
  $$H_{max} = \log(n)$$
- Achieved when distribution is UNIFORM: p(xᵢ) = 1/n for all i
- **Normalized entropy**: H / H_max
  - Range: 0 (deterministic) to 1 (maximum uncertainty)
  - Useful for comparing systems with different numbers of states
- **Minimum entropy**: H = 0
  - When one outcome has probability 1 (deterministic)
  - No uncertainty, no information

**Code to produce**:

Function 3: `compute_max_entropy`
```
Parameters:
    n_states: int
    base: float = 2.0
Returns:
    float
Maximum possible entropy for n states
```

Function 4: `compute_normalized_entropy`
```
Parameters:
    probabilities: NDArray[np.float64]
    base: float = 2.0
Returns:
    float
Entropy normalized by maximum (range 0-1)
```

**Visualization 4**:
- Entropy vs "peakedness" of distribution
- X-axis: parameter controlling distribution shape
- Y-axis: entropy
- Show entropy decreasing as distribution becomes more peaked
- Horizontal line at H_max

---

## Section 6: Binary Entropy Function

**Purpose**: Special case that's commonly used.

**Content to cover**:
- **Binary entropy**: entropy of Bernoulli variable (2 outcomes)
- H(p) = -p × log₂(p) - (1-p) × log₂(1-p)
- Function of single parameter p
- Properties:
  - H(0) = H(1) = 0 (deterministic)
  - H(0.5) = 1 bit (maximum)
  - Symmetric: H(p) = H(1-p)
- Useful for:
  - Analyzing binary classifications
  - Understanding information in yes/no questions
  - Building intuition for entropy

**Code to produce**:

Function 5: `binary_entropy`
```
Parameters:
    p: float | NDArray[np.float64]
Returns:
    float | NDArray[np.float64]
Binary entropy function H(p) in bits
Handle edge cases p=0, p=1
```

**Visualization 5**:
- Plot binary entropy function
- X-axis: p from 0 to 1
- Y-axis: H(p) from 0 to 1
- Mark maximum at p=0.5
- Symmetric parabola-like shape
- Caption: "Binary entropy: uncertainty in yes/no questions"

---

## Section 7: Entropy of Continuous Variables

**Purpose**: Extend entropy to continuous signals (like EEG).

**Content to cover**:
- **Problem**: continuous variables have infinite possible values
- Can't directly sum over infinite outcomes
- **Solution 1**: Discretize (binning) → discrete entropy
- **Solution 2**: Differential entropy (continuous analogue)
- **Differential entropy**:
  $$h(X) = -\int p(x) \log p(x) dx$$
- **Important differences from discrete**:
  - Can be NEGATIVE (unlike discrete entropy)
  - Depends on units of measurement
  - Not directly comparable across different variable types
- For neural signals: usually use BINNING approach

**Visualization 6**:
- Continuous distribution (e.g., Gaussian)
- Show discretization into bins
- Histogram approximating the distribution
- Caption: "Binning converts continuous to discrete"

---

## Section 8: Binning Strategies

**Purpose**: Practical methods for estimating entropy from data.

**Content to cover**:
- **Step 1**: Choose number of bins (critical choice!)
- **Step 2**: Define bin edges (uniform width or adaptive)
- **Step 3**: Count samples in each bin
- **Step 4**: Convert counts to probabilities
- **Step 5**: Compute discrete entropy
- **Binning methods**:
  1. **Uniform width**: equal-sized bins across data range
  2. **Uniform count (equiprobable)**: bins with equal number of samples
  3. **Adaptive**: data-driven bin edges
- **Choosing number of bins**:
  - Too few: lose information, underestimate entropy
  - Too many: sparse bins, biased estimation
  - Rules of thumb: √n, Sturges' rule, Freedman-Diaconis

**Code to produce**:

Function 6: `compute_entropy_continuous`
```
Parameters:
    signal: NDArray[np.float64]
    n_bins: int | str = "auto"  # or integer, or "sturges", "fd"
    method: str = "uniform"  # or "equiprobable"
Returns:
    Tuple[float, int]
    (entropy, actual_n_bins)
Estimate entropy of continuous signal via binning
Return entropy and number of bins used
```

Function 7: `optimal_n_bins`
```
Parameters:
    n_samples: int
    method: str = "sturges"  # or "sqrt", "fd"
Returns:
    int
Compute optimal number of bins for entropy estimation
```

**Visualization 7**:
- Same signal, different number of bins: 5, 20, 100, 500
- Four histograms
- Entropy value annotated on each
- Show how entropy changes with binning
- Caption: "Binning choice affects entropy estimate"

---

## Section 9: Bias in Entropy Estimation

**Purpose**: Understand and correct estimation bias.

**Content to cover**:
- **Problem**: entropy estimated from data is BIASED
- Finite samples → can't observe all possible outcomes
- Estimated entropy is typically TOO LOW
- **Miller-Madow correction**:
  $$\hat{H}_{corrected} = \hat{H} + \frac{m - 1}{2n}$$
  - m = number of bins with non-zero counts
  - n = total number of samples
- **Other approaches**:
  - Bayesian estimators (NSB)
  - Jackknife correction
  - Plug-in with shrinkage
- For practical use: be aware of bias, use enough data, report method

**Code to produce**:

Function 8: `compute_entropy_miller_madow`
```
Parameters:
    signal: NDArray[np.float64]
    n_bins: int
Returns:
    float
Entropy with Miller-Madow bias correction
```

**Visualization 8**:
- Simulation: known entropy, estimate from samples
- X-axis: sample size (100, 500, 1000, 5000, 10000)
- Y-axis: estimated entropy
- Show: raw estimate (biased), corrected estimate, true value
- Demonstrate convergence to true value

---

## Section 10: Entropy of Neural Signals

**Purpose**: Apply entropy to EEG-like data.

**Content to cover**:
- What does entropy of EEG tell us?
- **High entropy**: signal is unpredictable, complex
  - Could indicate: active processing, noise, disorder
- **Low entropy**: signal is predictable, regular
  - Could indicate: dominant rhythm, simple dynamics, coma
- **Applications**:
  - Anesthesia depth monitoring (lower entropy = deeper)
  - Sleep stage classification
  - Epilepsy detection (seizures can have low entropy)
  - Consciousness assessment
- **Caution**: interpretation depends on preprocessing!

**Code to produce**:
- Generate synthetic EEG with different characteristics:
  - Regular alpha rhythm (low entropy)
  - Mixed rhythms (medium entropy)
  - Noisy/random (high entropy)

**Visualization 9**:
- Three synthetic signals with different entropy
- Three rows:
  - Signal time series
  - Histogram of values
  - Entropy value annotated
- Clear visual of entropy reflecting signal complexity

---

## Section 11: Spectral Entropy

**Purpose**: Entropy applied to frequency domain.

**Content to cover**:
- **Spectral entropy**: entropy of normalized power spectrum
- Treats PSD as a probability distribution
- Normalize: p(f) = PSD(f) / Σ PSD(f)
- Then compute: H = -Σ p(f) × log p(f)
- **Interpretation**:
  - High spectral entropy: power spread across frequencies (white noise)
  - Low spectral entropy: power concentrated in few frequencies (pure tone)
- Measures "complexity" of frequency content
- Common in EEG analysis, especially for anesthesia monitoring

**Code to produce**:

Function 9: `compute_spectral_entropy`
```
Parameters:
    signal: NDArray[np.float64]
    fs: float
    nperseg: int = 256
    normalize: bool = True
Returns:
    float
Spectral entropy from power spectrum
If normalize=True, return H / H_max (range 0-1)
```

**Visualization 10**:
- Three signals:
  - Pure sine wave (low spectral entropy)
  - EEG-like signal (medium spectral entropy)
  - White noise (high spectral entropy)
- For each: time series + PSD + spectral entropy value
- 3×2 grid

---

## Section 12: Sample Entropy (Preview)

**Purpose**: Introduce regularity-based entropy for time series.

**Content to cover**:
- Shannon entropy ignores temporal ORDER of samples
- **Sample entropy**: measures regularity/predictability of patterns
- Based on probability that similar patterns remain similar
- Parameters:
  - m: pattern length
  - r: similarity threshold
- Lower sample entropy → more regular/predictable
- Higher sample entropy → more complex/irregular
- Common in HRV analysis, EEG complexity
- Brief introduction here; can be expanded in future notebooks

**Code to produce**:

Function 10: `compute_sample_entropy`
```
Parameters:
    signal: NDArray[np.float64]
    m: int = 2  # embedding dimension
    r: float | None = None  # threshold (default: 0.2 * std)
Returns:
    float
Sample entropy measuring pattern regularity
```

**Visualization 11**:
- Two signals: regular and irregular
- Same Shannon entropy (similar amplitude distribution)
- Different sample entropy (different temporal structure)
- Side-by-side comparison
- Caption: "Sample entropy captures temporal structure"

---

## Section 13: Entropy in Hyperscanning Context

**Purpose**: Connect to workshop theme.

**Content to cover**:
- Individual entropy: how complex is each participant's signal?
- **Joint entropy**: uncertainty about BOTH signals together (next notebook!)
- **Mutual information**: shared information between signals (D02)
- **Transfer entropy**: directed information flow (D03)
- Entropy is the FOUNDATION for all these
- In hyperscanning:
  - Do participants' signal complexities correlate?
  - Do they share information? (MI)
  - Does information flow from one to the other? (TE)
- This notebook provides the building blocks

**Visualization 12**:
- Conceptual diagram showing:
  - H(X): entropy of participant 1
  - H(Y): entropy of participant 2
  - Preview: overlapping circles for MI (Venn diagram)
- Caption: "Entropy is foundation for information-theoretic connectivity"

---

## Section 14: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Coin Flip Entropy
- Compute entropy for coins with bias: 0.1, 0.3, 0.5, 0.7, 0.9
- Plot entropy vs bias
- Verify maximum at 0.5
- What's the entropy of a coin with 0.01 bias?

**Exercise 2**: Binning Effects
- Generate Gaussian signal (1000 samples)
- Compute entropy with bins: 5, 10, 20, 50, 100, 200
- Plot entropy vs number of bins
- What happens with too few/many bins?
- Apply Miller-Madow correction — does it help?

**Exercise 3**: Signal Complexity
- Generate three signals:
  - Pure 10 Hz sine wave
  - Sine wave + harmonics (10, 20, 30 Hz)
  - Pink noise
- Compute Shannon entropy and spectral entropy for each
- Which measure better captures "complexity"?

**Exercise 4**: Sample Entropy Comparison
- Generate two signals:
  - Regular: repeating pattern
  - Irregular: random walk
- Make them have similar amplitude distributions
- Compute Shannon entropy (should be similar)
- Compute sample entropy (should differ)
- Interpret the difference

**Exercise 5**: EEG-like Analysis
- Generate synthetic EEG with:
  - Baseline (mixed rhythms)
  - Event (strong alpha burst)
- Compute entropy in sliding windows
- Plot entropy over time
- Does entropy change during the event?

**Exercise 6**: Entropy Estimation Accuracy
- Known distribution: uniform on [0, 1]
- True entropy: log₂(n_bins) for n_bins uniform bins
- Generate samples: 100, 500, 1000, 5000
- Estimate entropy each time
- Plot: estimated vs true, with error bars
- How many samples needed for accurate estimation?

---

## Section 15: Functions for src/information.py

**Purpose**: Identify functions for new information theory module.

**New file**: src/information.py

Functions to extract:
- `compute_entropy_discrete`
- `compute_entropy_from_counts`
- `compute_max_entropy`
- `compute_normalized_entropy`
- `binary_entropy`
- `compute_entropy_continuous`
- `optimal_n_bins`
- `compute_entropy_miller_madow`
- `compute_spectral_entropy`
- `compute_sample_entropy`

---

## Summary

**Key takeaways to list**:
- Information = reduction of uncertainty (Shannon, 1948)
- Entropy H(X) = average surprise = uncertainty about X
- Computed as: H = -Σ p(x) log p(x)
- Units: bits (log₂) or nats (ln)
- Maximum entropy: uniform distribution
- Minimum entropy: deterministic (one outcome certain)
- Continuous signals: use binning to estimate entropy
- Binning choice critically affects estimates
- Bias correction (Miller-Madow) helps with finite samples
- Spectral entropy: complexity of frequency content
- Sample entropy: temporal regularity/predictability
- Entropy is foundation for MI, TE, and information-theoretic connectivity

---

## Discussion Questions

1. A patient's EEG shows lower entropy during anesthesia compared to wakefulness. What does this tell us about brain activity? Could low entropy ever indicate increased processing?

2. You analyze the same EEG signal with 10 bins and 100 bins and get very different entropy values. Which is "correct"? How would you decide on the appropriate number of bins?

3. Two signals have the same Shannon entropy but different sample entropy. What does this tell you about their structure? Can you think of real-world examples?

4. In hyperscanning, participant A has higher signal entropy than participant B. Does this mean A's brain is "doing more"? What alternative interpretations exist?

5. Why is it important that entropy is measured in bits? What does it mean that a fair coin flip provides exactly 1 bit of information?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 3 bar charts | Distributions with different uncertainty | signal_1, signal_2, signal_3 |
| 2 | Single plot | -p log(p) contribution function | signal_1 |
| 3 | 4 bar charts | Example distributions with entropy | various |
| 4 | Line plot | Entropy vs distribution peakedness | signal_1 |
| 5 | Single plot | Binary entropy function | signal_1 |
| 6 | Distribution + histogram | Continuous to discrete | signal_1, signal_2 |
| 7 | 4 histograms | Different binning of same signal | signal_1 |
| 8 | Line plot | Bias vs sample size | signal_1, signal_2, reference |
| 9 | 3×2 grid | Signals with different entropy | signal colors |
| 10 | 3×2 grid | Spectral entropy examples | signal colors |
| 11 | 2×2 comparison | Shannon vs sample entropy | signal_1, signal_2 |
| 12 | Conceptual Venn | Entropy foundation diagram | subject_1, subject_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Handle p=0 edge case (0 log 0 = 0)
- [ ] Support both bits (log2) and nats (ln)
- [ ] Binning functions handle edge cases
- [ ] Sample entropy implementation is efficient
- [ ] Clear documentation of bias and limitations