# G02: Phase Lag Index (PLI) — Copilot Directive

## Metadata

- **Notebook ID**: G02
- **Title**: Phase Lag Index (PLI)
- **Branch**: `feature/G02_phase_lag_index`
- **Duration**: 55 minutes
- **Prerequisites**: G01 (Phase Locking Value), C01 (Volume Conduction Problem)
- **Next**: G03 (Weighted Phase Lag Index)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Explain why PLV fails for volume conduction
2. Define PLI as asymmetry of phase difference distribution
3. Implement PLI computation
4. Interpret PLI values (0 to 1)
5. Understand the sign inconsistency issue with PLI
6. Apply PLI to hyperscanning analysis
7. Recognize when to use PLI vs PLV

---

## Required Imports
```
numpy, matplotlib, scipy.signal
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/signals.py, src/phase.py, src/plv.py
```

---

## Section 1: The Problem with PLV (Revisited)

**Purpose**: Motivate PLI strongly.

**Content to cover**:
- **Recall G01**: PLV measures phase consistency
- **Recall C01**: Volume conduction creates zero-lag signals
- **The issue**:
  - Volume conduction → phase difference ≈ 0 (or π)
  - Phase difference very consistent (always near 0)
  - PLV = very high!
  - But it's SPURIOUS — not true connectivity
- **What we need**: A metric that ignores zero-phase relationships
- **PLI solution**: Only count non-zero phase differences

**Key message**: "PLI ignores phase differences at 0 and π, eliminating volume conduction artifacts."

**Visualization 1**:
- Phase difference distributions:
  - Left: Volume conduction — peaked at 0
  - Middle: True connection — peaked elsewhere (e.g., 45°)
  - Right: No connection — uniform
- PLV values annotated (both left and middle have high PLV!)
- Caption: "PLV can't distinguish volume conduction from true connectivity"

**Length**: 3-4 paragraphs

---

## Section 2: The Key Insight — Phase Difference Sign

**Purpose**: Build intuition for PLI.

**Content to cover**:
- **Volume conduction signature**: Δφ ≈ 0
- **True connectivity**: Δφ consistently non-zero (some delay)
- **New idea**: Look at the SIGN of sin(Δφ)
  - sin(Δφ) > 0 when 0 < Δφ < π (Y leads X)
  - sin(Δφ) < 0 when -π < Δφ < 0 (X leads Y)
  - sin(Δφ) = 0 when Δφ = 0 or π (volume conduction zone!)
- **PLI principle**:
  - If true connection with consistent lag: signs mostly same
  - If volume conduction (Δφ ≈ 0): equal positive and negative (noise fluctuates both ways)
- **Measure**: Asymmetry of sign distribution

**Visualization 2**:
- Sign of sin(Δφ) diagram:
  - Unit circle with phase difference
  - Upper half: sin > 0 (Y leads)
  - Lower half: sin < 0 (X leads)
  - On axis: sin = 0 (volume conduction zone)
- Caption: "Phase difference sign: positive, negative, or zero"

**Visualization 3**:
- Two scenarios:
  - Volume conduction: Δφ fluctuates around 0, signs balance
  - True connection: Δφ consistently positive, signs imbalanced
- Caption: "Sign asymmetry distinguishes true connectivity"

---

## Section 3: The PLI Formula

**Purpose**: Formal definition.

**Content to cover**:
- **PLI definition** (Stam et al., 2007):
  $$PLI = \left| \frac{1}{N} \sum_{t=1}^{N} sign(\sin(\Delta\phi(t))) \right|$$
  Or equivalently:
  $$PLI = \left| \langle sign(\sin(\Delta\phi)) \rangle \right|$$
- **Interpretation**:
  - Average the signs (+1, -1, or 0)
  - Take absolute value
  - If signs balance (volume conduction): average ≈ 0 → PLI ≈ 0
  - If signs imbalanced (true connection): average ≠ 0 → PLI > 0
- **Properties**:
  - Range: 0 to 1
  - PLI = 0: symmetric phase distribution (could be volume conduction OR no connection)
  - PLI = 1: perfect asymmetry (all phases same side)
  - PLI ignores MAGNITUDE of phase difference, only sign matters

**Visualization 4**:
- Formula visualization:
  - Phase difference time series
  - Sign extraction (color by +1 / -1)
  - Average calculation
  - PLI as absolute value
- Caption: "PLI computation steps"

---

## Section 4: Implementing PLI

**Purpose**: Practical computation.

**Content to cover**:
- **Pipeline**:
  1. Band-pass filter both signals
  2. Extract instantaneous phases (Hilbert)
  3. Compute phase difference Δφ(t)
  4. Compute sign(sin(Δφ(t)))
  5. Average the signs
  6. Take absolute value
- **Note**: Very similar to PLV, but sign instead of unit vector

**Code to produce**:

Function 1: `compute_pli`
```
Parameters:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
    filter_order: int = 4
Returns:
    float
Phase Lag Index between x and y
```

Function 2: `compute_pli_from_phases`
```
Parameters:
    phase_x: NDArray[np.float64]
    phase_y: NDArray[np.float64]
Returns:
    float
PLI from pre-computed phases
```

Function 3: `compute_sign_series`
```
Parameters:
    phase_diff: NDArray[np.float64]
Returns:
    NDArray[np.int64]
Sign of sin(phase_diff): +1, -1, or 0
```

**Visualization 5**:
- Comparison with PLV pipeline:
  - PLV path: difference → unit vector → average → magnitude
  - PLI path: difference → sin → sign → average → absolute
  - Highlight the branching point
- Caption: "PLI vs PLV computation"

---

## Section 5: PLI vs Volume Conduction

**Purpose**: Demonstrate the key advantage.

**Content to cover**:
- **Test 1**: Volume conduction simulation
  - Same signal at two electrodes
  - PLV = high
  - PLI ≈ 0 (fluctuations around 0 balance out)
- **Test 2**: True lagged connectivity
  - Consistent phase lag
  - PLV = high
  - PLI = high (signs imbalanced)
- **Conclusion**: PLI distinguishes what PLV cannot!

**Code to produce**:

Function 4: `demonstrate_pli_volume_conduction`
```
Parameters:
    n_samples: int = 10000
    fs: float = 500.0
    frequency: float = 10.0
    seed: int | None = None
Returns:
    dict[str, Any]
    {
        "vc_plv": float, "vc_pli": float,  # volume conduction
        "tc_plv": float, "tc_pli": float,  # true connection
        "signals_vc": Tuple, "signals_tc": Tuple
    }
Compare PLV and PLI for volume conduction vs true connectivity
```

**Visualization 6**:
- Two-row comparison:
  - Row 1: Volume conduction — phase diff histogram (peaked at 0), PLV high, PLI low
  - Row 2: True connection — phase diff histogram (peaked elsewhere), PLV high, PLI high
- Caption: "PLI correctly identifies volume conduction"

**Visualization 7**:
- Bar chart summary:
  - Volume conduction: PLV=0.95, PLI=0.05
  - True connection: PLV=0.85, PLI=0.75
  - Clear visual difference
- Caption: "PLI distinguishes what PLV cannot"

---

## Section 6: The Sensitivity vs Specificity Trade-off

**Purpose**: Honest assessment of PLI limitations.

**Content to cover**:
- **Advantage** (specificity): PLI doesn't false-alarm on volume conduction
- **Disadvantage** (sensitivity): PLI may miss true zero-lag connections
  - If true connectivity happens at exactly zero lag, PLI = 0
  - Such connections ARE possible (e.g., common input)
- **Noise sensitivity**:
  - Sign function is discontinuous
  - Small noise fluctuations around 0 cause sign flips
  - Can make PLI unstable for small phase differences
- **The trade-off**:
  - PLV: high sensitivity, low specificity (finds all, including false positives)
  - PLI: high specificity, lower sensitivity (confident results, may miss some)
- **Solution**: wPLI (G03) improves on PLI

**Visualization 8**:
- Sensitivity vs specificity diagram:
  - PLV: high sensitivity, low specificity region
  - PLI: lower sensitivity, high specificity region
  - "No free lunch"
- Caption: "PLV vs PLI: the sensitivity-specificity trade-off"

---

## Section 7: PLI Matrix

**Purpose**: Multi-channel analysis.

**Content to cover**:
- Compute PLI for all channel pairs → PLI matrix
- **Properties**:
  - Symmetric: PLI(X,Y) = PLI(Y,X) (absolute value)
  - Diagonal = 0 (no self-lag)
  - Values 0 to 1
- **Comparison with PLV matrix**:
  - PLI typically lower overall
  - Volume conduction pairs have much lower PLI than PLV

**Code to produce**:

Function 5: `compute_pli_matrix`
```
Parameters:
    data: NDArray[np.float64]  # (n_channels, n_samples)
    fs: float
    band: tuple[float, float]
Returns:
    NDArray[np.float64]  # (n_channels, n_channels)
PLI matrix for all channel pairs
```

Function 6: `compute_pli_matrix_bands`
```
Parameters:
    data: NDArray[np.float64]
    fs: float
    bands: dict[str, tuple[float, float]] | None = None
Returns:
    dict[str, NDArray[np.float64]]
PLI matrices for all frequency bands
```

**Visualization 9**:
- Side-by-side matrices:
  - PLV matrix
  - PLI matrix
  - Same data
  - Note which connections are "cleaned up" by PLI
- Caption: "PLV vs PLI matrices: volume conduction effects visible"

---

## Section 8: PLI for Hyperscanning

**Purpose**: Apply to inter-brain connectivity.

**Content to cover**:
- **Inter-brain PLI**:
  - No volume conduction between brains anyway!
  - PLI and PLV should be similar for between-brain
  - PLI still useful for within-brain blocks
- **Recommendation**:
  - Use PLI for consistency (same metric everywhere)
  - Or use PLI within-brain, PLV between-brain (justified by different concerns)
- **Research**: PLI increasingly used in hyperscanning for robustness

**Code to produce**:

Function 7: `compute_pli_hyperscanning`
```
Parameters:
    data_p1: NDArray[np.float64]
    data_p2: NDArray[np.float64]
    fs: float
    band: tuple[float, float]
Returns:
    dict[str, NDArray[np.float64]]
    {
        "within_p1": NDArray,
        "within_p2": NDArray,
        "between": NDArray,
        "full": NDArray
    }
PLI matrices for hyperscanning
```

**Visualization 10**:
- Hyperscanning comparison:
  - PLV full matrix
  - PLI full matrix
  - Highlight: within-brain differs more than between-brain
- Caption: "PLI cleans up within-brain connectivity"

---

## Section 9: Signed PLI (Optional)

**Purpose**: Advanced — directional information.

**Content to cover**:
- Standard PLI uses absolute value → loses direction
- **Signed PLI**: Don't take absolute value
  - Range: -1 to +1
  - Positive: Y leads X on average
  - Negative: X leads Y on average
- **Use case**: Want to know direction of phase relationship
- **Caution**: Sign depends on channel order!

**Code to produce**:

Function 8: `compute_signed_pli`
```
Parameters:
    phase_x: NDArray[np.float64]
    phase_y: NDArray[np.float64]
Returns:
    float
Signed PLI (without absolute value)
Positive = Y leads, Negative = X leads
```

**Visualization 11**:
- Signed PLI example:
  - Positive PLI: Y leads X
  - Negative PLI: X leads Y
  - Show phase relationship for each
- Caption: "Signed PLI indicates lead/lag direction"

---

## Section 10: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: PLI Basics
- Generate two signals with consistent 45° phase lag
- Compute PLV and PLI
- Both should be high (true connectivity)

**Exercise 2**: Volume Conduction Test
- Simulate volume conduction (same signal + small noise at two electrodes)
- Compute PLV → high
- Compute PLI → low!
- This is the key demonstration

**Exercise 3**: Sign Distribution
- Generate phase-locked signals
- Compute phase differences
- Histogram the signs (+1, -1)
- For true connectivity: imbalanced
- For volume conduction: balanced

**Exercise 4**: Zero-Lag True Connection
- Create two signals with genuine zero-lag relationship (common input)
- Compute PLI → low (even though connection is "real")
- This demonstrates PLI's limitation

**Exercise 5**: PLI Matrix
- Create 8-channel data with:
  - Some truly connected pairs (lagged)
  - Some volume-conduction pairs (zero-lag)
- Compute PLV and PLI matrices
- Which pairs differ most?
- PLI "cleans up" the volume conduction

**Exercise 6**: Hyperscanning Application
- Create two-participant data
- Add within-brain volume conduction artifacts
- Add between-brain true connectivity
- Compute PLV and PLI hyperscanning matrices
- Verify: within-brain PLI << PLV
- Verify: between-brain PLI ≈ PLV

---

## Section 11: Add Functions to src/plv.py

**Purpose**: Extend PLV module to include PLI.

**Add to src/plv.py**:
- `compute_pli`
- `compute_pli_from_phases`
- `compute_sign_series`
- `demonstrate_pli_volume_conduction`
- `compute_pli_matrix`
- `compute_pli_matrix_bands`
- `compute_pli_hyperscanning`
- `compute_signed_pli`

---

## Summary

**Key takeaways to list**:
- **PLV problem**: High for volume conduction (zero-lag)
- **PLI solution**: Measures asymmetry of phase difference SIGN
- Formula: PLI = |mean(sign(sin(Δφ)))|
- Range: 0 (symmetric/volume conduction) to 1 (fully asymmetric)
- **How it works**: Volume conduction → Δφ ≈ 0 → signs balance → PLI ≈ 0
- True connectivity → consistent lag → signs imbalanced → PLI > 0
- **Trade-off**: Higher specificity but lower sensitivity than PLV
- May miss true zero-lag connections
- Sign function makes PLI sensitive to noise
- **For hyperscanning**: PLI safer for within-brain; between-brain less critical
- **Signed PLI**: preserves lead/lag direction
- **Next**: wPLI (G03) improves noise robustness

---

## Discussion Questions

1. You find PLV = 0.7 but PLI = 0.1 between two nearby electrodes. What's your interpretation? What would you conclude?

2. PLI = 0 could mean volume conduction OR true zero-lag coupling. How problematic is this ambiguity? Can you think of scenarios where it matters?

3. The sign function in PLI is discontinuous — small changes in phase difference can flip the sign. How might this affect PLI stability? What's the solution (hint: G03)?

4. For between-brain hyperscanning, is PLI necessary since there's no volume conduction? Make arguments for both using PLI and using PLV in this case.

5. You want to compare connectivity between patients and controls. Would you use PLV or PLI? What factors influence your choice?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | 3 histograms | Phase diff distributions | signal_1 |
| 2 | Unit circle | Sign regions | signal_1, signal_2 |
| 3 | Comparison | Sign balance vs imbalance | signal_1, signal_2 |
| 4 | Pipeline | PLI computation | process colors |
| 5 | Comparison | PLI vs PLV pipeline | signal_1, signal_2 |
| 6 | 2-row | VC vs TC with PLV/PLI | signal_1, signal_2 |
| 7 | Bar chart | PLV/PLI comparison | signal_1, signal_2 |
| 8 | Diagram | Sensitivity-specificity | signal_1, signal_2 |
| 9 | 2 matrices | PLV vs PLI | viridis |
| 10 | 2 matrices | Hyperscanning comparison | subject_1, subject_2 |
| 11 | Waveforms | Signed PLI direction | signal_1, signal_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Sign function handles exact zeros correctly
- [ ] Phase wrapping handled properly
- [ ] PLI clipped to [0, 1] if numerical issues
- [ ] Signed PLI clearly distinguished from standard PLI
- [ ] Volume conduction simulation realistic
- [ ] Comparison with PLV clearly presented