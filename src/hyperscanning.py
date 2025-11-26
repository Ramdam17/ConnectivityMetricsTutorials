"""Hyperscanning utilities for dual-brain connectivity analysis.

This module provides functions for organizing and analyzing hyperscanning data,
including data structure creation, connectivity block extraction, and
pseudo-pair analysis for statistical validation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def create_hyperscanning_data_structure(
    data_p1: NDArray[np.float64],
    data_p2: NDArray[np.float64],
    channel_names_p1: List[str],
    channel_names_p2: List[str],
) -> Dict[str, Any]:
    """Create a unified data structure for hyperscanning analysis.

    Combines data from two participants into a single structure with
    prefixed channel names (P1_, P2_) for clear identification.

    Parameters
    ----------
    data_p1 : NDArray[np.float64]
        EEG data from participant 1, shape (n_channels_p1, n_samples).
    data_p2 : NDArray[np.float64]
        EEG data from participant 2, shape (n_channels_p2, n_samples).
    channel_names_p1 : List[str]
        Channel names for participant 1.
    channel_names_p2 : List[str]
        Channel names for participant 2.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "data_combined": Combined data array (n_total_channels, n_samples)
        - "channel_names": List of prefixed channel names
        - "n_channels_p1": Number of channels for P1
        - "n_channels_p2": Number of channels for P2
        - "participant_labels": Array of participant IDs (1 or 2)

    Raises
    ------
    ValueError
        If data dimensions don't match channel names or sample counts differ.

    Examples
    --------
    >>> data_p1 = np.random.randn(4, 1000)
    >>> data_p2 = np.random.randn(4, 1000)
    >>> channels = ["Fz", "Cz", "Pz", "Oz"]
    >>> result = create_hyperscanning_data_structure(
    ...     data_p1, data_p2, channels, channels
    ... )
    >>> result["data_combined"].shape
    (8, 1000)
    """
    # Validate inputs
    if data_p1.shape[0] != len(channel_names_p1):
        raise ValueError(
            f"P1 data has {data_p1.shape[0]} channels but "
            f"{len(channel_names_p1)} channel names provided"
        )
    if data_p2.shape[0] != len(channel_names_p2):
        raise ValueError(
            f"P2 data has {data_p2.shape[0]} channels but "
            f"{len(channel_names_p2)} channel names provided"
        )
    if data_p1.shape[1] != data_p2.shape[1]:
        raise ValueError(
            f"Sample counts differ: P1 has {data_p1.shape[1]}, "
            f"P2 has {data_p2.shape[1]}"
        )

    n_channels_p1 = data_p1.shape[0]
    n_channels_p2 = data_p2.shape[0]

    # Combine data
    data_combined = np.vstack([data_p1, data_p2])

    # Create prefixed channel names
    channel_names = [f"P1_{ch}" for ch in channel_names_p1] + [
        f"P2_{ch}" for ch in channel_names_p2
    ]

    # Create participant labels
    participant_labels = np.array(
        [1] * n_channels_p1 + [2] * n_channels_p2, dtype=np.int8
    )

    return {
        "data_combined": data_combined,
        "channel_names": channel_names,
        "n_channels_p1": n_channels_p1,
        "n_channels_p2": n_channels_p2,
        "participant_labels": participant_labels,
    }


def extract_connectivity_blocks(
    full_matrix: NDArray[np.float64],
    n_channels_p1: int,
) -> Dict[str, NDArray[np.float64]]:
    """Extract connectivity blocks from a combined hyperscanning matrix.

    Separates a full (n_p1 + n_p2) x (n_p1 + n_p2) connectivity matrix
    into within-participant and between-participant blocks.

    Parameters
    ----------
    full_matrix : NDArray[np.float64]
        Full connectivity matrix, shape (n_total, n_total).
    n_channels_p1 : int
        Number of channels for participant 1.

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary containing:
        - "within_p1": Connectivity within P1, shape (n_p1, n_p1)
        - "within_p2": Connectivity within P2, shape (n_p2, n_p2)
        - "between": Connectivity between P1 and P2, shape (n_p1, n_p2)

    Examples
    --------
    >>> matrix = np.random.rand(8, 8)
    >>> blocks = extract_connectivity_blocks(matrix, n_channels_p1=4)
    >>> blocks["within_p1"].shape
    (4, 4)
    >>> blocks["between"].shape
    (4, 4)
    """
    n_total = full_matrix.shape[0]
    n_channels_p2 = n_total - n_channels_p1

    within_p1 = full_matrix[:n_channels_p1, :n_channels_p1]
    within_p2 = full_matrix[n_channels_p1:, n_channels_p1:]
    between = full_matrix[:n_channels_p1, n_channels_p1:]

    return {
        "within_p1": within_p1,
        "within_p2": within_p2,
        "between": between,
    }


def combine_connectivity_blocks(
    within_p1: NDArray[np.float64],
    within_p2: NDArray[np.float64],
    between: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine connectivity blocks into a full hyperscanning matrix.

    Reconstructs a full (n_p1 + n_p2) x (n_p1 + n_p2) matrix from
    the within-participant and between-participant blocks.

    Parameters
    ----------
    within_p1 : NDArray[np.float64]
        Connectivity within P1, shape (n_p1, n_p1).
    within_p2 : NDArray[np.float64]
        Connectivity within P2, shape (n_p2, n_p2).
    between : NDArray[np.float64]
        Connectivity between P1 and P2, shape (n_p1, n_p2).

    Returns
    -------
    NDArray[np.float64]
        Full connectivity matrix, shape (n_p1 + n_p2, n_p1 + n_p2).

    Examples
    --------
    >>> within_p1 = np.eye(4)
    >>> within_p2 = np.eye(4)
    >>> between = np.ones((4, 4)) * 0.5
    >>> full = combine_connectivity_blocks(within_p1, within_p2, between)
    >>> full.shape
    (8, 8)
    """
    n_p1 = within_p1.shape[0]
    n_p2 = within_p2.shape[0]
    n_total = n_p1 + n_p2

    full_matrix = np.zeros((n_total, n_total), dtype=np.float64)

    # Fill blocks
    full_matrix[:n_p1, :n_p1] = within_p1
    full_matrix[n_p1:, n_p1:] = within_p2
    full_matrix[:n_p1, n_p1:] = between
    full_matrix[n_p1:, :n_p1] = between.T  # Symmetric assumption

    return full_matrix


def create_pseudo_pairs(
    participant_ids: List[str],
    real_pairs: List[Tuple[str, str]],
    n_pseudo: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Generate pseudo-pairs from participants who never interacted.

    Creates pairs of participants from different real pairs for use
    as a null distribution in statistical testing.

    Parameters
    ----------
    participant_ids : List[str]
        List of all participant IDs.
    real_pairs : List[Tuple[str, str]]
        List of tuples representing actual interaction pairs.
    n_pseudo : Optional[int]
        Number of pseudo-pairs to generate. If None, generates all possible.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[str, str]]
        List of pseudo-pair tuples.

    Examples
    --------
    >>> ids = ["A1", "A2", "B1", "B2", "C1", "C2"]
    >>> real = [("A1", "A2"), ("B1", "B2"), ("C1", "C2")]
    >>> pseudo = create_pseudo_pairs(ids, real, n_pseudo=5, seed=42)
    >>> len(pseudo)
    5
    """
    if seed is not None:
        np.random.seed(seed)

    # Create set of real pairs (both orderings)
    real_set = set()
    for p1, p2 in real_pairs:
        real_set.add((p1, p2))
        real_set.add((p2, p1))

    # Generate all possible pseudo-pairs
    pseudo_pairs = []
    for i, p1 in enumerate(participant_ids):
        for p2 in participant_ids[i + 1 :]:
            if (p1, p2) not in real_set:
                pseudo_pairs.append((p1, p2))

    # Subsample if requested
    if n_pseudo is not None and n_pseudo < len(pseudo_pairs):
        indices = np.random.choice(len(pseudo_pairs), n_pseudo, replace=False)
        pseudo_pairs = [pseudo_pairs[i] for i in indices]

    return pseudo_pairs


def compute_pseudo_pair_null(
    connectivity_values: Dict[Tuple[str, str], float],
    real_pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Build null distribution from pseudo-pair connectivity values.

    Parameters
    ----------
    connectivity_values : Dict[Tuple[str, str], float]
        Dictionary mapping pair IDs to connectivity values.
    real_pairs : List[Tuple[str, str]]
        List of real interaction pairs.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "null_distribution": Array of pseudo-pair values
        - "null_mean": Mean of null distribution
        - "null_std": Standard deviation of null distribution
        - "real_values": Array of real pair values
        - "real_mean": Mean of real pairs

    Examples
    --------
    >>> values = {("A", "B"): 0.8, ("C", "D"): 0.7, ("A", "C"): 0.3}
    >>> real = [("A", "B"), ("C", "D")]
    >>> result = compute_pseudo_pair_null(values, real)
    >>> result["null_mean"]
    0.3
    """
    real_set = set(real_pairs) | {(p2, p1) for p1, p2 in real_pairs}

    null_values = []
    real_values = []

    for pair, value in connectivity_values.items():
        if pair in real_set:
            real_values.append(value)
        else:
            null_values.append(value)

    null_array = np.array(null_values)
    real_array = np.array(real_values)

    return {
        "null_distribution": null_array,
        "null_mean": float(np.mean(null_array)) if len(null_array) > 0 else np.nan,
        "null_std": float(np.std(null_array)) if len(null_array) > 0 else np.nan,
        "real_values": real_array,
        "real_mean": float(np.mean(real_array)) if len(real_array) > 0 else np.nan,
    }


def test_against_pseudo_pairs(
    real_value: float,
    null_distribution: NDArray[np.float64],
) -> Dict[str, float]:
    """Test a real pair value against the pseudo-pair null distribution.

    Computes p-value, z-score, and percentile for a real pair's
    connectivity value relative to the null distribution.

    Parameters
    ----------
    real_value : float
        Connectivity value from a real pair.
    null_distribution : NDArray[np.float64]
        Array of connectivity values from pseudo-pairs.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - "pvalue": One-tailed p-value (proportion of null >= real)
        - "zscore": Z-score relative to null distribution
        - "percentile": Percentile of real value in null distribution

    Examples
    --------
    >>> null = np.random.normal(0.3, 0.1, 100)
    >>> result = test_against_pseudo_pairs(0.6, null)
    >>> result["pvalue"] < 0.05
    True
    """
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)

    # Z-score
    zscore = (real_value - null_mean) / null_std if null_std > 0 else 0.0

    # One-tailed p-value (testing if real > null)
    pvalue = np.mean(null_distribution >= real_value)

    # Percentile
    percentile = np.mean(null_distribution <= real_value) * 100

    return {
        "pvalue": float(pvalue),
        "zscore": float(zscore),
        "percentile": float(percentile),
    }
