"""
Information theory functions for entropy and related measures.

This module provides functions for computing various entropy measures
used in signal analysis and connectivity metrics.

Functions
---------
compute_entropy_discrete
    Compute Shannon entropy of a discrete probability distribution.
compute_entropy_from_counts
    Estimate entropy from observed counts.
compute_max_entropy
    Compute maximum possible entropy for n states.
compute_normalized_entropy
    Compute entropy normalized by maximum (range 0-1).
binary_entropy
    Compute binary entropy function H(p).
optimal_n_bins
    Compute optimal number of bins for entropy estimation.
compute_entropy_continuous
    Estimate entropy of a continuous signal via binning.
compute_entropy_miller_madow
    Compute entropy with Miller-Madow bias correction.
compute_spectral_entropy
    Compute spectral entropy of a signal.
compute_sample_entropy
    Compute sample entropy of a time series.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union
from scipy.signal import welch


def compute_entropy_discrete(
    probabilities: NDArray[np.float64],
    base: float = 2.0
) -> float:
    """
    Compute Shannon entropy of a discrete probability distribution.
    
    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability distribution (must sum to 1).
    base : float, optional
        Logarithm base. Use 2 for bits, np.e for nats. Default is 2.
    
    Returns
    -------
    float
        Shannon entropy of the distribution.
    
    Examples
    --------
    >>> compute_entropy_discrete(np.array([0.5, 0.5]))
    1.0  # Fair coin = 1 bit
    >>> compute_entropy_discrete(np.array([0.25, 0.25, 0.25, 0.25]))
    2.0  # Uniform over 4 states = 2 bits
    """
    # Ensure probabilities sum to 1 (with tolerance)
    assert np.abs(np.sum(probabilities) - 1.0) < 1e-9, "Probabilities must sum to 1"
    
    # Filter out zeros to avoid log(0)
    p = probabilities[probabilities > 0]
    
    # Compute entropy
    if base == np.e:
        entropy = -np.sum(p * np.log(p))
    else:
        entropy = -np.sum(p * np.log(p) / np.log(base))
    
    return float(entropy)


def compute_entropy_from_counts(
    counts: NDArray[np.int64],
    base: float = 2.0
) -> float:
    """
    Estimate entropy from observed counts.
    
    Parameters
    ----------
    counts : NDArray[np.int64]
        Count of observations for each outcome.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Estimated Shannon entropy.
    
    Examples
    --------
    >>> compute_entropy_from_counts(np.array([50, 50]))
    1.0  # Equal counts = 1 bit
    """
    # Convert counts to probabilities
    total = np.sum(counts)
    if total == 0:
        return 0.0
    
    probabilities = counts / total
    return compute_entropy_discrete(probabilities, base)


def compute_max_entropy(n_states: int, base: float = 2.0) -> float:
    """
    Compute maximum possible entropy for n states.
    
    The maximum entropy is achieved when all states are equally likely
    (uniform distribution).
    
    Parameters
    ----------
    n_states : int
        Number of possible states/outcomes.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Maximum entropy = log(n_states).
    
    Examples
    --------
    >>> compute_max_entropy(2)
    1.0  # log2(2) = 1 bit
    >>> compute_max_entropy(8)
    3.0  # log2(8) = 3 bits
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    
    if base == np.e:
        return float(np.log(n_states))
    else:
        return float(np.log(n_states) / np.log(base))


def compute_normalized_entropy(
    probabilities: NDArray[np.float64],
    base: float = 2.0
) -> float:
    """
    Compute entropy normalized by maximum (range 0-1).
    
    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability distribution.
    base : float, optional
        Logarithm base. Default is 2.
    
    Returns
    -------
    float
        Normalized entropy in range [0, 1].
        0 = deterministic, 1 = maximum uncertainty.
    
    Examples
    --------
    >>> compute_normalized_entropy(np.array([0.5, 0.5]))
    1.0  # Maximum entropy for 2 states
    >>> compute_normalized_entropy(np.array([1.0, 0.0]))
    0.0  # Deterministic
    """
    n_states = len(probabilities)
    if n_states <= 1:
        return 0.0
    
    H = compute_entropy_discrete(probabilities, base)
    H_max = compute_max_entropy(n_states, base)
    
    return H / H_max if H_max > 0 else 0.0


def binary_entropy(p: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Compute binary entropy function H(p) in bits.
    
    The binary entropy function gives the entropy of a Bernoulli
    random variable with probability p.
    
    Parameters
    ----------
    p : float or NDArray
        Probability value(s) in range [0, 1].
    
    Returns
    -------
    float or NDArray
        Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Examples
    --------
    >>> binary_entropy(0.5)
    1.0  # Maximum at p=0.5
    >>> binary_entropy(0.0)
    0.0  # Deterministic
    """
    p = np.asarray(p)
    
    # Handle edge cases
    result = np.zeros_like(p, dtype=float)
    
    # Only compute for valid probabilities (not 0 or 1)
    valid = (p > 0) & (p < 1)
    p_valid = p[valid]
    result[valid] = -p_valid * np.log2(p_valid) - (1 - p_valid) * np.log2(1 - p_valid)
    
    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def optimal_n_bins(n_samples: int, method: str = "sturges") -> int:
    """
    Compute optimal number of bins for entropy estimation.
    
    Parameters
    ----------
    n_samples : int
        Number of data samples.
    method : str, optional
        Method for determining bins:
        - "sturges": 1 + log2(n) (default, good for normal distributions)
        - "sqrt": sqrt(n) (simple rule)
        - "rice": 2 * n^(1/3) (better for larger datasets)
    
    Returns
    -------
    int
        Recommended number of bins.
    
    Examples
    --------
    >>> optimal_n_bins(1000, "sturges")
    11
    >>> optimal_n_bins(1000, "sqrt")
    32
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    if method == "sturges":
        return max(1, int(np.ceil(1 + np.log2(n_samples))))
    elif method == "sqrt":
        return max(1, int(np.ceil(np.sqrt(n_samples))))
    elif method == "rice":
        return max(1, int(np.ceil(2 * n_samples ** (1/3))))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sturges', 'sqrt', or 'rice'.")


def compute_entropy_continuous(
    signal: NDArray[np.float64],
    n_bins: Union[int, str] = "auto",
    method: str = "uniform"
) -> Tuple[float, int]:
    """
    Estimate entropy of a continuous signal via binning.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Continuous signal to analyze.
    n_bins : int or str, optional
        Number of bins, or "auto"/"sturges"/"sqrt" for automatic.
        Default is "auto" (uses Sturges' rule).
    method : str, optional
        Binning method:
        - "uniform": Equal width bins (default)
        - "equiprobable": Equal count bins
    
    Returns
    -------
    Tuple[float, int]
        (entropy in bits, actual number of bins used)
    
    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> H, n_bins = compute_entropy_continuous(signal)
    """
    n_samples = len(signal)
    
    # Determine number of bins
    if isinstance(n_bins, str):
        if n_bins == "auto" or n_bins == "sturges":
            actual_bins = optimal_n_bins(n_samples, "sturges")
        elif n_bins == "sqrt":
            actual_bins = optimal_n_bins(n_samples, "sqrt")
        else:
            actual_bins = optimal_n_bins(n_samples, "sturges")
    else:
        actual_bins = int(n_bins)
    
    # Compute histogram
    if method == "uniform":
        counts, _ = np.histogram(signal, bins=actual_bins)
    elif method == "equiprobable":
        # Create bins with equal number of samples
        percentiles = np.linspace(0, 100, actual_bins + 1)
        bin_edges = np.percentile(signal, percentiles)
        counts, _ = np.histogram(signal, bins=bin_edges)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute entropy from counts
    entropy = compute_entropy_from_counts(counts)
    
    return entropy, actual_bins


def compute_entropy_miller_madow(
    signal: NDArray[np.float64],
    n_bins: int,
    base: float = 2.0
) -> float:
    """
    Compute entropy with Miller-Madow bias correction.
    
    The Miller-Madow correction adds (m-1)/(2n) to the raw entropy
    estimate, where m is the number of non-empty bins and n is the
    sample size.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Continuous signal to analyze.
    n_bins : int
        Number of bins for discretization.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Bias-corrected entropy estimate.
    
    Examples
    --------
    >>> signal = np.random.uniform(0, 1, 100)
    >>> H_corrected = compute_entropy_miller_madow(signal, n_bins=20)
    """
    n_samples = len(signal)
    
    # Compute histogram
    counts, _ = np.histogram(signal, bins=n_bins)
    
    # Number of non-empty bins
    m = np.sum(counts > 0)
    
    # Raw entropy
    H_raw = compute_entropy_from_counts(counts, base)
    
    # Miller-Madow correction
    if base == np.e:
        correction = (m - 1) / (2 * n_samples)
    else:
        correction = (m - 1) / (2 * n_samples * np.log(base))
    
    return H_raw + correction


def compute_spectral_entropy(
    signal: NDArray[np.float64],
    fs: float,
    nperseg: int = 256,
    freq_range: Optional[Tuple[float, float]] = None,
    normalize: bool = True
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spectral entropy of a signal.
    
    Spectral entropy measures how spread the power is across frequencies.
    High spectral entropy indicates broadband signal (power spread),
    low spectral entropy indicates narrowband signal (power concentrated).
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Time series signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    freq_range : Tuple[float, float], optional
        Frequency range (fmin, fmax) to consider. Default is None (all).
    normalize : bool, optional
        Whether to normalize by maximum entropy. Default is True.
    
    Returns
    -------
    Tuple[float, NDArray, NDArray]
        (spectral_entropy, frequencies, psd)
    
    Examples
    --------
    >>> t = np.arange(0, 10, 1/256)
    >>> signal = np.sin(2 * np.pi * 10 * t)  # Pure sine
    >>> H_spec, freqs, psd = compute_spectral_entropy(signal, fs=256)
    >>> # H_spec will be low (narrowband)
    """
    # Compute PSD using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    
    # Apply frequency range if specified
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[mask]
    
    # Normalize PSD to make it a probability distribution
    psd_norm = psd / np.sum(psd)
    
    # Remove zeros
    psd_valid = psd_norm[psd_norm > 0]
    
    # Compute entropy
    H_spectral = -np.sum(psd_valid * np.log2(psd_valid))
    
    # Normalize if requested
    if normalize:
        H_max = np.log2(len(psd_valid))
        if H_max > 0:
            H_spectral = H_spectral / H_max
    
    return float(H_spectral), freqs, psd


def compute_sample_entropy(
    signal: NDArray[np.float64],
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute sample entropy of a time series.
    
    Sample entropy measures the complexity/irregularity of a signal
    by comparing patterns at different time points. It does not
    require binning.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Time series signal.
    m : int, optional
        Embedding dimension (pattern length). Default is 2.
    r : float, optional
        Tolerance for pattern matching. Default is 0.2 * std(signal).
    
    Returns
    -------
    float
        Sample entropy value.
        Higher values indicate more complexity/irregularity.
        Lower values indicate more self-similarity/regularity.
    
    Notes
    -----
    Computational complexity is O(n²), so use shorter signals
    (e.g., 500-2000 samples) for reasonable computation time.
    
    Examples
    --------
    >>> regular_signal = np.sin(np.linspace(0, 10*np.pi, 500))
    >>> random_signal = np.random.randn(500)
    >>> se_regular = compute_sample_entropy(regular_signal)
    >>> se_random = compute_sample_entropy(random_signal)
    >>> # se_random > se_regular
    """
    N = len(signal)
    
    if r is None:
        r = 0.2 * np.std(signal)
    
    def count_matches(template_length: int) -> int:
        """Count pairs of matching templates."""
        templates = np.array([
            signal[i:i + template_length] 
            for i in range(N - template_length)
        ])
        
        count = 0
        n_templates = len(templates)
        
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                # Chebyshev distance (max absolute difference)
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        
        return count
    
    # Count matches for m and m+1
    A = count_matches(m + 1)  # matches for length m+1
    B = count_matches(m)      # matches for length m
    
    # Sample entropy
    if A == 0 or B == 0:
        return np.inf  # No matches found
    
    return -np.log(A / B)
