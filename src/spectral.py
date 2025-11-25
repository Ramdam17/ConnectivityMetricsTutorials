"""Spectral analysis functions for frequency domain operations.

This module provides functions for computing FFT, amplitude spectrum,
phase spectrum, and frequency resolution for signal analysis.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq


def compute_fft(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Compute the Fast Fourier Transform of a signal.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    fft_values : NDArray[np.complexfloating]
        Complex FFT coefficients.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> frequencies, fft_values = compute_fft(signal, fs=1000)
    """
    n_samples = len(signal)
    fft_values = fft(signal)
    frequencies = fftfreq(n_samples, 1 / fs)
    return frequencies, fft_values


def compute_amplitude_spectrum(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the amplitude spectrum (positive frequencies only).

    The amplitude is properly scaled to recover the original signal amplitudes.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of positive frequency values in Hz.
    amplitude : NDArray[np.floating]
        Amplitude at each frequency (properly scaled).

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = 2.0 * np.sin(2 * np.pi * 10 * t)  # Amplitude = 2
    >>> freqs, amps = compute_amplitude_spectrum(signal, fs=1000)
    >>> # Peak at 10 Hz should be approximately 2.0
    """
    n_samples = len(signal)
    frequencies, fft_values = compute_fft(signal, fs)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    fft_pos = fft_values[positive_mask]

    # Scale: divide by N and multiply by 2 (except DC)
    amplitude = np.abs(fft_pos) * 2 / n_samples
    amplitude[0] /= 2  # DC component shouldn't be doubled

    return frequencies_pos, amplitude


def compute_phase_spectrum(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the phase spectrum (positive frequencies only).

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of positive frequency values in Hz.
    phase : NDArray[np.floating]
        Phase angle at each frequency in radians (-π to π).

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 10 * t + np.pi/4)  # Phase = π/4
    >>> freqs, phases = compute_phase_spectrum(signal, fs=1000)
    """
    frequencies, fft_values = compute_fft(signal, fs)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    fft_pos = fft_values[positive_mask]

    # Extract phase
    phase = np.angle(fft_pos)

    return frequencies_pos, phase


def compute_frequency_resolution(fs: float, n_samples: int) -> float:
    """Compute the frequency resolution of an FFT.

    The frequency resolution (Δf) determines the spacing between
    frequency bins and the ability to distinguish nearby frequencies.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of samples in the signal.

    Returns
    -------
    float
        Frequency resolution in Hz (Δf = fs / N).

    Notes
    -----
    To resolve two frequencies that are Δf apart, you need at least
    1/Δf seconds of data. For example, to resolve 1 Hz difference,
    you need at least 1 second of data.

    Examples
    --------
    >>> compute_frequency_resolution(fs=1000, n_samples=1000)
    1.0
    >>> compute_frequency_resolution(fs=1000, n_samples=2000)
    0.5
    """
    return fs / n_samples
