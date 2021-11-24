import numpy as np
from resampy import resample
from typing import Union


def rms(signal: np.ndarray):
    return np.sqrt(np.mean(np.power(signal, 2)))


def db2mag(magnitude_db: np.ndarray):
    return np.power(10, magnitude_db / 20)


def mag2db(magnitude: np.ndarray):
    return 20 * np.log10(magnitude)


def mel2hz(mel):
    return 700 * (np.power(10, mel / 2595) - 1)


def hz2mel(hz):
    return (2595 * np.log10(1 + hz / 700))


def erb(frequency):
    """
    Equivalent rectangular bandwidth of an auditory filter at a given frequency
    :param frequency:
    :type frequency:
    :return:
    :rtype:
    """

    return (
            6.23 * np.power(frequency / 1000, 2)
            + 93.39 * frequency / 1000
            + 28.52
    )


def magnitude_spectrum(signal: np.array,
                       fs,
                       in_db=False):

    # Number of samples in the signal.
    n = len(signal)

    # Create a frequency axes
    frequency = np.linspace(
        0.0,
        0.5 * fs,
        n // 2
    )

    # Calculate the magnitude spectrum
    spectrum = np.abs(np.fft.fft(signal))[:(n // 2)]

    # Convert to dB
    if in_db:
        spectrum = mag2db(spectrum)

    return frequency, spectrum
