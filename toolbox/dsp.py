import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchaudio as ta
import torchaudio.functional as taf

from resampy import resample
from scipy.signal import hilbert
from typing import Union
from toolbox.type_conversion import ensure_torch


def rms(signal: Union[np.ndarray, torch.Tensor]):
    if isinstance(signal, torch.Tensor):
        return torch.sqrt(torch.mean(torch.float_power(signal, 2)))
    else:
        return np.sqrt(np.mean(np.power(signal, 2)))


def db2mag(magnitude_db: Union[np.ndarray,
                               torch.Tensor,
                               float,
                               int]):
    if isinstance(magnitude_db, torch.Tensor):
        return torch.float_power(10, magnitude_db / 20)
    else:
        return np.power(10, magnitude_db / 20)


def mag2db(magnitude: Union[np.ndarray,
                            torch.Tensor,
                            float,
                            int]):
    if isinstance(magnitude, torch.Tensor):
        return 20 * torch.log10(magnitude)
    else:
        return 20 * np.log10(magnitude)


def mel2hz(mel: Union[np.ndarray,
                      torch.Tensor,
                      float,
                      int]):
    if isinstance(mel, torch.Tensor):
        return 700 * (torch.float_power(10, mel / 2595) - 1)
    else:
        return 700 * (np.power(10, mel / 2595) - 1)


def hz2mel(hz: Union[np.ndarray,
                     torch.Tensor,
                     float,
                     int]):
    if isinstance(hz, torch.Tensor):
        return 2595 * torch.log10(1 + hz / 700)
    else:
        return 2595 * np.log10(1 + hz / 700)


def erb(frequency: Union[np.ndarray,
                         torch.Tensor,
                         float,
                         int]):
    """
    Equivalent rectangular bandwidth of an auditory filter at a given frequency
    :param frequency:
    :type frequency:
    :return:
    :rtype:
    """
    if isinstance(frequency, torch.Tensor):
        return (
                6.23 * torch.float_power(frequency / 1000, 2)
                + 93.39 * frequency / 1000
                + 28.52
        )
    else:
        return (
                6.23 * np.power(frequency / 1000, 2)
                + 93.39 * frequency / 1000
                + 28.52
        )


def magnitude_spectrum(signal: Union[np.ndarray, torch.Tensor],
                       fs,
                       in_db=False):
    if isinstance(signal, torch.Tensor):

        # Make the tensor one-dimensional.
        signal = signal.squeeze()

        # Number of samples in the signal.
        n = signal.shape[0]

        # Create a frequency axes
        frequency = torch.linspace(
            0.0,
            0.5 * fs,
            n // 2 + 1
        )

        # Calculate the magnitude spectrum
        spectrum = torch.abs(torch.fft.rfft(signal))

        # Convert to dB
        if in_db:
            spectrum = mag2db(spectrum)

        return frequency, spectrum

    else:
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


def envelope(x: Union[np.ndarray, torch.Tensor],
             axis=-1):
    if isinstance(x, torch.Tensor):
        N = x.shape[axis]
        Xf = torch.fft.fft(x, N)
        h = torch.zeros(N)

        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1: N // 2] = 2
        else:
            h[0] = 1
            h[1: (N + 1) // 2] = 2

        if x.ndim > 1:
            new_shape = [1 for _ in range(x.ndim)]
            new_shape[axis] = len(h)
            h = h.reshape(*tuple(new_shape))

        # Get the input envelope
        amplitude_envelope = torch.abs(torch.fft.ifft(Xf * h))

    else:
        amplitude_envelope = np.abs(hilbert(x, axis=axis))

    return amplitude_envelope


def filter_signal(waveforms: torch.Tensor,
                  h: torch.Tensor,
                  gain: torch.Tensor = None,
                  joint_response=False):
    """
    waveforms: tensor size (n_observations, n_samples)
    h: tensor size (n_channels, n_fir_coeff)
    gain: tensor size: (n_filter_channels)
    """

    # Apply gain.
    if gain is not None:
        # Make gain a column vector
        gain = gain.reshape(-1, 1)

        # Check dimensions.
        if gain.shape[0] != h.shape[0]:
            raise ValueError('The number of channels of the gain does not '
                             'match the number of channels in the filter')

        # Apply gain to each channel.
        h = gain * h


    # If chosen, sum all the filters
    if joint_response:
        h = h.sum(dim=0).reshape(1, -1)

    # Calculate response.
    response = nnf.conv1d(
        waveforms.reshape(
            waveforms.shape[0],
            1,
            waveforms.shape[1]
        ),
        h.reshape(h.shape[0], 1, h.shape[1]),
        padding=h.shape[-1] // 2
    )

    # If joint_response, remove the superficial channel dimension.
    if joint_response:
        response = response.squeeze(1)

    return response


