import datetime
import numpy as np
import toolbox.dsp
import torch
import torchaudio as ta
import torchaudio.functional as taf

from toolbox.initialization import *
from toolbox.dsp import db2mag, mag2db, mel2hz, hz2mel, erb, envelope
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy.interpolate import interp1d
from typing import Union, io


class DHASP:
    """
    Based on:
    "resources/literature/DIFFERENTIABLE HEARING AID SPEECH PROCESSING.pdf"
    """

    def __init__(self, fs, n_taps=500):

        # Signal level corresponding to 60 dB SPL
        self.db_ref_60_db = -18

        # Number of taps for FIR filters.
        self.n_taps = n_taps

        # Sampling frequency
        self.fs = fs

        # Number of filter frequencies
        self.I = 32

        # The centre frequencies of the analysis filterbank f_a are in the
        # Mel scale covering the range from 80 Hz to 8 kHz:
        self.f_a = mel2hz(torch.linspace(
            hz2mel(80),
            hz2mel(8000),
            self.I
        )).reshape(self.I, 1)

        # The bandwidths the auditory filter for a normal hearing person at f_a
        # b_NH_a are in the equivalent rectangular bandwidth (ERB)
        # scale for the normal hearing model:
        self.b_NH_a = erb(self.f_a)

        # Hearing loss for outer cells at f_a:
        self.attn_o = 0 * torch.ones_like(self.f_a)

        # Max allowed value is 50
        self.attn_o[self.attn_o > 50] = 50

        # The bandwidths of the control filters correspond to the maximum
        # bandwidth allowed in the model, i.e. 50 dB attenuation for
        # outer-hair cell.The control filters are set wider so that they
        # could reduce the gain of a signal outside the bandwidth of the
        # analysis filters but still within the control filters. Each center
        # frequency of the control filter f_c_i is shifted higher relative to
        # the center frequency of the corresponding analysis filter f_a_i
        # a using a human frequency-position function
        s = 0.02
        self.f_c = 165.4 * (
                torch.float_power(
                    10,
                    (1 + s) * torch.log10(1 + self.f_a / 165.4)
                )
                - 1
        )

        # The bandwidths b_NH a are in the equivalent rectangular bandwidth (
        # ERB) scale for the normal hearing model. To approximate the
        # behaviour that the auditory filter bandwidths increase along with
        # the hearing loss the bandwidths b_HL_a of the hearing loss model is
        # expressed as:
        self.b_HL_a = (1
                       + self.attn_o / 50
                       + 2 * torch.float_power(self.attn_o / 50, 6)
                       ) * self.b_NH_a

        # The bandwidths of the control filters correspond to the maximum
        # bandwidth allowed in the model, i.e. 50 dB attenuation for
        # outer-hair cells.
        self.b_c = (1
                    + 50 / 50
                    + 2 * np.float_power(50 / 50, 6)
                    ) * self.b_NH_a

        # Calculate the filter coefficients for the analysis and control
        # filterbanks.
        self.h_a = self.__calculate_h('analysis')
        self.h_c = self.__calculate_h('control')

        # Compression thresholds.
        self.theta_low = self.attn_o + 30
        self.theta_high = float(100)

        # Compression ratio.
        CR = interp1d(
            np.array([0, 80, 8e3, 20e3]),
            np.array([1.25, 1.25, 3.5, 3.5]),
        )(self.f_c.numpy())

        self.CR = torch.from_numpy(CR)

    def __calculate_h(self, filter_variant):
        """
        Transfer function of the FIR filters in the analysis filterbank.
        Based on Eq 1 in the paper:
        :return:
        :rtype:
        """
        # Filter order
        N = 4

        if filter_variant == 'analysis':
            f = self.f_a
            b = self.b_NH_a

        elif filter_variant == 'control':
            f = self.f_c
            b = self.b_c

        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".'
                             f'Fiter variant should nw "analysis" or '
                             f'"control".')

        t = torch.arange(self.n_taps) * 1 / self.fs

        # Angular frequencies
        w_f = 2 * np.pi * f
        w_b = 2 * np.pi * b

        h = (
                torch.float_power(t, N - 1)
                * torch.exp(-w_b * t)
                * torch.cos(w_f * t)
        )

        # Normalize so that peak gain = 1
        peak_gain = torch.linalg.norm(torch.vstack([
            torch.sum(h * torch.cos(w_f * t), axis=1),
            torch.sum(h * torch.sin(w_f * t), axis=1)
        ]),
            axis=0
        )

        h = h / peak_gain.reshape(self.I, 1)

        return h

    def __get_filter_coefficents(self, filter_variant):
        if filter_variant == 'analysis':
            return self.h_a

        elif filter_variant == 'control':
            return self.h_c
        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".'
                             f'Fiter variant should nw "analysis" or '
                             f'"control".')

    def __to_dbspl(self, magnitude: torch.Tensor):
        return mag2db(magnitude) - self.db_ref_60_db + 60

    def calculate_G(self, outputs_control_filter):
        # Calculate envelopes of the outputs of the control filter.
        E_c = self.__to_dbspl(
            envelope(outputs_control_filter, axis=1)
        ).to(torch.float64)

        # Apply the thresholds
        E_hat_c = torch.where(
            E_c > self.theta_high,
            self.theta_high,
            E_c
        )

        E_hat_c = torch.where(
            E_hat_c < self.theta_low,
            self.theta_low,
            E_hat_c
        )

        # Get the gain in dB
        G = (-self.attn_o
             - (1 - 1 / self.CR) * (E_hat_c - self.theta_low))

        return(G)

    def apply_filter(self,
                     filter_variant,
                     signal: torch.Tensor):
        # Get the filter coefficients.
        h = self.__get_filter_coefficents(filter_variant)

        # Preallocate the response matrix.
        response = torch.zeros((h.shape[0], signal.shape[1]))

        # Compute the denominator
        a = torch.zeros(h.shape[1])
        a[0] = 1

        # Apply the filters.
        for i in range(h.shape[0]):
            response[i, :] = taf.lfilter(signal,
                                         a,
                                         h[i, :].squeeze().to(torch.float32),
                                         clamp=True)

        return response

    def apply_compression(self,
                          G: torch.Tensor,
                          outputs_anaysis_flter: torch.Tensor):

        return outputs_anaysis_flter * db2mag(G)

    def show_filterbank_responses(
            self,
            filter_variant,
            show_every=1
    ):
        # Get the filter center frequencies and the corresponding filter
        # coefficients.
        if filter_variant == 'analysis':
            frequencies = self.f_a.squeeze().numpy()
            h = self.h_a.numpy()

        elif filter_variant == 'control':
            frequencies = self.f_c.squeeze().numpy()
            h = self.h_c.numpy()

        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".'
                             f'Fiter variant should be "analysis" or '
                             f'"control".')

        # Create figure and axes for the plot
        figure, axes = plt.subplots(figsize=(12, 8))

        for idx_freq, frequency in enumerate(frequencies):
            if not (idx_freq % show_every):
                frequency_axis, response = freqz(h[idx_freq, :],
                                                 fs=self.fs)
                axes.semilogx(frequency_axis,
                              mag2db(np.abs(response)),
                              linewidth=2,
                              label=f'$f_c$ = {frequency:,.0f} Hz')
                axes.set_xlabel('Frequency [Hz])')
                axes.set_ylabel('Gain [dB]')
                axes.set_title(
                    f'Frequency response of the filters in the '
                    f'{filter_variant} filterbank.'
                    f'\nShowing every {show_every} filters.')

        axes.legend()
        axes.grid()

    def show_filterbank_joint_response(self,
                                       filter_variant):
        # Get filter coefficient
        h = self.__calculate_h(filter_variant)

        # Calculate a joint response of the filter
        joint_h = h.sum(dim=0)

        # Create figure and axes for the plot
        figure, axes = plt.subplots(figsize=(12, 8))

        frequency_axis, response = freqz(joint_h, fs=self.fs)

        # Plot
        axes.semilogx(frequency_axis,
                      mag2db(np.abs(response)),
                      linewidth=2, )
        axes.set_xlabel('Frequency [Hz]')
        axes.set_ylabel('Gain [dB]')
        axes.set_title(f'Joint frequency response of the {filter_variant}'
                       f' filterbank')
        axes.grid()

    def compression(self, filter_variant):
        if filter_variant == 'analysis':
            frequencies = self.f_a
        elif filter_variant == 'control':
            frequencies = self.f_c
        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".'
                             f'Fiter variant should nw "analysis" or '
                             f'"control".')

        # Lower and upper threshold
        theta_low = self.attn_o + 30
        theta_high = 100 * np.ones_like(self.attn_o)

        # Compression rate at frequencies
        CR = interp1d(
            np.array([80, 8e3]),
            np.array([1.25, 3.5]),
        )(frequencies)
