import toolbox as t
from toolbox.imports import *
from toolbox.dsp import db2mag, mag2db, mel2hz, hz2mel, erb, envelope
from scipy.signal import freqz
from scipy.interpolate import interp1d



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

        # Frequencies and bandwidths for the EQ that will be applied to noisy
        # signal when optimizing for highest intelligibility.
        self.f_eq = torch.from_numpy(np.geomspace(100, 8e3, 8)).reshape(-1, 1)
        self.b_eq = self.f_eq / 2.3

        # Calculate the fir filter coefficients for the EQ
        self.h_eq = self.__calculate_h('eq')



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
            total_gain = 1

        elif filter_variant == 'control':
            f = self.f_c
            b = self.b_c
            total_gain = 1

        elif filter_variant == 'eq':
            f = self.f_eq
            b = self.b_eq
            total_gain = db2mag(2.5)

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

        h = total_gain * h / peak_gain.reshape(-1, 1)

        return h

    def __get_filter_coefficents(self, filter_variant):
        if filter_variant == 'analysis':
            return self.h_a

        elif filter_variant == 'control':
            return self.h_c

        elif filter_variant == 'eq':
            return self.h_eq
        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".'
                             f'Filter variant should be "analysis", '
                             f'"control", or "eq".')

    def __to_dbspl(self, magnitude: torch.Tensor):
        return mag2db(magnitude) - self.db_ref_60_db + 60

    def apply_filter(self,
                     filter_variant,
                     waveforms: torch.Tensor):
        # Get the filter coefficients.
        h = (self
             .__get_filter_coefficents(filter_variant)
             .to(torch.float32))

        # Compute the filter responses.
        return tnnf.conv1d(
            waveforms.reshape(
                waveforms.shape[0],
                1,
                waveforms.shape[1]
            ),
            h.reshape(h.shape[0], 1, h.shape[1]),
            padding=h.shape[-1] // 2
        )

    def calculate_G(self, outputs_control_filter):
        # Calculate envelopes of the outputs of the control filter.
        E_c = self.__to_dbspl(
            envelope(outputs_control_filter).to(torch.float64)
        )

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

        return (G)

    def calculate_output(self,
                         waveforms: torch.Tensor):

        # Get the outputs of the control and analysis filters
        outputs_control_filter = self.apply_filter('control', waveforms)
        outputs_analysis_filter = self.apply_filter('analysis', waveforms)

        # Calculate the dynamic range compression gain.
        G = self.calculate_G(outputs_control_filter)

        # Calculate the output of the auditory model in each frequency band.
        output = outputs_analysis_filter * db2mag(G)

        # Calculate the envelopes of the output in dB in each frequency band.
        output_envelope = mag2db(envelope(output))

        return output, output_envelope

    def smooth_output_envelope(self, output_envelope: torch.Tensor):
        """
        :param envelope_model:  Tensor with 32 rows representing the envelopes
        (ind dB) of the output of the auditory model in each filterbank.
        :type envelope_model: torch.Tensor
        :return:
        :rtype:
        """

        # Make sure that the data type is "float32".
        output_envelope = output_envelope.to(torch.float32)

        # Number of samples corresponding to 16 ms.
        n_samples_16_ms = int(16e-3 * self.fs)

        # Overlap is set to 50 %
        stride = n_samples_16_ms // 2

        # Get the weights of the Hanning window we will use for smoothing.
        weights = torch.hann_window(n_samples_16_ms)
        weights = weights / weights.sum()

        # Smooth the envelope.

        # Initialize the variable containing the smoothed envelope
        # (so that the function will always have an output)
        envelope_smoothed = None

        for idx_waveform in range(output_envelope.shape[0]):
            envelope = output_envelope[idx_waveform, :, :]

            envelope_smoothed_temp = tnnf.conv1d(
                envelope.reshape(envelope.shape[0],
                                 1,
                                 envelope.shape[1]),
                weights.reshape(1, 1, -1),
                stride=stride
            ).permute((1, 0, -1))

            # Gather the smoothed envelopes in a tensor whose first two
            # dimensions are the same as the ones of the non-smoothed
            # envelopes.
            if idx_waveform == 0:
                envelope_smoothed = envelope_smoothed_temp
            else:
                envelope_smoothed = torch.cat(
                    [envelope_smoothed, envelope_smoothed_temp],
                    dim=0
                )

        return envelope_smoothed, stride

    def calculate_C(self,
                    from_value: torch.Tensor,
                    from_value_type='smoothed_output_envelope'):

        # Compute the tensor with indices of the auditory filters
        i = torch.arange(self.I).reshape(self.I, 1)

        # Compute the tensor with inices of the basis functions
        j = torch.arange(2, 7)

        # Compute the basis functions for every combination oi i (rows)
        # and j (columns).
        bas = torch.cos((j - 1) * np.pi * i / (self.I - 1))

        # Get the smoothed envelope of the output of the auditory model.
        if from_value_type == 'smoothed_output_envelope':
            output_envelope_smoothed = from_value

        elif from_value_type == 'waveforms':
            waveforms = from_value
            _, output_envelope = self.calculate_output(waveforms)
            output_envelope_smoothed, _ = \
                self.smooth_output_envelope(output_envelope)

        else:
            raise ValueError(f'Invalid from value type: "{from_value_type}".')

        # Initialize the output variable.
        C = torch.zeros([
            output_envelope_smoothed.shape[0],
            j.shape[-1],
            output_envelope_smoothed.shape[2]
        ])

        # Compute the cepstral sequence:
        for idx_waveform in range(output_envelope_smoothed.shape[0]):
            C[idx_waveform, :, :] = torch.matmul(
                bas.T,
                output_envelope_smoothed[idx_waveform, :, :]
            )

        if from_value_type == 'smoothed_output_envelope':
            return C
        elif from_value_type == 'waveforms':
            return C, output_envelope_smoothed

    def calculate_R(self,
                    C_r: torch.Tensor,
                    C_p: torch.Tensor):
        """
        Calculate correlation of cepstral sequences
        C_r: Target cepstral sequence:  Tensor of size (j,  m)
        C_p: Cepstral sequence of the processed sequences:
             Tensor of size (n_waveforms, j, m)
        Output: Tensor of size (n_waveforms, j)
        """
        # Initialize R
        R = torch.zeros(
            C_p.shape[0],
            C_p.shape[1]
        )

        # Calculate the correlations for each processed waveform.
        for idx_waveform in range(C_p.shape[0]):
            c_p = C_p[idx_waveform, :]
            R[idx_waveform, :] = (
                    (C_r * c_p).sum(dim=1)
                    / (
                            torch.linalg.norm(C_r, dim=1)
                            * torch.linalg.norm(c_p, dim=1)
                    )
            )

        return R

    def calculate_L_e(self,
                      E_r: torch.Tensor,
                      E_p: torch.Tensor):
        """
        Calculate correlation of cepstral sequences
        E_r: Target smoothed output envelope:  Tensor of size (i,  m)
        E_p: Smoothed output envelope of the processed sequences:
             Tensor of size (n_waveforms, i, m)
        Output: Tensor of size (n_waveforms, i)
        """

        # Initialize R
        L_e = torch.zeros(
            E_p.shape[0],
            E_p.shape[1]
        )

        # Calculate the L_e for each processed waveform.
        for idx_waveform in range(E_p.shape[0]):
            e_p = E_p[idx_waveform, :, :]
            l_e = e_p - E_r
            l_e[l_e < 0] = 0
            l_e = l_e.sum(dim=1)
            L_e[idx_waveform, :] = l_e

            return L_e

    def calculate_L(self,
                    R: torch.Tensor,
                    L_e: torch.Tensor,
                    alpha=1e-5):
        """
        Calculate total loss.
        R: Correlation, tensor, size: (n_waveforms, j)
        L_e_: Energy loss, tensor, size (n_waveforms, i)
        alpha: energy loss weight
        """

        return -R.mean(dim=1) + alpha * L_e.sum(dim=1)

    def calculate_loss(self,
                       waveforms: torch.Tensor,
                       waveform_target: torch.Tensor,
                       alpha: float):
        """
        waveforms: tensor of size (n_waveforms, n_samples)
        waveforms_target: tensor of size (1, n_samples)
        """

        # Put all waveforms in one tensor.
        all_waveforms = torch.vstack([
            waveform_target,
            waveforms
        ])

        # Get the outputs of the control and analysis filters
        outputs_control_filter = self.apply_filter('control', all_waveforms)
        outputs_analysis_filter = self.apply_filter('analysis', all_waveforms)

        # Calculate the dynamic range compression gain.
        G = self.calculate_G(outputs_control_filter)

        # Calculate the output of the auditory model in each frequency band.
        output = outputs_analysis_filter * db2mag(G)

        # Calculate the envelopes of the output in dB in each frequency band.
        output_envelope = mag2db(envelope(output))

        # Smooth the output envelope
        E, _ = self.smooth_output_envelope(output_envelope)

        # Calculate the cepstral sequences
        C = self.calculate_C(E,
                             from_value_type='smoothed_output_envelope')

        # Separate the target from the rest.
        E_r = E[0, :, :].squeeze(0)
        E_p = E[1:, :, :]

        C_r = C[0, :, :].squeeze(0)
        C_p = C[1:, :, :]

        # Calculate cepstral correlations.
        R = self.calculate_R(C_r, C_p)

        # Calculate the energy loss.
        L_e = self.calculate_L_e(E_r, E_p)

        # Calculate the total loss.
        L = self.calculate_L(R, L_e)

        return L, R, L_e

    # --------- Plotting functions ---------
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

        elif filter_variant == 'eq':
            frequencies = self.f_eq.squeeze().numpy()
            h = self.h_eq.numpy()

        else:
            raise ValueError(f'Invalid filter variant: "{filter_variant}".')

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
                axes.set_xlabel('Frequency [Hz]')
                axes.set_ylabel('Gain [dB]')
                axes.set_title(
                    f'Frequency response of the filters in the filterbank: '
                    f'"{filter_variant}"'
                    f'\nShowing every {show_every} filters')

        axes.legend()
        axes.grid()
        t.plotting.apply_standard_formatting(figure)

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
        axes.set_title(f'Joint frequency response of the filterbank: '
                       f'"{filter_variant}"')
        axes.grid()

        t.plotting.apply_standard_formatting(figure)

        # Lower the yaxis range for the eq filter
        if filter_variant == 'eq':
            axes.set_ylim([-20, 10])