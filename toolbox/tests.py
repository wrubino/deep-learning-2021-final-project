from toolbox.initialization import *


class Tests:

    def test_all(self):
        self.test_dhasp_filters()
        self.test_apply_filter()
        self.test_envelope()
        self.test_compression()
        self.test_output_auditory_model()
        self.test_smoothing_output_envelope()
        self.test_cepstral_sequence()
        self.test_cepstral_correlation_and_total_loss()


    def test_apply_filter(self):
        # %% Initialize the dhasp model.
        fs_model = 24e3
        dhasp = t.dhasp.DHASP(fs_model)

        # %% Create a test signal (white noise)
        time = torch.arange(0, 2, 1 / fs_model)
        white_noise = 2 * torch.rand_like(time).unsqueeze(0) - 0.25
        white_noise_20_dB_louder = 10 * white_noise
        white_noise_filtered = dhasp.apply_filter('analysis', torch.vstack(
            [white_noise, white_noise_20_dB_louder]))

        # %% Test the analysis filterbank on audio

        # The index of the filter to apply
        idx_filter = 3
        for filterbank in ['analysis', 'control', 'eq']:
            white_noise_filtered = dhasp.apply_filter(filterbank, torch.vstack(
                [white_noise, white_noise_20_dB_louder]))

            if filterbank == 'analysis':
                frequencies = dhasp.f_a
            elif filterbank == 'control':
                frequencies = dhasp.f_c
            elif filterbank == 'eq':
                frequencies = dhasp.f_eq

            # Get the filter frequency.
            frequency = frequencies.numpy().flatten()[idx_filter]

            t.plotting.plot_magnitude_spectrum(
                {'Unfiltered, 0 dB': white_noise,
                 'Filtered, 0 dB': white_noise_filtered[0, idx_filter, :],
                 'Filtered, 20 dB': white_noise_filtered[1, idx_filter, :]},
                fs_model,
                title=f'Application of filter number {idx_filter + 1} '
                      f'with $f_c$={frequency:.0f} Hz '
                      f'from the filterbank: "{filterbank}" on white noise'
            )

    def test_cepstral_correlation_and_total_loss(self):
        # %% Load Audio

        # Load audio samples representing the same speech segment in different
        # acoustic conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Specify the target (clean) waveform
        waveform_target = audio_samples['clean']

        # Get the other versions of the waveform
        waveforms = torch.vstack(list(audio_samples.values()))

        # Get the waveform names for plotting.
        waveform_names = list(audio_samples.keys())

        # %% Initialize the dhasp model.
        dhasp = t.dhasp.DHASP(fs_model)

        # % Get C and E for target and the other waveforms

        # Get the cepstral sequences and the smoothed envelope of the output of the
        # auditory model
        C_r, E_r = dhasp.calculate_C(waveform_target,
                                     from_value_type='waveforms')
        C_r = C_r.squeeze(0)
        E_r = E_r.squeeze(0)

        C_p, E_p = dhasp.calculate_C(waveforms, from_value_type='waveforms')

        # %% Calculate the correlations.
        R = dhasp.calculate_R(C_r, C_p)

        # %% Calculate the energy loss.
        L_e = dhasp.calculate_L_e(E_r, E_p)

        # %% Calculate the total loss
        alpha = 1e-5
        L = dhasp.calculate_L(R, L_e, alpha)

        # %% Calculate everything in one hook.

        start_time = time.time()

        L_2, R_2, L_e_2 = dhasp.calculate_loss(waveforms, waveform_target,
                                               alpha)

        stop_time = time.time()
        print(f'Execution time: {stop_time - start_time:.2f} s.')

        # %% Show results
        data = {
            'Variant': waveform_names,
            'R': R.mean(dim=1).numpy(),
            'L_e': L_e.sum(dim=1).numpy(),
            'L': L.numpy()
        }

        df_results = pd.DataFrame(data)

        display(df_results)

    def test_cepstral_sequence(self):
        # %% Load Audio

        # Load audio samples representing the same speech segment in different
        # acoustic conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Get the first 3 seconds of the clean sample.
        waveform = audio_samples['clean'][: int(fs_model) * 3]

        # Make 2 versions, quiet and loud.
        waveforms = torch.vstack([
            waveform,
            10 * waveform
        ])

        # Assign names for plotting.
        waveform_names = ['0 dB', '+20 dB']

        # %% Initialize the dhasp model.
        dhasp = t.dhasp.DHASP(fs_model)

        # %% Get the output of the auditory model.
        output, output_envelope = dhasp.calculate_output(waveforms)

        # %% Get the smoothed envelope

        # Get the smoothed envelope and the stride used when smoothing.
        output_envelope_smoothed, stride = \
            dhasp.smooth_output_envelope(output_envelope)

        # %% Compute cepstral sequence
        C = dhasp.calculate_C(
            output_envelope_smoothed,
            from_value_type='smoothed_output_envelope'
        )

        # %% Visualize
        # The index of the filter and number of samples to show
        idx_filter = 10
        j = 5

        # Create the time axis for the plot.
        time_smoothed = \
            np.arange(output_envelope_smoothed.shape[-1]) / fs_model * stride

        # Create figure and axes for the plots.
        figure, axess = plt.subplots(2, 1, figsize=(12, 8))

        for idx_waveform, (waveform_name, axes) \
                in enumerate(zip(waveform_names, axess.flatten())):

            axes_right = axes.twinx()

            # Initialize a list of curve handles.
            curves = list()

            curves += axes.plot(
                time_smoothed,
                output_envelope_smoothed[idx_waveform, idx_filter, :],
                label='Left axis: Smoothed envelope [dB]',
            )

            curves += axes_right.plot(
                time_smoothed,
                C[idx_waveform, j - 2, :],
                label=f'Right axis: Cepstral sequence for j = {j - 2}',
                color='red'
            )

            curves += axes_right.plot(
                time_smoothed,
                C[idx_waveform, j - 2 + 1, :],
                label=f'Right axis: Cepstral sequence for j = {j + 1}',
                color='green'
            )

            # # Show grid
            axes.grid()

            if idx_waveform == 1:
                axes.legend(curves, [curve.get_label() for curve in curves],
                            bbox_to_anchor=(0, -0.15, 1, 0),
                            loc="upper left",
                            mode="expand",
                            ncol=len(curves),
                            frameon=False
                            )

            axes.set_title(
                f'Waveform: {waveform_name}, '
                f'Filter number: {idx_filter + 1}, '
                f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz'
            )
            axes.set_xlabel('Time [s]')

        t.plotting.apply_standard_formatting(figure, include_grid=False)
        figure.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    def test_compression(self):
        # %% Load Audio

        # Load variations of the same speech segment in different conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Get the first 3 seconds of the clean sample.
        waveform = audio_samples['clean'][: int(fs_model) * 3]

        # Make 2 versions, quiet and loud.
        waveforms = torch.vstack([
            0.1 * waveform,
            10 * waveform
        ])

        # Assign names for plotting.
        waveform_names = ['-20 dB', '+20 dB']

        # %% Initialize the dhasp model

        dhasp = t.dhasp.DHASP(fs_model)

        # %% Get the outputs of the filters

        outputs_control_filter = dhasp.apply_filter('control', waveforms)
        envelopes_control_filter = t.dsp.envelope(outputs_control_filter)

        # %% Get dynamic range compression gain
        G = dhasp.calculate_G(outputs_control_filter)

        # %% Visualize
        # The index of the filter and number of samples to show
        idx_filter = 2
        n_samples = int(3e3)

        # Create the time axis for the plot
        time = np.arange(n_samples) / fs_model

        # Initialize a list of curve handles.
        curves = list()

        # Plot filter output and its envelope.
        figure, axess = plt.subplots(2, 1, figsize=(12, 8))

        for idx_waveform, (waveform_name, axes) \
                in enumerate(zip(waveform_names, axess.flatten())):
            curves = list()
            axes_right = axes.twinx()

            curves += axes.plot(
                time,
                outputs_control_filter[idx_waveform, idx_filter, :n_samples],
                label=f'Left axis: Output of control filter'
            )

            curves += axes.plot(
                time,
                envelopes_control_filter[idx_waveform, idx_filter, :n_samples],
                label='Left axis: Envelope'
            )

            curves += axes_right.plot(
                time,
                G[idx_waveform, idx_filter, :n_samples],
                color='red',
                label='Right axis: Compression gain [dB]'
            )

            axes.grid()
            if idx_waveform == 1:
                axes.legend(curves, [curve.get_label() for curve in curves],
                            bbox_to_anchor=(0, -0.15, 1, 0),
                            loc="upper left",
                            mode="expand",
                            ncol=len(curves),
                            frameon=False
                            )

            axes.set_title(f'Waveform: {waveform_name}, '
                           f'Filter number: {idx_filter + 1}, '
                           f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
            axes.set_xlabel('Time [s]')
            axes.set_ylabel('Signal value')
            axes_right.set_ylabel('Compression gain [dB]')

        t.plotting.apply_standard_formatting(figure, include_grid=False)
        figure.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    def test_dhasp_filters(self):
        fs_model = 24e3
        dhasp = t.dhasp.DHASP(fs_model)

        # %% Show the analysis and control filterbank of the auditory model.

        # Analysis filterbank
        dhasp.show_filterbank_responses('analysis', show_every=2)
        plt.show()
        dhasp.show_filterbank_joint_response('analysis')
        plt.show()

        # Control filterbank
        dhasp.show_filterbank_responses('control', show_every=2)
        plt.show()
        dhasp.show_filterbank_joint_response('control')
        plt.show()

        # EQ of the signal under optimization
        dhasp.show_filterbank_responses('control', show_every=1)
        plt.show()
        dhasp.show_filterbank_joint_response('eq')
        plt.show()

    def test_dsp_functions(self):
        def test_fun(fn, x):
            inputs = {'np': x,
                      'torch': t.type_conversion.np2torch(x)}
            # %%

            out = dict()
            for var_type, var in inputs.items():
                print()
                print(f'{var_type}: ')
                out[var_type] = fn(var)
                display(out[var_type])

            return out

        # %%
        fs_model = 24e3

        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        audio_np = audio_samples['clean'].squeeze().numpy()

        # %&
        out_rms = test_fun(t.dsp.rms, audio_np)

        # %%
        out_db2mag = test_fun(t.dsp.db2mag, np.array([20, 6]))

        # %%
        out_mag2db = test_fun(t.dsp.mag2db, np.array([0.0001, 10]))

        # %%
        out_mel2hz = test_fun(t.dsp.mel2hz, np.array([10, 500]))

        # %%
        out_hz2mel = test_fun(t.dsp.hz2mel, np.array([10, 500]))

        # %%
        out_erb = test_fun(t.dsp.erb, np.array([10, 500]))

        # %%
        out_envelope = test_fun(t.dsp.envelope, audio_np[:100])

    def test_envelope(self):
        # %% Load Audio

        # Load variations of the same speech segment in different conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Get the first 3 seconds of the clean sample.
        waveform = audio_samples['clean'][: int(fs_model) * 3]

        # Make 2 versions, quiet and loud.
        waveforms = torch.vstack([
            waveform,
            2 * waveform
        ])

        # %% Calculate envelopes

        envelopes = t.dsp.envelope(waveforms)

        # %% Visualize

        # Create a time axis.
        time_to_show_s = 0.1
        samples_to_show = int(time_to_show_s * fs_model)
        time_axis = np.arange(samples_to_show) / fs_model

        # Plot filter output and its envelope.
        figure, axess = plt.subplots(2, 1, figsize=(12, 8))

        axess[0].plot(time_axis,
                      waveforms[0, :samples_to_show],
                      label='Waveform')

        axess[1].plot(time_axis,
                      waveforms[1, :samples_to_show],
                      label='Waveform')

        axess[0].plot(time_axis,
                      envelopes[0, :samples_to_show],
                      label='Envelope')

        axess[1].plot(time_axis,
                      envelopes[1, :samples_to_show],
                      label='Envelope')

        axess[0].set_title('Calculation of envelope for signals with 2 '
                           'different amplitudes'
                           '\n\nOriginal signal')
        axess[1].set_title('Original signal + 6 dB')

        for axes in axess.flatten():
            axes.legend()
            axes.set_xlabel('Time [s]')
            axes.set_ylabel('Signal value')
            axes.grid()

        t.plotting.apply_standard_formatting(figure)
        plt.show()

    def test_output_auditory_model(self):
        # %% Load Audio

        # Load variations of the same speech segment in different conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Get the first 3 seconds of the clean sample.
        waveform = audio_samples['clean'][: int(fs_model) * 3]

        # Make 2 versions, quiet and loud.
        waveforms = torch.vstack([
            waveform,
            10 * waveform
        ])

        # Assign names for plotting.
        waveform_names = ['0 dB', '+20 dB']

        # %% Initialize the dhasp model
        dhasp = t.dhasp.DHASP(fs_model)

        # %% Get the output of the auditory model
        output_auditory_model, envelope_auditory_model = \
            dhasp.calculate_output(waveforms)

        # %% Visualize

        # The index of the filter and number of samples to show
        idx_filter = 2
        n_samples = int(waveform.shape[-1])

        # Create the time axis for the plot
        time = np.arange(n_samples) / fs_model

        # Create figure and axes for the plots.
        figure, axess = plt.subplots(2, 1, figsize=(12, 8))

        for idx_waveform, (waveform_name, axes) \
                in enumerate(zip(waveform_names, axess.flatten())):

            # Initialize a list of curve handles.
            curves = list()

            # Plot filter output and its envelope.
            axes_right = axes.twinx()

            curves += axes.plot(
                time,
                waveforms[idx_waveform, :n_samples],
                label=f'Original waveform'
            )

            curves += axes.plot(
                time,
                output_auditory_model[idx_waveform, idx_filter, :n_samples],
                label=f'Output of the auditory model'
            )

            curves += axes_right.plot(
                time,
                envelope_auditory_model[idx_waveform, idx_filter, :n_samples],
                label='Envelope of the output [dB]',
                color='red'
            )
            axes.grid()
            if idx_waveform == 1:
                axes.legend(curves, [curve.get_label() for curve in curves],
                            bbox_to_anchor=(0, -0.15, 1, 0),
                            loc="upper left",
                            mode="expand",
                            ncol=len(curves),
                            frameon=False
                            )

            axes.set_title(
                f'Waveform: {waveform_name}, '
                f'Filter number: {idx_filter + 1}, '
                f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
            axes.set_xlabel('Time [s]')

        t.plotting.apply_standard_formatting(figure, include_grid=False)
        figure.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    def test_smoothing_output_envelope(self):
        # %% Load Audio

        # Load audio samples representing the same speech segment in different
        # acoustic conditions.
        fs_model = 24e3
        audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
            dict(
                speaker='joanna',
                length='5s',
                segment=10
            ),
            fs_model
        )

        # Get the first 3 seconds of the clean sample.
        waveform = audio_samples['clean'][: int(fs_model) * 3]

        # Make 2 versions, quiet and loud.
        waveforms = torch.vstack([
            waveform,
            10 * waveform
        ])

        # Assign names for plotting.
        waveform_names = ['0 dB', '+20 dB']

        # %% Initialize the dhasp model
        dhasp = t.dhasp.DHASP(fs_model)

        # %% Get the output of the auditory model
        output_model, envelope_model = dhasp.calculate_output(waveforms)

        # %% Smooth the output envelope
        envelope_smoothed, stride = dhasp.smooth_output_envelope(envelope_model)

        # %% Visualize

        # The index of the filter and number of samples to show
        idx_filter = 2

        # Create the time axis for the plot.
        time_original = np.arange(envelope_model.shape[-1]) / fs_model
        time_smoothed = np.arange(
            envelope_smoothed.shape[-1]) / fs_model * stride

        # Create figure and axes for the plots.
        figure, axess = plt.subplots(2, 1, figsize=(12, 8))

        for idx_waveform, (waveform_name, axes) \
                in enumerate(zip(waveform_names, axess.flatten())):

            # Create another y axis on the right.
            axes_right = axes.twinx()

            # Initialize a list of curve handles.
            curves = list()

            curves += axes.plot(
                time_original,
                envelope_model[idx_waveform, idx_filter, :],
                label=f'Original envelope of the output of the model [dB]'
            )
            curves += axes.plot(
                time_smoothed,
                envelope_smoothed[idx_waveform, idx_filter, :],
                label='Smoothed envelope [dB]',
                color='red'
            )

            axes.grid()
            if idx_waveform == 1:
                axes.legend(curves, [curve.get_label() for curve in curves],
                            bbox_to_anchor=(0, -0.15, 1, 0),
                            loc="upper left",
                            mode="expand",
                            ncol=len(curves),
                            frameon=False
                            )

            axes.set_title(
                f'Waveform: {waveform_name}, '
                f'Filter number: {idx_filter + 1}, '
                f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz'
            )

            axes.set_xlabel('Time [s]')

        t.plotting.apply_standard_formatting(figure, include_grid=False)
        figure.tight_layout(rect=[0, 0, 1, 1])
        plt.show()







