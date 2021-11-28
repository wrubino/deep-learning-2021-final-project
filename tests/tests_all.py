# %% Imports
from toolbox.initialization import *

# %% Test the filterbanks of the auditory model

# Show responses of the filters in the auditory model.
import tests.test_dhasp_filters


# %% Test: Apply filter

# Test that the application of the filters in the auditory model works.
import tests.test_apply_filter


# %% Test: Envelope.

# Test that the amplitude envelopes are generated correctly.
import tests.test_envelope


# %% Test: Dynamic range compression

# Test that the dynamic range compression gain is calculated correctly.
import tests.test_compression


# %% Test: Output of the auditory model

# Test that auditory model generates a meaningful output.
import tests.test_output_auditory_model

# %% Test: Smoothing of the output envelope

# Test that the smoothing of the envelope of the output of the auditory model
# functions correctly.
import tests.test_smoothing_output_envelope


# %% Test: Cepstral sequence

# Test that the cepstral sequencies are generated correctly.
import tests.test_cepstral_sequence

# %% Test: Correlation of cepstral sequences, energy loss, and total loss

# Test that that calculation of the cepstral correlations, energy loss, and the
# Total loss functions correcly.
import tests.test_cepstral_correlation_and_total_loss

