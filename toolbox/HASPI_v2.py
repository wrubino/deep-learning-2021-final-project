import numpy as np
from resampy import resample
from scipy.interpolate import interp1d
from scipy.signal import butter
from scipy.signal import cheby2
from scipy.signal import firwin2
from scipy.signal import convolve
from scipy.signal import convolve2d
from scipy.signal import group_delay
from scipy.signal import lfilter
from numpy.random import default_rng
from matplotlib.pyplot import xcorr

import warnings

# from smop.libsmop import *
from typing import Union, Sequence


def HASPI_v2(x: np.ndarray,
             fx,
             y: np.ndarray,
             fy,
             HL: np.ndarray,
             Level1=None):
    '''
    function [Intel,raw] = HASPI_v2(x,fx,y,fy,HL,Level1)
    % Function to compute the HASPI intelligibility index using the
    % auditory model followed by computing the envelope cepstral
    % correlation and BM vibration high-level covariance. The reference
    % signal presentation level for NH listeners is assumed to be 65 dB
    % SPL. The same model is used for both normal and impaired hearing. This
    % version of HASPI uses a modulation filterbank followed by an ensemble of
    % neural networks to compute the estimated intelligibility.
    %
    % Calling arguments:
    % x			Clear input reference speech signal with no noise or distortion.
    %           If a hearing loss is specified, no amplification should be provided.
    % fx        Sampling rate in Hz for signal x
    % y			Output signal with noise, distortion, HA gain, and/or processing.
    % fy        Sampling rate in Hz for signal y.
    % HL		(1,6) vector of hearing loss at the 6 audiometric frequencies
    %			  [250, 500, 1000, 2000, 4000, 6000] Hz.
    % Level1    Optional input specifying level in dB SPL that corresponds to a
    %           signal RMS = 1. Default is 65 dB SPL if argument not provided.
    %
    % Returned values:
    % Intel     Intelligibility estimated by passing the cepstral coefficients
    %           through a modulation filterbank followed by an ensemble of
    %           neural networks.
    % raw       vector of 10 cep corr modulation filterbank outputs, averaged
    %           over basis funct 2-6.
    %
    % James M. Kates, 5 August 2013.
    '''

    # Set the RMS reference level
    if Level1 is None:
        Level1 = 65

    # Auditory model for intelligibility
    # Reference is no processing, normal hearing
    itype = 0
    # .\HASPI_v2.m:37

    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel(
        x, fx, y, fy, HL, itype, Level1)
    # .\HASPI_v2.m:38

    # ---------------------------------------
    # Envelope modulation features
    # LP filter and subsample the envelopes
    fLP = 320
    # .\HASPI_v2.m:44

    fsub = np.dot(8, fLP)
    # .\HASPI_v2.m:45

    xLP, yLP = ebm_EnvFilt(xenv, yenv, fLP, fsub, fsamp)

    # .\HASPI_v2.m:46
    # Compute the cepstral coefficients as a function of subsampled time
    nbasis = 6
    # .\HASPI_v2.m:49

    thr = 2.5
    # .\HASPI_v2.m:50

    dither = 0.1
    # .\HASPI_v2.m:51

    xcep, ycep = ebm_CepCoef(xLP, yLP, thr, dither, nbasis)
    # .\HASPI_v2.m:52

    # Cepstral coefficients filtered at each modulation rate
    # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
    # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
    xmod, ymod, cfmod = ebm_ModFilt(xcep, ycep, fsub)
    # .\HASPI_v2.m:57

    # Cross-correlations between the cepstral coefficients for the degraded and
    # ref signals at each modulation rate, averaged over basis functions 2-6
    # aveCM:  cep corr modulation filterbank outputs, ave over basis funct 2-6
    aveCM = ebm_ModCorr(xmod, ymod)
    # .\HASPI_v2.m:62

    # ---------------------------------------
    # Intelligibility prediction
    # Get the neural network parameters and the weights for an ensemble of
    # 10 networks
    NNparam, Whid, Wout, b = ebm_GetNeuralNet
    # .\HASPI_v2.m:68

    # Average the neural network outputs for the modulation filterbank values
    model = NNfeedfwdEns(aveCM, NNparam, Whid, Wout)
    # .\HASPI_v2.m:71

    model = model / b
    # .\HASPI_v2.m:72

    # Return the intelligibility estimate and raw modulation filter outputs
    Intel = np.copy(model)
    # .\HASPI_v2.m:75

    raw = np.copy(aveCM)
    # .\HASPI_v2.m:76

    return Intel, raw


def NNfeedforward(data: np.ndarray,
                  NNparam: np.ndarray,
                  Whid: np.ndarray,
                  Wout: np.ndarray):
    '''
    # Function to compute the outputs at each layer of a neural network given
    # the input to the network and the weights. The activiation function is an
    # offset logistic function that gives either a logsig or hyperbolic
    # tangent; the outputs from each layer have been reduced by the offset. The
    # structure of the network is an input layer, one hidden layer, and an
    # output layer. The first values in vectors hidden and output are set to 1
    # by the function, and the remaining values correspond to the outputs at
    # each neuron in the layer.

    # Calling arguments:
    # data       feature vector input to the neural network
    # NNparam    network parameters from NNinit
    # Whid       matrix of weights for the hidden layer
    # Wout       matrix of weights for the output layer

    # Returned values:
    # hidden     vector of outputs from the hidden layer
    # output     vector of outputs from the output layer

    # James M. Kates, 26 October 2010.
    '''

    # Extract the parameters from the parameter array
    nx = NNparam[0]
    # .\NNfeedforward.m:24

    nhid = NNparam[1]
    # .\NNfeedforward.m:25

    nout = NNparam[2]
    # .\NNfeedforward.m:26

    beta = NNparam[3]
    # .\NNfeedforward.m:27

    offset = NNparam[4]
    # .\NNfeedforward.m:28

    again = NNparam[5]
    # .\NNfeedforward.m:29

    # Initialize the array storage
    x = np.zeros([nx + 1, 1])
    # .\NNfeedforward.m:32

    hidden = np.zeros([nhid + 1, 1])
    # .\NNfeedforward.m:33

    output = np.zeros([nout + 1, 1])
    # .\NNfeedforward.m:34

    # Initialize the nodes used for constants
    x[0] = 1
    # .\NNfeedforward.m:37

    hidden[0] = 1.0
    # .\NNfeedforward.m:38

    output[0] = 1.0
    # .\NNfeedforward.m:39

    # Input layer
    for i in np.arange(1, nx + 1, dtype=int):
        x[i] = data[i - 1]
    # .\NNfeedforward.m:43

    # Response of the hidden layer
    for j in np.arange(1, nhid + 1, dtype=int):
        sumhid = np.sum(np.dot(Whid[:, j - 1], x))
        hidden[j] = (again / (1.0 + np.exp(-beta * sumhid))) - offset

    # .\NNfeedforward.m:49

    # Response of the output layer
    for k in np.arange(1, nout + 1, dtype=int):
        sumout = np.sum(np.dot(Wout[:, k - 1], hidden))

        # .\NNfeedforward.m:54
        output[k] = (again / (1.0 + np.exp(- beta * sumout))) - offset
    # .\NNfeedforward.m:55

    return hidden, output


def NNfeedfwdEns(data: np.ndarray,
                 NNparam: np.ndarray,
                 Whid: Sequence,
                 Wout: Sequence):
    '''
    function model = NNfeedfwdEns(data,NNparam,Whid,Wout)
    % Function to compute the neural network ensemble response to a set of
    % inputs. The neural network is defined in NNfeedforwardZ.
    %
    % Calling arguments:
    % data      array of features input to the neural network
    % NNparam   vector of neural network paramters
    % Whid      cell array of hidden layer weights for each network
    % Wout      cell array of output layer weights for each network
    %
    % Returned value:
    % model     neural network output vector averaged over the ensemble
    %
    % James M. Kates, 20 September 2011.
    '''

    # Data and network parameters
    ncond = np.size(data, 0)  # Number of conditions in the input data

    K = len(Whid)  # Number of networks in the ensemble

    # Ensemble average of the predictions over the set of neural
    # networks used for training
    predict = np.zeros([ncond, K])

    for k in np.arange(K, dtype=int):
        for n in np.arange(ncond, dtype=int):
            d = data[n, :]
            _, output = NNfeedforward(d, NNparam, Whid[k], Wout[k])
            predict[n, k] = output[1]

    model = np.mean(predict, 1)

    return model


def eb_BMaddnoise(x: np.ndarray,
                  thr: Union[int, float],
                  Level1: Union[int, float]):
    '''
    % Function to apply the IHC attenuation to the BM motion and to add a
    function y=eb_BMaddnoise(x,thr,Level1)
    % low-level Gaussian noise to give the auditory threshold.
    %
    % Calling arguments:
    % x         BM motion to be attenuated
    % thr       additive noise level in dB re:auditory threshold
    % Level1    an input having RMS=1 corresponds to Leve1 dB SPL
    %
    % Returned values:
    % y         attenuated signal with threhsold noise added
    %
    % James M. Kates, 19 June 2012.
    % Just additive noise, 2 Oct 2012.
    '''

    # Additive noise
    # Linear gain for the noise
    gn = np.power(10, (thr - Level1) / 20.0)

    # Gaussian RMS = 1, then attenuated
    noise = gn * default_rng().standard_normal(len(x))
    y = x + noise

    return y


def eb_BWadjust(control,
                BWmin,
                BWmax,
                Level1):
    '''
    function BW=eb_BWadjust(control,BWmin,BWmax,Level1)
    % Function to compute the increase in auditory filter bandwidth in response
    % to high signal levels.
    %
    % Calling arguments:
    % control     envelope output in the control filter band
    % BWmin       auditory filter bandwidth computed for the loss (or NH)
    % BWmax       auditory filter bandwidth at maximum OHC damage
    % Level1      RMS=1 corresponds to Level1 dB SPL
    %
    % Returned value:
    % BW          filter bandwidth increased for high signal levels
    %
    % James M. Kates, 21 June 2011.
    '''

    # Compute the control signal level
    cRMS = np.sqrt(np.mean(np.power(control, 2)))  # RMS signal intensity

    cdB = 20 * np.log10(cRMS) + Level1  # Convert to dB SPL

    # Adjust the auditory filter bandwidth
    if cdB < 50:
        # No BW adjustment for the signal below 50 dB SPL
        BW = BWmin

    elif cdB > 100:
        # Maximum BW if signal above 100 dB SPL
        BW = BWmax

    else:
        # Linear interpolation between BW at 50 dB and max BW at 100 dB SPL
        BW = BWmin + ((cdB - 50) / 50) * (BWmax - BWmin)

    return BW


def eb_CenterFreq(nchan,
                  shift=None):
    '''
    function cf=eb_CenterFreq(nchan,shift)
    % Function to compute the ERB frequency spacing for the gammatone
    % filter bank. The equation comes from Malcolm Slaney (1993).
    %
    % Calling variables
    % nchan		number of filters in the filter bank
    % shift     optional frequency shift of the filter bank specified as a
    %           fractional shift in distance along the BM. A positive shift
    %           is an increase in frequency (basal shift), and negative is
    %           a decrease in frequency (apical shift). The total length of
    %           the BM is normalized to 1. The frequency-to-distance map is
    %           from D.D. Greenwood (1990), JASA 87, 2592-2605, Eq (1).
    %
    % James M. Kates, 25 January 2007.
    % Frequency shift added 22 August 2008.
    % Lower and upper frequencies fixed at 80 and 8000 Hz, 19 June 2012.
    '''

    # Parameters for the filter bank
    lowFreq = 80.0  # Lowest center frequency
    highFreq = 8000.0  # Highest center frequency

    # Moore and Glasberg ERB values
    EarQ = 9.26449
    minBW = 24.7

    # Frequency shift is an optional parameter
    if shift is not None:
        k = 1
        A = 165.4
        a = 2.1  # shift specified as a fraction of the total length

        #   Locations of the low and high frequencies on the BM between 0 and 1
        xLow = (1 / a) * np.log10(k + (lowFreq / A))
        xHigh = (1 / a) * np.log10(k + (highFreq / A))

        #   Shift the locations
        xLow = xLow * (1 + shift)
        xHigh = xHigh * (1 + shift)

        #   Compute the new frequency range
        lowFreq = A * (np.power(10, (a * xLow)) - k)
        highFreq = A * (np.power(10, (a * xHigh)) - k)

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank" by Malcolm Slaney.
    cf = -(EarQ * minBW) \
         + np.exp(
            np.arange(1, nchan) * (
                - np.log(highFreq + EarQ * minBW)
                + np.log(lowFreq + EarQ * minBW)
            ) / (nchan - 1)
    ) * (highFreq + EarQ * minBW)

    # Last center frequency is set to highFreq
    cf = np.hstack([highFreq, cf])
    cf = np.flip(cf)  # Reorder to put the low frequencies first

    return cf


def eb_EarModel(x: np.ndarray,
                xsamp,
                y: np.ndarray,
                ysamp,
                HL: np.ndarray,
                itype,
                Level1):
    '''
    function [xdB,xBM,ydB,yBM,xSL,ySL,fsamp]=...
        eb_EarModel(x,xsamp,y,ysamp,HL,itype,Level1)
    % Function to implement a cochlear model that includes the middle ear,
    % auditory filter bank, OHC dynamic-range compression, and IHC attenuation.
    % The inputs are the reference and processed signals that are to be
    % compared. The reference x is at the reference intensity (e.g. 65 dB SPL
    % or with NAL-R amplification) and has no other processing. The processed
    % signal y is the hearing-aid output, and is assumed to have the same or
    % greater group delay compared to the reference. The function outputs are
    % the envelopes of the signals after OHC compression and IHC loss
    % attenuation.
    %
    % Calling arguments:
    % x        reference signal: should be adjusted to 65 dB SPL (itype=0 or 1)
    %            or to 65 dB SPL plus NAL-R gain (itype=2)
    % xsamp    sampling rate for the reference signal, Hz
    % y        processed signal (e.g. hearing-aid output) includes HA gain
    % ysamp    sampling rate for the processed signal, Hz
    % HL       audiogram giving the hearing loss in dB at six audiometric
    %            frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
    % itype    purpose for the calculation:
    %          0=intelligibility: reference is nornal hearing and must not
    %            include NAL-R EQ
    %          1=quality: reference does not include NAL-R EQ
    %          2=quality: reference already has NAL-R EQ applied
    % Level1   level calibration: signal RMS=1 corresponds to Level1 dB SPL
    %
    % Returned values:
    % xdB      envelope for the reference in each band
    % xBM      BM motion for the reference in each band
    % ydB      envelope for the processed signal in each band
    % yBM      BM motion for the processed signal in each band
    % xSL      compressed RMS average reference in each band converted to dB SL
    % ySL      compressed RMS average output in each band converted to dB SL
    % fsamp    sampling rate in Hz for the model outputs
    %
    % James M. Kates, 27 October 2011.
    % BM motion added 30 Dec 2011.
    % Revised 19 June 2012.
    % Remove match of reference RMS level to processed 29 August 2012.
    % IHC adaptation added 1 October 2012.
    % BM envelope coverted to dB SL, 2 Oct 2012.
    % Filterbank group delay corrected, 14 Dec 2012.
    '''

    # Processing parameters
    # OHC and IHC parameters for the hearing loss
    # Auditory filter center frequencies span 80 to 8000 Hz.
    nchan = 32  # Use 32 auditory frequency bands
    mdelay = 1  # Compensate for the gammatone group delay
    cfreq = eb_CenterFreq(nchan)  # Center frequencies on an ERB scale

    # Cochlear model parameters for the processed signal
    attnOHCy, BWminy, lowkneey, CRy, attnIHCy = eb_LossParameters(HL, cfreq)

    # The cochlear model parameters for the reference are the same as for the
    # hearing loss if calculating quality, but are for normal hearing if
    # calculating intelligibility.
    if itype == 0:
        HLx = 0 * HL
    else:
        HLx = HL

    attnOHCx, BWminx, lowkneex, CRx, attnIHCx = eb_LossParameters(HLx, cfreq)

    # Parameters for the control filter bank
    HLmax = 100 * np.array([1, 1, 1, 1, 1, 1])
    shift = 0.02  # Basal shift of 0.02 of the basilar membrane length
    cfreq1 = eb_CenterFreq(nchan, shift)  # Center frequencies for the control
    _, BW1, _, _, _ = eb_LossParameters(HLmax,
                                        cfreq1)  # Maximum BW for the control

    # ---------------------------------------
    # Input signal adjustments
    # Force the input signals to be row vectors
    x = x.flatten('F')
    y = y.flatten('F')

    # Convert the signals to 24 kHz sampling rate. Using 24 kHz guarantees that
    # all of the cochlear filters have the same shape independent of the
    # incoming signal sampling rates
    x24, _ = eb_Resamp24kHz(x, xsamp)
    y24, fsamp = eb_Resamp24kHz(y, ysamp)

    # Check the file sizes
    nxy = np.min([len(x24), len(y24)])
    x24 = x24[0: nxy]
    y24 = y24[0: nxy]

    # Bulk broadband signal alignment
    x24, y24 = eb_InputAlign(x24, y24)
    nsamp = len(x24)

    # Add NAL-R equalization if the quality reference doesn't already have it.
    if itype == 1:
        nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
        nalr, _ = eb_NALR(HL, nfir, fsamp)  # Design the NAL-R filter
        x24 = np.convolve(x24, nalr)  # Apply the NAL-R filter
        x24 = x24[nfir: nfir + nsamp]

    # ---------------------------------------
    # Cochlear model
    # Middle ear
    xmid = eb_MiddleEar(x24, fsamp).T
    ymid = eb_MiddleEar(y24, fsamp).T

    # Initialize storage
    # Reference and processed envelopes and BM motion
    xdB = np.zeros([nchan, nsamp])
    ydB = xdB
    xBM = xdB
    yBM = ydB

    # Reference and processed average spectral values
    xave = np.zeros([nchan, 1])  # Reference
    yave = xave  # Processed
    xcave = xave  # Reference control
    ycave = yave  # Processed control

    # Filter bandwidths adjusted for intensity
    BWx = np.zeros([nchan, 1])
    BWy = BWx

    # Loop over each filter in the auditory filter bank
    for n in np.arange(nchan, dtype=int):
        # Control signal envelopes for the reference and processed signals
        xcontrol, ycontrol = eb_GammatoneEnv2(
            xmid, BW1[n], ymid, BW1[n], fsamp, cfreq1[n])

        # Adjust the auditory filter bandwidths for the average signal level.
        BWx[n] = eb_BWadjust(xcontrol, BWminx[n], BW1[n], Level1)  # Reference
        BWy[n] = eb_BWadjust(ycontrol, BWminy[n], BW1[n], Level1)  # Processed

        #   Envelopes and BM motion of the reference and processed signals
        xenv, xbm, yenv, ybm = eb_GammatoneBM2(
            xmid, BWx[n], ymid, BWy[n], fsamp, cfreq[n])

        #   RMS levels of the ref and output envelopes for linear metric
        xave[n] = np.sqrt(
            np.mean0(np.power(xenv, 2)))  # Ave signal mag in each band
        yave[n] = np.sqrt(np.mean(np.power(yenv, 2)))
        xcave[n] = np.sqrt(np.mean(np.power(xcontrol, 2)))  # Ave control signal
        ycave[n] = np.sqrt(np.mean(np.power(ycontrol, 2)))

        # Cochlear compression for the signal envelopes and BM motion
        xc, xb = eb_EnvCompressBM(
            xenv, xbm, xcontrol, attnOHCx[n], lowkneex[n], CRx[n],
            fsamp, Level1
        )

        yc, yb = eb_EnvCompressBM(
            yenv, ybm, ycontrol, attnOHCy[n], lowkneey[n], CRy[n],
            fsamp, Level1
        )

        #   Correct for the delay between the reference and output
        yc = eb_EnvAlign(xc, yc)  # Align processed envelope to reference
        yb = eb_EnvAlign(xb, yb)  # Align processed BM motion to reference

        #   Convert the compressed envelopes and BM vibration envelopes to dB SL
        xc, xb = eb_EnvSL2(xc, xb, attnIHCx[n], Level1)
        yc, yb = eb_EnvSL2(yc, yb, attnIHCy[n], Level1)

        #   Apply the IHC rapid and short-term adaptation
        delta = 2.0
        # Amount of overshoot
        xdB[n, :], xb = eb_IHCadapt(xc, xb, delta, fsamp)
        ydB[n, :], yb = eb_IHCadapt(yc, yb, delta, fsamp)

        #   Additive noise level to give the auditory threshold
        IHCthr = -10.0  # Additive noise level, dB re: auditory threshold
        xBM[n, :] = eb_BMaddnoise(xb, IHCthr, Level1)
        yBM[n, :] = eb_BMaddnoise(yb, IHCthr, Level1)

    # Correct for the gammatone filterbank interchannel group delay.
    # Function eb_EnvAlign matches the processed signal delay to the reference,
    # so the filterbank delay correction should be for the reference.
    if mdelay > 0:
        xdB = eb_GroupDelayComp(xdB, BWx, cfreq, fsamp)
        ydB = eb_GroupDelayComp(ydB, BWx, cfreq, fsamp)
        xBM = eb_GroupDelayComp(xBM, BWx, cfreq, fsamp)
        yBM = eb_GroupDelayComp(yBM, BWx, cfreq, fsamp)

    # Convert average gammatone outputs to dB SL
    xSL = eb_aveSL(xave, xcave, attnOHCx, lowkneex, CRx, attnIHCx, Level1)
    ySL = eb_aveSL(yave, ycave, attnOHCy, lowkneey, CRy, attnIHCy, Level1)

    return xdB, xBM, ydB, yBM, xSL, ySL, fsamp


def eb_EnvAlign(x, y):
    '''
    function y = eb_EnvAlign(x,y)
    % Function to align the envelope of the processed signal to that of the
    % reference signal.
    %
    % Calling arguments:
    % x      envelope or BM motion of the reference signal
    % y      envelope or BM motion of the output signal
    %
    % Returned values:
    % y      shifted output envelope to match the input
    %
    % James M. Kates, 28 October 2011.
    % Absolute value of the cross-correlation peak removed, 22 June 2012.
    % Cross-correlation range reduced, 13 August 2013.
    '''

    # Correlation parameters
    # Reduce the range of the xcorr calculation to save computation time
    fsamp = 24000  # Sampling rate in Hz
    range = 100  # Range in msec for the xcorr calculation
    lags = np.round(0.001 * range * fsamp)  # Range in samples
    npts = len(x)
    lags = np.min([lags, npts])  # Use min of lags, length of the sequence

    # Cross-correlate the two sequences over the lag range
    lags, c = xcorr(x, y, maxlags=lags - 1)[0: 2]
    delay = -lags[np.argmax(c)]

    # Time shift the output sequence
    if delay > 0:
        #   Output delayed relative to the reference
        y = np.array([y[delay: npts], np.zeros([delay, 1])])  # Remove the delay
    elif delay < 0:
        #   Output advanced relative to the reference
        delay = -delay
        y = np.array(
            [np.zeros([delay, 1]), y[0: (npts - delay)]])  # Add advance

    return y


def eb_EnvCompressBM(envsig: np.ndarray,
                     bm: np.ndarray,
                     control,
                     attnOHC,
                     thrLow,
                     CR,
                     fsamp,
                     Level1):
    '''
    function[y, b] = eb_EnvCompressBM(envsig, bm, control, attnOHC, thrLow, CR,
                                      fsamp, Level1)
    # Function to compute the cochlear compression in one auditory filter
    # band. The gain is linear below the lower threshold, compressive with
    # a compression ratio of CR:1 between the lower and upper thresholds,
    # and reverts to linear above the upper threshold. The compressor
    # assumes that auditory thresold is 0 dB SPL.
    #
    #Calling variables:
    #envsig	analytic signal envelope (magnitude) returned by the
    #			gammatone filter bank
    #bm        BM motion output by the filter bank
    #control	analytic control envelope returned by the wide control
    #			path filter bank
    #attnOHC	OHC attenuation at the input to the compressor
    #thrLow	kneepoint for the low-level linear amplification
    #CR		compression ratio
    #fsamp		sampling rate in Hz
    #Level1	dB reference level: a signal having an RMS value of 1 is
    #			assigned to Level1 dB SPL.
    #
    #Function outputs:
    #y			compressed version of the signal envelope
    #b         compressed version of the BM motion
    #
    #James M. Kates, 19 January 2007.
    #James M. Kates, 19 January 2007.
    #LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
    #Version to compress the envelope, 20 Feb 2007.
    #Change in the OHC I/O function, 9 March 2007.
    #Two-tone suppression added 22 August 2008.

    '''

    # Initialize the compression parameters
    thrHigh = 100.0  # Upper compression threshold

    # Convert the control envelope to dB SPL
    small = 1.0e-30
    logenv = np.max(
        [control, small])  # Don't want to take logarithm of zero or neg
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.min(
        [logenv, thrHigh])  # Clip signal levels above the upper threshold
    logenv = np.max([logenv, thrLow])  # Clip signal at the lower threshold

    # Compute the compression gain in dB
    gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))

    # Convert the gain to linear and apply a LP filter to give a 0.2 msec delay
    gain = np.power(10, (gain / 20))
    flp = 800
    b, a = butter(1, flp / (0.5 * fsamp))
    gain = lfilter(b, a, gain)

    # Apply the gain to the signals
    y = gain * envsig  # Linear envelope
    b = gain * bm  # BM motion

    return y, b


def eb_EnvSL2(env: np.ndarray,
              bm: np.ndarray,
              attnIHC,
              Level1):
    '''
    function[y, b] = eb_EnvSL2(env, bm, attnIHC, Level1)
    # Function to convert the compressed envelope returned by
    # cochlea_envcomp to dB SL.
    #
    # Calling arguments
    # env			linear envelope after compression
    # bm            linear basilar membrane vibration after compression
    # attnIHC		IHC attenuation at the input to the synapse
    # Level1		level in dB SPL corresponding to 1 RMS
    #
    # Return
    # y				envelope in dB SL
    # b             BM vibration with envelope converted to dB SL
    #
    # James M. Kates, 20 Feb 07.
    # IHC attenuation added 9 March 2007.
    # Basilar membrane vibration conversion added 2 October 2012.
    '''

    # Convert the envelope to dB SL
    small = 1.0e-30  # To prevent taking log of 0
    y = Level1 - attnIHC + 20 * np.log10(env + small)
    y = np.max([y, 0.0])

    # Convert the linear BM motion to have a dB SL envelope
    gain = (y + small) / (
            env + small)  # Gain that converted the env from lin to dB SPL
    b = gain * bm  # Apply gain to BM motion

    return y, b


def eb_GammatoneBM2(x: np.ndarray,
                    BWx,
                    y: np.ndarray,
                    BWy,
                    fs,
                    cf):
    '''
    function[envx, BMx, envy, BMy] = eb_GammatoneBM2(x, BWx, y, BWy, fs, cf)
    # 4th-order gammatone auditory filter. This implementation is based
    # on the c program published on-line by Ning Ma, U. Sheffield, UK,
    # that gives an implementation of the Martin Cooke (1991) filters:
    # an impulse-invariant transformation of the gammatone filter. The
    # signal is demodulated down to baseband using a complex exponential,
    # and then passed through a cascade of four one-pole low-pass filters.
    #
    # This version filters two signals that have the same sampling rate and the
    # same gammatone filter center frequencies. The lengths of the two signals
    # should match; if they don't, the signals are truncated to the shorter of
    # the two lengths.
    #
    # Calling variables:
    # x			first sequence to be filtered
    # BWx	    bandwidth for x relative to that of a normal ear
    # y			second sequence to be filtered
    # BWy	    bandwidth for x relative to that of a normal ear
    # fs		sampling rate in Hz
    # cf		filter center frequency in Hz
    #
    # Returned values:
    # envx      filter envelope output (modulated down to baseband) 1st signal
    # BMx       BM motion for the first signal
    # envy      filter envelope output (modulated down to baseband) 2nd signal
    # BMy       BM motion for the second signal

    # James M. Kates, 8 Jan 2007.
    # Vectorized version for efficient MATLAB execution, 4 February 2007.
    # Cosine and sine generation, 29 June 2011.
    # Cosine/sine loop speed increased, 9 August 2013.
    '''

    # Filter ERB from Moore and Glasberg (1983)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)

    # Check the lengths of the two signals
    nx = len(x)
    ny = len(y)
    nsamp = np.min([nx, ny])
    x = x[:nsamp]
    y = y[:nsamp]

    # ---------------------------------------
    # Filter the first signal
    # Initialize the filter coefficients
    tpt = 2 * np.pi / fs
    tptBW = BWx * tpt * ERB * 1.019;
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Initialize the complex demodulation
    npts = len(x)
    cn = np.cos(tpt * cf)
    sn = np.sin(tpt * cf)
    coscf = np.zeros([npts, 1])
    sincf = coscf
    cold = 1
    sold = 0
    coscf[0] = cold
    sincf[0] = sold
    for n in np.arange(1, npts, dtype=int):
        arg = cold * cn + sold * sn
        sold = sold * cn - cold * sn
        cold = arg
        coscf[n] = cold
        sincf[n] = sold

    # Filter the real and imaginary parts of the signal using previous sines
    ureal = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    x * coscf)

    uimag = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    x * sincf)

    # Extract the BM velocity and the envelope
    BMx = gain * (ureal * coscf + uimag * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag)

    # ---------------------------------------
    # Filter the second signal using the existing cosine and sine sequences
    # Initialize the filter coefficients
    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Filter the real and imaginary parts of the signal
    ureal = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    y * coscf)
    uimag = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    y * sincf)

    # Extract the BM velocity and the envelope
    BMy = gain * (ureal * coscf + uimag * sincf)
    envy = gain * np.sqrt(ureal * ureal + uimag * uimag)

    return envx, BMx, envy, BMy


def eb_GammatoneEnv2(x: np.ndarray,
                     BWx,
                     y: np.ndarray,
                     BWy,
                     fs,
                     cf):
    '''
    function[envx, envy] = eb_GammatoneEnv2(x, BWx, y, BWy, fs, cf)
    # 4th-order gammatone auditory filter. This implementation is based
    # on the c program published on-line by Ning Ma, U. Sheffield, UK,
    # that gives an implementation of the Martin Cooke (1991) filters:
    # an impulse-invariant transformation of the gammatone filter. The
    # signal is demodulated down to baseband using a complex exponential,
    # and then passed through a cascade of four one-pole low-pass filters.
    #
    # This version filters two signals that have the same sampling rate and the
    # same gammatone filter center frequencies. The lengths of the two signals
    # should match; if they don't, the signals are truncated to the shorter of
    # the two lengths.
    #
    # Calling variables:
    # x			first sequence to be filtered
    # BWx	    bandwidth for x relative to that of a normal ear
    # y			second sequence to be filtered
    # BWy	    bandwidth for x relative to that of a normal ear
    # fs		sampling rate in Hz
    # cf		filter center frequency in Hz
    #
    # Returned values:
    # envx      filter envelope output (modulated down to baseband) 1st signal
    # envy      filter envelope output (modulated down to baseband) 2nd signal

    # James M. Kates, 8 Jan 2007.
    # Vectorized version for efficient MATLAB execution, 4 February 2007.
    # Cosine and sine generation, 29 June 2011.
    # Output sine and cosine sequences, 19 June 2012.
    # Cosine/sine loop speed increased, 9 August 2013.
    '''

    # Filter ERB from Moore and Glasberg (1983)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)

    # Check the lengths of the two signals
    nx = len(x)
    ny = len(y)
    nsamp = min([nx, ny])
    x = x[:nsamp]
    y = y[:nsamp]

    # ---------------------------------------
    # Filter the first signal
    # Initialize the filter coefficients
    tpt = 2 * np.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Initialize the complex demodulation
    npts = len(x)
    cn = np.cos(tpt * cf)
    sn = np.sin(tpt * cf)
    coscf = np.zeros([npts, 1])
    sincf = coscf
    cold = 1
    sold = 0
    coscf[1] = cold
    sincf[1] = sold
    for n in np.arange(1, npts, dtype=int):
        arg = cold * cn + sold * sn
        sold = sold * cn - cold * sn
        cold = arg
        coscf[n] = cold
        sincf[n] = sold

    # Filter the real and imaginary parts of the signal
    ureal = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    x * coscf)
    uimag = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    x * sincf)

    # Extract the BM velocity and the envelope
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag)

    # ---------------------------------------
    # Filter the second signal using the existing cosine and sine sequences
    # Initialize the filter coefficients
    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Filter the real and imaginary parts of the signal
    ureal = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    y * coscf)
    uimag = lfilter(np.array([1, a1, a5]),
                    np.array([1 - a1 - a2 - a3 - a4]),
                    y * sincf)

    # Extract the BM velocity and the envelope
    envy = gain * np.sqrt(ureal * ureal + uimag * uimag);

    return envx, envy


def eb_GroupDelayComp(xenv: np.ndarray,
                      BW: np.ndarray,
                      cfreq,
                      fsamp):
    '''
    function yenv = eb_GroupDelayComp(xenv, BW, cfreq, fsamp)
    # Function to compensate for the group delay of the gammatone filter bank.
    # The group delay is computed for each filter at its center frequency. The
    # firing rate output of the IHC model is then adjusted so that all outputs
    # have the same group delay.
    #
    # Calling variables:
    # xenv     matrix of signal envelopes or BM motion
    # BW       gammatone filter bandwidths adjusted for loss
    # cfreq    center frequencies of the bands
    # fsamp    sampling rate for the input signal in Hz (e.g. 24,000 Hz)
    #
    # Returned values:
    # yenv    envelopes or BM motion compensated for the group delay
    #
    # James M. Kates, 28 October 2011.
    '''

    # Processing parameters
    nchan = len(BW)

    # Filter ERB from Moore and Glasberg (1983)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cfreq / earQ)

    # Initialize the gamatone filter coefficients
    tpt = 2 * np.pi / fsamp
    tptBW = tpt * 1.019 * BW * ERB
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a

    # Compute the group delay in samples at fsamp for each filter
    gd = np.zeros([nchan, 1])
    for n in range(nchan):
        _, gd[n] = group_delay((np.array([1, a1[n], a5[n]]),
                                np.array([1 - a1[n] - a2[n] - a3[n] - a4[n]])),
                               1)

    gd = np.round(gd)  # Convert to integer samples

    # Compute the delay correction
    gmin = np.min(gd)
    gd = gd - gmin  # Remove the minimum delay from all of the values
    gmax = np.max(gd)  # Maximum of the adjusted delay
    correct = gmax - gd  # Samples delay needed to add to give alignment

    # Add the delay correction to each frequency band
    yenv = np.zeros(xenv.shape)  # Allocate space for the output
    for n in np.arange(nchan, dtype=int):
        r = xenv[n, :]  # Extract the IHC firing rate
        npts = len[r]
        r = np.hstack([np.zeros([1, correct[n]]),
                       r[:(npts - correct[n])]])
        yenv[n, :] = r

    return yenv


def eb_IHCadapt(xdB,
                xBM: np.ndarray,
                delta,
                fsamp):
    '''
    function[ydB, yBM] = eb_IHCadapt(xdB, xBM, delta, fsamp)
    # Function to provide inner hair cell (IHC) adaptation. The adaptation is
    # based on an equivalent RC circuit model, and the derivatives are mapped
    # into 1st-order backward differences. Rapid and short-term adaptation are
    # provided. The input is the signal envelope in dB SL, with IHC attenuation
    # already applied to the envelope. The outputs are the envelope in dB SL
    # with adaptation providing overshoot of the long-term output level, and
    # the BM motion is multiplied by a gain vs. time function that reproduces
    # the adaptation. IHC attenuation and additive noise for the equivalent
    # auditory threshold are provided by a subsequent call to eb_BMatten.
    #
    # Calling variables:
    # xdB      signal envelope in one frequency band in dB SL
    #          contains OHC compression and IHC attenuation
    # xBM      basilar membrane vibration with OHC compression but no IHC atten
    # delta    overshoot factor = delta x steady-state
    # fsamp    sampling rate in Hz
    #
    # Returned values:
    # ydB      envelope in dB SL with IHC adaptation
    # yBM      BM motion multiplied by the IHC adaptation gain function
    #
    # James M. Kates, 1 October 2012.
    '''

    # Test the amount of overshoot
    dsmall = 1.0001
    if delta < dsmall:
        delta = dsmall

    # Initialize the adaptation time constants
    tau1 = 2  # Rapid adaptation in msec
    tau2 = 60  # Short-term adaptation in msec
    tau1 = 0.001 * tau1
    tau2 = 0.001 * tau2  # Convert to seconds

    # Equivalent circuit parameters
    T = 1 / fsamp  # Sampling period
    R1 = 1 / delta
    R2 = 0.5 * (1 - R1)
    R3 = R2
    C1 = tau1 * (R1 + R2) / (R1 * R2)
    C2 = tau2 / ((R1 + R2) * R3)

    # Intermediate values used for the voltage update matrix inversion
    a11 = R1 + R2 + R1 * R2 * (C1 / T)
    a12 = -R1
    a21 = -R3
    a22 = R2 + R3 + R2 * R3 * (C2 / T)
    denom = 1.0 / (a11 * a22 - a21 * a12)

    # Additional intermediate values
    R1inv = 1.0 / R1
    R12C1 = R1 * R2 * (C1 / T)
    R23C2 = R2 * R3 * (C2 / T)

    # Initialize the outputs and state of the equivalent circuit
    nsamp = len(xdB)
    gain = np.ones(
        xdB.shape)  # Gain vector to apply to the BM motion, default is 1
    ydB = np.zeros(xdB.shape)  # Assign storage
    V1 = 0.0
    V2 = 0.0
    small = 1.0e-30

    # Loop to process the envelope signal
    # The gain asymptote is 1 for an input envelope of 0 dB SL
    for n in np.arange(nsamp, dtype=int):
        V0 = xdB[n]
        b1 = V0 * R2 + R12C1 * V1
        b2 = R23C2 * V2
        V1 = denom * (a22 * b1 - a12 * b2)
        V2 = denom * (-a21 * b1 + a11 * b2)
        out = (V0 - V1) * R1inv
        out = np.max([out, 0.0])  # Envelope can not drop below threshold
        ydB[n] = out  # Envelope with IHC adaptation
        gain[n] = (out + small) / (V0 + small)  # Avoid division by zero

    # Apply the gain to the BM vibration
    yBM = gain * xBM

    return ydB, yBM


def eb_InputAlign(x: np.ndarray,
                  y: np.ndarray):
    '''
    function[xp, yp] = eb_InputAlign(x, y)
    # Function to provide approximate temporal alignment of the reference and
    # processed output signals. Leading and trailing zeros are then pruned.
    # The function assumes that the two sequences have the same sampling rate:
    # call eb_Resamp24kHz for each sequence first, then call this function to
    # align the signals.
    #
    # Calling variables:
    # x       input reference sequence
    # y       hearing-aid output sequence
    #
    # Returned values:
    # xp   pruned and shifted reference
    # yp   pruned and shifted hearing-aid output
    #
    # James M. Kates, 12 July 2011.
    '''

    # Match the length of the processed output to the reference for the
    # purposes of computing the cross-covariance
    nx = len(x)
    ny = len(y)
    nsamp = np.min([nx, ny])

    # Determine the delay of the output relative to the reference
    lags, c = xcorr(x[:nsamp],
                    y[:nsamp],
                    maxlags=int(np.min([24000 * 0.5,
                                        len(y) * 0.7])))[
              0:2]  # Cross-covariance of the ref and output

    delay = -lags[np.argmax(c)]
    print(delay / 24000)

    # Back up 2 msec to allow for dispersion
    fsamp = 24000  # Cochlear model input sampling rate in Hz
    delay = int(delay - 2 * fsamp / 1000)  # Back up 2 msec

    # Align the output with the reference allowing for the dispersion
    if delay > 0:
        #   Output delayed relative to the reference
        print('positive')
        y = np.hstack([y[delay: ny], np.zeros(delay)])  # Remove the delay
    else:
        print('negative')
        #   Output advanced relative to the reference
        y = np.hstack([np.zeros(-delay), y[:(ny - delay)]])  # Add advance

    # Find the start and end of the noiseless reference sequence
    xabs = np.abs(x)
    xmax = np.max(xabs)
    xthr = 0.001 * xmax  # Zero detection threshold

    for n in np.arange(nx, dtype=int):
        # First value above the threshold working forwards from the beginning
        if xabs[n] > xthr:
            nx0 = n
            break

    for n in np.arange(nx - 1, -1, -1, dtype=int):
        # First value above the threshold working backwards from the end
        if xabs[n] > xthr:
            nx1 = n
            break

    # Prune the sequences to remove the leading and trailing zeros
    if nx1 > ny:
        nx1 = ny

    xp = x[nx0: nx1]
    yp = y[nx0: nx1]

    return xp, yp


def eb_LossParameters(HL: np.ndarray,
                      cfreq: np.ndarray):
    '''
    function [attnOHC,BW,lowknee,CR,attnIHC]=eb_LossParameters(HL,cfreq)
    # Function to apportion the hearing loss to the outer hair cells (OHC)
    # and the inner hair cells (IHC) and to increase the bandwidth of the
    # cochlear filters in proportion to the OHC fraction of the total loss.
    #
    # Calling variables:
    # HL		hearing loss at the 6 audiometric frequencies
    # cfreq		array containing the center frequencies of the gammatone filters
    #			arranged from low to high
    #
    # Returned values:
    # attnOHC	attenuation in dB for the OHC gammatone filters
    # BW		OHC filter bandwidth expressed in terms of normal
    # lowknee	Lower kneepoint for the low-level linear amplification
    # CR		Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for normal
    #			hearing. Reduced in proportion to the OHC loss to 1:1.
    # attnIHC	attenuation in dB for the input to the IHC synapse
    #
    # James M. Kates, 25 January 2007.
    # Version for loss in dB and match of OHC loss to CR, 9 March 2007.
    # Low-frequency extent changed to 80 Hz, 27 Oct 2011.
    # Lower kneepoint set to 30 dB, 19 June 2012.
    '''

    # Audiometric frequencies in Hz
    aud = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Interpolation to give the loss at the gammatone center frequencies
    # Use linear interpolation in dB. The interpolation assumes that
    # cfreq(1) < aud(1) and cfreq(nfilt) > aud(6)
    nfilt = len(cfreq)  # Number of filters in the filter bank
    fv = np.hstack([cfreq[0], aud,
                    cfreq[nfilt - 1]])  # Frequency vector for the interpolation
    x = np.arange(0, 10)
    y = np.exp(-x / 3.0)

    loss = interp1d(fv, np.hstack([HL[0], HL, HL[5]]))(
        cfreq)  # Interpolated gain in dB
    loss[loss < 0] = 0  # Make sure there are no negative losses

    # Compression ratio changes linearly with ERB rate from 1.25:1 in
    # the 80-Hz frequency band to 3.5:1 in the 8-kHz frequency band
    CR = np.zeros([nfilt, 1])
    for n in range(nfilt):
        CR[n] = 1.25 + 2.25 * (n - 1) / (nfilt - 1)

    # Maximum OHC sensitivity loss depends on the compression ratio.
    # The compression I/O curves assume linear below 30 and above 100
    # dB SPL in normal ears.
    maxOHC = 70 * (1 - (1 / CR))  # OHC loss that results in 1:1 compression
    thrOHC = 1.25 * maxOHC  # Loss threshold for adjusting the OHC parameters

    # Apportion the loss in dB to the outer and inner hair cells based on
    # the data of Moore et al (1999), JASA 106, 2761-2778. Reduce the CR
    # towards 1:1 in proportion to the OHC loss.
    attnOHC = np.zeros([nfilt, 1])  # Default is 0 dB attenuation
    attnIHC = np.zeros([nfilt, 1])

    for n in range(nfilt):
        if loss[n] < thrOHC[n]:
            attnOHC[n] = 0.8 * loss[n]
            attnIHC[n] = 0.2 * loss[n]
        else:
            attnOHC[n] = 0.8 * thrOHC[n]  # Maximum OHC attenuation
            attnIHC[n] = 0.2 * thrOHC[n] + (loss[n] - thrOHC[n])

    # Adjust the OHC bandwidth in proportion to the OHC loss
    BW = np.ones([nfilt, 1])  # Default is normal hearing gammatone bandwidth
    BW = BW + (attnOHC / 50.0) + 2.0 * np.power((attnOHC / 50.0), 6)

    # Compute the compression lower kneepoint and compression ratio
    lowknee = attnOHC + 30  # Lower kneepoint
    upamp = 30 + 70. / CR  # Output level for an input of 100 dB SPL
    CR = (100 - lowknee) / (
            upamp + attnOHC - lowknee)  # OHC loss Compression ratio

    return attnOHC, BW, lowknee, CR, attnIHC


def eb_MiddleEar(x: np.ndarray,
                 fsamp):
    '''
    function xout=eb_MiddleEar(x, fsamp)
    # Function to design the middle ear filters and process the input
    # through the cascade of filters. The middle ear model is a 2-pole HP
    # filter at 350 Hz in series with a 1-pole LP filter at 5000 Hz. The
    # result is a rough approximation to the equal-loudness contour at
    # threshold.
    #
    # Calling variables:
    #	x		input signal
    #	fsamp	sampling rate in Hz
    #
    # Function output:
    #	xout	filtered output
    #
    # James M. Kates, 18 January 2007.
    '''

    # Design the 1-pole Butterworth LP using the bilinear transformation
    bLP, aLP = butter(1, 5000 / (0.5 * fsamp))  # 5000-Hz LP

    # LP filter the input
    y = lfilter(bLP, aLP, x)

    # Design the 2-pole Butterworth HP using the bilinear transformation
    bHP, aHP = butter(2, 350 / (0.5 * fsamp), 'high')  # 350-Hz HP

    # HP filter the signal
    xout = lfilter(bHP, aHP, y);

    return xout


def eb_NALR(HL: np.ndarray,
            nfir,
            fsamp):
    '''
    function[nalr, delay] = eb_NALR(HL, nfir, fsamp)
    # Function to design an FIR NAL-R equalization filter and a flat filter
    # having the same linear-phase time delay.
    #
    # Calling variables:
    # HL        Hearing loss at the audiometric frequencies
    # nfir		Order of the NAL-R EQ filter and the matching delay
    # fsamp     sampling rate in Hz
    #
    # Returned arrays:
    # nalr		linear-phase filter giving the NAL-R gain function
    # delay		pure delay equal to that of the NAL-R filter
    #
    # James M. Kates, 27 December 2006.
    # Version for noise estimation system, 27 Oct 2011.
    '''

    # Processing parameters
    fmax = 0.5 * fsamp  # Nyquist frequency

    # Audiometric frequencies
    aud = [250, 500, 1000, 2000, 4000, 6000]  # Audiometric frequencies in Hz

    # Design a flat filter having the same delay as the NAL-R filter
    delay = np.zeros([1, nfir + 1])
    delay[int(nfir / 2)] = 1.0

    # Design the NAL-R filter for HI listener
    mloss = np.max(HL)  # Test for hearing loss
    if mloss > 0:
        #	Compute the NAL-R frequency response at the audiometric frequencies
        bias = np.array([-17, -8, 1, -1, -2, -2])
        t3 = HL[1] + HL[2] + HL[3];  # Three-frequency average loss
        if t3 <= 180:
            xave = 0.05 * t3
        else:
            xave = 9.0 + 0.116 * (t3 - 180)

        gdB = xave + 0.31 * HL + bias
        gdB = np.max([gdB, 0])  # Remove negative gains

        #	Design the linear-phase FIR filter
        fv = [0, aud, fmax];  # Frequency vector for the interpolation
        cfreq = np.arange(
            nfir + 1) / nfir;  # Uniform frequency spacing from 0 to 1
        gain = interp1d(fv, np.array([gdB[0], gdB, gdB[5]]))(
            fmax * cfreq)  # Interpolated gain in dB
        glin = np.power(10, gain / 20.0)  # Convert gain from dB to linear
        nalr = firwin2(nfir, cfreq, glin)  # Design the filter (length = nfir+1)
    else:
        #	Filters for the normal-hearing subject
        nalr = delay

    return nalr, delay


def eb_Resamp24kHz(x: np.ndarray,
                   fsampx):
    '''
    function[y, fsamp] = eb_Resamp24kHz(x, fsampx)
    # Function to resample the input signal at 24 kHz. The input sampling rate
    # is rounded to the nearest kHz to comput the sampling rate conversion
    # ratio.
    #
    # Calling variables:
    # x         input signal
    # fsampx    sampling rate for the input in Hz
    #
    # Returned argument:
    # y         signal resampled at 24 kHz
    # fsamp     output sampling rate in Kz
    #
    # James M. Kates, 20 June 2011.
    '''

    # Sampling rate information
    fsamp = 24000  # Output sampling rate in Hz
    fy = np.round(fsamp / 1000)  # Ouptut rate in kHz
    fx = np.round(fsampx / 1000)  # Input rate to nearest kHz

    # Resample the signal
    if fx == fy:
        #   No resampling performed if the rates match
        y = x

    elif fx < fy:
        #   Resample for the input rate lower than the output
        y = resample(x, fy, fx)

        #   Match the RMS level of the resampled signal to that of the input
        xRMS = np.sqrt(np.mean(np.power(x, 2)))
        yRMS = np.sqrt(np.mean(np.power(y, 2)))
        y = (xRMS / yRMS) * y

    else:
        #   Resample for the input rate higher than the output
        #   Resampling includes an anti-aliasing filter.
        y = resample(x, fy, fx)

        #   Reduce the input signal bandwidth to 21 kHz:
        #   Chebychev Type 2 LP (smooth passband)
        order = 7  # Filter order
        atten = 30  # Sidelobe attenuation in dB
        fcutx = 21 / fx  # Cutoff frequency as a fraction of the sampling rate
        [bx, ax] = cheby2(order, atten, fcutx)
        xfilt = lfilter(bx, ax, x)

        #   Reduce the resampled signal bandwidth to 21 kHz
        fcuty = 21 / fy
        [by, ay] = cheby2(order, atten, fcuty)
        yfilt = lfilter(by, ay, y)

        #   Compute the input and output RMS levels within the 21 kHz bandwidth
        #   and match the output to the input
        xRMS = np.sqrt(np.mean(np.power(xfilt, 2)))
        yRMS = np.sqrt(np.mean(np.power(yfilt, 2)))
        y = (xRMS / yRMS) * y

    return y, fsamp


def eb_aveSL(env: np.array,
             control,
             attnOHC: np.array,
             thrLow,
             CR,
             attnIHC,
             Level1):
    '''
    function xdB=eb_aveSL(env,control,attnOHC,thrLow,CR,attnIHC,Level1)
    # Function to covert the RMS average output of the gammatone filter bank
    # into dB SL. The gain is linear below the lower threshold, compressive
    # with a compression ratio of CR:1 between the lower and upper thresholds,
    # and reverts to linear above the upper threshold. The compressor
    # assumes that auditory thresold is 0 dB SPL.
    #
    # Calling variables:
    # env		analytic signal envelope (magnitude) returned by the
    #			gammatone filter bank, RMS average level
    # control   control signal envelope
    # attnOHC	OHC attenuation at the input to the compressor
    # thrLow	kneepoint for the low-level linear amplification
    # CR		compression ratio
    # attnIHC	IHC attenuation at the input to the synapse
    # Level1	dB reference level: a signal having an RMS value of 1 is
    #			assigned to Level1 dB SPL.
    #
    # Function output:
    # xdB		compressed output in dB above the impaired threshold
    #
    # James M. Kates, 6 August 2007.
    # Version for two-tone suppression, 29 August 2008.
    '''

    # Initialize the compression parameters
    thrHigh = 100.0  # Upper compression threshold

    # Convert the control to dB SPL
    small = 1.0e-30
    logenv = np.max(
        [control, small])  # Don't want to take logarithm of zero or neg
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.min(
        [logenv, thrHigh])  # Clip signal levels above the upper threshold
    logenv = np.max([logenv, thrLow])  # Clip signal at the lower threshold

    # Compute the compression gain in dB
    gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))  # Gain in dB

    # Convert the signal envelope to dB SPL
    logenv = np.max([env, small])  # Don't want to take logarithm of zero or neg
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.max([logenv, 0])  # Clip signal at auditory threshold
    xdB = logenv + gain - attnIHC  # Apply gain to the log spectrum
    xdB = max([xdB, 0.0])  # dB SL

    return xdB


def ebm_AddNoise(ydB: np.ndarray,
                 thrdB):
    '''
    function zdB = ebm_AddNoise(ydB,thrdB)
    # Function to add independent random Gaussian noise to the subsampled
    # signal envelope in each auditory frequency band.
    #
    # Calling arguments:
    # ydB      subsampled envelope in dB re:auditory threshold
    # thrdB    additive noise RMS level (in dB)
    # Level1   an input having RMS=1 corresponds to Level1 dB SPL
    #
    # Returned values:
    # zdB      envelope with threshold noise added, in dB re:auditory threshold
    #
    # James M. Kates, 23 April 2019.
    '''

    # Additive noise sequence
    noise = thrdB * default_rng().standard_normal(
        ydB.shape)  # Gaussian noise with RMS=1, then scaled

    # Add the noise to the signal envelope
    zdB = ydB + noise

    return zdB


def ebm_CepCoef(xdB: np.ndarray,
                ydB: np.ndarray,
                thrCep,
                thrNerve,
                nbasis):
    '''
    function [xcep,ycep] = ebm_CepCoef(xdB,ydB,thrCep,thrNerve,nbasis)
    # Function to compute the cepstral correlation coefficients between the
    # reference signal and the distorted signal log envelopes. The silence
    # portions of the signals are removed prior to the calculation based on the
    # envelope of the reference signal. For each time sample, the log spectrum
    # in dB SL is fitted with a set of half-cosine basis functions. The
    # cepstral coefficients then form the input to the cepstral correlation
    # calculation.
    #
    # Calling variables:
    # xdB       subsampled reference signal envelope in dB SL in each band
    # ydB	    subsampled distorted output signal envelope
    # thrCep    threshold in dB SPL to include sample in calculation
    # thrNerve  additive noise RMS for IHC firing (in dB)
    # nbasis    number of cepstral basis functions to use
    #
    # Output:
    # xcep      cepstral coefficient matrix for the ref signal (nsamp,nbasis)
    # ycep      cepstral coefficient matrix for the output signal (nsamp,nbasis)
    #             each column is a separate basis function, from low to high
    #
    # James M. Kates, 23 April 2015.
    # Gammawarp version to fit the basis functions, 11 February 2019.
    # Additive noise for IHC firing rates, 24 April 2019.
    '''

    # Processing parameters
    nbands = xdB.shape[0]  # Number of auditory frequency bands

    # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
    freq = np.arange(nbasis)
    k = np.arange(nbands)
    cepm = np.zeros([nbands, nbasis])
    for nb in range(nbasis):
        basis = np.cos(freq[nb] * np.pi * k / (nbands - 1))  # Basis functions
        cepm[:, nb] = basis / np.linalg.norm(basis)

    # Find the reference segments that lie sufficiently above the quiescent rate
    xLinear = np.power(10,
                       xdB / 20)  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear, 2) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.nonzero(xsum > thrCep)  # Identify those segments above threshold
    nsamp = len(index)  # Number of segments above threshold

    # Exit if not enough segments above zero
    if nsamp <= 1:
        xcep = []  # Return empty matrices
        ycep = []
        print('Function ebm_CepCoef: Signal below threshold');
        return xcep, ycep

    # Remove the silent samples
    xdB = xdB[index, :]
    ydB = ydB[index, :]

    # Add low-level noise to provide IHC firing jitter
    xdB = ebm_AddNoise(xdB, thrNerve)
    ydB = ebm_AddNoise(ydB, thrNerve)

    # Compute the mel cepstrum coefficients using only those samples above
    # threshold
    xcep = xdB * cepm
    ycep = ydB * cepm

    # Remove the average value from the cepstral coefficients. The
    # cepstral cross-correlation will thus be a cross-covariance, and there
    # is no effect of the absolute signal level in dB.
    for n in range(nbasis):
        x = xcep[:, n]
        x = x - np.mean(x)
        xcep[:, n] = x
        y = ycep[:, n]
        y = y - np.mean(y)
        ycep[:, n] = y

    return xcep, ycep


def ebm_EnvFilt(xdB: np.ndarray,
                ydB: np.ndarray,
                fcut,
                fsub,
                fsamp):
    '''
    function [xLP,yLP] = ebm_EnvFilt(xdB, ydB, fcut, fsub, fsamp)
    # Function to lowpass filter and subsample the envelope in dB SL produced
    # by the model of the auditory periphery. The LP filter uses a von Hann
    # raised cosine window to ensure that there are no negative envelope values
    # produced by the filtering operation.
    #
    # Calling arguments
    # xdB    matrix: env in dB SL for the ref signal in each auditory band
    # ydB    matrix: env in dB SL for the degraded signal in each auditory band
    # fcut   LP filter cutoff frequency for the filtered envelope, Hz
    # fsub   subsampling frequency in Hz for the LP filtered envelopes
    # fsamp  sampling rate in Hz for the signals xdB and ydB
    #
    # Returned values:
    # xLP    LP filtered and subsampled reference signal envelope
    #        Each frequency band is a separate column
    # yLP    LP filtered and subsampled degraded signal envelope
    #
    # James M. Kates, 12 September 2019.
    '''

    # Check the filter design parameters
    xLP = np.array([])  # Default error outputs
    yLP = np.array([])

    if fsub > fsamp:
        print(' Error in ebm_EnvFilt: Subsampling rate too high.')
        return xLP, yLP

    if fcut > 0.5 * fsub:
        print(' Error in ebm_EnvFilt: LP cutoff frequency too high.')
        return xLP, yLP

    # Check the data matrix orientation
    # Require each frequency band to be a separate column
    nrow = xdB.shape[0]  # Number of rows
    ncol = xdB.shape[1]  # Number of columns

    if ncol > nrow:
        xdB = xdB.T
        ydB = ydB.T

    nbands = xdB.shape[1]
    nsamp = xdB.shape[0]

    # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
    tfilt = 1000 * (1 / fcut)  # Filter length in ms
    tfilt = 0.7 * tfilt  # Empirical adjustment of the filter length
    nfilt = np.round(0.001 * tfilt * fsamp)  # Filter length in samples
    nhalf = np.floor(nfilt / 2)
    nfilt = 2 * nhalf  # Force an even filter length

    # Design the FIR LP filter using a von Hann window to ensure that there are
    # no negative envelope values
    benv = np.hanning(nfilt + 2)[1:-1]
    benv = benv / np.sum(benv);

    # LP filter for the envelopes at fsamp
    xenv = convolve2d(xdB, benv)  # 2-D convolution
    xenv = xenv[nhalf: (nhalf + nsamp), :]  # Remove the filter transients
    yenv = convolve2d(ydB, benv)
    yenv = yenv[nhalf: (nhalf + nsamp), :]

    # Subsample the LP filtered envelopes
    space = np.floor(fsamp / fsub)
    index = list(range(0, nsamp + 1, space))
    xLP = xenv[index, :]
    yLP = yenv[index, :]

    return xLP, yLP


def ebm_GetNeuralNet():
    '''
    function [NNparam,Whid,Wout,b] = ebm_GetNeuralNet
    # Function to provide the weights derived for the ensemble of ten neural
    # networks used for the HASPI_v2 intelligibility model. The neural networks
    # have ten inputs, 4 neurons in the hidden layer, and one output neuron.
    # The logsig activation function is used.
    #
    # Calling arguments: None
    #
    # Returned values:
    # NNparam  vector of parameters defining the neural network
    # Whid     cell array 10 x 1 for the weights linking the input to the hidden
    #          layer. Each cell is a 11 x 4 matrix of weights
    # Wout     call array 5 x 1 for the weights linking the hidden to the output
    #          layer. Each cell is a 5 x 1 vector of weights.
    # b        normalization so that the maximum neural net output is exactly 1.
    #
    # James M. Kates, 8 October 2019.
    # Version for new neural network using actual TFS scores, 24 October 2019.
    '''

    # Set up the neural network parameters
    NNparam = np.zeros([6, 1])
    NNparam[0] = 10  # Number of neurons in the input layer
    NNparam[1] = 4  # Number of neurons in the hidden layer
    NNparam[2] = 1  # Number of neurons in the output layer
    NNparam[3] = 1  # Activation function is logsig
    NNparam[4] = 0  # No offset for the activation function
    NNparam[5] = 1  # Maximum activation function value

    # Input to hidden layer weights
    Whid = np.empty(10, 1, dtype=object)
    Whid[0] = np.array(np.matrix(
        '4.9980 -13.0590 9.5478 -11.6760;'
        '18.9793 -8.5842 -6.6974 8.0382;'
        '-37.8234  26.9420 -6.6279  2.6069;'
        '4.1423 5.2106 10.3397 9.4643;'
        '-13.8839 3.1211 -5.7794 -1.9207;'
        '-17.0784 -8.5065 -16.7409 -1.6916;'
        '-0.0696 -19.9487 -13.9734 -20.3328;'
        '-10.5754 15.5461 -3.9137 -2.0726;'
        '-4.7145 5.0427 10.5728 28.7840;'
        '21.0595 -3.8171 2.2084 2.1929;'
        '17.2857  16.7562 -27.3290 1.1543'
    ))

    Whid[1] = np.array(np.matrix(
        '-11.8283 -12.3466 8.8198 5.6027;'
        '-8.3142 6.2553 -4.1575 13.7958;'
        '27.6080 3.3801 -7.9607 -33.7865;'
        '1.3185 5.7276 8.3761 0.8153;'
        '4.0206 3.4737 -7.0282 -9.8338;'
        '-7.3265 -4.0271 -12.3923 -12.5861;'
        '-17.9111 -23.1330 -16.2176 0.2218;'
        '15.0623 -3.9181 -2.3266 -21.2808;'
        '1.0537 34.5512 8.7196 -8.7648;'
        '-5.0357 -2.3610 -0.3678 31.4586;'
        '20.8312 7.8687 -28.9087 19.4417'
    ))

    Whid[2] = np.array(np.matrix(
        '9.5379 4.4994 -13.1308 0.9024;'
        '3.9544 -2.4002 2.6777 22.9810;'
        '-30.9452 -2.2645 15.2613 -23.8526;'
        '3.1327 18.3449 7.4923 -2.3167;'
        '-4.3189 6.5696 2.5123 -15.6430;'
        '-4.3704 -10.0506 2.0855 -19.4876;'
        '-9.6746 -9.9613 -30.5541 3.4877;'
        '-5.7179 -14.4015 9.3838 -14.9651;'
        '5.0717 -6.2980 26.6210 -6.7466;'
        '8.5857 -8.5345 -16.3236 18.1852;'
        '3.1709 -41.1078 6.7127 11.5747'
    ))

    Whid[3] = np.array(np.matrix(
        '9.3527 -13.3654 -2.1263 5.1205;'
        '9.4885 1.9584 21.8489 -8.0495;'
        '-32.0886 16.0934 -13.0956 -0.9466;'
        '-4.9347 6.1343 -0.7237 21.6024;'
        '-7.2456 6.2478 -16.2268 8.1160;'
        '-5.9809 0.7872 -20.7517 -9.8755;'
        '-7.6038 -32.4284 -0.3817 -10.7850;'
        '-5.5069 11.0813 -14.9053 -18.0625;'
        '8.9225 27.1473 -10.8270 -7.0454;'
        '7.4362 -19.8990 12.3480 -6.7305;'
        '6.3910 7.1670 11.7919 -38.1848'
    ))

    Whid[4] = np.array(np.matrix(
        '-12.0509 8.7151 12.9841 -12.7563;'
        '-8.0669 18.9493 -9.1899 7.8737;'
        '20.6577 -35.4767 -18.5397 2.8544;'
        '6.0629 -6.5786 10.9516 9.3709;'
        '5.0354 -18.6275 -0.5501 1.3219;'
        '21.0090 -21.7111 5.1285 -0.5481;'
        '8.3379 -5.0779 8.1280 -29.8252;'
        '19.6124 -5.0156 -0.1799 -5.3723;'
        '6.8287 4.5828 16.1024 40.0935;'
        '-30.5649 10.5307 -11.8234 0.4014;'
        '-9.4186 15.6892 -44.0505 1.4371'
    ))

    Whid[5] = np.array(np.matrix(
        '8.9905 -16.4000 13.3395 8.9068;'
        '11.0010 11.3797 14.8502 -14.2547;'
        '-23.8174 4.4221 -34.6896 -9.9423;'
        '-8.1285 4.0386 -5.7528 7.6275;'
        '-17.7683 3.2188 -0.4409 3.8280;'
        '-14.2883 2.4917 -16.7262 13.1258;'
        '-5.8409 -13.2882 -4.2047 22.9793;'
        '1.7396 4.2947 -13.9206 4.2493;'
        '7.8760 21.4827 -14.9673 -8.3899;'
        '6.7850 -4.3356 18.5928 - 12.0981;'
        '7.4116 -2.0622 4.7621 -40.2684'
    ))

    Whid[6] = np.array(np.matrix(
        '-13.2736 9.9119 3.4659 2.8783;'
        '0.4675 -0.8187 0.3497 20.7397;'
        '17.4133 -27.7575 -1.4997 -23.8363;'
        '3.9760 4.8989 15.8285 -6.6393;'
        '7.6936 1.1009 5.0979 -15.8340;'
        '-0.2380 -4.6432 -8.9580 -17.8548;'
        '-31.1510 -14.2219 -11.0122 3.0247;'
        '9.6552 -7.9702 -14.6836 -12.9456;'
        '25.9963 6.3569 -5.0912 -5.4249;'
        '-15.9809 9.4330 -10.4158 15.9834;'
        '6.1126 0.1713 -43.7492 14.7425'
    ))

    Whid[7] = np.array(np.matrix(
        '-11.6727 -15.7084 9.9095 -7.3946;'
        '4.4142 -4.4821 10.9888 0.0966;'
        '6.4298 25.5445 -32.7311 4.1951;'
        '8.4468 16.3594 7.0755 7.2817;'
        '-2.5481 15.7296 -12.2159 -2.5490;'
        '-3.2812 -0.6972 -13.1754 -0.7216;'
        '-19.5254 -25.2440 -7.6636 -15.0124;'
        '2.3548 8.5716 -6.7492 3.8422;'
        '26.9615 6.6441 3.1680 15.6611;'
        '6.6129 -15.7791 9.3453 2.7809;'
        '-3.6429 -0.8727 0.2410 -0.7045'
    ))

    Whid[8] = np.array(np.matrix(
        '-13.9106 3.1943 8.7525 7.8378;'
        '4.1210 0.4603 -7.2471 16.2216;'
        '9.3064 -3.8093 -14.4721 -34.2848;'
        '11.6147 17.6926 -1.5339 2.6700;'
        '5.3305 4.0299 -13.0022 -15.3827;'
        '-3.5035 -7.2305 6.8711 -12.6676;'
        '-25.5936 -9.8940 10.5552 2.4690;'
        '7.7159 -17.8905 6.5517 -17.6486;'
        '26.7162 -5.0092 -3.5613 -0.0383;'
        '-11.7304 -6.5251 -4.2616 19.8528;'
        '3.2551 -35.4889 -2.2133 6.7308'
    ))

    Whid[9] = np.array(np.matrix(
        '13.5754 -13.4585 2.5816 7.5809;'
        '-9.7189 7.6225 -3.0220 17.7773;'
        '-25.6273 4.1225 4.2090 -35.4511;'
        '5.3909 11.0694 15.5337 -1.3336;'
        '-1.2964 5.5829 6.9950 -9.9642;'
        '10.1510 2.2819 -9.6950 -14.6332;'
        '12.5032 -31.1403 -13.2782 0.1385;'
        '-2.6178 6.8453 -20.5308 -16.9705;'
        '-2.5462 30.2576 -3.5750 1.3910;'
        '-6.2286 -14.7841 -7.3953 17.8740;'
        '-15.8615 3.6023 -40.9104 7.7481'
    ))

    # Hidden to output layer weights
    # Column vectors, separate values with semi-colons
    Wout = np.array(10, 1, dtype=object)

    Wout[0] = np.array(np.matrix('-0.1316; -2.5182; 1.6401; -3.2093; 1.7924'))
    Wout[1] = np.array(np.matrix('-0.1653; 1.7375; 1.5526; -3.2349; -2.2877'))
    Wout[2] = np.array(np.matrix('0.1847; -3.1987; -2.4941; 2.7106; -1.8048'))
    Wout[3] = np.array(np.matrix('0.3962; -3.2952; 3.0003; -2.2602; -2.3269'))
    Wout[4] = np.array(np.matrix('-0.0646; 1.3288; -3.4087; -2.0046; 1.8565'))
    Wout[5] = np.array(np.matrix('1.3676; -3.4129; 1.6895; -1.8913; -1.5595'))
    Wout[6] = np.array(np.matrix('0.8124; 2.7171; -3.0867; -2.3310; -2.3657'))
    Wout[7] = np.array(np.matrix('-0.2743; 1.4949; 0.7896; -4.0589; 1.1257'))
    Wout[8] = np.array(np.matrix('0.1307; 2.2788; -2.3633; -1.5073; -2.9985'))
    Wout[9] = np.array(np.matrix('0.1024; -0.9517; 2.2123; -2.4008; -3.1655]'))

    # Normalization factor
    b = 0.9508

    return NNparam, Whid, Wout, b


def ebm_ModCorr(Xmod: np.ndarray,
                Ymod: np.ndarray):
    '''
    function aveCM = ebm_ModCorr(Xmod, Ymod)
    # Function to compute the cross-correlations between the input signal
    # time-frequency envelope and the distortion time-frequency envelope. The
    # cepstral coefficients or envelopes in each frequency band have been
    # passed through the modulation filterbank using function ebm_ModFilt.
    #
    # Calling variables:
    # Xmod	   cell array containing the reference signal output of the
    #          modulation filterbank. Xmod is of size {nchan,nmodfilt} where
    #          nchan is the number of frequency channels or cepstral basis
    #          functions in Xenv, and nmodfilt is the number of modulation
    #          filters used in the analysis. Each cell contains a column vector
    #          of length nsamp, where nsamp is the number of samples in each
    #          envelope sequence contained in the columns of Xenv.
    # Ymod	   subsampled distorted output signal envelope
    #
    # Output:
    # aveCM    modulation correlations averaged over basis functions 2-6
    #          vector of size nmodfilt
    #
    # James M. Kates, 21 February 2019.
    '''

    # Processing parameters
    nchan = Xmod.shape[0]  # Number of basis functions
    nmod = Xmod.shape[1]  # Number of modulation filters
    small = 1.0e-30  # Zero threshold

    # Compute the cross-covariance matrix
    CM = np.zeros([nchan, nmod])
    for m in range(nmod):
        for j in range(nchan):
            # Index j gives the input reference band
            xj = Xmod[j, m]  # Input freq band j, modulation freq m
            xj = xj - np.mean(xj)
            xsum = np.sum(np.power(xj, 2))

            # Processed signal band
            yj = Ymod[j, m]  # Processed freq band j, modulation freq m
            yj = yj - np.mean(yj)
            ysum = np.sum(np.power(yj, 2))

            #       Cross-correlate the reference and processed signals
            if (xsum < small) | (ysum < small):
                CM[j, m] = 0
            else:
                CM[j, m] = np.abs(np.sum(xj * yj)) / np.sqrt(xsum * ysum)

    aveCM = np.mean(CM[1:6, :], 1);  # Average over basis functions 2 - 6

    return aveCM


def ebm_ModFilt(Xenv: np.ndarray,
                Yenv: np.ndarray,
                fsub):
    '''
    function[Xmod, Ymod, cf] = ebm_ModFilt(Xenv, Yenv, fsub)
    # Function to apply an FIR modulation filterbank to the reference envelope
    # signals contained in matrix Xenv and the processed signal envelope
    # signals in matrix Yenv. Each column in Xenv and Yenv is a separate filter
    # band or cepstral coefficient basis function. The modulation filters use a
    # lowpass filter for the lowest modulation rate, and complex demodulation
    # followed by a lowpass filter for the remaining bands. The onset and
    # offset transients are removed from the FIR convolutions to temporally
    # align the modulation filter outputs.
    #
    # Calling arguments:
    # Xenv     matrix containing the subsampled reference envelope values. Each
    #          column is a different frequency band or cepstral basis function
    #          arranged from low to high.
    # Yenv     matrix containing the subsampled processed envelope values
    # fsub     envelope sub-sampling rate in Hz
    #
    # Returned values:
    # Xmod     cell array containing the reference signal output of the
    #          modulation filterbank. Xmod is of size {nchan,nmodfilt} where
    #          nchan is the number of frequency channels or cepstral basis
    #          functions in Xenv, and nmodfilt is the number of modulation
    #          filters used in the analysis. Each cell contains a column vector
    #          of length nsamp, where nsamp is the number of samples in each
    #          envelope sequence contained in the columns of Xenv.
    # Ymod     cell array containing the processed signal output of the
    #          modulation filterbank.
    # cf       vector of the modulation rate filter center frequencies
    #
    # James M. Kates, 14 February 2019.
    # Two matrix version of gwarp_ModFiltWindow, 19 February 2019.
    '''

    # Input signal properties
    nsamp = Xenv.shape[0]
    nchan = Xenv.shape[1]

    # Modulation filter band cf and edges, 10 bands
    # Band spacing uses the factor of 1.6 used by Dau
    cf = np.array(
        [2, 6, 10, 16, 25, 40, 64, 100, 160, 256])  # Band center frequencies
    nmod = len(cf)
    edge = np.zeros(1, nmod + 1)  # Includes lowest and highest edges
    edge[0: 3] = np.array(
        [0, 4, 8])  # Uniform spacing for lowest two modulation filters

    for k in range(3, nmod + 1):
        #   Log spacing for remaining constant-Q modulation filters
        edge[k] = np.power(cf[k - 1], 2) / edge[k - 1];

    # Allowable filters based on envelope subsampling rate
    fNyq = 0.5 * fsub
    index = edge < fNyq
    edge = edge[index]  # Filter upper band edges less than Nyquist rate
    nmod = len(edge) - 1
    cf = cf[:nmod]

    # Assign FIR filter lengths. Setting t0=0.2 gives a filter Q of about
    # 1.25 to match Ewert et al. (2002), and t0=0.33 corresponds to Q=2 (Dau et
    # al 1997a). Moritz et al. (2015) used t0=0.29. General relation Q=6.25*t0,
    # compromise with t0=0.24 which gives Q=1.5
    t0 = 0.24  # Filter length in sec for the lowest modulation frequency band
    t = np.zeros([nmod, 1])
    t[1] = t0
    t[2] = t0
    t[2: nmod] = t0 * cf[4] / cf[2: nmod]  # Constant-Q filters above 10 Hz
    nfir = 2 * np.floor(t * fsub / 2)  # Force even filter lengths in samples
    nhalf = nfir / 2

    # Design the family of lowpass windows
    b = np.empty(nmod, 1,
                 dtype=object)  # Filter coefficients, one filter impulse reponse per cell
    for k in range(nmod):
        b[k] = np.hanning(nfir[k] + 1 + 2)[1: -1]  # Lowpass window
        b[k] = b[k] / sum(b[k])  # Normalize to 0-dB gain at cf

    # Pre-compute the cosine and sine arrays
    co = np.empty(nmod, 1, dtype=object)  # cosine array, one frequency per cell
    si = np.empty(nmod, 1, dtype=object)  # sine array, one frequency per cell
    n = list(range(1, nsamp + 1));
    for k in range(nmod):
        if k == 0:
            co[k] = 1
            si[k] = 0
        else:
            co[k] = (np.sqrt(2) * np.cos(np.pi * n * cf[k] / fNyq)).T
            si[k] = (np.sqrt(2) * np.sin(np.pi * n * cf[k] / fNyq)).T

    # Convolve the input and output envelopes with the modulation filters
    Xmod = np.empty(nchan, nmod, dtype=object)
    Ymod = Xmod.copy()
    for k in range(nmod):  # Loop over the modulation filters
        bk = b[k]  # Extract the lowpass filter impulse response
        nh = nhalf[k]  # Transient duration for the filter
        c = co[k]  # Cosine and sine for complex modulation
        s = si[k]
        for m in range(nchan):  # Loop over the input signal vectors
            #       Reference signal
            x = Xenv[:, m]  # Extract the frequency or cepstral coefficient band
            u = convolve((x * c - 1j * x * s),
                         bk)  # Complex demodulation, then LP filter
            u = u[nh: (nh + nsamp)]  # Truncate the filter transients
            xfilt = np.real(u) * c - np.imag(
                u) * s  # Modulate back up to the carrier freq
            Xmod[m, k] = xfilt.copy()  # Save the filtered signal

            #       Processed signal
            y = Yenv[:, m]  # Extract the frequency or cepstral coefficient band
            v = convolve((y * c - 1j * y * s),
                         bk)  # Complex demodulation, then LP filter
            v = v[nh: (nh + nsamp)]  # Truncate the filter transients
            yfilt = np.real(v) * c - np.imag(
                v) * s  # Modulate back up to the carrier freq
            Ymod[m, k] = yfilt  # Save the filtered signal
