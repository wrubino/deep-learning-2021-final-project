import numpy as np
from numpy.random import default_rng
from matplotlib.pyplot import xcorr

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
        x, fx, y, fy, HL, itype, Level1, nargout=7)
    # .\HASPI_v2.m:38

    # ---------------------------------------
    # Envelope modulation features
    # LP filter and subsample the envelopes
    fLP = 320
    # .\HASPI_v2.m:44

    fsub = dot(8, fLP)
    # .\HASPI_v2.m:45

    xLP, yLP = ebm_EnvFilt(xenv, yenv, fLP, fsub, fsamp, nargout=2)

    # .\HASPI_v2.m:46
    # Compute the cepstral coefficients as a function of subsampled time
    nbasis = 6
    # .\HASPI_v2.m:49

    thr = 2.5
    # .\HASPI_v2.m:50

    dither = 0.1
    # .\HASPI_v2.m:51

    xcep, ycep = ebm_CepCoef(xLP, yLP, thr, dither, nbasis, nargout=2)
    # .\HASPI_v2.m:52

    # Cepstral coefficients filtered at each modulation rate
    # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
    # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
    xmod, ymod, cfmod = ebm_ModFilt(xcep, ycep, fsub, nargout=3)
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
    x = np.zeros(nx + 1, 1)
    # .\NNfeedforward.m:32

    hidden = np.zeros(nhid + 1, 1)
    # .\NNfeedforward.m:33

    output = np.zeros(nout + 1, 1)
    # .\NNfeedforward.m:34

    # Initialize the nodes used for constants
    x[0] = 1
    # .\NNfeedforward.m:37

    hidden[0] = 1.0
    # .\NNfeedforward.m:38

    output[0] = 1.0
    # .\NNfeedforward.m:39

    # Input layer
    for i in arange(1, nx + 1):
        x[i] = data[i - 1]
    # .\NNfeedforward.m:43

    # Response of the hidden layer
    for j in arange(1, nhid + 1):
        sumhid = np.sum(np.dot(Whid[:, j - 1], x))
        hidden[j] = (again / (1.0 + exp(-beta * sumhid))) - offset

    # .\NNfeedforward.m:49

    # Response of the output layer
    for k in arange(1, nout + 1):
        sumout = np.sum(np.dot(Wout[:, k - 1], hidden))

        # .\NNfeedforward.m:54
        output[k] = (again / (1.0 + exp(- beta * sumout))) - offset
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
    predict = np.zeros(ncond, K)

    for k in np.arange(K):
        for n in np.arange(ncond):
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
    cf = (
            -(EarQ * minBW)
            + np.exp(np.arange(1, nchan)) * -(
            np.log(highFreq + EarQ * minBW)
            + np.log(lowFreq + EarQ * minBW) / (nchan - 1)
    ) * (highFreq + EarQ * minBW)
    )

    # Last center frequency is set to highFreq
    cf = np.array([[highFreq], [cf]])
    cf = np.flipud(cf)  # Reorder to put the low frequencies first

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
    y24 = y24(0: nxy]

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
    xdB = np.zeros(nchan, nsamp)
    ydB = xdB
    xBM = xdB
    yBM = ydB

    # Reference and processed average spectral values
    xave = np.zeros(nchan, 1)  # Reference
    yave = xave  # Processed
    xcave = xave  # Reference control
    ycave = yave  # Processed control

    # Filter bandwidths adjusted for intensity
    BWx = np.zeros(nchan, 1)
    BWy = BWx

    # Loop over each filter in the auditory filter bank
    for n in np.arange(nchan):
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
    xy = xcorr(x, y, maxlags=lags - 1)
    location = np.argmax(xy)  # Find the peak

    # Compute the delay
    delay = lags - location

    # Time shift the output sequence
    if delay > 0:
        #   Output delayed relative to the reference
        y = np.array([y[delay : npts], np.zeros(delay, 1)])  # Remove the delay
    elif delay < 0:
        #   Output advanced relative to the reference
        delay = -delay
        y =np.array([np.zeros(delay, 1), y[0 : (npts - delay)]])  # Add advance

    return y
