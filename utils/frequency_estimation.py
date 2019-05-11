from __future__ import division
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate, nonzero
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

# Faster version from http://projects.scipy.org/scipy/browser/trunk/scipy/signal/signaltools.py
# from signaltoolsmod import fftconvolve
from utils.parabolic import parabolic

def freq_from_crossings(sig, fs):
    """
    Estimates frequency by counting zero crossings
    """

    # Find all indices right before a rising-edge zero crossing
    idx = nonzero((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in idx]
    
    if not crossings:
        median_freq = 0
    else:
        median_freq = fs / np.median(diff(crossings[0]))
    
    return median_freq

def freq_from_fft(sig, fs):
    """
    Estimates frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)
    pw_spectrum = np.abs(f)
    pw_spectrum[0:15] = np.nan

    # Find the peak
    i = argmax(pw_spectrum[16:])+15
    peak_freq = fs * i / len(windowed)

    return peak_freq

def freq_from_welch(sig, fs):
    """
    Estimates frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)
    pw_spectrum = np.abs(f)
    pw_spectrum[0:15] = np.nan

    f_welch, S_welch = welch(
    y, fs=Fs, nperseg=nperseg, noverlap=(nperseg // 2),
    detrend=None, scaling='density', window='hanning')
    
    # Find the peak
    i = argmax(pw_spectrum[16:])+15
    peak_freq = fs * i / len(windowed)

