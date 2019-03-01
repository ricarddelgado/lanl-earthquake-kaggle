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
    
    diff_crossings = np.diff(crossings[0])
    min_diff_crossings = np.min(diff_crossings)
    max_diff_crossings = np.max(diff_crossings)
    mean_diff_crossings = np.mean(diff_crossings)
    median_diff_crossings = np.median(diff_crossings) 

    mean_freq = fs / mean_diff_crossings
    median_freq = fs / median_diff_crossings

    mean_freq = fs / mean(diff(crossings[0]))
    median_freq = fs / np.median(diff(crossings[0]))
    
    #plt.figure(figsize=(16, 8))
    #plt.hist(diff_crossings, bins=500)
    #plt.title(f'Histogram of diff crossings mean={mean_diff_crossings:0.1f} - {round(mean_freq)}Hz, median={median_diff_crossings:0.1f} - {round(median_freq)}')
    #plt.ylabel('Count')
    #plt.xlim(0,200)
    #plt.axvline(x=mean_diff_crossings, color='g')
    #plt.axvline(x=median_diff_crossings, color='r')
    #plt.show()
    
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

    #plt.figure(figsize=(16, 8))
    #plt.plot(pw_spectrum)
    #plt.axvline(x=i, color='r')
    #plt.title(f'Power spectrum peak={peak_freq}Hz')
    #plt.ylabel('Power')
    #plt.xlabel('Sample')
    #plt.show()   

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

    #plt.figure(figsize=(16, 8))
    #plt.plot(pw_spectrum)
    #plt.axvline(x=i, color='r')
    #plt.title(f'Power spectrum peak={peak_freq}Hz')
    #plt.ylabel('Power')
    #plt.xlabel('Sample')
    #plt.show() 
