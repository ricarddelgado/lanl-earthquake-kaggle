import numpy as np
import pandas as pd
import scipy.signal as sg
from itertools import product
import pywt 
from scipy import signal
from scipy.signal import butter, deconvolve
from scipy import stats
from scipy.signal import hilbert, hann, convolve
from sklearn.linear_model import LinearRegression
from tsfresh.feature_extraction import feature_calculators

from utils.frequency_estimation import freq_from_crossings

NY_FREQ_IDX = 75000
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500
borders = list(range(-4000, 4001, 1000))

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff, SAMPLE_RATE):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def change_rate(x, method='original'):
    if method == 'original':
        rate = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    if method == 'modified':
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        rate = np.mean(change)
    return rate

def sta_lta_ratio(x, length_sta, length_lta, method='original'):
    if method=='original':
        sta = np.cumsum(x ** 2)
        # Convert to float
        sta = np.require(sta, dtype=np.float)
        # Copy for LTA
        lta = sta.copy()
        # Compute the STA and the LTA
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta /= length_sta
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta /= length_lta
        # Pad zeros
        sta[:length_lta - 1] = 0
        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny
        ratio = sta / lta
        
    elif method == 'modified':
        x_abs = np.abs(x)
        # Convert to float
        x_abs = np.require(x_abs, dtype=np.float)
        # Compute the STA and the LTA
        sta = np.cumsum(x_abs)
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta = sta[length_sta - 1:] / length_sta
        sta = sta[:-(length_lta-length_sta)]
        lta = x_abs.copy()
        lta = np.cumsum(lta)
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta = lta[length_lta - 1:] / length_lta
        ratio = sta / lta

    return ratio

def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a

def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a

def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high / NY_FREQ_IDX), btype='bandpass')
    return b, a

def create_all_features(seg_id, seg, X, fs):
    xc = pd.Series(seg['acoustic_data'].values)
    zc = np.fft.fft(xc)
    
    # Generic stats
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    
    #FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()
    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()
    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()
    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()
    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()
    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()
    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()
    
    X.loc[seg_id, 'mean_diff'] = np.mean(np.diff(xc))
    X.loc[seg_id, 'mean_abs_diff'] = np.mean(np.abs(np.diff(xc)))
    X.loc[seg_id, 'mean_change_rate'] = change_rate(xc, method='original')
    X.loc[seg_id, 'mean_change_rate_v2'] = change_rate(xc, method='modified')
    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()
    
    # Classical stats by segment
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()
    
    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()
    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()
    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()
    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()
    
    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()
    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()
    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()
    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()
    
    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()
    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()
    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()
    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()
    
    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())
    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])
    X.loc[seg_id, 'sum'] = xc.sum()
    
    X.loc[seg_id, 'mean_change_rate_first_50000'] = change_rate(xc[:50000], method='original')
    X.loc[seg_id, 'mean_change_rate_last_50000'] = change_rate(xc[-50000:], method='original')
    X.loc[seg_id, 'mean_change_rate_first_10000'] = change_rate(xc[:10000], method='original')
    X.loc[seg_id, 'mean_change_rate_last_10000'] = change_rate(xc[-10000:], method='original')

    X.loc[seg_id, 'mean_change_rate_first_50000_v2'] = change_rate(xc[:50000], method='modified')
    X.loc[seg_id, 'mean_change_rate_last_50000_v2'] = change_rate(xc[-50000:], method='modified')
    X.loc[seg_id, 'mean_change_rate_first_10000_v2'] = change_rate(xc[:10000], method='modified')
    X.loc[seg_id, 'mean_change_rate_last_10000_v2'] = change_rate(xc[-10000:], method='modified')

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    
    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)
    
    X.loc[seg_id, 'trend'] = add_trend_feature(xc)
    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()
    
    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()
    
    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

    sta_lta_method = 'original'
    classic_sta_lta1 = sta_lta_ratio(xc, 500, 10000, method=sta_lta_method)
    classic_sta_lta2 = sta_lta_ratio(xc, 5000, 100000, method=sta_lta_method)
    classic_sta_lta3 = sta_lta_ratio(xc, 3333, 6666, method=sta_lta_method)
    classic_sta_lta4 = sta_lta_ratio(xc, 10000, 25000, method=sta_lta_method)
    classic_sta_lta5 = sta_lta_ratio(xc, 50, 1000, method=sta_lta_method)
    classic_sta_lta6 = sta_lta_ratio(xc, 100, 5000, method=sta_lta_method)
    classic_sta_lta7 = sta_lta_ratio(xc, 333, 666, method=sta_lta_method)
    classic_sta_lta8 = sta_lta_ratio(xc, 4000, 10000, method=sta_lta_method)
    
    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta1.mean()
    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta2.mean()
    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta3.mean()
    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta4.mean()
    X.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta5.mean()
    X.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta6.mean()
    X.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta7.mean()
    X.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta8.mean()

    X.loc[seg_id, 'classic_sta_lta1_q95'] = np.quantile(classic_sta_lta1, 0.95)
    X.loc[seg_id, 'classic_sta_lta2_q95'] = np.quantile(classic_sta_lta2, 0.95)
    X.loc[seg_id, 'classic_sta_lta3_q95'] = np.quantile(classic_sta_lta3, 0.95)
    X.loc[seg_id, 'classic_sta_lta4_q95'] = np.quantile(classic_sta_lta4, 0.95)
    X.loc[seg_id, 'classic_sta_lta5_q95'] = np.quantile(classic_sta_lta5, 0.95)
    X.loc[seg_id, 'classic_sta_lta6_q95'] = np.quantile(classic_sta_lta6, 0.95)
    X.loc[seg_id, 'classic_sta_lta7_q95'] = np.quantile(classic_sta_lta7, 0.95)
    X.loc[seg_id, 'classic_sta_lta8_q95'] = np.quantile(classic_sta_lta8, 0.95)   

    X.loc[seg_id, 'classic_sta_lta1_q05'] = np.quantile(classic_sta_lta1, 0.05)
    X.loc[seg_id, 'classic_sta_lta2_q05'] = np.quantile(classic_sta_lta2, 0.05)
    X.loc[seg_id, 'classic_sta_lta3_q05'] = np.quantile(classic_sta_lta3, 0.05)
    X.loc[seg_id, 'classic_sta_lta4_q05'] = np.quantile(classic_sta_lta4, 0.05)
    X.loc[seg_id, 'classic_sta_lta5_q05'] = np.quantile(classic_sta_lta5, 0.05)
    X.loc[seg_id, 'classic_sta_lta6_q05'] = np.quantile(classic_sta_lta6, 0.05)
    X.loc[seg_id, 'classic_sta_lta7_q05'] = np.quantile(classic_sta_lta7, 0.05)
    X.loc[seg_id, 'classic_sta_lta8_q05'] = np.quantile(classic_sta_lta8, 0.05)

    sta_lta_method = 'modified'
    classic_sta_lta1 = sta_lta_ratio(xc, 500, 10000, method=sta_lta_method)
    classic_sta_lta2 = sta_lta_ratio(xc, 5000, 100000, method=sta_lta_method)
    classic_sta_lta3 = sta_lta_ratio(xc, 3333, 6666, method=sta_lta_method)
    classic_sta_lta4 = sta_lta_ratio(xc, 10000, 25000, method=sta_lta_method)
    classic_sta_lta5 = sta_lta_ratio(xc, 50, 1000, method=sta_lta_method)
    classic_sta_lta6 = sta_lta_ratio(xc, 100, 5000, method=sta_lta_method)
    classic_sta_lta7 = sta_lta_ratio(xc, 333, 666, method=sta_lta_method)
    classic_sta_lta8 = sta_lta_ratio(xc, 4000, 10000, method=sta_lta_method)
    
    X.loc[seg_id, 'modified_sta_lta1_mean'] = classic_sta_lta1.mean()
    X.loc[seg_id, 'modified_sta_lta2_mean'] = classic_sta_lta2.mean()
    X.loc[seg_id, 'modified_sta_lta3_mean'] = classic_sta_lta3.mean()
    X.loc[seg_id, 'modified_sta_lta4_mean'] = classic_sta_lta4.mean()
    X.loc[seg_id, 'modified_sta_lta5_mean'] = classic_sta_lta5.mean()
    X.loc[seg_id, 'modified_sta_lta6_mean'] = classic_sta_lta6.mean()
    X.loc[seg_id, 'modified_sta_lta7_mean'] = classic_sta_lta7.mean()
    X.loc[seg_id, 'modified_sta_lta8_mean'] = classic_sta_lta8.mean()

    X.loc[seg_id, 'modified_sta_lta1_q95'] = np.quantile(classic_sta_lta1, 0.95)
    X.loc[seg_id, 'modified_sta_lta2_q95'] = np.quantile(classic_sta_lta2, 0.95)
    X.loc[seg_id, 'modified_sta_lta3_q95'] = np.quantile(classic_sta_lta3, 0.95)
    X.loc[seg_id, 'modified_sta_lta4_q95'] = np.quantile(classic_sta_lta4, 0.95)
    X.loc[seg_id, 'modified_sta_lta5_q95'] = np.quantile(classic_sta_lta5, 0.95)
    X.loc[seg_id, 'modified_sta_lta6_q95'] = np.quantile(classic_sta_lta6, 0.95)
    X.loc[seg_id, 'modified_sta_lta7_q95'] = np.quantile(classic_sta_lta7, 0.95)
    X.loc[seg_id, 'modified_sta_lta8_q95'] = np.quantile(classic_sta_lta8, 0.95)   

    X.loc[seg_id, 'modified_sta_lta1_q05'] = np.quantile(classic_sta_lta1, 0.05)
    X.loc[seg_id, 'modified_sta_lta2_q05'] = np.quantile(classic_sta_lta2, 0.05)
    X.loc[seg_id, 'modified_sta_lta3_q05'] = np.quantile(classic_sta_lta3, 0.05)
    X.loc[seg_id, 'modified_sta_lta4_q05'] = np.quantile(classic_sta_lta4, 0.05)
    X.loc[seg_id, 'modified_sta_lta5_q05'] = np.quantile(classic_sta_lta5, 0.05)
    X.loc[seg_id, 'modified_sta_lta6_q05'] = np.quantile(classic_sta_lta6, 0.05)
    X.loc[seg_id, 'modified_sta_lta7_q05'] = np.quantile(classic_sta_lta7, 0.05)
    X.loc[seg_id, 'modified_sta_lta8_q05'] = np.quantile(classic_sta_lta8, 0.05)

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_30000_mean'] = xc.rolling(window=30000).mean().mean(skipna=True)

    ewma = pd.Series.ewm
    X.loc[seg_id, 'exp_Moving_average_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_6000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=30000).mean().mean(skipna=True)

    # rdg: TODO it seems a parameter to tune
    no_of_std = 2
    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    X.loc[seg_id, 'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    X.loc[seg_id, 'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()
    
    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    X.loc[seg_id, 'q999'] = np.quantile(xc, 0.999)
    X.loc[seg_id, 'q001'] = np.quantile(xc, 0.001)
    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

    X.loc[seg_id, 'freq_cross_first_50000'] = freq_from_crossings(xc.values[:50000], fs)
    X.loc[seg_id, 'freq_cross_last_50000'] = freq_from_crossings(xc.values[-50000:], fs)
    X.loc[seg_id, 'freq_cross_first_10000'] = freq_from_crossings(xc.values[:10000], fs)
    X.loc[seg_id, 'freq_cross_last_10000'] = freq_from_crossings(xc.values[-10000:], fs)
    
    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values
        
        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.abs(np.diff(x_roll_std)))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = change_rate(pd.Series(x_roll_std), method='original')
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows) + 'v2'] = change_rate(pd.Series(x_roll_std), method='modified')
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.abs(np.diff(x_roll_mean)))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = change_rate(pd.Series(x_roll_mean), method = 'original')
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows) + '_v2'] = change_rate(pd.Series(x_roll_mean), method = 'modified')
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

def compute_bp_features_block(xc, seg_id, X):

    xcdm = xc - np.mean(xc)

    # Features by bandwith
    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = sg.lfilter(b, a, xcdm)
    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = sg.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3), pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]
    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_{i}'] = sig.mean()
        X.loc[seg_id, 'std_{i}'] = sig.std()
        X.loc[seg_id, 'max_{i}'] = sig.max()
        X.loc[seg_id, 'min_{i}'] = sig.min()

        X.loc[seg_id, 'mean_change_abs_{i}'] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_{i}'] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        X.loc[seg_id, 'abs_max_{i}'] = np.abs(sig).max()
        X.loc[seg_id, 'abs_min_{i}'] = np.abs(sig).min()

        X.loc[seg_id, 'std_first_50000_{i}'] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_{i}'] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_{i}'] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_{i}'] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_{i}'] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_{i}'] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_{i}'] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_{i}'] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_{i}'] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_{i}'] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_{i}'] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_{i}'] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_{i}'] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_{i}'] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_{i}'] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_{i}'] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_{i}'] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_{i}'] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_{i}'] = len(sig[np.abs(sig) > 500])
        X.loc[seg_id, 'sum_{i}'] = sig.sum()

        X.loc[seg_id, 'mean_change_rate_first_50000_{i}'] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_50000_{i}'] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_first_10000_{i}'] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_10000_{i}'] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)
        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)
        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)
        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_{i}'] = sig.mad()
        X.loc[seg_id, 'kurt_{i}'] = sig.kurtosis()
        X.loc[seg_id, 'skew_{i}'] = sig.skew()
        X.loc[seg_id, 'med_{i}'] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_{i}'] = np.abs(hilbert(sig)).mean()

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = sta_lta_ratio(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = sta_lta_ratio(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = sta_lta_ratio(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = sta_lta_ratio(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)
        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)


def compute_standard_features_block(xc, seg_id, X, fs, prefix=''):
    
    # Generic stats
    X.loc[seg_id, prefix + 'mean'] = xc.mean()
    X.loc[seg_id, prefix + 'std'] = xc.std()
    X.loc[seg_id, prefix + 'max'] = xc.max()
    X.loc[seg_id, prefix + 'min'] = xc.min()
    X.loc[seg_id, prefix + 'hmean'] = stats.hmean(np.abs(xc[np.nonzero(xc)[0]]))
    X.loc[seg_id, prefix + 'gmean'] = stats.gmean(np.abs(xc[np.nonzero(xc)[0]])) 
    X.loc[seg_id, prefix + 'mad'] = xc.mad()
    X.loc[seg_id, prefix + 'kurt'] = xc.kurtosis()
    X.loc[seg_id, prefix + 'skew'] = xc.skew()
    X.loc[seg_id, prefix + 'med'] = xc.median()

    for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
        X.loc[seg_id, prefix + f'percentile_{p}'] = np.percentile(xc, p)
        X.loc[seg_id, prefix + f'abs_percentile_{p}'] = np.percentile(np.abs(xc), p)

    X.loc[seg_id, prefix + 'num_crossing_0'] = feature_calculators.number_crossing_m(xc, 0)

    for p in [95,99]:
        X.loc[seg_id, prefix + f'binned_entropy_{p}'] = feature_calculators.binned_entropy(xc, p)

    # Andrew stats
    X.loc[seg_id, prefix + 'mean_diff'] = np.mean(np.diff(xc))
    X.loc[seg_id, prefix + 'mean_abs_diff'] = np.mean(np.abs(np.diff(xc)))
    X.loc[seg_id, prefix + 'mean_change_rate'] = change_rate(xc, method='original')
    X.loc[seg_id, prefix + 'mean_change_rate_v2'] = change_rate(xc, method='modified')
    X.loc[seg_id, prefix + 'abs_max'] = np.abs(xc).max()
    X.loc[seg_id, prefix + 'abs_min'] = np.abs(xc).min()
    X.loc[seg_id, prefix + 'mean_change_abs'] = np.mean(np.diff(xc))

    # Classical stats by segment
    for agg_type, slice_length, direction in product(['std', 'min', 'max', 'mean'], [1000, 10000, 50000], ['first', 'last']):
        if direction == 'first':
            X.loc[seg_id, prefix + f'{agg_type}_{direction}_{slice_length}'] = xc[:slice_length].agg(agg_type)
        elif direction == 'last':
            X.loc[seg_id, prefix + f'{agg_type}_{direction}_{slice_length}'] = xc[-slice_length:].agg(agg_type)

    X.loc[seg_id, prefix + 'avg_first_50000'] = xc[:50000].mean()
    X.loc[seg_id, prefix + 'avg_last_50000'] = xc[-50000:].mean()
    X.loc[seg_id, prefix + 'avg_first_10000'] = xc[:10000].mean()
    X.loc[seg_id, prefix + 'avg_last_10000'] = xc[-10000:].mean()

    # k-statistic and moments
    for i in range(1, 5):
        X.loc[seg_id, prefix + f'kstat_{i}'] = stats.kstat(xc, i)
        X.loc[seg_id, prefix + f'moment_{i}'] = stats.moment(xc, i)

    for i in [1, 2]:
        X.loc[seg_id, prefix + f'kstatvar_{i}'] = stats.kstatvar(xc, i)

    X.loc[seg_id, prefix + 'range_minf_m4000'] = feature_calculators.range_count(xc, -np.inf, -4000)
    X.loc[seg_id, prefix + 'range_p4000_pinf'] = feature_calculators.range_count(xc, 4000, np.inf)
    for i, j in zip(borders, borders[1:]):
        X.loc[seg_id, prefix + f'range_{i}_{j}'] = feature_calculators.range_count(xc, i, j)
        X.loc[seg_id, prefix + 'ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(xc)

    X.loc[seg_id, prefix + 'max_to_min'] = xc.max() / np.abs(xc.min())
    X.loc[seg_id, prefix + 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
    X.loc[seg_id, prefix + 'count_big'] = len(xc[np.abs(xc) > 500])
    X.loc[seg_id, prefix + 'sum'] = xc.sum()

    # calc_change_rate on slices of data
    for slice_length, direction in product([1000, 10000, 50000], ['first', 'last']):
        if direction == 'first':
            X.loc[seg_id, prefix + f'mean_change_rate_{direction}_{slice_length}'] = change_rate(xc[:slice_length], method='original')
            X.loc[seg_id, prefix + f'mean_change_rate_{direction}_{slice_length}_v2'] = change_rate(xc[:slice_length], method='modified')
        elif direction == 'last':
            X.loc[seg_id, prefix + f'mean_change_rate_{direction}_{slice_length}'] = change_rate(xc[-slice_length:], method='original')
            X.loc[seg_id, prefix + f'mean_change_rate_{direction}_{slice_length}_v2'] = change_rate(xc[-slice_length:], method='modified')

    X.loc[seg_id, prefix + 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, prefix + 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, prefix + 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, prefix + 'q01'] = np.quantile(xc, 0.01)

    X.loc[seg_id, prefix + 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
    X.loc[seg_id, prefix + 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
    X.loc[seg_id, prefix + 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
    X.loc[seg_id, prefix + 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    X.loc[seg_id, prefix + 'trend'] = add_trend_feature(xc)
    X.loc[seg_id, prefix + 'abs_trend'] = add_trend_feature(xc, abs_values=True)
    X.loc[seg_id, prefix + 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, prefix + 'abs_std'] = np.abs(xc).std()

    X.loc[seg_id, prefix + 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
    X.loc[seg_id, prefix + 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
    for hw in [50, 150, 1500, 15000]:
        X.loc[seg_id, prefix + f'Hann_window_mean_{hw}'] = (convolve(xc, hann(hw), mode='same') / sum(hann(hw))).mean()

    sta_lta_method = 'original'
    classic_sta_lta1 = sta_lta_ratio(xc, 500, 10000, method=sta_lta_method)
    classic_sta_lta2 = sta_lta_ratio(xc, 5000, 100000, method=sta_lta_method)
    classic_sta_lta3 = sta_lta_ratio(xc, 3333, 6666, method=sta_lta_method)
    classic_sta_lta4 = sta_lta_ratio(xc, 10000, 25000, method=sta_lta_method)
    classic_sta_lta5 = sta_lta_ratio(xc, 50, 1000, method=sta_lta_method)
    classic_sta_lta6 = sta_lta_ratio(xc, 100, 5000, method=sta_lta_method)
    classic_sta_lta7 = sta_lta_ratio(xc, 333, 666, method=sta_lta_method)
    classic_sta_lta8 = sta_lta_ratio(xc, 4000, 10000, method=sta_lta_method)

    X.loc[seg_id, prefix + 'classic_sta_lta1_mean'] = classic_sta_lta1.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta2_mean'] = classic_sta_lta2.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta3_mean'] = classic_sta_lta3.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta4_mean'] = classic_sta_lta4.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta5_mean'] = classic_sta_lta5.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta6_mean'] = classic_sta_lta6.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta7_mean'] = classic_sta_lta7.mean()
    X.loc[seg_id, prefix + 'classic_sta_lta8_mean'] = classic_sta_lta8.mean()

    X.loc[seg_id, prefix + 'classic_sta_lta1_q95'] = np.quantile(classic_sta_lta1, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta2_q95'] = np.quantile(classic_sta_lta2, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta3_q95'] = np.quantile(classic_sta_lta3, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta4_q95'] = np.quantile(classic_sta_lta4, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta5_q95'] = np.quantile(classic_sta_lta5, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta6_q95'] = np.quantile(classic_sta_lta6, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta7_q95'] = np.quantile(classic_sta_lta7, 0.95)
    X.loc[seg_id, prefix + 'classic_sta_lta8_q95'] = np.quantile(classic_sta_lta8, 0.95)

    X.loc[seg_id, prefix + 'classic_sta_lta1_q05'] = np.quantile(classic_sta_lta1, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta2_q05'] = np.quantile(classic_sta_lta2, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta3_q05'] = np.quantile(classic_sta_lta3, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta4_q05'] = np.quantile(classic_sta_lta4, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta5_q05'] = np.quantile(classic_sta_lta5, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta6_q05'] = np.quantile(classic_sta_lta6, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta7_q05'] = np.quantile(classic_sta_lta7, 0.05)
    X.loc[seg_id, prefix + 'classic_sta_lta8_q05'] = np.quantile(classic_sta_lta8, 0.05)

    sta_lta_method = 'modified'
    classic_sta_lta1 = sta_lta_ratio(xc, 500, 10000, method=sta_lta_method)
    classic_sta_lta2 = sta_lta_ratio(xc, 5000, 100000, method=sta_lta_method)
    classic_sta_lta3 = sta_lta_ratio(xc, 3333, 6666, method=sta_lta_method)
    classic_sta_lta4 = sta_lta_ratio(xc, 10000, 25000, method=sta_lta_method)
    classic_sta_lta5 = sta_lta_ratio(xc, 50, 1000, method=sta_lta_method)
    classic_sta_lta6 = sta_lta_ratio(xc, 100, 5000, method=sta_lta_method)
    classic_sta_lta7 = sta_lta_ratio(xc, 333, 666, method=sta_lta_method)
    classic_sta_lta8 = sta_lta_ratio(xc, 4000, 10000, method=sta_lta_method)

    X.loc[seg_id, prefix + 'modified_sta_lta1_mean'] = classic_sta_lta1.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta2_mean'] = classic_sta_lta2.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta3_mean'] = classic_sta_lta3.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta4_mean'] = classic_sta_lta4.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta5_mean'] = classic_sta_lta5.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta6_mean'] = classic_sta_lta6.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta7_mean'] = classic_sta_lta7.mean()
    X.loc[seg_id, prefix + 'modified_sta_lta8_mean'] = classic_sta_lta8.mean()

    X.loc[seg_id, prefix + 'modified_sta_lta1_q95'] = np.quantile(classic_sta_lta1, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta2_q95'] = np.quantile(classic_sta_lta2, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta3_q95'] = np.quantile(classic_sta_lta3, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta4_q95'] = np.quantile(classic_sta_lta4, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta5_q95'] = np.quantile(classic_sta_lta5, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta6_q95'] = np.quantile(classic_sta_lta6, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta7_q95'] = np.quantile(classic_sta_lta7, 0.95)
    X.loc[seg_id, prefix + 'modified_sta_lta8_q95'] = np.quantile(classic_sta_lta8, 0.95)

    X.loc[seg_id, prefix + 'modified_sta_lta1_q05'] = np.quantile(classic_sta_lta1, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta2_q05'] = np.quantile(classic_sta_lta2, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta3_q05'] = np.quantile(classic_sta_lta3, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta4_q05'] = np.quantile(classic_sta_lta4, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta5_q05'] = np.quantile(classic_sta_lta5, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta6_q05'] = np.quantile(classic_sta_lta6, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta7_q05'] = np.quantile(classic_sta_lta7, 0.05)
    X.loc[seg_id, prefix + 'modified_sta_lta8_q05'] = np.quantile(classic_sta_lta8, 0.05)

    X.loc[seg_id, prefix + 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'Moving_average_30000_mean'] = xc.rolling(window=30000).mean().mean(skipna=True)

    ewma = pd.Series.ewm
    X.loc[seg_id, prefix + 'exp_Moving_average_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_6000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_30000_mean'] = ewma(xc, span=30000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_50000_mean'] = ewma(xc, span=50000).mean().mean(skipna=True)

    X.loc[seg_id, prefix + 'exp_Moving_average_300_std'] = ewma(xc, span=300).mean().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_3000_std'] = ewma(xc, span=3000).mean().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_6000_std'] = ewma(xc, span=6000).mean().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_30000_std'] = ewma(xc, span=30000).mean().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_average_50000_std'] = ewma(xc, span=50000).mean().std(skipna=True)

    X.loc[seg_id, prefix + 'exp_Moving_std_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_6000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_30000_mean'] = ewma(xc, span=30000).mean().mean(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_50000_mean'] = ewma(xc, span=50000).mean().mean(skipna=True)
    
    X.loc[seg_id, prefix + 'exp_Moving_std_300_std'] = ewma(xc, span=300).std().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_3000_std'] = ewma(xc, span=3000).std().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_6000_std'] = ewma(xc, span=6000).std().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_30000_std'] = ewma(xc, span=30000).std().std(skipna=True)
    X.loc[seg_id, prefix + 'exp_Moving_std_50000_std'] = ewma(xc, span=50000).std().std(skipna=True)

    no_of_std = 2
    X.loc[seg_id, prefix + 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    X.loc[seg_id, prefix + 'MA_700MA_BB_high_mean'] = (X.loc[seg_id, prefix + 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, prefix + 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, prefix + 'MA_700MA_BB_low_mean'] = (X.loc[seg_id, prefix + 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, prefix + 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, prefix + 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    X.loc[seg_id, prefix + 'MA_400MA_BB_high_mean'] = (X.loc[seg_id, prefix + 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, prefix + 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, prefix + 'MA_400MA_BB_low_mean'] = (X.loc[seg_id, prefix + 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, prefix + 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, prefix + 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    X.loc[seg_id, prefix + 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    X.loc[seg_id, prefix + 'iqr1'] = np.subtract(*np.percentile(xc, [95, 5]))

    X.loc[seg_id, prefix + 'q999'] = np.quantile(xc, 0.999)
    X.loc[seg_id, prefix + 'q001'] = np.quantile(xc, 0.001)
    X.loc[seg_id, prefix + 'ave10'] = stats.trim_mean(xc, 0.1)

    X.loc[seg_id, prefix + 'freq_cross_first_50000'] = freq_from_crossings(xc.values[:50000], fs)
    X.loc[seg_id, prefix + 'freq_cross_last_50000'] = freq_from_crossings(xc.values[-50000:], fs)
    X.loc[seg_id, prefix + 'freq_cross_first_10000'] = freq_from_crossings(xc.values[:10000], fs)
    X.loc[seg_id, prefix + 'freq_cross_last_10000'] = freq_from_crossings(xc.values[-10000:], fs)

    for peak in [10, 20, 50, 100]:
        X.loc[seg_id, prefix + f'num_peaks_{peak}'] = feature_calculators.number_peaks(xc, peak)

    for c in [1, 5, 10, 50, 100]:
        X.loc[seg_id, prefix + f'spkt_welch_density_{c}'] = list(feature_calculators.spkt_welch_density(xc, [{'coeff': c}]))[0][1]
        X.loc[seg_id, prefix + f'time_rev_asym_stat_{c}'] = feature_calculators.time_reversal_asymmetry_statistic(xc, c) 

    for autocorr_lag in [5, 10, 50, 100, 500, 1000, 5000, 10000]:
        X.loc[seg_id, prefix + f'autocorrelation_{autocorr_lag}'] = feature_calculators.autocorrelation(xc, autocorr_lag)
        X.loc[seg_id, prefix + f'c3_{autocorr_lag}'] = feature_calculators.c3(xc, autocorr_lag)

    for windows in [10, 50, 100, 500, 1000, 10000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
            X.loc[seg_id, prefix + f'percentile_roll_std_{p}_window_{windows}'] = np.percentile(x_roll_std, p)
            X.loc[seg_id, prefix + f'percentile_roll_mean_{p}_window_{windows}'] = np.percentile(x_roll_mean, p)

        X.loc[seg_id, prefix + 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, prefix + 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, prefix + 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, prefix + 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, prefix + 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, prefix + 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, prefix + 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, prefix + 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, prefix + 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.abs(np.diff(x_roll_std)))
        X.loc[seg_id, prefix + 'av_change_rate_roll_std_' + str(windows)] = change_rate(pd.Series(x_roll_std), method='original')
        X.loc[seg_id, prefix + 'av_change_rate_roll_std_' + str(windows) + 'v2'] = change_rate(pd.Series(x_roll_std), method='modified')
        X.loc[seg_id, prefix + 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        X.loc[seg_id, prefix + 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, prefix + 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, prefix + 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, prefix + 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, prefix + 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, prefix + 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, prefix + 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, prefix + 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, prefix + 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.abs(np.diff(x_roll_mean)))
        X.loc[seg_id, prefix + 'av_change_rate_roll_mean_' + str(windows)] = change_rate(pd.Series(x_roll_mean), method='original')
        X.loc[seg_id, prefix + 'av_change_rate_roll_mean_' + str(windows) + '_v2'] = change_rate(pd.Series(x_roll_mean), method='modified')
        X.loc[seg_id, prefix + 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
        X.loc[seg_id, prefix + f'percentile_roll_std_{p}'] = X.loc[seg_id, prefix + f'percentile_roll_std_{p}_window_10000']
        X.loc[seg_id, prefix + f'percentile_roll_mean_{p}'] = X.loc[seg_id, prefix + f'percentile_roll_mean_{p}_window_10000']
        
        
def compute_fft_features_block(xc, seg_id, X):
    
    xcdm = xc - np.mean(xc)
    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)
    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT stats
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    magFFT = np.abs(zc)
    phzFFT = np.angle(zc)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    X.loc[seg_id, 'FFT_Rmean_last_5000'] = realFFT[-5000:].mean()
    X.loc[seg_id, 'FFT_Rstd_last_5000'] = realFFT[-5000:].std()
    X.loc[seg_id, 'FFT_Rmax_last_5000'] = realFFT[-5000:].max()
    X.loc[seg_id, 'FFT_Rmin_last_5000'] = realFFT[-5000:].min()

    X.loc[seg_id, 'FFT_Rmean_last_15000'] = realFFT[-15000:].mean()
    X.loc[seg_id, 'FFT_Rstd_last_15000'] = realFFT[-15000:].std()
    X.loc[seg_id, 'FFT_Rmax_last_15000'] = realFFT[-15000:].max()
    X.loc[seg_id, 'FFT_Rmin_last_15000'] = realFFT[-15000:].min()

    for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
        X.loc[seg_id, f'fft_{coeff}_{attr}'] = list(feature_calculators.fft_coefficient(xc, [{'coeff': coeff, 'attr': attr}]))[0][1]

def create_all_features_extended(seg_id, seg, X, fs):

    xc = pd.Series(seg['acoustic_data'].values.astype('float64'))

    compute_fft_features_block(xc, seg_id, X)
    compute_bp_features_block(xc, seg_id, X)
    compute_standard_features_block(xc, seg_id, X, fs)

    x = pd.Series(xc)
    zc = np.fft.fft(x)
    realFFT = pd.Series(np.real(zc))
    imagFFT = pd.Series(np.imag(zc))
    compute_standard_features_block(realFFT, seg_id, X, fs, prefix='fftr_')
    compute_standard_features_block(imagFFT, seg_id, X, fs, prefix='ffti_')

def create_all_features_extended_denoised(seg_id, seg, X, fs):

    xc_noisy = pd.Series(seg['acoustic_data'].values)
    xc = denoise_signal(high_pass_filter(xc_noisy, low_cutoff=10000, SAMPLE_RATE=fs), wavelet='haar', level=1)

    xc = pd.Series(xc)
    
    compute_fft_features_block(xc, seg_id, X)
    compute_bp_features_block(xc, seg_id, X)
    compute_standard_features_block(xc, seg_id, X, fs)

    x = pd.Series(xc)
    zc = np.fft.fft(x)
    realFFT = pd.Series(np.real(zc))
    imagFFT = pd.Series(np.imag(zc))
    compute_standard_features_block(realFFT, seg_id, X, fs, prefix='fftr_')
    compute_standard_features_block(imagFFT, seg_id, X, fs, prefix='ffti_')
