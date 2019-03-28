import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import hilbert, hann, convolve
from utils.frequency_estimation import freq_from_crossings

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
    X.loc[seg_id, 'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean']
                                              + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean']
                                             - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    X.loc[seg_id, 'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean']
                                              + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean']
                                             - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
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
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = change_rate(pd.Series(x_roll_std), 
                                                                               method='original')
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows) + 'v2'] = change_rate(pd.Series(x_roll_std),
                                                                                      method='modified')
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
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = change_rate(pd.Series(x_roll_mean),
                                                                               method = 'original')
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows) + '_v2'] = change_rate(pd.Series(x_roll_mean),
                                                                                       method = 'modified')
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()