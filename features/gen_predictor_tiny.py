import numpy as np
from scipy.special import erf

def activation_func(x, activation=None):
    if activation == 'tanh':
        return np.tanh(x)
    elif activation == 'abs':
        return x/(1+np.abs(x))
    elif activation=='sqrt':
        return x/np.sqrt(1+(x)**2)
    elif activation=='atan':
        return (2/np.pi)*np.arctan((np.pi/2)*x) 
    elif activation=='erf':
        return erf((np.sqrt(np.pi)/2)*x)
    else:
        print('Activation not defined')
        return


def gpi_tiny(data, activation=None):

    def activation_func_local(x):
        return activation_func(x, activation=activation)

    return (5.683668 +
            1.0*activation_func_local(((((data["abs_percentile_50"]) - (((data["percentile_roll_std_75_window_500"]) * 2.0)))) - (((((((((((data["percentile_roll_std_75_window_500"]) + (data["num_peaks_10"]))) * 2.0)) + (data["abs_percentile_50"]))) * 2.0)) * 2.0)))) +
            1.0*activation_func_local(((-1.0) - (((((data["percentile_roll_std_30_window_1000"]) + ((((data["binned_entropy_95"]) + (data["num_peaks_10"]))/2.0)))) + (data["percentile_roll_std_75_window_500"]))))) +
            1.0*activation_func_local(((((((((data["abs_percentile_50"]) + (data["binned_entropy_99"]))) * ((((data["percentile_roll_std_75_window_500"]) + (data["num_peaks_10"]))/2.0)))) * ((-1.0*((((data["percentile_roll_std_75_window_500"]) + (data["percentile_roll_std_30_window_1000"])))))))) * 2.0)) +
            1.0*activation_func_local((((-1.0*((((((data["fftr_exp_Moving_std_3000_mean"]) * (((1.0) + (((data["percentile_roll_std_30_window_1000"]) * 2.0)))))) / 2.0))))) * ((((data["num_peaks_10"]) + (data["abs_percentile_50"]))/2.0)))) +
            1.0*activation_func_local(((((data["exp_Moving_std_3000_mean"]) - (data["percentile_roll_std_75_window_500"]))) + (((data["num_peaks_10"]) * (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_exp_Moving_std_3000_mean"]))))))))

def gpii_tiny(data, activation=None):

    def activation_func_local(x):
        return activation_func(x, activation=activation)

    return (5.683668 +
            1.0*activation_func_local((((((((((-1.0*((((data["num_peaks_10"]) + (data["percentile_roll_std_75_window_500"])))))) * 2.0)) - (((data["num_peaks_10"]) + (data["abs_percentile_50"]))))) * 2.0)) * 2.0)) +
            1.0*activation_func_local((((((((data["exp_Moving_std_3000_mean"]) + (-2.0))) + (((((((data["percentile_roll_std_80_window_500"]) + (data["autocorrelation_5"]))) / 2.0)) / 2.0)))/2.0)) - (((((data["percentile_roll_std_75_window_500"]) * 2.0)) * 2.0)))) +
            1.0*activation_func_local((((-1.0*((data["fftr_exp_Moving_std_3000_mean"])))) * (((((data["num_peaks_10"]) + (data["abs_percentile_50"]))) * (((((data["percentile_roll_std_30_window_1000"]) * 2.0)) + (((1.0) + (data["percentile_roll_std_30_window_1000"]))))))))) +
            1.0*activation_func_local(((data["exp_Moving_std_3000_mean"]) - (((data["percentile_roll_std_75_window_500"]) + ((((((((data["abs_percentile_50"]) + (data["percentile_roll_std_75_window_500"]))/2.0)) * ((((data["abs_percentile_50"]) + (data["percentile_roll_std_75_window_500"]))/2.0)))) * (data["fftr_exp_Moving_std_3000_mean"]))))))) +
            1.0*activation_func_local((((((((((data["percentile_roll_std_80_window_500"]) - (data["fftr_exp_Moving_std_3000_mean"]))) + (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_classic_sta_lta2_mean"]))))/2.0)) + (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_exp_Moving_std_3000_mean"]))))) * (data["num_peaks_10"]))))