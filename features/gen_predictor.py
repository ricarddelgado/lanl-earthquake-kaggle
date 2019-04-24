import numpy as np

def gpi(data):
    return (5.591662 +
            0.0399999991*np.tanh(((data["mean_change_rate_first_10000"]) - (((73.0) * ((((((7.0) * (((((((((data["q95"]) + (((0.3183098733) + (data["q05_roll_std_100"]))))) * 2.0)) * 2.0)) + (((data["q95_roll_std_10"]) + (data["iqr"]))))))) + (data["iqr"]))/2.0)))))) +
            0.0399999991*np.tanh(((((3.6923100948) + (data["Moving_average_6000_mean"]))) * (((73.0) * (((data["MA_400MA_BB_low_mean"]) - (((((data["Moving_average_6000_mean"]) * (data["q05_roll_std_10"]))) + (((data["iqr"]) + (((data["q05_roll_std_10"]) * (((3.6923100948) * 2.0)))))))))))))) +
            0.0399999991*np.tanh(((((((((((((((((((((((data["q05"]) * 2.0)) * 2.0)) - (data["iqr"]))) - (data["max_roll_std_1000"]))) * 2.0)) * 2.0)) * 2.0)) - (data["iqr"]))) - (3.6923100948))) - (((data["q05_roll_std_10"]) * (73.0))))) * 2.0)) +
            0.0399999991*np.tanh(((((((((((((((((((data["q05_roll_std_10"]) * (((((-3.0) * 2.0)) * 2.0)))) - (((data["iqr"]) + (data["max_roll_std_1000"]))))) * 2.0)) * 2.0)) * 2.0)) - (data["max_roll_mean_100"]))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((73.0) * ((((data["max_first_50000"]) + ((((data["max_first_50000"]) + (((73.0) * (((data["q05"]) - (((data["q05_roll_std_100"]) + (((0.5555559993) + (((data["mad"]) + (data["q95"]))))))))))))/2.0)))/2.0)))) +
            0.0399999991*np.tanh(((((((data["q05_roll_std_10"]) * (73.0))) * 2.0)) - ((((((((data["q05_roll_std_10"]) * (73.0))) + (((((((((data["iqr"]) + (((data["MA_1000MA_std_mean"]) + (data["abs_max_roll_mean_10"]))))) * 2.0)) * 2.0)) * 2.0)))/2.0)) * (73.0))))) +
            0.0399999991*np.tanh(((((data["mean_diff"]) - (((((data["q95"]) + (((data["q95"]) + (((((data["q05_roll_std_100"]) / 2.0)) + (((((((0.3183098733) * 2.0)) - (data["q05"]))) - (data["q05"]))))))))) * (73.0))))) * 2.0)) +
            0.0399999991*np.tanh((-1.0*(((((((7.48210620880126953)) * 2.0)) + (((((data["iqr"]) + ((((-1.0*((data["avg_first_10000"])))) * 2.0)))) + (((73.0) * (((data["iqr"]) + (((((data["q05_roll_std_10"]) * 2.0)) * 2.0))))))))))))) +
            0.0399999991*np.tanh((((((data["q01_roll_std_10"]) + (((((data["q05_roll_mean_10"]) - (((((data["q05_roll_std_100"]) - (((data["q05"]) - (0.3026320040))))) + (((((((0.5434780121) + (0.3026320040))/2.0)) + (data["iqr"]))/2.0)))))) * (73.0))))/2.0)) * (73.0))) +
            0.0399999991*np.tanh(((((data["trend"]) + (((data["q95_roll_std_10"]) + (data["trend"]))))) - (((73.0) * (((((((data["ave_roll_std_10"]) + (((((((data["q95_roll_mean_10"]) + (data["q05_roll_std_1000"]))/2.0)) + (data["q05_roll_std_1000"]))/2.0)))/2.0)) + ((((0.5434780121) + (data["q95_roll_mean_10"]))/2.0)))/2.0)))))) +
            0.0399999991*np.tanh(((73.0) - (((((73.0) * 2.0)) * (((73.0) * ((((((((((data["q05_roll_std_1000"]) / 2.0)) + (data["iqr"]))/2.0)) + (data["q05_roll_std_10"]))) + (((0.2183910012) + (data["q95"]))))))))))) +
            0.0399999991*np.tanh(((73.0) * (((73.0) * (((((73.0) * (((data["q05_roll_mean_10"]) - (((((data["q05_roll_std_100"]) + (data["q95"]))) + (0.5434780121))))))) + (data["q05_roll_std_100"]))))))) +
            0.0399999991*np.tanh((-1.0*((((73.0) * (((73.0) * (((73.0) * (((data["q05_roll_std_100"]) + ((((data["q95_roll_mean_10"]) + ((((data["Moving_average_3000_mean"]) + (((((data["iqr"]) + (0.9756100178))) + (data["abs_mean"]))))/2.0)))/2.0))))))))))))) +
            0.0399999991*np.tanh(((73.0) * (((((data["classic_sta_lta3_mean"]) + (data["mad"]))) + (((((((73.0) * (((data["q05_roll_mean_10"]) - (((data["q05_roll_std_100"]) + (((data["mad"]) + (0.5555559993))))))))) - (data["iqr"]))) - (data["iqr"]))))))) +
            0.0399999991*np.tanh(((((np.tanh((data["q95_roll_std_10"]))) + (((((((data["q05_roll_std_10"]) + ((((((0.2439019978) + (data["ave10"]))/2.0)) / 2.0)))) * 2.0)) * 2.0)))) * (((data["q05_roll_std_10"]) - (((73.0) * (((73.0) * (73.0))))))))) +
            0.0399999991*np.tanh(((3.6923100948) * (((73.0) * (((3.6923100948) * ((((((-1.0*((data["q99_roll_mean_1000"])))) * (data["avg_last_50000"]))) - (((data["q05_roll_std_10"]) * (((3.6923100948) + (((3.6923100948) + (data["abs_max_roll_mean_100"]))))))))))))))) +
            0.0399999991*np.tanh(((((((data["Moving_average_1500_mean"]) + (((((((((data["MA_400MA_BB_low_mean"]) + ((((-1.0*((data["q05_roll_std_1000"])))) - (data["iqr"]))))) * (73.0))) - (data["min_roll_std_1000"]))) - (data["q05_roll_std_1000"]))))) * (73.0))) * (73.0))) +
            0.0399999991*np.tanh(((73.0) - (((73.0) * (((73.0) * (((data["q05_roll_std_10"]) + (np.tanh(((((((data["exp_Moving_average_3000_mean"]) + (((data["iqr"]) + (np.tanh((((data["q05_roll_std_100"]) + (data["exp_Moving_average_6000_mean"]))))))))/2.0)) / 2.0)))))))))))) +
            0.0399999991*np.tanh(((73.0) * (((73.0) * (((data["q05_roll_mean_10"]) - (((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((data["q05_roll_std_100"]))))))))))) + ((((data["q95_roll_std_10"]) + (np.tanh((2.8437500000))))/2.0)))))))))) +
            0.0399999991*np.tanh(((((((data["iqr"]) / 2.0)) + (((data["mean"]) + (((data["q05_roll_std_100"]) + (((data["q05_roll_std_10"]) * 2.0)))))))) * ((-1.0*((((73.0) - (((((((((data["mean"]) * 2.0)) * 2.0)) * 2.0)) * 2.0))))))))) +
            0.0399999991*np.tanh(((((((((0.5217390060) - (((((data["q05_roll_std_10"]) * ((-1.0*((7.0)))))) + (data["exp_Moving_average_3000_mean"]))))) * ((-1.0*((73.0)))))) - (((data["classic_sta_lta4_mean"]) * (7.0))))) - ((((7.0) + (data["classic_sta_lta4_mean"]))/2.0)))) +
            0.0399999991*np.tanh(((((((((((((((((((np.tanh(((-1.0*((((data["q05_roll_mean_100"]) + (0.2183910012)))))))) - ((((data["q05_roll_std_100"]) + (((data["q05_roll_std_10"]) * 2.0)))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["exp_Moving_average_6000_mean"]))) * 2.0)) +
            0.0399999991*np.tanh(((data["trend"]) - (((73.0) * ((((3.6923100948) + ((((9.0)) * (((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((data["q05_roll_std_100"]))))))))))))))) - (data["q05_roll_mean_10"]))))))/2.0)))))) +
            0.0399999991*np.tanh(((data["q05_roll_std_1000"]) - (((73.0) * (((data["avg_last_50000"]) + (((((73.0) * 2.0)) * (((data["ave_roll_std_100"]) + (((data["Moving_average_6000_mean"]) + (((data["iqr"]) + (((data["q05_roll_std_1000"]) + (((data["q95"]) * 2.0)))))))))))))))))) +
            0.0399999991*np.tanh(((((data["q05_roll_mean_10"]) - (2.0769200325))) - ((((((6.08570814132690430)) * (((2.0769200325) + (((((((((data["q05_roll_std_10"]) - (data["q05_roll_mean_10"]))) * 2.0)) * 2.0)) * 2.0)))))) + (((((data["av_change_rate_roll_std_100"]) - (data["avg_last_10000"]))) * 2.0)))))) +
            0.0399999991*np.tanh((-1.0*((((((((((((((((((((((((((((((((data["q05_roll_std_1000"]) + (data["q05_roll_mean_100"]))/2.0)) * 2.0)) + (data["iqr"]))/2.0)) + (data["q95_roll_mean_10"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) +
            0.0399999991*np.tanh((((-1.0*((((((((((data["q05_roll_std_100"]) - (np.tanh(((-1.0*((((data["MA_400MA_std_mean"]) + (((data["exp_Moving_average_300_mean"]) * (((data["q05_roll_std_100"]) * 2.0))))))))))))) * 2.0)) * 2.0)) + (np.tanh((data["q05_roll_std_100"])))))))) * (73.0))) +
            0.0399999991*np.tanh(((((data["abs_q05"]) - ((((((2.0769200325) * ((((((73.0) * ((((((0.5555559993) + (((data["q05_roll_std_100"]) + (0.7446810007))))/2.0)) + (data["q05_roll_std_100"]))))) + (0.5555559993))/2.0)))) + (0.5555559993))/2.0)))) * 2.0)) +
            0.0399999991*np.tanh((((10.75933361053466797)) - (((7.0) * (((7.0) * (((7.0) * ((((((data["q95_roll_mean_10"]) + (data["q05_roll_std_100"]))) + ((((data["q95_roll_mean_10"]) + (((data["iqr"]) + (((data["Moving_average_700_mean"]) * 2.0)))))/2.0)))/2.0)))))))))) +
            0.0399999991*np.tanh(((73.0) * (((((data["q05_roll_std_1000"]) + ((((data["q05_roll_std_10"]) + ((((((data["q05_roll_std_1000"]) + (np.tanh((data["q05_roll_std_1000"]))))/2.0)) - (((data["q95_roll_mean_1000"]) - (np.tanh((73.0))))))))/2.0)))) * (((2.8437500000) - (73.0))))))) +
            0.0399999991*np.tanh(((((((((-3.0) + (data["Moving_average_3000_mean"]))) - (((73.0) * (((data["ave_roll_mean_10"]) - ((((((data["q05_roll_std_100"]) + (data["q95_roll_mean_10"]))/2.0)) * (((-3.0) - (((data["Moving_average_700_mean"]) + (data["q95_roll_mean_10"]))))))))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((data["classic_sta_lta3_mean"]) * 2.0)) + (data["q05_roll_mean_100"]))) - (((73.0) * (((((((0.6582279801) + (data["q05_roll_std_10"]))) - (((data["q05_roll_mean_10"]) * 2.0)))) + ((((-1.0*((data["abs_q05"])))) + (((data["q05_roll_std_100"]) * 2.0)))))))))) +
            0.0399999991*np.tanh(((((data["skew"]) + (((data["kurt"]) + (data["med"]))))) - ((((7.02540111541748047)) * (((((data["iqr"]) + (((data["q95_roll_std_10"]) + (((((data["q95_roll_mean_10"]) + (((data["iqr"]) * (data["Moving_average_1500_mean"]))))) * 2.0)))))) * 2.0)))))) +
            0.0399999991*np.tanh(((((data["q05_roll_std_100"]) - (((-2.0) + (((((data["iqr"]) * (-2.0))) - ((-1.0*((data["av_change_rate_roll_std_10"])))))))))) - ((((((data["q05_roll_std_100"]) + (np.tanh((((data["q05_roll_std_100"]) * (data["ave_roll_mean_10"]))))))/2.0)) * (73.0))))) +
            0.0399999991*np.tanh(((((((((((data["q05_roll_std_10"]) + (((data["av_change_abs_roll_std_1000"]) - (((data["av_change_abs_roll_mean_10"]) * 2.0)))))) * 2.0)) + (data["av_change_abs_roll_std_1000"]))) * 2.0)) - (((73.0) * (((((data["abs_max_roll_mean_1000"]) + (((0.5555559993) + (data["q05_roll_std_10"]))))) * 2.0)))))) +
            0.0399999991*np.tanh((((((((((((((((((np.tanh((((((data["q95_roll_mean_10"]) + (data["q95_roll_std_100"]))) - (data["q05_roll_std_1000"]))))) + (data["q95_roll_std_100"]))/2.0)) + (data["q95_roll_std_100"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (((73.0) * (data["q95_roll_mean_10"]))))) * 2.0)) +
            0.0399999991*np.tanh(((73.0) - (((((((((((data["q05_roll_std_10"]) * 2.0)) - (data["ave10"]))) + (((data["q05_roll_std_100"]) + (0.9756100178))))) * (73.0))) * (((73.0) * 2.0)))))) +
            0.0399999991*np.tanh(((((data["Moving_average_6000_mean"]) + (((7.0) + (((data["ave10"]) * 2.0)))))) * (((((((((data["MA_700MA_BB_low_mean"]) * (data["avg_last_50000"]))) + ((-1.0*((((data["q95_roll_mean_10"]) * 2.0))))))) - (((data["ave_roll_mean_10"]) + (data["q05_roll_std_100"]))))) * 2.0)))) +
            0.0399999991*np.tanh(((73.0) * (((data["q05_roll_std_100"]) - (((((((data["iqr"]) - (np.tanh((((73.0) * (((data["ave10"]) - (((data["iqr"]) + (((((data["q05_roll_std_1000"]) + (0.5434780121))) * 2.0)))))))))))) * 2.0)) * 2.0)))))) +
            0.0399999991*np.tanh(((((((((((((((data["exp_Moving_average_6000_mean"]) * (((data["exp_Moving_average_6000_mean"]) + ((-1.0*((data["q01_roll_std_10"])))))))) - (((data["q05_roll_std_10"]) + (((data["q05_roll_std_10"]) + (((data["q95_roll_mean_100"]) + (data["q05_roll_std_10"]))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((73.0) - (((((data["q05_roll_std_100"]) * (73.0))) * (((73.0) * (((data["q05_roll_std_100"]) * (((data["q05_roll_std_100"]) - (((data["q05"]) - ((((0.2525250018) + (2.0))/2.0)))))))))))))) +
            0.0399999991*np.tanh(((((((((7.0) - ((((data["sum"]) + (((73.0) * (((((data["sum"]) + (data["q05_roll_std_10"]))) + (((((((np.tanh((data["abs_q05"]))) + (data["q05_roll_std_10"]))) * 2.0)) * 2.0)))))))/2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh((((((1.0)) - (((((((((data["q05_roll_std_100"]) + (np.tanh((((data["abs_q05"]) * (((((data["exp_Moving_average_6000_mean"]) - (data["abs_q05"]))) + (((data["q05_roll_std_100"]) + (((data["q95_roll_mean_10"]) * 2.0)))))))))))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +
            0.0399999991*np.tanh(((((np.tanh((((np.tanh((data["Moving_average_1500_mean"]))) * 2.0)))) - (((data["Moving_average_1500_mean"]) + ((-1.0*((((((data["q05_roll_std_100"]) + (np.tanh((0.6582279801))))) * 2.0))))))))) * ((((-1.0*((((data["Moving_average_1500_mean"]) * 2.0))))) - (73.0))))) +
            0.0399999991*np.tanh(((73.0) - ((((((((((((data["q05_roll_std_100"]) + (((data["q05_roll_std_100"]) * (data["Moving_average_6000_mean"]))))/2.0)) + (((data["Moving_average_6000_mean"]) + (((data["q05_roll_std_100"]) + (data["q95"]))))))) * 2.0)) * 2.0)) * (((73.0) + (data["Moving_average_6000_mean"]))))))) +
            0.0399999991*np.tanh(((((((((((((data["mean_change_rate_last_10000"]) - (data["mean_diff"]))) - (((((((((((((data["q05_roll_std_10"]) + (((0.5434780121) - (data["q05"]))))) * 2.0)) - (data["Hann_window_mean"]))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((-1.0*((((data["avg_last_50000"]) * (3.6923100948)))))) + (((73.0) / 2.0)))/2.0)) - (((73.0) * ((((((data["Moving_average_6000_mean"]) + (((data["q05_roll_std_10"]) - (((data["exp_Moving_average_6000_mean"]) * (data["exp_Moving_average_6000_mean"]))))))/2.0)) + (data["q05_roll_std_10"]))))))) +
            0.0399999991*np.tanh(((73.0) * ((-1.0*((((data["q01_roll_std_1000"]) + (((73.0) * (((data["q05_roll_std_100"]) + ((((1.4411799908) + ((((((data["q01_roll_std_1000"]) - (data["q99_roll_mean_1000"]))) + (((data["q99_roll_mean_1000"]) * (data["q99_roll_mean_1000"]))))/2.0)))/2.0))))))))))))) +
            0.0399999991*np.tanh((((12.50784111022949219)) + ((((12.50784111022949219)) - ((((((np.tanh((((data["q05_roll_std_10"]) * (data["q95_roll_mean_1000"]))))) + (((((((data["q05_roll_std_10"]) + (np.tanh((((data["q05_roll_std_10"]) * (data["exp_Moving_average_6000_mean"]))))))) * 2.0)) * 2.0)))/2.0)) * (73.0))))))) +
            0.0399999991*np.tanh(((((((data["skew"]) - (((data["ave_roll_mean_100"]) + (((((((data["ave_roll_mean_100"]) * (data["abs_q05"]))) + (((((2.0) + (((((data["q05_roll_std_100"]) - (data["q05_roll_mean_10"]))) * 2.0)))) * 2.0)))) * 2.0)))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["q95_roll_mean_1000"]) * (data["ave_roll_mean_100"]))) * 2.0)) - (((((data["exp_Moving_average_6000_mean"]) + (((((((((data["exp_Moving_average_6000_mean"]) * (data["q05_roll_std_10"]))) + (((((data["q05_roll_std_10"]) - (0.2439019978))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["q05_roll_std_10"]) - (((((data["q95"]) - (((((data["iqr"]) - (((data["exp_Moving_average_3000_mean"]) * (np.tanh((((data["q05_roll_std_10"]) * 2.0)))))))) - (((data["q05_roll_std_10"]) * 2.0)))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((data["q95_roll_mean_100"]) + (((data["mean_change_rate_first_10000"]) + (((data["abs_q05"]) + (data["abs_q05"]))))))) * (3.6923100948))) + (((((((((data["q05_roll_std_1000"]) + (0.7446810007))) * (((3.6923100948) - (73.0))))) * 2.0)) * 2.0)))) +
            0.0399999991*np.tanh((-1.0*((((((73.0) * ((((((data["q01_roll_mean_1000"]) - (((data["q05"]) * 2.0)))) + (((data["q05_roll_std_1000"]) + (((((data["q05_roll_std_100"]) - (data["q95_roll_mean_1000"]))) * (data["Moving_average_700_mean"]))))))/2.0)))) + (((((data["Moving_average_1500_mean"]) * 2.0)) * 2.0))))))) +
            0.0399999991*np.tanh(((((((((data["q05"]) - (((73.0) * ((((-1.0*((((data["q05"]) * 2.0))))) + (((data["q05_roll_std_1000"]) + (1.9019600153))))))))) + (((((((data["q05"]) + (data["max_to_min"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((data["av_change_abs_roll_mean_100"]) - (((data["q01_roll_mean_1000"]) * 2.0)))) - (((73.0) * (((((((data["Hann_window_mean"]) * (((data["q95"]) - (data["q95_roll_mean_1000"]))))) / 2.0)) + (((((data["q95"]) * 2.0)) + (data["q01_roll_mean_1000"]))))))))) * 2.0)) +
            0.0399999991*np.tanh(((((((((((data["skew"]) - (((((data["q05_roll_mean_10"]) + (((((data["iqr"]) + (((3.1415927410) * (((1.0) - (((data["q05_roll_mean_10"]) * 2.0)))))))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh((((((((6.0)) - (((((((((data["q05_roll_std_100"]) + (((data["q05_roll_std_100"]) * 2.0)))) * 2.0)) + (((((data["ave10"]) * (((((data["q05_roll_std_100"]) * 2.0)) * 2.0)))) + (np.tanh((data["q05_roll_std_100"]))))))) * 2.0)))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["q05_roll_std_10"]) + (((((((data["q95_roll_mean_1000"]) + (((((((data["skew"]) + (((((((data["abs_q05"]) - (((data["q05_roll_std_10"]) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["max_to_min"]) + (data["q95_roll_std_1000"]))) - (((((73.0) * ((((data["q05_roll_std_1000"]) + (((0.2183910012) + (data["abs_q05"]))))/2.0)))) * (((data["q05_roll_std_1000"]) * (((data["q05_roll_std_10"]) + (data["q01_roll_std_10"]))))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((((data["abs_q05"]) + (data["q05_roll_std_100"]))) - (((7.0) * (((data["iqr"]) - (((data["q99_roll_mean_1000"]) + ((-1.0*((((((((data["q05_roll_std_100"]) + (1.0))) * 2.0)) * 2.0))))))))))))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh((-1.0*((((7.0) * (((73.0) * (((((7.0) * (((((data["Moving_average_6000_mean"]) * (((((-1.0*((data["Moving_average_6000_mean"])))) + (np.tanh((np.tanh((data["q01_roll_std_10"]))))))/2.0)))) + (data["q95_roll_mean_10"]))))) + (data["q95_roll_mean_10"])))))))))) +
            0.0399999991*np.tanh(((((((((((data["skew"]) + (data["ave10"]))) + (data["q95_roll_std_100"]))) - (((7.0) * (((((data["q05_roll_std_100"]) * (data["q95"]))) * ((((((data["q05_roll_std_100"]) * (7.0))) + (7.0))/2.0)))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh((((((9.64285850524902344)) * (((((data["Moving_average_700_mean"]) - ((((((9.64285850524902344)) * ((((data["q05_roll_std_100"]) + (data["exp_Moving_average_6000_mean"]))/2.0)))) - ((((data["skew"]) + ((3.98402547836303711)))/2.0)))))) * 2.0)))) * (((data["iqr"]) * (data["q05_roll_std_100"]))))) +
            0.0399999991*np.tanh(((((((((((((((((data["Moving_average_1500_mean"]) - (((np.tanh((data["exp_Moving_average_6000_mean"]))) + (np.tanh((((data["abs_q05"]) + ((((data["q95_roll_std_10"]) + (data["q01_roll_std_10"]))/2.0)))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["iqr"]))) * 2.0)) +
            0.0399999991*np.tanh((((-1.0*((data["q95"])))) * (((73.0) + ((-1.0*((((data["abs_q05"]) * ((((((((-1.0*((data["abs_q05"])))) - (data["min_roll_std_10"]))) - (73.0))) - ((((-1.0*((data["q05_roll_mean_10"])))) * (73.0)))))))))))))) +
            0.0399999991*np.tanh(((data["q05_roll_std_10"]) * (((((((((data["q05_roll_std_100"]) * ((11.90774154663085938)))) * (((data["ave10"]) - (-2.0))))) - ((11.90774154663085938)))) * ((-1.0*((((data["q05_roll_std_1000"]) + (((data["q05_roll_std_100"]) - (-2.0)))))))))))) +
            0.0399999991*np.tanh(((((((data["iqr"]) * ((9.63457870483398438)))) * (((data["q95_roll_mean_1000"]) - ((((9.63457870483398438)) * (((data["q05_roll_std_10"]) * ((((data["q01_roll_std_10"]) + (((data["iqr"]) * (data["q95_roll_mean_1000"]))))/2.0)))))))))) - (((-3.0) * (data["q01_roll_std_10"]))))) +
            0.0399999991*np.tanh(((((((((-1.0) - (((np.tanh((data["q05_roll_std_10"]))) + (((((((data["q05_roll_std_100"]) - (np.tanh((((data["classic_sta_lta1_mean"]) + (((((((data["q95_roll_std_100"]) * 2.0)) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["q95_roll_mean_10"]) - (((((((data["q95_roll_mean_10"]) + (((((data["q05_roll_std_10"]) + (((3.6923100948) * ((((data["ave10"]) + (((data["q05_roll_std_10"]) + (-1.0))))/2.0)))))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((data["q05_roll_std_100"]) + (((((data["q95_roll_mean_1000"]) + (((((data["q05_roll_std_100"]) + ((((3.1415927410) + (data["q05_roll_std_1000"]))/2.0)))) * (((((((data["q05_roll_mean_10"]) * 2.0)) * 2.0)) * (((3.1415927410) * (data["q05_roll_std_100"]))))))))) * 2.0)))) * 2.0)) +
            0.0399999991*np.tanh(((((((((((((data["mean_change_rate_first_10000"]) - (data["Moving_average_6000_mean"]))) * 2.0)) - (data["q01_roll_mean_1000"]))) - (((data["av_change_abs_roll_mean_10"]) + (((data["mean_change_rate_last_50000"]) + (((data["av_change_abs_roll_std_100"]) - ((((((13.93686485290527344)) / 2.0)) * (data["q05"]))))))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["q05_roll_std_10"]) * (data["q05_roll_std_10"]))) * ((((-1.0*((((2.0) - ((-1.0*((((data["q05_roll_std_100"]) + (((((data["q05_roll_std_100"]) * (data["abs_q05"]))) * 2.0)))))))))))) * 2.0)))) - ((-1.0*((data["kurt"])))))) * 2.0)) +
            0.0399999991*np.tanh(((((((((((((((data["q05"]) - (np.tanh((((data["iqr"]) + (((((data["Moving_average_1500_mean"]) + (((data["q05"]) + (data["Moving_average_3000_mean"]))))) * 2.0)))))))) * 2.0)) + (data["q05"]))) * 2.0)) * 2.0)) - (data["av_change_abs_roll_mean_10"]))) * 2.0)) +
            0.0399999991*np.tanh(((((((((np.tanh((((((((((((((data["Moving_average_6000_mean"]) - (data["q05_roll_std_100"]))) * 2.0)) * 2.0)) - (((1.6428600550) + (data["q01_roll_std_10"]))))) * 2.0)) - (data["av_change_abs_roll_std_100"]))))) * 2.0)) - (data["abs_q05"]))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh((((((((((((((data["skew"]) + (data["q95_roll_std_10"]))/2.0)) - (((((data["ave_roll_std_100"]) + (((data["max_roll_mean_10"]) * 2.0)))) * (data["abs_q05"]))))) + (((((np.tanh((data["q95_roll_std_100"]))) - (data["q01_roll_std_10"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((data["skew"]) + (((((data["q01_roll_std_100"]) * 2.0)) + (((data["Moving_average_700_mean"]) + (((data["q05_roll_std_10"]) * (((data["q05"]) * ((((((73.0) * (((data["Moving_average_700_mean"]) + (((data["q05_roll_std_100"]) / 2.0)))))) + (data["q05_roll_std_10"]))/2.0)))))))))))) +
            0.0399999991*np.tanh(((((((data["q05_roll_mean_10"]) * (((data["q05_roll_std_100"]) * (data["q05_roll_std_100"]))))) * (((((data["q05_roll_std_100"]) * 2.0)) * (((data["q05_roll_std_1000"]) * 2.0)))))) - (((((((data["q05_roll_std_1000"]) * 2.0)) * (data["q05_roll_std_100"]))) - (((data["q01_roll_std_1000"]) * 2.0)))))) +
            0.0399999991*np.tanh(((((((-1.0) + (((((((np.tanh((data["q01_roll_std_10"]))) * 2.0)) * 2.0)) + (((data["max_to_min_diff"]) + (((data["q05"]) * ((((14.98138904571533203)) + (((data["mean_change_rate_last_10000"]) * 2.0)))))))))))) * (73.0))) * (73.0))) +
            0.0399999991*np.tanh(((((((data["q05_roll_std_10"]) - (((((data["q05_roll_std_1000"]) + (0.7446810007))) * (((data["q05_roll_std_10"]) * (((((((data["q01_roll_std_10"]) * (((data["ave10"]) + (((data["q95_roll_mean_10"]) * (data["q05_roll_std_10"]))))))) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((np.tanh((data["q01_roll_std_100"]))) - (((((((((data["q05_roll_std_1000"]) + (((((data["q95"]) - (np.tanh((data["q95_roll_std_100"]))))) * 2.0)))) - (np.tanh((np.tanh((data["q95"]))))))) - (data["std_roll_mean_1000"]))) * 2.0)))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((data["q05_roll_std_10"]) + (((data["ave_roll_mean_1000"]) * 2.0)))) * (((((((data["ave_roll_mean_1000"]) * (data["ave_roll_mean_1000"]))) - (data["q05_roll_std_1000"]))) + (((data["q05_roll_mean_10"]) * (((data["Moving_average_3000_mean"]) + (((((data["q05_roll_std_10"]) * 2.0)) * 2.0)))))))))) +
            0.0399999991*np.tanh(((((((((((data["q05_roll_std_1000"]) + (((data["q05_roll_std_100"]) - (((data["Moving_average_6000_mean"]) * (data["Moving_average_6000_mean"]))))))) * (((data["q95_roll_std_100"]) * ((((-1.0*((data["iqr"])))) - (((data["Moving_average_6000_mean"]) + (data["Moving_average_1500_mean"]))))))))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((((data["sum"]) + (0.9756100178))) * 2.0)) * 2.0)) * (((data["q01_roll_std_100"]) - (((np.tanh((((data["q05_roll_std_10"]) * (data["q05_roll_std_10"]))))) + (((data["q05_roll_std_1000"]) * (((((data["q01_roll_std_100"]) * (data["q01_roll_std_10"]))) * 2.0)))))))))) +
            0.0399999991*np.tanh((((((((((-1.0*((data["q05_roll_std_100"])))) - (((data["q01_roll_std_100"]) * (data["mean"]))))) * (((data["q05_roll_std_100"]) * (data["q05_roll_std_100"]))))) * (((data["q05_roll_std_100"]) * (data["q05_roll_std_100"]))))) + (((((data["av_change_rate_roll_std_10"]) * (data["max_to_min"]))) * 2.0)))) +
            0.0399999991*np.tanh(((((((data["q05_roll_std_10"]) * 2.0)) * 2.0)) * (((((0.5434780121) * 2.0)) - (((((data["q05_roll_std_10"]) * (((((((((data["q95_roll_std_10"]) * 2.0)) * (data["q05_roll_std_10"]))) + (data["q05_roll_std_10"]))) * (data["q05_roll_std_10"]))))) + (data["skew"]))))))) +
            0.0399999991*np.tanh(((((data["abs_q05"]) * (data["q95_roll_std_100"]))) - (((((data["q05_roll_std_100"]) * (((data["q05_roll_std_100"]) * (((data["q05_roll_std_100"]) * (((data["q05_roll_std_100"]) * (((data["abs_q05"]) * (data["q05_roll_std_100"]))))))))))) + ((((data["min_roll_std_100"]) + (data["av_change_abs_roll_mean_100"]))/2.0)))))) +
            0.0399999991*np.tanh(((((data["Moving_average_6000_mean"]) + (0.8873239756))) * (((((data["q05_roll_std_100"]) + (data["q95_roll_std_1000"]))) * (((1.6428600550) - (((data["q05_roll_std_1000"]) * (((((data["q05_roll_std_100"]) + (((data["exp_Moving_average_3000_mean"]) + (0.9756100178))))) + (data["q05_roll_std_100"]))))))))))) +
            0.0399999991*np.tanh(((((data["q95_roll_mean_10"]) + (((data["q95_roll_std_1000"]) * (((data["med"]) + (((((data["q05_roll_mean_100"]) + (((data["q95_roll_mean_10"]) * (data["q95_roll_mean_10"]))))) * (data["q05_roll_std_100"]))))))))) * (((((data["exp_Moving_average_6000_mean"]) - (data["q05_roll_std_100"]))) - (data["q05_roll_std_1000"]))))) +
            0.0399999991*np.tanh(((((((data["min"]) + ((((-1.0*((data["MA_400MA_BB_high_mean"])))) + (np.tanh((((((0.3183098733) + (((((data["kurt"]) * 2.0)) * 2.0)))) * ((((3.0)) + ((-1.0*((data["kurt"])))))))))))))) * 2.0)) * 2.0)) +
            0.0399999991*np.tanh(((((((data["min_roll_mean_1000"]) * 2.0)) * 2.0)) + ((((-1.0*((((((data["q05_roll_std_1000"]) + (((data["q01_roll_std_10"]) * (data["q95"]))))) + (((data["avg_last_50000"]) * (((data["q01_roll_std_10"]) - (data["avg_last_50000"])))))))))) * (data["q01_roll_std_100"]))))) +
            0.0399921872*np.tanh(((((data["q95_roll_std_1000"]) - ((((((((data["av_change_abs_roll_mean_10"]) + (data["std_first_10000"]))/2.0)) + (((data["q95"]) - (data["min_last_10000"]))))) - (((data["min"]) + (((data["q95_roll_std_1000"]) + (data["mean_change_rate_last_10000"]))))))))) - (((data["q95"]) + (data["q95"]))))) +
            0.0399999991*np.tanh(((((((data["min_roll_mean_1000"]) - (((data["q05_roll_std_1000"]) * (((np.tanh((((data["med"]) * 2.0)))) + ((((((data["skew"]) * (data["kurt"]))) + (data["q05_roll_std_1000"]))/2.0)))))))) + (((data["kurt"]) * (data["skew"]))))) + (data["min_roll_mean_1000"]))) +
            0.0399921872*np.tanh(((data["exp_Moving_average_6000_mean"]) * (((((data["max_roll_std_100"]) - (((((data["exp_Moving_average_6000_mean"]) * (data["MA_400MA_std_mean"]))) + (data["q05_roll_mean_10"]))))) * (((((data["mean_change_rate_first_10000"]) * (73.0))) - (((((data["abs_mean"]) * (73.0))) * (73.0))))))))) +
            0.0399999991*np.tanh(((data["q01_roll_std_10"]) + ((-1.0*((((data["q05_roll_std_1000"]) * (((data["av_change_rate_roll_mean_100"]) + ((((-1.0*((data["classic_sta_lta4_mean"])))) + (((data["med"]) + (((data["q05_roll_std_1000"]) + (((data["iqr"]) * (((data["iqr"]) + (data["max_last_10000"])))))))))))))))))))) +
            0.0399921872*np.tanh((((((((data["q05"]) + (data["mean_change_rate_first_10000"]))/2.0)) + (((data["kurt"]) - (((((data["kurt"]) * (data["abs_max_roll_std_1000"]))) * (data["kurt"]))))))) + (((((data["av_change_abs_roll_std_100"]) * ((-1.0*((data["classic_sta_lta3_mean"])))))) * 2.0)))) +
            0.0399999991*np.tanh(((((((((((((((data["skew"]) - ((((data["min_roll_std_1000"]) + (data["q95_roll_mean_100"]))/2.0)))) + (((data["q95_roll_mean_100"]) * (data["q05"]))))) * (data["MA_400MA_std_mean"]))) * (data["MA_400MA_BB_high_mean"]))) * 2.0)) * 2.0)) + ((((data["mean_diff"]) + (data["min_roll_std_1000"]))/2.0)))) +
            0.0399921872*np.tanh((((((4.12892913818359375)) * ((((((data["exp_Moving_average_300_mean"]) + (data["med"]))/2.0)) - (data["q01_roll_std_100"]))))) * (((data["skew"]) + (((0.2439019978) + ((((data["MA_1000MA_std_mean"]) + ((((((data["av_change_abs_roll_std_1000"]) + (data["med"]))) + (data["iqr"]))/2.0)))/2.0)))))))) +
            0.0399999991*np.tanh(((((((((data["av_change_abs_roll_std_10"]) * (((((data["mean_change_rate_last_50000"]) + (data["av_change_abs_roll_mean_10"]))) + (data["min_last_50000"]))))) * 2.0)) * 2.0)) + ((((((((1.0)) + (((data["av_change_rate_roll_std_10"]) + (data["skew"]))))/2.0)) + (data["q01_roll_mean_1000"]))/2.0)))) +
            0.0399999991*np.tanh(((((((((data["q05_roll_std_10"]) * (((((((data["med"]) + (data["q05_roll_std_10"]))) * (data["q05_roll_mean_10"]))) * (data["q95_roll_std_1000"]))))) * (data["q05_roll_std_10"]))) + (((((data["q05_roll_mean_10"]) + (data["q95_roll_std_100"]))) + (data["q95_roll_std_100"]))))) * 2.0)) +
            0.0399921872*np.tanh((((data["skew"]) + (((((((((data["min_roll_std_10"]) * ((-1.0*((data["min_roll_std_10"])))))) + (data["q05_roll_mean_1000"]))) * (data["Hilbert_mean"]))) - ((((data["av_change_rate_roll_std_100"]) + (((data["max_roll_std_100"]) * (((data["skew"]) * (data["skew"]))))))/2.0)))))/2.0)) +
            0.0399999991*np.tanh(((data["q05_roll_std_10"]) * (((((((((data["q01_roll_mean_1000"]) - (np.tanh((((data["Moving_average_6000_mean"]) - (((data["ave_roll_std_10"]) * (((data["kurt"]) - ((((data["av_change_abs_roll_mean_10"]) + ((((data["Moving_average_6000_mean"]) + (data["ave_roll_std_10"]))/2.0)))/2.0)))))))))))) * 2.0)) * 2.0)) * 2.0)))) +
            0.0399843715*np.tanh(((data["min"]) + ((-1.0*((((data["av_change_rate_roll_std_100"]) * ((((((((data["mean_change_rate"]) + (data["min"]))/2.0)) + (data["min_roll_std_1000"]))) + (((np.tanh((data["min_last_10000"]))) + ((((((data["min_roll_std_100"]) * (data["av_change_rate_roll_mean_10"]))) + (data["av_change_abs_roll_std_10"]))/2.0))))))))))))) +
            0.0399999991*np.tanh(((data["min_roll_mean_100"]) * (((((data["min_roll_std_1000"]) * (((data["q01_roll_std_1000"]) * (((((((data["q01_roll_std_1000"]) * (data["q99_roll_std_100"]))) + (data["exp_Moving_average_300_mean"]))) + (data["avg_first_10000"]))))))) + (((((data["mean_change_rate"]) * (3.1415927410))) - (data["avg_first_10000"]))))))) +
            0.0399921872*np.tanh(((((data["q95_roll_mean_1000"]) * (((((((((data["q95_roll_mean_1000"]) + (((data["avg_last_10000"]) + (((3.0) / 2.0)))))/2.0)) + (data["q05_roll_mean_100"]))/2.0)) - (data["q95_roll_std_1000"]))))) * (((((((data["q95_roll_std_1000"]) * 2.0)) + (data["q99_roll_mean_10"]))) + (data["classic_sta_lta4_mean"]))))) +
            0.0399296731*np.tanh((((((((((((data["q01_roll_std_100"]) + (np.tanh((((data["q01"]) + ((-1.0*((((data["q01_roll_std_100"]) * 2.0))))))))))/2.0)) * ((-1.0*(((((data["q95"]) + (((data["exp_Moving_average_6000_mean"]) * 2.0)))/2.0))))))) * 2.0)) * 2.0)) * 2.0)) +
            0.0399921872*np.tanh(((((np.tanh((data["q99"]))) + ((-1.0*(((((data["max_first_10000"]) + (((data["av_change_rate_roll_mean_1000"]) - (data["q99"]))))/2.0))))))) - ((((((((data["av_change_rate_roll_mean_1000"]) * (data["av_change_rate_roll_mean_1000"]))) + (data["q05_roll_std_10"]))/2.0)) * (((data["q05_roll_std_10"]) * (data["iqr"]))))))) +
            0.0399999991*np.tanh(((data["min_roll_mean_1000"]) * (((((((((((data["std_roll_std_100"]) + (((data["q05"]) + (((((data["q95_roll_mean_1000"]) * (data["q95_roll_std_1000"]))) + (((data["q95_roll_mean_1000"]) * (((data["q95_roll_mean_1000"]) * (data["q95_roll_std_1000"]))))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +
            0.0399921872*np.tanh(((((data["mean_diff"]) * (data["min_roll_std_10"]))) + (((((((data["std_last_50000"]) * (3.6923100948))) * (((data["q95_roll_mean_1000"]) + (((data["classic_sta_lta4_mean"]) - (data["min_roll_std_10"]))))))) * ((((((data["min_roll_std_10"]) + (data["q05_roll_mean_100"]))/2.0)) - (data["max_roll_std_10"]))))))) +
            0.0399999991*np.tanh(((((data["q01_roll_std_10"]) + (data["min_roll_std_100"]))) * (((data["abs_q05"]) - (((data["avg_last_50000"]) - (((((data["classic_sta_lta2_mean"]) / 2.0)) + (((data["max_roll_mean_1000"]) * (((((data["avg_last_50000"]) - (np.tanh((data["min_roll_std_100"]))))) - (data["min_roll_std_100"]))))))))))))) +
            0.0399999991*np.tanh(((((data["q95_roll_mean_100"]) * (data["min_roll_mean_1000"]))) - (((((data["av_change_abs_roll_std_100"]) * (((((data["classic_sta_lta3_mean"]) + (data["mean_change_rate_first_50000"]))) + (((data["q95_roll_mean_100"]) * (data["classic_sta_lta3_mean"]))))))) + (((data["mean_diff"]) * (((data["classic_sta_lta3_mean"]) - (data["mean_change_rate_first_50000"]))))))))) +
            0.0398359038*np.tanh(((data["q95_roll_mean_10"]) * (((((data["q95_roll_mean_10"]) * (((data["min_roll_mean_1000"]) + (data["av_change_rate_roll_std_10"]))))) - (((data["max_to_min"]) + (((((((data["max_to_min"]) * 2.0)) * 2.0)) * (((data["av_change_rate_roll_std_10"]) + (data["q01_roll_mean_1000"]))))))))))) +
            0.0399999991*np.tanh(((data["q01_roll_std_10"]) * (((((((data["q01_roll_std_10"]) * (data["skew"]))) + (((data["classic_sta_lta4_mean"]) - (data["classic_sta_lta2_mean"]))))) + ((((data["av_change_abs_roll_mean_1000"]) + (((((-1.0*((data["q01_roll_std_100"])))) + (((data["q01_roll_std_10"]) * ((-1.0*((data["q01_roll_std_10"])))))))/2.0)))/2.0)))))) +
            0.0385153331*np.tanh((((((data["std_roll_mean_1000"]) + (data["av_change_abs_roll_mean_100"]))) + (((data["max_first_10000"]) * (((data["MA_700MA_std_mean"]) * (((73.0) * (((np.tanh((data["max_to_min"]))) - (((data["av_change_rate_roll_mean_1000"]) + (((((data["std_roll_mean_1000"]) * 2.0)) * 2.0)))))))))))))/2.0)) +
            0.0399999991*np.tanh(((((data["std_roll_mean_1000"]) * (((data["av_change_rate_roll_std_1000"]) - (((((((data["skew"]) - (data["min_roll_std_10"]))) - (data["classic_sta_lta4_mean"]))) + (((data["std_roll_mean_1000"]) - (2.0))))))))) + (((data["classic_sta_lta4_mean"]) * (((data["av_change_abs_roll_mean_1000"]) - (data["q99"]))))))) +
            0.0399843715*np.tanh((-1.0*((((data["kurt"]) * (((data["kurt"]) * ((((((data["av_change_abs_roll_mean_10"]) + (((((data["med"]) - (((data["med"]) * (data["av_change_abs_roll_mean_10"]))))) - (data["min_roll_std_10"]))))/2.0)) + (((data["q99_roll_std_1000"]) * (data["kurt"])))))))))))) +
            0.0399921872*np.tanh(((data["mean_change_rate_first_10000"]) * (((data["av_change_abs_roll_mean_100"]) * ((-1.0*((((((np.tanh((((((data["classic_sta_lta3_mean"]) + (data["mean_change_rate_first_10000"]))) * 2.0)))) + (((1.9019600153) - (data["av_change_abs_roll_mean_100"]))))) - (((((data["classic_sta_lta3_mean"]) * 2.0)) * (data["mean_change_rate_first_10000"])))))))))))) +
            0.0399921872*np.tanh(((((data["av_change_rate_roll_std_100"]) * (((data["classic_sta_lta3_mean"]) * (((((data["min_roll_std_1000"]) - (data["min"]))) * (data["q05"]))))))) + (((data["mean_change_rate_last_10000"]) * (((((data["av_change_abs_roll_std_1000"]) - (data["min"]))) - (((data["classic_sta_lta3_mean"]) * (data["q01_roll_std_1000"]))))))))) +
            0.0399999991*np.tanh(((data["q95_roll_std_1000"]) + (((data["q05_roll_mean_10"]) - ((-1.0*((((data["q95_roll_std_1000"]) * (((data["q95_roll_std_1000"]) * (((((data["q01_roll_std_10"]) * ((((((((data["q95_roll_std_1000"]) / 2.0)) + (data["q01_roll_mean_1000"]))/2.0)) - (data["med"]))))) * (data["q95_roll_std_1000"])))))))))))))) +
            0.0399843715*np.tanh(((data["av_change_abs_roll_mean_100"]) * (((data["classic_sta_lta4_mean"]) * (((((((((3.65123486518859863)) + ((((3.65123486518859863)) * (data["abs_max_roll_std_10"]))))/2.0)) + (((((data["min_last_50000"]) * (((data["abs_max_roll_std_1000"]) - (data["av_change_abs_roll_mean_10"]))))) - (data["classic_sta_lta4_mean"]))))) * (data["ave_roll_mean_10"]))))))) +
            0.0399999991*np.tanh(((((data["max_to_min_diff"]) * (((data["min_roll_std_100"]) * (((data["av_change_rate_roll_std_100"]) + (((data["q01_roll_std_100"]) + (((((((data["av_change_rate_roll_mean_100"]) + (data["Hilbert_mean"]))) * (data["q95_roll_mean_1000"]))) + (((data["q01"]) * (data["min_roll_std_100"]))))))))))))) * 2.0)) +
            0.0399999991*np.tanh(((((np.tanh((((((data["q05"]) + (((np.tanh((data["avg_first_10000"]))) + (data["mean_change_rate_last_10000"]))))) + (data["q05"]))))) + (((((data["kurt"]) * (data["mean_change_rate_last_10000"]))) + (data["q95_roll_std_1000"]))))) * (((data["q01_roll_mean_1000"]) + (data["av_change_rate_roll_mean_100"]))))) +
            0.0399921872*np.tanh(((data["min_roll_mean_1000"]) + ((((((((((((data["av_change_rate_roll_std_10"]) + (((data["av_change_abs_roll_std_1000"]) * (data["min_roll_mean_1000"]))))/2.0)) * (data["classic_sta_lta3_mean"]))) + (data["av_change_rate_roll_std_10"]))/2.0)) + (((data["abs_max_roll_mean_100"]) - (((data["av_change_abs_roll_std_1000"]) * (data["classic_sta_lta1_mean"]))))))/2.0)))) +
            0.0399999991*np.tanh(((data["max_to_min"]) * (((((data["MA_400MA_std_mean"]) - (data["q95_roll_std_100"]))) + ((((((data["min_roll_std_10"]) * (((((((data["min_roll_std_10"]) * (data["classic_sta_lta1_mean"]))) - (data["min_roll_std_10"]))) - (data["kurt"]))))) + ((((data["kurt"]) + (data["min_roll_std_10"]))/2.0)))/2.0)))))) +
            0.0399921872*np.tanh(((data["min_roll_std_10"]) * ((((((((((((np.tanh((data["min_roll_std_10"]))) + (data["avg_last_10000"]))/2.0)) - (data["q95_roll_std_100"]))) - (data["MA_700MA_BB_high_mean"]))) + (((data["abs_max_roll_mean_100"]) - (data["classic_sta_lta4_mean"]))))) * (((data["min_roll_std_10"]) - (((3.0) / 2.0)))))))) +
            0.0399765596*np.tanh(((((data["avg_last_10000"]) * (((data["min_roll_mean_1000"]) + (((((data["avg_last_10000"]) + (np.tanh((data["mean_change_rate_last_50000"]))))) * (((data["min_roll_std_10"]) * (data["min_last_10000"]))))))))) + (np.tanh((np.tanh((((((data["mean_change_rate_last_50000"]) * (data["min_roll_std_1000"]))) * 2.0)))))))) +
            0.0399921872*np.tanh(np.tanh((np.tanh((np.tanh((((((((data["q95_roll_std_10"]) - (data["q05"]))) * (((data["abs_q05"]) * (((data["mean_change_rate_first_10000"]) + (((((data["mean_change_rate_first_10000"]) + (data["q05"]))) * (((data["abs_q05"]) * (data["q95_roll_mean_1000"]))))))))))) * 2.0)))))))) +
            0.0399843715*np.tanh(((data["q95_roll_std_1000"]) * (((data["max_roll_mean_100"]) + (((((((((data["q01_roll_mean_1000"]) * 2.0)) * 2.0)) * (data["min_roll_mean_1000"]))) * (((((((data["q01_roll_mean_1000"]) + ((-1.0*((data["max_to_min"])))))) * 2.0)) * 2.0)))))))) +
            0.0399765596*np.tanh(((data["q05_roll_std_100"]) * ((-1.0*((np.tanh((((((data["med"]) + (((((data["med"]) + (((data["q01_roll_std_10"]) * (((data["avg_first_50000"]) + (data["q01_roll_std_10"]))))))) * ((((1.16074943542480469)) + (data["avg_first_50000"]))))))) * 2.0))))))))) +
            0.0399062298*np.tanh((-1.0*((((data["classic_sta_lta2_mean"]) * ((((((np.tanh((data["kurt"]))) + (data["avg_last_50000"]))/2.0)) + (((data["q05_roll_mean_100"]) * ((-1.0*(((((((((data["kurt"]) + (data["q05_roll_mean_100"]))/2.0)) - (np.tanh((data["kurt"]))))) + (data["std_last_10000"]))))))))))))))) +
            0.0399453007*np.tanh(((data["std_last_50000"]) * (((((np.tanh((((((((((((data["av_change_abs_roll_mean_100"]) + (np.tanh((np.tanh((((((((((data["skew"]) + (data["Moving_average_6000_mean"]))) * 2.0)) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)))) +
            0.0390935726*np.tanh((-1.0*((((data["max_roll_mean_100"]) * (((((((np.tanh((((((data["classic_sta_lta3_mean"]) + ((((data["q99_roll_mean_1000"]) + (data["mean_change_rate_first_10000"]))/2.0)))) + (((data["av_change_abs_roll_mean_10"]) + (((data["classic_sta_lta3_mean"]) * (data["q95_roll_mean_1000"]))))))))) * 2.0)) * 2.0)) * 2.0))))))) +
            0.0399296731*np.tanh(((data["q95_roll_std_100"]) + ((-1.0*((((data["q95_roll_std_10"]) + ((((((((data["kurt"]) * ((((data["mean_change_rate_last_10000"]) + (data["trend"]))/2.0)))) * (data["kurt"]))) + ((((((data["med"]) + (data["max_first_50000"]))/2.0)) * (data["min_roll_std_100"]))))/2.0))))))))) +
            0.0398984179*np.tanh(((((np.tanh((((data["q01"]) - (data["q05"]))))) + (((((((((np.tanh((data["q05_roll_std_100"]))) + (data["q01"]))) * 2.0)) + (data["q95_roll_mean_1000"]))) * 2.0)))) * ((-1.0*((((data["q01"]) - (data["q05"])))))))) +
            0.0399999991*np.tanh(np.tanh((np.tanh(((-1.0*((((data["mean_change_rate_first_10000"]) * (((data["min_last_10000"]) + (((data["min_last_10000"]) + (((((data["abs_q95"]) + (((data["avg_last_50000"]) + ((((data["min_last_10000"]) + (0.8873239756))/2.0)))))) * ((-1.0*((data["q05_roll_std_100"]))))))))))))))))))) +
            0.0399999991*np.tanh(((data["min_first_50000"]) * ((((-1.0*((data["av_change_abs_roll_std_100"])))) - (((data["ave_roll_mean_10"]) * ((-1.0*((((data["max_to_min"]) - (((((((data["MA_400MA_BB_low_mean"]) - (((data["av_change_rate_roll_mean_100"]) + (((data["q01_roll_mean_1000"]) * (data["av_change_rate_roll_mean_100"]))))))) * 2.0)) * 2.0))))))))))))) +
            0.0399999991*np.tanh(((((((((data["max_to_min_diff"]) * (((data["av_change_rate_roll_std_100"]) + (((((data["q95_roll_std_10"]) + (data["max_first_50000"]))) + (((data["mean_diff"]) + (((((data["MA_400MA_BB_low_mean"]) * (data["max_first_50000"]))) * (data["max_first_50000"]))))))))))) * 2.0)) * 2.0)) * (data["min_roll_std_100"]))) +
            0.0399999991*np.tanh(((((data["max"]) * (((data["std_first_50000"]) * ((((((-1.0*((data["max_roll_std_100"])))) + (((((((((data["q001"]) + (((data["q001"]) * (data["classic_sta_lta4_mean"]))))) + (data["classic_sta_lta4_mean"]))) * 2.0)) * 2.0)))) * 2.0)))))) * (data["q05_roll_std_1000"]))) +
            0.0399765596*np.tanh(((((data["abs_max_roll_std_1000"]) * 2.0)) * (((data["q95_roll_std_10"]) - (((data["q999"]) - (np.tanh((((data["max_last_10000"]) * (((((((((((((data["abs_max_roll_std_1000"]) + (data["Moving_average_6000_mean"]))) * 2.0)) + (data["mean_diff"]))) * 2.0)) * 2.0)) * 2.0)))))))))))) +
            0.0399687439*np.tanh(((data["min_roll_std_10"]) * ((((((data["avg_last_50000"]) + (((((((data["q01_roll_std_1000"]) * (data["classic_sta_lta4_mean"]))) * (data["mean_change_rate_last_50000"]))) - (((data["min_roll_std_10"]) * (data["av_change_abs_roll_mean_10"]))))))/2.0)) + ((((((data["classic_sta_lta4_mean"]) * (data["Moving_average_6000_mean"]))) + (data["mean_change_rate_last_50000"]))/2.0)))))) +
            0.0398906022*np.tanh((((np.tanh((((data["av_change_abs_roll_mean_1000"]) - (((data["Hann_window_mean"]) - (data["skew"]))))))) + (((((((data["min_roll_mean_1000"]) * 2.0)) * 2.0)) * (((data["skew"]) + (((((data["std_roll_mean_1000"]) - (data["q99_roll_mean_10"]))) - (data["med"]))))))))/2.0)) +
            0.0399999991*np.tanh(((data["std_first_10000"]) * ((-1.0*((((data["min_roll_std_10"]) * (((((((((data["kurt"]) + (((((data["av_change_rate_roll_std_1000"]) - (data["trend"]))) + (data["mean_change_rate_first_10000"]))))) - (data["max_last_10000"]))) + (((data["trend"]) * (data["trend"]))))) * 2.0))))))))) +
            0.0399765596*np.tanh(((data["q99"]) * ((-1.0*((((data["iqr"]) - (((((((1.9019600153) + (((data["abs_max_roll_mean_100"]) + (data["q99"]))))) * (((((data["std_roll_mean_1000"]) * 2.0)) * (((data["max_roll_mean_1000"]) - (data["std_roll_mean_1000"]))))))) + (data["min_roll_std_1000"])))))))))) +
            0.0399765596*np.tanh(((((data["mean_change_rate_last_10000"]) + (np.tanh((((((np.tanh((data["skew"]))) + (((data["max_roll_std_1000"]) * 2.0)))) * 2.0)))))) * (((data["abs_max_roll_std_10"]) + ((-1.0*((((data["skew"]) * (((((data["abs_max"]) / 2.0)) - (data["ave_roll_std_1000"])))))))))))) +
            0.0355928876*np.tanh((((((((data["min_roll_std_100"]) * (data["q05_roll_mean_1000"]))) * (data["min_roll_std_1000"]))) + (((((data["min_roll_std_100"]) * (data["min_roll_std_100"]))) - (((((data["max_to_min"]) + ((((((data["q999"]) + (data["min_roll_std_100"]))/2.0)) + (data["q999"]))))) * (data["classic_sta_lta3_mean"]))))))/2.0)) +
            0.0399843715*np.tanh((((((-1.0*((((((((data["q01_roll_std_10"]) * (data["q05_roll_std_1000"]))) * (data["max_first_50000"]))) * (data["q01_roll_std_10"])))))) * (((data["max_first_50000"]) + (data["max_to_min_diff"]))))) * (((((data["min"]) - (data["q05"]))) - (data["q05"]))))) +
            0.0399765596*np.tanh(((data["med"]) * (((np.tanh((((((((((data["av_change_abs_roll_std_100"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) - (((((data["min_roll_std_10"]) + ((((data["q01_roll_std_1000"]) + ((((data["max_to_min_diff"]) + (data["min_roll_std_1000"]))/2.0)))/2.0)))) * (((data["av_change_abs_roll_std_100"]) * 2.0)))))))) +
            0.0387106836*np.tanh((-1.0*((((((data["av_change_rate_roll_mean_10"]) * (((data["q95_roll_std_1000"]) + (((((((((((data["min_roll_mean_1000"]) * 2.0)) * 2.0)) * (((data["q95_roll_std_1000"]) + (data["min_roll_mean_1000"]))))) * 2.0)) * (((data["q95_roll_std_100"]) - (((data["min_roll_mean_1000"]) * 2.0)))))))))) * 2.0))))) +
            0.0399999991*np.tanh(((data["max_last_10000"]) * ((((((((data["MA_700MA_BB_low_mean"]) * (((((data["avg_first_10000"]) - (data["av_change_abs_roll_mean_1000"]))) - (((data["avg_first_10000"]) * (((data["avg_first_10000"]) - (((data["min_roll_std_1000"]) * 2.0)))))))))) * 2.0)) + (((data["avg_first_10000"]) - (data["mean_change_rate_last_10000"]))))/2.0)))) +
            0.0399999991*np.tanh(np.tanh((((((((((data["av_change_rate_roll_std_100"]) * (((data["min"]) * (((data["q05_roll_std_100"]) + (((((data["av_change_abs_roll_std_100"]) + (((data["mean_change_rate_first_10000"]) * (data["av_change_rate_roll_std_100"]))))) + (((data["med"]) - (data["mean_change_rate_first_10000"]))))))))))) * 2.0)) * 2.0)) * 2.0)))) +
            0.0359679610*np.tanh((-1.0*((((np.tanh((((((((((((((((((data["min_roll_std_1000"]) - (data["av_change_rate_roll_std_1000"]))) - (((-2.0) + ((-1.0*((data["min_roll_std_1000"])))))))) - (np.tanh((data["av_change_abs_roll_std_100"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) / 2.0))))) +
            0.0373354182*np.tanh((-1.0*((((data["max_to_min"]) * (((data["avg_first_50000"]) * (((((data["ave_roll_std_100"]) - (((data["classic_sta_lta1_mean"]) + (data["mean_change_rate_first_10000"]))))) * (((data["classic_sta_lta2_mean"]) - ((-1.0*((((data["ave_roll_std_100"]) / 2.0)))))))))))))))) +
            0.0398671627*np.tanh(((((0.8873239756) + (data["avg_last_50000"]))) * (((data["iqr"]) * (((((0.8873239756) + (data["Moving_average_6000_mean"]))) + (((data["q05_roll_mean_10"]) * (((((((data["ave10"]) + (data["q01_roll_std_10"]))) * (data["q01_roll_std_10"]))) * (data["iqr"]))))))))))) +
            0.0399453007*np.tanh((((np.tanh((((((((((((data["mean_change_rate_last_10000"]) + (np.tanh(((((((-1.0*((data["mean_change_rate_last_50000"])))) * 2.0)) * (data["q95_roll_mean_1000"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) + (np.tanh((np.tanh(((((-1.0*((data["mean_change_rate_last_50000"])))) * 2.0)))))))/2.0)) +
            0.0399687439*np.tanh(((((((data["max_roll_std_1000"]) + (((data["max_roll_std_1000"]) + (data["trend"]))))) + (data["count_big"]))) * (((data["max_to_min"]) * (((((((-1.0*((data["max_to_min"])))) + (data["max_roll_mean_1000"]))/2.0)) - (data["q99_roll_std_10"]))))))) +
            0.0399843715*np.tanh(((data["classic_sta_lta2_mean"]) * (((data["abs_mean"]) * (((data["classic_sta_lta1_mean"]) * (((((((data["mean_change_rate_first_50000"]) + ((-1.0*((data["mean_diff"])))))) + (data["classic_sta_lta4_mean"]))) + (((((((-2.0) + (data["av_change_rate_roll_std_10"]))/2.0)) + (data["av_change_abs_roll_std_1000"]))/2.0)))))))))) +
            0.0399687439*np.tanh(((data["min_roll_std_100"]) * (((((((data["mean_diff"]) + (((((((data["av_change_abs_roll_std_100"]) * (data["min_roll_std_100"]))) + (data["min_roll_std_1000"]))) * (((data["std_roll_mean_1000"]) * (((data["min_last_10000"]) + (data["av_change_abs_roll_std_100"]))))))))) * 2.0)) * (data["av_change_abs_roll_std_100"]))))) +
            0.0399374850*np.tanh(((((data["min_first_10000"]) - (((data["mean_change_rate_first_10000"]) * (data["kurt"]))))) * (((((((data["abs_trend"]) * (data["kurt"]))) - (data["mean_diff"]))) + (((data["std_roll_mean_1000"]) + (((data["std_roll_mean_1000"]) + (((data["min_first_10000"]) + (data["abs_std"]))))))))))) +
            0.0399921872*np.tanh(((data["q95_roll_std_10"]) * ((-1.0*((((((-1.0*((data["abs_max_roll_mean_100"])))) + ((((-1.0*((((data["classic_sta_lta4_mean"]) / 2.0))))) - (((data["q001"]) * (((data["q95_roll_mean_1000"]) - ((-1.0*((((((((data["classic_sta_lta4_mean"]) / 2.0)) / 2.0)) / 2.0))))))))))))/2.0))))))) +
            0.0399843715*np.tanh(((((data["mean_change_rate_last_10000"]) * (((data["abs_max"]) * (((((((((((data["min_roll_std_1000"]) + (data["max_to_min"]))) - (data["count_big"]))) * 2.0)) - (data["max_to_min"]))) + (((data["av_change_abs_roll_std_1000"]) + (data["mean_diff"]))))))))) + (data["count_big"]))) +
            0.0383434258*np.tanh(((((data["med"]) * ((-1.0*((((((((data["trend"]) * (((np.tanh((data["av_change_abs_roll_std_100"]))) - (data["mean_diff"]))))) + (data["mean_diff"]))) * 2.0))))))) * (((((data["av_change_abs_roll_std_100"]) + (np.tanh((data["min_last_10000"]))))) + (data["av_change_abs_roll_std_100"]))))) +
            0.0399999991*np.tanh(np.tanh((((((((data["min_roll_mean_100"]) * 2.0)) * 2.0)) * (((((((data["max_last_10000"]) - (0.5434780121))) * (data["av_change_rate_roll_mean_100"]))) + (((data["av_change_rate_roll_std_10"]) - (((data["min_roll_std_10"]) * (data["av_change_rate_roll_std_10"]))))))))))) +
            0.0399921872*np.tanh((-1.0*((((((((((data["min_roll_std_100"]) + (data["av_change_abs_roll_std_10"]))/2.0)) * (((data["av_change_abs_roll_mean_1000"]) + (data["av_change_abs_roll_std_10"]))))) + (((data["av_change_abs_roll_std_100"]) * (((data["classic_sta_lta4_mean"]) * (((data["av_change_abs_roll_std_10"]) + (((data["min_roll_std_1000"]) * (data["av_change_abs_roll_std_100"]))))))))))/2.0))))) +
            0.0293416679*np.tanh(np.tanh((((data["abs_q95"]) * (((data["q01_roll_std_10"]) * (((data["q01_roll_std_10"]) * (((((data["max_last_50000"]) - ((((((data["av_change_abs_roll_mean_10"]) - (data["classic_sta_lta4_mean"]))) + (data["av_change_abs_roll_mean_100"]))/2.0)))) - ((-1.0*((((data["av_change_abs_roll_mean_10"]) * (data["classic_sta_lta4_mean"])))))))))))))))) +
            0.0399765596*np.tanh(((((data["trend"]) * (data["max_first_10000"]))) * (((data["med"]) + (((data["av_change_rate_roll_mean_1000"]) + (((((2.0769200325) - (((np.tanh((((((((((data["classic_sta_lta1_mean"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) / 2.0)))))))) +
            0.0399921872*np.tanh(((((data["mean_change_rate_last_50000"]) * ((-1.0*((((data["min_first_10000"]) - (data["std_first_10000"])))))))) * ((((((((((-1.0*((data["std_first_10000"])))) - (data["min_last_50000"]))) - (data["abs_max_roll_std_10"]))) - (((data["mean_change_rate_last_50000"]) * (data["std_first_10000"]))))) - (data["classic_sta_lta2_mean"]))))) +
            0.0399921872*np.tanh(((data["av_change_abs_roll_mean_10"]) * (((data["abs_trend"]) + ((((((((data["mean_diff"]) * (data["min_last_10000"]))) + (((data["mean_change_rate_first_10000"]) * (data["max_first_10000"]))))) + ((((data["av_change_abs_roll_mean_10"]) + (((((data["med"]) - (data["q95_roll_mean_1000"]))) - (data["mean_change_rate_first_10000"]))))/2.0)))/2.0)))))) +
            0.0399531126*np.tanh(((data["ave_roll_std_100"]) * (((data["min_roll_std_100"]) * (((((data["max_roll_mean_100"]) + ((((data["max_to_min_diff"]) + (data["q99"]))/2.0)))) * (((((((data["max_to_min_diff"]) * ((((-1.0*((data["std_first_50000"])))) * (data["q95_roll_std_100"]))))) - (data["skew"]))) * 2.0)))))))) +
            0.0399843715*np.tanh(((data["std_roll_std_100"]) * (((data["min_roll_std_1000"]) * (((data["max_roll_mean_10"]) - (((data["kurt"]) * ((((((data["avg_last_10000"]) * (data["avg_last_10000"]))) + ((((data["q99_roll_mean_100"]) + ((((((data["avg_last_10000"]) * (data["avg_last_10000"]))) + (data["avg_first_10000"]))/2.0)))/2.0)))/2.0)))))))))) +
            0.0353584699*np.tanh(((data["med"]) * (((np.tanh((((np.tanh(((((((-1.0*((data["max_to_min"])))) - (data["trend"]))) * 2.0)))) - (((data["min_roll_std_100"]) * (data["med"]))))))) - (np.tanh((np.tanh((data["min_roll_std_100"]))))))))) +
            0.0399765596*np.tanh((((((((data["ave_roll_mean_10"]) + (data["min_roll_std_100"]))/2.0)) * (((((data["classic_sta_lta4_mean"]) + (data["kurt"]))) + (((((data["q95_roll_mean_10"]) + (((data["q95"]) * (data["q95"]))))) * (data["sum"]))))))) * ((((data["q05_roll_std_100"]) + (data["ave_roll_mean_10"]))/2.0)))) +
            0.0399843715*np.tanh(((data["min_roll_std_10"]) * ((((((-1.0*((((data["av_change_abs_roll_std_10"]) * (data["av_change_abs_roll_mean_100"])))))) * (data["av_change_abs_roll_std_10"]))) + (np.tanh(((((-1.0*((data["q05"])))) + (((data["min_roll_std_1000"]) * ((-1.0*((((data["av_change_abs_roll_mean_100"]) * (data["av_change_abs_roll_mean_100"])))))))))))))))) +
            0.0399999991*np.tanh((-1.0*((((data["classic_sta_lta1_mean"]) * (((data["max_last_50000"]) + (((((data["abs_q95"]) + (((data["av_change_abs_roll_std_10"]) * 2.0)))) * (((((data["abs_std"]) + (data["max_last_50000"]))) * (((((data["iqr"]) * 2.0)) * (data["avg_last_10000"])))))))))))))) +
            0.0388200805*np.tanh(((((data["min_first_10000"]) * (((data["mean_change_rate"]) * (((((data["min_roll_std_100"]) + (data["max_last_50000"]))) + (((((data["mean_change_rate"]) * (((data["mean_diff"]) * (((data["min_roll_std_100"]) + (data["min_first_10000"]))))))) - (data["mean_diff"]))))))))) * 2.0)) +
            0.0389529206*np.tanh(((np.tanh((np.tanh((((data["max_to_min"]) + (data["max_to_min"]))))))) - (np.tanh((((((((((((((((((((data["min_first_10000"]) * 2.0)) + (data["max_to_min"]))) * 2.0)) * 2.0)) + (data["max_last_50000"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))) +
            0.0399843715*np.tanh((-1.0*((((((data["max_first_50000"]) * 2.0)) * (((np.tanh(((((((-1.0*((((data["q95_roll_std_1000"]) + (((data["q01_roll_std_10"]) + (data["av_change_abs_roll_std_1000"])))))))) * 2.0)) * 2.0)))) + (((data["q01_roll_std_10"]) + (((data["max_first_50000"]) * (data["mean_diff"])))))))))))) +
            0.0399453007*np.tanh((((((data["av_change_abs_roll_std_1000"]) + (data["mean_diff"]))/2.0)) * (((data["av_change_abs_roll_std_100"]) + (((data["trend"]) + (((((data["av_change_abs_roll_mean_1000"]) + ((((data["av_change_abs_roll_std_100"]) + (((data["av_change_rate_roll_std_100"]) + (((data["av_change_abs_roll_std_1000"]) + (data["min_roll_std_10"]))))))/2.0)))) * (data["av_change_abs_roll_mean_1000"]))))))))) +
            0.0399140455*np.tanh(((data["std_first_50000"]) * (((data["mean_change_rate_last_50000"]) * ((-1.0*((((data["avg_first_50000"]) * ((((data["max_first_10000"]) + (((data["min_roll_mean_10"]) - (((data["avg_first_50000"]) - (((((data["mean_change_rate_last_50000"]) - (data["min_last_10000"]))) * (data["min_last_10000"]))))))))/2.0))))))))))) +
            0.0399999991*np.tanh(((((data["min_last_10000"]) + (np.tanh((((data["ave_roll_mean_1000"]) + (((((((data["av_change_abs_roll_mean_100"]) + (data["av_change_abs_roll_std_10"]))) * (data["ave_roll_mean_1000"]))) * (((data["min_roll_std_1000"]) * 2.0)))))))))) * (((data["min_roll_std_1000"]) * (((((data["std_roll_mean_1000"]) * 2.0)) * 2.0)))))) +
            0.0399765596*np.tanh((((((data["av_change_abs_roll_mean_100"]) + (data["q99_roll_std_10"]))/2.0)) * (np.tanh((((np.tanh((((((((((((((data["avg_first_10000"]) + (data["max_roll_std_1000"]))) + (data["count_big"]))) * (73.0))) - (data["av_change_rate_roll_mean_100"]))) * 2.0)) + (data["avg_first_10000"]))))) * 2.0)))))) +
            0.0371556953*np.tanh(((((data["max_last_10000"]) * (((((data["std_last_50000"]) * (data["avg_last_50000"]))) + (((((((data["min_roll_std_10"]) * 2.0)) + (((data["min_last_10000"]) * (((data["max_first_50000"]) * (((((data["min_roll_std_10"]) * 2.0)) * 2.0)))))))) * (data["max_first_10000"]))))))) * 2.0)) +
            0.0399609283*np.tanh(((data["av_change_abs_roll_std_10"]) * ((((((((((data["min_last_50000"]) + (((data["trend"]) * (data["av_change_abs_roll_mean_10"]))))/2.0)) + (data["max_to_min"]))/2.0)) + (((data["min_last_50000"]) - (((data["min_last_50000"]) * (((data["min_last_50000"]) * (data["q05_roll_mean_1000"]))))))))/2.0)))) +
            0.0391638987*np.tanh(((((data["trend"]) - ((((data["min_roll_std_100"]) + (data["q001"]))/2.0)))) * (((np.tanh((np.tanh((((((data["classic_sta_lta3_mean"]) + ((((data["classic_sta_lta3_mean"]) + ((((data["kurt"]) + (data["q95_roll_mean_100"]))/2.0)))/2.0)))) * 2.0)))))) * ((-1.0*((data["min_roll_std_100"])))))))) +
            0.0398749746*np.tanh(((data["ave10"]) * ((-1.0*((((data["av_change_abs_roll_mean_10"]) * (((data["av_change_abs_roll_mean_100"]) + ((-1.0*(((((data["av_change_abs_roll_mean_10"]) + (((((data["classic_sta_lta2_mean"]) * 2.0)) + (((data["mean_change_rate_last_10000"]) * (data["av_change_rate_roll_mean_100"]))))))/2.0)))))))))))))) +
            0.0385778472*np.tanh((-1.0*(((((np.tanh((((((((data["trend"]) + (((data["min_roll_std_10"]) + (data["Moving_average_6000_mean"]))))) * 2.0)) * 2.0)))) + (((((-1.0*((((data["q95_roll_std_100"]) * (data["ave10"])))))) + (((data["mean_change_rate_first_50000"]) * (data["av_change_abs_roll_mean_10"]))))/2.0)))/2.0))))) +
            0.0399921872*np.tanh(((((((((data["min_roll_std_10"]) * ((((data["ave_roll_std_10"]) + (data["avg_last_50000"]))/2.0)))) + (((((((((data["avg_last_10000"]) + (data["min_roll_std_10"]))/2.0)) + (data["min_roll_std_10"]))/2.0)) - (data["ave_roll_std_10"]))))/2.0)) + (((((data["q95_roll_std_100"]) - (data["ave_roll_std_10"]))) * 2.0)))/2.0)) +
            0.0399921872*np.tanh((((((-1.0*((((data["mean_change_rate_first_10000"]) * 2.0))))) * ((((((((((data["av_change_rate_roll_std_10"]) * (data["avg_last_10000"]))) + (((data["min_last_10000"]) * (data["av_change_rate_roll_std_10"]))))) - (((data["avg_first_10000"]) / 2.0)))) + (data["av_change_rate_roll_std_1000"]))/2.0)))) * (data["av_change_abs_roll_std_1000"]))) +
            0.0399453007*np.tanh(((data["av_change_abs_roll_std_1000"]) * (((data["avg_last_10000"]) * (((((((data["avg_last_10000"]) * (((data["avg_last_10000"]) * (((((data["med"]) * (((data["q001"]) - (data["classic_sta_lta1_mean"]))))) - (data["min_roll_std_10"]))))))) - (data["max_roll_mean_10"]))) - (data["min_roll_std_10"]))))))) +
            0.0398749746*np.tanh(np.tanh((np.tanh((((((((((((-1.0*((data["av_change_abs_roll_std_100"])))) + ((-1.0*((data["mean_change_rate_first_50000"])))))/2.0)) / 2.0)) - (((data["mean_diff"]) * (((((((data["avg_last_50000"]) * (data["q99_roll_mean_10"]))) - (data["mean_change_rate_first_50000"]))) - (data["std_last_10000"]))))))) * 2.0)))))) +
            0.0363899209*np.tanh(((data["max_last_10000"]) * (((((np.tanh((((((((data["av_change_rate_roll_mean_100"]) + (((data["min_roll_std_1000"]) * 2.0)))) * 2.0)) * 2.0)))) + (((data["av_change_abs_roll_std_100"]) + ((-1.0*((data["mean_diff"])))))))) + (((((data["av_change_rate_roll_mean_10"]) * (data["max_last_50000"]))) * 2.0)))))) +
            0.0399999991*np.tanh(((data["min_roll_std_100"]) * (((np.tanh((data["min_last_10000"]))) + (((((np.tanh((data["std_first_10000"]))) - ((((np.tanh((data["min_last_10000"]))) + (((data["exp_Moving_average_6000_mean"]) * (data["min_last_10000"]))))/2.0)))) * ((((((data["avg_last_50000"]) + (data["mean_diff"]))/2.0)) * 2.0)))))))) +
            0.0351396762*np.tanh(np.tanh((((((data["av_change_abs_roll_std_1000"]) * (data["mean_change_rate_last_10000"]))) + (((data["med"]) * (np.tanh((((data["av_change_rate_roll_std_100"]) - (((((data["avg_last_10000"]) - (((data["mean_change_rate_first_10000"]) - (((data["avg_last_10000"]) * (data["mean_change_rate_last_10000"]))))))) - (data["av_change_abs_roll_std_1000"]))))))))))))) +
            0.0399921872*np.tanh((((np.tanh((((((((((data["min_roll_std_10"]) + (((data["mean_change_rate_first_50000"]) + (((((data["mean_change_rate_first_50000"]) + (data["mean_change_rate_first_50000"]))) * (data["classic_sta_lta4_mean"]))))))) * (data["av_change_rate_roll_mean_1000"]))) * 2.0)) * 2.0)))) + (((data["av_change_abs_roll_mean_10"]) * (np.tanh((data["med"]))))))/2.0)) +
            0.0379214697*np.tanh((-1.0*((((((data["max_to_min_diff"]) * (((((((((((((data["min_last_10000"]) - (data["med"]))) - (data["mean_change_rate_first_10000"]))) + (data["q05_roll_std_100"]))) + (data["q01_roll_std_1000"]))) * 2.0)) * 2.0)))) * ((-1.0*((((data["mean_change_rate_first_10000"]) * (data["min_last_10000"]))))))))))) +
            0.0399218611*np.tanh(((data["min_first_10000"]) * (np.tanh((((((((((np.tanh((((data["max_to_min"]) * 2.0)))) + (((((((data["min_first_50000"]) * 2.0)) + (((np.tanh((np.tanh((data["mean_change_rate_last_10000"]))))) + (data["min_roll_mean_10"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))) +
            0.0328423530*np.tanh((-1.0*((((np.tanh((((data["kurt"]) * (((((((data["mean_change_rate_last_50000"]) * 2.0)) * 2.0)) * (data["skew"]))))))) + (np.tanh((((data["skew"]) * (((((data["mean_change_rate_last_50000"]) * 2.0)) * (data["skew"])))))))))))) +
            0.0346083194*np.tanh((((((data["av_change_abs_roll_std_10"]) / 2.0)) + ((-1.0*((((np.tanh((data["q05_roll_mean_1000"]))) + (np.tanh((((((data["min_roll_std_100"]) - (data["mean_change_rate_last_50000"]))) - (((((data["q95_roll_mean_1000"]) - (data["av_change_abs_roll_std_10"]))) - (((data["av_change_abs_roll_mean_1000"]) - (data["mean_change_rate_last_50000"])))))))))))))))/2.0)) +
            0.0395155288*np.tanh(((data["avg_last_50000"]) * ((-1.0*(((((np.tanh((np.tanh((((((((((((data["av_change_rate_roll_mean_100"]) * ((-1.0*((data["max_to_min"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))) + ((((((data["av_change_rate_roll_std_10"]) * (data["avg_last_10000"]))) + (data["av_change_rate_roll_std_10"]))/2.0)))/2.0))))))) +
            0.0316311792*np.tanh(((data["max_first_50000"]) - (((data["classic_sta_lta3_mean"]) * (((((data["max_to_min_diff"]) + (data["mean_diff"]))) - (((data["trend"]) * (((((data["mean_diff"]) - (((np.tanh((data["max_to_min_diff"]))) * 2.0)))) - (data["av_change_abs_roll_mean_100"]))))))))))) +
            0.0399609283*np.tanh(((data["trend"]) * (((((((data["min_roll_std_1000"]) * (data["std_last_10000"]))) * ((((data["av_change_abs_roll_mean_10"]) + (((data["std_last_50000"]) - ((((((data["av_change_rate_roll_mean_10"]) * (data["q05_roll_std_1000"]))) + (data["min_roll_std_1000"]))/2.0)))))/2.0)))) + (((data["min_roll_std_1000"]) - (data["q05_roll_std_1000"]))))))) +
            0.0362180099*np.tanh((((np.tanh((((((((((((data["q01_roll_std_100"]) - (data["mean_change_rate_last_10000"]))) - ((((data["av_change_abs_roll_mean_10"]) + (data["abs_max_roll_mean_100"]))/2.0)))) - (0.2525250018))) * (data["min_roll_std_10"]))) * (73.0))))) + (((data["min_roll_mean_1000"]) + (data["abs_max_roll_mean_100"]))))/2.0)) +
            0.0399999991*np.tanh((((np.tanh((((73.0) * (((((73.0) * (data["av_change_abs_roll_std_100"]))) - (data["av_change_abs_roll_mean_100"]))))))) + (np.tanh((((((((((data["av_change_rate_roll_mean_100"]) - (data["av_change_abs_roll_std_100"]))) - (data["av_change_abs_roll_std_100"]))) - (data["std_first_50000"]))) - (data["av_change_abs_roll_std_100"]))))))/2.0)) +
            0.0374213718*np.tanh((((-1.0*((((data["q05"]) * (((data["max_last_10000"]) * (((((data["min_roll_std_10"]) - (data["std_last_10000"]))) - (data["std_last_10000"])))))))))) + (((data["max_last_50000"]) * (((data["mean_change_rate_last_10000"]) - (np.tanh((((((data["av_change_rate_roll_std_10"]) * 2.0)) * 2.0)))))))))) +
            0.0331470966*np.tanh(((((((((data["abs_max_roll_std_1000"]) * 2.0)) * 2.0)) * 2.0)) * (((((data["mean_change_rate_last_50000"]) * ((((((data["abs_max_roll_std_1000"]) * (((data["mean_diff"]) * 2.0)))) + ((((data["std_first_50000"]) + (data["mean_change_rate_last_50000"]))/2.0)))/2.0)))) + (((data["mean_diff"]) * (data["av_change_rate_roll_std_1000"]))))))) +
            0.0396796241*np.tanh(((np.tanh((np.tanh((((((data["avg_first_10000"]) * (((((data["Moving_average_3000_mean"]) + (data["skew"]))) * 2.0)))) + (np.tanh(((-1.0*((((((data["ave_roll_mean_100"]) + (((data["av_change_abs_roll_std_1000"]) + (((data["avg_first_10000"]) * 2.0)))))) * 2.0))))))))))))) / 2.0)) +
            0.0399765596*np.tanh(((np.tanh((((((data["av_change_abs_roll_mean_10"]) + (data["exp_Moving_average_300_mean"]))) * (((((data["classic_sta_lta4_mean"]) + (data["av_change_abs_roll_mean_10"]))) + (((((data["av_change_abs_roll_mean_10"]) + (((np.tanh((((np.tanh((data["avg_last_50000"]))) * 2.0)))) * 2.0)))) * (data["av_change_abs_roll_mean_10"]))))))))) / 2.0)) +
            0.0396014862*np.tanh((((((data["max_last_50000"]) + (((((np.tanh((np.tanh((((((np.tanh((np.tanh(((((data["q95"]) + (data["mean_diff"]))/2.0)))))) + (((data["av_change_abs_roll_std_100"]) * 2.0)))) * 2.0)))))) * 2.0)) - (data["av_change_abs_roll_std_100"]))))/2.0)) * (data["classic_sta_lta4_mean"]))) +
            0.0317015015*np.tanh(((np.tanh((((((data["avg_last_10000"]) - (((data["iqr"]) - (((((data["avg_last_10000"]) - (((data["iqr"]) * (data["av_change_rate_roll_std_100"]))))) - (((((((((data["min_roll_std_100"]) * (data["av_change_rate_roll_std_100"]))) * 2.0)) * 2.0)) * 2.0)))))))) * 2.0)))) / 2.0)) +
            0.0399296731*np.tanh((-1.0*((((data["q05_roll_std_1000"]) * (((((((((data["avg_first_10000"]) + (data["mean_diff"]))/2.0)) * (data["std_last_50000"]))) + (((((((data["avg_last_10000"]) + (((data["mean_change_rate_first_10000"]) * (data["mean_diff"]))))/2.0)) + (data["max_to_min_diff"]))/2.0)))/2.0))))))) +
            0.0377104878*np.tanh((((np.tanh((np.tanh((((((((data["count_big"]) + (((data["min_last_10000"]) * (data["min_first_50000"]))))) * 2.0)) * 2.0)))))) + ((((data["count_big"]) + ((-1.0*((((data["max_roll_mean_100"]) - (np.tanh((((data["min_last_10000"]) * (data["avg_last_10000"])))))))))))/2.0)))/2.0)) +
            0.0309357308*np.tanh(((np.tanh((((((((data["max_to_min"]) * (data["classic_sta_lta2_mean"]))) + (((data["classic_sta_lta2_mean"]) * (((data["max_to_min"]) * (data["classic_sta_lta2_mean"]))))))) * 2.0)))) / 2.0)) +
            0.0355303772*np.tanh(((((np.tanh((np.tanh((((((((((data["av_change_abs_roll_std_10"]) - (np.tanh((((data["skew"]) * 2.0)))))) * (data["mean_change_rate_first_10000"]))) * (data["mean_change_rate_first_10000"]))) * (((data["mean_change_rate_first_10000"]) + (data["av_change_abs_roll_std_10"]))))))))) * (data["mean_change_rate_first_10000"]))) * (data["mean_change_rate_first_10000"]))) +
            0.0399843715*np.tanh(((data["av_change_abs_roll_std_1000"]) * (((((((data["trend"]) + (np.tanh((((data["min_roll_std_1000"]) * ((((8.0)) + (((data["min_roll_std_1000"]) * ((8.0)))))))))))/2.0)) + ((-1.0*((((data["q01_roll_std_1000"]) + (((data["q01_roll_std_1000"]) * (data["mean_change_rate_first_50000"])))))))))/2.0)))) +
            0.0398437195*np.tanh((-1.0*((((data["std_first_10000"]) * ((((((data["q99"]) / 2.0)) + (((data["skew"]) + (((((data["std_first_10000"]) * 2.0)) * (((data["std_first_10000"]) * (((data["std_first_10000"]) * (((data["std_first_10000"]) - (((data["MA_700MA_std_mean"]) * 2.0)))))))))))))/2.0))))))) +
            0.0399843715*np.tanh((-1.0*((((data["mean_diff"]) * ((((((data["av_change_abs_roll_mean_1000"]) + ((((((((data["abs_max_roll_mean_10"]) + (((((data["min_roll_mean_100"]) * (data["av_change_abs_roll_mean_1000"]))) * (data["min_roll_std_10"]))))) - ((((data["min_roll_std_10"]) + (data["av_change_abs_roll_std_100"]))/2.0)))) + (data["skew"]))/2.0)))/2.0)) / 2.0))))))) +
            0.0295057632*np.tanh(((((((((((((((data["q99_roll_mean_100"]) + (((data["q99_roll_mean_100"]) + (data["q95_roll_mean_10"]))))) * (((data["q05_roll_std_100"]) + ((((((data["av_change_abs_roll_std_1000"]) - (data["min_roll_std_100"]))) + (data["av_change_abs_roll_mean_100"]))/2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (data["av_change_abs_roll_std_1000"]))) +
            0.0346864611*np.tanh(((np.tanh((((((((data["mean_change_rate"]) * (((data["mean_change_rate_first_10000"]) + (((data["mean_change_rate_first_10000"]) + (data["mean_change_rate_last_50000"]))))))) + (((data["q01_roll_std_100"]) * (((((data["mean_change_rate_first_10000"]) * (((data["q99_roll_std_10"]) * 2.0)))) + (data["mean_change_rate_first_10000"]))))))) * 2.0)))) / 2.0)) +
            0.0397265106*np.tanh(np.tanh((((((data["min_roll_std_100"]) * ((-1.0*((((data["av_change_abs_roll_mean_100"]) * ((((data["q05_roll_mean_100"]) + (data["mean_change_rate_first_10000"]))/2.0))))))))) - (((((((-1.0*((data["classic_sta_lta4_mean"])))) / 2.0)) + ((((data["av_change_rate_roll_mean_100"]) + ((((data["Moving_average_6000_mean"]) + (data["mean_change_rate_first_10000"]))/2.0)))/2.0)))/2.0)))))) +
            0.0399687439*np.tanh(((np.tanh((((data["min_first_50000"]) * (((((((((np.tanh((((((data["q01_roll_std_1000"]) + (np.tanh(((-1.0*((((((data["skew"]) + (data["classic_sta_lta2_mean"]))) + (data["q95_roll_std_10"])))))))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))) / 2.0)) +
            0.0365696438*np.tanh(((np.tanh((((((((((((((((data["mean_change_rate_last_10000"]) * (((data["ave_roll_std_10"]) - (((data["q95"]) - ((((((data["abs_mean"]) + (((data["iqr"]) / 2.0)))/2.0)) / 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) / 2.0)) +
            0.0393358096*np.tanh(((((data["max_roll_std_10"]) * (((((data["q95_roll_mean_100"]) + (((data["q95_roll_mean_100"]) + (((data["q05_roll_mean_1000"]) * (data["classic_sta_lta2_mean"]))))))) * (((data["max_roll_std_10"]) * (((((data["skew"]) * (((data["q05_roll_mean_100"]) + (data["classic_sta_lta2_mean"]))))) * 2.0)))))))) * 2.0)) +
            0.0381871462*np.tanh((((np.tanh((((((((data["q05_roll_mean_10"]) - (((data["q05"]) + (data["ave_roll_std_100"]))))) * (73.0))) * (73.0))))) + (((((((data["abs_q05"]) * (((data["std_roll_mean_1000"]) * (data["av_change_rate_roll_std_10"]))))) * 2.0)) - (data["q05"]))))/2.0)) +
            0.0383043550*np.tanh(((data["max_last_50000"]) * (((((data["av_change_rate_roll_std_100"]) * 2.0)) * (np.tanh((((((((((((((np.tanh((((data["mean_diff"]) - (data["classic_sta_lta3_mean"]))))) - (data["classic_sta_lta3_mean"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))))) +
            0.0397812054*np.tanh(np.tanh((np.tanh((((data["min_roll_mean_100"]) * (((((data["av_change_rate_roll_mean_1000"]) - (data["min_roll_mean_100"]))) - (((((((data["min_last_50000"]) * (((data["av_change_rate_roll_mean_1000"]) - (np.tanh((((data["min_roll_std_10"]) - (((data["min_roll_mean_100"]) * 2.0)))))))))) * 2.0)) * 2.0)))))))))) +
            0.0399921872*np.tanh((-1.0*((((((np.tanh((((((((((((((((((data["mean_diff"]) * 2.0)) - ((((((-1.0*((data["mean_change_rate_last_50000"])))) * 2.0)) - (data["trend"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) / 2.0)) / 2.0))))) +
            0.0329673775*np.tanh(np.tanh(((-1.0*((((data["std_first_10000"]) * (((((((((data["av_change_abs_roll_mean_10"]) + (data["max_last_10000"]))) * (((((data["max_last_10000"]) * (0.8873239756))) + (data["classic_sta_lta2_mean"]))))) * 2.0)) * (data["max_last_10000"])))))))))) +
            0.0367962494*np.tanh(((data["avg_last_10000"]) * (((((data["std_roll_mean_1000"]) * ((-1.0*(((((((((data["mean_change_rate_first_50000"]) - (((data["mean_diff"]) * (data["max_last_10000"]))))) + (data["max_to_min_diff"]))/2.0)) - (((data["max_last_10000"]) * (((data["max_to_min"]) * (data["q05_roll_mean_100"])))))))))))) * 2.0)))) +
            0.0399921872*np.tanh(((data["av_change_abs_roll_mean_10"]) * (((((data["av_change_abs_roll_std_1000"]) - ((((data["av_change_abs_roll_mean_10"]) + ((((data["min_roll_std_10"]) + (((((((data["av_change_rate_roll_mean_10"]) * (data["av_change_abs_roll_std_10"]))) + (data["av_change_abs_roll_mean_10"]))) * (data["av_change_abs_roll_mean_10"]))))/2.0)))/2.0)))) * (data["min_roll_std_10"]))))) +
            0.0395311601*np.tanh(((np.tanh(((-1.0*((((data["std_first_10000"]) + (data["min_first_10000"])))))))) + (np.tanh((((data["max_roll_std_10"]) * (((data["q95_roll_mean_100"]) * ((((((((data["mean_diff"]) + (data["max_roll_std_10"]))/2.0)) - ((-1.0*((data["min_first_10000"])))))) + (data["mean_change_rate_last_50000"]))))))))))) +
            0.0399296731*np.tanh(((((data["min_roll_std_10"]) + (((data["min_roll_std_10"]) * (data["med"]))))) * ((((((data["classic_sta_lta1_mean"]) + (data["min_roll_mean_1000"]))/2.0)) - (((data["q05_roll_std_10"]) * ((((((data["med"]) * ((((data["classic_sta_lta1_mean"]) + (data["min_roll_mean_1000"]))/2.0)))) + (data["med"]))/2.0)))))))) +
            0.0399453007*np.tanh(np.tanh((((((((np.tanh((data["min_roll_std_10"]))) * (data["std_roll_mean_1000"]))) * 2.0)) + ((((((((((((data["max_to_min_diff"]) * 2.0)) * (data["std_last_50000"]))) + (data["min_roll_std_10"]))/2.0)) / 2.0)) + (((data["min_roll_std_10"]) * (np.tanh((data["max_to_min_diff"]))))))))))) +
            0.0394999012*np.tanh(((data["mean_change_rate_first_10000"]) * (((data["min_first_10000"]) * ((((((((data["min_roll_std_10"]) * (((data["av_change_abs_roll_std_100"]) + (data["av_change_abs_roll_std_100"]))))) + ((((data["mean_change_rate_first_10000"]) + (((data["min_roll_std_10"]) * (data["mean_change_rate_first_10000"]))))/2.0)))/2.0)) + (((data["min_roll_std_10"]) - (data["av_change_abs_roll_std_100"]))))))))) +
            0.0399687439*np.tanh(((data["max_first_10000"]) * (((((data["std_first_10000"]) * (((data["max_first_10000"]) * (((data["std_first_10000"]) * (((np.tanh((((data["std_first_10000"]) * (data["av_change_abs_roll_mean_10"]))))) + (data["std_roll_mean_100"]))))))))) - (((((data["av_change_abs_roll_mean_10"]) * (data["min_roll_std_100"]))) / 2.0)))))) +
            0.0397499502*np.tanh(((data["trend"]) * ((((((data["std_first_50000"]) + (np.tanh((np.tanh((((data["av_change_rate_roll_mean_10"]) + (((((((data["av_change_rate_roll_mean_1000"]) + ((((data["min_roll_std_100"]) + (((data["Hilbert_mean"]) + (data["av_change_rate_roll_mean_10"]))))/2.0)))) * 2.0)) * 2.0)))))))))/2.0)) * (data["min_roll_std_100"]))))) +
            0.0274897423*np.tanh((((np.tanh((((((((((((((data["q95_roll_std_100"]) - (data["ave_roll_std_10"]))) - (data["abs_max_roll_mean_10"]))) - (data["max_last_50000"]))) - (((data["min_first_50000"]) + (data["ave_roll_std_10"]))))) * 2.0)) * 2.0)))) + (((((data["q95_roll_std_100"]) - (data["ave_roll_std_10"]))) * 2.0)))/2.0)) +
            0.0399531126*np.tanh(((((data["mean_change_rate_first_10000"]) * (np.tanh((((((((((data["av_change_abs_roll_mean_10"]) + (data["mean_change_rate_last_10000"]))/2.0)) * (((data["avg_first_10000"]) - (np.tanh((data["av_change_abs_roll_mean_100"]))))))) + (np.tanh((np.tanh((data["avg_first_10000"]))))))/2.0)))))) * ((((-1.0*((data["mean_diff"])))) * 2.0)))) +
            0.0340613388*np.tanh(((data["skew"]) * (((data["max_first_50000"]) * (((((((((np.tanh((((((((data["iqr"]) * 2.0)) - (data["av_change_abs_roll_mean_10"]))) - (data["std_last_10000"]))))) + (data["av_change_abs_roll_mean_10"]))) - (((data["av_change_rate_roll_mean_10"]) * (data["avg_first_50000"]))))) * 2.0)) * 2.0)))))) +
            0.0399921872*np.tanh((((((data["min_roll_std_10"]) + (((data["min_roll_std_10"]) * (data["avg_first_10000"]))))/2.0)) * (np.tanh((((73.0) * (((data["avg_first_10000"]) * (((data["min_roll_std_10"]) + (((73.0) * (((73.0) * ((-1.0*((data["av_change_abs_roll_std_1000"])))))))))))))))))) +
            0.0356475860*np.tanh(np.tanh((((data["av_change_abs_roll_mean_1000"]) * ((((((((((((((data["avg_last_50000"]) + (data["kurt"]))) + (0.2783510089))/2.0)) + (data["av_change_rate_roll_std_1000"]))) + (0.2783510089))) + (data["kurt"]))) * ((((np.tanh((data["avg_first_10000"]))) + (data["mean_change_rate_first_50000"]))/2.0)))))))) +
            0.0329986326*np.tanh((((((((((data["kurt"]) + (data["q05_roll_mean_10"]))) * (data["min_roll_mean_10"]))) + (np.tanh((((((data["q05_roll_mean_10"]) - (((((data["max_to_min"]) - (data["kurt"]))) + (((data["q05_roll_mean_10"]) * (data["max_to_min"]))))))) * 2.0)))))/2.0)) / 2.0)) +
            0.0399999991*np.tanh(((data["std_roll_mean_1000"]) * (((data["med"]) * ((((-3.0) + (((((data["avg_first_10000"]) * 2.0)) * (((data["mean_change_rate_first_10000"]) - (((((((data["std_roll_mean_1000"]) / 2.0)) * (((data["std_roll_mean_1000"]) * (-3.0))))) * (data["abs_max_roll_mean_1000"]))))))))/2.0)))))) +
            0.0303496774*np.tanh((-1.0*((((np.tanh((np.tanh((((((((data["classic_sta_lta2_mean"]) + (((data["av_change_abs_roll_std_10"]) + (((data["av_change_abs_roll_mean_10"]) + (data["classic_sta_lta1_mean"]))))))) * (data["abs_q05"]))) * (3.6923100948))))))) / 2.0))))) +
            0.0399687439*np.tanh((((-1.0*((data["max_first_50000"])))) * (((((data["std_roll_mean_1000"]) + (((data["av_change_abs_roll_std_10"]) * (((data["ave_roll_mean_10"]) + (((data["med"]) + (1.0))))))))) * (((data["avg_last_50000"]) + (((data["med"]) + (2.0769200325))))))))) +
            0.0399843715*np.tanh(((np.tanh((((data["av_change_rate_roll_mean_100"]) * (((np.tanh((((((((((data["av_change_abs_roll_mean_100"]) * 2.0)) + (data["mean_change_rate_first_50000"]))) * 2.0)) * 2.0)))) - (((data["av_change_abs_roll_mean_100"]) * (((((data["av_change_rate_roll_std_1000"]) - (data["classic_sta_lta2_mean"]))) - (data["classic_sta_lta2_mean"]))))))))))) / 2.0)) +
            0.0399999991*np.tanh(((data["mean_diff"]) * ((((data["av_change_abs_roll_std_100"]) + (np.tanh((((((((((((((((((data["min_first_10000"]) - (data["av_change_rate_roll_std_10"]))) - (((data["av_change_abs_roll_std_100"]) * (3.6923100948))))) * 2.0)) * 2.0)) * 2.0)) + (data["mean_diff"]))) * 2.0)) * 2.0)))))/2.0)))) +
            0.0285446383*np.tanh(((np.tanh((((((73.0) * (data["mean_change_rate_first_10000"]))) + (((((((((data["trend"]) + (((data["av_change_abs_roll_mean_10"]) + (data["min_last_50000"]))))) + (data["mean_change_rate_first_10000"]))) * 2.0)) * 2.0)))))) + ((-1.0*((np.tanh((data["mean_change_rate_first_10000"])))))))) +
            0.0399687439*np.tanh(((((np.tanh((((((((((((((((((((np.tanh(((((((data["kurt"]) + (data["classic_sta_lta4_mean"]))/2.0)) - (data["max_first_10000"]))))) + (data["skew"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) / 2.0)) / 2.0)) +
            0.0398593470*np.tanh(((np.tanh(((-1.0*((np.tanh((((data["avg_first_10000"]) * (((data["min_roll_std_10"]) - (((((data["mean_change_rate_last_10000"]) * (data["min_roll_std_10"]))) + ((((data["mean_diff"]) + ((((((data["ave_roll_mean_100"]) + (data["avg_first_10000"]))/2.0)) + (data["skew"]))))/2.0))))))))))))))) / 2.0)) +
            0.0399296731*np.tanh(((np.tanh((np.tanh((np.tanh((np.tanh((((((((data["mean_diff"]) * ((((data["av_change_abs_roll_mean_100"]) + (0.7446810007))/2.0)))) - (((np.tanh((data["skew"]))) + ((((data["trend"]) + (0.7446810007))/2.0)))))) * 2.0)))))))))) / 2.0)) +
            0.0387341268*np.tanh((((((((data["min_roll_std_100"]) * (np.tanh((((((((data["av_change_rate_roll_std_100"]) * 2.0)) + (((data["abs_q05"]) + (np.tanh((data["iqr"]))))))) * 2.0)))))) + (np.tanh((((np.tanh((((data["iqr"]) * 2.0)))) - (data["min_roll_std_10"]))))))/2.0)) / 2.0)) +
            0.0384997055*np.tanh(((data["av_change_abs_roll_mean_1000"]) * (((np.tanh((((((np.tanh((data["std_first_50000"]))) + ((((-1.0*((data["max_to_min"])))) * (data["std_first_10000"]))))) + (((data["av_change_abs_roll_mean_1000"]) * (data["min_roll_mean_1000"]))))))) + ((((-1.0*((data["min_roll_std_10"])))) * (data["std_first_10000"]))))))) +
            0.0384918936*np.tanh(((np.tanh((np.tanh((((((73.0) * (((73.0) * (((((data["q95_roll_std_100"]) + (data["std_roll_mean_1000"]))) * 2.0)))))) * (((data["std_first_50000"]) + (((np.tanh((np.tanh((data["max_roll_std_1000"]))))) * (data["kurt"]))))))))))) / 2.0)) +
            0.0343192033*np.tanh(((((-1.0) - (np.tanh((((((((0.8873239756) + (data["mean_diff"]))) - (data["Hann_window_mean"]))) * (((((((((data["Hann_window_mean"]) - (data["min_last_10000"]))) * 2.0)) * 2.0)) + (data["av_change_rate_roll_std_100"]))))))))) / 2.0)) +
            0.0372416489*np.tanh(((np.tanh((((((data["classic_sta_lta2_mean"]) + (((data["q95_roll_mean_1000"]) * (data["classic_sta_lta2_mean"]))))) + ((((-1.0*((((((data["std_last_10000"]) + (data["MA_400MA_BB_low_mean"]))) + (np.tanh((((data["min_roll_std_100"]) + (data["classic_sta_lta2_mean"])))))))))) * (data["avg_last_10000"]))))))) / 2.0)) +
            0.0377808139*np.tanh(np.tanh((((((data["mean_change_rate_last_10000"]) * (((((((data["mean_change_rate_first_10000"]) + ((((data["min_roll_std_10"]) + ((-1.0*((-3.0)))))/2.0)))/2.0)) + (data["Moving_average_1500_mean"]))/2.0)))) * (np.tanh((((((data["mean_change_rate_last_10000"]) + (((data["av_change_abs_roll_std_1000"]) + (1.9019600153))))) * 2.0)))))))) +
            0.0369056463*np.tanh(np.tanh((np.tanh((((((data["max_first_50000"]) * (data["max_first_50000"]))) * ((((((((data["max_first_50000"]) + (data["kurt"]))/2.0)) - (data["avg_last_50000"]))) - (((np.tanh((((data["mean_change_rate_first_10000"]) * 2.0)))) - (((data["kurt"]) - (0.5555559993))))))))))))))

def gpi1(data):
    return (-0.000008 +
            1.0*np.tanh(((((data["ave_roll_std_100"]) + ((((-1.0*((((data["q95_roll_mean_1000"]) + (((((data["sum"]) * 2.0)) * 2.0))))))) - (data["mean"]))))) * 2.0)) +
            1.0*np.tanh((-1.0*((((data["med"]) + (((data["mean_change_rate"]) + (((((data["exp_Moving_average_6000_mean"]) + (data["q05_roll_mean_100"]))) * 2.0))))))))) +
            1.0*np.tanh((((-1.0*((((data["med"]) * 2.0))))) - (((((((data["q05_roll_mean_100"]) * 2.0)) - ((-1.0*((data["q001"])))))) * 2.0)))) +
            1.0*np.tanh((((((((-1.0*((data["q01"])))) - (((data["ave_roll_mean_1000"]) + (((data["av_change_rate_roll_mean_1000"]) + (((data["Moving_average_6000_mean"]) * 2.0)))))))) * 2.0)) * 2.0)) +
            1.0*np.tanh(((((data["q01_roll_std_100"]) - (((data["exp_Moving_average_300_mean"]) * 2.0)))) + ((-1.0*((((data["med"]) * 2.0))))))) +
            1.0*np.tanh(((data["q05_roll_std_100"]) - (((data["ave_roll_mean_1000"]) + (((((((data["ave_roll_mean_100"]) * 2.0)) + (data["q99_roll_mean_1000"]))) + (data["exp_Moving_average_300_mean"]))))))) +
            1.0*np.tanh((((-1.0*((((data["av_change_rate_roll_mean_100"]) + (data["sum"])))))) - (((data["q95_roll_mean_1000"]) + (((data["med"]) + (data["exp_Moving_average_3000_mean"]))))))) +
            1.0*np.tanh(((((data["q05_roll_std_100"]) - (((data["q05_roll_mean_1000"]) + (((data["Moving_average_700_mean"]) + (data["q05_roll_mean_100"]))))))) - (data["Moving_average_6000_mean"]))) +
            1.0*np.tanh(((data["q99_roll_mean_1000"]) - (((data["med"]) + (((((((((data["q99_roll_mean_1000"]) * 2.0)) * 2.0)) + (((data["med"]) * 2.0)))) * 2.0)))))) +
            1.0*np.tanh(((((((-1.0) - (((data["mean"]) * 2.0)))) * 2.0)) * 2.0)) +
            1.0*np.tanh((((-1.0*((((((data["abs_q05"]) * 2.0)) - (data["MA_400MA_std_mean"])))))) - (((data["med"]) + (np.tanh((((data["med"]) * 2.0)))))))) +
            1.0*np.tanh((-1.0*((((((((data["abs_q05"]) * 2.0)) - (((data["q05_roll_std_1000"]) / 2.0)))) - (((((data["max_roll_mean_10"]) - (data["Moving_average_3000_mean"]))) * 2.0))))))) +
            1.0*np.tanh((-1.0*((((data["Moving_average_1500_mean"]) + (((data["Moving_average_3000_mean"]) + (((data["q05_roll_mean_100"]) + (data["max"])))))))))) +
            1.0*np.tanh((((-1.0*((((((((data["med"]) + (data["abs_max_roll_mean_1000"]))) * 2.0)) + (data["med"])))))) * 2.0)) +
            1.0*np.tanh(((data["max_roll_mean_100"]) - (((((((data["abs_q05"]) + (((data["abs_q05"]) * 2.0)))) * 2.0)) + (((data["abs_q05"]) * 2.0)))))) +
            1.0*np.tanh(((data["ave10"]) + ((((((-1.0*((((data["med"]) - (data["abs_max_roll_std_1000"])))))) * 2.0)) - (((data["abs_q05"]) * 2.0)))))) +
            1.0*np.tanh((((-1.0*((((data["ave_roll_mean_10"]) - (data["q99_roll_std_10"])))))) - (((data["abs_q05"]) - ((-1.0*((((data["abs_q05"]) - (data["q95_roll_std_10"])))))))))) +
            1.0*np.tanh((((((((-1.0*((((data["med"]) * 2.0))))) * 2.0)) - (((data["q99_roll_mean_1000"]) * 2.0)))) * 2.0)) +
            1.0*np.tanh(((data["classic_sta_lta1_mean"]) - (((data["abs_q05"]) + (((((data["abs_q05"]) - (data["max_last_10000"]))) * 2.0)))))) +
            1.0*np.tanh(((((3.0) + (((data["av_change_rate_roll_mean_1000"]) - (data["ave_roll_mean_1000"]))))) - (((((data["ave_roll_mean_100"]) - (data["ave_roll_std_10"]))) * 2.0)))) +
            1.0*np.tanh((-1.0*((((((data["abs_q05"]) + ((((data["abs_max_roll_mean_10"]) + ((((data["abs_q05"]) + (data["med"]))/2.0)))/2.0)))) * 2.0))))) +
            1.0*np.tanh(((((data["q01_roll_std_100"]) + ((((4.00706386566162109)) - (((((data["exp_Moving_average_300_mean"]) + (data["med"]))) * 2.0)))))) * 2.0)) +
            1.0*np.tanh(((((((data["abs_q05"]) * 2.0)) - (((((((data["abs_q05"]) + (data["abs_q05"]))) * 2.0)) * 2.0)))) - (data["abs_q05"]))) +
            1.0*np.tanh(((((((data["q05_roll_std_1000"]) + (((((((np.tanh((3.0))) - (data["Hann_window_mean"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
            1.0*np.tanh(((((((((data["q05_roll_mean_1000"]) * (((data["abs_q05"]) + (((data["abs_q05"]) - (data["Moving_average_6000_mean"]))))))) * 2.0)) * 2.0)) - (data["abs_q05"]))) +
            1.0*np.tanh(((((data["med"]) * (((((data["std_roll_mean_10"]) * 2.0)) * 2.0)))) - (((data["ave_roll_mean_100"]) * (((data["avg_first_50000"]) * (data["q95_roll_mean_100"]))))))) +
            1.0*np.tanh((((-1.0*((((data["iqr"]) * (((((data["abs_mean"]) / 2.0)) - (data["q05_roll_mean_100"])))))))) - (data["abs_q05"]))) +
            1.0*np.tanh((((((data["ave10"]) + (data["Moving_average_700_mean"]))) + (((data["Moving_average_700_mean"]) * (((data["ave10"]) * ((((-1.0*((data["Moving_average_700_mean"])))) * 2.0)))))))/2.0)) +
            1.0*np.tanh(((((((data["iqr"]) * (((((((data["med"]) * 2.0)) * 2.0)) + (data["med"]))))) - (data["Moving_average_6000_mean"]))) - (data["avg_last_10000"]))) +
            1.0*np.tanh(((data["iqr"]) - (((data["Moving_average_1500_mean"]) + (((np.tanh(((-1.0*((((data["av_change_rate_roll_std_100"]) * (data["med"])))))))) * 2.0)))))) +
            1.0*np.tanh(((((data["Moving_average_6000_mean"]) * 2.0)) * (((((data["std_last_50000"]) + (((data["abs_q05"]) * 2.0)))) - (data["ave_roll_mean_100"]))))) +
            1.0*np.tanh((((((((data["av_change_rate_roll_std_10"]) * (data["med"]))) - (((data["q01_roll_mean_1000"]) * (((data["avg_last_10000"]) * (data["q05_roll_mean_1000"]))))))) + (data["q01_roll_mean_1000"]))/2.0)) +
            1.0*np.tanh(((data["ave_roll_mean_10"]) + ((((data["Moving_average_1500_mean"]) + (((data["exp_Moving_average_300_mean"]) * (((((data["q05_roll_mean_1000"]) * ((-1.0*((data["exp_Moving_average_6000_mean"])))))) * 2.0)))))/2.0)))) +
            1.0*np.tanh((((((data["std_roll_std_100"]) + ((-1.0*(((-1.0*((data["q05_roll_mean_1000"]))))))))/2.0)) * (((((data["Moving_average_700_mean"]) / 2.0)) * ((-1.0*((data["Moving_average_700_mean"])))))))) +
            1.0*np.tanh(((((data["MA_700MA_BB_high_mean"]) + (data["std_last_10000"]))) * (((((data["med"]) + (data["abs_trend"]))) - (((data["Moving_average_6000_mean"]) * (data["av_change_rate_roll_mean_1000"]))))))) +
            1.0*np.tanh(((data["abs_q05"]) * (((data["av_change_rate_roll_mean_100"]) * (((data["MA_700MA_BB_low_mean"]) * 2.0)))))) +
            0.952325*np.tanh(((((data["max_last_50000"]) * (np.tanh((((data["av_change_rate_roll_mean_100"]) * (((((data["iqr"]) + (data["iqr"]))) + (data["iqr"]))))))))) * 2.0)) +
            1.0*np.tanh(((((data["q99_roll_mean_100"]) * ((-1.0*((data["avg_last_10000"])))))) * (((data["sum"]) - (data["q05_roll_std_10"]))))) +
            0.957405*np.tanh(np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((((np.tanh((((data["sum"]) * 2.0)))) * 2.0)))))))))))))) +
            1.0*np.tanh(((data["max_roll_std_100"]) * ((((data["ave_roll_mean_1000"]) + ((((((data["std_roll_mean_100"]) + (((data["av_change_abs_roll_std_100"]) + (data["mad"]))))/2.0)) - (data["iqr"]))))/2.0)))) +
            1.0*np.tanh(((((data["std_roll_mean_1000"]) / 2.0)) - (((data["classic_sta_lta4_mean"]) * (((((-1.0*((data["med"])))) + (((data["q05_roll_std_100"]) * (data["abs_q05"]))))/2.0)))))) +
            1.0*np.tanh(((data["q05_roll_mean_100"]) * (((data["abs_max_roll_mean_1000"]) * (((data["abs_max_roll_mean_10"]) - (((data["q05_roll_mean_100"]) / 2.0)))))))) +
            0.885893*np.tanh((((((((data["mean_change_rate_last_10000"]) + (((((np.tanh((data["abs_q05"]))) / 2.0)) / 2.0)))) + (data["abs_q05"]))/2.0)) / 2.0)) +
            1.0*np.tanh(((data["q95_roll_mean_10"]) * ((((data["q05_roll_mean_10"]) + (((((((((data["MA_700MA_BB_low_mean"]) * (data["abs_q99"]))) + (data["abs_max_roll_mean_1000"]))) / 2.0)) * 2.0)))/2.0)))) +
            0.922626*np.tanh(((((((data["std_roll_std_100"]) * (((data["max"]) + (data["std_roll_mean_100"]))))) * (data["max_roll_std_100"]))) * (data["med"]))) +
            1.0*np.tanh(((((data["abs_q05"]) * (((data["abs_max_roll_std_100"]) * (((data["abs_max_roll_std_1000"]) * (data["av_change_rate_roll_std_100"]))))))) * (data["max_roll_std_100"]))) +
            0.999609*np.tanh(((np.tanh(((((((np.tanh((((data["abs_max_roll_mean_1000"]) / 2.0)))) - (((data["avg_first_10000"]) / 2.0)))) + (data["min_last_10000"]))/2.0)))) / 2.0)) +
            0.854631*np.tanh(((np.tanh((((data["q05_roll_mean_1000"]) / 2.0)))) * ((((((data["mean"]) + (((data["abs_max_roll_std_10"]) * (data["mean"]))))/2.0)) * (data["q95_roll_std_1000"]))))) +
            0.966784*np.tanh(((data["max_roll_mean_1000"]) * (((data["q95_roll_mean_1000"]) - (((data["max_roll_mean_1000"]) * (data["abs_min"]))))))) +
            1.0*np.tanh(((data["avg_first_10000"]) * (((data["max"]) * (((data["std_last_10000"]) + ((((((data["MA_400MA_BB_high_mean"]) + (data["max_roll_mean_1000"]))/2.0)) + (data["std_first_10000"]))))))))) +
            1.0*np.tanh((((((((data["abs_trend"]) + (data["q01_roll_std_1000"]))/2.0)) * (((((((data["abs_q05"]) / 2.0)) + (data["abs_q05"]))) / 2.0)))) / 2.0)) +
            1.0*np.tanh((((((((data["q99_roll_std_1000"]) - ((-1.0*((((data["min_roll_std_10"]) * 2.0))))))) + (data["std_roll_std_10"]))/2.0)) * (((data["mean_change_rate"]) * (data["std_roll_std_10"]))))) +
            1.0*np.tanh((((((((((((((data["q99_roll_mean_100"]) * (data["q001"]))) + (data["std_roll_std_10"]))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.989058*np.tanh(((((((((data["min_first_10000"]) / 2.0)) / 2.0)) / 2.0)) * ((((data["count_big"]) + (np.tanh(((((-3.0) + (data["MA_700MA_BB_low_mean"]))/2.0)))))/2.0)))) +
            0.909340*np.tanh(((((data["max_last_10000"]) * (np.tanh((((np.tanh((data["abs_q01"]))) - (np.tanh((np.tanh((data["Moving_average_1500_mean"]))))))))))) / 2.0)) +
            0.804220*np.tanh(((((-1.0*((data["max_to_min_diff"])))) + (((((data["abs_min"]) / 2.0)) * (data["abs_min"]))))/2.0)) +
            1.0*np.tanh(((data["max_to_min_diff"]) * (((((((data["abs_q99"]) + (np.tanh((data["abs_q99"]))))/2.0)) + (data["abs_q99"]))/2.0)))) +
            1.0*np.tanh((-1.0*((((((((0.0) + (((data["std_first_50000"]) / 2.0)))/2.0)) + (data["abs_trend"]))/2.0))))) +
            1.0*np.tanh((((((((data["min_first_10000"]) / 2.0)) + ((((((data["max_roll_mean_10"]) - (data["std_roll_mean_100"]))) + (data["mean_diff"]))/2.0)))/2.0)) / 2.0)) +
            1.0*np.tanh(((data["q99_roll_mean_100"]) * ((-1.0*((((data["ave_roll_mean_1000"]) * ((((data["q05_roll_mean_10"]) + (data["q99_roll_mean_100"]))/2.0))))))))) +
            0.682689*np.tanh(((data["mean_change_rate_last_10000"]) * (((((((((data["min_last_50000"]) + (data["MA_400MA_BB_high_mean"]))/2.0)) * 2.0)) + (data["MA_400MA_BB_high_mean"]))/2.0)))) +
            0.962485*np.tanh(((((((((data["abs_q05"]) + (((data["max_last_10000"]) * (((data["abs_q05"]) + (data["std_roll_mean_1000"]))))))) / 2.0)) / 2.0)) / 2.0)) +
            1.0*np.tanh((((((data["min_last_10000"]) * (np.tanh((((data["iqr"]) / 2.0)))))) + (((data["abs_q01"]) * (np.tanh((data["abs_min"]))))))/2.0)) +
            0.833920*np.tanh(((data["abs_q01"]) / 2.0)) +
            1.0*np.tanh(((0.0) * (np.tanh((data["abs_q01"]))))) +
            0.923017*np.tanh(((data["q95_roll_mean_100"]) * (((((data["std_first_50000"]) - (np.tanh((((((data["abs_min"]) / 2.0)) * 2.0)))))) * (data["av_change_rate_roll_mean_100"]))))) +
            1.0*np.tanh((((((data["count_big"]) + (((data["mean_diff"]) + (np.tanh((np.tanh((data["avg_last_50000"]))))))))/2.0)) * (((data["min_first_10000"]) / 2.0)))) +
            1.0*np.tanh(((data["avg_first_50000"]) * ((-1.0*((((data["max_roll_mean_1000"]) * (((data["avg_first_50000"]) / 2.0))))))))) +
            0.782337*np.tanh(np.tanh((((((data["iqr"]) / 2.0)) * (((((data["exp_Moving_average_300_mean"]) / 2.0)) * (((data["iqr"]) + (data["avg_first_10000"]))))))))) +
            0.908558*np.tanh((((((np.tanh((np.tanh(((-1.0*((data["iqr"])))))))) + (((((data["min_roll_std_10"]) * (data["iqr"]))) / 2.0)))/2.0)) / 2.0)) +
            0.769832*np.tanh((((((data["min_roll_std_10"]) * ((-1.0*((((data["max_roll_std_10"]) / 2.0))))))) + (((data["mean_change_rate_last_10000"]) * ((-1.0*((data["max_roll_mean_10"])))))))/2.0)) +
            0.926925*np.tanh(((((np.tanh((((((data["kurt"]) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) +
            1.0*np.tanh((((((((data["max_first_50000"]) + (data["q01_roll_mean_100"]))) + (0.0))/2.0)) * (((data["av_change_abs_roll_std_10"]) * (np.tanh((data["q99"]))))))) +
            0.982024*np.tanh((((((((((data["av_change_abs_roll_std_1000"]) / 2.0)) * (data["av_change_abs_roll_std_10"]))) + (np.tanh((np.tanh((np.tanh((data["std_roll_mean_100"]))))))))/2.0)) / 2.0)) +
            0.794060*np.tanh(np.tanh((0.0))) +
            0.995701*np.tanh(((data["abs_q01"]) * (((0.0) * (data["std_roll_mean_1000"]))))) +
            1.0*np.tanh(((((data["av_change_abs_roll_std_100"]) / 2.0)) * (((((data["av_change_abs_roll_std_100"]) / 2.0)) * ((((data["abs_q05"]) + (np.tanh((data["abs_min"]))))/2.0)))))) +
            0.784291*np.tanh(((data["Moving_average_6000_mean"]) * (((data["Moving_average_6000_mean"]) * (((data["q99_roll_mean_100"]) * (((data["av_change_rate_roll_mean_10"]) * ((((-1.0*((data["Hann_window_mean"])))) / 2.0)))))))))) +
            0.973427*np.tanh((((data["abs_q01"]) + (((np.tanh((data["iqr"]))) * ((-1.0*((((data["classic_sta_lta4_mean"]) / 2.0))))))))/2.0)) +
            0.870262*np.tanh(((data["iqr"]) * (((data["iqr"]) * (np.tanh(((((data["av_change_rate_roll_std_10"]) + (((data["iqr"]) * (data["abs_trend"]))))/2.0)))))))) +
            1.0*np.tanh(np.tanh(((((((np.tanh((data["q01_roll_mean_100"]))) + (0.0))/2.0)) * (np.tanh((data["q99_roll_mean_1000"]))))))) +
            0.668230*np.tanh(((data["av_change_abs_roll_mean_10"]) * (data["abs_max_roll_mean_1000"]))) +
            0.949199*np.tanh(((np.tanh((np.tanh((np.tanh((((data["std_last_50000"]) * (((((-1.0*((data["Moving_average_1500_mean"])))) + (data["med"]))/2.0)))))))))) / 2.0)) +
            1.0*np.tanh(((((data["max_to_min"]) * (np.tanh((((data["min_roll_mean_1000"]) * (((np.tanh((((data["abs_max_roll_std_100"]) * 2.0)))) * 2.0)))))))) / 2.0)) +
            0.945682*np.tanh((((-1.0*((data["max_roll_mean_1000"])))) * ((((data["classic_sta_lta1_mean"]) + (((data["q99_roll_std_10"]) * (data["classic_sta_lta1_mean"]))))/2.0)))) +
            1.0*np.tanh(((data["abs_q05"]) * (((((data["max_last_50000"]) * (data["std_last_50000"]))) * ((((data["std_last_50000"]) + (data["max"]))/2.0)))))) +
            0.999609*np.tanh(((np.tanh((data["max_last_10000"]))) * (((data["std_last_10000"]) - (np.tanh((data["max_last_10000"]))))))) +
            1.0*np.tanh(((((np.tanh((((data["min_last_10000"]) + (((data["q01"]) * (((data["q01_roll_mean_1000"]) + (data["abs_q99"]))))))))) / 2.0)) / 2.0)) +
            0.814771*np.tanh(np.tanh((np.tanh((((data["max_to_min_diff"]) * (data["std_last_50000"]))))))) +
            0.999218*np.tanh(((((((((data["min_first_50000"]) * (data["max_last_10000"]))) * (data["abs_max"]))) * (data["max_last_10000"]))) * (data["max_last_10000"]))) +
            0.921063*np.tanh(np.tanh((np.tanh((((data["q01_roll_mean_100"]) * (((data["MA_400MA_std_mean"]) * (data["Moving_average_6000_mean"]))))))))) +
            0.783118*np.tanh((((((-1.0*((data["ave_roll_std_10"])))) + (data["q99_roll_std_100"]))) / 2.0)) +
            0.977335*np.tanh(((data["skew"]) * (((data["skew"]) * (((data["min"]) * (((data["skew"]) * (data["std_roll_std_10"]))))))))) +
            0.557249*np.tanh((((np.tanh((((data["max_roll_std_10"]) - (np.tanh((data["abs_max_roll_std_100"]))))))) + (data["std_first_50000"]))/2.0)) +
            1.0*np.tanh(((((((((np.tanh((np.tanh((data["max_to_min_diff"]))))) + ((((data["min_last_50000"]) + (data["classic_sta_lta3_mean"]))/2.0)))/2.0)) + (np.tanh((data["abs_max_roll_mean_1000"]))))/2.0)) / 2.0)) +
            0.886284*np.tanh(data["abs_q01"]) +
            0.998046*np.tanh(data["abs_min"]) +
            0.847597*np.tanh(((((((data["classic_sta_lta3_mean"]) * (((data["med"]) * ((-1.0*((data["abs_max_roll_std_1000"])))))))) - (((((data["med"]) / 2.0)) / 2.0)))) / 2.0)) +
            0.818679*np.tanh(((np.tanh((((((np.tanh((data["Moving_average_6000_mean"]))) + (data["q01_roll_mean_100"]))) * (data["q01_roll_mean_100"]))))) / 2.0)) +
            0.620946*np.tanh(((((((((-1.0*((((1.0) / 2.0))))) / 2.0)) / 2.0)) + ((((data["MA_700MA_BB_high_mean"]) + (data["MA_700MA_BB_low_mean"]))/2.0)))/2.0)) +
            0.836655*np.tanh(((data["std_roll_std_1000"]) * (((data["max_last_10000"]) * (((data["min_roll_std_10"]) + (((data["av_change_abs_roll_std_100"]) + (((data["max_last_10000"]) / 2.0)))))))))) +
            1.0*np.tanh(((data["count_big"]) * (((((data["iqr"]) * (((((data["iqr"]) + (data["MA_400MA_BB_low_mean"]))) * 2.0)))) / 2.0)))) +
            0.937085*np.tanh(((((data["std_first_50000"]) * (((data["max_last_50000"]) * ((-1.0*((data["kurt"])))))))) * 2.0)) +
            1.0*np.tanh(((((data["max_last_50000"]) * ((((((data["classic_sta_lta1_mean"]) + (((data["std_roll_std_10"]) * (data["max_roll_mean_100"]))))/2.0)) / 2.0)))) * (data["std_first_10000"]))) +
            1.0*np.tanh(((np.tanh((((data["max_last_50000"]) * (np.tanh(((((((((data["std_roll_mean_1000"]) * 2.0)) * (data["max_last_50000"]))) + (data["std_roll_mean_1000"]))/2.0)))))))) / 2.0)) +
            0.908558*np.tanh((((((((data["min_roll_mean_100"]) + (data["std_roll_mean_10"]))) + ((((((data["min_roll_mean_100"]) + (data["std_roll_mean_10"]))/2.0)) / 2.0)))/2.0)) / 2.0)) +
            0.881985*np.tanh(((data["abs_q05"]) * (np.tanh((np.tanh((np.tanh((np.tanh((((data["abs_max_roll_std_100"]) * (data["MA_400MA_BB_low_mean"]))))))))))))) +
            0.575615*np.tanh((-1.0*(((-1.0*((((np.tanh((((0.0) / 2.0)))) / 2.0)))))))) +
            0.679172*np.tanh(((np.tanh((np.tanh((((((((data["Moving_average_3000_mean"]) - (data["iqr"]))) * 2.0)) - (data["iqr"]))))))) / 2.0)) +
            0.889019*np.tanh(((np.tanh(((((((data["q95_roll_std_10"]) + ((-1.0*((((data["avg_last_50000"]) - (data["abs_q05"])))))))/2.0)) / 2.0)))) / 2.0)) +
            0.999609*np.tanh(((((((np.tanh((((((data["av_change_abs_roll_std_100"]) + (data["med"]))) * 2.0)))) / 2.0)) * (data["av_change_abs_roll_std_1000"]))) / 2.0)) +
            0.998437*np.tanh(((((data["av_change_abs_roll_mean_10"]) * (((((((((data["av_change_abs_roll_mean_100"]) - (((-1.0) * 2.0)))) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) +
            0.555295*np.tanh(((((data["mean_diff"]) * (data["std_last_10000"]))) / 2.0)) +
            0.736616*np.tanh((-1.0*((((data["abs_q05"]) * (np.tanh((np.tanh((np.tanh((((data["iqr"]) * ((((data["av_change_rate_roll_std_10"]) + (data["q95_roll_std_10"]))/2.0))))))))))))))) +
            0.954279*np.tanh(((((np.tanh(((((data["sum"]) + (((data["sum"]) * ((((data["classic_sta_lta2_mean"]) + (data["sum"]))/2.0)))))/2.0)))) / 2.0)) / 2.0)) +
            0.844471*np.tanh(((data["abs_max_roll_std_100"]) * (((data["ave_roll_mean_100"]) * ((((((data["q01_roll_mean_1000"]) / 2.0)) + (data["max_last_10000"]))/2.0)))))) +
            0.587730*np.tanh(((np.tanh((((data["avg_first_50000"]) - (np.tanh((data["std_roll_mean_1000"]))))))) * (data["std_roll_mean_1000"]))) +
            0.875342*np.tanh(((data["av_change_rate_roll_std_100"]) * (((((data["av_change_abs_roll_mean_1000"]) * (((((data["trend"]) / 2.0)) / 2.0)))) / 2.0)))) +
            0.859711*np.tanh((((np.tanh((((data["abs_max_roll_std_1000"]) - (data["max_roll_std_100"]))))) + ((((((data["trend"]) + (data["min_last_50000"]))/2.0)) / 2.0)))/2.0)) +
            0.999218*np.tanh((((((((((data["abs_q05"]) + (((data["q01_roll_std_100"]) * (data["abs_q05"]))))/2.0)) / 2.0)) / 2.0)) / 2.0)) +
            1.0*np.tanh((((-1.0*(((-1.0*(((-1.0*(((0.02310276590287685))))))))))) * (((data["Moving_average_6000_mean"]) - (data["min_first_10000"]))))) +
            0.867917*np.tanh((((((data["abs_q01"]) + (((np.tanh((((np.tanh((data["abs_max_roll_mean_100"]))) - (((data["av_change_rate_roll_std_100"]) + (0.0))))))) / 2.0)))/2.0)) / 2.0)) +
            0.847206*np.tanh(((0.0) * (0.0))) +
            0.998828*np.tanh(((((data["min_last_10000"]) * ((((np.tanh(((((data["av_change_rate_roll_mean_1000"]) + (data["min_last_10000"]))/2.0)))) + (np.tanh((data["q05_roll_mean_100"]))))/2.0)))) / 2.0)) +
            0.843689*np.tanh(((np.tanh((((data["av_change_rate_roll_mean_100"]) * (((0.0) - (np.tanh((np.tanh((np.tanh((np.tanh((data["iqr"]))))))))))))))) / 2.0)) +
            0.881594*np.tanh(np.tanh((np.tanh((((((data["min_roll_mean_1000"]) * (((0.0) - (np.tanh((data["av_change_rate_roll_std_10"]))))))) * (data["mean"]))))))) +
            1.0*np.tanh(((((((((np.tanh((((((data["mean_change_rate"]) - (((data["med"]) / 2.0)))) / 2.0)))) / 2.0)) * 2.0)) / 2.0)) / 2.0)) +
            0.581477*np.tanh(((np.tanh((np.tanh(((-1.0*((((np.tanh((np.tanh((data["mean_change_rate_last_10000"]))))) / 2.0))))))))) / 2.0)) +
            0.707698*np.tanh(np.tanh((np.tanh(((((((np.tanh((data["std_roll_mean_1000"]))) / 2.0)) + ((-1.0*((((data["trend"]) * (data["max_to_min_diff"])))))))/2.0)))))) +
            0.999218*np.tanh(((data["abs_q01"]) * ((((0.0) + (np.tanh((data["ave_roll_mean_10"]))))/2.0)))) +
            0.884330*np.tanh(((data["std_first_10000"]) * (((np.tanh((((data["av_change_rate_roll_mean_1000"]) - (data["Moving_average_3000_mean"]))))) / 2.0)))) +
            0.999609*np.tanh(0.0) +
            1.0*np.tanh(data["abs_q01"]) +
            0.787808*np.tanh(((data["min_last_10000"]) * (((((((((((data["std_first_10000"]) * (((((data["std_first_10000"]) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) +
            0.994529*np.tanh(((((((((data["abs_min"]) * (0.0))) / 2.0)) / 2.0)) / 2.0)) +
            0.850332*np.tanh(((((data["Moving_average_1500_mean"]) * (((data["Moving_average_1500_mean"]) * (((((((((data["abs_min"]) + (data["std_first_10000"]))/2.0)) + (data["std_first_10000"]))/2.0)) / 2.0)))))) / 2.0)) +
            0.699101*np.tanh(np.tanh((((((data["max_first_10000"]) * (((data["abs_q05"]) - (np.tanh(((((data["max_first_10000"]) + (data["Moving_average_6000_mean"]))/2.0)))))))) / 2.0)))) +
            0.996874*np.tanh((((((((((((-1.0*((data["avg_first_10000"])))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.314576*np.tanh(((np.tanh((((((np.tanh((((data["MA_400MA_BB_low_mean"]) / 2.0)))) / 2.0)) / 2.0)))) / 2.0)) +
            0.731536*np.tanh(((data["mean_change_rate_first_50000"]) * (((((((data["mean_change_rate_first_50000"]) * (((data["max_roll_mean_10"]) * (data["av_change_abs_roll_std_10"]))))) / 2.0)) * (data["av_change_abs_roll_std_10"]))))) +
            0.998437*np.tanh(((((((((data["min_roll_mean_10"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.661977*np.tanh((((((0.0) + (((((np.tanh((data["avg_first_50000"]))) / 2.0)) / 2.0)))/2.0)) / 2.0)) +
            0.932005*np.tanh(np.tanh((((data["Moving_average_1500_mean"]) - (data["ave10"]))))) +
            0.753810*np.tanh((((((((-1.0*(((((((-1.0*((((data["av_change_abs_roll_mean_1000"]) / 2.0))))) / 2.0)) / 2.0))))) / 2.0)) / 2.0)) / 2.0)) +
            0.998046*np.tanh(((np.tanh((((data["classic_sta_lta2_mean"]) * (np.tanh((((data["av_change_rate_roll_mean_100"]) * (np.tanh((data["min_first_50000"]))))))))))) / 2.0)) +
            0.996092*np.tanh(((data["Moving_average_1500_mean"]) - (data["ave_roll_mean_100"]))) +
            0.423603*np.tanh(data["abs_min"]) +
            0.998828*np.tanh(np.tanh((np.tanh((((data["classic_sta_lta3_mean"]) * ((-1.0*((((data["abs_max_roll_mean_1000"]) * (((((data["abs_max_roll_mean_1000"]) * (data["std_roll_mean_1000"]))) * 2.0))))))))))))) +
            0.647909*np.tanh((-1.0*((((np.tanh((np.tanh(((((np.tanh((data["abs_trend"]))) + (data["count_big"]))/2.0)))))) / 2.0))))) +
            0.801485*np.tanh(data["abs_q01"]) +
            0.999609*np.tanh(((((((((((data["mean_change_rate_last_50000"]) / 2.0)) / 2.0)) * (((data["av_change_abs_roll_std_1000"]) - (((data["Moving_average_6000_mean"]) / 2.0)))))) / 2.0)) / 2.0)) +
            0.999218*np.tanh(data["abs_q01"]) +
            0.998828*np.tanh(((np.tanh((((data["abs_max_roll_std_10"]) * (np.tanh((((data["abs_max_roll_std_10"]) * (np.tanh((data["med"]))))))))))) / 2.0)) +
            0.939039*np.tanh(((np.tanh((((((data["av_change_abs_roll_std_10"]) * (((data["kurt"]) * (data["kurt"]))))) * (((data["kurt"]) * (data["kurt"]))))))) / 2.0)) +
            0.546307*np.tanh((((-1.0*(((((((((data["av_change_abs_roll_mean_1000"]) / 2.0)) + (data["av_change_abs_roll_std_10"]))/2.0)) * (data["av_change_abs_roll_std_1000"])))))) / 2.0)) +
            0.998828*np.tanh((((data["abs_min"]) + ((-1.0*((((((((data["abs_q01"]) / 2.0)) / 2.0)) / 2.0))))))/2.0)) +
            1.0*np.tanh((((((-1.0*(((((((data["av_change_abs_roll_std_100"]) / 2.0)) + (((np.tanh((data["med"]))) / 2.0)))/2.0))))) / 2.0)) / 2.0)) +
            0.998437*np.tanh(0.0) +
            0.785854*np.tanh(((np.tanh((((np.tanh(((-1.0*((((data["mean_diff"]) * (data["avg_first_10000"])))))))) / 2.0)))) / 2.0)) +
            0.705354*np.tanh(np.tanh((((data["exp_Moving_average_300_mean"]) * (np.tanh((((data["exp_Moving_average_300_mean"]) * (np.tanh((((data["abs_max_roll_std_10"]) + ((-1.0*((data["max_roll_std_100"])))))))))))))))) +
            0.781946*np.tanh(data["abs_min"]) +
            0.921454*np.tanh(((((data["av_change_abs_roll_mean_100"]) / 2.0)) * ((((((((((data["av_change_abs_roll_mean_100"]) / 2.0)) + (data["abs_q01"]))/2.0)) / 2.0)) / 2.0)))) +
            1.0*np.tanh((((-1.0*((((np.tanh((np.tanh((((data["abs_q05"]) / 2.0)))))) / 2.0))))) * ((-1.0*((data["std_last_10000"])))))) +
            0.731145*np.tanh(((((data["std_last_50000"]) * ((-1.0*((((((data["max_to_min_diff"]) * 2.0)) * (((np.tanh((np.tanh((data["Moving_average_6000_mean"]))))) / 2.0))))))))) / 2.0)) +
            0.999609*np.tanh(((((((data["av_change_abs_roll_std_10"]) * (data["min_last_50000"]))) * (data["min_roll_std_100"]))) * ((-1.0*((np.tanh((data["min_first_50000"])))))))) +
            0.668621*np.tanh((((-1.0*((data["max_first_10000"])))) * ((((data["min_last_10000"]) + (np.tanh((((data["av_change_rate_roll_std_1000"]) * (data["min_last_10000"]))))))/2.0)))) +
            0.932786*np.tanh((((-1.0*(((((((((((0.0) / 2.0)) / 2.0)) / 2.0)) + ((((data["abs_min"]) + (np.tanh((data["max_first_50000"]))))/2.0)))/2.0))))) / 2.0)) +
            0.728019*np.tanh(((np.tanh((((((data["MA_700MA_BB_low_mean"]) * (data["std_first_50000"]))) * (data["abs_trend"]))))) / 2.0)) +
            0.828058*np.tanh(((data["min_first_10000"]) * ((((((np.tanh((((data["max_last_50000"]) / 2.0)))) + (data["max_first_50000"]))/2.0)) / 2.0)))) +
            0.923017*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(((data["q05_roll_std_1000"]) * (((((data["abs_q01"]) + (((data["Hann_window_mean"]) - (data["Moving_average_6000_mean"]))))) + (((data["Hann_window_mean"]) - (data["Moving_average_6000_mean"]))))))) +
            0.917155*np.tanh(np.tanh((((((((((np.tanh((data["std_roll_mean_1000"]))) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) +
            1.0*np.tanh(0.0) +
            0.630715*np.tanh(((((np.tanh((np.tanh((data["std_first_10000"]))))) / 2.0)) / 2.0)) +
            0.501368*np.tanh(data["abs_q01"]) +
            0.423603*np.tanh((-1.0*((((data["abs_q01"]) + (((data["ave_roll_mean_1000"]) - (data["Moving_average_1500_mean"])))))))) +
            0.722157*np.tanh(np.tanh((((np.tanh((np.tanh((np.tanh((((data["min_last_10000"]) * (data["abs_max_roll_mean_10"]))))))))) / 2.0)))) +
            0.424775*np.tanh(((((((np.tanh((((data["std_roll_mean_10"]) * (((data["max_roll_std_100"]) - (data["Moving_average_700_mean"]))))))) / 2.0)) / 2.0)) * 2.0)) +
            0.999609*np.tanh(((((np.tanh((((data["mean_diff"]) * ((-1.0*((np.tanh((data["count_big"])))))))))) / 2.0)) / 2.0)) +
            0.812427*np.tanh((((((((-1.0*((((((np.tanh((np.tanh((data["abs_q95"]))))) * 2.0)) / 2.0))))) / 2.0)) / 2.0)) / 2.0)) +
            0.540055*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(data["abs_q01"]) +
            0.998046*np.tanh(data["abs_min"]) +
            0.998437*np.tanh(data["abs_min"]) +
            1.0*np.tanh(((data["av_change_abs_roll_mean_10"]) * ((((((((((data["Moving_average_6000_mean"]) + (((((data["av_change_abs_roll_mean_10"]) / 2.0)) / 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)))) +
            0.738179*np.tanh(0.0) +
            0.790934*np.tanh(0.0) +
            0.930442*np.tanh(((((data["min_roll_std_10"]) * (((((np.tanh((np.tanh((data["av_change_rate_roll_std_10"]))))) / 2.0)) / 2.0)))) / 2.0)) +
            0.980852*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(((((((((((((data["abs_min"]) - (((data["Hann_window_mean"]) - (data["exp_Moving_average_300_mean"]))))) * 2.0)) * 2.0)) - (0.0))) * 2.0)) * 2.0)) +
            0.991012*np.tanh(((data["Moving_average_1500_mean"]) - (data["Hann_window_mean"]))) +
            0.855021*np.tanh(0.0) +
            0.997655*np.tanh(((0.0) / 2.0)) +
            0.980852*np.tanh(0.0) +
            0.991794*np.tanh(((data["exp_Moving_average_3000_mean"]) - (data["exp_Moving_average_6000_mean"]))) +
            1.0*np.tanh(((((((data["av_change_abs_roll_mean_100"]) / 2.0)) * (((((((np.tanh((((data["max_last_10000"]) * 2.0)))) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) +
            0.999609*np.tanh(((data["sum"]) * (((data["sum"]) * (((data["min_last_10000"]) * (np.tanh(((((0.0) + (data["std_last_50000"]))/2.0)))))))))) +
            1.0*np.tanh((((-1.0*((data["min_last_10000"])))) * (((((((((np.tanh((data["abs_q05"]))) * ((-1.0*((data["min_last_10000"])))))) / 2.0)) / 2.0)) / 2.0)))) +
            0.723329*np.tanh(np.tanh((np.tanh((((((((np.tanh((((data["Moving_average_1500_mean"]) * (data["Moving_average_1500_mean"]))))) / 2.0)) / 2.0)) / 2.0)))))) +
            0.607659*np.tanh((((((((((data["min_roll_mean_10"]) / 2.0)) + (((np.tanh((((data["q05_roll_std_10"]) - (3.0))))) / 2.0)))/2.0)) / 2.0)) / 2.0)) +
            0.920281*np.tanh(np.tanh((((data["std_last_10000"]) * (((((((((data["Hilbert_mean"]) / 2.0)) * (data["max_last_10000"]))) / 2.0)) / 2.0)))))) +
            0.843689*np.tanh(((((((((data["std_last_50000"]) * (np.tanh((((((data["mean_change_rate"]) - (((data["MA_1000MA_std_mean"]) * 2.0)))) * 2.0)))))) / 2.0)) / 2.0)) / 2.0)) +
            0.623290*np.tanh((-1.0*((((((np.tanh((((data["min_roll_mean_10"]) * (data["min_roll_std_1000"]))))) / 2.0)) / 2.0))))) +
            1.0*np.tanh(data["abs_q01"]) +
            0.836655*np.tanh((((((-1.0*((((data["abs_q01"]) - (((np.tanh((data["min_last_50000"]))) / 2.0))))))) / 2.0)) / 2.0)) +
            0.695584*np.tanh(((((((((-1.0*((np.tanh((((data["min_roll_mean_1000"]) * 2.0))))))) / 2.0)) + (((np.tanh((data["classic_sta_lta3_mean"]))) / 2.0)))/2.0)) / 2.0)) +
            0.893708*np.tanh((((((((((((((-1.0) + (((data["abs_min"]) * 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.998828*np.tanh(((data["min_last_10000"]) * ((((0.0) + (((((-1.0*((0.0)))) + (((data["ave_roll_mean_100"]) + ((-1.0*((data["sum"])))))))/2.0)))/2.0)))) +
            0.845643*np.tanh(((((((data["q01_roll_std_100"]) - (data["q01_roll_std_1000"]))) * ((-1.0*((np.tanh(((((data["med"]) + (0.0))/2.0))))))))) / 2.0)) +
            0.893708*np.tanh(np.tanh((((((data["max_last_10000"]) / 2.0)) * (((((((data["std_last_50000"]) / 2.0)) - (data["q99_roll_mean_100"]))) / 2.0)))))) +
            0.998046*np.tanh(((((((data["max_roll_mean_10"]) - (data["max_roll_std_100"]))) * (((data["std_last_50000"]) / 2.0)))) / 2.0)) +
            1.0*np.tanh(np.tanh((((((np.tanh((((data["med"]) * (data["max_roll_mean_100"]))))) / 2.0)) / 2.0)))) +
            0.770223*np.tanh(((((((np.tanh((data["max_roll_mean_1000"]))) / 2.0)) / 2.0)) / 2.0)) +
            0.999218*np.tanh(((np.tanh((((data["med"]) / 2.0)))) * (((((((data["kurt"]) / 2.0)) / 2.0)) / 2.0)))) +
            0.771004*np.tanh(((data["Moving_average_1500_mean"]) * (((data["std_first_10000"]) * (np.tanh((np.tanh((((data["min_roll_mean_1000"]) + (data["std_roll_mean_100"]))))))))))) +
            0.856585*np.tanh(0.0) +
            0.484173*np.tanh(((((np.tanh((((((((((data["Moving_average_700_mean"]) - (data["ave_roll_mean_100"]))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
            0.858148*np.tanh(((((((((((data["min_last_10000"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.999218*np.tanh(((((((data["std_first_10000"]) * (data["ave_roll_mean_1000"]))) * (((data["std_first_10000"]) + (data["min_first_10000"]))))) / 2.0)) +
            0.996092*np.tanh(((((data["exp_Moving_average_300_mean"]) - (data["exp_Moving_average_6000_mean"]))) * (((data["q99_roll_mean_100"]) / 2.0)))) +
            0.899961*np.tanh(data["abs_q01"]) +
            0.554513*np.tanh(((((((data["q999"]) + ((-1.0*((data["std_roll_mean_10"])))))/2.0)) + (0.0))/2.0)) +
            0.878859*np.tanh((((((((((np.tanh((np.tanh((((data["mean"]) * (data["ave10"]))))))) + ((-1.0*((data["skew"])))))/2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.0*np.tanh(data["abs_min"]) +
            0.999218*np.tanh((((((-1.0*((np.tanh((((data["abs_max_roll_std_100"]) * (np.tanh((((((data["avg_first_50000"]) * 2.0)) * 2.0))))))))))) / 2.0)) / 2.0)) +
            1.0*np.tanh(((((((((data["abs_q05"]) * (data["max_last_10000"]))) * (data["max_last_10000"]))) * (np.tanh((((data["abs_max"]) * 2.0)))))) / 2.0)) +
            0.998437*np.tanh(((((data["av_change_abs_roll_std_10"]) * (np.tanh(((((-1.0*((((data["q99_roll_std_100"]) * (((data["med"]) * 2.0))))))) / 2.0)))))) / 2.0)) +
            0.471278*np.tanh((-1.0*((((data["min_roll_std_10"]) * (((data["min_roll_std_10"]) * (((((((((data["count_big"]) / 2.0)) / 2.0)) / 2.0)) / 2.0))))))))) +
            0.878859*np.tanh(((np.tanh((((data["abs_q05"]) * ((((((((data["min_roll_std_100"]) + ((((data["min_roll_std_100"]) + (data["mean_diff"]))/2.0)))/2.0)) / 2.0)) / 2.0)))))) / 2.0)) +
            0.556467*np.tanh(((np.tanh((((((np.tanh((data["min_last_10000"]))) / 2.0)) / 2.0)))) / 2.0)) +
            0.457210*np.tanh(((np.tanh((((((((data["av_change_abs_roll_mean_100"]) * (data["av_change_abs_roll_mean_100"]))) * (data["q99_roll_mean_10"]))) / 2.0)))) / 2.0)) +
            0.677608*np.tanh(data["abs_q01"]) +
            0.824541*np.tanh(np.tanh((((np.tanh((((data["q99_roll_std_10"]) * (((data["av_change_abs_roll_mean_1000"]) * (data["min_roll_mean_10"]))))))) / 2.0)))) +
            0.999609*np.tanh(0.0) +
            0.865572*np.tanh(np.tanh((((((((data["max_roll_std_10"]) * (((((np.tanh(((((data["q001"]) + (data["abs_max_roll_mean_10"]))/2.0)))) / 2.0)) / 2.0)))) / 2.0)) / 2.0)))) +
            0.563111*np.tanh(((((((np.tanh((np.tanh((((data["av_change_abs_roll_mean_100"]) / 2.0)))))) / 2.0)) / 2.0)) / 2.0)) +
            0.816725*np.tanh(np.tanh((((((data["classic_sta_lta4_mean"]) * (((np.tanh((data["min_last_10000"]))) / 2.0)))) / 2.0)))) +
            0.0*np.tanh(((((np.tanh((((np.tanh((data["std_roll_mean_10"]))) / 2.0)))) / 2.0)) / 2.0)) +
            0.767097*np.tanh(((((((np.tanh((((np.tanh((((data["q95_roll_std_100"]) - (data["abs_min"]))))) - (data["q95"]))))) / 2.0)) / 2.0)) / 2.0)) +
            0.719031*np.tanh(((np.tanh((((data["max_to_min"]) * (((data["abs_q99"]) * ((((data["max_first_50000"]) + ((((data["abs_q99"]) + (data["abs_q99"]))/2.0)))/2.0)))))))) / 2.0)) +
            0.617429*np.tanh(((np.tanh((((np.tanh(((((((np.tanh((data["max_roll_mean_1000"]))) / 2.0)) + (data["abs_max_roll_std_1000"]))/2.0)))) / 2.0)))) / 2.0)) +
            0.997265*np.tanh(((np.tanh((((((((((data["abs_q05"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) +
            0.999609*np.tanh(((np.tanh((((data["av_change_rate_roll_mean_10"]) * (((((data["med"]) - (data["Moving_average_3000_mean"]))) / 2.0)))))) / 2.0)) +
            0.859320*np.tanh(((np.tanh((((data["max_roll_mean_10"]) * (np.tanh((((data["max_first_50000"]) * (data["std_roll_mean_1000"]))))))))) * (data["ave_roll_mean_10"]))) +
            0.297382*np.tanh(((np.tanh((np.tanh((np.tanh(((((((-1.0*((((((data["abs_q01"]) / 2.0)) + (data["q99"])))))) / 2.0)) / 2.0)))))))) / 2.0)) +
            0.467761*np.tanh(((((data["max_to_min_diff"]) * (((np.tanh((np.tanh((((data["max_last_10000"]) * (np.tanh((data["av_change_abs_roll_mean_1000"]))))))))) * (data["max_last_10000"]))))) / 2.0)) +
            0.999218*np.tanh((0.0)) +
            0.999218*np.tanh(((((np.tanh(((-1.0*((((((data["q99_roll_std_10"]) * (data["avg_first_50000"]))) * (data["classic_sta_lta1_mean"])))))))) / 2.0)) / 2.0)) +
            0.892927*np.tanh(((np.tanh((np.tanh(((((data["abs_max_roll_mean_100"]) + ((((-1.0*((((data["q99_roll_std_10"]) - (data["q999"])))))) * 2.0)))/2.0)))))) / 2.0)) +
            0.763579*np.tanh((((((-1.0*((((np.tanh((data["q99_roll_std_1000"]))) / 2.0))))) / 2.0)) / 2.0)) +
            0.999218*np.tanh(((((((np.tanh((data["abs_q01"]))) * 2.0)) * (np.tanh((data["abs_q01"]))))) / 2.0)) +
            0.739351*np.tanh(data["abs_q01"]) +
            0.867136*np.tanh(((((data["std"]) + ((-1.0*((data["std_roll_std_100"])))))) / 2.0)) +
            0.725674*np.tanh(((np.tanh((np.tanh((((np.tanh((data["exp_Moving_average_6000_mean"]))) * (((data["q01_roll_mean_100"]) + (((data["abs_max_roll_std_100"]) * (data["count_big"]))))))))))) / 2.0)) +
            0.996874*np.tanh(((((((np.tanh((((data["q01_roll_std_10"]) * ((-1.0*((data["q01_roll_std_10"])))))))) / 2.0)) / 2.0)) / 2.0)) +
            1.0*np.tanh(data["abs_q01"]))

def gpi2(data):
    return (-0.000001 +
            1.0*np.tanh(((((-1.0) * 2.0)) + ((((((-1.0*((((((data["q95_roll_mean_10"]) * 2.0)) * 2.0))))) * 2.0)) * 2.0)))) +
            1.0*np.tanh((((-1.0*((((data["q05_roll_std_100"]) * 2.0))))) - (((((((data["q05_roll_std_10"]) * 2.0)) * 2.0)) + (((((data["iqr"]) * 2.0)) * 2.0)))))) +
            1.0*np.tanh((((-1.0*((((np.tanh((data["sum"]))) + (((((data["q95_roll_std_10"]) + (data["abs_std"]))) + (((data["q01_roll_std_100"]) * 2.0))))))))) * 2.0)) +
            1.0*np.tanh((((((((((-1.0*((data["med"])))) - (data["q01_roll_std_10"]))) - (data["q95_roll_std_1000"]))) * 2.0)) - (((data["q05_roll_std_100"]) * 2.0)))) +
            1.0*np.tanh((((((-1.0*((data["q01_roll_std_1000"])))) + ((-1.0*((((((data["MA_700MA_BB_high_mean"]) * 2.0)) * 2.0))))))) * 2.0)) +
            1.0*np.tanh(((0.0) - (((((data["q01_roll_std_100"]) * 2.0)) + (((((data["med"]) * 2.0)) + (((data["q05_roll_std_100"]) * 2.0)))))))) +
            1.0*np.tanh((((((((-1.0*((data["q95_roll_mean_10"])))) - ((((((data["sum"]) + (data["av_change_rate_roll_std_100"]))/2.0)) / 2.0)))) * 2.0)) - (data["q01_roll_std_100"]))) +
            1.0*np.tanh((((((((-1.0*((((data["ave_roll_mean_100"]) + (data["q05_roll_std_10"])))))) * 2.0)) - ((((data["q05_roll_std_1000"]) + (data["av_change_rate_roll_mean_1000"]))/2.0)))) * 2.0)) +
            1.0*np.tanh(((((data["abs_q05"]) - (((((((((((data["MA_400MA_BB_high_mean"]) * 2.0)) * 2.0)) - (data["abs_q05"]))) + (data["q05_roll_std_10"]))) * 2.0)))) * 2.0)) +
            1.0*np.tanh((-1.0*((((((data["q05_roll_std_100"]) + (((data["q05_roll_std_100"]) + (data["med"]))))) + (((data["q95_roll_mean_1000"]) * (data["ave_roll_mean_100"])))))))) +
            1.0*np.tanh((((-1.0*((data["iqr"])))) - (((data["MA_700MA_BB_high_mean"]) + (((data["q95_roll_mean_100"]) * 2.0)))))) +
            1.0*np.tanh(((((((data["q05_roll_mean_10"]) - (data["min_roll_std_100"]))) - (((data["ave_roll_std_10"]) / 2.0)))) + (((((data["q05_roll_mean_10"]) * 2.0)) * 2.0)))) +
            1.0*np.tanh((((((-1.0*((((data["exp_Moving_average_300_mean"]) + (data["iqr"])))))) * 2.0)) - (((((data["Moving_average_3000_mean"]) + (((data["std_roll_std_10"]) * 2.0)))) * 2.0)))) +
            1.0*np.tanh((((((-1.0*((data["abs_max"])))) + (data["q01_roll_mean_10"]))) - ((((data["med"]) + (data["min_roll_std_1000"]))/2.0)))) +
            1.0*np.tanh(((data["q01_roll_mean_10"]) + ((((((((-1.0*((data["q05_roll_mean_1000"])))) * (data["q05_roll_mean_100"]))) + (data["q01_roll_mean_10"]))) - ((-1.0*((data["min_roll_mean_10"])))))))) +
            1.0*np.tanh(((((((((data["min_roll_mean_10"]) + (((((data["abs_q05"]) + (data["MA_700MA_BB_low_mean"]))) / 2.0)))) - (data["abs_max_roll_std_100"]))) * 2.0)) / 2.0)) +
            1.0*np.tanh((((((((data["av_change_rate_roll_mean_100"]) + ((-1.0*((((data["exp_Moving_average_300_mean"]) * 2.0))))))/2.0)) + (data["min_roll_mean_10"]))) * 2.0)) +
            1.0*np.tanh(((((data["min_roll_mean_10"]) + ((-1.0*((((data["iqr"]) * (((data["abs_q05"]) * 2.0))))))))) + (np.tanh((data["av_change_rate_roll_std_100"]))))) +
            1.0*np.tanh(((((((((data["Hann_window_mean"]) * (data["q05_roll_std_1000"]))) * 2.0)) + (data["abs_q05"]))) + (data["MA_400MA_BB_low_mean"]))) +
            1.0*np.tanh((((((((data["q001"]) / 2.0)) / 2.0)) + (((((data["q001"]) + (((data["Hilbert_mean"]) * (data["exp_Moving_average_300_mean"]))))) * (data["abs_q05"]))))/2.0)) +
            1.0*np.tanh(((((((data["exp_Moving_average_300_mean"]) * (((data["MA_400MA_BB_high_mean"]) * 2.0)))) * (data["q05_roll_mean_10"]))) - (data["q999"]))) +
            1.0*np.tanh((((data["q01_roll_mean_10"]) + (data["min_first_10000"]))/2.0)) +
            1.0*np.tanh(((((data["abs_trend"]) - (data["abs_q95"]))) * (data["abs_q05"]))) +
            1.0*np.tanh(((data["max_roll_mean_10"]) * (((data["q05_roll_mean_10"]) + (data["min_roll_mean_10"]))))) +
            1.0*np.tanh(((((data["min_roll_mean_1000"]) * (((data["max"]) - (data["min_roll_mean_1000"]))))) * 2.0)) +
            1.0*np.tanh((((data["av_change_abs_roll_std_100"]) + ((((((data["av_change_rate_roll_mean_1000"]) + (data["min_roll_mean_10"]))/2.0)) - (data["std_first_10000"]))))/2.0)) +
            1.0*np.tanh(((data["Moving_average_3000_mean"]) * (((((data["std_first_10000"]) * (data["MA_700MA_BB_low_mean"]))) * 2.0)))) +
            1.0*np.tanh(((data["q05_roll_std_1000"]) * ((((data["q01_roll_mean_100"]) + (((data["q01_roll_std_10"]) + (data["q01_roll_mean_100"]))))/2.0)))) +
            1.0*np.tanh(data["min_roll_mean_1000"]) +
            1.0*np.tanh(((((data["min"]) + (((((data["min"]) / 2.0)) * 2.0)))) * (data["abs_max"]))) +
            1.0*np.tanh(((((data["abs_max_roll_mean_100"]) * (((data["q05"]) + (((data["min_last_10000"]) * (data["av_change_rate_roll_std_100"]))))))) * (data["abs_max_roll_mean_100"]))) +
            1.0*np.tanh(((data["MA_400MA_std_mean"]) * (((((((data["av_change_rate_roll_std_1000"]) * (((data["Moving_average_3000_mean"]) - (data["min_first_10000"]))))) * 2.0)) * 2.0)))) +
            1.0*np.tanh(((data["abs_q05"]) * (((data["mean_change_rate"]) - (((data["MA_1000MA_std_mean"]) + ((((data["std_roll_std_1000"]) + ((((data["q99_roll_std_100"]) + (data["std_roll_mean_10"]))/2.0)))/2.0)))))))) +
            1.0*np.tanh(((np.tanh((((np.tanh((((((((data["q95_roll_mean_10"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) - (data["q95_roll_mean_10"]))) +
            1.0*np.tanh((((data["min_last_10000"]) + (((data["med"]) * (data["MA_400MA_BB_high_mean"]))))/2.0)) +
            1.0*np.tanh(((data["max_roll_mean_10"]) + (((data["min_last_50000"]) * (np.tanh((((data["abs_trend"]) * (((data["abs_max_roll_mean_10"]) + (((data["abs_max_roll_std_100"]) * 2.0)))))))))))) +
            0.952325*np.tanh(((((data["max_first_50000"]) - (data["abs_q05"]))) * (((data["iqr"]) - ((((((-1.0*((data["ave_roll_std_10"])))) * (data["max_roll_mean_100"]))) / 2.0)))))) +
            1.0*np.tanh(((data["min_roll_std_1000"]) * (((data["q01_roll_std_10"]) * (((data["q05_roll_mean_10"]) - (((((data["med"]) - (((data["q01"]) / 2.0)))) * 2.0)))))))) +
            0.957405*np.tanh(np.tanh((np.tanh((((((((data["Moving_average_700_mean"]) + (((data["std_roll_std_100"]) + (np.tanh((data["q05_roll_std_100"]))))))) * 2.0)) * 2.0)))))) +
            1.0*np.tanh(((((-1.0*((np.tanh((data["iqr"])))))) + (np.tanh((((data["iqr"]) * (data["med"]))))))/2.0)) +
            1.0*np.tanh((((np.tanh((data["std_first_50000"]))) + (((data["std"]) * (((((data["q001"]) * (data["q95_roll_mean_1000"]))) + (data["min_roll_std_100"]))))))/2.0)) +
            1.0*np.tanh((-1.0*(((((((data["av_change_rate_roll_std_100"]) + ((((data["av_change_abs_roll_std_100"]) + (np.tanh((data["max_to_min_diff"]))))/2.0)))/2.0)) * ((((data["av_change_rate_roll_mean_10"]) + (data["q99_roll_std_100"]))/2.0))))))) +
            0.885893*np.tanh((((np.tanh((data["min_last_50000"]))) + (np.tanh((np.tanh((((data["q01_roll_std_100"]) * 2.0)))))))/2.0)) +
            1.0*np.tanh(((data["q99_roll_mean_10"]) * (((data["MA_400MA_BB_low_mean"]) + (((np.tanh((data["iqr"]))) + (np.tanh((data["iqr"]))))))))) +
            0.922626*np.tanh((-1.0*((((data["min_roll_mean_1000"]) - (((data["abs_q05"]) * ((((-1.0*((data["q05_roll_std_100"])))) / 2.0))))))))) +
            1.0*np.tanh((((((data["max_last_10000"]) * (data["q01_roll_mean_1000"]))) + ((((np.tanh((data["abs_max_roll_mean_1000"]))) + ((-1.0*((data["min_roll_std_10"])))))/2.0)))/2.0)) +
            0.999609*np.tanh(np.tanh(((((np.tanh((((((data["abs_max_roll_mean_1000"]) + (np.tanh(((-1.0*((((data["av_change_rate_roll_mean_100"]) * 2.0))))))))) / 2.0)))) + (data["abs_max_roll_mean_1000"]))/2.0)))) +
            0.854631*np.tanh(((((data["iqr"]) * (((data["q95_roll_std_10"]) + ((((data["std_roll_mean_100"]) + (((data["Moving_average_700_mean"]) + (data["q95_roll_std_10"]))))/2.0)))))) * (data["av_change_rate_roll_mean_1000"]))) +
            0.966784*np.tanh((-1.0*(((((np.tanh((data["av_change_rate_roll_mean_1000"]))) + ((((((((np.tanh((data["av_change_rate_roll_std_1000"]))) / 2.0)) * (data["av_change_rate_roll_std_1000"]))) + (data["min_roll_std_100"]))/2.0)))/2.0))))) +
            1.0*np.tanh(np.tanh(((((((data["iqr"]) * (((data["q99_roll_mean_1000"]) - (((((data["classic_sta_lta2_mean"]) / 2.0)) / 2.0)))))) + (np.tanh((data["q95_roll_mean_10"]))))/2.0)))) +
            1.0*np.tanh((((((data["av_change_rate_roll_mean_10"]) + (data["abs_max_roll_mean_10"]))/2.0)) * (((data["q001"]) + ((((data["q95_roll_std_100"]) + (data["q001"]))/2.0)))))) +
            1.0*np.tanh(((data["q01_roll_mean_100"]) * (((((((((data["mean_change_rate"]) / 2.0)) * (data["av_change_abs_roll_mean_100"]))) * (data["classic_sta_lta4_mean"]))) - (((data["min_roll_std_1000"]) / 2.0)))))) +
            1.0*np.tanh(((data["Moving_average_1500_mean"]) * (((data["max_last_10000"]) * (((data["ave_roll_std_100"]) * (((data["max_last_10000"]) - (((data["abs_max_roll_std_1000"]) * 2.0)))))))))) +
            0.989058*np.tanh(np.tanh((((data["av_change_rate_roll_mean_100"]) * (((data["med"]) * (((((data["q001"]) + (data["iqr"]))) * 2.0)))))))) +
            0.909340*np.tanh(((data["max_roll_mean_100"]) * ((-1.0*((((data["med"]) - (((((data["q95_roll_mean_10"]) - (data["std_last_10000"]))) * ((-1.0*((data["med"]))))))))))))) +
            0.804220*np.tanh(((((((-1.0*((np.tanh((((0.0) - (data["max_first_50000"])))))))) + (data["abs_q95"]))/2.0)) * ((((data["classic_sta_lta2_mean"]) + (data["classic_sta_lta3_mean"]))/2.0)))) +
            1.0*np.tanh(((((data["q95_roll_std_1000"]) - (np.tanh((((data["q95_roll_mean_10"]) * 2.0)))))) * ((((((np.tanh((data["Moving_average_1500_mean"]))) / 2.0)) + (data["max_first_10000"]))/2.0)))) +
            1.0*np.tanh((((((((data["std_first_50000"]) + (((data["Hilbert_mean"]) * (((data["MA_400MA_BB_low_mean"]) + (data["av_change_abs_roll_std_100"]))))))/2.0)) / 2.0)) * (data["av_change_abs_roll_std_100"]))) +
            1.0*np.tanh(((data["max_first_50000"]) * (((((((data["min_roll_mean_100"]) * (data["count_big"]))) / 2.0)) * (data["std_last_50000"]))))) +
            1.0*np.tanh(data["abs_min"]) +
            0.682689*np.tanh((((((((data["std_first_50000"]) * (np.tanh((data["abs_q01"]))))) / 2.0)) + (((data["std_first_50000"]) * (data["ave_roll_mean_1000"]))))/2.0)) +
            0.962485*np.tanh(((((data["av_change_rate_roll_mean_100"]) * (((data["MA_700MA_BB_high_mean"]) * ((-1.0*((((((data["q95_roll_mean_1000"]) * 2.0)) / 2.0))))))))) / 2.0)) +
            1.0*np.tanh((((((((data["av_change_abs_roll_std_10"]) * (data["q01_roll_std_100"]))) + (((data["std_roll_mean_1000"]) - (((data["std_last_10000"]) / 2.0)))))/2.0)) * (np.tanh((data["std_last_10000"]))))) +
            0.833920*np.tanh(((data["med"]) * (((((data["iqr"]) - (((data["q01_roll_std_1000"]) - (data["q01_roll_mean_100"]))))) / 2.0)))) +
            1.0*np.tanh(((data["std_last_10000"]) * (((data["min_roll_mean_1000"]) + ((((data["q05_roll_std_10"]) + ((-1.0*(((((data["min_roll_mean_1000"]) + (data["std_last_10000"]))/2.0))))))/2.0)))))) +
            0.923017*np.tanh(np.tanh((((((data["av_change_abs_roll_std_100"]) * (np.tanh((np.tanh((((data["av_change_abs_roll_std_1000"]) * (data["med"]))))))))) * (data["av_change_abs_roll_std_1000"]))))) +
            1.0*np.tanh((((((np.tanh((data["max_last_10000"]))) + (0.0))/2.0)) / 2.0)) +
            1.0*np.tanh((((((np.tanh((((data["abs_q05"]) * (data["av_change_abs_roll_mean_100"]))))) / 2.0)) + (np.tanh((((((np.tanh((data["kurt"]))) / 2.0)) / 2.0)))))/2.0)) +
            0.782337*np.tanh(((data["abs_min"]) * ((-1.0*((0.0)))))) +
            0.908558*np.tanh(data["abs_min"]) +
            0.769832*np.tanh(((np.tanh((((data["mean_change_rate"]) * ((((data["MA_700MA_BB_high_mean"]) + ((-1.0*((data["avg_first_10000"])))))/2.0)))))) / 2.0)) +
            0.926925*np.tanh(np.tanh((((data["std_first_10000"]) * (((((((data["mean_change_rate_last_50000"]) + (data["abs_q01"]))/2.0)) + (data["classic_sta_lta4_mean"]))/2.0)))))) +
            1.0*np.tanh(((((((((data["Moving_average_6000_mean"]) - (np.tanh((((data["std_last_10000"]) / 2.0)))))) * (((data["std_last_10000"]) / 2.0)))) / 2.0)) / 2.0)) +
            0.982024*np.tanh(((((data["av_change_abs_roll_mean_10"]) * (((data["av_change_abs_roll_mean_10"]) * (((data["q05_roll_std_100"]) / 2.0)))))) / 2.0)) +
            0.794060*np.tanh(((data["q01_roll_std_1000"]) * (((data["MA_700MA_std_mean"]) * ((-1.0*(((((data["ave_roll_std_10"]) + (data["min_first_10000"]))/2.0))))))))) +
            0.995701*np.tanh(((data["q05_roll_std_100"]) * (((data["q05_roll_std_100"]) * (((data["abs_q01"]) + (((data["ave_roll_mean_1000"]) * (data["count_big"]))))))))) +
            1.0*np.tanh(((((((((((((((((data["q01_roll_mean_100"]) / 2.0)) / 2.0)) / 2.0)) + (data["av_change_abs_roll_mean_100"]))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.784291*np.tanh(data["abs_min"]) +
            0.973427*np.tanh((((((((data["q95_roll_mean_10"]) * (data["Hilbert_mean"]))) * (data["abs_q05"]))) + (((((data["av_change_abs_roll_std_10"]) / 2.0)) * (data["mean_change_rate_first_50000"]))))/2.0)) +
            0.870262*np.tanh((((((-1.0*((((data["abs_max_roll_std_10"]) * (np.tanh((((data["q99_roll_mean_10"]) * (data["std"])))))))))) / 2.0)) / 2.0)) +
            1.0*np.tanh(((((((data["ave_roll_std_100"]) * (((((((data["q99"]) * (data["mean_change_rate_last_10000"]))) / 2.0)) - (((data["av_change_abs_roll_mean_1000"]) / 2.0)))))) / 2.0)) / 2.0)) +
            0.668230*np.tanh(((((np.tanh((((np.tanh((((data["av_change_rate_roll_std_1000"]) * (((((data["abs_max_roll_std_10"]) / 2.0)) * (data["av_change_abs_roll_std_1000"]))))))) * 2.0)))) * 2.0)) / 2.0)) +
            0.949199*np.tanh(((0.0) + ((((-1.0*((np.tanh((((data["kurt"]) * (((0.0) + (data["max_first_10000"])))))))))) / 2.0)))) +
            1.0*np.tanh(((data["abs_min"]) / 2.0)) +
            0.945682*np.tanh(((np.tanh((((data["max_to_min"]) * ((((data["min_last_10000"]) + (((data["abs_min"]) / 2.0)))/2.0)))))) / 2.0)) +
            1.0*np.tanh(0.0) +
            0.999609*np.tanh(np.tanh((((((data["abs_q01"]) / 2.0)) / 2.0)))) +
            1.0*np.tanh(0.0) +
            0.814771*np.tanh((((np.tanh((((data["av_change_abs_roll_std_1000"]) * (data["MA_400MA_std_mean"]))))) + (((data["av_change_abs_roll_std_1000"]) * (((data["av_change_abs_roll_std_1000"]) * (((data["std_roll_mean_1000"]) / 2.0)))))))/2.0)) +
            0.999218*np.tanh((((((((((((data["min_roll_mean_10"]) + (((data["q99_roll_std_1000"]) / 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.921063*np.tanh(((((((np.tanh((((((data["exp_Moving_average_3000_mean"]) + (data["abs_q01"]))) * (data["exp_Moving_average_3000_mean"]))))) / 2.0)) / 2.0)) / 2.0)) +
            0.783118*np.tanh(((((data["av_change_rate_roll_std_10"]) / 2.0)) * (((data["max_last_50000"]) * (((data["max_last_50000"]) * (((data["abs_q01"]) - (data["max_last_50000"]))))))))) +
            0.977335*np.tanh(((data["abs_q01"]) * (((((data["abs_min"]) * (np.tanh((data["abs_q01"]))))) / 2.0)))) +
            0.557249*np.tanh(((((np.tanh((((np.tanh((((data["mean_change_rate"]) * (data["Hilbert_mean"]))))) * 2.0)))) / 2.0)) - (0.0))) +
            1.0*np.tanh((0.0)) +
            0.886284*np.tanh((((((0.0) + (np.tanh((((((((((data["Moving_average_1500_mean"]) / 2.0)) + (np.tanh((data["q05_roll_std_100"]))))) * 2.0)) * 2.0)))))/2.0)) / 2.0)) +
            0.998046*np.tanh((-1.0*(((((((((data["min_roll_std_1000"]) / 2.0)) + ((((np.tanh((data["classic_sta_lta1_mean"]))) + (data["sum"]))/2.0)))/2.0)) / 2.0))))) +
            0.847597*np.tanh(((data["abs_min"]) / 2.0)) +
            0.818679*np.tanh(data["abs_q01"]) +
            0.620946*np.tanh((((((((data["mean_diff"]) + (((((((((((data["mean_diff"]) + (data["abs_q01"]))/2.0)) + (0.0))/2.0)) / 2.0)) / 2.0)))/2.0)) / 2.0)) / 2.0)) +
            0.836655*np.tanh(data["abs_q01"]) +
            1.0*np.tanh(data["abs_q01"]) +
            0.937085*np.tanh(data["abs_min"]) +
            1.0*np.tanh(data["abs_min"]) +
            1.0*np.tanh(np.tanh((((((((np.tanh((((((data["max_roll_std_1000"]) * (data["std_first_10000"]))) * 2.0)))) / 2.0)) / 2.0)) / 2.0)))) +
            0.908558*np.tanh(((((((data["std_first_10000"]) * ((((data["q05_roll_mean_1000"]) + (data["abs_q01"]))/2.0)))) / 2.0)) / 2.0)) +
            0.881985*np.tanh(((((np.tanh((((data["count_big"]) / 2.0)))) / 2.0)) / 2.0)) +
            0.575615*np.tanh(((data["skew"]) * (np.tanh((np.tanh((((((np.tanh((np.tanh((data["max_roll_mean_10"]))))) / 2.0)) / 2.0)))))))) +
            0.679172*np.tanh(data["abs_q01"]) +
            0.889019*np.tanh((-1.0*((((((((np.tanh((0.0))) * (0.0))) * 2.0)) / 2.0))))) +
            0.999609*np.tanh(0.0) +
            0.998437*np.tanh(data["abs_min"]) +
            0.555295*np.tanh(((data["Moving_average_6000_mean"]) - (data["ave_roll_mean_1000"]))) +
            0.736616*np.tanh(0.0) +
            0.954279*np.tanh(data["abs_min"]) +
            0.844471*np.tanh(data["abs_q01"]) +
            0.587730*np.tanh(data["abs_min"]) +
            0.875342*np.tanh(data["abs_min"]) +
            0.859711*np.tanh(np.tanh((0.0))) +
            0.999218*np.tanh(((((data["avg_last_10000"]) * (data["Moving_average_3000_mean"]))) * ((((data["max_roll_std_1000"]) + (((0.0) - (data["std_roll_std_1000"]))))/2.0)))) +
            1.0*np.tanh(((np.tanh((((np.tanh((((data["ave_roll_mean_10"]) * (((data["av_change_rate_roll_std_10"]) * (data["q01_roll_mean_10"]))))))) / 2.0)))) / 2.0)) +
            0.867917*np.tanh(((np.tanh((((data["q01_roll_mean_10"]) * (((data["med"]) * (np.tanh((data["MA_700MA_BB_high_mean"]))))))))) / 2.0)) +
            0.847206*np.tanh(np.tanh((np.tanh((((((((data["q99_roll_mean_100"]) - ((-1.0*((data["min"])))))) / 2.0)) / 2.0)))))) +
            0.998828*np.tanh(((((np.tanh((((((data["q95_roll_mean_1000"]) - (data["avg_first_10000"]))) / 2.0)))) / 2.0)) / 2.0)) +
            0.843689*np.tanh(0.0) +
            0.881594*np.tanh(((data["min_last_50000"]) * ((((-1.0*(((-1.0*(((((data["abs_q05"]) + (((0.0) / 2.0)))/2.0)))))))) * (data["mean_diff"]))))) +
            1.0*np.tanh(((((0.0) / 2.0)) * 2.0)) +
            0.581477*np.tanh(0.0) +
            0.707698*np.tanh(0.0) +
            0.999218*np.tanh(((((np.tanh((((np.tanh((((data["av_change_rate_roll_mean_100"]) * ((-1.0*((data["iqr"])))))))) / 2.0)))) / 2.0)) / 2.0)) +
            0.884330*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(((((np.tanh(((((0.0) + (np.tanh((((data["sum"]) * (data["Moving_average_6000_mean"]))))))/2.0)))) / 2.0)) / 2.0)) +
            1.0*np.tanh(((((((np.tanh((np.tanh((((np.tanh((((((-1.0) / 2.0)) * 2.0)))) / 2.0)))))) / 2.0)) / 2.0)) / 2.0)) +
            0.787808*np.tanh(((((((data["std_last_10000"]) / 2.0)) * ((((((np.tanh((data["ave_roll_mean_1000"]))) * (data["max_last_10000"]))) + (((data["abs_q01"]) / 2.0)))/2.0)))) / 2.0)) +
            0.994529*np.tanh(((((((((data["abs_q01"]) + (np.tanh((np.tanh((data["abs_max_roll_mean_10"]))))))/2.0)) + (((((-1.0*(((0.17002706229686737))))) + (data["abs_q01"]))/2.0)))/2.0)) / 2.0)) +
            0.850332*np.tanh(0.0) +
            0.699101*np.tanh(((data["abs_q01"]) / 2.0)) +
            0.996874*np.tanh(((data["Moving_average_1500_mean"]) - (data["mean"]))) +
            0.314576*np.tanh(((((data["std_last_10000"]) * (np.tanh(((((data["std"]) + (data["min_last_10000"]))/2.0)))))) / 2.0)) +
            0.731536*np.tanh(data["abs_min"]) +
            0.998437*np.tanh(((((data["q95_roll_mean_10"]) + ((-1.0*((((data["q95"]) + (data["abs_q01"])))))))) / 2.0)) +
            0.661977*np.tanh(np.tanh((((((np.tanh((((data["abs_min"]) - ((((np.tanh((data["max_roll_mean_1000"]))) + (data["q95"]))/2.0)))))) / 2.0)) / 2.0)))) +
            0.932005*np.tanh(((np.tanh(((((((-1.0*((data["max_roll_mean_100"])))) + (data["abs_std"]))) / 2.0)))) / 2.0)) +
            0.753810*np.tanh(((((((np.tanh(((((data["kurt"]) + (((data["classic_sta_lta4_mean"]) - (data["abs_mean"]))))/2.0)))) / 2.0)) / 2.0)) / 2.0)) +
            0.998046*np.tanh(0.0) +
            0.996092*np.tanh(0.0) +
            0.423603*np.tanh(((np.tanh((((((data["mean_change_rate_first_10000"]) / 2.0)) / 2.0)))) / 2.0)) +
            0.998828*np.tanh(data["abs_q01"]) +
            0.647909*np.tanh((0.0)) +
            0.801485*np.tanh((((((-1.0*((((data["q05_roll_std_100"]) * (((((data["max_to_min"]) / 2.0)) / 2.0))))))) / 2.0)) / 2.0)) +
            0.999609*np.tanh(((((((data["av_change_abs_roll_std_1000"]) * (((data["min_roll_mean_100"]) - (data["min_roll_mean_10"]))))) / 2.0)) / 2.0)) +
            0.999218*np.tanh(((((((((((data["av_change_abs_roll_std_100"]) / 2.0)) / 2.0)) / 2.0)) - (np.tanh(((((data["abs_q01"]) + (((data["av_change_abs_roll_std_100"]) / 2.0)))/2.0)))))) / 2.0)) +
            0.998828*np.tanh(data["abs_q01"]) +
            0.939039*np.tanh(((np.tanh((np.tanh((data["abs_min"]))))) / 2.0)) +
            0.546307*np.tanh(data["abs_q01"]) +
            0.998828*np.tanh(((data["mean_change_rate_first_50000"]) * ((((-1.0*((np.tanh((((((np.tanh((((data["Moving_average_1500_mean"]) / 2.0)))) / 2.0)) / 2.0))))))) / 2.0)))) +
            1.0*np.tanh(0.0) +
            0.998437*np.tanh(((data["abs_q05"]) * ((((data["min_first_10000"]) + (data["std_first_10000"]))/2.0)))) +
            0.785854*np.tanh(((((((0.0) / 2.0)) / 2.0)) * 2.0)) +
            0.705354*np.tanh(0.0) +
            0.781946*np.tanh(((((np.tanh(((-1.0*((((((data["av_change_abs_roll_mean_10"]) / 2.0)) + (np.tanh((data["av_change_abs_roll_std_1000"])))))))))) / 2.0)) / 2.0)) +
            0.921454*np.tanh(np.tanh((((np.tanh((np.tanh((np.tanh((np.tanh((((((((data["mean_change_rate_first_50000"]) * (data["std_first_50000"]))) * 2.0)) * 2.0)))))))))) / 2.0)))) +
            1.0*np.tanh(0.0) +
            0.731145*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(data["abs_q01"]) +
            0.668621*np.tanh(data["abs_min"]) +
            0.932786*np.tanh(((((((data["mean_diff"]) * ((((((data["max_last_10000"]) + (data["q01_roll_mean_1000"]))/2.0)) / 2.0)))) / 2.0)) / 2.0)) +
            0.728019*np.tanh((-1.0*((((((((data["av_change_abs_roll_mean_10"]) / 2.0)) / 2.0)) * ((((data["Moving_average_700_mean"]) + (0.0))/2.0))))))) +
            0.828058*np.tanh(np.tanh((((((((data["std_first_50000"]) / 2.0)) * (((np.tanh((data["min_roll_std_10"]))) / 2.0)))) * (data["q05_roll_mean_10"]))))) +
            0.923017*np.tanh(((np.tanh(((((data["count_big"]) + (((np.tanh((data["std_roll_mean_1000"]))) - (data["abs_max_roll_mean_1000"]))))/2.0)))) / 2.0)) +
            0.999609*np.tanh(data["abs_min"]) +
            0.917155*np.tanh(((((data["av_change_abs_roll_std_100"]) * (((np.tanh((data["mean_change_rate_last_50000"]))) / 2.0)))) / 2.0)) +
            1.0*np.tanh(((((((np.tanh(((-1.0*((np.tanh((((data["Moving_average_6000_mean"]) * (np.tanh((data["q05_roll_std_100"])))))))))))) / 2.0)) / 2.0)) / 2.0)) +
            0.630715*np.tanh(((((data["mean_diff"]) * (((data["std_last_10000"]) * (np.tanh((data["min_first_50000"]))))))) / 2.0)) +
            0.501368*np.tanh(np.tanh((((np.tanh((data["min_first_10000"]))) * ((-1.0*((((np.tanh((((data["classic_sta_lta1_mean"]) - (0.0))))) / 2.0))))))))) +
            0.423603*np.tanh(((np.tanh((((data["std_first_50000"]) * (((data["max_last_10000"]) * (((data["q01_roll_mean_100"]) * 2.0)))))))) / 2.0)) +
            0.722157*np.tanh(((((np.tanh(((-1.0*((((data["min_first_50000"]) / 2.0))))))) / 2.0)) / 2.0)) +
            0.424775*np.tanh(((data["exp_Moving_average_300_mean"]) - (data["Moving_average_6000_mean"]))) +
            0.999609*np.tanh(data["abs_min"]) +
            0.812427*np.tanh(np.tanh(((((-1.0*((data["classic_sta_lta3_mean"])))) * (((((((data["std_first_50000"]) / 2.0)) / 2.0)) * (data["std_first_10000"]))))))) +
            0.540055*np.tanh(((((np.tanh((np.tanh(((((((data["q05"]) + (data["abs_min"]))/2.0)) / 2.0)))))) / 2.0)) * (data["max_first_10000"]))) +
            0.999609*np.tanh(((np.tanh((((((((data["min_roll_std_10"]) / 2.0)) / 2.0)) * (((data["min_roll_std_10"]) / 2.0)))))) / 2.0)) +
            0.998046*np.tanh(np.tanh((((((data["max_first_50000"]) * (((data["abs_max_roll_std_100"]) - (data["MA_400MA_std_mean"]))))) / 2.0)))) +
            0.998437*np.tanh((((np.tanh((((data["q01_roll_std_100"]) - (data["q05_roll_std_100"]))))) + (np.tanh((((data["iqr"]) * (data["std_roll_mean_1000"]))))))/2.0)) +
            1.0*np.tanh(data["abs_q01"]) +
            0.738179*np.tanh(((((np.tanh(((-1.0*((((((data["skew"]) * ((-1.0*((((data["abs_max_roll_mean_100"]) + (data["q05_roll_mean_10"])))))))) * 2.0))))))) / 2.0)) / 2.0)) +
            0.790934*np.tanh(((np.tanh(((((((-1.0*(((((data["av_change_rate_roll_mean_10"]) + (((np.tanh((data["std_roll_mean_100"]))) / 2.0)))/2.0))))) / 2.0)) * (data["av_change_abs_roll_mean_100"]))))) / 2.0)) +
            0.930442*np.tanh((((((((((data["av_change_abs_roll_mean_10"]) + (data["abs_q01"]))/2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.980852*np.tanh(((np.tanh((((((np.tanh(((-1.0*((data["abs_q05"])))))) / 2.0)) * (((data["mad"]) / 2.0)))))) / 2.0)) +
            0.999609*np.tanh(data["abs_min"]) +
            0.991012*np.tanh(data["abs_q01"]) +
            0.855021*np.tanh(((((np.tanh(((-1.0*((((data["min_last_10000"]) * ((((data["MA_400MA_std_mean"]) + (data["min_last_10000"]))/2.0))))))))) / 2.0)) / 2.0)) +
            0.997655*np.tanh(((((((np.tanh((((np.tanh((((data["Moving_average_1500_mean"]) * (data["ave_roll_mean_100"]))))) / 2.0)))) / 2.0)) / 2.0)) / 2.0)) +
            0.980852*np.tanh((-1.0*((((((((np.tanh((np.tanh(((((1.0) + ((-1.0*((np.tanh((data["std_roll_mean_10"])))))))/2.0)))))) / 2.0)) / 2.0)) / 2.0))))) +
            0.991794*np.tanh(0.0) +
            1.0*np.tanh(((((((((((data["max_last_10000"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.999609*np.tanh(data["abs_q01"]) +
            1.0*np.tanh(data["abs_q01"]) +
            0.723329*np.tanh(data["abs_min"]) +
            0.607659*np.tanh(data["abs_q01"]) +
            0.920281*np.tanh((((((np.tanh((((data["av_change_abs_roll_mean_100"]) * (data["std_roll_std_1000"]))))) + (((0.0) / 2.0)))/2.0)) / 2.0)) +
            0.843689*np.tanh(data["abs_q01"]) +
            0.623290*np.tanh(data["abs_min"]) +
            1.0*np.tanh(np.tanh((((((data["kurt"]) - (data["exp_Moving_average_3000_mean"]))) * (((data["kurt"]) * (((data["Moving_average_3000_mean"]) - (data["exp_Moving_average_3000_mean"]))))))))) +
            0.836655*np.tanh(0.0) +
            0.695584*np.tanh(data["abs_q01"]) +
            0.893708*np.tanh((((((np.tanh(((((-1.0*((data["iqr"])))) * 2.0)))) + (np.tanh((np.tanh((data["q05_roll_std_10"]))))))/2.0)) / 2.0)) +
            0.998828*np.tanh((((((-1.0*((np.tanh((((data["max_last_10000"]) * (data["av_change_abs_roll_std_10"])))))))) / 2.0)) / 2.0)) +
            0.845643*np.tanh(((((((((np.tanh((data["q95_roll_std_1000"]))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
            0.893708*np.tanh(0.0) +
            0.998046*np.tanh(0.0) +
            1.0*np.tanh(((data["exp_Moving_average_6000_mean"]) - (data["ave_roll_mean_10"]))) +
            0.770223*np.tanh(((np.tanh((((data["abs_max"]) * ((-1.0*((data["mean"])))))))) * (((((data["max_to_min"]) * (np.tanh((data["max_to_min_diff"]))))) / 2.0)))) +
            0.999218*np.tanh(((np.tanh((((((((data["ave_roll_mean_10"]) * (data["std_last_50000"]))) / 2.0)) / 2.0)))) / 2.0)) +
            0.771004*np.tanh(((data["std_first_10000"]) * ((((((data["count_big"]) + (((0.0) * (np.tanh(((((data["count_big"]) + (0.0))/2.0)))))))/2.0)) / 2.0)))) +
            0.856585*np.tanh(np.tanh((((((((((np.tanh(((-1.0*((data["count_big"])))))) + (((np.tanh((data["classic_sta_lta3_mean"]))) / 2.0)))) / 2.0)) / 2.0)) / 2.0)))) +
            0.484173*np.tanh(data["abs_q01"]) +
            0.858148*np.tanh(((((np.tanh(((((((((0.0) + ((((0.0) + (data["min_roll_mean_1000"]))/2.0)))/2.0)) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) +
            0.999218*np.tanh(0.0) +
            0.996092*np.tanh(data["abs_q01"]) +
            0.899961*np.tanh((((((np.tanh((0.0))) + (((np.tanh((np.tanh((np.tanh((((data["q01_roll_std_100"]) + (data["q01_roll_std_10"]))))))))) / 2.0)))/2.0)) / 2.0)) +
            0.554513*np.tanh(np.tanh((((((np.tanh((((np.tanh((data["MA_400MA_BB_low_mean"]))) / 2.0)))) / 2.0)) / 2.0)))) +
            0.878859*np.tanh(((np.tanh((((data["abs_q01"]) / 2.0)))) / 2.0)) +
            0.0*np.tanh(data["abs_min"]) +
            0.999218*np.tanh(data["abs_q01"]) +
            1.0*np.tanh(data["abs_min"]) +
            0.998437*np.tanh(0.0) +
            0.471278*np.tanh(((data["av_change_abs_roll_std_100"]) * ((-1.0*((((data["min_last_10000"]) * ((((((data["av_change_abs_roll_std_100"]) + (data["abs_min"]))/2.0)) / 2.0))))))))) +
            0.878859*np.tanh(0.0) +
            0.556467*np.tanh((((-1.0*(((((((((0.0) + ((((data["avg_last_10000"]) + (((data["avg_last_10000"]) / 2.0)))/2.0)))/2.0)) / 2.0)) / 2.0))))) / 2.0)) +
            0.457210*np.tanh(((((np.tanh((data["q01_roll_mean_100"]))) / 2.0)) / 2.0)) +
            0.677608*np.tanh(((((np.tanh((((np.tanh((data["std_first_50000"]))) / 2.0)))) / 2.0)) / 2.0)) +
            0.824541*np.tanh(data["abs_min"]) +
            0.999609*np.tanh(np.tanh(((((-1.0*((((((np.tanh((((data["mean_diff"]) * (np.tanh((data["av_change_rate_roll_mean_100"]))))))) / 2.0)) / 2.0))))) / 2.0)))) +
            0.865572*np.tanh(data["abs_min"]) +
            0.563111*np.tanh(0.0) +
            0.816725*np.tanh(data["abs_q01"]) +
            0.0*np.tanh(data["abs_min"]) +
            0.767097*np.tanh(0.0) +
            0.719031*np.tanh(data["abs_min"]) +
            0.617429*np.tanh(data["abs_q01"]) +
            0.997265*np.tanh((((-1.0*((np.tanh((((((((np.tanh(((((data["av_change_rate_roll_std_10"]) + (0.0))/2.0)))) / 2.0)) / 2.0)) / 2.0))))))) / 2.0)) +
            0.999609*np.tanh((0.0)) +
            0.859320*np.tanh((-1.0*((((data["sum"]) * (((data["av_change_abs_roll_mean_10"]) * (0.0)))))))) +
            0.297382*np.tanh(data["abs_q01"]) +
            0.467761*np.tanh(0.0) +
            0.999218*np.tanh(((data["min_roll_std_1000"]) * (((data["min_roll_std_1000"]) * (((data["ave_roll_mean_1000"]) - (data["ave_roll_mean_10"]))))))) +
            0.999218*np.tanh(data["abs_min"]) +
            0.892927*np.tanh(0.0) +
            0.763579*np.tanh(((((((np.tanh((np.tanh((data["max_roll_std_1000"]))))) * (((((data["q01_roll_std_10"]) * (data["abs_q05"]))) * 2.0)))) / 2.0)) / 2.0)) +
            0.999218*np.tanh(((((np.tanh((((data["std_first_50000"]) * ((-1.0*((data["std_last_10000"])))))))) * (data["min_roll_std_10"]))) / 2.0)) +
            0.739351*np.tanh(((((((np.tanh((((data["ave_roll_std_100"]) + (((data["Moving_average_1500_mean"]) * (np.tanh((data["exp_Moving_average_3000_mean"]))))))))) / 2.0)) / 2.0)) / 2.0)) +
            0.867136*np.tanh((((((((-1.0*(((((1.0) + (data["q05_roll_std_1000"]))/2.0))))) / 2.0)) / 2.0)) / 2.0)) +
            0.725674*np.tanh(0.0) +
            0.996874*np.tanh(data["abs_min"]) +
            1.0*np.tanh(((((data["av_change_abs_roll_std_100"]) * (((((((data["av_change_abs_roll_mean_100"]) * (data["min_roll_std_100"]))) / 2.0)) / 2.0)))) / 2.0)))
