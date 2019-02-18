import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data(acoustic_data, ttf):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(acoustic_data, 'g')
    ax2.plot(ttf, 'b')

    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Acoustic Data', color='g')
    ax2.set_ylabel('Time to failure0', color='b')

    plt.show()