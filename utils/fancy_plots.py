import numpy as np
import matplotlib.pyplot as plt

def compare_methods(evaluation):
    plt.subplots(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(len(evaluation.index.tolist())), evaluation['mean'], yerr=evaluation['std'], align='center',
           alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('CV score')
    plt.xlabel('model')
    plt.xticks(np.arange(len(evaluation.index.tolist())), evaluation.index.tolist())
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(evaluation['mean'], evaluation['std'])

    model_list = evaluation.index.tolist()
    score_mean_list = evaluation['mean'].tolist()
    score_std_list = evaluation['std'].tolist()

    for i in range(len(model_list)):
        plt.annotate(model_list[i], (score_mean_list[i]+0.001, score_std_list[i]))

    plt.ylabel('std')
    plt.xlabel('mean')
    plt.grid()

    plt.show()