import os

import numpy as np
import matplotlib.pyplot as plt


def epsilon_linear_scheduler(epsilon, min_epsilon, decay):
    return max(decay * epsilon, min_epsilon)


def numpy_ewma_vectorized_v2(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def make_performance_plot(model_path: str):
    performance = np.load(os.path.join(model_path, "performance.npy"), allow_pickle=True)
    performance = performance.astype(float)

    steps, rewards, length = performance[:, 0], performance[:, 1], performance[:, 2]

    window = 150

    # Smooth rewards and lengths
    ewma_rewards = numpy_ewma_vectorized_v2(rewards, window)
    ewma_length = numpy_ewma_vectorized_v2(length, window)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(steps, ewma_rewards)
    ax[0].set_title("EWMA Reward")

    ax[1].plot(steps, ewma_length)
    ax[1].set_title("EWMA Length")

    fig.savefig(os.path.join(model_path, "performance.png"), dpi=80)
    plt.close(fig)
