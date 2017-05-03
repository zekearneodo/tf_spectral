import numpy as np
from core.data import overlap
from core.tensorops import real_fft


def spectrogram_tf(X, log=True, db_cut=65, fft_size=512, step_size=64, window=None):
    """
    creates a spectrogram of the 1d time series X using tensorflow to compute the fft
    :param x: ndarray, shape=(n_samples, ) Input time series
    :param db_cut: float, threshold (in dB, relative to total max of the whole spectrogram)
    :param fft_size: int, (samples) size of the window to use for every spectral slice
    :param step_size: int, (samples) stride for the spectral slices
    :param log: boolean, whether to take the log
    :param window: ndarray, shape=(fft_size, ). If entered, the window function for the fft.
                   must be an array of the same size of fft_size
    :return: ndarray, shape=(n_steps, fft_size/2) with the spectrogram
    """
    x = overlap(X, fft_size, step_size)
    specgram = real_fft(x, only_abs=True, logarithmic=log, window=window)
    max_specgram = np.max(specgram)

    if log:
        threshold = pow(10, max_specgram) * pow(10, -db_cut*0.05)
        specgram[specgram < np.log10(threshold)] = np.log10(threshold)
        #specgram /= specgram.max()  # volume normalize to max 1
    else:
        threshold = max_specgram * pow(10, -db_cut*0.05)
        specgram[specgram < threshold] = threshold  # set anything less than the threshold as the threshold
    return specgram
