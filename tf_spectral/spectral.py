import numpy as np
from tf_spectral.core.data import overlap
from tf_spectral.core.tensorops import real_fft


def spectrogram_tf(X, log=True, db_cut=65, fft_size=512, step_size=64, window=None, device='/gpu:0'):
    """
    creates a spectrogram of the 1d time series X using tensorflow to compute the fft
    :param x: ndarray, shape=(n_samples, ) Input time series
    :param db_cut: float, threshold (in dB, relative to total max of the whole spectrogram)
    :param fft_size: int, (samples) size of the window to use for every spectral slice
    :param step_size: int, (samples) stride for the spectral slices
    :param log: boolean, whether to take the log
    :param window: ndarray, shape=(fft_size, ). If entered, the window function for the fft.
                   must be an array of the same size of fft_size
    :param device: str, device to use; ('/cpu:0', for instance).
    :return: ndarray, shape=(n_steps, fft_size/2) with the spectrogram
    """
    x = overlap(X, fft_size, step_size)
    specgram = real_fft(x, only_abs=True, logarithmic=log, window=window, device=device)
    max_specgram = np.max(specgram)

    # do the cut_off. Values are amplitude, not power, hence db = -20*log(V/V_0)
    if log:
        # threshold = pow(10, max_specgram) * pow(10, -db_cut*0.05)
        # specgram[specgram < np.log10(threshold)] = np.log10(threshold)
        log10_threshhold = max_specgram - db_cut*0.05
        specgram[specgram < log10_threshhold] = log10_threshhold
        # specgram /= specgram.max()  # volume normalize to max 1
    else:
        threshold = max_specgram * pow(10, -db_cut*0.05)
        specgram[specgram < threshold] = threshold  # set anything less than the threshold as the threshold
    return specgram
