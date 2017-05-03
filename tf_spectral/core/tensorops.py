import tensorflow as tf
import numpy as np


def real_fft(x, only_abs=True, logarithmic=False, window=None):
    """
    Computes fft of a stack of time series.
    :param x: ndarray, shape=(n_series, n_samples)
            Input stack of time series
    :param only_abs: boolean
            return only absolute values (power spectra)
    :param logarithmic: boolean
            return in logarithmic scale; ignored if only_abs==False
    :param window: ndarray, shape=(n_samples, )
            boxing window (default none), np.array (n_samples)
    :return: ndarray, shape=(n_series, n_samples/2)
            with fft computed for every row
             dtype=np.float32 (if only_abs==True) or dtype=np.complex64 (if only_abs==False)
    """
    n_series, n_samples = x.shape

    tensor_x = tf.Variable(x, dtype=tf.float32)
    if window is not None:
        assert (window.size == n_samples), 'Window must be size n_samples'
        vector_win = tf.Variable(window.flatten(), dtype=tf.float32)
        tensor_win = tf.reshape(tf.tile(vector_win, [n_series]), [n_series, n_samples])
        real_x = tf.multiply(tensor_x, tensor_win)
    else:
        real_x = tensor_x
    img_x = tf.Variable(np.zeros_like(x), dtype=tf.float32)
    complex_x = tf.complex(real_x, img_x)
    complex_y = tf.fft(complex_x)[:, :int(x.shape[-1] / 2)]

    if only_abs:
        amps_y = tf.abs(complex_y)
        if logarithmic:
            log_10_inv = tf.constant(1./np.log(10.), dtype=tf.float32)
            log_amps = tf.log(amps_y)
            fft = tf.multiply(log_amps, log_10_inv)
        else:
            fft = amps_y
    else:
        fft = complex_y

    model = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(model)
        s = sess.run(fft)
    return s
