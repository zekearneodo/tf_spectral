import tensorflow as tf


def tf_version():
    tf_ver = tf.__version__
    bin_tf_ver = ''.join([c for c in tf_ver if c not in '.'])
    int_tf_ver = int(bin_tf_ver, 2)
    # 1.0.x is 4-5
    return int_tf_ver


TFVERSION = tf_version()
