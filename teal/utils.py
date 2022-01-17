import tensorflow as tf


def tf_log10(input_tensor):
    numerator = tf.math.log(input_tensor)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
