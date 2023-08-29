from sklearn.impute import SimpleImputer
import tensorflow as tf
import numpy as np
import pandas as pd


def make_train_set(window_size, raw_data, predict_size):
    data = tf.data.Dataset.from_tensor_slices(raw_data)
    data = data.window(window_size + predict_size, drop_remainder=True, shift=1)
    data = data.flat_map(lambda window: window.batch(window_size + predict_size))
    data = data.shuffle(raw_data.shape[0] + predict_size).map(
        lambda window: (window[:-predict_size], window[-predict_size:])
    )
    data = data.batch(32).prefetch(1)
    return data