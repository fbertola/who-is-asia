import os

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Permute, multiply


def create_path_labels(path: os.PathLike):
    path = Path(path)
    all_classes = os.listdir(path)
    label_dict = {}
    img_paths, label_list = [], []
    for label_name in all_classes:
        label_num, dog_name = label_name.split(".")
        # Start with 0
        label_num = int(label_num) - 1
        label_dict[int(label_num)] = dog_name
        for image_name in os.listdir(path / label_name):
            img_paths.append(path / label_name / image_name)
            label_list.append(label_num)
    df = pd.DataFrame({"img_path": img_paths, "label": label_list})
    return label_dict, df


def squeeze_excite_block(tensor, ratio=16):
    # From: https://github.com/titu1994/keras-squeeze-excite-network
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = K.int_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)

    if K.image_data_format() == "channels_first":
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def combine_prediction(predictions, weights=(1.0, 1.0)):
    predictions = np.array(predictions)
    weights = np.array(weights).reshape(predictions.shape[0], 1, 1)
    return np.mean(np.multiply(predictions, weights), axis=0)


def calc_accuracy(predictions, truth):
    if type(predictions) != list:
        predictions = [predictions]
    accuracy = []
    for prediction in predictions:
        prediction = np.argmax(prediction, axis=-1)
        correct_nums = (prediction == truth).sum()
        accuracy.append(correct_nums / len(prediction))
    return accuracy
