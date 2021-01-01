import os
from pathlib import Path
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Concatenate, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from src.model.utils import create_path_labels, squeeze_excite_block

# Xception pretrain model input size
image_shape = (299, 299, 3)

data_path = Path(__file__).parent.parent.parent / "data" / "dog-images"

# 133
total_classes = len(os.listdir(data_path / "train"))

num_label_dict, df_train = create_path_labels(data_path / "train")
_, df_val = create_path_labels(data_path / "valid")
_, df_test = create_path_labels(data_path / "test")

xception = Xception(
    include_top=False, weights="imagenet", input_shape=image_shape, pooling=None
)
x = xception.output

# Original branch
gavg = GlobalAveragePooling2D()(x)
gmax = GlobalMaxPooling2D()(x)
original_concat = Concatenate(axis=-1)([gavg, gmax])
original_concat = Dropout(0.5)(original_concat)
original_final = Dense(total_classes, activation="softmax", name="original_out")(
    original_concat
)

# SE branch
se_out = squeeze_excite_block(x)
se_gavg = GlobalAveragePooling2D()(se_out)
se_gmax = GlobalMaxPooling2D()(se_out)
se_concat = Concatenate(axis=-1)([se_gavg, se_gmax,])
se_concat = Dropout(0.5)(se_concat)
se_final = Dense(total_classes, activation="softmax", name="se_out")(se_concat)

xception_model = Model(inputs=xception.input, outputs=[original_final, se_final])
