import os

import pandas as pd
import tensorflow.keras.backend as K
from imgaug import augmenters as iaa
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import SGD

from src.main import calc_accuracy, combine_prediction
from src.model import (
    df_train,
    image_shape,
    total_classes,
    df_val,
    df_test,
    xception_model,
)
from src.model.data_generators import *
from src.training.utils import create_augmenter, scheduler

batch_size = 20

train_datagen = MultiOutputDataGenerator(
    images_paths=df_train["img_path"].values,
    labels=df_train["label"].values,
    batch_size=batch_size,
    image_dimensions=image_shape,
    shuffle=True,
    augmenter=create_augmenter(train=True),
    preprocessor=preprocess_input,
    return_label=True,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

val_datagen = MultiOutputDataGenerator(
    images_paths=df_val["img_path"].values,
    labels=df_val["label"].values,
    batch_size=5,
    image_dimensions=image_shape,
    shuffle=True,
    augmenter=None,
    preprocessor=preprocess_input,
    return_label=True,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

test_datagen = MultiOutputDataGenerator(
    images_paths=df_test["img_path"].values,
    labels=df_test["label"].values,
    batch_size=1,
    image_dimensions=image_shape,
    shuffle=False,
    augmenter=None,
    preprocessor=preprocess_input,
    return_label=False,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

checkpointer = ModelCheckpoint(
    filepath="trained_models/weights.best.Xception_best.hdf5",
    verbose=1,
    save_best_only=True,
)

logdir = "./logs/warmup"

if not os.path.exists(logdir):
    os.mkdir(logdir)

tensorboard_callback = TensorBoard(log_dir=logdir)

early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=15, restore_best_weights=True
)

lr_scheduler = LearningRateScheduler(scheduler)

for layer in xception_model.layers[:-15]:
    layer.trainable = False

xception_model.compile(
    optimizer="adam",
    loss={
        "original_out": "categorical_crossentropy",
        "se_out": "categorical_crossentropy",
    },
    loss_weights={"original_out": 1.0, "se_out": 1.0},
    metrics=["accuracy"],
)
history = xception_model.fit_generator(
    generator=train_datagen,
    validation_data=val_datagen,
    epochs=5,
    callbacks=[tensorboard_callback, early_stop, checkpointer],
    verbose=1,
)

for layer in xception_model.layers[:-15]:
    layer.trainable = True

logdir = "./logs/whole"

if not os.path.exists(logdir):
    os.mkdir(logdir)

tensorboard_callback = TensorBoard(log_dir=logdir)

xception_model.compile(
    optimizer="sgd",
    loss={
        "original_out": "categorical_crossentropy",
        "se_out": "categorical_crossentropy",
    },
    loss_weights={"original_out": 1.0, "se_out": 1.0},
    metrics=["accuracy"],
)

history = xception_model.fit_generator(
    generator=train_datagen,
    validation_data=val_datagen,
    epochs=100,
    callbacks=[tensorboard_callback, checkpointer],
    verbose=1,
)

xception_model.save_weights("trained_models/xception_whole_model.hdf5")
K.clear_session()

xception_model.load_weights("trained_models/xception_whole_model.hdf5")
pred = xception_model.predict_generator(generator=test_datagen, verbose=1,)

calc_accuracy(combine_prediction(pred, [1.0, 1.0]), df_test["label"].values)
xception_model.load_weights("trained_models/xception_best.hdf5")
pred = xception_model.predict_generator(generator=test_datagen, verbose=1,)
len(df_test["label"].values)
# Same weight
calc_accuracy(combine_prediction(pred, [1.0, 1.0]), df_test["label"].values)

# Get Augmentors
tta_augmentors = [iaa.Fliplr(1.0), iaa.Flipud(1.0)]

tta_test_datagen = MultiOutputDataGenerator(
    images_paths=df_test["img_path"].values,
    labels=df_test["label"].values,
    batch_size=1,
    image_dimensions=image_shape,
    shuffle=False,
    augmenter=None,
    preprocessor=preprocess_input,
    return_label=False,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
    tta_augmentors=tta_augmentors,
)

all_predictions = np.zeros((df_test.shape[0], total_classes))
count = 0
total_len = len(tta_test_datagen)
for images in tta_test_datagen:
    count += 1
    print("{}/{}".format(count, total_len), end="\r")
    preds = []
    for image in images:
        pred = xception_model.predict_on_batch(image)
        pred = combine_prediction(pred, [1.0, 1.0])
        preds.append(pred)
    all_predictions[count - 1] = combine_prediction(preds, [1.0, 1.0, 1.0])

calc_accuracy(np.array(all_predictions).reshape((836, 133)), df_test["label"].values)

val_pred = xception_model.predict_generator(generator=val_datagen, verbose=1,)
val_pred = combine_prediction(val_pred, [1.0, 1.0])

test_datagen = MultiOutputDataGenerator(
    images_paths=df_test["img_path"].values,
    labels=df_test["label"].values,
    batch_size=2,
    image_dimensions=image_shape,
    shuffle=False,
    augmenter=None,
    preprocessor=preprocess_input,
    return_label=False,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

test_pred = xception_model.predict_generator(generator=test_datagen, verbose=1,)

test_pred = combine_prediction(test_pred, [1.0, 1.0])

val_pseudolabels = np.argmax(val_pred, axis=-1)
test_pseudolabels = np.argmax(test_pred, axis=-1)

pseudo_data = pd.DataFrame(
    {
        "img_path": np.concatenate(
            [df_val["img_path"].values, df_test["img_path"].values]
        ),
        "label": np.concatenate([val_pseudolabels, test_pseudolabels]),
    }
)
pseudo_train = pd.concat([pseudo_data, df_train], axis=0)

pseudo_train_datagen = MultiOutputDataGenerator(
    images_paths=pseudo_train["img_path"].values,
    labels=pseudo_train["label"].values,
    batch_size=batch_size,
    image_dimensions=image_shape,
    shuffle=True,
    augmenter=create_augmenter(train=True),
    preprocessor=preprocess_input,
    return_label=True,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

checkpointer = ModelCheckpoint(
    filepath="trained_models/xception_pl.hdf5", verbose=1, save_best_only=True,
)

logdir = "./logs/pseudolabeling"
# Create target Directory if don't exist
if not os.path.exists(logdir):
    os.mkdir(logdir)

tensorboard_callback = TensorBoard(log_dir=logdir)

early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=15, restore_best_weights=True
)

xception_model.load_weights("trained_models/xception_whole_model.hdf5")

# Freeze pretrained part
for layer in xception_model.layers[:-15]:
    layer.trainable = True

sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=True)

xception_model.compile(
    optimizer=sgd,
    loss={
        "original_out": "categorical_crossentropy",
        "se_out": "categorical_crossentropy",
    },
    loss_weights={"original_out": 1.0, "se_out": 1.0},
    metrics=["accuracy"],
)

history = xception_model.fit_generator(
    generator=pseudo_train_datagen,
    validation_data=val_datagen,
    epochs=100,
    callbacks=[tensorboard_callback, early_stop, checkpointer, lr_scheduler],
    verbose=1,
)

pred = xception_model.predict_generator(generator=test_datagen, verbose=1,)

calc_accuracy(combine_prediction(pred, [1.0, 1.0]), df_test["label"].values)

xception_model.load_weights("trained_models/xception_pl.hdf5")
pred = xception_model.predict_generator(generator=test_datagen, verbose=1,)
calc_accuracy(combine_prediction(pred, [1.0, 1.0]), df_test["label"].values)
