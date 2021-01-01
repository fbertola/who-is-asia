from tensorflow.keras.applications.xception import preprocess_input

from src.model import xception_model, num_label_dict, total_classes, image_shape
from src.model.data_generators import MultiOutputDataGenerator
from src.model.utils import calc_accuracy, combine_prediction

xception_model.load_weights("../trained_models/xception_pl.hdf5")

input_image = "<INPUT_IMAGE>>"

test_datagen = MultiOutputDataGenerator(
    images_paths=[input_image],
    labels=[""],
    batch_size=1,
    image_dimensions=image_shape,
    shuffle=False,
    augmenter=None,
    preprocessor=preprocess_input,
    return_label=False,
    total_classes=total_classes,
    output_names=["original_out", "se_out"],
)

pred = xception_model.predict(test_datagen, verbose=1)

calc_accuracy(combine_prediction(pred, [1.0, 1.0]), num_label_dict)

with open("labels.txt", mode="w") as label_file:
    for key in sorted(num_label_dict):
        label_file.write("{} {}\n".format(key, num_label_dict[key]))
