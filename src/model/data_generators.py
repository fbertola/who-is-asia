import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


# reference: https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
class BaseDataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        images_paths,
        labels,
        batch_size=64,
        image_dimensions=(512, 512, 3),
        shuffle=False,
        augmenter=None,
        preprocessor=None,
        return_label=True,
        total_classes=None,
    ):
        self.labels = labels  # array of labels
        self.images_paths = images_paths  # array of image paths
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.shuffle = shuffle  # shuffle bool
        self.augmenter = augmenter  # augmenter
        self.preprocessor = preprocessor
        self.return_label = return_label
        self.total_classes = total_classes
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def gather_batch_item(self, index):
        "Generate one batch of data"
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        images = [cv2.imread(self.images_paths[k]) for k in indexes]

        # preprocess and augment data
        if self.augmenter:
            images = self.augmenter.augment_images(images)

        images = np.array(
            [self.preprocess_image(cv2.resize(img, self.dim[:2])) for img in images]
        )

        if self.return_label:
            labels = np.array([self.labels[k] for k in indexes])
            labels = to_categorical(labels, num_classes=self.total_classes)
            return images, labels
        else:
            return images

    def __getitem__(self, index):
        return self.gather_batch_item(index)

    def preprocess_image(self, images):
        if self.preprocessor is None:
            images = images / 255.0
            pass
        else:
            images = self.preprocessor(images)
        return images


class MultiOutputDataGenerator(BaseDataGenerator):
    "Generates multiple output data for Keras"

    def __init__(
        self,
        images_paths,
        labels,
        batch_size=64,
        image_dimensions=(512, 512, 3),
        shuffle=False,
        augmenter=None,
        preprocessor=None,
        return_label=True,
        total_classes=None,
        output_names=None,
        tta_augmentors=None,
    ):
        # Init parent's parameter
        super().__init__(
            images_paths,
            labels,
            batch_size,
            image_dimensions,
            shuffle,
            augmenter,
            preprocessor,
            return_label,
            total_classes,
        )

        self.output_names = output_names
        self.tta_augmentors = tta_augmentors

    def __getitem__(self, index):
        if self.return_label:
            images, labels = self.gather_batch_item(index)
            output_dict = {}
            # Copy labels to each output name
            for output_name in self.output_names:
                output_dict[output_name] = labels
            if self.tta_augmentors != None:
                images = self.get_tta_images(images)
            return images, output_dict
        else:
            images = self.gather_batch_item(index)
            if self.tta_augmentors != None:
                images = self.get_tta_images(images)
            return images

    def get_tta_images(self, images):
        aug_images = []
        # Original
        aug_images.append(images)
        for augmentor in self.tta_augmentors:
            aug_images.append(augmentor.augment_images(images))
        images = aug_images
        return images
