import imgaug as ia
import tensorflow as tf
from imgaug import augmenters as iaa


def create_augmenter(train=True):
    # from https://github.com/aleju/imgaug
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    if train:
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(
                    iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,  # random mode from all available modes will be sampled per image.
                        pad_cval=(0, 255)
                        # The constant value to use if the pad mode is constant or the end value to use if the mode is linear_ramp
                    )
                ),
                sometimes(
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        cval=(
                            0,
                            255,
                        ),  # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL,
                        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
            ],
        )
    else:
        pass
    return seq


def scheduler(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))
