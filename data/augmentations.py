from typing import Tuple
from types import SimpleNamespace
import imgaug.augmenters as iaa


def contrast(min=0.8, max=1.2, p=0.5):
    return iaa.Sometimes(p, iaa.contrast.LinearContrast(alpha=(min, max)))


def gamma(min=0.8, max=1.2, p=0.5):
    return iaa.Sometimes(p, iaa.Multiply((min, max), per_channel=0.1))


def motionblur(k=(3, 5), p=0.2):
    return iaa.Sometimes(p, iaa.blur.AverageBlur(k=k))


def jpeg_compr(min=5, max=20, p=0.1):
    return iaa.Sometimes(p, iaa.JpegCompression(compression=(min, max)))


def flip_lr(p=0.5):
    return iaa.Fliplr(p)


def cutout(min=0.2, max=0.5, p=0.2, squared=False, position='normal', cval=0):
    return iaa.Sometimes(p, iaa.Cutout(size=[min, max], squared=squared, cval=cval, position=position))


def train_augs():
    return iaa.Sequential([
        contrast(),
        gamma(),
        motionblur(),
        jpeg_compr(),
        flip_lr(),
        cutout()]
    )
