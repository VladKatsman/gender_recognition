import os
import re
import numpy as np
from PIL import Image


def resize_keep_aspect_ratio(path,
                             target_shape=(224, 224)) -> Image:

    image = Image.open(path)
    ht, wt = target_shape
    h, w = image.size

    # Minimum distortion rescaling
    hr, wr = (ht, int(w * ht / h)) if ht / h < wt / w else (int(h * wt / w), wt)
    # Padding size
    pad_h = ht - hr
    pad_w = wt - wr

    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    # Perform resize
    resized_image = image.resize((hr, wr), Image.BILINEAR)
    resized_image = np.array(resized_image)
    resized_image = np.pad(resized_image,
                          [(pad_w_before, pad_w_after), (pad_h_before, pad_h_after), (0, 0)])

    return Image.fromarray(resized_image)


def list_files_from_dirs_subdirs(root) -> list:
    """List all files of specified root folder

    Returns:
        list: list of all files and subdirs
    """
    return natural_sort(os.path.join(path, name)\
                         for path, subdirs, files in\
                            os.walk(root) for name in files)


def natural_sort(lst) -> list:
    """
    :return: Human-like sorted lst
    """
    convert      = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)


def split_list_to_chunks(lst, n) -> list:
    return [lst[i:i + n] for i in range(0, len(lst), n)]
