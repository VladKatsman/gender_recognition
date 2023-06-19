import os
from p_tqdm import p_map
from .utils import resize_keep_aspect_ratio, list_files_from_dirs_subdirs


def remove_suffix_and_resize_images(p_to_dir: str) -> None:
    """ We would like:
    1. Remove redundant suffix ('.jpg')
    2. Resize all data to 224x224 keeping aspect ratio
    """
    images = list_files_from_dirs_subdirs(p_to_dir)

    # tqdm + multiprocessing
    p_map(remove_suffix_and_resize_image, images)


def remove_suffix_and_resize_image(p_to_image) -> None:
    image = resize_keep_aspect_ratio(p_to_image)
    image.save(p_to_image, quality=95)
    remove_suffix(p_to_image)


def remove_suffix(src: str) -> None:
    dst = src[:-4]
    os.rename(src, dst)


if __name__ == '__main__':
    p_to_images = '/home/noteme/data/gender_recognition'
    remove_suffix_and_resize_images(p_to_images)
