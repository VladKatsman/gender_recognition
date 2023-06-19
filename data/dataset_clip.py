import random

import numpy as np
from PIL import Image
from imgaug import augmenters
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from .utils import list_files_from_dirs_subdirs, split_list_to_chunks
MASK = 0xffffffff
import clip


class CLIPItemDataset(Dataset):
    """The first stage of the data loading pipeline based on the ItemDict class.
    It implements a map-style Dataset that loads item images
    using the tuples (item_id, image_index)."""

    def __init__(self, p_to_images,
                 mean=(0.48145466, 0.4578275, 0.40821073),
                 std=(0.26862954, 0.26130258, 0.27577711),
                 augmenter: augmenters.Augmenter = None,
                 seed=1337,
                 shuffle=False):

        super(CLIPItemDataset, self).__init__()
        # # set reproducible seed
        np.random.seed(seed)
        random.seed(seed)
        self.mean = mean
        self.std = std
        self.images = list_files_from_dirs_subdirs(p_to_images)
        self.labels = get_labels(self.images)
        self.captions = get_captions(self.images)
        self.augmenter = augmenter
        self.seed = seed
        # classification in arcface format
        if shuffle:
            self.shuffle()

    def shuffle(self):
        tmp_list = list(zip(self.images, self.labels))
        random.shuffle(tmp_list)
        self.images, self.labels = zip(*tmp_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        caption = self.captions[index]
        caption = clip.tokenize(caption)[0]
        image = np.array(Image.open(image))

        # Perform augmentations
        if self.augmenter is not None:
            if self.seed is None:
                self.augmenter.reseed(np.random.randint(0, MASK, dtype=np.uint32))
            image = self.augmenter.augment(image=image)

        # Convert data to the torch format
        image = ToTensor()(image.astype(np.uint8))
        image = Normalize(mean=self.mean, std=self.std)(image)

        return image, label, caption


def get_labels(images_list: list) -> list:
    """
    Build labels for images based upon folder name
    male: 0
    female: 1
    """
    return [1 if 'female' in x else 0 for x in images_list]


def get_captions(images_list: list) -> list:
    """
    Build captions for images based upon folder name
    male: 0
    female: 1
    """
    return ['a photo of a woman' if 'female' in x else 'a photo of a man' for x in images_list]


def dataloader(p_to_images, batch_size, augs=None, num_workers=0, drop_last=False, shuffle=False):
    dataset = CLIPItemDataset(p_to_images=p_to_images,
                              augmenter=augs,
                              shuffle=shuffle)
    dataset.path_batches = split_list_to_chunks(dataset.images, batch_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last)
