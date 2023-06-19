import numpy as np
import os.path

from PIL import Image, ImageDraw, ImageOps


def open_and_add_text_to_img(p, title, description,
                             pad=60,
                             border_color=(255,255,255),
                             text_color=(0,0,0)):

    img = Image.open(p)
    img = ImageOps.expand(img, border=pad, fill=border_color)
    if img.width >= 200:
        img = img.crop((pad,0,img.width - pad, img.height))

    draw = ImageDraw.Draw(img)

    _,_,w,h = draw.textbbox((0,0), title)
    draw.text(((img.width - w)//2, (pad-h)//2), title, text_color, align='center')

    label, name = p.split(os.sep)[-2:]
    image_name  = os.path.join(label, name)
    txt = description + f'\n{image_name}'

    _,_,w,_ = draw.textbbox((0,0), txt)
    draw.text(((img.width - w)//2, img.height - pad), txt, text_color, align='center')

    return img


def combine_images(test_image, knn_images, centroid_images, rows=3, columns=3, space=16):
    width_max = test_image.width
    height_max = test_image.height
    background_width = width_max * columns + (space * columns) - space
    background_height = height_max * rows + (space * rows) - space
    background = Image.new('RGB', (background_width, background_height), (255, 255, 255, 255))

    def put_test_img():
        x = width_max + space
        y = 0
        put_img(test_image, background, x, y)

    def put_knn_images():
        y = (height_max + space) * 2
        for idx, img in enumerate(knn_images):
            x = (width_max + space) * idx
            put_img(img, background, x, y)

    def put_centroid_images():
        y = (height_max + space)
        for idx, img in enumerate(centroid_images):
            x = (width_max + space) * idx
            if len(centroid_images) == 1:
                x = (width_max + space)
            put_img(img, background, x, y)

    def put_img(img, background, x, y):
        x_offset = int((width_max - img.width) / 2)
        y_offset = int((height_max - img.height) / 2)
        background.paste(img, (x + x_offset, y + y_offset))

    put_test_img()
    put_centroid_images()
    put_knn_images()

    return background


def center_print(msg, length=100) -> None:
    """Print center line

    Args:
        msg (_type_): _description_
        length (int, optional): _description_. Defaults to 100.
    """
    print(length * '=' + '\n' + f'{msg}'.center(length) +'\n' + length * '=')
