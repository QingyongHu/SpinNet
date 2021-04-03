from __future__ import division
from __future__ import print_function

from collections import defaultdict
from pathlib import Path
import cv2
import json
import numpy as np
import os
import os.path as osp
import re
import shutil


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------- #
# Common IO
# ---------------------------------------------------------------------------- #
def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, sort=True):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter))]
    if sort:
        return sorted_alphanum(file_list)
    else:
        return file_list


def list_folders(folder_path, name_filter=None, sort=True):
    folders = list()
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            folder_name = subfolder.name
            if name_filter is not None:
                if name_filter in folder_name:
                    folders.append(folder_name)
            else:
                folders.append(folder_name)
    if sort:
        return sorted_alphanum(folders)
    else:
        return folders


def read_lines(file_path):
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_json(filepath):
    with open(filepath, 'r') as fh:
        ret = json.load(fh)
    return ret


# ---------------------------------------------------------------------------- #
# Image IO
# ---------------------------------------------------------------------------- #
def read_color_image(file_path):
    """
    Args:
        file_path (str):

    Returns:
        np.array: RGB.
    """
    img = cv2.imread(file_path)
    return img[..., ::-1]


def read_gray_image(file_path):
    """Load a gray image

    Args:
        file_path (str):

    Returns:
        np.array: np.uint8, max 255.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img


def read_16bit_image(file_path):
    """Load a 16bit image

    Args:
        file_path (str):

    Returns:
        np.array: np.uint16, max 65535.
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return img


def write_color_image(file_path, image):
    """
    Args:
        file_path (str):
        image (np.array): in RGB.

    Returns:
        str:
    """
    cv2.imwrite(file_path, image[..., ::-1])
    return file_path


def write_gray_image(file_path, image):
    """
    Args:
        file_path (str):
        image (np.array):

    Returns:
        str:
    """
    cv2.imwrite(file_path, image)
    return file_path


def write_image(file_path, image):
    """
    Args:
        file_path (str):
        image (np.array):

    Returns:
        str:
    """
    if image.ndim == 2:
        return write_gray_image(file_path, image)
    elif image.ndim == 3:
        return write_color_image(file_path, image)
    else:
        raise RuntimeError('Image dimensions are not correct!')
