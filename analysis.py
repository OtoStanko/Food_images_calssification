import os
import numpy as np
from PIL import Image
from colorthief import Colorthief



WORKDIR = ""
DATA = ""
mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}
"""
        Analysis functions
"""


def get_image_depth():
    set_of_depths = {}
    with open(WORKDIR + "\\logs.txt", "w") as logs:
        for root, _, files in os.walk(DATA):
            for file in files:
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                set_of_depths[mode_to_bpp[image.mode]] = set_of_depths.get(mode_to_bpp[image.mode], 0) + 1
    return set_of_depths


def get_image_size():
    set_of_sizes = set()
    num_of_files = 0
    for root, _, files in os.walk(DATA):
        num_of_files += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            set_of_sizes.add(image.size)
    return set_of_sizes, num_of_files


def image_count_per_species():
    names = {}
    for i in ["test", "train", "validation"]:
        for root, directories, files in os.walk(
                DATA + i):
            for directory in directories:
                if names.get(directory) is not None:
                    names[directory] += [(len(os.listdir(os.path.join(root, directory))))]
                else:
                    names[directory] = [len(os.listdir(os.path.join(root, directory)))]
    return names


def dominant_color_per_picture_andrej():
    colors = []
    for root, directories, files in os.walk(DATA +  "\\dataset\\validation"):
        dominant = [0, 0, 0]
        count = 0
        pole = []
        for file in files:
            count += 1
            file_path = os.path.join(root, file)
            color = list(ColorThief(file_path).get_color(quality=1))
            dominant[0] += color[0]
            dominant[1] += color[1]
            dominant[2] += color[2]
            pole.append([color[0], color[1], color[2]])
        colors.append(dominant)
    return colors


def dominant_color_per_picture():
    colors = []
    for root, directories, files in os.walk(DATA +  "\\dataset\\validation"):
        dominant = [0, 0, 0]
        count = 0
        for file in files:
            count += 1
            file_path = os.path.join(root, file)
            color = list(ColorThief(file_path).get_color(quality=1))
            dominant[0] += color[0]
            dominant[1] += color[1]
            dominant[2] += color[2]
        if count != 0:
            dominant[0] /= count
            dominant[1] /= count
            dominant[2] /= count
        colors.append(dominant)
    return colors


def dominant_color_per_pixel():
    colors = {}
    for root, directories, files in os.walk(
            DATA + "\\test"):
        dominant = [0, 0, 0]
        count = 0
        for file in files:
            file_path = os.path.join(root, file)
            color = list(ColorThief(file_path).get_color(quality=1))
            image_op = Image.open(file_path)
            num_pixels = (image_op.width * image_op.height)
            count += num_pixels
            dominant[0] += color[0] * num_pixels
            dominant[1] += color[1] * num_pixels
            dominant[2] += color[2] * num_pixels
        if count != 0:
            dominant[0] /= count
            dominant[1] /= count
            dominant[2] /= count
        colors[root.split('\\')[-1]] = dominant
        print(root)
    return colors
