import os
import numpy as np
from PIL import Image
from random import random




PATH = ""
IMAGE_SIZE = 150
categories = {"apple": 1, "banana": 2, "beetroot": 3,
              "bell pepper": 4, "cabbage": 5, "capsicum":6,
              "carrot": 7, "cauliflower": 8, "chilli pepper": 9,
              "corn": 10, "cucumber": 11, "eggplant": 12,
              "garlic": 13, "ginger": 14, "grapes": 15,
              "jalepeno": 16, "kiwi": 17, "lemon": 18,
              "lettuce": 19, "mango": 20, "onion": 21,
              "orange": 22, "paprika": 23, "pear": 24,
              "peas": 25, "pineapple": 26, "pomegranate": 27,
              "potato": 28, "raddish": 29, "soy beans": 30,
              "spinach": 31, "sweetcorn": 32, "sweetpotato": 33,
              "tomato": 34, "turnip": 35, "watermelon": 36}
"""
        Preprocessing
"""


def change_bit_depth():
    for root, directories, files in os.walk(
        PATH + "\\" + "unified_size"):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            with Image.open(file_path) as image:
                image = image / 255
                path = "\\" + root.split("\\")[-1] + "_" + file.split(".")[0]
                with open(PATH + "\\" + "unified_bit_depth" + path, "w") as img:
                    image.save(img)

def change_size():
    print(PATH + "\\" + "dataset\train")
    for root, directories, files in os.walk(PATH + "\\" + "dataset\\train"):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            with Image.open(file_path) as image:
                path = "\\" + root.split("\\")[-1] + "_" + file.split(".")[0] + ".jpg"
                image = image.resize((720, 720))
                with open(PATH + "\\" + "unified_size" + path, "w") as img:
                    np.set_printoptions(threshold=np.inf)
                    image = image.convert('RGB')
                    image.save(img)


def rotate_or_flip(image):
    if random() < 0.5:
        f = random()
        if f <= 0.3:
            return np.rot90(image, 1)
        elif f <= 0.6:
            return np.rot90(image, 2)
        return np.rot90(image, 3)
    if random() < 0.5:
        return np.flipud(image)
    return np.fliplr(image)

    
def change_size_and_pickle(subfolder="train"):
    X = []
    y = []

    for root, directories, files in os.walk(PATH + "\\" + "dataset\\" + subfolder):
        for file in files:
            file_path = os.path.join(root, file)
            with Image.open(file_path) as image:
                path = "\\" + root.split("\\")[-1] + "_" + file.split(".")[0] + ".jpg"
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                image = np.array(image.convert('RGB'))

                if random() < 0.5:
                    new_image = rotate_or_flip(image)
                    #np.set_printoptions(threshold=np.inf)
                    X.append(new_image)
                    y.append(categories[root.split("\\")[-1]]) 
                #np.set_printoptions(threshold=np.inf)
                X.append(image)
                y.append(categories[root.split("\\")[-1]])
                
    y = np.array(y)
    X = np.array(X)

    X = X / 255.0
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    with open('X_pickle_' + subfolder, 'wb') as pickle_out:
        pickle.dump(X, pickle_out)
    with open('y_pickle_' + subfolder, 'wb') as pickle_out:
        pickle.dump(y, pickle_out)
    
