import os
#import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from colorthief import ColorThief
from matplotlib.image import imread

from keras.preprocessing.image import ImageDataGenerator

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
from math import sqrt, inf
from random import random


PATH = r"D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3"
IMAGE_SIZE = 150
BATCH_SIZE = 128


mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}

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
for key, value in categories.items():
    categories[key] = value-1

rev_cat = {}
for veg, num in categories.items():
    rev_cat[num] = veg


def get_format():
    set_of_file_ends = set()
    for root, _, files in os.walk("C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Food_Images_Data"):
        for file in files:
            file_end = file[file.index("."):]
            set_of_file_ends.add(file_end)

    return set_of_file_ends


"""
        Analysis functions
"""


def get_image_depth():
    set_of_depths = {}
    with open(r"C:\Users\vitko\Desktop\MUNI\IB031-Úvod_do_strojového_učení\Projekt\Food_Images\logs.txt", "w") as logs:
        for root, _, files in os.walk(
                "C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Projekt\\Food_Images_Data"):
            for file in files:
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                set_of_depths[mode_to_bpp[image.mode]] = set_of_depths.get(mode_to_bpp[image.mode], 0) + 1
    return set_of_depths


def get_image_size():
    set_of_sizes = set()
    num_of_files = 0
    for root, _, files in os.walk("C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Food_Images_Data"):
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
                "C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Proejtk\\Food_Images_Data\\" + i):
            for directory in directories:
                if names.get(directory) is not None:
                    names[directory] += [(len(os.listdir(os.path.join(root, directory))))]
                else:
                    names[directory] = [len(os.listdir(os.path.join(root, directory)))]
    return names


def dominant_color_per_picture_andrej():
    colors = []
    for root, directories, files in os.walk(PATH +  "\dataset\\validation"):
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
    for root, directories, files in os.walk(PATH +  "\dataset\\validation"):
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
            "C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Projekt\\Food_Images_Data\\test"):
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
    image_size = 150
    X = []
    y = []

    for root, directories, files in os.walk(PATH + "\\" + "dataset\\" + subfolder):
        for file in files:
            file_path = os.path.join(root, file)
            with Image.open(file_path) as image:
                path = "\\" + root.split("\\")[-1] + "_" + file.split(".")[0] + ".jpg"
                image = image.resize((image_size, image_size))
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
    X = X.reshape(-1, image_size, image_size, 3)

    with open('X_pickle_' + subfolder, 'wb') as pickle_out:
        pickle.dump(X, pickle_out)
    with open('y_pickle_' + subfolder, 'wb') as pickle_out:
        pickle.dump(y, pickle_out)
    

"""
        Models
"""

def get_data():
    X = pickle.load(open('X_pickle_train', 'rb'))
    y = pickle.load(open('y_pickle_train', 'rb'))

    X_val = pickle.load(open('X.pickle_validation', 'rb'))
    y_val = pickle.load(open('y.pickle_validation', 'rb'))

    X_test = pickle.load(open('X_pickle_test', 'rb'))
    y_test = pickle.load(open('y_pickle_test', 'rb'))
    return X, y, X_val, y_val, X_test, y_test


def non_CNN_models(model="knn"):
    X, y, X_val, y_val, X_test, y_test = get_data()
    samples, nx, ny, nrgb = X.shape
    X = X.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))

    if model == "knn":
        classifier = KNeighborsClassifier(n_neighbors=7)
    elif model == "rf":
        classifier = RandomForestClassifier()
    elif model == "dt":
        classifier = DecisionTreeClassifier()
    
    classifier.fit(X, y)
    pred = classifier.predict(X_test)
    print("PREDICTIONS:")
    print(pred)
    print("ACCURACY SCORE:")
    print(accuracy_score(pred, y_test))
    print("REPORT:")
    print(classification_report(pred,y_test))
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(pred,y_test)
    print_cnf(cm)


def print_row(row):
    line = ""
    for elem in row:
        line = line + "{:2d} ".format(int(elem))
    print(line)


def print_cnf(cm):
    for row in cm:
        print_row(row)
    

    
def model_CNN(X, y, X_val, y_val, epochs=6):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(36))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, batch_size=128, validation_data=(X_val, y_val), epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=2,
                                                                    restore_best_weights=True)])
    return model, history


def predict_CNN_results(epochs):
    X, y, X_val, y_val, X_test, y_test = get_data()

    m, h = model_CNN(X, y, X_val, y_val, epochs)

    pred = m.predict(X_test)
    actual = []
    counter = 0
    print(pred.shape)
    best_pred = []
    for i in range(len(pred)):
      best_pred.append(np.argmax(pred[i]))
    print(y_test.shape)
    cm = confusion_matrix(best_pred, y_test)
    for i in range(len(pred)):
        j = np.argmax(pred[i])
        actual.append(j)
        if (abs(y_test[i] - j) > 1):
            print("Predicted: {:13s} ({:2d}) | actually: {:13s} ({:2d})".format(rev_cat[j], j, rev_cat[y_test[i]], y_test[i]))
            counter += 1
    print(len(actual), len(best_pred))
    print(actual[0], best_pred[0])
    print(f1_score(actual, best_pred, labels=actual))
    print("missed {:3d} out of {:3d}".format(counter, len(pred)))
    print("Accuracy on testing data: {:3.3f} %".format((1-counter/len(pred))*100))
    process_history(h, cm, y_test)


def naive_color_baseline(average_colors, X_test, y_test):
    #X, y, X_val, y_val, X_test, y_test = get_data()
    print(len(X_test), len(y_test), len(average_colors))
    correct_guesses = 0
    number_of_guesses = 0
    average_color = [[160.2, 82.9, 73.0], [207.0, 180.33333333333334, 86.44444444444444], [123.1, 102.7, 80.8], [164.33333333333334, 140.22222222222223, 93.55555555555556], [119.7, 148.4, 99.5], [165.0, 139.7, 71.1], [218.44444444444446, 126.66666666666667, 62.55555555555556], [172.7, 163.3, 131.2], [182.55555555555554, 146.44444444444446, 124.88888888888889], [185.2, 172.6, 81.4], [134.1, 144.9, 91.1], [76.6, 72.3, 61.7], [156.0, 142.3, 128.4], [186.9, 143.5, 90.4], [111.33333333333333, 93.77777777777777, 82.44444444444444], [86.0, 106.88888888888889, 57.666666666666664], [151.1, 155.6, 77.3], [200.5, 180.8, 86.0], [113.11111111111111, 161.44444444444446, 53.111111111111114], [219.5, 166.4, 81.7], [191.2, 137.5, 114.8], [222.33333333333334, 164.11111111111111, 53.22222222222222], [143.9, 73.7, 34.3], [160.6, 158.1, 74.5], [103.9, 142.4, 56.9], [136.7, 131.4, 93.7], [147.6, 50.8, 52.3], [188.6, 147.2, 94.7], [184.77777777777777, 186.88888888888889, 159.66666666666666], [181.6, 145.7, 97.9], [135.1, 158.8, 102.1], [198.4, 177.1, 70.1], [127.0, 97.4, 78.2], [149.6, 103.3, 65.6], [148.2, 143.1, 121.2], [146.5, 112.9, 86.7]]

    for i in range(len(X_test)):
        correct = y_test[i]
        #colors, count = np.unique(X_test[i].reshape(-1,X_test[i].shape[-1]), axis=0, return_counts=True)
        #avg_color = colors[count.argmax()]
        avg_color_per_row = np.average(X_test[i], axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)

        min_dist = inf
        category = -1
        for j in range(len(average_colors)):
            dist = sqrt((avg_color[0] - (average_colors[j][0]/255)) ** 2 +
                        (avg_color[1] - (average_colors[j][1]/255)) ** 2 +
                        (avg_color[2] - (average_colors[j][2]/255)) ** 2)
            if dist < min_dist:
                min_dist = dist
                category = j
        if correct == category:
            correct_guesses += 1
            print(correct)
        number_of_guesses += 1
    print(correct_guesses, number_of_guesses)
    return correct_guesses / number_of_guesses



def model_DT(X, y, X_val, y_val, X_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(36))
    model.add(Activation('softmax'))
    features = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)

    model_f = features.predict(X)
    model2 = features.predict(X_test)

    dt = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth=11, min_samples_leaf=8)
    dt.fit(model_f, y)
    return dt.score(model2, y_test)


def model_lin_svm():
    X, y, X_val, y_val, X_test, y_test = get_data()
    X_new = []
    X_test_new = []
    X_val_new = []

    print("Data loaded")
    for i in range(len(X)):
        X_new.append(X[i].flatten())
    for i in range(len(X_test)):
        X_test_new.append(X_test[i].flatten())
    for i in range(len(X_val)):
        X_val_new.append(X_val[i].flatten())

    X_new = np.array(X_new)
    X = X_new
    X_test_new = np.array(X_test_new)
    X_test = X_test_new
    X_val_new = np.array(X_val_new)
    X_val = X_val_new
    print(X_val.shape, y_val.shape)

    param_grid = {'kernel': ['rbf', 'poly']}
    print("param_grid created")
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    print("model created")
    model.fit(X, y)
    print(model.best_params_)

    y_pred = model.predict(X_test)
    print("The predicted Data is :", y_pred)
    print("The actual data is:", np.array(y_test))
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")






"""
        Results
"""

def predict_baseline(average_colors=None):
    X, y, X_val, y_val, X_test, y_test = get_data()
    if average_colors is None:
        average_colors = dominant_color_per_picture()
    print(average_colors)
    acc = naive_color_baseline(average_colors, X, y)
    print(acc)


def predict_CNN_results(epochs, m=None):
    if m is None:
        m = model_CNN(epochs)
        #joblib.dump(m, "model_CNN")
        #pickle.dump(m, open("model_CNN", "wb"))
        with open('./model_CNN.pkl', 'wb') as f:
            pickle.dump(m, f)
    else:
        print("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl")
        print(os.path.isfile("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl"))
        with open("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl", "rb") as f:
            m = pickle.load(f)
        #m = pickle.load(open('model_CNN', "rb"))
        #m = joblib.load("model_CNN")
    _, _, _, _, X_test, y_test = get_data()

    pred = m.predict(X_test)
    counter = 0
    for i in range(len(pred)):
        j = np.argmax(pred[i])
        if (abs(y_test[i] - j) > 1):
            print("Predicted: {:13s} ({:2d}) | actually: {:13s} ({:2d})".format(rev_cat[j], j, rev_cat[y_test[i]], y_test[i]))
            counter += 1
    print("missed {:3d} out of {:3d}".format(counter, len(pred)))
    print("Accuracy on testing data: {:3.3f}%".format((1-counter/len(pred))*100))

                                       
#change_size()
#change_bit_depth()
def model(epochs=4):
    X = pickle.load(open('X.pickle', 'rb'))
    y = pickle.load(open('y.pickle', 'rb'))
    print(len(X), len(y))
    
    X = X / 255.0

    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, validation_split=0.1, epochs=epochs)


def model2():
    X = pickle.load(open('X.pickle', 'rb'))
    y = pickle.load(open('y.pickle', 'rb'))

    batch_size = 32
    img_height = 180
    img_width = 180

    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(36)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    print(y[0])

    epochs=10
    history = model.fit(x=X, y=y, validation_split=0.1, batch_size=256, epochs=3)


def KNN_with_grid():
    X, y, X_val, y_val, X_test, y_test = get_data()

    samples, nx, ny, nrgb = X.shape
    X = X.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))
    
    knn_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('knn', KNeighborsClassifier())])

    params = [{'knn__n_neighbors': [3, 5, 7, 9, 11, 13],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [5, 6, 15, 18, 36]}]
    gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
    gs_knn.fit(X, y)
    print(gs_knn.best_params_)
    # find best model score
    print(gs_knn.score(X_test, y_test))
    return gs_knn


def RF_with_grid():
    X, y, X_val, y_val, X_test, y_test = get_data()
    print(X.shape)
    samples, nx, ny, nrgb = X_val.shape
    X_val = X_val.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))
    
    rf = RandomForestClassifier()

    params = [{'n_estimators': [75, 100, 125],
         'min_samples_split': [2, 5, 7],
         'min_samples_leaf': [1, 3, 5]}]
    gs_rf = GridSearchCV(rf,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
    gs_rf.fit(X_val, y_val)
    print(gs_rf.best_params_)
    # find best model score
    #print(gs_rf.score(X, y))
    return gs_rf

a = RF_with_grid()

#a = RF_with_grid()


#a = KNN_with_grid()
=======
import os
#import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from colorthief import ColorThief
from matplotlib.image import imread

from keras.preprocessing.image import ImageDataGenerator

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
from math import sqrt, inf
from random import random


PATH = r"D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3"
IMAGE_SIZE = 150
BATCH_SIZE = 128


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
for key, value in categories.items():
    categories[key] = value-1

rev_cat = {}
for veg, num in categories.items():
    rev_cat[num] = veg


def get_format():
    set_of_file_ends = set()
    for root, _, files in os.walk("C:\\Users\\vitko\\Desktop\\MUNI\\IB031-Úvod_do_strojového_učení\\Food_Images_Data"):
        for file in files:
            file_end = file[file.index("."):]
            set_of_file_ends.add(file_end)

    return set_of_file_ends




"""
        Models
"""

def get_data():
    X = pickle.load(open('X_pickle_train', 'rb'))
    y = pickle.load(open('y_pickle_train', 'rb'))

    X_val = pickle.load(open('X.pickle_validation', 'rb'))
    y_val = pickle.load(open('y.pickle_validation', 'rb'))

    X_test = pickle.load(open('X_pickle_test', 'rb'))
    y_test = pickle.load(open('y_pickle_test', 'rb'))
    return X, y, X_val, y_val, X_test, y_test


def non_CNN_models(model="knn"):
    X, y, X_val, y_val, X_test, y_test = get_data()
    samples, nx, ny, nrgb = X.shape
    X = X.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))

    if model == "knn":
        classifier = KNeighborsClassifier(n_neighbors=7)
    elif model == "rf":
        classifier = RandomForestClassifier()
    elif model == "dt":
        classifier = DecisionTreeClassifier()
    
    classifier.fit(X, y)
    pred = classifier.predict(X_test)
    print("PREDICTIONS:")
    print(pred)
    print("ACCURACY SCORE:")
    print(accuracy_score(pred, y_test))
    print("REPORT:")
    print(classification_report(pred,y_test))
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(pred,y_test)
    print_cnf(cm)


def print_row(row):
    line = ""
    for elem in row:
        line = line + "{:2d} ".format(int(elem))
    print(line)


def print_cnf(cm):
    for row in cm:
        print_row(row)
    

    
def model_CNN(X, y, X_val, y_val, epochs=6):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(36))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, batch_size=128, validation_data=(X_val, y_val), epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=2,
                                                                    restore_best_weights=True)])
    return model, history


def predict_CNN_results(epochs):
    X, y, X_val, y_val, X_test, y_test = get_data()

    m, h = model_CNN(X, y, X_val, y_val, epochs)

    pred = m.predict(X_test)
    actual = []
    counter = 0
    print(pred.shape)
    best_pred = []
    for i in range(len(pred)):
      best_pred.append(np.argmax(pred[i]))
    print(y_test.shape)
    cm = confusion_matrix(best_pred, y_test)
    for i in range(len(pred)):
        j = np.argmax(pred[i])
        actual.append(j)
        if (abs(y_test[i] - j) > 1):
            print("Predicted: {:13s} ({:2d}) | actually: {:13s} ({:2d})".format(rev_cat[j], j, rev_cat[y_test[i]], y_test[i]))
            counter += 1
    print(len(actual), len(best_pred))
    print(actual[0], best_pred[0])
    print(f1_score(actual, best_pred, labels=actual))
    print("missed {:3d} out of {:3d}".format(counter, len(pred)))
    print("Accuracy on testing data: {:3.3f} %".format((1-counter/len(pred))*100))
    process_history(h, cm, y_test)


def naive_color_baseline(average_colors, X_test, y_test):
    #X, y, X_val, y_val, X_test, y_test = get_data()
    print(len(X_test), len(y_test), len(average_colors))
    correct_guesses = 0
    number_of_guesses = 0
    average_color = [[160.2, 82.9, 73.0], [207.0, 180.33333333333334, 86.44444444444444], [123.1, 102.7, 80.8], [164.33333333333334, 140.22222222222223, 93.55555555555556], [119.7, 148.4, 99.5], [165.0, 139.7, 71.1], [218.44444444444446, 126.66666666666667, 62.55555555555556], [172.7, 163.3, 131.2], [182.55555555555554, 146.44444444444446, 124.88888888888889], [185.2, 172.6, 81.4], [134.1, 144.9, 91.1], [76.6, 72.3, 61.7], [156.0, 142.3, 128.4], [186.9, 143.5, 90.4], [111.33333333333333, 93.77777777777777, 82.44444444444444], [86.0, 106.88888888888889, 57.666666666666664], [151.1, 155.6, 77.3], [200.5, 180.8, 86.0], [113.11111111111111, 161.44444444444446, 53.111111111111114], [219.5, 166.4, 81.7], [191.2, 137.5, 114.8], [222.33333333333334, 164.11111111111111, 53.22222222222222], [143.9, 73.7, 34.3], [160.6, 158.1, 74.5], [103.9, 142.4, 56.9], [136.7, 131.4, 93.7], [147.6, 50.8, 52.3], [188.6, 147.2, 94.7], [184.77777777777777, 186.88888888888889, 159.66666666666666], [181.6, 145.7, 97.9], [135.1, 158.8, 102.1], [198.4, 177.1, 70.1], [127.0, 97.4, 78.2], [149.6, 103.3, 65.6], [148.2, 143.1, 121.2], [146.5, 112.9, 86.7]]

    for i in range(len(X_test)):
        correct = y_test[i]
        #colors, count = np.unique(X_test[i].reshape(-1,X_test[i].shape[-1]), axis=0, return_counts=True)
        #avg_color = colors[count.argmax()]
        avg_color_per_row = np.average(X_test[i], axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)

        min_dist = inf
        category = -1
        for j in range(len(average_colors)):
            dist = sqrt((avg_color[0] - (average_colors[j][0]/255)) ** 2 +
                        (avg_color[1] - (average_colors[j][1]/255)) ** 2 +
                        (avg_color[2] - (average_colors[j][2]/255)) ** 2)
            if dist < min_dist:
                min_dist = dist
                category = j
        if correct == category:
            correct_guesses += 1
            print(correct)
        number_of_guesses += 1
    print(correct_guesses, number_of_guesses)
    return correct_guesses / number_of_guesses



def model_DT(X, y, X_val, y_val, X_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(36))
    model.add(Activation('softmax'))
    features = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)

    model_f = features.predict(X)
    model2 = features.predict(X_test)

    dt = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth=11, min_samples_leaf=8)
    dt.fit(model_f, y)
    return dt.score(model2, y_test)


def model_lin_svm():
    X, y, X_val, y_val, X_test, y_test = get_data()
    X_new = []
    X_test_new = []
    X_val_new = []

    print("Data loaded")
    for i in range(len(X)):
        X_new.append(X[i].flatten())
    for i in range(len(X_test)):
        X_test_new.append(X_test[i].flatten())
    for i in range(len(X_val)):
        X_val_new.append(X_val[i].flatten())

    X_new = np.array(X_new)
    X = X_new
    X_test_new = np.array(X_test_new)
    X_test = X_test_new
    X_val_new = np.array(X_val_new)
    X_val = X_val_new
    print(X_val.shape, y_val.shape)

    param_grid = {'kernel': ['rbf', 'poly']}
    print("param_grid created")
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    print("model created")
    model.fit(X, y)
    print(model.best_params_)

    y_pred = model.predict(X_test)
    print("The predicted Data is :", y_pred)
    print("The actual data is:", np.array(y_test))
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")






"""
        Results
"""

def predict_baseline(average_colors=None):
    X, y, X_val, y_val, X_test, y_test = get_data()
    if average_colors is None:
        average_colors = dominant_color_per_picture()
    print(average_colors)
    acc = naive_color_baseline(average_colors, X, y)
    print(acc)


def predict_CNN_results(epochs, m=None):
    if m is None:
        m = model_CNN(epochs)
        #joblib.dump(m, "model_CNN")
        #pickle.dump(m, open("model_CNN", "wb"))
        with open('./model_CNN.pkl', 'wb') as f:
            pickle.dump(m, f)
    else:
        print("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl")
        print(os.path.isfile("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl"))
        with open("D:\MUNI\FI\semester_6\IB031_strojove_uceni\project-Food-Images-3\model_CNN.pkl", "rb") as f:
            m = pickle.load(f)
        #m = pickle.load(open('model_CNN', "rb"))
        #m = joblib.load("model_CNN")
    _, _, _, _, X_test, y_test = get_data()

    pred = m.predict(X_test)
    counter = 0
    for i in range(len(pred)):
        j = np.argmax(pred[i])
        if (abs(y_test[i] - j) > 1):
            print("Predicted: {:13s} ({:2d}) | actually: {:13s} ({:2d})".format(rev_cat[j], j, rev_cat[y_test[i]], y_test[i]))
            counter += 1
    print("missed {:3d} out of {:3d}".format(counter, len(pred)))
    print("Accuracy on testing data: {:3.3f}%".format((1-counter/len(pred))*100))

                                       
#change_size()
#change_bit_depth()
def model(epochs=4):
    X = pickle.load(open('X.pickle', 'rb'))
    y = pickle.load(open('y.pickle', 'rb'))
    print(len(X), len(y))
    
    X = X / 255.0

    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, validation_split=0.1, epochs=epochs)


def model2():
    X = pickle.load(open('X.pickle', 'rb'))
    y = pickle.load(open('y.pickle', 'rb'))

    batch_size = 32
    img_height = 180
    img_width = 180

    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(36)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    print(y[0])

    epochs=10
    history = model.fit(x=X, y=y, validation_split=0.1, batch_size=256, epochs=3)


def KNN_with_grid():
    X, y, X_val, y_val, X_test, y_test = get_data()

    samples, nx, ny, nrgb = X.shape
    X = X.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))
    
    knn_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('knn', KNeighborsClassifier())])

    params = [{'knn__n_neighbors': [3, 5, 7, 9, 11, 13],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [5, 6, 15, 18, 36]}]
    gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
    gs_knn.fit(X, y)
    print(gs_knn.best_params_)
    # find best model score
    print(gs_knn.score(X_test, y_test))
    return gs_knn


def RF_with_grid():
    X, y, X_val, y_val, X_test, y_test = get_data()
    print(X.shape)
    samples, nx, ny, nrgb = X_val.shape
    X_val = X_val.reshape((samples, nx*ny*nrgb))

    samples, nx, ny, nrgb = X_test.shape
    X_test = X_test.reshape((samples, nx*ny*nrgb))
    
    rf = RandomForestClassifier()

    params = [{'n_estimators': [75, 100, 125],
         'min_samples_split': [2, 5, 7],
         'min_samples_leaf': [1, 3, 5]}]
    gs_rf = GridSearchCV(rf,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
    gs_rf.fit(X_val, y_val)
    print(gs_rf.best_params_)
    # find best model score
    #print(gs_rf.score(X, y))
    return gs_rf

a = RF_with_grid()

#a = RF_with_grid()


#a = KNN_with_grid()
>>>>>>> 468c8df (project functions together)
