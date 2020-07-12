import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras


# %% load folders into arrays and return array
def loadImages(path):
    # Put files into lists and return them as one list
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return image_files


# %% Process images; resizing, remove noise
def processing(data):
    img = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in data if cv2.imread(i, cv2.IMREAD_GRAYSCALE) is not None]

    # setting dimension of resize
    width = height = 220
    dim = (width, height)
    res_img = []
    no_noise = []

    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

        # remove noise using Gaussian function
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        no_noise.append(blur)

        if i % 1000 == 0: print(f"Resizing and Blur: {i}")

    # Checking the size
    print("RESIZED", res_img[0].shape)

    return np.array(no_noise)


# %% main function
def main():
    # define path file
    train_nsfw = "not-safe-for-work/train/nsfw"
    train_sfw = "not-safe-for-work/train/sfw"
    test_nsfw = "not-safe-for-work/test/nsfw"
    test_sfw = "not-safe-for-work/test/sfw"

    # variables with images
    train_nsfw_images = loadImages(train_nsfw)
    train_sfw_images = loadImages(train_sfw)
    test_nsfw_images = loadImages(test_nsfw)
    test_sfw_images = loadImages(test_sfw)

    # split data to run on virtual server
    def passing(data):
        i = 0
        j = i + 5000
        array = processing(data[i:j])
        i = j
        j += 5000
        while i < len(data):
            arr = processing(data[i:j])
            array = np.concatenate((array, arr))
            i = j
            j += 5000
            if j > len(data): j = len(data)

        return array

    # pass images for pre-processing
    proc_train_nsfw = passing(train_nsfw_images)
    x1labels = np.zeros(len(proc_train_nsfw), dtype=int)
    proc_train_sfw = passing(train_sfw_images)
    x2labels = np.ones(len(proc_train_sfw), dtype=int)
    train_images = np.concatenate((proc_train_nsfw, proc_train_sfw))
    train_labels = np.concatenate((x1labels, x2labels))

    proc_test_nsfw = processing(test_nsfw_images)
    y1labels = np.zeros(len(proc_test_nsfw), dtype=int)
    proc_test_sfw = processing(test_sfw_images)
    y2labels = np.ones(len(proc_test_sfw), dtype=int)

    test_images = np.concatenate((proc_test_nsfw, proc_test_sfw))
    test_labels = np.concatenate((y1labels, y2labels))

    # model development
    class_names = ['nsfw', 'sfw']
    # normalize/shrink data down within a range, avoid large numbers
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # create model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(220, 220)),  # flatten data (input layer)
        keras.layers.Dense(128, activation="relu"),  # dense layer--fully connected layer (hidden layer)
        keras.layers.Dense(2, activation="softmax")  # softmax to add up to 1 (output layer)
    ])

    # Compiling the model is just picking the optimizer, loss function and metrics to keep track of
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train model, epochs defines iteration of seeing a single input
    model.fit(train_images, train_labels, batch_size=512, epochs=5, validation_split=0.15)

    # save model
    model.save("nsfw_model.h5")


# %%
main()
