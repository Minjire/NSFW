import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# load folders into arrays and return array
def loadImages(path):
    # Put files into lists and return them as one list
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return image_files


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


test_nsfw = "not-safe-for-work/test/nsfw"
test_sfw = "not-safe-for-work/test/sfw"

test_nsfw_images = loadImages(test_nsfw)
test_sfw_images = loadImages(test_sfw)

proc_test_nsfw = processing(test_nsfw_images)
y1labels = np.zeros(len(proc_test_nsfw), dtype=int)
proc_test_sfw = processing(test_sfw_images)
y2labels = np.ones(len(proc_test_sfw), dtype=int)

test_images = np.concatenate((proc_test_nsfw, proc_test_sfw))
test_labels = np.concatenate((y1labels, y2labels))

class_names = ['nsfw', 'sfw']
test_images = test_images / 255.0

# load model
model = keras.models.load_model("nsfw_model.h5")

# test for model accuracy after training
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
prediction = model.predict(test_images)

array = []
for i in range(len(prediction)):
    if np.argmax(prediction):
        array.append(np.argmax[i])
    else:
        array.append(np.argmax(prediction[i]))

array = np.array(array)

print(prediction[0])


results = confusion_matrix(test_labels, array)

print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(test_labels, array))
print('Report : ')
print(classification_report(test_labels, array))


# display the first 5 images and their predictions
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

