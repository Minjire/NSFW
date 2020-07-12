# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
import time


# %% load folders into arrays and return array
def loadImages(path):
    # Put files into lists and return them as one list
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg')])
    return image_files


# %% functions to display images
# display one image
def displayOne(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


# display 2 images
def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# %% Process images; resizing,
def processing(data):
    # random integer for indexing
    ind = random.randint(0, (len(data) - 1))
    # loading image in color, BGR mode
    img = [cv2.imread(i, cv2.IMREAD_COLOR) for i in data]

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

        if i % 100 == 0: print(f"Resizing and Blur: {i}")

    # print(f"Last index: {i}")
    # Checking the size
    print("RESIZED", res_img[ind].shape)

    # Visualizing one of the images in the array
    original = res_img[ind]
    # displayOne(original)

    image = no_noise[ind]
    # display(original, image, 'Original', 'Blurred')

    seg_image = []
    seg_bac_img = []
    final_proc_img = []

    for i in range(len(no_noise)):
        # segment the image, separating background from foreground objects
        '''
        # skip images already in grayscale
        if len(no_noise[i].shape) == 2:
            gray = no_noise[i]
        else:'''
        gray = cv2.cvtColor(no_noise[i], cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        seg_image.append(thresh)

        # further noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(seg_image[i], cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        seg_bac_img.append(sure_bg)

        # separate different objects in the image with markers
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers += 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(no_noise[i], markers)
        no_noise[i][markers == -1] = [255, 0, 0]

        final_proc_img.append(markers)

        if i % 100 == 0: print(f"Segmentation: {i}")

    # display(original, seg_image[ind], 'Original', 'Segmented')

    print("ArrayShape:", np.array(final_proc_img).shape)

    # Displaying segmented back ground
    # display(original, seg_bac_img[ind], 'Original', 'Segmented Background')

    # Displaying markers on the specific image
    display(no_noise[ind], final_proc_img[ind], 'Original', 'Marked')

    return final_proc_img


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

    # pass images for pre-processing
    '''
    proc_train_nsfw = processing(train_nsfw_images)
    proc_train_sfw = processing(train_sfw_images)
    proc_test_nsfw = processing(test_nsfw_images)
    proc_test_sfw = processing(test_sfw_images)
    proc_car_images = processing(car_images)'''

    t1 = time.time()
    proc_train_sfw = processing(train_nsfw_images[:4500])
    t2 = time.time()
    # np.save('train_sfw_arr.npy', proc_train_sfw)
    t3 = time.time()
    print(f"Image Processing Time: {t2 - t1} seconds.")
    print(f"Saving Time: {t3 - t2} seconds.")
    # array_reloaded = np.load('train_sfw_arr.npy', allow_pickle=True)
    # print(f"Array Shape: {array_reloaded.shape}")
    # print(array_reloaded)
    # print(array_reloaded.shape)
    # displayOne(array_reloaded[0])
    # print(test_sfw_images[0])


# %%
main()

# %%
'''
data = np.array([np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([3, 3, 4, 5, 2, 5]), np.array([7, 5, 4, 90])])
# save to npy file
print(data.shape)
np.save('fnumpy.npy', data)
np.savez_compressed('data.npz', data)
# %%
array_reloaded = np.load('fnumpy.npy', allow_pickle=True)
# extract the first array

print(array_reloaded)
dict_data = np.load('data.npz', allow_pickle=True)
# extract the first array
data = dict_data['arr_0']
# print the array
print(data)'''
