import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import transform
from tqdm import tqdm

from paths import root_dir, mkdir_if_not_exist
from sklearn.utils import shuffle

origin_data_dir = os.path.join(root_dir, 'origin_data')
data_dir = os.path.join(root_dir, 'data')
mkdir_if_not_exist(dir_list=[data_dir])


def process_datasets():
    images = []
    labels = []

    for AK_or_SK_dir in tqdm(os.listdir(origin_data_dir)):
        # AK ==> [1, 0]   SK ==> [0, 1]
        if 'AK' in AK_or_SK_dir:
            label = [1, 0]
        elif 'SK' in AK_or_SK_dir:
            label = [0, 1]
        else:
            print("AK_or_SK_dir error")

        for person_dir in tqdm(os.listdir(os.path.join(origin_data_dir, AK_or_SK_dir))):
            for fname in os.listdir(os.path.join(origin_data_dir, AK_or_SK_dir, person_dir)):
                img_fname = os.path.join(origin_data_dir, AK_or_SK_dir,
                                         person_dir, fname)
                image = io.imread(img_fname)
                image = transform.resize(image, (224, 224),
                                         order=1, mode='constant',
                                         cval=0, clip=True,
                                         preserve_range=True,
                                         anti_aliasing=True)
                image = image.astype(np.uint8)

                images.append(image)
                labels.append(label)

    images = np.stack(images).astype(np.uint8)
    labels = np.stack(labels, axis=0)

    return images, labels


def load_datasets():
    images_npy_filename = os.path.join(data_dir, 'images_data.npy')
    labels_npy_filename = os.path.join(data_dir, 'labels.npy')

    if os.path.exists(images_npy_filename) and os.path.exists(labels_npy_filename):
        images = np.load(images_npy_filename)
        labels = np.load(labels_npy_filename)
    else:
        images, labels = process_datasets()
        # 将数据打乱后保存
        images, labels = shuffle(images, labels)
        np.save(images_npy_filename, images)
        np.save(labels_npy_filename, labels)

    return images, labels


if __name__ == '__main__':
    X, y = load_datasets()
    print(X.shape,y.shape)
    # plt.imshow(X[5])
    # plt.show()
    y = np.argmax(y, axis=1)
    print(y[:20])
    count_SK = np.count_nonzero(y)
    print("SK图片数量：", count_SK)

