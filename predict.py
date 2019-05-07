import os

import cv2 as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

from paths import root_dir

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

clsss_name = {0: "AK", 1: "SK"}

if __name__ == '__main__':
    # 加载模型
    model_path = os.path.join(root_dir, 'model_data', 'model_no_cross.h5')
    model = load_model(model_path)

    for AK_or_SK_dir in os.listdir(os.path.join(root_dir, "images")):
        for fname in os.listdir(os.path.join(root_dir, "images", AK_or_SK_dir)):
            # 读取图片
            img_path = os.path.join(root_dir, "images", AK_or_SK_dir, fname)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)

            # 扩充维度
            img = np.expand_dims(img, axis=0)

            # 预测
            pred = model.predict(img)

            # 打印图片类别
            y_pred = np.argmax(pred, axis=1)
            img_name = clsss_name[y_pred[0]]
            print(fname, "的预测概率是：")
            print(pred, " ==> ", img_name)
