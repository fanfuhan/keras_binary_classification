import os
from keras.models import load_model
from paths import root_dir
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
from load_train_test_data import load_test_data

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


if __name__ == '__main__':
    # 加载模型
    model_path = os.path.join(root_dir, 'model_data', 'model_no_cross.h5')
    model = load_model(model_path)

    # 评估数据
    X_test, y_test = load_test_data()

    # y预测
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(y_test)
    print(y_pred)

    # 准确率，精确率，召回率，F1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("accuracy_score = %.2f" % accuracy)
    print("precision_score = %.2f" % precision)
    print("recall_score = %.2f" % recall)
    print("f1_score = %.2f" % f1)
