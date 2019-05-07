import os

from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from load_train_test_data import load_train_data
from paths import root_dir


# 超参数
test_split = 0.2   # 验证机划分比例
num_classes = 2
lr = 1e-4
epochs = 30
dropout_rate = 0.5
kernel_regularizer = regularizers.l1(1e-4)   # 正则化
batch_size = 64
use_data_aug = True  # 是否使用数据增强
use_cross_validation = False   # 是否使用交叉验证
k_fold = 5  # k折交叉验证的k


def build_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    img_input = base_model.output

    outputs = GlobalAveragePooling2D(name='avg_pool_my')(img_input)

    if dropout_rate > 0.:
        outputs = Dropout(rate=dropout_rate)(outputs)

    outputs = Dense(256, activation='elu', name='fc1',
                    kernel_regularizer=kernel_regularizer)(outputs)
    outputs = Dropout(rate=dropout_rate)(outputs)
    outputs = Dense(128, activation='elu', name='fc2',
                    kernel_regularizer=kernel_regularizer)(outputs)
    outputs = Dropout(rate=dropout_rate)(outputs)
    outputs = Dense(num_classes, activation='softmax', name='predictions',
                    kernel_regularizer=kernel_regularizer)(outputs)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.summary()

    model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy,
                  metrics=[categorical_accuracy, ])

    return model


def train_model(model, X_train, y_train, X_valid, y_valid):
    # 模型保存路径
    model_path = os.path.join(root_dir, 'model_data', 'model_no_cross.h5')

    # 定义回调函数
    callbacks = [
        # 当标准评估停止提升时，降低学习速率
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.25,
                          patience=2,
                          verbose=1,
                          mode='auto',
                          min_lr=1e-7),
        # 在每个训练期之后保存模型，最后保存的是最佳模型
        ModelCheckpoint(model_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=True),
        # tensorboard 可视化
        TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=False,
                    write_grads=True,
                    write_images=True,
                    update_freq='epoch')
    ]

    if use_data_aug:
        datagen = ImageDataGenerator(rotation_range=180,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 #featurewise_center=True,  # 均值为0
                                 #featurewise_std_normalization=True  # 标准化
                                 )

        model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size * 2,
                        epochs=epochs,
                        initial_epoch=0,
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks)
    else:
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_valid, y_valid),
                  callbacks=callbacks)


def set_gpu():
    # 指定使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    ## keras 默认占满gpu所有内存，所以要手动设定内存使用情况
    config = tf.ConfigProto()
    '''
    # keras 设置gpu内存使用比例
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    '''
    # keras 设置gpu内存按需分配
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


if __name__ == '__main__':
    # 指定GPU
    set_gpu()

    # 构建模型
    model = build_model()

    if use_cross_validation:
        data = load_train_data(use_cross_validation=use_cross_validation, k_fold=k_fold)
        for i in range(k_fold):
            # 加载数据
            X_train, X_valid, y_train, y_valid = data[i]

            # 训练模型
            train_model(model, X_train, y_train, X_valid, y_valid)
    else:
        # 加载数据
        X_train, X_valid, y_train, y_valid = load_train_data(test_split=test_split)

        # 训练模型
        train_model(model, X_train, y_train, X_valid, y_valid)


