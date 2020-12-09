# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from net import create_model


def read_csv(path):

    csvs = os.listdir(path)
    data = []
    prediction = []
    for csv in csvs:
        prediction.append(int(csv.split('-')[4]))
        csv_path = os.path.join(path, csv)
        data_df = pd.read_csv(csv_path, header=None)
        data.append(data_df.values.reshape(-1,8192,1))
    print(prediction)

    dataset = tf.data.Dataset.from_tensor_slices(np.asarray(data))


    return dataset


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # model = create_model()
    model = load_model(filepath='./model/test6.h5')

    predict = model.predict(read_csv(path = './eva_dataset'))
    label=[]
    for i in range(len(predict)):
        if predict[i][0]>predict[i][1]:
            label.append(0)
            # print('类别为0')
        else:
            label.append(1)
            # print('类别为1')
    print(label)