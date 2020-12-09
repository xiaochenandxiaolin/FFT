# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import re


def csv_data(file_list, batch_size):
    i = 0
    while True:
        if i * batch_size >= len(file_list):
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i * batch_size:(i + 1) * batch_size]
            data = []
            labels = []
            data_list = []
            for b in range(len(file_list)):
                w = np.array(file_list[b], dtype=np.str_)
                data_list.append(w)
            data_index = list(file_list)
            # print(len(data_index))
            data_list = np.asarray(data_list)

            for file in file_chunk:
                temp = pd.read_csv(open(file, 'r'), header=None)
                data.append(temp.values.reshape(8192, 1))
            # for a in range(batch_size):
                index = list(file_chunk)
                # print(index)
                label_classes = tf.constant(data_list[data_index.index(file)].split('.')[0])
                pattern = tf.constant(data_list[data_index.index(file)].split('.')[0])
                if re.match(label_classes.numpy(), pattern.numpy()):
                    labels.append(data_list[data_index.index(file)].split('-')[4])
            data = np.asarray(data).reshape(-1, 8192, 1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1