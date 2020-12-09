from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np
import glob
import os


from net import create_model
from dataset import csv_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
fault_0_files = glob.glob("./data/0/*")
fault_1_files = glob.glob("./data/1/*")


def plot_learning_data(histories):

    plt.figure(figsize=(18, 10), dpi=80)
    for num, history in enumerate(histories):

        plt.subplot(2, 5, num+1)
        time = range(len(history.history["loss"]))
        plt.plot(time, history.history["loss"], label='loss', color='#152523', linestyle='--')
        plt.plot(time, history.history["accuracy"], label='accuracy', color='#5512fc')
        plt.plot(time, history.history["val_loss"], label='val_loss', color='#94A350', linestyle='--')
        plt.plot(time, history.history["val_accuracy"], label='val_accuracy', color='#CD2504')
        plt.ylim(0, 1.1)

        plt.legend(loc=0)

    plt.savefig("training loss and accuracy.png")
    # plt.show()


def main(batch_size,epochs):
    filepath = 'test6.h5'
    train_log = []

    fault_0_train, fault_0_val = train_test_split(fault_0_files, test_size=0.2, random_state=5)
    fault_1_train, fault_1_val = train_test_split(fault_1_files, test_size=0.2, random_state=54)
    fault_0_train, fault_0_test = train_test_split(fault_0_train, test_size=0.1, random_state=1)
    fault_1_train, fault_1_test = train_test_split(fault_1_train, test_size=0.1, random_state=12)

    train_file_names = fault_0_train + fault_1_train
    validation_file_names = fault_0_val + fault_1_val
    test_file_names = fault_0_test + fault_1_test
    print("Number of train_files:", len(train_file_names))
    print("Number of validation_files:", len(validation_file_names))
    print("Number of test_files:", len(test_file_names))


    train_dataset = tf.data.Dataset.from_generator(csv_data, args=[train_file_names, batch_size],
                                                   output_shapes=((None, 8192, 1), (None,)),
                                                   output_types=(tf.float64, tf.float64))
    validation_dataset = tf.data.Dataset.from_generator(csv_data, args=[validation_file_names, batch_size],
                                                        output_shapes=((None, 8192, 1), (None,)),
                                                        output_types=(tf.float64, tf.float64))

    test_dataset = tf.data.Dataset.from_generator(csv_data, args=[test_file_names, batch_size],
                                                  output_shapes=((None, 8192, 1), (None,)),
                                                  output_types=(tf.float64, tf.float64))

    # model = create_model()
    # model_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # model_optimizer = tf.keras.optimizers.SGD()

    # model.compile(loss=model_loss, optimizer=model_optimizer, metrics=["accuracy"])

    steps_per_epoch = np.int(np.ceil(len(train_file_names) / batch_size))
    validation_steps = np.int(np.ceil(len(validation_file_names) / batch_size))
    steps = np.int(np.ceil(len(test_file_names) / batch_size))
    print("steps_per_epoch = ", steps_per_epoch)
    print("steps = ", steps)
    model = load_model(filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath)
    # train_ = model.fit(train_dataset,validation_data=validation_dataset,
    #                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,epochs=epochs,callbacks=checkpoint)
    # train_ = model.fit(train_dataset,validation_data=validation_dataset,
    #                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,epochs=epochs)
    # model.save_weights('weights.h5')
    # test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps)
    # print("Test loss: ", test_loss)
    # print("Test accuracy:", test_accuracy)

    # train_log.append(train_)

    # plot_learning_data(train_log)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    main(batch_size=8,epochs=10)