from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

    plt.savefig("training loss.png")
    # plt.show()


def main(k,batch_size,epochs):
    kf = KFold(n_splits=k)
    kf.get_n_splits(fault_0_files)
    print('kf', kf)
    train_0_file_names,test_0_file_names = [],[]
    train_1_file_names,test_1_file_names = [],[]

    for train_index, test_index in kf.split(fault_0_files):
        train_0, test_0 = list(np.asarray(fault_0_files)[train_index]), list(np.asarray(fault_0_files)[test_index])
        train_0_file_names.append(train_0)
        test_0_file_names.append(test_0)
    kf.get_n_splits(fault_1_files)
    for train_index, test_index in kf.split(fault_1_files):
        train_1, test_1 = list(np.asarray(fault_1_files)[train_index]), list(np.asarray(fault_1_files)[test_index])
        train_1_file_names.append(train_1)
        test_1_file_names.append(test_1)

    train_numbers = train_0 + train_1
    test_numbers = test_0 + test_1
    train_0 = []
    all_acc =[]
    print("Number of train_files:", len(train_numbers))
    print("Number of test_files:", len(test_numbers))


    for i in range(k):

        filepath = 'test' + str(i) + '.h5'
        train = train_1_file_names[i]+train_0_file_names[i]
        test = test_1_file_names[i]+test_0_file_names[i]

        train_dataset = tf.data.Dataset.from_generator(csv_data, args=[train, batch_size],
                                                       output_shapes=((None, 8192, 1), (None,)),
                                                       output_types=(tf.float64, tf.float64))

        test_dataset = tf.data.Dataset.from_generator(csv_data, args=[test, batch_size],
                                                      output_shapes=((None, 8192, 1), (None,)),
                                                      output_types=(tf.float64, tf.float64))

        model = create_model()
        model_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model_optimizer = tf.keras.optimizers.SGD()

        model.compile(loss=model_loss, optimizer=model_optimizer, metrics=["accuracy"])

        steps_per_epoch = np.int(np.ceil(len(train) / batch_size))
        steps = np.int(np.ceil(len(test) / batch_size))
        print("steps_per_epoch = ", steps_per_epoch)
        print("steps = ", steps)
        # model.load_weights(filepath)
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        train_log = model.fit(train_dataset,validation_data=test_dataset,
                            steps_per_epoch=steps_per_epoch, validation_steps=steps,epochs=epochs)

        model.save(filepath)
        test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps)
        print("Test loss: ", test_loss)
        print("Test accuracy:", test_accuracy)

        train_0.append(train_log)
        all_acc.append(test_accuracy)

    print("all the accuracy:{}, test accuracy:{:.4f}.".format(all_acc, sum(all_acc) / len(all_acc)))

    plot_learning_data(train_0)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    main(k=10,batch_size=8,epochs=1000)