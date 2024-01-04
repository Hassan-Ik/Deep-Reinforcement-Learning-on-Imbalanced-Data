import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class CustomImageDataset:
    def __init__(self, X, y, image_size = (512, 512), batch_size=8):

        self.X = X
        self.y = y
        self.image_size = image_size
        self.batch_size = batch_size
        self.create_dataset()

        self.get_rho()
        self.get_minority_classes()

    def imbalance_the_data(self):
        pass

    def load_image_and_label_from_path(self, image_path, label):
        # img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=1)
        # Resize images to a fixed size (e.g., 224x224)
        img = tf.reshape(image_path, self.image_size)
        img = tf.cast(img, tf.float32) / 255.
        return img, label

    def create_dataset(self):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.2)
        
        # training_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train))
        # testing_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test))

        # AUTOTUNE = tf.data.experimental.AUTOTUNE

        # self.training_data = training_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
        # self.testing_data = testing_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
        X_train, X_test, y_train, y_test = tf.keras.datasets.cifar10.load_data()
        
        # self.training_data_batches = training_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        # self.testing_data_batches = testing_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        

    def get_class_num(self):
        # get number of all classes
        _, nums_cls = np.unique(self.y, return_counts=True)
        print("No of total samples in dataset and their distribution: ", np.unique(self.y, return_counts=True))
        
        return nums_cls

    def get_minority_classes(self):
        label, label_count = np.unique(self.y, return_counts=True)
        labels_with_counts = {}
        for i in range(len(label)):
            labels_with_counts[label[i]] = label_count[i]
        labels_with_counts = sorted(labels_with_counts.items())

        # We are going to get 25% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        no_of_minority_classes_to_get = int(np.round(len(label) * 0.25))

        self.minority_classes = []
        for i in range(no_of_minority_classes_to_get):
            self.minority_classes.append(labels_with_counts[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, this paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_class_num()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))