import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Cifar10ImageDataset:
    def __init__(self, batch_size=32):

        self.batch_size = batch_size
        self.create_dataset()
        self.get_labels_counts()
        self.get_rho()
        self.get_minority_classes()

    def create_dataset(self):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        X_train = X_train.reshape(-1, 32, 32, 3)
        y_train = y_train.reshape(y_train.shape[0], )
        X_test = X_test.reshape(-1, 32, 32, 3)
        y_test = y_test.reshape(y_test.shape[0], )
        X_train = X_train / 255.
        X_test = X_test / 255.
        
        # We are going to get 25% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        self.no_of_minority_classes_to_get = int(np.round(len(np.unique(y_train)) * 0.25))
        
        
        # Specify the percentage of label 2 data to remove
        percentage_to_remove = 90
        for class_to_remove in range(self.no_of_minority_classes_to_get):
            # Use the filter_data function to create a new dataset with filtered data
            # Find indices of samples with the chosen label
            indices_to_remove = np.where(y_train == class_to_remove)[0]

            # Calculate the number of samples to remove
            num_samples_to_remove = int(percentage_to_remove / 100 * len(indices_to_remove))

            # Randomly select indices to remove
            indices_to_remove = np.random.choice(indices_to_remove, num_samples_to_remove, replace=False)

            # Remove the selected samples
            X_train = np.delete(X_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            percentage_to_remove -= 10
            
        
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test
        
        self.length_of_dataset = len(y_train)
    
    # Define a function to filter out data with label 2 based on a percentage
    def filter_data(self, image, label, label_to_drop, percentage_to_remove):
        # Assuming label 2 corresponds to the class you want to remove
        if label == label_to_drop.numpy() and tf.random.uniform(()) < percentage_to_remove:
            return False
        return True
    
    def get_labels_counts(self):
        # Get unique labels and their counts
        self.unique_labels, self.label_counts = np.unique(self.y_train, return_counts=True)
        
        return self.label_counts
        
    def get_minority_classes(self):
        unique_labels_counts_dict = dict(zip(self.unique_labels, self.label_counts))
        unique_labels_counts_dict = sorted(unique_labels_counts_dict.items())
        
        self.minority_classes = []
        for i in range(self.no_of_minority_classes_to_get):
            self.minority_classes.append(unique_labels_counts_dict[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, this paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        labels_counts = self.get_labels_counts()
        raw_reward_set = 1 / labels_counts
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))

class CassavaLeafDataset:
    def __init__(self, image_size = (512, 512), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.reading_csv("./cassava-leaf-disease-classification/train_images/", "./cassava-leaf-disease-classification/train.csv")
        self.create_dataset()

        self.get_rho()
        self.get_minority_classes()

    def reading_csv(self, folder_path, file_path):
        df = pd.read_csv(file_path) # Load train image file names and each label data
        df["filepath"] = folder_path + df["image_id"] # Create path by adding folder name and image name for load images easily
        df = df.drop(['image_id'],axis=1) # Drop image names which is useless.
        self.X = df.drop(columns=["label"])
        self.y = df["label"]

    
    def imbalance_the_data(self):
        pass

    def load_image_and_label_from_path(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        return img, label
    
    def create_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.2)
        

        training_data = tf.data.Dataset.from_tensor_slices((X_train.filepath.values, y_train))
        testing_data = tf.data.Dataset.from_tensor_slices((X_test.filepath.values, y_test))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        training_data = training_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
        testing_data = testing_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

        self.training_data_batches = training_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        self.testing_data_batches = testing_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        self.length_of_dataset = len(list(self.training_data_batches.as_numpy_iterator()))
        return self.training_data_batches, self.testing_data_batches

    def get_class_num(self):
        # get number of all classes
        _, nums_cls = np.unique(self.y, return_counts=True)
        print("No of total samples in dataset and their distribution: ", np.unique(self.y, return_counts=True))
        
        return nums_cls

    def get_minority_classes(self):
        label, label_count = np.unique(self.y, return_counts=True)
        print("Label is ", label)
        labels_with_counts = {}
        for i in range(len(label)):
            labels_with_counts[label[i]] = label_count[i]
        print("Labels with count",labels_with_counts)
        labels_with_counts = sorted(labels_with_counts.items())

        # We are going to get 35% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        no_of_minority_classes_to_get = int(np.round(len(label) * 0.35))

        self.minority_classes = []
        for i in range(no_of_minority_classes_to_get):
            self.minority_classes.append(labels_with_counts[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, research paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_class_num()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))


class PersonalityDataset:
    def __init__(self, batch_size=100):

        self.batch_size = batch_size
        self.create_dataset()
        self.get_rho()
        self.get_minority_classes()

    def create_dataset(self):
        df = pd.read_csv("../16P/16P.csv", encoding='cp1252')
        
        df = df.dropna()

        self.X = df.drop(["Personality", "Response Id"], axis = 1)
        self.y = df["Personality"]

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        
        
        X_train, X_test, y_train, y_test = train_test_split(self.X.values, self.y, random_state=42, test_size=0.2)
        
        # y_train = self.label_encoder.fit_transform(y_train)
        # y_test = self.label_encoder.fit_transform(y_test)

        # We are going to get 25% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        self.no_of_minority_classes_to_get = int(np.round(len(np.unique(y_train)) * 0.25))
        unique_labels = np.unique(y_train)
        # Specify the percentage of label 2 data to remove
        percentage_to_remove = 90
        for class_to_remove in range(self.no_of_minority_classes_to_get):
            
            # Use the filter_data function to create a new dataset with filtered data
            # Find indices of samples with the chosen label
            indices_to_remove = np.where(y_train == unique_labels[class_to_remove])[0]

            # Calculate the number of samples to remove
            num_samples_to_remove = int(percentage_to_remove / 100 * len(indices_to_remove))

            # Randomly select indices to remove
            indices_to_remove = np.random.choice(indices_to_remove, num_samples_to_remove, replace=False)

            # Remove the selected samples
            X_train = np.delete(X_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            percentage_to_remove -= 10

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.length_of_dataset = len(X_train)
        
    def get_labels_counts(self):
        # Get unique labels and their counts
        self.unique_labels, self.label_counts = np.unique(self.y_train, return_counts=True)
        
        return self.label_counts
        
    def get_minority_classes(self):
        unique_labels_counts_dict = dict(zip(self.unique_labels, self.label_counts))
        unique_labels_counts_dict = sorted(unique_labels_counts_dict.items())
        
        self.minority_classes = []
        for i in range(self.no_of_minority_classes_to_get):
            self.minority_classes.append(unique_labels_counts_dict[i][0])

    def get_minority_classes(self):
        label, label_count = np.unique(self.y, return_counts=True)
        print("Label is ", label)
        labels_with_counts = {}
        for i in range(len(label)):
            labels_with_counts[label[i]] = label_count[i]
        print("Labels with count",labels_with_counts)
        labels_with_counts = sorted(labels_with_counts.items())

        # We are going to get 35% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        no_of_minority_classes_to_get = int(np.round(len(label) * 0.35))

        self.minority_classes = []
        for i in range(no_of_minority_classes_to_get):
            self.minority_classes.append(labels_with_counts[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, research paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_labels_counts()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))