import tensorflow as tf
import numpy as np

class Cifar10ImageDataset:
    def __init__(self, batch_size=32):

        self.batch_size = batch_size
        self.create_dataset()
        self.get_labels_counts()
        self.get_rho()
        self.get_minority_classes()

    def create_dataset(self):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        # Convert labels to integers
        y = y.flatten()

        # Create a TensorFlow Dataset from the CIFAR-10 data
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # We are going to get 25% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        self.no_of_minority_classes_to_get = int(np.round(len(np.unique(y)) * 0.25))
        
        
        # Specify the percentage of label 2 data to remove
        percentage_to_remove = 0.9
        for min_ in range(self.no_of_minority_classes_to_get):
            # Use the filter_data function to create a new dataset with filtered data
            dataset = dataset.filter(lambda x, y: tf.py_function(self.filter_data, inp=[x, y, min_, percentage_to_remove], Tout=tf.bool))
            percentage_to_remove -= 0.1
        
        self.dataset = dataset
        
        self.length_of_dataset = len(list(self.dataset.as_numpy_iterator()))
    
    # Define a function to filter out data with label 2 based on a percentage
    def filter_data(self, image, label, label_to_drop, percentage_to_remove):
        # Assuming label 2 corresponds to the class you want to remove
        if label == label_to_drop.numpy() and tf.random.uniform(()) < percentage_to_remove:
            return False
        return True
    
    def get_labels_counts(self):
        labels_dataset = self.dataset.map(lambda x, y: y)
        
        # Convert the labels dataset to a NumPy array
        labels_array = list(labels_dataset.as_numpy_iterator())

        # Get unique labels and their counts
        self.unique_labels, self.label_counts = np.unique(labels_array, return_counts=True)
        
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