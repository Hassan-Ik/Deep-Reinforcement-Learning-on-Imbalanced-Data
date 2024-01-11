import pickle
import datetime
import os
import random

from collections import deque

import numpy as np
import tensorflow as tf
from sklearn import metrics  
import matplotlib.pyplot as plt

import seaborn as sns


class DQNAgent:
    def __init__(self, network, dataset, state_size, action_size, memory, gamma, epsilon, network_type: str = 'ddqn'):
        """
        Defining the Deep Q Learning Agent for our Imbalanced Data Classification Reinforcement Learning.

        Args:
            network (q_network): Our custom deep q network for training and prediction
            dataset (_type_): Imbalanced Dataset we are going to use to test our proposed network 
            state_size (_type_): Input shape, since we are using image dataset it would be like (32, 32, 3)
            action_size (_type_): Its is the predicted classes i-e 5 for ten labels 
            memory (_type_): A queue where we will store our network's actions, states, rewards and terminal
            epsilon (_type_): Based on which we will decide the rewarding and penalty values during episodes
        """
        self.network = network
        self.network_type = network_type
        
        self.dataset = dataset
        self.memory = memory
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1024)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = 1e-3
        self.epsilon_min = 0.01
        self.decay_steps = 500
        self.power = 4

    def remember(self, state, action, reward, next_state, terminal):
        """
        Remembering the state, action, reward, possible next_state, and terminal which means if our model predicted minority class or not

        Args:
            state (_type_): Data, i-e for image data its an image
            action (_type_): predicted class 
            reward (_type_): reward calculated based on reciprocal of the different class distributions in dataset
            next_state (_type_): Possible next image
            terminal (_type_): 1 if our model predicted minority class wrong and 0 for everything else
        """
        self.memory.append((state, action, reward, next_state, terminal))

    def act(self, state, is_training=True):
        """
        Getting the possilble action for a state i-e for image data it would be label

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if np.random.rand() < self.epsilon and is_training:
            return np.random.choice(self.action_size)
        q_values = self.network.model.predict(np.reshape(state, (-1, self.state_size[0])), verbose=0)
        return np.argmax(q_values[0])
    
    def get_reward_and_terminal(self, label, action):
        """
        Get the reward and terminal for the predicted action

        Args:
            label (int): Real label/class of the image
            action (int): Predicted label/class of the image.

        Returns:
            _type_: _description_
        """
        terminal = 0
        if action == label:
            reward = self.dataset.reward_set[label]
        else:
            reward = - self.dataset.reward_set[label]
            # End of an episode if the agent misjudgement about a minority class
            if label in self.dataset.minority_classes:
                terminal = 1
        return reward, terminal
    
    def replay(self, batch_size, step):
        """
         Replay the prediction of the saved states in memory

        Args:
            batch_size (int): _description_
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminal in minibatch:
            if len(self.state_size) > 2:
                next_state = np.reshape(next_state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
            else:
                next_state = np.reshape(next_state, [-1, self.state_size[0]])
            
            q_mnet, q_tnet = self.network.model.predict(next_state, verbose=0), self.network.target_model.predict(next_state, verbose=0)

            a_wrt_qmnet = np.argmax(q_mnet, axis=1)[:, np.newaxis]
            max_q_ = q_tnet[np.arange(len(q_tnet)), a_wrt_qmnet.flatten()]

            with tf.GradientTape() as tape:
                if len(self.state_size) > 2:
                    state = np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
                else:
                    state = np.reshape(state, [-1, self.state_size[0]])
                q_values = self.network.model(state, training=True)
                selected_q_values = tf.reduce_sum(tf.one_hot(action, self.action_size) * q_values, axis=1)
                target_q_values = reward + (1 - terminal) * self.gamma * max_q_
                loss = tf.keras.losses.CategoricalCrossentropy()(target_q_values, selected_q_values)

            gradients = tape.gradient(loss, self.network.model.trainable_variables)
            tf.keras.optimizers.Adam(learning_rate=self.network.learning_rate).apply_gradients(zip(gradients, self.network.model.trainable_variables))
            
            # tf.keras.backend.clear_session()
        self.update_epsilon(step)
        
    def update_epsilon(self, step, decay_type='polynomial'):
        """
        Using Exponential Decay/Polynomial Decay for updating epsilon value

        Args:
            step (_type_): _description_

        Returns:
            _type_: _description_
        """
        if decay_type == 'polynomial':
            # Polynomial decay function for epsilon
            decay_factor = max(0, 1.0 - step / self.decay_steps)  # Ensure decay factor is between 0 and 1
            self.epsilon = self.epsilon_min + (self.initial_epsilon - self.epsilon_min) * decay_factor**self.power
            return self.epsilon
        else:
            self.epsilon = self.epsilon_min + (self.initial_epsilon - self.epsilon_min) * np.exp(
                -self.gamma * step / self.decay_steps
            )
            return self.epsilon

    def save_model(self, save_path):
        """
        Saving the trained DQN model

        Args:
            save_path (str): path and name of the model weights
        """
        self.network.model.save_weights(save_path)

    def train_cassava(self, num_episodes: int =10, steps: int = 20):
        """
            Funciton for training on the cassava image dataset
        """
        for episode in range(num_episodes):
            int_val = np.random.randint(0, self.dataset.length_of_dataset)
            state_batch = self.dataset.training_data_batches.take(int_val)
            total_reward = 0
            for states, labels in list(state_batch):
                for index in range(states.shape[0]):
                    state = states[index].numpy()
                    label = labels[index].numpy()
                    state = np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])

                    action = self.act(state)
                    
                    print("Predicted Action", action)
                    if index < states.shape[0] - 1:
                        next_state = states[index + 1]
                    else:
                        next_state = states[0]
                        
                    print(f"Action is {action} and label is {label}")

                    reward, terminal = self.get_reward_and_terminal(label, action)

                    # print(f"Reward is {reward}")

                    self.remember(state, action, reward, next_state, terminal)
                    total_reward += reward
                
                    if terminal == 1:
                        if self.network_type == 'ddqn':
                            self.network.update_target_network()
                        break
                
            print("Total Reward: {} after {} Episodes and epsilon is {}".format(total_reward, episode, self.epsilon))
                    
            self.replay(self.dataset.batch_size, episode)
            
    def evaluate_cassava(self):
        # Testing the model
        total_reward = 0
        real_labels = []
        predictions = []
        for states, labels in list(self.dataset.testing_data_batches.as_numpy_iterator()):
            for index in range(states.shape[0]):
                state = states[index]
                label = labels[index]

                state = np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
                action = self.act(state, False)
                reward, terminal = self.get_reward_and_terminal(label, action)
                total_reward += reward
                real_labels.append(label)
                predictions.append(action)

        print("Test Accuracy: {:.2%}".format(total_reward / self.dataset.length_of_dataset))
        f1_all_cls = metrics.f1_score(real_labels, predictions, average=None)
        f1_macro_avg = metrics.f1_score(real_labels, predictions, average='weighted')
        print("F1 None averaged score of our Cassava Deep Q Network model: ", f1_all_cls)
        print("F1 Weighted averaged score of our Cassava Deep Q Network model: ", f1_macro_avg)
    
    def train_cifar10(self, num_episodes: int =10, steps: int = 100):
        """
            Function to train for cifar10 images dataset
        """
        for episode in range(num_episodes):
            random_index = np.random.randint(0, self.dataset.length_of_dataset)
            state = self.dataset.X_train[random_index]
            state = np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
            label = self.dataset.y_train[random_index]
            for i in range(steps):
                action = self.act(state)
                random_index = np.random.randint(0, self.dataset.length_of_dataset)
                next_state = self.dataset.X_train[random_index]
                next_state = np.reshape(next_state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
                next_label = self.dataset.y_train[random_index]
                
                # print(f"Action is {action} and label is {label}")
                
                reward, terminal = self.get_reward_and_terminal(label, action)
                
                # print(f"Reward is {reward}")
                
                self.remember(state, action, reward, next_state, terminal)
                state = next_state
                label = next_label

                total_reward += reward
                
                if terminal == 1:
                    if self.network_type == 'ddqn':
                        self.network.update_target_network()
                    break
            
            print("Total Reward: {} after {} Episodes and epsilon is {}".format(total_reward, episode, self.epsilon))

            self.replay(self.dataset.batch_size, episode)
                    
    def evaluate_cifar10(self):
        # Testing the model
        total_reward = 0
        train_labels = []
        train_predictions = []
        test_labels = []
        test_predictions = []
        
        for index, image in enumerate(self.dataset.X_train):
            label = self.dataset.y_train[index]
            image = np.reshape(image, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
            action = self.act(image, False)
            # action = np.argmax(self.network.model.predict(image), axis=1)
            reward, terminal = self.get_reward_and_terminal(label, action)
            total_reward += reward
            train_labels.append(label)
            train_predictions.append(action)

        for index, image in enumerate(self.dataset.X_test):
            label = self.dataset.y_test[index]
            image = np.reshape(image, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
            action = self.act(image, False)
            # action = np.argmax(self.network.model.predict(image), axis=1)
            reward, terminal = self.get_reward_and_terminal(label, action)
            total_reward += reward
            test_labels.append(label)
            test_predictions.append(action)

        # Calculate accuracy
        train_accuracy = metrics.accuracy_score(train_labels, train_predictions)
        test_accuracy = metrics.accuracy_score(test_labels, test_predictions)

        class_labels = np.unique(self.dataset.y_train)

        print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        # Confusion matrix for train set
        train_conf_matrix = metrics.confusion_matrix(train_labels, train_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - Train Set')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        plt.savefig("train_conf_matrix.png")

        # Confusion matrix for test set
        test_conf_matrix = metrics.confusion_matrix(test_labels, test_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        plt.savefig("test_conf_matrix.png")

        print("Test Accuracy: {:.2%}".format(total_reward / self.dataset.length_of_dataset))

        f1_all_cls = metrics.f1_score(train_labels, train_predictions, average=None)
        f1_macro_avg = metrics.f1_score(train_labels, train_predictions, average='weighted')
        print("weighted macro avg on training dataset: {:.3f}".format(f1_macro_avg))

        f1_all_cls = metrics.f1_score(test_labels, test_predictions, average=None)
        f1_macro_avg = metrics.f1_score(test_labels, test_predictions, average='weighted')
        print("weighted macro avg on testing dataset: {:.3f}".format(f1_macro_avg))    
    
    def train_personality(self, num_episodes: int =1000, steps: int = 100):
        """
            Function to train on personality dataset
        """

        for episode in range(num_episodes):
            random_index = np.random.randint(0, self.dataset.length_of_dataset)
            state = self.dataset.X_train[random_index]
            label = self.dataset.y_train[random_index]
            total_reward = 0
            for i in range(steps):
                action = self.act(state)
                random_index = np.random.randint(0, self.dataset.length_of_dataset)
                next_state = self.dataset.X_train[random_index]
                next_label = self.dataset.y_train[random_index]
                
                reward, terminal = self.get_reward_and_terminal(label, action)
                
                self.remember(state, action, reward, next_state, terminal)
                state = next_state
                label = next_label

                total_reward += reward

                if terminal == 1:
                    if self.network_type == 'ddqn':
                        self.network.update_target_network()
                    break
            
            print("Total Reward: {} after {} Episodes and epsilon is {}".format(total_reward, episode, self.epsilon))

            self.replay(self.dataset.batch_size, episode)
                    
    def evaluate_personality(self):
        # Testing the model
        total_reward = 0
        train_labels = []
        train_predictions = []
        test_labels = []
        test_predictions = []
        
        for index, state in enumerate(self.dataset.X_train):
            label = self.dataset.y_train[index]
            state = np.reshape(state, [-1, self.state_size[0]])
            action = self.act(state, False)
            # action = np.argmax(self.network.model.predict(image), axis=1)
            reward, terminal = self.get_reward_and_terminal(label, action)
            total_reward += reward
            train_labels.append(label)
            train_predictions.append(action)

        for index, state in enumerate(self.dataset.X_test):
            label = self.dataset.y_test[index]
            state = np.reshape(state, [-1, self.state_size[0]])
            action = self.act(state, False)
            # action = np.argmax(self.network.model.predict(image), axis=1)
            reward, terminal = self.get_reward_and_terminal(label, action)
            total_reward += reward
            test_labels.append(label)
            test_predictions.append(action)

        # Calculate accuracy
        train_accuracy = metrics.accuracy_score(train_labels, train_predictions)
        test_accuracy = metrics.accuracy_score(test_labels, test_predictions)

        class_labels = np.unique(self.dataset.y_train)

        print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        # Confusion matrix for train set
        train_conf_matrix = metrics.confusion_matrix(train_labels, train_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - Train Set')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        plt.savefig("train_conf_matrix.png")

        # Confusion matrix for test set
        test_conf_matrix = metrics.confusion_matrix(test_labels, test_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        plt.savefig("test_conf_matrix.png")

        print("Test Accuracy: {:.2%}".format(total_reward / self.dataset.length_of_dataset))

        f1_all_cls = metrics.f1_score(train_labels, train_predictions, average=None)
        f1_macro_avg = metrics.f1_score(train_labels, train_predictions, average='weighted')
        print("weighted macro avg on training dataset: {:.3f}".format(f1_macro_avg))

        f1_all_cls = metrics.f1_score(test_labels, test_predictions, average=None)
        f1_macro_avg = metrics.f1_score(test_labels, test_predictions, average='weighted')
        print("weighted macro avg on testing dataset: {:.3f}".format(f1_macro_avg))    