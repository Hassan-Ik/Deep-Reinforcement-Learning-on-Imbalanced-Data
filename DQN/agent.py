import pickle
import datetime
import os
import random

from collections import deque

import numpy as np
import tensorflow as tf
from sklearn import metrics    
class DQNAgent:
    def __init__(self, network, dataset, state_size, action_size, memory, gamma, epsilon):
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
        self.dataset = dataset
        self.memory = memory
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1024)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

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

    def act(self, state):
        """
        Getting the possilble action for a state i-e for image data it would be label

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.network.model.predict(state)
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
    
    def replay(self, batch_size):
        """
         Replay the prediction of the saved states in memory

        Args:
            batch_size (int): _description_
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminal in minibatch:
            target = self.network.model.predict(np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]]))
            if terminal:
                target[0][action] = reward
            else:
                t = self.network.target_model.predict(np.reshape(next_state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]]))[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            self.network.model.fit(np.reshape(state, [-1, self.state_size[0], self.state_size[1], self.state_size[2]]), target, epochs=1, verbose=0)
        self.update_epsilon()
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, save_path):
        """
        Saving the trained DQN model

        Args:
            save_path (str): path and name of the model weights
        """
        self.network.model.save_weights(save_path)

    def train_cassava(self, num_episodes: int =10, steps: int = 20):
        """

        """
        for episode in range(num_episodes):
            int_val = np.random.randint(0, self.dataset.length_of_dataset)
            state_batch = self.dataset.training_data_batches.take(int_val)
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
                    if terminal == 1:
                        self.network.update_target_network()
                        print("Episode: {}, Reward: {}".format(episode, reward))
                        break

                    if len(self.memory) > self.dataset.batch_size:
                        self.replay(self.dataset.batch_size)
                    
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
                action = self.act(state)
                reward, terminal = self.get_reward_and_terminal(label, action)
                total_reward += reward
                real_labels.append(label)
                predictions.append(action)

        print("Test Accuracy: {:.2%}".format(total_reward / self.dataset.length_of_dataset))
        f1_all_cls = metrics.f1_score(real_labels, predictions, average=None)
        f1_macro_avg = metrics.f1_score(real_labels, predictions, average='weighted')
        print("F1 None averaged score of our Cassava Deep Q Network model: ", f1_all_cls)
        print("F1 Weighted averaged score of our Cassava Deep Q Network model: ", f1_macro_avg)

    def train_cifar(self, num_episodes: int =10, steps: int = 20):
        """

        """
        for episode in range(num_episodes):
            random_index = np.random.randint(0, self.dataset.length_of_dataset)
            state = self.dataset.X_train[random_index]
            state = np.reshape(state, [1, self.state_size[0], self.state_size[1], self.state_size[2]])
            label = self.dataset.y_train[random_index]
            for i in range(steps):
                action = self.act(state)
                random_index = np.random.randint(0, self.dataset.length_of_dataset)
                next_state = self.dataset.X_train[random_index]
                next_state = np.reshape(next_state, [1, self.state_size[0], self.state_size[1], self.state_size[2]])
                next_label = self.dataset.y_train[random_index]
                
                # print(f"Action is {action} and label is {label}")
                
                reward, terminal = self.get_reward_and_terminal(label, action)
                
                # print(f"Reward is {reward}")
                
                self.remember(state, action, reward, next_state, terminal)
                state = next_state
                label = next_label

                if terminal == 1:
                    self.network.update_target_network()
                    print("Episode: {}, Reward: {}".format(episode, reward))
                    break

                if len(self.memory) > self.dataset.batch_size:
                    self.replay(self.dataset.batch_size)
                    
    def evaluate_cifar10(self):
        # Testing the model
        total_reward = 0
        labels = []
        predictions = []
        for index, image in enumerate(self.dataset.X_test):
            label = self.dataset.y_test[index]
            image = np.reshape(image, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])
            action = self.act(image)
            reward, terminal = self.get_reward_and_terminal(label, action)
            total_reward += reward
            labels.append(label)
            predictions.append(action)

        print("Test Accuracy: {:.2%}".format(total_reward / len(self.dataset.length_of_dataset)))

        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            f1_all_cls = metrics.f1_score(label, prediction, average=None)
            f1_macro_avg = metrics.f1_score(label, prediction, average='weighted')
            for i, f1 in enumerate(f1_all_cls):
                print("class {} : {:.3f}".format(i, f1), end=", ")
            print("weighted macro avg : {:.3f}".format(f1_macro_avg))    