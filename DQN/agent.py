import pickle
import datetime
import os
import random

from collections import deque

import numpy as np
import tensorflow as tf
from sklearn import metrics

class DQNAgent:
    def __init__(self, network, dataset, state_size, action_size, memory, epsilon):
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
            target = self.network.model.predict(np.reshape(state, [-1, self.state_size]))
            if terminal:
                target[0][action] = reward
            else:
                t = self.network.target_model.predict(next_state)[0]
                target[0][action] = reward + self.network.gamma * np.amax(t)

            print("Target is ", target)
            self.network.model.fit(state, target, epochs=1, verbose=0)

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

    def train(self, num_episodes):
        """
            
        """
        for episode in range(num_episodes):
            int_val = np.random.randint(0, self.dataset.length_of_dataset)
            state_batch = self.dataset.dataset.take(int_val)
            for state, label in list(state_batch):
                action = self.act(state)
                next_state = self.dataset.dataset.take(np.random.randint(0, self.dataset.length_of_dataset))
                next_state = tuple(next_state)[0][0]
                
                print(f"Action is {action} and label is {label}")
                
                reward, terminal = self.get_reward_and_terminal(label, action)
                
                print(f"Reward is {reward}")
                
                self.remember(state, action, reward, next_state, terminal)
                state = next_state

                if terminal == 1:
                    self.network.update_target_network()
                    print("Episode: {}, Reward: {}".format(episode, reward))
                    break

                if len(self.memory) > self.dataset.batch_size:
                    self.replay(self.dataset.batch_size)
                    
    def evaluate(self, percent_data = 0.2):
        # Testing the model
        total_reward = 0
        labels = []
        predictions = []
        no_of_data = self.dataset.length_of_dataset * percent_data
        for image, label in self.dataset[:no_of_data]:
            image = np.reshape(image, [-1, self.state_size])
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