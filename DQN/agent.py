import pickle
import datetime
import os
import random

from collections import deque

import numpy as np
import tensorflow as tf
from sklearn import metrics
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DQNAgent:
    def __init__(self, network, dataset, state_size, action_size, memory, epsilon):
        self.network = network
        self.dataset = dataset
        self.memory = memory
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1024)
        
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.network.model.predict(state)
        return np.argmax(q_values[0])
    
    def get_reward_and_terminal(self, label, action):
        terminal = 0
        if action == label:
            reward = self.dataset.reward_set[label]
        else:
            reward = - self.dataset.reward_set[label]
            # End of an episode if the agent misjudgement about a minority class
            if label in self.dataset.minority_classes:
                terminal = 1
        return [reward], [terminal]
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.network.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.network.target_model.predict(next_state)[0]
                target[0][action] = reward + self.network.gamma * np.amax(t)

            self.network.model.fit(state, target, epochs=1, verbose=0)

        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, save_path):
        self.network.model.save_weights(save_path)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state_batch = self.dataset.training_data_batches[np.random.randint(0, len(self.dataset.training_data_batches))]
            
            for state, label in state_batch.take(self.dataset.batch_size):
                action = self.act(state)
                next_state = state_batch[np.random.randint(0, len(state_batch))]
                
                reward, terminal = self.get_reward_and_terminal(label, action)
                
                self.remember(state, action, reward, next_state, terminal)
                state = next_state

                if terminal == 1:
                    self.network.update_target_model()
                    print("Episode: {}, Reward: {}".format(episode, reward))
                    break

                if len(self.memory) > self.dataset.batch_size:
                    self.replay(self.dataset.batch_size)
                
        # Testing the model
        # total_reward = 0
        # for test_state in X_test:
        #     test_state = np.reshape(test_state, [1, state_size])
        #     action = self.act(test_state)
        #     reward = 1 if y_test[np.argmax(test_state)] == 1 else -1
        #     total_reward += reward

        # print("Test Accuracy: {:.2%}".format(total_reward / len(X_test)))
        
    def evaluate(self, train_label, train_prediction, val_label, val_prediction, step, show_phase="Both"):
        # Calculate f1 score of each class and weighted macro average
        print("train_step : {}, epsilon : {:.3f}".format(step, self.epsilon))
        if show_phase == "Both":
            phase = ["Train Data.", "Validation Data."]
            labels = [train_label, val_label]
            predictions = [train_prediction, val_prediction]
        elif show_phase == "Train":
            phase = ["Train Data."]
            labels = [train_label]
            predictions = [train_prediction]
        elif show_phase == "Validation":
            phase = ["Validation Data."]
            labels = [val_label]
            predictions = [val_prediction]

        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            f1_all_cls = metrics.f1_score(label, prediction, average=None)
            f1_macro_avg = metrics.f1_score(label, prediction, average='weighted')
            print("\t\t {:<20} f1-score of ".format(phase[idx]), end="")
            for i, f1 in enumerate(f1_all_cls):
                print("class {} : {:.3f}".format(i, f1), end=", ")
            print("weighted macro avg : {:.3f}".format(f1_macro_avg))