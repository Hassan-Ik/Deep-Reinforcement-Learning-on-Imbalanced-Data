import tensorflow as tf
import numpy as np
from collections import deque
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ImbalancedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        # Main Q-network
        self.model = self.build_model()

        # Target Q-network
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Generate imbalanced data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize ImbalancedDQNAgent
state_size = X_train.shape[1]
action_size = 2  # Binary classification
agent = ImbalancedDQNAgent(state_size, action_size)

# Training the Imbalanced DQN
batch_size = 32
num_episodes = 100

for episode in range(num_episodes):
    state = X_train[np.random.randint(0, len(X_train))]
    state = np.reshape(state, [1, state_size])

    for _ in range(500):
        action = agent.act(state)
        next_state = X_train[np.random.randint(0, len(X_train))]
        next_state = np.reshape(next_state, [1, state_size])

        reward = 1 if y_train[np.argmax(next_state)] == 1 else -1
        done = False

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            agent.update_target_model()
            print("Episode: {}, Reward: {}".format(episode, reward))
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Testing the model
total_reward = 0
for test_state in X_test:
    test_state = np.reshape(test_state, [1, state_size])
    action = agent.act(test_state)
    reward = 1 if y_test[np.argmax(test_state)] == 1 else -1
    total_reward += reward

print("Test Accuracy: {:.2%}".format(total_reward / len(X_test)))
