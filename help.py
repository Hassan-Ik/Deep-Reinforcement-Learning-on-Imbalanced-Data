import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import gym

# Environment
env = gym.make('Pendulum-v1')  # Replace with your environment

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_high):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high

        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 64
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        
        # Actor and Critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Target networks
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        # Initialize target networks with the same weights as the online networks
        self.update_target_networks(tau=1.0)

        # Replay buffer
        self.memory = []

    def build_actor(self):
        actor = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh')  # Assuming actions are in the range [-1, 1]
        ])
        actor.compile(optimizer=optimizers.Adam(learning_rate=self.actor_lr), loss='mse')
        return actor

    def build_critic(self):
        critic = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_dim + self.action_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        critic.compile(optimizer=optimizers.Adam(learning_rate=self.critic_lr), loss='mse')
        return critic

    def update_target_networks(self, tau):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        target_actor_weights = self.target_actor.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(target_actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]

        for i in range(len(target_critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action = self.actor.predict(state)
        return action.flatten()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        dones = np.vstack(dones)

        # Target Q-values using the target networks
        target_actions = self.target_actor.predict(next_states)
        target_q_values = self.target_critic.predict(np.hstack([next_states, target_actions]))

        # Compute target Q-values
        target_values = rewards + self.gamma * target_q_values * (1 - dones)

        # Update the Critic
        self.critic.fit(np.hstack([states, actions]), target_values, epochs=1, verbose=0)

        # Update the Actor
        action_gradients = np.reshape(self.critic.gradient(np.hstack([states, actions]), -target_values), (-1, self.action_dim))
        self.actor.train_on_batch(states, action_gradients)

        # Update target networks
        self.update_target_networks(tau=0.001)

# Instantiate the DDPG agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high[0]
agent = DDPGAgent(state_dim, action_dim, action_high)

# Training loop
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, (1, state_dim))
    total_reward = 0

    for time in range(500):  # Adjust the maximum number of steps
        # env.render()  # Uncomment if rendering is desired

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, state_dim))

        # Modify the reward based on your environment's requirements if needed

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            agent.replay()
            print("episode: {}/{}, score: {}".format(episode, episodes, total_reward))
            break

#
