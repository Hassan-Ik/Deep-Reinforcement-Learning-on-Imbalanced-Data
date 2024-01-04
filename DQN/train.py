from qnetwork import DQNetwork
from dataset import Cifar10ImageDataset
from agent import DQNAgent
from collections import deque


if __name__ =="__main__":
    dataset = Cifar10ImageDataset(10)
    
    network = DQNetwork(state_size=(32, 32, 3), action_size=10, gamma = 0.95, epsilon = 1.0, learning_rate = 0.001)
    
    agent = DQNAgent(network, dataset, state_size=(32, 32, 3), action_size = 10, memory=deque(maxlen=2000), epsilon=1.0)
    agent.train(100)
    agent.evaluate()