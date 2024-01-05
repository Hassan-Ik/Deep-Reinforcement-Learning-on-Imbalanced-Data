from qnetwork import DQNetwork
from dataset import Cifar10ImageDataset, CassavaLeafDataset
from agent import DQNAgent
from collections import deque
import argparse


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    parser.add_argument("dataset", help="Dataset to use for training",
                    type=str, default="cifar10")
    parser.add_argument("gamma", help="Gamma value for q network",
                    type=float, default=0.95)
    parser.add_argument("batch_size", help="Batch size of our dataset ",
                    type=int, default=32)
    parser.add_argument("epsilon", help="Epsilon starting value for our Greedy policy",
                    type=float, default=1.0)
    parser.add_argument("learning_rate", help="Learning rate of our model",
                    type=float, default=1e-4)
    parser.add_argument("episodes", help="Maximum number of episode to train our Deep Q Network on",
                    type=int, default=100)
    args = parser.parse_args()
    
    if args.dataset == "cifar10":
        dataset = Cifar10ImageDataset(args.batch_size)
        
        network = DQNetwork(state_size=(32, 32, 3), action_size=10, learning_rate = args.learning_rate)
        
        agent = DQNAgent(network, dataset, state_size=(32, 32, 3), action_size = 10, memory=deque(maxlen=2000), gamma = args.gamma,epsilon=args.epsilon)
        agent.train(args.episodes)
        agent.save_model("./model/cifar10_model.h5")
        agent.evaluate()
    elif args.dataset == "cassava":
        dataset = CassavaLeafDataset((256, 256), args.batch_size)
        
        network = DQNetwork(state_size=(256, 256, 3), action_size=5, learning_rate = args.learning_rate)
        
        agent = DQNAgent(network, dataset, state_size=(256, 256, 3), action_size = 5, memory=deque(maxlen=2000), gamma = args.gamma, epsilon=args.epsilon)
        agent.train(args.episodes)
        agent.save_model("./model/cassava_model.h5")
        agent.evaluate()