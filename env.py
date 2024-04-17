import gym
from gym import spaces
import numpy as np


class GraphEnv(gym.Env):

    def __init__(self, graph, start_node):
        super(GraphEnv, self).__init__()
        self.graph = graph
        self.current_node = start_node
        self.action_space = spaces.Discrete(len(graph))
        self.observation_space = spaces.Discrete(len(graph))

    def step(self, action):
        self.current_node = action
        reward = self.graph[self.current_node]['reward']
        done = self.current_node == len(self.graph) - 1
        return self.current_node, reward, done, {}

    def reset(self):
        self.current_node = 0
        return self.current_node

