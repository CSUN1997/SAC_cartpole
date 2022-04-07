import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def __len__(self):
        return len(self.data)

    def remember(self, state, action, reward, state_next, done):
        self.data.append((state, action, reward, state_next, done))
        if len(self) > self.capacity:
            self.data = self.data[-self.capacity:]

    def sample(self, batch):
        state = []
        action = []
        reward = []
        state_next = []
        done = []
        for idx in batch:
            state.append(self.data[idx][0])
            action.append(self.data[idx][1])
            reward.append(self.data[idx][2])
            state_next.append(self.data[idx][3])
            done.append(self.data[idx][4])
        return state, action, reward, state_next, done