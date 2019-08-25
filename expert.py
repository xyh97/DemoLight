import numpy as np
import random
import pickle


class Expert(object):

    def __init__(self, expert_dir, state_dim, action_dim, batch_size):
        file = open(expert_dir, "rb")
        self.expert_data = pickle.load(file)
        self.expert_size = len(self.expert_data["state"])
        self.actions = np.random.normal(scale=0.35, size=(self.expert_size, action_dim))
        self.rewards = np.random.normal(scale=0.35, size=(self.expert_size, ))
        self.states = np.random.normal(scale=0.35, size=(self.expert_size, state_dim))
        self.terminals = np.zeros(self.expert_size, dtype=np.float32)
        self.batch_size = batch_size

        self.prestates = np.empty((self.batch_size, 1, state_dim), dtype=np.float32)
        self.poststates = np.empty((self.batch_size, 1, state_dim), dtype=np.float32)
        self.add()

    def add(self):
        for idx in range(self.expert_size):
            self.actions[idx, ...] = self.expert_data["action"][idx]
            self.rewards[idx] = self.expert_data["reward"][idx][0]
            self.states[idx, ...] = self.expert_data["state"][idx]
            self.terminals[idx] = self.expert_data["done"][idx]

    def get_state(self, index):
        # if is not in the beginning of matrix
        if index >= 0:
            # use faster slicing
            return self.states[index:(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(1))]
            return self.states[indexes, ...]

    def sample(self):
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(1, self.expert_size-1)
                # if wraps over current pointer, then get new one

                # if wraps over episode end, then get new one
                # poststate (last screen) can be terminal state!
                if self.terminals[(index - 1):index].any():
                    continue
                # otherwise use this index
                break

            # having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.poststates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        actions = self.actions[indexes, ...]
        rewards = self.rewards[indexes, ...]
        terminals = self.terminals[indexes]

        return np.squeeze(self.prestates, axis=1), actions, rewards, \
               np.squeeze(self.poststates, axis=1), terminals


if __name__ == "__main__":
    expert = Expert("./sample.pkl", 16, 8, 20)
    print("hello")


