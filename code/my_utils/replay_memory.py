from collections import namedtuple
import random
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward', 'latent_code'))

class Memory(object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            # index = random.sample(range(len(self.memory)), batch_size)
            # random_batch = [self.memory[i] for i in index]    # super slow 
            # random_batch = random.sample(self.memory, batch_size)   # still slow

            random_batch = random.choices(self.memory, k=batch_size)    # the fastest somehow
            return Transition(*zip(*random_batch))

    ## For N-step returns update. 
    ## This function return a list (sequence) length N of Transition tuple. 
    def sample_n_step(self, batch_size=None, N=1):
        if N == 1:
            return [self.sample(batch_size)]    # return a list always

        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            index = random.sample(range(len(self.memory)-N), batch_size)    # cannot select the last N element, since we have not observe future N step yet
            out = []
            for n in range(0, N):
                random_batch = [self.memory[i+n] for i in index]
                out += [Transition(*zip(*random_batch))]
            return out

    def append(self, new_memory):
        self.memory += new_memory.memory
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def reset(self):
        self.memory = []

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)
