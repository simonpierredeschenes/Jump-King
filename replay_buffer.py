from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size
        self.__permanent_buffer = deque()
        self.__volatile_buffer = deque()

    def store(self, element, permanent=False):
        if permanent:
            self.__permanent_buffer.append(element)
        else:
            self.__volatile_buffer.append(element)

        if len(self.__permanent_buffer) + len(self.__volatile_buffer) > self.__buffer_size:
            self.__volatile_buffer.popleft()

    def get_batch(self, batch_size, epsilon_priorization=0.5):
        permanent_samples = random.sample(self.__permanent_buffer, round(epsilon_priorization * batch_size))
        volatile_samples = random.sample(self.__volatile_buffer, round((1 - epsilon_priorization) * batch_size))
        batch = permanent_samples + volatile_samples
        random.shuffle(batch)
        return batch

    def get_buffer(self):
        return self.__permanent_buffer + self.__volatile_buffer

    def get_size(self):
        return len(self.__volatile_buffer)

    def get_size_permanent(self):
        return len(self.__permanent_buffer)

    def get_permanent_buffer(self):
        return self.__permanent_buffer

    def __getitem__(self, index):
        return (self.__permanent_buffer + self.__volatile_buffer)[index]

    def closest_distance_state_and_historic(self, state):
        list_historic = []
        for x in self.__permanent_buffer:
            list_historic.append(np.sqrt(((x[0][0] - state[0]) ** 2) + ((x[0][1] - state[1]) ** 2)))
        index = np.argmin(list_historic)
        return self.__permanent_buffer[index]
