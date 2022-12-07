from collections import deque
import random


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

    def get_batch(self, batch_size):
        return random.sample(self.__permanent_buffer + self.__volatile_buffer, batch_size)

    def get_buffer(self):
        return self.__permanent_buffer + self.__volatile_buffer

    def get_size(self):
        return len(self.__permanent_buffer) + len(self.__volatile_buffer)

    def __getitem__(self, index):
        return (self.__permanent_buffer + self.__volatile_buffer)[index]
