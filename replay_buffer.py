from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size,epsilon_P=0.5):
        self.__buffer_size = buffer_size
        self.__permanent_buffer = deque()
        self.__volatile_buffer = deque()
        self.epsilon_Priorization=epsilon_P

    def store(self, element, permanent=False):
        if permanent:
            self.__permanent_buffer.append(element)
        else:
            self.__volatile_buffer.append(element)

        if len(self.__permanent_buffer) + len(self.__volatile_buffer) > self.__buffer_size:
            self.__volatile_buffer.popleft()

    def get_batch(self, batch_size):
        sample_Volatile=random.sample(self.__volatile_buffer, round((1-self.epsilon_Priorization)*batch_size))
        sample_Permanent=random.sample(self.__permanent_buffer, round((self.epsilon_Priorization)*batch_size))
        list=sample_Permanent+sample_Volatile
        random.shuffle(list)
        return list

    def get_buffer(self):
        return self.__permanent_buffer + self.__volatile_buffer

    def get_size(self):
        return len(self.__volatile_buffer)

    def __getitem__(self, index):
        return (self.__permanent_buffer + self.__volatile_buffer)[index]
