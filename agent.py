# The agent is called 10 times per second approximately
# Each time, it must compute the next action based on its historic
#
# A state is a vector of size 5
#   The first element is the global height of the player in the game (0 is at the bottom)
#   The second element is the x position of the player in the screen (0 is at the left)
#   The third element is the y position of the player in the screen (0 is at the top)
#   The fourth element is true if the player is touching the ground, false otherwise
#   The fifth element is a vector containing all the solid edges in the screen under the form (x1, y1, x2, y2)
#
# An action is a vector of size 2
#   The first element is either -1 to move left, 0 for no lateral movement or 1 to move right
#   The second element is true to jump, false otherwise
#
# A historic entry is a vector of size 4
#   The first element is the previous state
#   The second element is the previous action
#   The third element is the reward received after the previous action
#   The fourth element is the next state

import csv
import numpy as np
from copy import deepcopy
from deep_q_learning import DQN, NNModel, format_batch, format_state, dqn_loss
from replay_buffer import ReplayBuffer
from demonstration import Demonstration
import torch

ACTIONS = np.array([[-1, 1], [1, 1], [-1, 0], [1, 0], [0, 1], [0, 0]])
BATCH_SIZE = 200
GAMMA = 0.99
BUFFER_SIZE = 10000
TAU = 1e-3
TRAINING_INTERVAL = 20
LEARNING_RATE = 5e-4


class Agent:
    def __init__(self, historic=None):
        self.epsilon = 1
        self.model = NNModel(204, 6, 6, 2040)
        self.source_network = DQN(ACTIONS.shape[0], self.model, torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE), loss_function=dqn_loss)
        self.target_network = deepcopy(self.source_network)
        self.total_n_steps = 0
        self.G = 0
        self.last_loss_episode = 0
        self.loss = 0
        self.is_demonstrating = True
        self.demonstration = Demonstration()
        if historic is None:
            self.historic = ReplayBuffer(BUFFER_SIZE)
        else:
            self.historic = historic

    def add_entry_to_historic(self, previous_state, action, reward, next_state):
        action_index = 0
        for i in range(len(ACTIONS)):
            if action[0] == ACTIONS[i][0] and action[1] == ACTIONS[i][1]:
                action_index = i
                break
        self.historic.store((previous_state, action_index, reward, next_state), permanent=self.is_demonstrating)

    def choose_action(self):
        action = self.demonstration.get_next_action()
        if action is None:
            self.is_demonstrating = False
            action = ACTIONS[self.choose_action_NN()]

        if self.historic.get_size() % 500 == 0:
            with open('historique' + str(self.historic.get_size()) + '.csv', 'w+') as file:
                writer = csv.writer(file)
                batch = format_batch(self.historic.get_buffer(), self.target_network, GAMMA)
                for row in range(len(batch)):
                    writer.writerow(batch[row][:])

        return action

    def choose_action_NN(self):
        next_state = self.historic[-1][3]
        state = self.historic[-1][0]
        action = self.historic[-1][1]
        reward = self.historic[-1][2]
        self.G += reward

        self.historic.store((state, action, reward, next_state))
        self.total_n_steps += 1

        if self.historic.get_size() > BATCH_SIZE and self.total_n_steps % TRAINING_INTERVAL == 0:
            minibatch = self.historic.get_batch(BATCH_SIZE)
            formatted_minibatch = format_batch(minibatch, self.target_network, GAMMA)
            self.last_loss_episode = self.source_network.train_on_batch(*formatted_minibatch)
            self.target_network.soft_update(self.source_network, TAU)

        if self.total_n_steps == 1:
            with open("dql.csv", "w+", newline="") as file:
                file.write("total_nb_steps,cumulative_reward,loss\n")
                file.write(str(self.total_n_steps) + "," + str(self.G) + "," + str(self.last_loss_episode) + "\n")
                print("Après " + str(self.total_n_steps) + " actions: reward " + "," + str(self.G) + ", dernier loss: " + str(self.last_loss_episode) + "\n")
        else:
            with open("dql.csv", "a", newline="") as file:
                file.write(str(self.total_n_steps) + "," + str(self.G) + "," + str(self.last_loss_episode) + "\n")
                print("Après " + str(self.total_n_steps) + " actions: reward " + "," + str(
                    self.G) + ", dernier loss: " + str(self.last_loss_episode) + "\n")

            state = format_state(self.historic[-1][-1])
            action = self.source_network.get_action(state, self.epsilon)
            self.epsilon = max(self.epsilon * 0.9999, 0.05)

        return action
