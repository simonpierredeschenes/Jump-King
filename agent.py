# The agent is called 10 times per second approximately
# Each time, it must compute the next action based on its historic
#
# A state is a vector of size 5
#   The first element is the global height of the player in the game (0 is at the bottom)
#   The second element is the x position of the player in the screen (0 is at the left)
#   The third element is the y position of the player in the screen (0 is at the top)
#   The fourth element is true if the player is touching the ground, false otherwise
#   The fifth element is a counter of since how many time steps is the jump button held
#   The sixth element is a vector containing all the solid edges in the screen under the form (x1, y1, x2, y2)
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

import numpy as np
from copy import deepcopy
from deep_q_learning import DQN, NNModel, format_batch, format_state, dqn_loss
from replay_buffer import ReplayBuffer
from demonstration import Demonstration
import torch

ACTIONS = np.array([[-1, 1], [1, 1], [-1, 0], [1, 0], [0, 1], [0, 0]])
NB_PRE_TRAINING_UPDATES = 200
BATCH_SIZE = 200
GAMMA = 0.99
BUFFER_SIZE = 10000
TAU = 5e-3
TRAINING_INTERVAL = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY=0.01
NBRE_COUCHE_NN=6
NB_NEURONES_NN=2050
EPSILON=0.9999
LAMBDA_1=0.7
LAMBDA_2=0.5

class Agent:
    def __init__(self, historic=None):
        self.epsilon = 1
        self.model = NNModel(205, 6, NBRE_COUCHE_NN, NB_NEURONES_NN)
        self.source_network = DQN(ACTIONS.shape[0], self.model, torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY), loss_function=dqn_loss)
        self.target_network = deepcopy(self.source_network)
        self.total_nb_steps = 0
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
        if self.is_demonstrating:
            action = self.demonstration.get_next_action()
            if action is None:
                self.is_demonstrating = False
                self.pre_train_network()
                action = ACTIONS[self.choose_action_NN()]
        else:
            action = ACTIONS[self.choose_action_NN()]

        return action

    def pre_train_network(self):
        for i in range(NB_PRE_TRAINING_UPDATES):
            minibatch = self.historic.get_batch(BATCH_SIZE, epsilon_priorization=1)
            formatted_minibatch = format_batch(minibatch, self.target_network, GAMMA, self.historic,[LAMBDA_1,LAMBDA_2])
            self.source_network.train_on_batch(*formatted_minibatch)
            self.target_network.soft_update(self.source_network, TAU)
            if (i+1) % (NB_PRE_TRAINING_UPDATES // 10) == 0:
                print(f"Pre-training {100 * (i+1) / NB_PRE_TRAINING_UPDATES:.1f}% completed")

    def choose_action_NN(self):
        state = self.historic[-1][-1]
        self.G += self.historic[-1][2]
        self.total_nb_steps += 1

        formatted_state = format_state(state)
        action = self.source_network.get_action(formatted_state, self.epsilon)
        self.epsilon = max(self.epsilon * EPSILON, 0.05)

        if self.historic.get_size() > BATCH_SIZE and self.total_nb_steps % TRAINING_INTERVAL == 0:
            minibatch = self.historic.get_batch(BATCH_SIZE)
            formatted_minibatch = format_batch(minibatch, self.target_network, GAMMA, self.historic,[LAMBDA_1,LAMBDA_2])
            self.last_loss_episode = self.source_network.train_on_batch(*formatted_minibatch)
            self.target_network.soft_update(self.source_network, TAU)
            if self.total_nb_steps % 600 == 0:
                self.source_network.save_weights("source_weights.pth")
                self.target_network.save_weights("target_weights.pth")

        return action
