# The agent is called 5 times per second approximately
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
import deep_q_learning
import torch
from copy import deepcopy

class Agent:
    def __init__(self):
        self.historic = []
        #Dans le contexte des tests, je vais implémenter les paramètres directement en attribut de l'objet
        #batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate
        self.actions=[[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0]]
        self.batchSize=100
        self.gamma=0.99
        self.bufferSize=10000
        self.tau=1e-3
        self.trainingInterval=20
        self.learningRate=5e-4
        self.model=deep_q_learning.NNModel(125,2,6,250)
        self.source_network=deep_q_learning.DQN(self.actions,self.model,torch.optim.Adam(self.model.parameters(), lr=self.learningRate), loss_function=deep_q_learning.dqn_loss)
        self.target_network=deepcopy(self.source_network)
        self.replayBuffer=deep_q_learning.ReplayBuffer(self.bufferSize)


    def add_entry_to_historic(self, previous_state, action, reward, next_state):
        self.historic.append((previous_state, action, reward, next_state))

    def choose_action(self):
        direction = -1
        if len(self.historic) >= 5:
            direction = self.historic[-1][1][0] * -1
            for i in range(5):
                if self.historic[-1][1][0] != self.historic[len(self.historic)-1-i][1][0]:
                    direction = self.historic[-1][1][0]
                    break

        jump = True
        if len(self.historic) >= 4:
            jump = not self.historic[-1][1][1]
            for i in range(4):
                if self.historic[-1][1][1] != self.historic[len(self.historic)-1-i][1][1]:
                    jump = self.historic[-1][1][1]
                    break

        if len(self.historic)%100==0:
            A=deep_q_learning.format_batch(self.historic,self.target_network,self.gamma)

            f = open('historique'+str(len(self.historic))+'.csv', 'w',newline='')
            writer=csv.writer(f)
            for row in range(len(A)):
                writer.writerow(self.historic[row][:])
        if len(self.historic)>=500:
            pass

        return direction, jump
