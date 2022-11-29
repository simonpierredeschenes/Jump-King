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

import numpy

import deep_q_learning
import torch
from copy import deepcopy

class Agent:
    def __init__(self):
        self.historic = []
        #Dans le contexte des tests, je vais implémenter les paramètres directement en attribut de l'objet
        #batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate
        self.actions=numpy.array([[-1,1],[1,1],[-1,0],[1,0],[0,1],[0,0]])
        self.batchSize=100
        self.gamma=0.99
        self.bufferSize=10000
        self.tau=1e-3
        self.epsilon=1
        self.trainingInterval=20
        self.learningRate=5e-4
        self.model=deep_q_learning.NNModel(204,6,6,2040)
        self.source_network=deep_q_learning.DQN(self.actions,self.model,torch.optim.Adam(self.model.parameters(), lr=self.learningRate), loss_function=deep_q_learning.dqn_loss)
        self.target_network=deepcopy(self.source_network)
        self.replayBuffer=deep_q_learning.ReplayBuffer(self.bufferSize)
        self.total_n_steps=0
        self.nbTrajectories=2000
        self.G=0
        self.last_loss_episode=0
        self.pretraining=200

    def add_entry_to_historic(self, previous_state, action, reward, next_state):
        actionIndex=0
        for k in range(len(self.actions)):
            if action[0]==self.actions[k][0] and action[1]==self.actions[k][1]:
                actionIndex=k
            else:
                pass
        self.historic.append((previous_state, actionIndex, reward, next_state))

    def choose_action(self):
        if len(self.historic) < self.pretraining:
            direction = -1
            if len(self.historic) >= 5 and self.actions[self.historic[-1][1]][0] != 0:
                direction = self.actions[self.historic[-1][1]][0] * -1
                for i in range(5):
                    if self.actions[self.historic[-1][1]][0] != self.actions[self.historic[len(self.historic)-1-i][1]][0]:
                        direction = self.actions[self.historic[-1][1]][0]
                        break

            jump = True
            if len(self.historic) >= 4:
                jump = not self.actions[self.historic[-1][1]][1]
                for i in range(4):
                    if self.actions[self.historic[-1][1]][1] != self.actions[self.historic[len(self.historic)-1-i][1]][1]:
                        jump = self.actions[self.historic[-1][1]][1]
                        break
        else:
            if len(self.historic)==self.pretraining:
                print("--->Début du NN<----")
            direction=self.actions[self.choose_action_NN()][0]
            jump=self.actions[self.choose_action_NN()][1]

        if len(self.historic)%500==0:
            A=deep_q_learning.format_batch(self.historic,self.target_network,self.gamma)

            f = open('historique'+str(len(self.historic))+'.csv', 'w',newline='')
            writer=csv.writer(f)
            for row in range(len(A)):
                writer.writerow(A[row][:])

        return direction, jump

    def choose_action_NN(self):
        next_state = self.historic[-1][3]
        state=self.historic[-1][0]
        action=self.historic[-1][1]
        reward=self.historic[-1][2]
        self.G += reward

        self.replayBuffer.store((state, action, reward, next_state))
        self.total_n_steps += 1

        if self.replayBuffer.get_size() > self.batchSize and self.total_n_steps % self.trainingInterval == 0:
            minibatch = self.replayBuffer.get_batch(self.batchSize)
            formatted_minibatch = deep_q_learning.format_batch(minibatch, self.target_network, self.gamma)
            input=torch.FloatTensor(formatted_minibatch[0])
            output=torch.FloatTensor(formatted_minibatch[1])
            self.last_loss_episode = self.source_network.train_on_batch(input,output)
            self.target_network.soft_update(self.source_network, self.tau)

        state = next_state
        with open("dql.csv", "w+") as file:
            file.write("total_nb_steps,cumulative_reward,loss\n")

            trajectory_done = False
            #state, _ = environment.reset(seed=seed)
            state = deep_q_learning.state_format(self.historic[-1][-1])
            if not trajectory_done:
                action = self.source_network.get_action(state, self.epsilon)
            self.epsilon = max(self.epsilon * 0.99, 0.05)
            # condition d'arrêt pour trajectory_done
        return action
