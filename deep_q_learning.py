import random

from poutyne import Model
from copy import deepcopy  # NEW

import numpy as np
import gym
import torch

from collections import deque

NB_TRAJECTORIES = 1200
RUN_VISUALIZATION = False

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size
        # TODO: add any needed attributes
        self.__buffer = deque()

    def store(self, element):
        '''
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        '''
        
        # TODO: implement
        self.__buffer.append(element)
        if len(self.__buffer) > self.__buffer_size:
            self.__buffer.popleft()

    def get_batch(self, batch_size):
        '''
        Returns a list of batch_size elements from the buffer.
        '''
        
        # TODO: implement
        return random.sample(self.__buffer, batch_size)

    def get_size(self):
        return len(self.__buffer)


class DQN(Model):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def get_action(self, state, epsilon):
        '''
        Returns the selected action according to an epsilon-greedy policy.
        '''
        
        # TODO: implement
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            q_vals = self.predict_on_batch(state)
            return np.argmax(q_vals)

    def soft_update(self, other, tau):
        '''
        Code for the soft update between a target network (self) and
        a source network (other).

        The weights are updated according to the rule in the assignment.
        '''
        new_weights = {}

        own_weights = self.get_weight_copies()
        other_weights = other.get_weight_copies()

        for k in own_weights:
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

        self.set_weights(new_weights)


class NNModel(torch.nn.Module,n_hidden_layers=3,hidden_dim=64):
    '''
    Neural Network with 3 hidden layers of hidden dimension 64.
    '''

    def __init__(self, in_dim, out_dim, n_hidden_layers=3, hidden_dim=64):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.fa = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fa(x)


def format_batch(batch, target_network, gamma):
    '''
    Input : 
        - batch, a list of n=batch_size elements from the replay buffer
        - target_network, the target network to compute the one-step lookahead target
        - gamma, the discount factor

    Returns :
        - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
        - (actions, targets) : where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.
    '''
    
    # TODO: Modifier pour que cela fonctionne avec le format de Jump King
    states = np.vstack([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch])
    next_states = np.vstack([x[3] for x in batch])
    dones = np.array([x[4] for x in batch])
    next_q_vals = target_network.predict_on_batch(next_states)
    max_q_vals = np.max(next_q_vals, axis=-1)
    targets = (rewards + gamma * max_q_vals * (1 - dones)).astype(np.float32)
    return states, (actions, targets)


def dqn_loss(y_pred, y_target):
    '''
    Input :
        - y_pred, (batch_size, n_actions) Tensor outputted by the network
        - y_target = (actions, targets), where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.

    Returns :
        - The DQN loss 
    '''
    
    # TODO: implement
    actions, Q_target = y_target
    Q_predict = y_pred.gather(1, actions.unsqueeze(-1).to(torch.int64)).squeeze()
    return torch.nn.functional.mse_loss(Q_predict, Q_target)


def set_random_seed(seed, environment):
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW


def run(batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate):
    environment = gym.make("LunarLander-v2")
    set_random_seed(seed, environment)

    # TODO: implement the Deep Q-Learning strategy here
    model = NNModel(environment.observation_space.shape[0], environment.action_space.n)
    source_network = DQN(environment.action_space, model, torch.optim.Adam(model.parameters(), lr=learning_rate), loss_function=dqn_loss)
    target_network = deepcopy(source_network)
    replay_buffer = ReplayBuffer(buffer_size)
    epsilon = 1.0
    total_n_steps = 0
    with open("dql.csv", "w+") as file:
        file.write("total_nb_steps,cumulative_reward,loss\n")
        for n_trajectories in range(NB_TRAJECTORIES):
            trajectory_done = False
            G = 0
            last_loss_episode = 0
            state, _ = environment.reset(seed=seed)
            state = state.astype(np.float32)
            while not trajectory_done:
                action = source_network.get_action(state, epsilon)

                next_state, reward, terminated, truncated, _ = environment.step(action)
                trajectory_done = terminated or truncated
                next_state = next_state.astype(np.float32)

                G += reward

                replay_buffer.store((state, action, reward, next_state, trajectory_done))
                total_n_steps += 1

                if replay_buffer.get_size() > batch_size and total_n_steps % training_interval == 0:
                    minibatch = replay_buffer.get_batch(batch_size)
                    formatted_minibatch = format_batch(minibatch, target_network, gamma)
                    last_loss_episode = source_network.train_on_batch(*formatted_minibatch)
                    target_network.soft_update(source_network, tau)

                state = next_state

            print(f"After {n_trajectories + 1} trajectories, we have G_0 = {G:.2f}, {epsilon:4f}")
            epsilon = max(epsilon * 0.99, 0.05)
            file.write(f"{total_n_steps},{G},{last_loss_episode}\n")
            environment.close()

    if RUN_VISUALIZATION:
        environment = gym.make("LunarLander-v2", render_mode="human")
        set_random_seed(seed, environment)
        state, _ = environment.reset(seed=seed)
        done = False
        while not done:
            action = source_network.get_action(state, 0)
            state, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            environment.render()
        environment.close()


if __name__ == "__main__":
    '''
    All hyperparameter values and overall code structure are only given as a baseline. 
    
    You can use them if they help  you, but feel free to implement from scratch the
    required algorithms if you wish!
    '''
    batch_size = 64
    gamma = 0.99
    buffer_size = 1e5
    seed = 42
    tau = 1e-3
    training_interval = 4
    learning_rate = 5e-4
    run(batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate)
