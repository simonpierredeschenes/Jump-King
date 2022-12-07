from poutyne import Model
import numpy as np
import torch

NB_TRAJECTORIES = 1200
RUN_VISUALIZATION = False


class DQN(Model):
    def __init__(self, nb_actions, *args, **kwargs):
        self.nb_actions = nb_actions
        super().__init__(*args, **kwargs)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            rand = np.random.randint(0, self.nb_actions)
            return rand
        else:
            q_vals = self.predict_on_batch(state)
            return np.argmax(q_vals)

    def soft_update(self, other, tau):
        new_weights = {}

        own_weights = self.get_weight_copies()
        other_weights = other.get_weight_copies()

        for k in own_weights:
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

        self.set_weights(new_weights)


class NNModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden_layers=3, hidden_dim=128):
        super().__init__()

        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.fa = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fa(x)


def format_batch(batch, target_network, gamma):
    state_height = np.vstack([x[0][0] for x in batch])
    state_x = np.vstack([x[0][1] for x in batch])
    state_y = np.vstack([x[0][2] for x in batch])
    jumping = np.vstack([x[0][3] for x in batch])
    edges = []
    i = 0
    for x in batch: # format_state?
        j = 0
        edges.append(np.zeros(200))
        for y in range(len(x[0][4])):
            edges[i][j] = x[0][4][y][0]
            edges[i][j + 1] = x[0][4][y][1]
            edges[i][j + 2] = x[0][4][y][2]
            edges[i][j + 3] = x[0][4][y][3]
            j += 4
        i += 1
    edges = np.vstack(edges)
    actions = np.array([x[1] for x in batch]).astype(np.float32)
    rewards = np.array([x[2] for x in batch])
    next_state_height = np.vstack([x[3][0] for x in batch])
    next_state_x = np.vstack([x[3][1] for x in batch])
    next_state_y = np.vstack([x[3][2] for x in batch])

    next_jumping = np.vstack([x[3][3] for x in batch])
    next_edges = []
    i = 0
    for x in batch: # format_state?
        j = 0
        next_edges.append(np.zeros(200))
        for y in range(len(x[3][4])):
            next_edges[i][j] = x[3][4][y][0]
            next_edges[i][j + 1] = x[3][4][y][1]
            next_edges[i][j + 2] = x[3][4][y][2]
            next_edges[i][j + 3] = x[3][4][y][3]
            j += 4
        i += 1
    next_edges = np.vstack(next_edges)
    states = np.concatenate((state_height, state_x, state_y, jumping, edges), axis=1).astype(np.float32)
    next_states = np.concatenate((next_state_height, next_state_x, next_state_y, next_jumping, next_edges), axis=1).astype(np.float32)
    next_q_vals = target_network.predict_on_batch(next_states)
    max_q_vals = np.max(next_q_vals, axis=-1)
    targets = (rewards + gamma * max_q_vals).astype(np.float32)
    return states, (actions, targets)


def format_state(state):
    state_height = state[0]
    state_x = state[1]
    state_y = state[2]
    jumping = state[3]
    edges = np.zeros(200)
    for i in range(min(len(state[4]), 50)):
        edges[4 * i] = state[4][i][0]
        edges[4 * i + 1] = state[4][i][1]
        edges[4 * i + 2] = state[4][i][2]
        edges[4 * i + 3] = state[4][i][3]
    return np.concatenate(([state_height, state_x, state_y, jumping], edges), axis=0).astype(np.float32)


def dqn_loss(y_pred, y_target):
    actions, Q_target = y_target
    Q_predict = y_pred.gather(1, actions.unsqueeze(-1).to(torch.int64)).squeeze()
    return torch.nn.functional.mse_loss(Q_predict, Q_target)
