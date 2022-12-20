from poutyne import Model
import numpy as np
import torch
import scipy.optimize

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


def format_batch(batch, target_network, gamma, historic):
    state_height = np.vstack([x[0][0] for x in batch])
    state_x = np.vstack([x[0][1] for x in batch])
    state_y = np.vstack([x[0][2] for x in batch])
    jumping = np.vstack([x[0][3] for x in batch])
    jump_counter = np.vstack([x[0][4] for x in batch])
    edges = []
    i = 0
    for x in batch:  # format_state?
        j = 0
        edges.append(np.zeros(200))
        for y in range(len(x[0][5])):
            edges[i][j] = x[0][5][y][0]
            edges[i][j + 1] = x[0][5][y][1]
            edges[i][j + 2] = x[0][5][y][2]
            edges[i][j + 3] = x[0][5][y][3]
            j += 4
        i += 1
    edges = np.vstack(edges)
    actions = np.array([x[1] for x in batch]).astype(np.float32)
    rewards = np.array([x[2] for x in batch])
    next_state_height = np.vstack([x[3][0] for x in batch])
    next_state_x = np.vstack([x[3][1] for x in batch])
    next_state_y = np.vstack([x[3][2] for x in batch])

    next_jumping = np.vstack([x[3][3] for x in batch])
    next_jump_counter = np.vstack([x[3][4] for x in batch])
    next_edges = []
    i = 0
    for x in batch:  # format_state?
        j = 0
        next_edges.append(np.zeros(200))
        for y in range(len(x[3][5])):
            next_edges[i][j] = x[3][5][y][0]
            next_edges[i][j + 1] = x[3][5][y][1]
            next_edges[i][j + 2] = x[3][5][y][2]
            next_edges[i][j + 3] = x[3][5][y][3]
            j += 4
        i += 1
    next_edges = np.vstack(next_edges)
    states = np.concatenate((state_height, state_x, state_y, jumping, jump_counter, edges), axis=1).astype(np.float32)
    next_states = np.concatenate((next_state_height, next_state_x, next_state_y, next_jumping, next_jump_counter, next_edges), axis=1).astype(np.float32)
    next_q_vals = target_network.predict_on_batch(next_states)
    max_q_vals = np.max(next_q_vals, axis=-1)
    targets = (rewards + gamma * max_q_vals).astype(np.float32)
    permanent_buffer=historic.get_permanent_buffer()
    Q_permanent=[]
    for i in range(len(permanent_buffer)):
        state_permanent_buffer=format_state(permanent_buffer[i][0])
        Q_actuel=target_network.predict_on_batch(state_permanent_buffer)
        Q_permanent.append(Q_actuel[permanent_buffer[i][1]])
    return states, (actions, targets, states, historic,Q_permanent)


def format_state(state):
    state_height = state[0]
    state_x = state[1]
    state_y = state[2]
    jumping = state[3]
    jump_counter = state[4]
    edges = np.zeros(200)
    for i in range(min(len(state[5]), 50)):
        edges[4 * i] = state[5][i][0]
        edges[4 * i + 1] = state[5][i][1]
        edges[4 * i + 2] = state[5][i][2]
        edges[4 * i + 3] = state[5][i][3]
    return np.concatenate(([state_height, state_x, state_y, jumping, jump_counter], edges), axis=0).astype(np.float32)


# def dqn_loss(y_pred, y_target):
#     actions, Q_target,states,historic = y_target
#     Q_predict = y_pred.gather(1, actions.unsqueeze(-1).to(torch.int64)).squeeze()
#
#     return torch.nn.functional.mse_loss(Q_predict, Q_target)

def dqn_loss(y_pred, y_target):
    actions, Q_target, states, historic,Q_permanent = y_target
    Q_predict = y_pred.gather(1, actions.unsqueeze(-1).to(torch.int64)).squeeze()
    ###Section pour le Large-Margin Approach###

    L=[]
    index_Permanent=[]
    for i in range(0,len(Q_target),1):
        temp=[]
        index_Permanent.append(closest_distance_state_and_historic_index(historic,states[i]))
        for j in range(0,len(y_pred[0]),1):

            if int((closest_distance_state_and_historic(historic,states[i])[1]))!=j:
                temp.append(50)
            else:
                temp.append(0)
        L.append(temp)

    Large_Margin=torch.tensor(L)
    Q_avec_marge=Large_Margin+y_pred
    Q_max=[]
    for z in range(len(Q_avec_marge)):
        Q_max.append(np.max(Q_avec_marge[z].detach().numpy())-Q_permanent[index_Permanent[z]])
    Q_max=torch.tensor(np.mean(np.array(Q_max)))
    mse_loss=torch.nn.functional.mse_loss(Q_predict, Q_target)
    loss_network = Q_max + mse_loss

    return loss_network


def closest_distance_state_and_historic(historic, state):
    list_historic = []
    for x in range(0, historic.get_size_permanent(), 1):
        list_historic.append(
            np.sqrt(((historic.get_permanent_buffer()[x][0][0] - state[0].item()) ** 2) + ((historic.get_permanent_buffer()[x][0][1] - state[1].item()) ** 2)))
    index = np.argmin(list_historic)
    return historic.get_permanent_buffer()[index]

def closest_distance_state_and_historic_index(historic, state):
    list_historic = []
    for x in range(0, historic.get_size_permanent(), 1):
        list_historic.append(
            np.sqrt(((historic.get_permanent_buffer()[x][0][0] - state[0].item()) ** 2) + ((historic.get_permanent_buffer()[x][0][1] - state[1].item()) ** 2)))
    index = np.argmin(list_historic)
    return index