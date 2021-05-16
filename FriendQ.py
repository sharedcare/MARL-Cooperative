import numpy as np
from envs import GridWorldEnv, MGEnv, Agent
from config import get_config

import matplotlib
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000


def plot_error(error, name):
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    plt.plot(error, color='black', linestyle='-')
    plt.ylim(0, 0.5)
    plt.title(name)
    plt.xlabel("Iteration")
    plt.ylabel("Q value difference")
    fig.savefig(name + ".png", dpi=fig.dpi)
    return

ACTIONS = {'MOVE_LEFT': [0, -1],  # Move left
           'MOVE_RIGHT': [0, 1],  # Move right
           'MOVE_UP': [-1, 0],  # Move up
           'MOVE_DOWN': [1, 0],  # Move down
           'STAY': [0, 0]  # don't move
           }

class FriendQAgent(object):
    def __init__(self):
        super().__init__()
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999995

        args = get_config()
        args.env_name = "EscalationGW"
        self.env = GridWorldEnv(args)
        self.no_iter = 200000
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999995
        # stateSpace = self.env.observation_space
        # actSpace = self.env.action_space
        # dimOfQ = np.concatenate((stateSpace, actSpace))
        # self.Q = np.ones(dimOfQ)

    def state_encode(self, pos):
        return pos[0] * self.env.length + pos[1]

    def act(self, Q, state, epsilon):
        state_a = self.state_encode(state[0:2])
        state_b = self.state_encode(state[2:4])
        state_esc = self.state_encode(state[4:6])
        temp = np.random.random()
        if temp < epsilon:
            action = np.random.randint(0,4)
        else:
            actions = np.where(Q[state_a, state_b, state_esc, :, :] == np.max(Q[state_a, state_b, state_esc, :, :]))
            action = actions[0][np.random.choice(range(len(actions[0])), 1)[0]]
        return action

    def learn(self):
        errors = []
        n_states = self.env.length*self.env.length
        Q_a = np.zeros((n_states, n_states, n_states, 5, 5))
        Q_b = np.zeros((n_states, n_states, n_states, 5, 5))
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma

        i = 0
        while i < self.no_iter:
            states_a, states_b = self.env.reset()
            state = [self.state_encode(states_a[0:2]), self.state_encode(states_a[2:4]), self.state_encode(states_a[4:6])]
            while True:
                if i % 10000 == 1:
                    print(str(errors[-1]))
                before_value = Q_a[2][1][1][2][4]
                actions = [self.act(Q_a, states_a, epsilon), self.act(Q_b, states_b, epsilon)]
                states_new, rewards, done, infos = self.env.step(actions)
                states_a_new, states_b_new = states_new
                state_new = [self.state_encode(states_a_new[0:2]), self.state_encode(states_a_new[2:4]), self.state_encode(states_a_new[4:6])]
                i += 1
                if done:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2],
                                                                                    actions[0], actions[1]] + \
                                                                                alpha * (rewards[0] - Q_a[
                        state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2],
                                                                                    actions[1], actions[0]] + \
                                                                                alpha * (rewards[1] - Q_b[
                        state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    break
                else:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2],
                                                                                    actions[0], actions[1]] + \
                                                                                alpha * (rewards[0] + gamma * np.max(
                        Q_a[state_new[0], state_new[1], state_new[2], :, :]) -
                                                                                         Q_a[state[0], state[1], state[
                                                                                             2], actions[0], actions[
                                                                                                 1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2],
                                                                                    actions[1], actions[0]] + \
                                                                                alpha * (rewards[1] + gamma * np.max(
                        Q_b[state_new[0], state_new[1], state_new[2], :, :]) -
                                                                                         Q_b[state[0], state[1], state[
                                                                                             2], actions[1], actions[
                                                                                                 0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    state = state_new
                epsilon *= self.epsilon_decay
                epsilon = max(self.epsilon_min, epsilon)
                # alpha *= self.alpha_decay
                # alpha = max(self.alpha_min, alpha)
                alpha = 1 / (i / self.alpha_min / self.no_iter + 1)
        plot_error(errors, "friend_q_learning_3_2")
        return

if __name__ == "__main__":
    learner = FriendQAgent()
    learner.learn()