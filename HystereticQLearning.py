import random
import gym
import copy
import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from envs import GridWorldEnv, MGEnv, Agent
from config import get_config


# Some Hyper Parameters
Gamma = 0.95
NUM_EPISODE = 10000


class Policy(object):
    def action_prob(self, state: list, action: int) -> float:
        """
        input:
            state: 6-dim int which indicate:
                x-y position of the agent,
                x-y position of the other agent,
                x-y position of the target
            action: int indicate which action
        return:
            \pi(a|s)
        """
        raise NotImplementedError("Implement in the subclass")

    def action(self, state: list) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError("Implement in the subclass")


class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        """
        input:
            nA: number of actions
        """
        self.nA = nA
        if p is not None:
            assert len(p) == self.nA
            self.p = p
        else:
            self.p = np.array([1.0/self.nA] * self.nA)

    def action_prob(self, state, action=None):
        if action is not None:
            return self.p[action]
        else:
            return self.p[0]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, nA, gw_size=5, Q=None, epsilon=0.0):
        """
        input:
            nA: number of actions
            gw_size: the size of the matrix grid-world (default: 5)
        """
        # init the Q-value
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.zeros([nA, gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        self.epsilon = epsilon
        self.nAction = nA
        self.gw_size = gw_size

    def action_prob(self, state, action):
        assert len(state) == 6          # the state (observation of the agent) should be 6-dim
        current_state_Q = self.Q[:, state[0], state[1], state[2], state[3], state[4], state[5]]
        q_best = np.max(current_state_Q)
        best_action = np.where(current_state_Q == q_best)[0]
        if action in best_action:
            return (1 - self.epsilon) / len(best_action) + self.epsilon / self.nAction
        else:
            return self.epsilon / self.nAction

    def action(self, state):
        assert len(state) == 6          # the state (observation of the agent) should be 6-dim
        if np.random.rand() < self.epsilon:
            # random choose an action
            return np.random.choice(np.arange(self.nAction))
        else:
            # greedy choose (break tie randomly)
            current_state_Q = self.Q[:, state[0], state[1], state[2], state[3], state[4], state[5]]
            q_best = np.max(current_state_Q)
            return np.random.choice(np.where(current_state_Q == q_best)[0])


class ReinforcementLearning(object):
    def __init__(self, env):
        """
        :param env: environment
        """
        self.env = env

    def QLearning(self, nEpisodes, epsilon=0.1, alpha=0.1, max_step=10000):
        # get the number of action
        nAction = self.env.action_space[0].n
        gw_size = self.env.length
        n_agents = self.env.num_agents
        assert n_agents == 2
        # init the Q-table, since we have 2 agents, we need to assign a Q-table for each one of them
        Q1 = np.zeros([nAction, gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        Q2 = np.zeros([nAction, gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        # init the policy table
        policy1 = np.zeros([gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        policy2 = np.zeros([gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        # Save the return value for each episode
        total_reward_episodes = []

        # start training
        for episode in range(nEpisodes):
            # reset the env
            init_observ = env.reset()
            # get the position for agent1, agent2 and escalation(target)
            s1 = init_observ[0]
            s2 = init_observ[1]
            # reset the done flag
            done = False
            steps_cnt = 0
            R_traj_agent_1 = 0.0
            R_traj_agent_2 = 0.0
            coop_num_cnt = 0
            collective_reward = 0
            cumu_discount = 1.0
            while not done:
                # choose A from S using policy derived from Q
                behavior_policy_agent_1 = EpsilonGreedyPolicy(nAction, gw_size, Q1, epsilon)
                behavior_policy_agent_2 = EpsilonGreedyPolicy(nAction, gw_size, Q2, epsilon)
                action_agent_1 = behavior_policy_agent_1.action(s1)
                action_agent_2 = behavior_policy_agent_2.action(s2)
                # convert agent action to 1-hot format
                action_agent_1_onehot = np.zeros(nAction)
                action_agent_1_onehot[action_agent_1] = 1
                action_agent_2_obehot = np.zeros(nAction)
                action_agent_2_obehot[action_agent_2] = 1
                # Take action A, observe R, S'
                observations, rewards, dones, infos = self.env.step([action_agent_1_onehot, action_agent_2_obehot])
                # Get the next state for each agent
                next_s1 = observations[0]
                next_s2 = observations[1]
                # get the reward for each agent
                reward_agent_1 = rewards[0]
                reward_agent_2 = rewards[1]
                # Update Q(S,A)
                Q1[action_agent_1, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5]] = \
                    (1 - alpha) * Q1[action_agent_1, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5]] \
                    + alpha * (reward_agent_1 + Gamma * np.max(Q1[:, next_s1[0], next_s1[1], next_s1[2], next_s1[3], next_s1[4], next_s1[5]]))
                # Update Q2(S,A)
                Q2[action_agent_2, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]] = \
                    (1 - alpha) * Q2[action_agent_2, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]] \
                    + alpha * (reward_agent_2 + Gamma * np.max(Q2[:, next_s2[0], next_s2[1], next_s2[2], next_s2[3], next_s2[4], next_s2[5]]))
                # Update state
                s1 = next_s1
                s2 = next_s2
                # Update the total reward
                R_traj_agent_1 += reward_agent_1
                R_traj_agent_2 += reward_agent_2
                cumu_discount *= Gamma
                # update the done flag
                if dones[0]:
                    assert dones[1]
                    done = True
                    coop_num_cnt = infos['coop&coop_num']
                    collective_reward = infos['collective_return']
                else:
                    done = False
                steps_cnt += 1
                if steps_cnt >= max_step:
                    coop_num_cnt = infos['coop&coop_num']
                    collective_reward = infos['collective_return']
                    break
            # append the logging statistics
            total_reward_episodes.append([episode + 1, R_traj_agent_1, R_traj_agent_2, steps_cnt, coop_num_cnt, collective_reward, epsilon])
            print('Episode: {0} \t Reward: {1} \t Total Steps: {2}'.format(episode+1, collective_reward, steps_cnt))

            # generate final policy based on Q-table
            for x1 in range(gw_size):
                for y1 in range(gw_size):
                    for x2 in range(gw_size):
                        for y2 in range(gw_size):
                            for t1 in range(gw_size):
                                for t2 in range(gw_size):
                                    q1_best = np.max(Q1[:, x1, y1, x2, y2, t1, t2])
                                    q2_best = np.max(Q2[:, x2, y2, x1, y1, t1, t2])
                                    a1 = np.random.choice(np.where(Q1[:, x1, y1, x2, y2, t1, t2] == q1_best)[0])
                                    a2 = np.random.choice(np.where(Q2[:, x2, y2, x1, y1, t1, t2] == q2_best)[0])
                                    policy1[x1, y1, x2, y2, t1, t2] = a1
                                    policy2[x2, y2, x1, y1, t1, t2] = a2

        # save statistics in dataframe
        total_reward_episodes_df = pd.DataFrame(total_reward_episodes, columns=['episode', 'reward1', 'reward2', 'length', 'coop', 'totalreward', 'epsilon'])

        return Q1, Q2, policy1, policy2, total_reward_episodes_df

    def HysQLearning(self, nEpisodes, epsilon=0.1, alpha=0.1, beta=0.01, max_step=10000):
        # get the number of action
        nAction = self.env.action_space[0].n
        gw_size = self.env.length
        n_agents = self.env.num_agents
        assert n_agents == 2
        # init the Q-table, since we have 2 agents, we need to assign a Q-table for each one of them
        Q1 = np.zeros([nAction, gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        Q2 = np.zeros([nAction, gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        # init the policy table
        policy1 = np.zeros([gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        policy2 = np.zeros([gw_size, gw_size, gw_size, gw_size, gw_size, gw_size])
        # Save the return value for each episode
        total_reward_episodes = []

        # start training
        for episode in range(nEpisodes):
            # reset the env
            init_observ = env.reset()
            # get the position for agent1, agent2 and escalation(target)
            s1 = init_observ[0]
            s2 = init_observ[1]
            # reset the done flag
            done = False
            steps_cnt = 0
            R_traj_agent_1 = 0.0
            R_traj_agent_2 = 0.0
            coop_num_cnt = 0
            collective_reward = 0
            cumu_discount = 1.0
            while not done:
                # choose A from S using policy derived from Q
                behavior_policy_agent_1 = EpsilonGreedyPolicy(nAction, gw_size, Q1, epsilon)
                behavior_policy_agent_2 = EpsilonGreedyPolicy(nAction, gw_size, Q2, epsilon)
                action_agent_1 = behavior_policy_agent_1.action(s1)
                action_agent_2 = behavior_policy_agent_2.action(s2)
                # convert agent action to 1-hot format
                action_agent_1_onehot = np.zeros(nAction)
                action_agent_1_onehot[action_agent_1] = 1
                action_agent_2_obehot = np.zeros(nAction)
                action_agent_2_obehot[action_agent_2] = 1
                # Take action A, observe R, S'
                observations, rewards, dones, infos = self.env.step([action_agent_1_onehot, action_agent_2_obehot])
                # Get the next state for each agent
                next_s1 = observations[0]
                next_s2 = observations[1]
                # get the reward for each agent
                reward_agent_1 = rewards[0]
                reward_agent_2 = rewards[1]
                # Update Q(S,A)
                TD_error_agent_1 = reward_agent_1 + \
                    Gamma * np.max(Q1[:, next_s1[0], next_s1[1], next_s1[2], next_s1[3], next_s1[4], next_s1[5]]) - \
                    Q1[action_agent_1, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5]]
                if TD_error_agent_1 >= 0.0:
                    Q1[action_agent_1, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5]] += alpha * TD_error_agent_1
                else:
                    Q1[action_agent_1, s1[0], s1[1], s1[2], s1[3], s1[4], s1[5]] += beta * TD_error_agent_1
                # Update Q2(S,A)
                TD_error_agent_2 = reward_agent_2 + \
                    Gamma * np.max(Q2[:, next_s2[0], next_s2[1], next_s2[2], next_s2[3], next_s2[4], next_s2[5]]) - \
                    Q2[action_agent_2, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]]
                if TD_error_agent_2 >= 0.0:
                    Q2[action_agent_2, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]] += alpha * TD_error_agent_2
                else:
                    Q2[action_agent_2, s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]] -= beta * TD_error_agent_2
                # Update state
                s1 = next_s1
                s2 = next_s2
                # Update the total reward
                R_traj_agent_1 += reward_agent_1
                R_traj_agent_2 += reward_agent_2
                cumu_discount *= Gamma
                # update the done flag
                if dones[0]:
                    assert dones[1]
                    done = True
                    coop_num_cnt = infos['coop&coop_num']
                    collective_reward = infos['collective_return']
                else:
                    done = False
                steps_cnt += 1
                if steps_cnt >= max_step:
                    coop_num_cnt = infos['coop&coop_num']
                    collective_reward = infos['collective_return']
                    break
            # append the logging statistics
            total_reward_episodes.append([episode + 1, R_traj_agent_1, R_traj_agent_2, steps_cnt, coop_num_cnt, collective_reward, epsilon])
            print('Episode: {0} \t Reward: {1} \t Total Steps: {2}'.format(episode+1, collective_reward, steps_cnt))

            # generate final policy based on Q-table
            for x1 in range(gw_size):
                for y1 in range(gw_size):
                    for x2 in range(gw_size):
                        for y2 in range(gw_size):
                            for t1 in range(gw_size):
                                for t2 in range(gw_size):
                                    q1_best = np.max(Q1[:, x1, y1, x2, y2, t1, t2])
                                    q2_best = np.max(Q2[:, x2, y2, x1, y1, t1, t2])
                                    a1 = np.random.choice(np.where(Q1[:, x1, y1, x2, y2, t1, t2] == q1_best)[0])
                                    a2 = np.random.choice(np.where(Q2[:, x2, y2, x1, y1, t1, t2] == q2_best)[0])
                                    policy1[x1, y1, x2, y2, t1, t2] = a1
                                    policy2[x2, y2, x1, y1, t1, t2] = a2

        # save statistics in dataframe
        total_reward_episodes_df = pd.DataFrame(total_reward_episodes, columns=['episode', 'reward1', 'reward2', 'length', 'coop', 'totalreward', 'epsilon'])

        return Q1, Q2, policy1, policy2, total_reward_episodes_df


if __name__ == '__main__':

    # Init the environment
    args = get_config()
    print("Using Environment: {}".format(args.env_name))

    # Since we only consider a case where we only has 2 agents
    assert args.num_agents == 2
    env = GridWorldEnv(args)
    print("Number of Agents: {}".format(env.num_agents))

    rl = ReinforcementLearning(env=env)

    epsilon_set = [0.1]
    alpha_set = [0.1]
    beta_set = [0.01]
    repeat_num = 1

    if args.algorithm_name == 'Q':
        print("Using algorithm Q-Learning ...")
        # Test Q-Learning
        data_results_Q = pd.DataFrame()
        for epsilon in epsilon_set:
            for alpha in alpha_set:
                for beta in beta_set:
                    for i in range(repeat_num):
                        Q1, Q2, P1, P2, episodes_results = rl.QLearning(nEpisodes=NUM_EPISODE, epsilon=epsilon, alpha=alpha)
                        # concat dataframe
                        data_results_Q = pd.concat([data_results_Q, episodes_results], ignore_index=True)
        # save the statistical result to file
        data_results_Q.to_csv('Decentralized_QL.csv')
        if args.plot_reward:
            # draw the reward
            sns.relplot(
                data=data_results_Q, x="episode", y="totalreward",
                hue="epsilon", kind="line", ci=None
            )
            plt.show()

    elif args.algorithm_name == 'HysQ':
        print("Using algorithm Hysteretic Q-Learning ...")
        # Test Hysteretic Q-Learning
        data_results_hysQ = pd.DataFrame()
        for epsilon in epsilon_set:
            for alpha in alpha_set:
                for beta in beta_set:
                    for i in range(repeat_num):
                        Q1, Q2, P1, P2, episodes_results = rl.HysQLearning(nEpisodes=NUM_EPISODE, epsilon=epsilon, alpha=alpha, beta=beta)
                        # concat dataframe
                        data_results_hysQ = pd.concat([data_results_hysQ, episodes_results], ignore_index=True)
        # save the statistical result to file
        data_results_hysQ.to_csv('Hysteretic_QL.csv')
        if args.plot_reward:
            # draw the reward
            sns.relplot(
                data=data_results_hysQ, x="episode", y="totalreward",
                hue="epsilon", kind="line", ci=None
            )
            plt.show()

    else:
        print("Currently not support this algorithm: {}".format(args.algorithm_name))
