import copy
import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from envs import GridWorldEnv, MGEnv, Agent
from config import get_config

Reward_returned=[]
Terminated_returned=[]
Info_returned=[]

args = get_config()
# print(args.env_name)

assert args.num_agents == 2, ("only 2 agens are supported")
env = GridWorldEnv(args)




'''Definition of Action see AAs '''
# 共五个action(0~1) 用一个array 表示， example: action 1--------------->[0,1,0,0,0]
'''
AGENT_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY'  # don't move
                }  # Rotate clockwise

ACTIONS = {'MOVE_LEFT': [0, -1],  # Move left
           'MOVE_RIGHT': [0, 1],  # Move right
           'MOVE_UP': [-1, 0],  # Move up
           'MOVE_DOWN': [1, 0],  # Move down
           'STAY': [0, 0]  # don't move
           }

'''
AAs=[]
AA=np.zeros(5)
AA[1]=1
AAs.append(AA)
AA1=np.zeros(5)
AA1[0]=1
AAs.append(AA1)


state, reward, done, infos = env.step(AAs)

print(state)  #observation ----------> 格式： Array1：Agent0 位置（前两位）， Agent1 位置（中两位）， Escalation 位置（后两位）
#                                               Array2： Agent1 位置（前两位）， Agent0 位置（中两位）， Escalation 位置（后两位）
print(reward)   #[Agent0, Agent1]

# print(done) # 这个没找到

print(infos)


AAs1=[]
AA1=np.zeros(5)
AA1[3]=1
AAs1.append(AA1)
AA11=np.zeros(5)
AA11[3]=1
AAs1.append(AA11)
state, reward, done, infos = env.step(AAs1)
print(state)
print(reward)
print(infos)
# print(state)
# print(np.zeros(env.action_space[0].n))
# print((env.action_space[1].n))


# print(AAs)



# AA.append([0,0,0,0,1])
# print(AA[7])
