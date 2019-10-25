# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:09:51 2019

@author: hba
"""



import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from Lib.envs.blackjack import BlackjackEnv
from Lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q,epsilon,nA):
    """
    基于给定的Q值函数和epsilon创建epsilon贪婪策略
    参数：
        Q:将状态映射成动作值函数的字典。每个值都是一个长度为nA的numpy数组。
        epsilon：选择随机动作的概率，为0与1之间的浮点数。
        nA：环境中的动作数。
    返回值：
        返回一个函数，该函数的输入为观察即状态，并以numpy数组（长度为nA）的形式返回每个动作的概率。
    """
    def policy_fn(obseration):
        A=np.ones(nA,dtype=float)*epsilon/nA
        best_action=np.argmax(Q[obseration])
        A[best_action]+=(1.0-epsilon)
        return A
    return policy_fn


def mc_control_epsilon_greedy(env,num_episodes,discount_factor=1.0,epsilon=0.1):
    """
    使用epsilon贪婪策略进行蒙特卡洛控制，寻找最优的epsilon贪婪策略。
    参数：
        env：OpenAI gym环境。
        num_eposides:采样代数。
        discount_factor:Gamma折扣因子。
        epsilon:选择随机动作的概率，为0与1之间的浮点数。
    返回值：
        返回一个元组(Q,policy).Q是将状态映射成动作值函数的字典，policy是一个函数，以obseration为输入，返回动作的概率
    """
    # 跟踪每个状态的次数和回报以计算平均，可以使用一个数组来保存所有
    # 返回(如书中所示)，但这是内存效率低下的.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    #最终的动作值函数
    #为嵌套字典，将state映射为(action -> action-value)
    Q=defaultdict(lambda:np.zeros(env.action_space.n))
    
    #遵循的策略
    policy=make_epsilon_greedy_policy(Q,epsilon,env.action_space.n)
    
    for i_episode in range(1,num_episodes+ 1):
        #打印当前代，用于调试
        if i_episode%1000==0:
            print("\rEpisode {}/{}.".format(i_episode,num_episodes),end="")
            sys.stdout.flush()
        #产生一代样本数据
        #一代是由tuple(state,action,reward)组成的array数列
        episode=[]
        #重置得到初始状态
        state=env.reset()
        for t in range(100):
            probs=policy(state)
            action=np.random.choice(np.arange(len(probs)),p=probs)
            next_state,reward,done,_=env.step(action)
            episode.append((state,action,reward))
            if done:
                break
            state=next_state
            
        #寻找当前代中所有已访问的状态-动作对
        #将每个状态转换成tuple以用作字典（通过set将重复的元素去除）
        sa_in_episode=set([(tuple(x[0]),x[1])for x in episode])
        for state,action in sa_in_episode:
            sa_pair=(state,action)
            #在当前代中寻找状态-动作对(state,action)第一次出现的位置
            first_occurence_idx=next(i for i,x in enumerate(episode) if x[0]==state and x[1]==action)
            #对从第一次出现开始的所有奖励进行求和
            G=sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            #计算该状态在所有采样代数上的平均回报
            returns_sum[sa_pair]+=G
            returns_count[sa_pair]+=1.0
            Q[state][action]=returns_sum[sa_pair]/returns_count[sa_pair]
            
            #通过修改Q字典，可以隐式地改进策略
    return Q,policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="最优值函数")