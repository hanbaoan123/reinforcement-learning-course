# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:22:21 2019

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
def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    蒙特卡洛预测算法：使用采样计算给定策略的值函数.
    
    参数:
        策略:将观察映射成动作概率的函数.
        env: OpenAI gym环境.
        num_episodes: 采样的迭代次数.
        discount_factor: Gamma折扣因子.
    
    返回值:
        值函数V.
    """

    # 跟踪每个状态的次数和回报以计算平均，可以使用一个数组来保存所有
    # 返回(如书中所示)，但这是内存效率低下的.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    # Implement this!
    for i_episode in range(1,num_episodes+1):
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
            action=policy(state)
            next_state,reward,done,_=env.step(action)
            episode.append((state,action,reward))
            if done :
                break
            state=next_state
        #寻找当前代已经访问过的所有状态
        #将每个状态转换成tuple以用作字典（通过set将重复的元素去除）
        states_in_episode=set([tuple(x[0])for x in episode])
        for state in states_in_episode:
            #在当前代state第一次出现的位置
            first_occurence_idx=next(i for i,x in enumerate(episode) if x[0]==state)
            #对从第一次出现开始的所有奖励进行求和
            G=sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            #计算该状态在所有采样代数上的平均回报
            returns_sum[state]+=G
            returns_count[state]+=1.0
            V[state]=returns_sum[state]/returns_count[state]
    return V

def sample_policy(observation):
    """
    采样策略：如果玩家得分>20,那么停牌，否则叫牌.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")