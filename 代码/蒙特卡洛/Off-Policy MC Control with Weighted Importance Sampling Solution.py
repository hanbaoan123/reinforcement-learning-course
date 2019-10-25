# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:52:35 2019

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

def create_random_policy(nA):
    """
    创建随机策略函数.
    
    参数:
        nA: 环境中动作的数量.
    
    返回值:
        返回一个函数，该函数以观察为输入，返回一个动作概率向量.
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    """
    基于Q值创建贪婪策略.
    
    参数:
        Q: 映射状态到动作值的字典
        
    返回值:
        返回一个函数，该函数以观察为输入，返回一个动作概率向量.
    """
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    使用加权重要度采样进行蒙特卡洛离策略控制，寻找最优贪婪策略.
    
    参数:
        env: OpenAI gym环境.
        num_episodes: 采样代数.
        behavior_policy: 生成各代样本数据时所遵循的行为策略，为给定观察下返回每个动作概率向量的函数.
        discount_factor: Gamma折扣因子.
    
    返回值:
        返回一个元组(Q, policy).
        Q是映射状态到动作值的字典.
        policy是一个以观察为输入，返回一个动作概率向量的函数，该策略为最优贪婪策略.
    """
    #最终的动作值函数
    #映射状态到动作值函数的字典
    Q=defaultdict(lambda:np.zeros(env.action_space.n))
    #加权重要性抽样公式的分母累积（贯穿于所有的代）
    C=defaultdict(lambda:np.zeros(env.action_space.n))
    
    #希望学习的贪婪策略即目标策略
    target_policy=create_greedy_policy(Q)
    
    for i_episode in range(1,num_episodes+1):
        #打印当前代，用于调试
        if i_episode%1000==0:
            print("\rEpisode {}/{}.".format(i_episode,num_episodes),end="")
            sys.stdout.flush()
            
        #生成一代样本数据
        #每一代是由元组（state,action,reward）组成的array数组
        episode=[]
        state=env.reset()
        for t in range(100):
            #从行为策略采样动作
            probs=behavior_policy(state)
            action=np.random.choice(np.arange(len(probs)),p=probs)
            next_state,reward,done,_=env.step(action)
            episode.append((state,action,reward))
            if done:
                break
            state=next_state
            
        #折扣回报总和
        G=0.0
        #重要度采样比（回报的权重）
        W=1.0
        #反向遍历当前代中所有的时间步(反向的目的是采样增量式求平均)
        for t in range(len(episode))[::-1]:
            state,action,reward=episode[t]
            #更新t时间步后的总奖励
            G=discount_factor*G+reward
            #更新加权重要性抽样公式的分母
            C[state][action]+=W
            #使用增量更新式更新动作值函数，同时也会改善引用Q的目标策略
            Q[state][action]+=(W/C[state][action])*(G-Q[state][action])
            #如果行为策略所采取的动作
            if action!=np.argmax(target_policy(state)):
                break
            W=W*1./behavior_policy(state)[action]
    return Q,target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)


# 输出：根据动作值函数，通过为每个状态选择最大的动作得到值函数
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="最优值函数")