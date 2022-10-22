"""
1. 构建马尔可夫决策过程
2. 计算动作-价值
3. 计算状态-价值
"""

# 导入工具函数:根据状态和行为生成操作相关字典的键，显示字典内容
from utils import display_dict
# 设置转移概率、奖励值以及读取它们方法
from utils import set_prob, set_reward, get_prob, get_reward
# 设置状态价值、策略概率以及读取它们的方法
from utils import set_pi, get_value, get_pi


"""
构建学生马尔科夫决策过程
    - 初始化状态、动作
    - 初始化状态转移概率 字典
    - 初始化奖励 字典
    - 初始化行为策略

"""
S = ['Phone', 'C1', 'C2', 'C3', 'rest'] # 状态
A = ['FB', 'Study', 'Quit', 'Pub', 'Sleep'] # 动作
R = {}  # 奖励Rsa字典
P = {}  # 状态转移概率Pss'a字典
gamma = 0.5  # 衰减因子

# 根据学生马尔科夫决策过程示例的数据设置状态转移概率和奖励，默认概率为1
# 注：概率为1：说明在某个状态下执行某个动作下一个行为一定是确定的；而概率不为1说明，在某个状态下执行某个动作下一个行为是不确定的
set_prob(P, S[0], A[0], S[0])  # 浏览手机中 - 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1])  # 浏览手机中 - 离开浏览 -> 第一节课
set_prob(P, S[1], A[0], S[0])  # 第一节课 - 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2])  # 第一节课 - 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3])  # 第二节课 - 学习 -> 第三节课
set_prob(P, S[2], A[4], S[4])  # 第二节课 - 退出学习 -> 退出休息
# set_prob(P, S[2], A[3], S[1], p=0.1)  # 第二节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[1], S[4])  # 第三节课 - 学习 -> 退出休息

# set_prob(P, S[3], A[3], S[2], p=0.4)  # 第三节课 - 泡吧 -> 第二节课
# set_prob(P, S[3], A[3], S[3], p=0.4)  # 第三节课 - 泡吧 -> 第三节课

set_reward(R, S[0], A[0], -1)  # 浏览手机中 - 浏览手机 ->-1
set_reward(R, S[0], A[2], 1)  # 浏览手机中 - 离开浏览 ->1
set_reward(R, S[1], A[0], 1)  # 第一节课 - 浏览手机 ->1
set_reward(R, S[1], A[1], -3)  # 第一节课 - 学习 ->-3
set_reward(R, S[2], A[1], -3)  # 第二节课 - 学习 ->-3
set_reward(R, S[2], A[4], 0)  # 第二节课 - 退出学习 ->0
set_reward(R, S[3], A[1], 12)  # 第三节课 - 学习 ->12
set_reward(R, S[3], A[3], +2)  # 第三节课 - 泡吧 ->2

MDP = (S, A, R, P, gamma)

print("----状态转移概率字典(矩阵)信息:----")
display_dict(P)
print("----奖励字典(函数)信息:----")
display_dict(R)

# 一个MDP中状态的价值是基于某一给定策略的，需要给定一个策略，在这里使用均匀随机策略:pi(a|.)=0.5
#
Pi = {}
set_pi(Pi, S[0], A[0], 0.5)  # 浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5)  # 浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5)  # 第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5)  # 第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5)  # 第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5)  # 第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5)  # 第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5)  # 第三节课 - 泡吧

print("----策略字典(函数)信息:----")
display_dict(Pi)
# 初始时价值为空，访问时会返回0
print("----价值字典(函数)信息:----")
# V = { 'Phone':10, 'C1':7, 'C2':9, 'C3':20, 'rest':0 }
V={}
display_dict(V)


def compute_q(MDP, V, s, a):
    """
    根据给定的MDP，价值函数V，计算状态行为对s,a的价值qsa
    :param MDP: MDP
    :param V: 价值函数
    :param s: 状态
    :param a: 动作
    :return: 动作价值
    """

    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
        q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa


def compute_v(MDP, V, Pi, s):
    """
    给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
    :param MDP:MDP
    :param V:价值函数
    :param Pi:策略
    :param s:状态
    :return: 状态价值
    """
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)
    return v_s
print('MDP:',MDP)
print('V',V)
print('S',S)
print('A',A)
print('compete q:-----------')
print('C1与Study',compute_q(MDP, V, 'C1', 'Study'))
print('C1与FB',compute_q(MDP, V, 'C1', 'FB'))
print('C1与Pub',compute_q(MDP, V, 'C1', 'Pub'))
print('C2与Pub',compute_q(MDP, V, 'C2', 'Pub'))
print('C3与Pub',compute_q(MDP, V, 'C3', 'Pub'))
print('compete v:-----------')


print('C1',compute_v(MDP, V, Pi, 'C1'))
print('C2',compute_v(MDP, V, Pi, 'C2'))
print('C3',compute_v(MDP, V, Pi, 'C3'))
print('rest',compute_v(MDP, V, Pi, 'rest'))