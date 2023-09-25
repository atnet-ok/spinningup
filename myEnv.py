# https://developers.agirobots.com/jp/openai-gym-custom-env/
# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import pairwise_distances
import random
import gymnasium as gym
from gymnasium import spaces


# def sim_signal(paras):

#     theta = np.linspace(0,100*2*np.pi,3000)
#     signal = np.sin(paras[0]*theta )*np.sin(paras[1]*theta)
#     signal = signal + np.random.randn(3000)*paras[2]
#     return signal

def sim_signal(paras):

    signal = np.zeros(3)

    signal[0] = paras[0]*paras[1]
    signal[1] = np.sin(paras[1]*3/4*np.pi)
    signal[2] = ((paras[2]-1)**2)/2 -1

    return signal

# gym.Envを継承したEasyMazeクラス
class EasyMaze(gym.Env):

    
    def __init__(self):
        super(EasyMaze, self).__init__()

        self.max_data = 500
        self.obs_dim =10
        self.plot_interval = 40
        self.step_n = 0

        self.reward_history = []
        self.fig_s = []

        # 行動空間として0から3までの4種類の離散値を対象とする
        # ちなみに、0は"left"、1は"top"、2は”right”、3は"down"に対応させた
        self.action_space = gym.spaces.Box(
                -1.0, 1.0, (3,)
            )

        # 状態はエージェントが存在するセルの位置(12種類)
        self.observation_space = gym.spaces.Box(
                -1.0, 1.0, (10,)
            )
        
        # 即時報酬の値は0から1の間とした
        self.reward_range = (0, 2)

    def get_observation(self,state):

        observation = np.array(random.sample(state, self.obs_dim))

        return observation

    def get_reward(self,state,signal):

        distances = pairwise_distances(state, signal.reshape(1, -1),  metric='cosine')
        reward =  np.mean(distances)

        return reward

    def reset(self):
        # 迷路のスタート位置は"s0"とする
        self.state = []

        for i in range(self.obs_dim):
            action = np.random.random(3)*2 -1
            signal = sim_signal(action)
            self.state.append(signal)
        # 初期状態の番号を観測として返す

        return self.get_observation(self.state)

    def step(self, action):
        # 現在の状態と行動から次の状態に遷移

        signal = sim_signal(action)
        reward = self.get_reward(self.state, signal)
        self.state.append(signal)
        self.reward_history.append(reward)
        if self.step_n % self.plot_interval==0:
            self.render()


        # ゴール状態"s11"に遷移していたら終了したことをdoneに格納＆報酬1を格納
        # その他の状態ならdone=False, reward=0とする
        if len(self.state) == self.max_data:
            done = True
        else:
            done = False

        # 今回の例ではinfoは使用しない
        info = {}

        observation = self.get_observation(self.state)

        self.step_n += 1

        return observation, reward, done, info

    # 描画関連の処理を実施

    def render(self, mode=None):
        # matplotlibを用いて迷路を作成
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        x = np.array(self.state)[:,0]
        y = np.array(self.state)[:,1]
        z = np.array(self.state)[:,2]

        ax.scatter(x, y, z, c='b')

        ax_max = 1
        ax.set_xlim(-ax_max, ax_max)
        ax.set_ylim(-ax_max, ax_max)
        ax.set_zlim(-ax_max, ax_max)
        
        self.fig_s.append(fig)


if __name__ == '__main__':
    env = EasyMaze()
    obs = env.reset()

    for _ in range(env.max_data):
        action = env.action_space.sample()
        obs, re, done, info = env.step(action)

    for fig in env.fig_s:
        plt.show()
