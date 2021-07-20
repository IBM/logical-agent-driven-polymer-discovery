
import gym
import numpy as np

from gym import Env, spaces


class DirectAgentDrivenPolymerDiscovery(Env):

    def __init__(self, config_files, n_initial=50, max_steps=100, n_targets=5, 
                       grid_size=7, step_size=3, rewards=None):
        """
        Wrapper environment for direct features for Agent Driven Polymer Discovery environment

        :param config_files: (str) path to csv file with environment config file details
        :param n_initial: (int) number of initial known reactions
        :param max_steps: (int) maximum steps allowed in an episode
        :param n_targets: (int) number of targets needed to be visited in an episode
        :param grid_size: (int) grid size for observation around current reaction or cell in lattice/grid
        :param step_size: (int) maximum step size for controllable parameters
        :param rewards: (dict) dictionary with values for various types of rewards
        """
        self.env = gym.make(id="md_envs:eADPD-v1", 
                            config_files=config_files,
                            n_initial=n_initial,
                            max_steps=max_steps,
                            n_targets=n_targets,
                            rewards=rewards)

        self.grid_size = grid_size
        self.step_size = step_size
        self.visited_reactions = set()
        self.current_reaction = None

        self.action_space = spaces.Discrete(4*self.step_size)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.grid_size, self.grid_size, 2), dtype=np.float32)

        self.reset()

    def reset(self, return_infos=False):
        obs, infos = self.env.reset()

        self.visited_reactions = dict([(_[0], _[1]) for _ in infos['prev_reactions']])
        self.current_reaction = infos['prev_reactions'][-1][0]

        direct_obs = self.get_observation()
        if return_infos:
            return direct_obs, infos

        return direct_obs

    def get_observation(self):
        data_1 = np.ones((self.grid_size, self.grid_size)) * -1
        data_2 = np.zeros((self.grid_size, self.grid_size))
        if self.grid_size == 20:
            for i, j in self.visited_reactions.items():
                if 0 <= i[0] < 20 and 0 <= i[1] < 20:
                    data_1[i] = j
                    data_2[i] = 0.5
            if 0 <= self.current_reaction[0] < 20 and 0 <= self.current_reaction[1] < 20:
                data_2[self.current_reaction] = 1
        else:
            grid_half = self.grid_size//2
            for i_, i in enumerate(range(-grid_half, -grid_half+self.grid_size)):
                for j_, j in enumerate(range(-grid_half, -grid_half+self.grid_size)):
                    ri, rj = self.convert_reaction((self.current_reaction[0]+i, self.current_reaction[1]+j))
                    data_1[i_, j_] = self.visited_reactions.get((ri, rj), -1)
                    if 0 <= ri < 20 and 0 <= rj < 20:
                        if (ri, rj) in self.visited_reactions:
                            data_2[i_, j_] = 0.5
                    else:
                        data_2[i_, j_] = -1
            data_2[grid_half, grid_half] = 1

        obs = np.zeros((self.grid_size, self.grid_size, 2))
        obs[:, :, 0] = data_1
        obs[:, :, 1] = data_2
        return obs

    def convert_reaction(self, reaction):
        rx, ry = reaction
        if rx < 0:
            rx = 20 + rx
        elif rx > 19:
            rx = rx % 20
        if ry < 0:
            ry = 20 + ry
        elif ry > 19:
            ry = ry % 20
        return (rx, ry)

    def step(self, action):
        action_direction = action // self.step_size
        action_step = action % self.step_size + 1
        if action_direction == 0:
            next_reaction = (self.current_reaction[0] - action_step, self.current_reaction[1])
        elif action_direction == 1:
            next_reaction = (self.current_reaction[0], self.current_reaction[1]+action_step)
        elif action_direction == 2:
            next_reaction = (self.current_reaction[0] + action_step, self.current_reaction[1])
        else:
            next_reaction = (self.current_reaction[0], self.current_reaction[1]-action_step)

        next_reaction = self.convert_reaction(next_reaction)

        obs, rewards, dones, infos = self.env.step(next_reaction)
        self.visited_reactions[infos['prev_reactions'][-1][0]] = infos['prev_reactions'][-1][1]
        self.current_reaction = infos['prev_reactions'][-1][0]

        obs_ = self.get_observation()
        return obs_, rewards, dones, infos

    def render(self, delay=0):
        self.env.render(delay)

