
import copy
import io
import json
import logging
import random
import time

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym import Env, spaces


class AgentDrivenPolymerDiscovery(Env):

    def __init__(self, config_files, n_initial=50, max_steps=100, n_targets=5, rewards=None, seed=None, log_level=logging.INFO):
        """
        Agent Driven Polymer Discovery environment

        :param config_files: (str) path to csv file with environment config file details
        :param n_initial: (int) number of initial known reactions
        :param max_steps: (int) maximum steps allowed in an episode
        :param n_targets: (int) number of targets needed to be visited in an episode
        :param rewards: (dict) dictionary with values for various types of rewards
        :param seed: (int) Seed for pseudo-random generators.
        """
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        base_path = '/'.join(config_files.split('/')[:-1])
        base_path = base_path + '/' if base_path else base_path
        df = pd.read_csv(config_files)
        df['config_file'] = base_path + df['config_file']
        df = df.sample(frac=1)

        self.all_env_name = []
        self.all_parameters = []
        self.all_properties = []
        self.all_no_go = []
        self.all_function = []
        self.all_function_data = []
        self.all_reward = []

        for env_name, cf in zip(df.env_name, df.config_file):
            cf_base_path = '/'.join(cf.split('/')[:-1])
            cf_base_path = cf_base_path + '/' if cf_base_path else cf_base_path
            with open(cf) as f:
                cf_data = json.load(f)
            self.all_env_name.append(env_name)
            self.all_parameters.append(cf_data['parameters'])
            self.all_properties.append(cf_data['properties'])
            if cf_data['no-go']['type'] == 'list':
                self.all_no_go.append(set([tuple(_) for _ in cf_data['no-go']['area']]))

            with open(cf_base_path + cf_data['function']['function_path'], 'rb') as f: 
                self.all_function.append(pickle.load(f))

            if cf_data['function']['data_type'] == 'npy':
                self.all_function_data.append(np.load(cf_base_path + cf_data['function']['data_path']))
            if rewards is None:
                self.all_reward.append(cf_data['reward'])
            else:
                self.all_reward.append(rewards)

        # Define action space based on the allowed values for the parameters of first environment
        # All environments in here are expected to have similar parameters
        self.n_initial = n_initial
        self.max_steps = max_steps
        self.n_targets = n_targets
        self.n_props = len(self.all_properties[0])
        self.n_params = len(self.all_parameters[0])

        lows = np.array([_["allowed"][0] for _ in self.all_parameters[0]])
        highs = np.array([_["allowed"][1] for _ in self.all_parameters[0]])
        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.int32)
        
        self.observation_space = spaces.Box(low=np.min(lows), 
                                            high=np.max(highs), 
                                            shape=(self.n_initial+self.max_steps, self.n_params+self.n_props), 
                                            dtype=np.float32)

        self.n_envs = len(self.all_env_name)
        self.current_env_id = None
        self.parameters, self.properties, self.no_go = None, None, None
        self.initial_reactions, self.initial_outputs = [], []
        self.visited_reactions, self.visited_outputs = [], []
        self.n_steps, self.target_steps = 0, 0
        self.obs = None

        self.all_target_cells = self.get_target_cells()

        self.reset()


    def reset(self):
        """
        Reset environment.

        :return: (np.ndarray) Feature representation of initial state of environment.
        """
        self.logger.debug("reset")

        self.current_env_id = 0 if self.current_env_id is None else (self.current_env_id+1) % self.n_envs

        self.parameters = self.all_parameters[self.current_env_id]
        self.properties = self.all_properties[self.current_env_id]
        self.no_go = self.all_no_go[self.current_env_id]
        self.reward = self.all_reward[self.current_env_id]
        self.initial_reactions, self.initial_outputs = [], []
        self.visited_reactions, self.visited_outputs = [], []
        self.n_steps, self.target_steps = 0, 0
        self.obs = np.zeros((self.n_initial+self.max_steps, self.n_params+self.n_props))

        c = 0
        while c < self.n_initial:
            random_reaction = tuple([_ for _ in self.action_space.sample()])
            if random_reaction in self.initial_reactions:
                continue
            elif random_reaction in self.no_go:
                r_output = None
            else:
                f_parameters = {"data": self.all_function_data[self.current_env_id], "parameters": random_reaction}
                r_output = self.all_function[self.current_env_id](**f_parameters)
                r_output = round(r_output, 4)
                if self.is_target(r_output):
                    continue
            self.initial_reactions.append(random_reaction)
            self.initial_outputs.append(r_output)
            c += 1

        for i, row in enumerate(list(zip(self.initial_reactions, self.initial_outputs))):
            r, o = row
            self.obs[i, :] = list(r) + [o]


        info = {'prev_reactions': list(zip(self.initial_reactions, self.initial_outputs)),
                'parameters': self.parameters,
                'properties': self.properties}

        return self.obs, info

    def step(self, action):
        """
        Run a single step on the environment with given action.

        :param action: tuple() reaction parameters

        :return: (float, float, bool, dict) observation, reward, status, info
        """
        self.logger.debug("step with action : {}".format(action))

        reward, r_output = self.execute(action)
        status = self.status()
        self.obs[self.n_initial+self.n_steps-1, :] = list(self.visited_reactions[-1]) + [self.visited_outputs[-1]]
        info = {'prev_reactions': list(zip(self.initial_reactions, self.initial_outputs)) + 
                                  list(zip(self.visited_reactions, self.visited_outputs)),
                'parameters': self.parameters,
                'properties': self.properties}
        return self.obs, reward, status, info

    def status(self):
        """
        Status of the episode

        :return: (bool) Whether the episode has ended or not
        """
        self.logger.debug("status")

        if self.n_steps == self.max_steps or self.target_steps == self.n_targets:
            return True

        return False

    def execute(self, action):
        """
        Execute action on the environment

        :param action: tuple() reaction parameters

        :return: (float) Reward for the action
        """
        self.logger.debug("execute")

        self.n_steps += 1
        self.visited_reactions.append(action)
        if action in self.no_go:
            self.visited_outputs.append(None)
            return self.reward['no-go'], None

        for parameter, action_i in zip(self.parameters, action):
            if parameter['type'] == 'continuous':
                allowed_min = parameter["allowed"][0]
                allowed_max = parameter["allowed"][1] + 1
                allowed_step = parameter["allowed"][2] 
                if action_i not in list(range(allowed_min, allowed_max, allowed_step)):
                    self.visited_outputs.append(None)
                    return self.reward['no-go'], None

        f_parameters = {"data": self.all_function_data[self.current_env_id], "parameters": action}
        r_output = self.all_function[self.current_env_id](**f_parameters)
        r_output = np.round(r_output, 4)

        self.visited_outputs.append(r_output)

        if action in self.initial_reactions or action in self.visited_reactions[:-1]:
            return self.reward['revisit'], r_output

        if self.is_target(r_output):
            self.target_steps += 1
            return self.reward['target'], r_output
        else:
            return self.reward['valid'], r_output


    def render(self, initial_reactions=True, trajectory=True, delay=0):
        """
        Render agent's action

        :initial_reactions: (bool) plot initial reactions with black '.'
        :trajectory: (bool) plot agent's trajectory with arrows
        :param delay: (int) Delay between action's of agent while rendering in seconds.
        """

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        im = ax.imshow(self.all_function_data[self.current_env_id])
        fig.colorbar(im, ax=ax)
        fig.canvas.draw()
        ax.scatter([_[1] for _ in self.all_target_cells[self.current_env_id]], 
                   [_[0] for _ in self.all_target_cells[self.current_env_id]], c='black', marker='*', s=150)
        ax.scatter([_[1] for _ in self.all_no_go[self.current_env_id]], 
                   [_[0] for _ in self.all_no_go[self.current_env_id]], c='black', marker='s', s=300)
        if initial_reactions:
            ax.scatter([_[1] for _ in self.initial_reactions], 
                       [_[0] for _ in self.initial_reactions], c='black', marker='.', s=75)
        if trajectory:
            for i in range(1, len(self.visited_reactions)):
                ax.annotate(s='', xy=self.visited_reactions[i][::-1], xytext=self.visited_reactions[i - 1][::-1],
                            arrowprops=dict(arrowstyle='->', color='white'))
                fig.canvas.draw()
                time.sleep(delay)

    def is_target(self, r_output):
        """
        Check if the given reaction output is a target property or not
        """

        if r_output is None:
            return False

        if self.properties[0]['target'][0][0] <= r_output <= self.properties[0]['target'][0][1]:
            return True

        else:
            return False

    def get_target_cells(self):
        """
        Get target cells for all given environments

        :return: list() list of list of target cells
        """
        all_target_cells = []
        for data, properties in zip(self.all_function_data, self.all_properties):
            target_min, target_max = properties[0]["target"][0][0], properties[0]["target"][0][1]
            target_cells = np.argwhere((data >= target_min) & (data <= target_max))
            target_cells = set([tuple(list(_)) for _ in target_cells])
            all_target_cells.append(target_cells)
        return all_target_cells
