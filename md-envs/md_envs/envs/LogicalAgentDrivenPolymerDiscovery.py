
import GPy
import gym
import numpy as np

from gym import Env, spaces
from sklearn.preprocessing import MinMaxScaler


def euclidean_dist(reaction_1, reaction_2):
    return np.sum([(x-y)**2 for x, y in zip(reaction_1, reaction_2)])**0.5


class LogicalAgentDrivenPolymerDiscovery(Env):

    def __init__(self, config_files, n_initial=50, max_steps=100, n_targets=5, 
                       n_random_samples=20, regressor_type='GPy', 
                       regressor_train_interval=5, rewards=None):

        """
        Wrapper environment for logical features and logical features with regressor
        for Agent Driven Polymer Discovery environment

        :param config_files: (str) path to csv file with environment config file details
        :param n_initial: (int) number of initial known reactions
        :param max_steps: (int) maximum steps allowed in an episode
        :param n_targets: (int) number of targets needed to be visited in an episode
        :param n_random_samples: (int) number of randomly proposed reactions at each step
        :param regressor_type: (int) type of internal regressor. Default 'GPy'. 
                                     None for logical features without regressor.
        :param regressor_train_interval: (int) number of steps before retraining internal regressor 
        :param rewards: (dict) dictionary with values for various types of rewards
        """

        self.env = gym.make(id="md_envs:eADPD-v1", 
                            config_files=config_files,
                            n_initial=n_initial,
                            max_steps=max_steps,
                            n_targets=n_targets,
                            rewards=rewards)

        self.regressor_type = regressor_type
        self.regressor_train_interval = regressor_train_interval
        self.n_random_samples = n_random_samples

        self.current_reaction = None
        self.visited_reactions = set()
        self.target = None
        self.regressor = None
        self.n_steps = 0

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.reset()

    def reset(self, return_infos=False):
        obs, infos = self.env.reset()

        self.visited_reactions = set([_[0] for _ in infos['prev_reactions']])
        self.target = np.mean(infos['properties'][0]['target'][0])
        self.current_reaction = infos['prev_reactions'][-1][0] if len(self.visited_reactions) > 0 else None
        self.n_steps = 0

        self.proposed_actions = self.propose_random_reactions()
        pred, std_dev = [0 for _ in self.proposed_actions], [0 for _ in self.proposed_actions]

        if self.regressor_type == "GPy":
            prev_reactions_x = np.array([list(_[0]) for _ in infos['prev_reactions']])
            prev_reactions_y = np.array([-1 if _[1] is None else _[1] for _ in infos['prev_reactions']]).reshape(-1, 1)
            k = GPy.kern.RBF(2, 2**-4)
            self.regressor = GPy.models.GPRegression(prev_reactions_x, prev_reactions_y, kernel=k)
            self.regressor.optimize('bfgs')

            pred, std_dev = self.regressor.predict(np.array([list(_) for _ in self.proposed_actions]))
            pred = [_[0] for _ in pred]
            std_dev = [_[0] for _ in std_dev]

        logical_obs = self.get_logical_observation([(tuple(r), p, s) for r, p, s in zip(self.proposed_actions, pred, std_dev)])

        if return_infos:
            return logical_obs, infos

        return logical_obs

    def get_logical_observation(self, proposed_reactions):
        # For GPy regressor, return logical vector with features  - 
        #                         ['visited', 'closer', 'confident', 'similar', 'feasible']
        # For no regressor, return logical vector with features -
        #                         ['visited', 'similar', 'feasible'] 
        #
        # for all proposed reactions

        logical_states = []
        for i, p_reaction in enumerate(proposed_reactions):
            reaction, r_output, r_err = p_reaction
            visited = 1 if reaction in self.visited_reactions else 0
            output_target_diff = abs(r_output - self.target)
            mean_dist = np.min([euclidean_dist(reaction, _) for _ in self.visited_reactions])
            feasible = 1 if 0 <= reaction[0] < 20 and 0 <= reaction[1] < 20 else 0
            logical_states.append([visited, output_target_diff, r_err, mean_dist, feasible])

        logical_states.append([0, 0, 0, 0, 0])
        scaler = MinMaxScaler()
        logical_states = scaler.fit_transform(logical_states)[:-1, :]
        logical_states[:, 1] = 1 - logical_states[:, 1]
        logical_states[:, 2] = 1 - logical_states[:, 2]
        logical_states[:, 3] = 1 - logical_states[:, 3]

        if self.regressor_type is None:
            logical_states = logical_states[:, [0, 3, 4]]

        return logical_states.reshape(-1,)


    def step(self, action):
        obs, rewards, dones, infos = self.env.step(self.proposed_actions[action])
        self.visited_reactions.add(infos['prev_reactions'][-1][0])
        self.current_reaction = infos['prev_reactions'][-1][0]
        self.n_steps += 1

        self.proposed_actions = self.propose_random_reactions()
        pred, std_dev = [0 for _ in self.proposed_actions], [0 for _ in self.proposed_actions]
        if self.regressor_type == "GPy" and self.n_steps%self.regressor_train_interval == 0:
            prev_reactions_x = np.array([list(_[0]) for _ in infos['prev_reactions']])
            prev_reactions_y = np.array([-1 if _[1] is None else _[1] for _ in infos['prev_reactions']]).reshape(-1, 1)
            try:
                self.regressor.set_XY(X=prev_reactions_x, Y=prev_reactions_y)
                self.regressor.optimize('bfgs')
            except Exception as e:
                print("Failed to optimize regressor with new data")
                pass

        if self.regressor_type == "GPy":
            try:
                pred, std_dev = self.regressor.predict(np.array([list(_) for _ in self.proposed_actions]))
                pred = [_[0] for _ in pred]
                std_dev = [_[0] for _ in std_dev]
            except Exception as e:
                print("Failed to predict property for proposed reactions")
                pass

        logical_obs = self.get_logical_observation([(tuple(r), p, s) for r, p, s in zip(self.proposed_actions, pred, std_dev)])
        return logical_obs, rewards, dones, infos

    def render(self, delay=0):
        self.env.render(delay)

    def propose_random_reactions(self):
        actions = set()
        while len(actions) < self.n_random_samples:
            actions.add(tuple([_ for _ in self.env.action_space.sample()]))
        return list(actions)

    def get_observation_space(self):
        n_actions = self.n_random_samples
        n_predicates = 3 if self.regressor_type is None else 5
        return spaces.Box(low=0.0, high=1.0, shape=(n_actions * n_predicates,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Discrete(self.n_random_samples)

