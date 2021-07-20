
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN, PPO2

import argparse
import gym
import numpy as np
import pandas as pd

from tqdm import tqdm

from policies import CustomMlpPolicy_1, CustomMlpPolicy_2
from policies import CustomLstmMlpPolicy_2, CustomLstmMlpPolicy_3
from policies import CustomCnnPolicy, CustomLstmCnnPolicy

def test_model(model, env, num_epochs=100, max_steps=100, vectorized_env=False, csv_file="steps_vs_targets.csv"):

    log_rewards = []
    log_targets = []
    log_target_steps = []
    log_first_target_steps = []
    log_target_perc = []
    log_valid_perc = []
    log_revisit_perc = []
    log_invalid_perc = []
    log_episode_lengths = []
    log_target_steps = []
    log_targets_per_step = {}
    for i in range(max_steps):
        log_targets_per_step[i+1] = []

    for episode in tqdm(range(num_epochs)):
        logical_state = env.reset()
        r = 0
        targets, valids, revisits, invalids = 0, 0, 0, 0
        first_target_steps = None
        target_steps = []
        episode_length = 0
        for i in range(max_steps):
            episode_length += 1
            action, _states = model.predict(logical_state)
            logical_state, reward, done, infos = env.step(action)
            if vectorized_env == True:
                reward, done = reward[0], done[0]

            r += reward
            if reward >= 1:
                targets += 1
                target_steps.append(i+1)
            elif reward > -0.2:
                valids += 1
            elif reward == -0.5:
                revisits += 1
            elif reward == -1:
                invalids += 1
            log_targets_per_step[i+1].append(targets)
            if done:
                break

        log_rewards.append(r)
        log_targets.append(targets)
        log_target_perc.append(targets*100.0/episode_length)
        log_valid_perc.append(valids*100.0/episode_length)
        log_revisit_perc.append(revisits*100.0/episode_length)
        log_invalid_perc.append(invalids*100.0/episode_length)
        log_episode_lengths.append(episode_length)
        log_target_steps.append(target_steps)

    print("\n" + "##" * 30)
    print("Avg Reward: {:.2f}".format(np.mean(log_rewards)))
    print("Avg episode length: {:.2f}".format(np.mean(log_episode_lengths)))
    print("Avg # of targets reached: {:.2f}".format(np.mean(log_targets)))
    first_target = [_[0] for _ in log_target_steps if len(_) > 0]
    print("Avg # of steps to find first target: {:.2f}".format(np.mean(first_target)))
    print("\n\n")
    print("Avg % of target actions: {:.2f}".format(np.mean(log_target_perc)))
    print("Avg % of valid actions: {:.2f}".format(np.mean(log_valid_perc)))
    print("Avg % of revisit actions: {:.2f}".format(np.mean(log_revisit_perc)))
    print("Avg % of no go actions: {:.2f}".format(np.mean(log_invalid_perc)))
    print("##" * 30)

    df = pd.DataFrame()
    df['steps'] = list(range(1, max_steps+1))
    df['Avg Targets'] = [np.mean(log_targets_per_step[i]) for i in range(1, max_steps+1)]
    df.to_csv(csv_file, index=False)


def train_dqn(train_env, custom_policy, gamma, learning_rate, 
              buffer_size, batch_size, training_timesteps, model_file):

    model = DQN(custom_policy, 
                train_env, 
                gamma=gamma, 
                learning_rate=learning_rate, 
                buffer_size=buffer_size, 
                batch_size=batch_size)

    model.learn(total_timesteps=training_timesteps, 
                log_interval=100)

    model.save(model_file)


def train_ppo(train_env, custom_policy, gamma, n_steps, learning_rate,
              training_timesteps, model_file):

    model = PPO2(custom_policy, 
                 train_env, 
                 gamma=gamma, 
                 n_steps=n_steps, 
                 learning_rate=learning_rate, 
                 nminibatches=1)

    model.learn(total_timesteps=training_timesteps, 
                log_interval=100)

    model.save(model_file)

def get_environment(feature_type, config_files, max_steps, algorithm):

    if algorithm not in ['dqn', 'ppo']:
        raise ValueError("algorithm is not in {dqn, ppo}")

    if feature_type == 'direct':
        env = gym.make(id="md_envs:direct-eADPD-v1",
                       config_files=config_files,
                       max_steps=max_steps,
                       rewards={"target": 1, "valid": -0.01, "revisit": -0.5, "no-go": -1})

        policy = CustomCnnPolicy if algorithm == 'dqn' else CustomLstmCnnPolicy
        env = env if algorithm == 'dqn' else DummyVecEnv([lambda: _ for _ in [env]])

    elif feature_type == 'logical':
        env = gym.make(id="md_envs:logical-eADPD-v1",
                       config_files=config_files,
                       max_steps=max_steps,
                       regressor_type=None,
                       rewards={"target": 1, "valid": -0.01, "revisit": -0.5, "no-go": -1})

        policy = CustomMlpPolicy_1 if algorithm == 'dqn' else CustomLstmMlpPolicy_2
        env = env if algorithm == 'dqn' else DummyVecEnv([lambda: _ for _ in [env]])

    elif feature_type == 'logical_with_regressor':
        env = gym.make(id="md_envs:logical-eADPD-v1",
                       config_files=config_files,
                       max_steps=max_steps,
                       rewards={"target": 1, "valid": -0.01, "revisit": -0.5, "no-go": -1})

        policy = CustomMlpPolicy_2 if algorithm == 'dqn' else CustomLstmMlpPolicy_3
        env = env if algorithm == 'dqn' else DummyVecEnv([lambda: _ for _ in [env]])
    else:
        raise ValueError("feature_type is not in {direct, logical, logical_with_regressor}")

    return env, policy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="python scripts/main.py", 
                                     description="scripts for Agent Driven Polymer Discovery environment")
    subparsers = parser.add_subparsers(dest="sub_command", help="sub-command help")


    parser_train = subparsers.add_parser("train", help="train a model")
    parser_train.add_argument('-f', "--feature_type", required=True,
                                    help="direct, logical or logical_with_regressor", metavar='')
    parser_train.add_argument('-a', "--algorithm", required=True, help="dqn or ppo", metavar='')
    parser_train.add_argument('-o', "--output_model", required=True, help="output model file", metavar='')

    parser_test = subparsers.add_parser('test', help="test a trained model")
    parser_test.add_argument('-f', "--feature_type", required=True,
                                    help="direct, logical or logical_with_regressor", metavar='')
    parser_test.add_argument('-a', "--algorithm", required=True, help="dqn or ppo", metavar='')
    parser_test.add_argument('-m', "--model", required=True, help="path to trained model file", metavar='')
    parser_test.add_argument('-r', "--results_file", 
                                   required=False, 
                                   help="output csv file for steps vs targets results. default steps_vs_targets.csv", 
                                   metavar='')

    args = parser.parse_args()

    if args.sub_command == "train":
        feature_type = args.feature_type.lower()
        algorithm = args.algorithm.lower()
        output_model = args.output_model

        env, policy = get_environment(feature_type, "./data/polymerDiscovery/train.csv", 200, algorithm)

        if algorithm == 'dqn' and feature_type == 'direct':
            train_dqn(env, policy, 0.8, 0.003, 20000, 64, 10000, output_model)

        elif algorithm == 'dqn' and feature_type == 'logical':
            train_dqn(env, policy, 0.8, 0.0003, 20000, 32, 500000, output_model)

        elif algorithm == 'dqn' and feature_type == 'logical_with_regressor':
            train_dqn(env, policy, 0.8, 0.0003, 20000, 32, 500000, output_model)

        elif algorithm == 'ppo' and feature_type == 'direct':
            train_ppo(env, policy, 0.8, 512, 0.003, 10000, output_model)

        elif algorithm == 'ppo' and feature_type == 'logical':
            train_ppo(env, policy, 0.8, 512, 0.003, 100000, output_model)

        elif algorithm == 'ppo' and feature_type == 'logical_with_regressor':
            train_ppo(env, policy, 0.8, 2048, 0.003, 500000, output_model)

    if args.sub_command == "test":
        feature_type = args.feature_type.lower()
        algorithm = args.algorithm.lower()
        model = args.model
        results_file = args.results_file
        results_file = results_file if results_file else "steps_vs_targets.csv"

        env, policy = get_environment(feature_type, "./data/polymerDiscovery/test.csv", 100, algorithm)

        if algorithm == 'dqn':
            model = DQN.load(model)
        else:
            model = PPO2.load(model)

        vectorized_env = False if algorithm == 'dqn' else True

        test_model(model, env, num_epochs=100, max_steps=100, vectorized_env=vectorized_env, csv_file=results_file)
