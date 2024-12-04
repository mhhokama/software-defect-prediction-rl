import sys
import os

project_root = os.path.dirname(os.path.realpath(__file__))
stable_path = os.path.join(project_root, "stable-baselines3")
sys.path.append(stable_path)

import numpy as np
from env.feature_selection import FPOO
from env.classifier import ClassifierModel
import pandas as pd

from model.Transformer_policy import NullFeatureExtractor, CustomActorCriticPolicy
from model.Transformer_graph_policy import CustomGraphActorCriticPolicy


def make_agent(env, device = "auto", state = "simple", tb_log_dir = "PPO_tb", 
               input_dim = 3840, features_dim = 32, n_steps= 10, max_seq = 20, num_heads = 4):
    if "simple" in state:
        state = "simple"
    elif "custom" in state:
        state = "custom"
    else:
        state = "graph"
    from stable_baselines3 import PPO
    if state == "graph":
        policy_kwargs = dict(net_arch=dict(pi=[features_dim], vf=[features_dim]),
                                features_extractor_class = NullFeatureExtractor,
                                features_extractor_kwargs = dict(features_dim=features_dim, 
                                                                num_heads = num_heads,
                                                                num_layers = 4,
                                                                mask_tensor = input_dim,
                                                                input_dim = 6),
                                share_features_extractor = True)   
        policy = CustomGraphActorCriticPolicy
        model = PPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir, 
                    policy_kwargs = policy_kwargs, n_steps= n_steps, batch_size=n_steps, gamma=1)
        print("Model created")    
    else:
        policy_kwargs = dict(net_arch=dict(pi=[features_dim], vf=[features_dim]),
                                features_extractor_class = NullFeatureExtractor,
                                features_extractor_kwargs = dict(features_dim=features_dim, 
                                                                max_seq = max_seq,
                                                                computing_setting = state,
                                                                num_heads = 4,
                                                                num_layers = 4,
                                                                input_dim = input_dim),
                                share_features_extractor = True)


        policy = CustomActorCriticPolicy
        model = PPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir, 
                    policy_kwargs = policy_kwargs, n_steps= n_steps, batch_size=n_steps, gamma=1)
        print("Model created")
    return model

def make_env(data, test_data, env_params, classifier, seed = 42):
    print("Second step: Initializing environment")
    env = FPOO(env_params['data_path'],data, test_data, env_params['max_features'], 
               data.shape[1]-1 , ClassifierModel, env_params["state"], classifier, 
               env_params["reward_mode"], env_params["stop"], env_params['cluster_beginning'],
               env_params['cluster_ending'], seed)
    print("Environment created")
    return env , env.input_dim

def load_data(defect_prediction_path_train, defect_prediction_path_eval):
    print("First step: loading data")
    defect_prediction = pd.read_csv(defect_prediction_path_train, usecols=lambda x: x != "Unnamed: 0")
    shuffled_columns = defect_prediction.columns.to_list()
    np.random.seed(42)
    np.random.shuffle(shuffled_columns)

    # Create a new DataFrame with shuffled columns
    defect_prediction_eval = pd.read_csv(defect_prediction_path_eval, usecols=lambda x: x != "Unnamed: 0")

    return (defect_prediction[shuffled_columns], defect_prediction_eval[shuffled_columns])

def load_model(path):

    from stable_baselines3 import PPO
    model = PPO.load(path=path)
    print("Model loaded")
    return model

def get_discounted_reward(rewards, gamma):
    discounted_reward = 0
    for i in range(len(rewards)):
        coeff = pow(gamma, i)
        r = coeff*rewards[i]
        discounted_reward += r
    return discounted_reward

def load_localized_model(path):
    from stable_baselines3 import PPO
    model = PPO.load(path=path)
    return model

def define_project(project_name):
    if project_name == "lucene":
        data_path = 'data/lucene-2.9.0_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/lucene-3.0.0_ground-truth-files_dataset_normalized.csv'
    elif project_name == "hive":
        data_path = 'data/hive-0.10.0_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/hive-0.12.0_ground-truth-files_dataset_normalized.csv'
    elif project_name == "hbase":
        data_path = 'data/hbase-0.95.0_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/hbase-0.95.2_ground-truth-files_dataset_normalized.csv'
    elif project_name == "groovy":
        data_path = 'data/groovy-1_5_7_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/groovy-1_6_BETA_2_ground-truth-files_dataset_normalized.csv'
    elif project_name == "camel":
        data_path = 'data/camel-2.10.0_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/camel-2.11.0_ground-truth-files_dataset_normalized.csv'
    elif project_name == "derby":
        data_path = 'data/derby-10.3.1.4_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/derby-10.5.1.1_ground-truth-files_dataset_normalized.csv'
    elif project_name == "wicket":
        data_path = 'data/wicket-1.3.0-incubating-beta-1_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/wicket-1.5.3_ground-truth-files_dataset_normalized.csv'
    elif project_name == "activemq":
        data_path = 'data/activemq-5.3.0_ground-truth-files_dataset_normalized.csv'
        test_data_path = 'data/activemq-5.8.0_ground-truth-files_dataset_normalized.csv'
    else:
        raise ValueError("Invalid project name")
    
    return data_path, test_data_path
