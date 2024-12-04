from utils import load_data, make_agent, make_env, load_model, define_project
from numpy import load
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

def train(env_params, train_params, tb_log_dir, tb_name, log_dir, seed):

    (data, test_data) = load_data(env_params['data_path'], env_params['test_data_path'])
    env, input_dim = make_env(data, test_data, env_params, env_params["classifier"], seed=seed)
    model = make_agent(env, train_params['device'], env_params["state"], 
                       tb_log_dir, input_dim = input_dim, 
                       features_dim= train_params["features_dim"], 
                       n_steps = train_params["n_steps"], max_seq= env_params["max_features"], 
                       num_heads = train_params["num_heads"])
    
    # model = load_model("PPO","plotting/tb_results/trained_model/custom-ppo-100k-20-SmoteFree-meanbased")
    # model.set_env(env)
    model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name, log_interval= 10)

    model.save(log_dir+tb_name)
    
    env.test_run()

def main():
    
    import argparse
    parser = argparse.ArgumentParser(description='Fault Prediction environment, feature selection')
    parser.add_argument('--state', choices=['simple', 'custom', 'simple_phero', 'custom_phero', 'graph', 'graph_phero'], default='custom')
    parser.add_argument('--classifier', choices=['random_forest', 'logistic_regression', 'svm', 'knn'], default='random_forest')
    parser.add_argument('--project', choices=['lucene', 'groovy', 'wicket', 'hive', 'hbase', 'camel' ,'activemq', 'derby'], default='lucene')
    parser.add_argument('--cluster_beginning', type=int, default=2)
    parser.add_argument('--cluster_ending', type=int, default=14)    
    parser.add_argument('--tb_log_dir', default='plotting/tb_results/tb_logs/')
    parser.add_argument('--tb_name', required=True)
    parser.add_argument('--log_dir', default='plotting/tb_results/trained_model/')
    parser.add_argument('--total_timesteps', type=int, default=41000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max_features', type = int, default=20) 
    parser.add_argument('--features_dim', type = int, default=16)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--reward_mode', choices=['auc', 'accuracy', 'f1', 'custom'], default='auc')
    parser.add_argument('--stop', type=int, choices=[0,1], default=0)

    args = parser.parse_args()
    data_path , test_data_path = define_project(args.project)
    train_params = {'total_timesteps': args.total_timesteps,
                    'device': args.device,
                    'features_dim': args.features_dim,
                    'n_steps': args.n_steps,
                    'num_heads': args.cluster_ending - args.cluster_beginning}

    env_params = {'classifier': args.classifier,
                  'data_path' : data_path,
                  'test_data_path' : test_data_path,
                  'max_features': args.max_features,
                  'state': args.state,
                  'reward_mode': args.reward_mode,
                  'stop': args.stop,
                  'cluster_beginning': args.cluster_beginning,
                  'cluster_ending': args.cluster_ending}

    train(env_params, train_params,
            tb_log_dir=args.tb_log_dir, log_dir=args.log_dir, tb_name=args.tb_name,
            seed=42)

if __name__ == '__main__':
    main()
