from utils import load_data, make_agent, make_env, load_model, define_project
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np
from env.classifier import ClassifierModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd


def main():
    """
    min_accuracy: the desired minimum accuracy (in this case reward) we expect our model to reach
    max_features: the desired maximum features for the model to select
    total_timesteps: maximum number of timesteps for the model to be trained
    max_episode_length: maximum number of timesteps within an episode
    """
    import argparse
    parser = argparse.ArgumentParser(description='Fault Prediction Evaluation, Feature Selection')
    parser.add_argument('--classifier', choices=['random_forest', 'logistic_regression', 'svm', 'knn'], default='random_forest')
    parser.add_argument('--project', choices=['lucene', 'groovy', 'wicket', 'hive', 'hbase', 'camel' ,'activemq', 'derby'], default='lucene')
    parser.add_argument('--selection', type=int, choices=[0,1], default=1)
    # parser.add_argument('--features', default=[0, 1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 20, 22, 25, 33, 34, 37], type=lambda s: [int(item) for item in s.split(',')]) # Simple-Phero/Secondary - AUC reward
    parser.add_argument('--features', default=[3458, 3338, 2956, 540, 2981, 169, 298, 2731, 1196, 448, 2753, 718, 2896, 351, 3039, 2406, 105, 746, 2025, 111], type=lambda s: [int(item) for item in s.split(',')]) # Custom-Phero/Secondary - AUC reward -- 5-15 --16
    # parser.add_argument('--features', default=[0, 4, 5, 6, 7, 10, 11, 14, 19, 22, 25, 34, 36, 37, 44, 45, 47, 48, 55, 58], type=lambda s: [int(item) for item in s.split(',')]) # Custom-Phero/Secondary - AUC reward -- 2-10 --16


    args = parser.parse_args()
    data_path, test_data_path = define_project(args.project)
    env_params = {'data_path' : data_path,
                  'test_data_path': test_data_path,
                  'features': args.features}

    cls_model = ClassifierModel(args.classifier)
    data, test_data = load_data(env_params['data_path'], env_params['test_data_path'])
    data, eval_data = train_test_split(data, test_size = 0.2, random_state=42)
    
    # Dropping the "Bug" column from the datasets to prepare the feature sets
    X_train = data.drop(columns=["Bug"])
    X_eval = eval_data.drop(columns=["Bug"])
    X_test = test_data.drop(columns=["Bug"])
    Y_train = data["Bug"]
    Y_eval = eval_data["Bug"]
    Y_test = test_data["Bug"]

    # Combining the training and evaluation data for PCA
    training_x = pd.concat([X_train, X_eval], ignore_index=True)
    training_y = pd.concat([Y_train, Y_eval], ignore_index=True)

    smote = SMOTE(random_state=42)
    training_x, training_y = smote.fit_resample(training_x, training_y)

    if args.selection:
        cls_inputs =dict(train_df = training_x.iloc[:,env_params['features']],
                eval_df = X_test.iloc[:, env_params['features']],
                train_label = training_y,
                eval_label = Y_test)
    else:
        cls_inputs = dict(train_df = training_x,
                        eval_df = X_test,
                        train_label = training_y,
                        eval_label = Y_test)       

    accuracy, report, (confusion, auc, f1), _= cls_model(**cls_inputs)
    print(f"evaluation accuracy: {accuracy}.\n corresponding classification report: {report}. \n confusion matrix:\n {confusion}\n auc score: {auc}")
if __name__ == '__main__':
    main()
