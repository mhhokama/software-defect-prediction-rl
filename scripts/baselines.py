from utils import load_data, make_agent, make_env, load_model, define_project
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np
from env.classifier import ClassifierModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
import random


def plot_metrics(metric_name, values, folder_path):
    """
    Helper function to plot metrics like TP, FP, FN, TN, accuracy, etc.
    """
    plt.figure()
    plt.plot(values)
    plt.title(f'{metric_name} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, f'{metric_name}.png'))
    plt.close()


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
    parser.add_argument('--strategy', choices=["PCA", "random"], default="PCA", help='Feature selection strategy')
    parser.add_argument('--output_folder', default='plotting/plots/random/Lucene-chunked-NotNormalized', help='Folder to save the generated plots')
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    data_path, test_data_path = define_project(args.project)

    cls_model = ClassifierModel(args.classifier)
    data, test_data = load_data(data_path, test_data_path)
    data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # Dropping the "Bug" column from the datasets to prepare the feature sets
    X_train = data.drop(columns=["Bug"])
    X_eval = eval_data.drop(columns=["Bug"])
    X_test = test_data.drop(columns=["Bug"])
    Y_train = data["Bug"]
    Y_eval = eval_data["Bug"]
    Y_test = test_data["Bug"]

    # Combine the training and evaluation data for feature selection
    training_x = pd.concat([X_train, X_eval], ignore_index=True)
    training_y = pd.concat([Y_train, Y_eval], ignore_index=True)
    
    from sklearn.decomposition import PCA

    # Applying PCA to reduce the feature dimensions to 20 components
    pca = PCA(n_components=20)
    # Fit the PCA on the combined training data and transform both training and test data
    training_x = pca.fit_transform(training_x)
    X_test = pca.transform(X_test)

    # Optionally convert back to DataFrames for better readability
    training_x = pd.DataFrame(training_x, columns=[f'PC{i+1}' for i in range(20)])
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(20)])
    
    smote = SMOTE(random_state=42)
    training_x, training_y = smote.fit_resample(training_x, training_y)
    
    cls_inputs = dict(train_df = training_x,
                eval_df = X_test,
                train_label = training_y,
                eval_label = Y_test)   
    accuracy, report, (confusion, auc, f1), _= cls_model(**cls_inputs)
    print(f"evaluation accuracy: {accuracy}.\n corresponding classification report: {report}. \n confusion matrix:\n {confusion}\n auc score: {auc}")

    # accuracies = []
    # auc_scores = []
    # tps, fps, fns, tns = [], [], [], []

    # for n in range(5, 51):
    #     run_accuracies = []
    #     run_auc_scores = []
    #     run_tps, run_fps, run_fns, run_tns = [], [], [], []

    #     for run in range(5):  # Run each setting 5 times
    #         if args.strategy == "PCA":
    #             # Applying PCA to reduce the feature dimensions to n components
    #             pca = PCA(n_components=n)
    #             training_x_pca = pca.fit_transform(training_x)
    #             X_test_pca = pca.transform(X_test)
    #         elif args.strategy == "random":
    #             # Randomly select 'n' features from the dataset
    #             random_columns = random.sample(list(training_x.columns), min(n, len(training_x.columns)))
    #             training_x_pca = training_x[random_columns]
    #             X_test_pca = X_test[random_columns]

    #         # Apply SMOTE to handle class imbalance
    #         smote = SMOTE(random_state=42)
    #         training_x_pca, training_y_pca = smote.fit_resample(training_x_pca, training_y)

    #         # Classifier inputs
    #         cls_inputs = dict(train_df=training_x_pca,
    #                           eval_df=X_test_pca,
    #                           train_label=training_y_pca,
    #                           eval_label=Y_test)

    #         # Train and evaluate the model
    #         accuracy, report, (confusion, auc, f1) = cls_model(**cls_inputs)

    #         # Append metrics for this run
    #         run_accuracies.append(accuracy)
    #         run_auc_scores.append(auc)
    #         run_tps.append(confusion[1][1])
    #         run_fps.append(confusion[0][1])
    #         run_fns.append(confusion[1][0])
    #         run_tns.append(confusion[0][0])

    #     # Calculate average for this setting after 5 runs
    #     accuracies.append(np.mean(run_accuracies))
    #     auc_scores.append(np.mean(run_auc_scores))
    #     tps.append(np.mean(run_tps))
    #     fps.append(np.mean(run_fps))
    #     fns.append(np.mean(run_fns))
    #     tns.append(np.mean(run_tns))

    #     # Print classification report for the current iteration
    #     print(f"Iteration {n} (Average over 5 runs):\n")
    #     print(f"Evaluation accuracy: {np.mean(run_accuracies)}")
    #     print(f"AUC score: {np.mean(run_auc_scores)}")
    #     print(f"Confusion matrix: \n[[{np.mean(run_tns)}, {np.mean(run_fps)}], [{np.mean(run_fns)}, {np.mean(run_tps)}]]")

    # # Plot and save the individual metrics
    # plot_metrics("TP", tps, args.output_folder)
    # plot_metrics("FP", fps, args.output_folder)
    # plot_metrics("FN", fns, args.output_folder)
    # plot_metrics("TN", tns, args.output_folder)

    # # Plot Accuracy and AUC on the same plot
    # plt.figure()
    # plt.plot(accuracies, label='Accuracy', color='blue')
    # plt.plot(auc_scores, label='AUC', color='red')
    # plt.title('Accuracy and AUC Over Iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_folder, 'Accuracy_AUC.png'))
    # plt.close()

if __name__ == '__main__':
    main()
