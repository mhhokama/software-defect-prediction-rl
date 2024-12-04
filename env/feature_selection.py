import gym
from gym import spaces
from gym.spaces import *
from gym.utils import seeding
import numpy as np
import time

import random
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



from scipy.special import softmax
import math

    
class FPOO(gym.Env):
    
    def __init__(self, data_name, data, test_data, max_features, 
                 num_features, cls_model, state_mode, classifier, 
                 mode, stop, cluster_beginning, cluster_ending, seed):

        import warnings
        warnings.filterwarnings("ignore")
        
        self.seed = seed
        self.total_time_step = 0
        
        # Loading train, eval, and test data
        self.test_data = test_data
        self.data, self.eval_data = train_test_split(data, test_size = 0.2, random_state = self.seed)
    
        while self.eval_data[self.eval_data["Bug"] == 0].shape[0] == self.eval_data.shape[0]:
            self.seed+=1
            print(f"Eval_data does not entail buggy data, trying again with seed: {self.seed}")
            self.data, self.eval_data = train_test_split(data, test_size = 0.2, random_state = self.seed)
            
        self.X_train = self.data.drop(columns= ["Bug"])
        self.X_eval = self.eval_data.drop(columns= ["Bug"])
        self.Y_train = self.data["Bug"]
        self.Y_eval = self.eval_data["Bug"]
        self.X_test = self.test_data.drop(columns= ["Bug"])
        self.Y_test = self.test_data["Bug"]

        # Augment the training data to balance the classes using SMOTE
        smote = SMOTE(random_state=42)
        self.X_train, self.Y_train = smote.fit_resample(self.X_train, self.Y_train)
        
        
        
        # ratio of not buggy to all eval_data
        self.accuracy_ratio_base = self.eval_data[self.eval_data["Bug"] == 0].shape[0] / self.eval_data.shape[0]
        print("Ratio of not buggy to all eval_data:", self.accuracy_ratio_base)
        
        self.num_features = num_features
        self.max_features = max_features
        
        
        # cls_model is the classifier model that will be used to assess the agen
        self.cls_model = cls_model(classifier)
        
        
        # Pheromone is a technique in feature selection that (mis)uses the processes stochasticity, inspired by a chemical ants leave from themselves
        self.phero = True if "phero" in state_mode else False
        # state_mode determines whether we use our custom vectors or not
        self.state_mode = state_mode
        # embedder either returns a 48 sized vector of (2-10) categorization results for each feature, or the feature index, depending on steate_mode
        self.embedder = Embedder(self.data, self.state_mode, 
                                 cluster_beginning, cluster_ending)
        print("Third step: embedder and cls_model set")
        self.embedding_size = self.embedder.embedding_size
        
        # Defining the input dim for the agent
        if "simple" in self.state_mode:
            self.input_dim = self.X_train.shape[1]
        elif "custom" in self.state_mode:
            self.input_dim = self.embedding_size
        else:
            self.input_dim = self.embedder.graph_mask
        
        # we will define the RL agents essentials in the following lines    
        self.actions = set()
        
        if "simple" in self.state_mode:
            self.action_space = spaces.Box(low = 0,high = 1, shape = (self.num_features,) , dtype=np.float32)
        elif "custom" in self.state_mode:
            low_bounds = [0] * (self.embedding_size-4) + [-1] + [0] + [-1] + [0]
            high_bounds = [1] * (self.embedding_size-4) + [1] + [5] + [1] + [5]
        
            self.action_space = spaces.Box(low = np.array(low_bounds), high = np.array(high_bounds), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.num_features)
        
        
        if "graph" in self.state_mode:
            self.observation_space = spaces.Box(low = -1, high = 6, shape=(self.num_features, 6), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-15, high=15, shape=(self.max_features, self.embedding_size+1), dtype=np.float32)
        
        
        
        # currently selected features are masked
        self.mask = np.ones(shape = (self.num_features,))
        if "graph" in self.state_mode:
            self.state = self.embedder.mean_variance
            self.state[:][4] = self.mask
            self.importance = np.zeros(shape = (self.num_features,))
            
        else:
            self.state = np.zeros(shape=(self.max_features, self.embedding_size+1))
            self.state[1:,-1] = -1
        
        
        self.time_step = 0
        self.max_episode_length = max_features 
        self.accumulative_reward = 0  
        self.reward_mode = mode
        self.stop = stop
        
        self.best_episode_reward = -np.inf
        self.best_action = set()
        self.best_action_report = None
        self.best_accuracy = 0
        self.best_auc = 0
        self.best_confusion = None
        self.best_f1 = 0
        
        
        self.pickle_address = f'{data_name}_{classifier}_{mode}_{max_features}.pkl'
        
        
        self.cusotm_reward_denominator = self.data[self.data["Bug"] == 1].shape[0]
        self.custom_reward_numerator_fn_coef, self.custom_reward_numerator_fp_coef = self.gather_custom_reward_coefs()
        
        if self.phero:
            self.pheromone_score = [0] * self.num_features
            self.pheromone_count = [0] * self.num_features
            self.pheromone_dist = [0] * self.num_features
            self.pheromone_selection_rate = int(self.max_features/3)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
    
        action = self.process_new_action(action)
        
        # Call classifier
        cls_inputs = self.process_actions()
        accuracy, report, (confusion, auc, f1_score), feature_importances = self.cls_model(**cls_inputs)
        
        reward = self.calculate_reward(accuracy, auc, f1_score, confusion)
                   
        self.time_step += 1
        self.total_time_step+=1
        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}
        done = self.time_step >= self.max_episode_length 

        if done:
            if self.accumulative_reward >= self.best_episode_reward:
                self.best_episode_reward = self.accumulative_reward
                self.best_action_report = report
                self.best_action = self.actions
                self.best_auc = auc
                self.best_confusion = confusion
                self.best_accuracy = accuracy
                self.best_f1 = f1_score
                self.save_current_stats(accuracy, report, confusion, auc)
                
        if self.total_time_step % 500 == 0:
            print(f"current time step: {self.total_time_step},\n best auc so far:", 
                  f"{self.best_auc},\n corresponding accuracy: {self.best_accuracy}.\n best actions are:", 
                  f"{self.best_action}\n classification report: {self.best_action_report} \n confusion matrix:", 
                  f"{self.best_confusion}")
        
        if self.phero:
            self.pheromone_count[action] = self.pheromone_count[action]+1
            self.pheromone_score[action] = self.pheromone_score[action]+reward
            self.pheromone_dist[action] = self.pheromone_score[action]/self.pheromone_count[action]
            
        return self.new_state(feature_importances), reward, done, info
    
    def reset(self):
        self.time_step = 0
        
        self.accumulative_reward = 0
        
        self.actions = set()
        
        self.mask = np.ones(shape = (self.num_features,))
        
        if "graph" in self.state_mode:
            self.state = self.embedder.mean_variance
            self.state[:][4] = self.mask
            self.importance = np.zeros(shape = (self.num_features,))        
        else:
            self.state = np.zeros(shape=(self.max_features, self.embedding_size+1))
            self.state[1:,-1] = -1
        
        feature_importances = []
        
        if self.phero:
            self.time_step = self.pheromone_selection_rate
            nums = np.array(self.pheromone_dist, dtype=np.float32)
            
            total_sum = np.sum(nums)
            if total_sum == 0:
                selected_actions = random.choices(range(self.num_features), k = self.pheromone_selection_rate)
            else:
                probabilities = nums / total_sum
                selected_actions = random.choices(range(self.num_features), weights=probabilities, k=self.pheromone_selection_rate)
            
            self.actions.update(selected_actions)
            for action in self.actions:
                self.mask[action] = 0
                
            cls_inputs = self.process_actions()
            accuracy, report, (confusion, auc, f1_score), feature_importances = self.cls_model(**cls_inputs)
            
            if self.reward_mode == "accuracy":
                reward = accuracy - self.accuracy_ratio_base
            elif self.reward_mode == "auc":
                reward = auc
            elif self.reward_mode == "f1":
                reward = f1_score
            else:
                reward = self.custom_reward(confusion)
                
            self.accumulative_reward = reward
            

        return self.new_state(feature_importances)

    
    def process_actions(self):
        
        return dict(train_df = self.X_train.iloc[:,list(self.actions)],
                    eval_df = self.X_eval.iloc[:, list(self.actions)],
                    train_label = self.Y_train,
                    eval_label = self.Y_eval)
    
    def process_graph_state(self, feature_importances):

        for feature, importance in zip(feature_importances.Feature, feature_importances.Importance):
            self.importance[self.X_train.columns.get_loc(feature)] = importance 


        self.state = self.embedder.mean_variance.T
        self.state[:,4] = self.mask
        self.state[:,5] = self.importance
        
        return self.state
        
    def new_state(self, feature_importance):
        if "graph" in self.state_mode:
            self.state = self.process_graph_state(feature_importance)
        else:
            count = 0
            for action in self.actions:
                self.state[count] = self.embedder(action)
                count+=1
                
        return self.state

    def save_current_stats(self, accuracy, report, confusion, auc):
        import pickle
        # Dictionary to hold all objects
        data_to_save = {
            'confusion_matrix': confusion,
            'classification_report': report,
            'auc_score': auc,
            'accuracy': accuracy,
            'integer_list': self.best_action
        }

        # Save the data to a pickle file
        with open(self.pickle_address, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    def test_run(self):
        
        X_train = self.data.drop(columns= ["Bug"])
        X_eval = self.eval_data.drop(columns= ["Bug"])
        Y_train = self.data["Bug"]
        Y_eval = self.eval_data["Bug"]

        training_x = pd.concat([X_train, X_eval], ignore_index=True)
        training_y = pd.concat([Y_train, Y_eval], ignore_index=True)
        
        smote = SMOTE(random_state=42)
        training_x, training_y = smote.fit_resample(training_x, training_y)
        
        # Either the pheromone vector suggestion or the best seen action
        selected_actions = self.get_final_actions()
        print(f"Final selected actions: {set(selected_actions)}")
        cls_inputs =dict(train_df = training_x.iloc[:,selected_actions],
                    eval_df = self.X_test.iloc[:, selected_actions],
                    train_label = training_y,
                    eval_label = self.Y_test)
        
        accuracy, report, (confusion, auc, f1_score), _ = self.cls_model(**cls_inputs)
        print("training on whole training set and testing on test set")
        print(f"accuracy: {accuracy}, \n auc: {auc},\n confusion matrix: {confusion},\n classification report: {report}")\
    
    def get_final_actions(self):
        if self.phero:
            nums = np.array(self.pheromone_dist, dtype=np.float32)
            
            actions = np.argsort(nums)[-self.max_features:]
        else:
            actions = list(self.best_action)
        return actions
        
    def gather_custom_reward_coefs(self,):      
        cls_inputs =dict(train_df = self.X_train,
                    eval_df = self.X_eval,
                    train_label = self.Y_train,
                    eval_label = self.Y_eval)
        
        accuracy, report, (confusion, auc, f1), feature_importance = self.cls_model(**cls_inputs)
        print("Whole data training results on the separated evaluation portion:")
        print(f"confusion matrix: {confusion}, \n {report}")
        # min criteria is used whenever stop is set to True, it is the minimum reward that the agent should achieve
        if self.reward_mode == "accuracy":
            self.min_criteria = accuracy
        elif self.reward_mode == "auc":
            self.min_criteria = auc
        elif self.reward_mode == "f1":
            self.min_criteria = f1
        else:
            TN, FP, FN, TP = confusion.ravel() 
            self.min_criteria = -(1 * FN + 2 * FP) / (self.cusotm_reward_denominator)
                
        print(f"Whole data reward in reward_mode {self.reward_mode} : ",self.min_criteria)
        """
        We assign the coefficients in reverse order. Intuitively given an iverse amount to their actual base values
        gives the already higher amount a lesser importance to lower, since it is easier to lower its value by the same
        size, compared to the other entity with lower amount. The process can intuitively be countes as scaling. Also, a
        simple adjustment is done so that the lesser value is not too signified or vice versa.
        """
        FN_coef, FP_coef = self.process_confusion_matrix(confusion)
        print(f"Coefficients for custom rewarding: \nFN_coef :{FN_coef}, FP_coef: {FP_coef}")
        return FN_coef, FP_coef
        
    def normalize_enclose(self,FP, FN, adjustment_factor=0.15):
        """
        Normalize FP and FN by enclosing them:
        - Increase the smaller value (FN).
        - Decrease the larger value (FP).
        - Move them towards each other based on an adjustment factor.

        Args:
            FP (float): False Positives (larger value).
            FN (float): False Negatives (smaller value).
            adjustment_factor (float): Factor controlling how much to adjust FP and FN.
                                    0 = no change, 1 = fully enclosed.

        Returns:
            Tuple: New enclosed FP and FN values.
        """
        # Calculate the difference between FP and FN
        diff = FP - FN
        
        # Ensure FP is the larger and FN is the smaller (if not, swap)
        if FN >= FP:
            diff = -diff
            
        # Adjust both FP and FN by enclosing them based on the adjustment factor
        FP_new = FP - diff * adjustment_factor  # Decrease FP
        FN_new = FN + diff * adjustment_factor  # Increase FN

        return FP_new, FN_new

    def process_confusion_matrix(self, confusion_matrix, adjustment_factor = 0.15):
        """
        Given a confusion matrix for a binary classification (labels 0 and 1),
        this function extracts the FP and FN values, applies the distance-based 
        normalization, and returns the results.
        
        Args:
            confusion_matrix (numpy array): A 2x2 confusion matrix.
            scale_factor (float): A factor to control how much to adjust the numbers.

        Returns:
            Tuple: Normalized FP and FN values.
        """
        # Extract elements from confusion matrix
        TN, FP, FN, TP = confusion_matrix.ravel()
        # Apply normalization based on distance
        normalized_FP, normalized_FN = self.normalize_enclose(FP, FN, adjustment_factor)

        return normalized_FP, normalized_FN    
    
    def custom_reward(self, confusion):
        # Extract values from the confusion matrix
        TN, FP, FN, TP = confusion.ravel()  # TN, FP, FN, TP in a flattened array
        
        # Calculating the custom reward signal which is : -(FP * FP_coef, FN * FN_coef)/P
        reward = -(self.custom_reward_numerator_fn_coef * FN + self.custom_reward_numerator_fp_coef * FP) / (self.cusotm_reward_denominator)
        
        return reward
    
    def calculate_reward(self, accuracy, auc, f1_score, confusion):
        
        if self.reward_mode == "accuracy":
            reward = accuracy - self.accuracy_ratio_base
        elif self.reward_mode == "auc":
            reward = auc
        elif self.reward_mode == "f1":
            reward = f1_score
        else:
            reward = self.custom_reward(confusion)
            
        reward -= self.accumulative_reward
        self.accumulative_reward += reward
        
        return reward
    
    def process_new_action(self, action):
        if "simple" in self.state_mode:
            action = np.argmax(action * self.mask)
            self.mask[action] = 0
        elif "graph" in self.state_mode:
            self.mask[action] = 0
        else:
            differences = np.sqrt(np.abs(np.array(self.embedder.clustered_data) - action))
            sums = np.sum(differences, axis=1)
            masked_sums = sums * self.mask
            masked_sums[masked_sums == 0.0] = np.inf
            
            action = np.argmin(masked_sums)
            self.mask[action] = 0
                
        self.actions.add(action)
        return action
                

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class Embedder:
    
    def __init__(self, data, setting, cluster_beginning = 2, cluster_ending = 14):
        self.data = data.drop(columns=["Bug"])
        
        # Filter the data for each label
        data_label_1 = data[data['Bug'] == 1].drop(columns=["Bug"])
        data_label_0 = data[data['Bug'] == 0].drop(columns=["Bug"])

        # Calculate mean and variance for Label 1
        mean_label_1 = data_label_1.mean()
        var_label_1 = data_label_1.var()

        # Calculate mean and variance for Label 0
        mean_label_0 = data_label_0.mean()
        var_label_0 = data_label_0.var()

        # Create a new DataFrame for the result
        self.result = pd.DataFrame([mean_label_1, var_label_1, mean_label_0, var_label_0],
                            index=['Mean_Label_1', 'Variance_Label_1', 'Mean_Label_0', 'Variance_Label_0'])

        # Embedding type logic
        if "simple" in setting:
            self.embedding = self.simple_embedding
            self.embedding_size = 1
        elif "custom" in setting:
            self.embedding = self.custom_embedding
            self.cluster_beginning = cluster_beginning
            self.cluster_ending = cluster_ending
            self.setup_custom_embedding()
        elif "graph" in setting:
            self.embedding = self.graph_embedding
            self.cluster_beginning = cluster_beginning
            self.cluster_ending = cluster_ending
            self.setup_graph_embedding()
    
    def setup_custom_embedding(self):
        k_values = range(self.cluster_beginning, self.cluster_ending)

        # Initialize a DataFrame to store the one-hot encoded vectors
        one_hot_encoded = pd.DataFrame(index=self.data.columns, columns=k_values)

        # Initialize a DataFrame to store cluster labels
        cluster_labels = pd.DataFrame(index=self.data.columns, columns=k_values, dtype=int)

        # Perform clustering and generate one-hot encoded vectors
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.result.T)
            
            # Store cluster labels
            cluster_labels[k] = kmeans.labels_

            # Generate one-hot encoded vectors
            encoder = OneHotEncoder(categories=[range(k)], sparse_output=False, dtype=int)
            one_hot_encoded_vectors = encoder.fit_transform(kmeans.labels_.reshape(-1, 1))
            
            # Create a DataFrame with one-hot encoded vectors
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded_vectors, index=self.data.columns, columns=[f'{k}_cluster_{i}' for i in range(k)])
            # Update the one-hot encoded DataFrame
            one_hot_encoded = pd.concat([one_hot_encoded, one_hot_encoded_df], axis=1, join='inner')

        # Combine one-hot encoded vectors with mean and variance
        final_output = pd.concat([one_hot_encoded, self.result.T], axis=1)
        final_output.drop(columns=np.arange(self.cluster_beginning, self.cluster_ending), inplace=True)

        self.clustered_data = final_output
        self.embedding_size = len(final_output.columns)

    def setup_graph_embedding(self):
        
        k_values = range(self.cluster_beginning, self.cluster_ending)
        num_features = len(self.data.columns)
        num_clusters = len(k_values)

        # Initialize the graph tensor with dimensions [K, N, N]
        self.graph_mask = np.ones((num_clusters, num_features, num_features), dtype=int)

        # Loop through each k-value and calculate adjacency matrices
        for idx, k in enumerate(k_values):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.result.T)
            cluster_labels = kmeans.labels_

            # Create adjacency matrix for k clusters
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    if cluster_labels[i] == cluster_labels[j]:
                        self.graph_mask[idx, i, j] = 0
                        self.graph_mask[idx, j, i] = 0

            # Ensure diagonal is all zeros
            np.fill_diagonal(self.graph_mask[idx], 0)

        zeros = np.zeros((2, self.result.shape[1]))

        # Create a tensor for the means and variances
        self.mean_variance = np.vstack((self.result, zeros))
        self.embedding_size = 6

    def __call__(self, obs_id):
        return self.embedding(obs_id)
    
    def simple_embedding(self, obs_id):
        return np.array([obs_id] + [obs_id])
    
    def custom_embedding(self, obs_id):
        return np.array(self.clustered_data.iloc[obs_id].values.tolist() + [obs_id])

    def graph_embedding(self, obs_id):
        # Return the mean/variance tensor and graph tensor for the given obs_id
        return self.mean_variance, self.graph_mask
