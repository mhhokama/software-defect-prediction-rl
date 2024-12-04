import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

class ClassifierModel:
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initializes the ClassifierModel with the specified model type.

        Parameters:
        model_type (str): The type of model to use. Options are 'random_forest', 'logistic_regression', 'knn', and 'svm'.
        random_state (int): Random seed for reproducibility (used for certain models like RandomForest).
        """
        self.random_state = random_state
        self.model_type = model_type
        self.model = self._initialize_model()

    def _initialize_model(self):
        """
        Initializes the model based on the selected type.
        
        Returns:
        A classifier model instance.
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state)
        elif self.model_type == 'knn':
            return KNeighborsClassifier()
        elif self.model_type == 'svm':
            return SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported. Choose from 'random_forest', 'logistic_regression', 'knn', or 'svm'.")

    def __call__(self, train_df, eval_df, train_label, eval_label):
        """
        Trains the selected classifier on the provided training data and evaluates it.

        Parameters:
        train_df (pd.DataFrame): Training dataframe with features and label.
        eval_df (pd.DataFrame): Evaluation dataframe with features and label.

        Returns:
        float: Accuracy score on the evaluation data.
        """
        X_train = train_df
        y_train = train_label
        X_eval = eval_df
        y_eval = eval_label

        # Train the model
        self.train(X_train, y_train)
        
        # Evaluate the model
        accuracy, report, (confusion, auc, f1) = self.evaluate(X_eval, y_eval)
        
        # Feature_importance vector
        feature_importances = self.get_feature_importance(X_train)
    
        return accuracy, report, (confusion, auc, f1), feature_importances

    def train(self, X_train, y_train):
        """
        Trains the classifier on the provided training data.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts labels for the provided test data.

        Parameters:
        X_test (array-like): Test features.

        Returns:
        array-like: Predicted labels.
        """
        return self.model.predict(X_test)
    
    def get_feature_importance(self, X):

        # NOTE: code not totally complete: SVM and KNN not counted.
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            feature_importance = self.model.coef_[0]
        else:
            feature_importance = np.zeros(len(X.columns))

        # Create a DataFrame to display feature importances
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        })
        
        return importance_df

        
        
    def evaluate(self, X_test, y_test):
        """
        Evaluates the classifier's accuracy on the provided test data.

        Parameters:
        X_test (array-like): Test features.
        y_test (array-like): True labels for test data.

        Returns:
        tuple: (accuracy_score, classification_report, (confusion_matrix, auc_score)).
        """
        y_pred = self.predict(X_test)
        
        try:
            # Check if the classifier has 'predict_proba' method
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
            # If 'predict_proba' is not available, fall back to 'decision_function'
            elif hasattr(self.model, 'decision_function'):
                y_proba = self.model.decision_function(X_test)
            else:
                raise ValueError("Model does not support probability or decision function.")
            
            # Calculate AUC using the predicted probabilities or decision function output
            auc = roc_auc_score(y_test, y_proba)
        except ValueError as e:
            print(f"Warning: {e}. Skipping AUC calculation.")
            auc = None  # Set AUC to None if we cannot compute it
        
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), (confusion_matrix(y_test, y_pred), auc, f1_score(y_test, y_pred, average='macro'))
