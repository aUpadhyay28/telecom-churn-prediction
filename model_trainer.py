"""
Model training module for customer churn prediction.
Trains multiple ML models and evaluates their performance.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from data_processor import ChurnDataProcessor

class ChurnModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate Decision Tree Classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            sklearn model: Trained Decision Tree model
        """
        print("Training Decision Tree Classifier...")
        
        # Hyperparameter tuning
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
        
        dt = DecisionTreeClassifier()
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_dt = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_dt.predict(X_test)
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        if hasattr(best_dt, 'feature_importances_'):
            self.feature_importance['decision_tree'] = best_dt.feature_importances_
        
        self.models['decision_tree'] = best_dt
        self.model_performance['decision_tree'] = performance
        
        print(f"Decision Tree - Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")
        return best_dt
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate Logistic Regression.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            sklearn model: Trained Logistic Regression model
        """
        print("Training Logistic Regression...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'random_state': [42],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression()
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_lr.predict(X_test)
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'best_params': grid_search.best_params_
        }
        
        # Feature importance (coefficients)
        if hasattr(best_lr, 'coef_'):
            self.feature_importance['logistic_regression'] = np.abs(best_lr.coef_[0])
        
        self.models['logistic_regression'] = best_lr
        self.model_performance['logistic_regression'] = performance
        
        print(f"Logistic Regression - Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")
        return best_lr
    
    def train_svm_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate SVM Classifier (instead of SVR for classification).
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            sklearn model: Trained SVM model
        """
        print("Training SVM Classifier...")
        
        # Hyperparameter tuning (limited for performance)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }
        
        svm = SVC(probability=True)  # Enable probability estimates
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_svm.predict(X_test)
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'best_params': grid_search.best_params_
        }
        
        self.models['svm_classifier'] = best_svm
        self.model_performance['svm_classifier'] = performance
        
        print(f"SVM Classifier - Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")
        return best_svm
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test):
        """
        Train an ensemble model combining all three models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            sklearn model: Trained ensemble model
        """
        print("Training Ensemble Model...")
        
        if len(self.models) < 3:
            print("Warning: Not all base models are trained. Train base models first.")
            return None
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('dt', self.models['decision_tree']),
                ('lr', self.models['logistic_regression']),
                ('svm', self.models['svm_classifier'])
            ],
            voting='soft'  # Use probability estimates
        )
        
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_pred = ensemble.predict(X_test)
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        self.models['ensemble'] = ensemble
        self.model_performance['ensemble'] = performance
        
        print(f"Ensemble Model - Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")
        return ensemble
    
    def train_all_models(self, data_processor, df):
        """
        Train all models using the data processor.
        
        Args:
            data_processor: ChurnDataProcessor instance
            df: Raw dataset
            
        Returns:
            dict: All trained models
        """
        print("Starting model training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = data_processor.prepare_data(df)
        
        # Train individual models
        self.train_decision_tree(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_svm_classifier(X_train, y_train, X_test, y_test)
        
        # Train ensemble model
        self.train_ensemble_model(X_train, y_train, X_test, y_test)
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        print("\nModel Training Complete!")
        print("="*50)
        for model_name, performance in self.model_performance.items():
            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {performance['accuracy']:.4f}")
            print(f"  Precision: {performance['precision']:.4f}")
            print(f"  Recall: {performance['recall']:.4f}")
            print(f"  F1-Score: {performance['f1_score']:.4f}")
            print()
        
        return self.models
    
    def get_feature_importance_df(self):
        """
        Get feature importance as a DataFrame for visualization.
        
        Returns:
            pd.DataFrame: Feature importance data
        """
        if not self.feature_importance or not hasattr(self, 'feature_names'):
            return pd.DataFrame()
        
        importance_data = []
        
        for model_name, importance in self.feature_importance.items():
            for i, feature in enumerate(self.feature_names):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance[i]
                })
        
        return pd.DataFrame(importance_data)
    
    def save_models(self, model_dir='models'):
        """
        Save all trained models and performance metrics.
        
        Args:
            model_dir (str): Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save performance metrics
        performance_path = os.path.join(model_dir, 'model_performance.joblib')
        joblib.dump(self.model_performance, performance_path)
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(model_dir, 'feature_importance.joblib')
            joblib.dump(self.feature_importance, importance_path)
        
        # Save feature names
        if hasattr(self, 'feature_names'):
            feature_names_path = os.path.join(model_dir, 'feature_names.joblib')
            joblib.dump(self.feature_names, feature_names_path)
        
        print(f"All models and metrics saved to {model_dir}")
    
    def load_models(self, model_dir='models'):
        """
        Load all trained models and performance metrics.
        
        Args:
            model_dir (str): Directory to load models from
        """
        try:
            # Load individual models
            model_files = ['decision_tree.joblib', 'logistic_regression.joblib', 
                          'svm_classifier.joblib', 'ensemble.joblib']
            
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                if os.path.exists(model_path):
                    model_name = model_file.replace('.joblib', '')
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} from {model_path}")
            
            # Load performance metrics
            performance_path = os.path.join(model_dir, 'model_performance.joblib')
            if os.path.exists(performance_path):
                self.model_performance = joblib.load(performance_path)
                print("Loaded model performance metrics")
            
            # Load feature importance
            importance_path = os.path.join(model_dir, 'feature_importance.joblib')
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
                print("Loaded feature importance data")
            
            # Load feature names
            feature_names_path = os.path.join(model_dir, 'feature_names.joblib')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
                print("Loaded feature names")
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

def main():
    """
    Main function to run the complete training pipeline.
    """
    # Initialize components
    data_processor = ChurnDataProcessor()
    model_trainer = ChurnModelTrainer()
    
    # Load data
    df = data_processor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Train all models
    models = model_trainer.train_all_models(data_processor, df)
    
    # Save models and preprocessor
    model_trainer.save_models()
    data_processor.save_preprocessor()
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
