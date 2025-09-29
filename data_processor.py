"""
Data processing module for customer churn prediction.
Handles data loading, cleaning, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class ChurnDataProcessor:
    def __init__(self):
        """Initialize the data processor with encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Churn'
        
    def load_data(self, file_path):
        """
        Load the telecom churn dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and data type issues.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (it might be stored as string)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Handle missing values in TotalCharges
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
        
        # Remove customerID as it's not useful for prediction
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        # Convert SeniorCitizen to categorical for consistency
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
        
        print(f"Data cleaned. Final shape: {df_clean.shape}")
        print(f"Missing values per column:\n{df_clean.isnull().sum()}")
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        df_eng = df.copy()
        
        # Average charges per month (avoiding division by zero)
        df_eng['AvgChargesPerMonth'] = df_eng['TotalCharges'] / (df_eng['tenure'] + 1)
        
        # Total services count
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Count of additional services (excluding "No" and "No internet service")
        df_eng['TotalServices'] = 0
        for col in service_cols:
            if col in df_eng.columns:
                df_eng['TotalServices'] += (df_eng[col] == 'Yes').astype(int)
        
        # Contract duration mapping
        contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df_eng['ContractMonths'] = df_eng['Contract'].map(contract_mapping)
        
        # High value customer indicator
        df_eng['HighValueCustomer'] = (df_eng['MonthlyCharges'] > df_eng['MonthlyCharges'].quantile(0.75)).astype(int)
        
        # Long tenure customer
        df_eng['LongTenureCustomer'] = (df_eng['tenure'] > 24).astype(int)
        
        print("Feature engineering completed.")
        return df_eng
    
    def encode_categorical_features(self, df, fit_encoders=True):
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Dataset with categorical features
            fit_encoders (bool): Whether to fit new encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns (excluding target)
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        # Encode categorical features
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    
                    # Map unknown values to the most frequent class
                    if unique_values - known_values:
                        most_frequent_class = self.label_encoders[col].classes_[0]
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: most_frequent_class if x not in known_values else x
                        )
                    
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        # Encode target variable
        if fit_encoders and self.target_column in df_encoded.columns:
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
            df_encoded[self.target_column] = self.label_encoders['target'].fit_transform(df_encoded[self.target_column])
        
        print(f"Categorical encoding completed for {len(categorical_cols)} columns.")
        return df_encoded
    
    def scale_features(self, X, fit_scaler=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Feature matrix
            fit_scaler (bool): Whether to fit the scaler or use existing one
            
        Returns:
            np.array: Scaled feature matrix
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        print("Feature scaling completed.")
        return X_scaled
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Complete data preparation pipeline.
        
        Args:
            df (pd.DataFrame): Raw dataset
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        """
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_eng = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_eng, fit_encoders=True)
        
        # Separate features and target
        X = df_encoded.drop(self.target_column, axis=1)
        y = df_encoded[self.target_column]
        
        # Store feature names
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit_scaler=True)
        X_test_scaled = self.scale_features(X_test, fit_scaler=False)
        
        print(f"Data preparation completed.")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_columns
    
    def prepare_single_prediction(self, customer_data):
        """
        Prepare a single customer's data for prediction.
        
        Args:
            customer_data (dict): Customer feature dictionary
            
        Returns:
            np.array: Processed feature vector
        """
        # Create DataFrame from customer data
        df_single = pd.DataFrame([customer_data])
        
        # Clean data (basic cleaning)
        df_single['TotalCharges'] = pd.to_numeric(df_single['TotalCharges'], errors='coerce')
        df_single['TotalCharges'].fillna(0, inplace=True)
        df_single['SeniorCitizen'] = df_single['SeniorCitizen'].astype(str)
        
        # Engineer features
        df_single = self.engineer_features(df_single)
        
        # Encode categorical features (using existing encoders)
        df_single = self.encode_categorical_features(df_single, fit_encoders=False)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df_single.columns:
                df_single[col] = 0
        
        # Select and order features
        X_single = df_single[self.feature_columns]
        
        # Scale features
        X_single_scaled = self.scale_features(X_single, fit_scaler=False)
        
        return X_single_scaled
    
    def save_preprocessor(self, filepath='models/preprocessor.joblib'):
        """
        Save the preprocessor (encoders, scaler, feature columns).
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.joblib'):
        """
        Load the preprocessor (encoders, scaler, feature columns).
        
        Args:
            filepath (str): Path to load the preprocessor from
        """
        try:
            preprocessor_data = joblib.load(filepath)
            self.label_encoders = preprocessor_data['label_encoders']
            self.scaler = preprocessor_data['scaler']
            self.feature_columns = preprocessor_data['feature_columns']
            print(f"Preprocessor loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
            return False
    
    def get_feature_info(self):
        """
        Get information about the features for the UI.
        
        Returns:
            dict: Feature information for form generation
        """
        feature_info = {
            'gender': {'type': 'categorical', 'options': ['Female', 'Male']},
            'SeniorCitizen': {'type': 'categorical', 'options': ['0', '1']},
            'Partner': {'type': 'categorical', 'options': ['No', 'Yes']},
            'Dependents': {'type': 'categorical', 'options': ['No', 'Yes']},
            'tenure': {'type': 'numerical', 'min': 0, 'max': 72, 'default': 24},
            'PhoneService': {'type': 'categorical', 'options': ['No', 'Yes']},
            'MultipleLines': {'type': 'categorical', 'options': ['No', 'No phone service', 'Yes']},
            'InternetService': {'type': 'categorical', 'options': ['DSL', 'Fiber optic', 'No']},
            'OnlineSecurity': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'OnlineBackup': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'DeviceProtection': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'TechSupport': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'StreamingTV': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'StreamingMovies': {'type': 'categorical', 'options': ['No', 'No internet service', 'Yes']},
            'Contract': {'type': 'categorical', 'options': ['Month-to-month', 'One year', 'Two year']},
            'PaperlessBilling': {'type': 'categorical', 'options': ['No', 'Yes']},
            'PaymentMethod': {'type': 'categorical', 'options': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']},
            'MonthlyCharges': {'type': 'numerical', 'min': 18.0, 'max': 120.0, 'default': 65.0},
            'TotalCharges': {'type': 'numerical', 'min': 18.0, 'max': 8500.0, 'default': 2300.0}
        }
        
        return feature_info
