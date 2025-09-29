"""
Prediction module for customer churn prediction.
Handles loading models and making predictions for new customers.
"""

import numpy as np
import pandas as pd
import joblib
import os
from data_processor import ChurnDataProcessor
from model_trainer import ChurnModelTrainer

class ChurnPredictor:
    def __init__(self, model_dir='models'):
        """
        Initialize the predictor with trained models and preprocessor.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        self.model_dir = model_dir
        self.data_processor = ChurnDataProcessor()
        self.model_trainer = ChurnModelTrainer()
        self.models_loaded = False
        
        # Load models and preprocessor
        self.load_all_components()
    
    def load_all_components(self):
        """Load all required components for prediction."""
        try:
            # Load preprocessor
            preprocessor_loaded = self.data_processor.load_preprocessor(
                os.path.join(self.model_dir, 'preprocessor.joblib')
            )
            
            # Load models
            models_loaded = self.model_trainer.load_models(self.model_dir)
            
            self.models_loaded = preprocessor_loaded and models_loaded
            
            if self.models_loaded:
                print("All components loaded successfully!")
            else:
                print("Warning: Some components failed to load. Predictions may not work correctly.")
                
        except Exception as e:
            print(f"Error loading components: {str(e)}")
            self.models_loaded = False
    
    def predict_single_customer(self, customer_data):
        """
        Make churn prediction for a single customer.
        
        Args:
            customer_data (dict): Customer feature dictionary
            
        Returns:
            dict: Prediction results from all models
        """
        if not self.models_loaded:
            return {"error": "Models not loaded properly"}
        
        try:
            # Preprocess customer data
            X_processed = self.data_processor.prepare_single_prediction(customer_data)
            
            # Make predictions with all models
            predictions = {}
            
            for model_name, model in self.model_trainer.models.items():
                try:
                    # Get prediction
                    pred = model.predict(X_processed)[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_processed)[0]
                        confidence = max(proba)
                        churn_probability = proba[1] if len(proba) > 1 else proba[0]
                    else:
                        confidence = 0.8  # Default confidence for models without probability
                        churn_probability = pred
                    
                    # Convert prediction to human readable
                    if 'target' in self.data_processor.label_encoders:
                        pred_label = self.data_processor.label_encoders['target'].inverse_transform([pred])[0]
                    else:
                        pred_label = "Yes" if pred == 1 else "No"
                    
                    predictions[model_name] = {
                        'prediction': pred_label,
                        'confidence': float(confidence),
                        'churn_probability': float(churn_probability),
                        'prediction_numeric': int(pred)
                    }
                    
                except Exception as e:
                    predictions[model_name] = {
                        'error': f"Prediction failed: {str(e)}"
                    }
            
            # Calculate ensemble prediction (majority vote)
            if len(predictions) >= 3:
                valid_predictions = [p for p in predictions.values() if 'error' not in p]
                if valid_predictions:
                    churn_votes = sum(1 for p in valid_predictions if p['prediction'] == 'Yes')
                    total_votes = len(valid_predictions)
                    
                    ensemble_pred = "Yes" if churn_votes > total_votes / 2 else "No"
                    ensemble_confidence = churn_votes / total_votes if ensemble_pred == "Yes" else (total_votes - churn_votes) / total_votes
                    
                    predictions['consensus'] = {
                        'prediction': ensemble_pred,
                        'confidence': float(ensemble_confidence),
                        'churn_probability': float(churn_votes / total_votes),
                        'votes': f"{churn_votes}/{total_votes}"
                    }
            
            return predictions
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, customer_data_list):
        """
        Make churn predictions for multiple customers.
        
        Args:
            customer_data_list (list): List of customer feature dictionaries
            
        Returns:
            list: List of prediction results
        """
        if not self.models_loaded:
            return [{"error": "Models not loaded properly"}] * len(customer_data_list)
        
        results = []
        for customer_data in customer_data_list:
            result = self.predict_single_customer(customer_data)
            results.append(result)
        
        return results
    
    def get_model_performance(self):
        """
        Get performance metrics for all trained models.
        
        Returns:
            dict: Model performance metrics
        """
        if hasattr(self.model_trainer, 'model_performance'):
            return self.model_trainer.model_performance
        else:
            return {}
    
    def get_feature_importance(self):
        """
        Get feature importance for models that support it.
        
        Returns:
            dict: Feature importance data
        """
        if hasattr(self.model_trainer, 'feature_importance'):
            return self.model_trainer.feature_importance
        else:
            return {}
    
    def explain_prediction(self, customer_data, model_name='decision_tree'):
        """
        Provide explanation for a prediction (simplified feature contribution).
        
        Args:
            customer_data (dict): Customer feature dictionary
            model_name (str): Model to use for explanation
            
        Returns:
            dict: Explanation of the prediction
        """
        if not self.models_loaded or model_name not in self.model_trainer.models:
            return {"error": "Model not available for explanation"}
        
        try:
            # Get prediction
            prediction_result = self.predict_single_customer(customer_data)
            
            if model_name not in prediction_result or 'error' in prediction_result[model_name]:
                return {"error": "Prediction failed"}
            
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            if model_name in feature_importance and hasattr(self.model_trainer, 'feature_names'):
                # Get top important features
                importance_scores = feature_importance[model_name]
                feature_names = self.model_trainer.feature_names
                
                # Create feature importance ranking
                feature_ranking = []
                for i, (feature, importance) in enumerate(zip(feature_names, importance_scores)):
                    if feature in customer_data:
                        feature_ranking.append({
                            'feature': feature,
                            'importance': float(importance),
                            'value': customer_data[feature]
                        })
                
                # Sort by importance
                feature_ranking.sort(key=lambda x: x['importance'], reverse=True)
                
                explanation = {
                    'prediction': prediction_result[model_name]['prediction'],
                    'confidence': prediction_result[model_name]['confidence'],
                    'top_features': feature_ranking[:5],  # Top 5 features
                    'model_used': model_name
                }
                
                return explanation
            else:
                return {
                    'prediction': prediction_result[model_name]['prediction'],
                    'confidence': prediction_result[model_name]['confidence'],
                    'explanation': 'Feature importance not available for this model'
                }
                
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}
    
    def is_ready(self):
        """
        Check if the predictor is ready to make predictions.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.models_loaded and len(self.model_trainer.models) > 0

def create_sample_customer():
    """
    Create a sample customer for testing purposes.
    
    Returns:
        dict: Sample customer data
    """
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': '0',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 49.85,
        'TotalCharges': 598.2
    }
    
    return sample_customer

def main():
    """
    Test the predictor with a sample customer.
    """
    predictor = ChurnPredictor()
    
    if not predictor.is_ready():
        print("Predictor is not ready. Please train models first.")
        return
    
    # Test with sample customer
    sample_customer = create_sample_customer()
    print("Testing with sample customer:")
    print(sample_customer)
    print("\nPrediction Results:")
    
    results = predictor.predict_single_customer(sample_customer)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            if 'churn_probability' in result:
                print(f"  Churn Probability: {result['churn_probability']:.3f}")

if __name__ == "__main__":
    main()
