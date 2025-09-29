"""
Main Streamlit application for customer churn prediction.
Provides interactive interface for data visualization, model training, and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from data_processor import ChurnDataProcessor
    from model_trainer import ChurnModelTrainer
    from predictor import ChurnPredictor
    from visualizations import ChurnVisualizations
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data():
    """Load the customer churn dataset."""
    data_file = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if os.path.exists(data_file):
        try:
            data_processor = ChurnDataProcessor()
            df = data_processor.load_data(data_file)
            if df is not None:
                df_clean = data_processor.clean_data(df)
                st.session_state.df = df_clean
                st.session_state.data_loaded = True
                return df_clean
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.error("Dataset not found. Please ensure the data file is in the correct location.")
    
    return None

def check_models_exist():
    """Check if trained models exist."""
    model_files = [
        'models/decision_tree.joblib',
        'models/logistic_regression.joblib',
        'models/svm_classifier.joblib',
        'models/preprocessor.joblib'
    ]
    return all(os.path.exists(file) for file in model_files)

def validate_training_data(df):
    """
    Validate uploaded training data against expected schema.
    
    Args:
        df (pd.DataFrame): Uploaded training data
        
    Returns:
        dict: Validation result with 'valid' flag and 'errors' list
    """
    result = {'valid': True, 'errors': []}
    
    # Get expected schema
    data_processor = ChurnDataProcessor()
    feature_info = data_processor.get_feature_info()
    expected_columns = set(feature_info.keys())
    expected_columns.add('Churn')  # Add target column
    
    # Check columns
    actual_columns = set(df.columns)
    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns
    
    if missing_columns:
        result['valid'] = False
        result['missing_columns'] = list(missing_columns)
        result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
    
    if extra_columns:
        result['extra_columns'] = list(extra_columns)
        result['errors'].append(f"Unexpected columns found: {', '.join(extra_columns)} (these will be ignored)")
    
    # Validate Churn column values if present
    if 'Churn' in df.columns:
        valid_churn_values = {'Yes', 'No', 'yes', 'no', 'YES', 'NO', '1', '0', 1, 0}
        actual_churn_values = set(df['Churn'].dropna().unique())
        invalid_churn = actual_churn_values - valid_churn_values
        
        if invalid_churn:
            result['valid'] = False
            result['errors'].append(f"Invalid Churn values: {invalid_churn}. Must be 'Yes', 'No', '1', '0', or similar variants.")
    
    # Check for minimum data requirements
    if len(df) < 10:
        result['valid'] = False
        result['errors'].append("Dataset too small. Need at least 10 records for training.")
    
    # Check for class balance
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        if len(churn_counts) < 2:
            result['valid'] = False
            result['errors'].append("Dataset must contain both churned and non-churned customers.")
        else:
            min_class_ratio = min(churn_counts) / len(df)
            if min_class_ratio < 0.05:  # Less than 5% of minority class
                result['errors'].append(f"Warning: Severe class imbalance detected. Minority class: {min_class_ratio:.1%}")
    
    # Validate data types for numerical columns
    for feature, info in feature_info.items():
        if feature in df.columns and info['type'] == 'numerical':
            try:
                pd.to_numeric(df[feature], errors='coerce')
            except:
                result['valid'] = False
                result['errors'].append(f"Column '{feature}' should contain numerical values.")
    
    return result

def train_models_page():
    """Model training page."""
    st.title("🤖 Model Training")
    
    # Training mode selection
    training_mode = st.radio(
        "Choose training mode:",
        ["📊 Train with Default Dataset", "📤 Upload New Training Data"],
        horizontal=True
    )
    
    if training_mode == "📊 Train with Default Dataset":
        train_with_default_data()
    else:
        train_with_uploaded_data()

def train_with_default_data():
    """Train models with the default dataset."""
    if not st.session_state.data_loaded:
        st.warning("Please load data first.")
        if st.button("Load Data"):
            df = load_data()
            if df is not None:
                st.success("Data loaded successfully!")
                st.rerun()
        return
    
    df = st.session_state.df
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        churn_count = len(df[df['Churn'] == 'Yes']) if 'Churn' in df.columns else 0
        st.metric("Churned Customers", churn_count)
    with col3:
        churn_rate = (churn_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    perform_training(df, "default dataset")

def train_with_uploaded_data():
    """Train models with uploaded training data."""
    st.subheader("📤 Upload New Training Data")
    
    st.info("""
    Upload a CSV file with training data to retrain the models.
    The CSV should contain all customer features plus a 'Churn' column with 'Yes'/'No' values.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file for training",
        type=['csv'],
        help="Upload a CSV file with customer data including the 'Churn' column"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            new_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(new_df)} records.")
            
            # Show preview of uploaded data
            st.subheader("📋 Training Data Preview")
            st.dataframe(new_df.head(), use_container_width=True)
            
            # Validate schema and data quality
            validation_result = validate_training_data(new_df)
            if not validation_result['valid']:
                st.error("❌ Data validation failed:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
                
                # Show expected schema
                if validation_result.get('missing_columns') or validation_result.get('extra_columns'):
                    st.subheader("📋 Expected Data Schema")
                    data_processor = ChurnDataProcessor()
                    feature_info = data_processor.get_feature_info()
                    expected_columns = list(feature_info.keys()) + ['Churn']
                    st.info(f"Expected columns: {', '.join(expected_columns)}")
                    
                    # Download training template
                    if st.button("📥 Download Training Data Template"):
                        template_data = {}
                        for feature, info in feature_info.items():
                            if info['type'] == 'categorical':
                                template_data[feature] = info['options'][0]
                            else:
                                template_data[feature] = info.get('default', info.get('min', 0))
                        template_data['Churn'] = 'No'  # Add target column
                        
                        template_df = pd.DataFrame([template_data])
                        template_csv = template_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Training Template CSV",
                            data=template_csv,
                            file_name="training_data_template.csv",
                            mime="text/csv"
                        )
                
                return
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(new_df))
            with col2:
                churn_count = len(new_df[new_df['Churn'] == 'Yes'])
                st.metric("Churned Customers", churn_count)
            with col3:
                churn_rate = (churn_count / len(new_df) * 100) if len(new_df) > 0 else 0
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            # Check data quality
            st.subheader("📊 Data Quality Check")
            missing_data = new_df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("⚠️ Missing data detected:")
                st.dataframe(missing_data[missing_data > 0].to_frame('Missing Count'), use_container_width=True)
            else:
                st.success("✅ No missing data detected")
            
            # Training section
            if len(new_df) < 100:
                st.warning("⚠️ Dataset is quite small. Consider adding more data for better model performance.")
            
            perform_training(new_df, "uploaded dataset")
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format with proper headers.")

def perform_training(df, dataset_name):
    """Perform the actual model training."""
    st.subheader("🤖 Model Training")
    
    models_exist = check_models_exist()
    if models_exist:
        st.info("Trained models found. You can retrain with new data or proceed to predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Train All Models with {dataset_name.title()}", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Initialize components
                    data_processor = ChurnDataProcessor()
                    model_trainer = ChurnModelTrainer()
                    
                    # Train models
                    models = model_trainer.train_all_models(data_processor, df)
                    
                    # Save models
                    model_trainer.save_models()
                    data_processor.save_preprocessor()
                    
                    st.session_state.models_trained = True
                    st.success(f"✅ Models trained and saved successfully using {dataset_name}!")
                    
                    # Display performance metrics
                    st.subheader("📈 Model Performance")
                    performance_df = pd.DataFrame(model_trainer.model_performance).T
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Performance comparison if previous models existed
                    if models_exist:
                        st.info("🔄 Models have been retrained and updated.")
                    
                    # Clear cache to reload updated models
                    st.cache_resource.clear()
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.error("Please check your data format and try again.")
    
    with col2:
        if models_exist and st.button("Load Existing Models"):
            try:
                predictor = ChurnPredictor()
                if predictor.is_ready():
                    st.session_state.models_trained = True
                    st.success("Models loaded successfully!")
                    
                    # Display performance metrics
                    performance = predictor.get_model_performance()
                    if performance:
                        st.subheader("Model Performance")
                        performance_df = pd.DataFrame(performance).T
                        st.dataframe(performance_df, use_container_width=True)
                else:
                    st.error("Failed to load models.")
            except Exception as e:
                st.error(f"Loading failed: {str(e)}")

def prediction_page():
    """Customer prediction page."""
    st.title("🔮 Customer Churn Prediction")
    
    # Check if models are available
    if not check_models_exist():
        st.warning("No trained models found. Please train models first.")
        return
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor():
        return ChurnPredictor()
    
    predictor = get_predictor()
    
    if not predictor.is_ready():
        st.error("Models not loaded properly. Please retrain models.")
        return
    
    # Prediction mode selection
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["🔮 Single Customer", "📊 Batch Prediction (CSV Upload)"],
        horizontal=True
    )
    
    if prediction_mode == "🔮 Single Customer":
        single_customer_prediction(predictor)
    else:
        batch_prediction(predictor)

def single_customer_prediction(predictor):
    """Handle single customer prediction."""
    st.subheader("Enter Customer Information")
    
    # Get feature info for form generation
    data_processor = ChurnDataProcessor()
    feature_info = data_processor.get_feature_info()
    
    # Create input form
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        customer_data = {}
        
        with col1:
            st.subheader("Demographics")
            customer_data['gender'] = st.selectbox("Gender", feature_info['gender']['options'])
            customer_data['SeniorCitizen'] = st.selectbox("Senior Citizen", ['0', '1'], format_func=lambda x: 'No' if x == '0' else 'Yes')
            customer_data['Partner'] = st.selectbox("Partner", feature_info['Partner']['options'])
            customer_data['Dependents'] = st.selectbox("Dependents", feature_info['Dependents']['options'])
            customer_data['tenure'] = st.slider("Tenure (months)", 0, 72, 24)
            
            st.subheader("Services")
            customer_data['PhoneService'] = st.selectbox("Phone Service", feature_info['PhoneService']['options'])
            customer_data['MultipleLines'] = st.selectbox("Multiple Lines", feature_info['MultipleLines']['options'])
            customer_data['InternetService'] = st.selectbox("Internet Service", feature_info['InternetService']['options'])
            customer_data['OnlineSecurity'] = st.selectbox("Online Security", feature_info['OnlineSecurity']['options'])
            customer_data['OnlineBackup'] = st.selectbox("Online Backup", feature_info['OnlineBackup']['options'])
        
        with col2:
            st.subheader("Additional Services")
            customer_data['DeviceProtection'] = st.selectbox("Device Protection", feature_info['DeviceProtection']['options'])
            customer_data['TechSupport'] = st.selectbox("Tech Support", feature_info['TechSupport']['options'])
            customer_data['StreamingTV'] = st.selectbox("Streaming TV", feature_info['StreamingTV']['options'])
            customer_data['StreamingMovies'] = st.selectbox("Streaming Movies", feature_info['StreamingMovies']['options'])
            
            st.subheader("Account Information")
            customer_data['Contract'] = st.selectbox("Contract", feature_info['Contract']['options'])
            customer_data['PaperlessBilling'] = st.selectbox("Paperless Billing", feature_info['PaperlessBilling']['options'])
            customer_data['PaymentMethod'] = st.selectbox("Payment Method", feature_info['PaymentMethod']['options'])
            customer_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.05)
            customer_data['TotalCharges'] = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=2300.0, step=0.05)
        
        submitted = st.form_submit_button("Predict Churn", type="primary")
    
    if submitted:
        display_single_prediction_results(predictor, customer_data)

def display_single_prediction_results(predictor, customer_data):
    """Display results for single customer prediction."""
    with st.spinner("Making predictions..."):
        try:
            # Make prediction
            predictions = predictor.predict_single_customer(customer_data)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            # Create columns for model results
            model_names = [k for k in predictions.keys() if 'error' not in predictions[k]]
            
            if model_names:
                # Display consensus if available
                if 'consensus' in predictions:
                    consensus = predictions['consensus']
                    
                    # Big result display
                    result_color = "🔴" if consensus['prediction'] == 'Yes' else "🟢"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {'#ffebee' if consensus['prediction'] == 'Yes' else '#e8f5e8'};">
                        <h2>{result_color} Customer Churn Prediction: {consensus['prediction']}</h2>
                        <p style="font-size: 18px;">Confidence: {consensus['confidence']:.1%}</p>
                        <p>Consensus: {predictions['consensus'].get('votes', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Individual model results
                st.subheader("Individual Model Predictions")
                cols = st.columns(len(model_names))
                
                for i, model_name in enumerate(model_names):
                    if model_name != 'consensus':
                        with cols[i % len(cols)]:
                            result = predictions[model_name]
                            if 'error' not in result:
                                icon = "🔴" if result['prediction'] == 'Yes' else "🟢"
                                st.markdown(f"""
                                **{model_name.replace('_', ' ').title()}**  
                                {icon} {result['prediction']}  
                                Confidence: {result['confidence']:.1%}
                                """)
                
                # Visualize prediction confidence
                viz = ChurnVisualizations()
                confidence_fig = viz.plot_prediction_confidence(predictions)
                st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Explanation (if available)
                st.subheader("🔍 Prediction Explanation")
                explanation = predictor.explain_prediction(customer_data, 'decision_tree')
                
                if 'error' not in explanation and 'top_features' in explanation:
                    st.write("**Top factors influencing this prediction:**")
                    for i, feature in enumerate(explanation['top_features'], 1):
                        st.write(f"{i}. **{feature['feature']}**: {feature['value']} (Importance: {feature['importance']:.3f})")
                else:
                    st.info("Detailed explanation not available.")
            
            else:
                st.error("All model predictions failed. Please check the input data.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def batch_prediction(predictor):
    """Handle batch prediction from CSV upload."""
    st.subheader("📊 Batch Prediction")
    
    st.info("""
    Upload a CSV file with customer data to predict churn for multiple customers at once.
    The CSV should contain the same columns as the training data (excluding 'Churn' column).
    """)
    
    # Sample CSV download
    st.subheader("📄 Sample CSV Format")
    data_processor = ChurnDataProcessor()
    feature_info = data_processor.get_feature_info()
    
    # Create sample data dynamically from feature_info
    sample_data = {}
    for feature, info in feature_info.items():
        if info['type'] == 'categorical':
            sample_data[feature] = info['options'][0]  # Use first option as default
        elif info['type'] == 'numerical':
            sample_data[feature] = info.get('default', info.get('min', 0))  # Use default or min value
    
    sample_df = pd.DataFrame([sample_data])
    st.dataframe(sample_df, use_container_width=True)
    
    # Download sample CSV
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="customer_data_template.csv",
        mime="text/csv"
    )
    
    # File upload
    st.subheader("📤 Upload Customer Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} customers.")
            
            # Show preview of uploaded data
            st.subheader("📋 Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Validate columns
            required_columns = list(feature_info.keys())
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your CSV contains all required columns as shown in the sample template.")
                return
            
            # Predict button
            if st.button("🔮 Predict Churn for All Customers", type="primary"):
                with st.spinner(f"Making predictions for {len(batch_df)} customers..."):
                    try:
                        # Convert DataFrame to list of dictionaries
                        customer_list = batch_df.to_dict('records')
                        
                        # Make batch predictions
                        batch_results = predictor.predict_batch(customer_list)
                        
                        # Process results
                        results_data = []
                        for i, (customer, predictions) in enumerate(zip(customer_list, batch_results)):
                            if 'error' not in predictions:
                                # Get consensus prediction if available
                                if 'consensus' in predictions:
                                    churn_pred = predictions['consensus']['prediction']
                                    confidence = predictions['consensus']['confidence']
                                else:
                                    # Use first available model prediction
                                    valid_preds = [p for p in predictions.values() if 'error' not in p]
                                    if valid_preds:
                                        churn_pred = valid_preds[0]['prediction']
                                        confidence = valid_preds[0]['confidence']
                                    else:
                                        churn_pred = 'Error'
                                        confidence = 0.0
                                
                                # Add individual model predictions
                                row_data = {
                                    'Customer_ID': i + 1,
                                    'Churn_Prediction': churn_pred,
                                    'Confidence': confidence,
                                    'Risk_Level': 'High' if churn_pred == 'Yes' and confidence > 0.7 else 'Medium' if churn_pred == 'Yes' else 'Low'
                                }
                                
                                # Add individual model results
                                for model_name, result in predictions.items():
                                    if model_name != 'consensus' and 'error' not in result:
                                        row_data[f'{model_name}_prediction'] = result['prediction']
                                        row_data[f'{model_name}_confidence'] = result['confidence']
                                
                                results_data.append(row_data)
                            else:
                                results_data.append({
                                    'Customer_ID': i + 1,
                                    'Churn_Prediction': 'Error',
                                    'Confidence': 0.0,
                                    'Risk_Level': 'Unknown',
                                    'Error': predictions.get('error', 'Unknown error')
                                })
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results_data)
                        
                        # Display results summary
                        st.subheader("📊 Batch Prediction Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_customers = len(results_df)
                            st.metric("Total Customers", total_customers)
                        with col2:
                            churn_count = len(results_df[results_df['Churn_Prediction'] == 'Yes'])
                            st.metric("Predicted Churners", churn_count)
                        with col3:
                            churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        with col4:
                            high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                            st.metric("High Risk Customers", high_risk)
                        
                        # Display detailed results
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        results_csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Prediction Results",
                            data=results_csv,
                            file_name=f"churn_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Risk distribution visualization
                        if 'Risk_Level' in results_df.columns:
                            risk_counts = results_df['Risk_Level'].value_counts()
                            
                            import plotly.express as px
                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Customer Risk Distribution",
                                color_discrete_map={
                                    'High': '#e74c3c',
                                    'Medium': '#f39c12',
                                    'Low': '#27ae60',
                                    'Unknown': '#95a5a6'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format.")

def analytics_page():
    """Data analytics and visualization page."""
    st.title("📊 Data Analytics Dashboard")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        df = load_data()
        if df is None:
            st.warning("Please load data first.")
            return
    else:
        df = st.session_state.df
    
    # Initialize visualizations
    viz = ChurnVisualizations()
    
    # Dashboard summary
    summary = viz.create_dashboard_summary(df)
    
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{summary['total_customers']:,}")
    with col2:
        st.metric("Churned Customers", f"{summary['churned_customers']:,}")
    with col3:
        st.metric("Churn Rate", f"{summary['churn_rate']:.1f}%")
    with col4:
        st.metric("Avg Monthly Charges", f"${summary['avg_monthly_charges']:.2f}")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Feature Analysis", "🤖 Model Performance", "📈 Advanced"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            churn_fig = viz.plot_churn_distribution(df)
            st.plotly_chart(churn_fig, use_container_width=True)
        
        with col2:
            # Tenure vs charges scatter
            scatter_fig = viz.plot_churn_by_tenure_and_charges(df)
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Numerical features distribution
        num_fig = viz.plot_numerical_features_distribution(df)
        st.plotly_chart(num_fig, use_container_width=True)
    
    with tab2:
        # Categorical features
        cat_fig = viz.plot_categorical_features(df)
        st.plotly_chart(cat_fig, use_container_width=True)
        
        # Correlation heatmap
        corr_fig = viz.plot_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    with tab3:
        if check_models_exist():
            try:
                predictor = ChurnPredictor()
                performance = predictor.get_model_performance()
                
                if performance:
                    # Model performance comparison
                    perf_fig = viz.plot_model_performance_comparison(performance)
                    st.plotly_chart(perf_fig, use_container_width=True)
                    
                    # Feature importance
                    importance = predictor.get_feature_importance()
                    if importance and hasattr(predictor.model_trainer, 'feature_names'):
                        importance_df = predictor.model_trainer.get_feature_importance_df()
                        if not importance_df.empty:
                            imp_fig = viz.plot_feature_importance(importance_df)
                            st.plotly_chart(imp_fig, use_container_width=True)
                    
                    # Performance metrics table
                    st.subheader("Detailed Performance Metrics")
                    performance_df = pd.DataFrame(performance).T
                    st.dataframe(performance_df, use_container_width=True)
                else:
                    st.info("No model performance data available.")
            except Exception as e:
                st.error(f"Error loading model performance: {str(e)}")
        else:
            st.warning("No trained models found. Please train models first to see performance metrics.")
    
    with tab4:
        # Customer Segmentation Analysis
        customer_segmentation_analysis(df, viz)

def customer_segmentation_analysis(df, viz):
    """Perform customer segmentation analysis using clustering algorithms."""
    st.subheader("🎯 Customer Segmentation Analysis")
    
    # Import clustering libraries
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Segmentation type selection
    segmentation_type = st.selectbox(
        "Choose Segmentation Analysis:",
        ["📊 Risk-Based Segmentation", "💰 Value-Based Segmentation", "📈 Behavioral Segmentation", "🤖 ML-Based Clustering"]
    )
    
    if segmentation_type == "📊 Risk-Based Segmentation":
        risk_based_segmentation(df)
    elif segmentation_type == "💰 Value-Based Segmentation":
        value_based_segmentation(df)
    elif segmentation_type == "📈 Behavioral Segmentation":
        behavioral_segmentation(df)
    else:
        ml_based_clustering(df)

def risk_based_segmentation(df):
    """Segment customers by churn risk levels."""
    st.subheader("📊 Risk-Based Customer Segmentation")
    
    # Calculate risk score based on multiple factors
    risk_factors = []
    
    # Contract type risk (month-to-month = high risk)
    if 'Contract' in df.columns:
        contract_risk = df['Contract'].map({
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }).fillna(2)
        risk_factors.append(contract_risk)
    
    # Payment method risk
    if 'PaymentMethod' in df.columns:
        payment_risk = df['PaymentMethod'].map({
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 1
        }).fillna(2)
        risk_factors.append(payment_risk)
    
    # Tenure risk (lower tenure = higher risk)
    if 'tenure' in df.columns:
        tenure_risk = pd.cut(df['tenure'], bins=[0, 12, 36, 100], labels=[3, 2, 1]).astype(float)
        risk_factors.append(tenure_risk)
    
    # Calculate combined risk score
    if risk_factors:
        df_analysis = df.copy()
        df_analysis['risk_score'] = sum(risk_factors) / len(risk_factors)
        
        # Define risk segments
        df_analysis['risk_segment'] = pd.cut(
            df_analysis['risk_score'],
            bins=[0, 1.5, 2.5, 3],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Display risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = df_analysis['risk_segment'].value_counts()
            risk_fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Customer Risk Distribution",
                color_discrete_map={
                    'Low Risk': '#28a745',
                    'Medium Risk': '#ffc107', 
                    'High Risk': '#dc3545'
                }
            )
            st.plotly_chart(risk_fig, use_container_width=True)
        
        with col2:
            # Risk vs actual churn rate
            if 'Churn' in df.columns:
                churn_by_risk = df_analysis.groupby('risk_segment')['Churn'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).reset_index()
                churn_by_risk.columns = ['Risk Segment', 'Churn Rate (%)']
                
                risk_churn_fig = px.bar(
                    churn_by_risk,
                    x='Risk Segment',
                    y='Churn Rate (%)',
                    title="Actual Churn Rate by Risk Segment",
                    color='Churn Rate (%)',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(risk_churn_fig, use_container_width=True)
        
        # Risk segment characteristics
        st.subheader("📋 Risk Segment Characteristics")
        
        segment_stats = []
        for segment in ['Low Risk', 'Medium Risk', 'High Risk']:
            segment_data = df_analysis[df_analysis['risk_segment'] == segment]
            if len(segment_data) > 0:
                stats = {
                    'Segment': segment,
                    'Count': len(segment_data),
                    'Avg Tenure': segment_data['tenure'].mean() if 'tenure' in segment_data.columns else 0,
                    'Avg Monthly Charges': segment_data['MonthlyCharges'].mean() if 'MonthlyCharges' in segment_data.columns else 0,
                    'Actual Churn Rate': (segment_data['Churn'] == 'Yes').mean() * 100 if 'Churn' in segment_data.columns else 0
                }
                segment_stats.append(stats)
        
        if segment_stats:
            segment_df = pd.DataFrame(segment_stats)
            st.dataframe(segment_df, use_container_width=True)
    else:
        st.warning("Insufficient data for risk-based segmentation analysis.")

def value_based_segmentation(df):
    """Segment customers by monetary value and usage patterns."""
    st.subheader("💰 Value-Based Customer Segmentation")
    
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df_analysis = df.copy()
        
        # Calculate customer lifetime value
        df_analysis['customer_lifetime_value'] = df_analysis['MonthlyCharges'] * df_analysis['tenure']
        
        # Create value segments using quartiles
        df_analysis['value_segment'] = pd.qcut(
            df_analysis['customer_lifetime_value'],
            q=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium Value']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Value distribution
            value_counts = df_analysis['value_segment'].value_counts()
            value_fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title="Customer Value Distribution",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(value_fig, use_container_width=True)
        
        with col2:
            # Value vs Churn relationship
            if 'Churn' in df.columns:
                churn_by_value = df_analysis.groupby('value_segment')['Churn'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).reset_index()
                churn_by_value.columns = ['Value Segment', 'Churn Rate (%)']
                
                value_churn_fig = px.bar(
                    churn_by_value,
                    x='Value Segment',
                    y='Churn Rate (%)',
                    title="Churn Rate by Customer Value",
                    color='Churn Rate (%)',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(value_churn_fig, use_container_width=True)
        
        # Value segment analysis
        st.subheader("📊 Value Segment Analysis")
        
        value_stats = []
        for segment in ['Low Value', 'Medium Value', 'High Value', 'Premium Value']:
            segment_data = df_analysis[df_analysis['value_segment'] == segment]
            if len(segment_data) > 0:
                stats = {
                    'Segment': segment,
                    'Count': len(segment_data),
                    'Avg CLV': segment_data['customer_lifetime_value'].mean(),
                    'Avg Monthly Charges': segment_data['MonthlyCharges'].mean(),
                    'Avg Tenure': segment_data['tenure'].mean(),
                    'Churn Rate (%)': (segment_data['Churn'] == 'Yes').mean() * 100 if 'Churn' in segment_data.columns else 0
                }
                value_stats.append(stats)
        
        if value_stats:
            value_df = pd.DataFrame(value_stats)
            # Format numeric columns
            for col in ['Avg CLV', 'Avg Monthly Charges', 'Avg Tenure', 'Churn Rate (%)']:
                if col in value_df.columns:
                    value_df[col] = value_df[col].round(2)
            st.dataframe(value_df, use_container_width=True)
        
        # CLV distribution visualization
        st.subheader("📈 Customer Lifetime Value Distribution")
        clv_fig = px.histogram(
            df_analysis,
            x='customer_lifetime_value',
            color='value_segment',
            title="Customer Lifetime Value Distribution by Segment",
            nbins=30,
            marginal="box"
        )
        st.plotly_chart(clv_fig, use_container_width=True)
        
    else:
        st.warning("MonthlyCharges and tenure columns required for value-based segmentation.")

def behavioral_segmentation(df):
    """Segment customers by service usage and behavioral patterns."""
    st.subheader("📈 Behavioral Customer Segmentation")
    
    # Define service usage patterns
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    available_services = [col for col in service_columns if col in df.columns]
    
    if available_services:
        df_analysis = df.copy()
        
        # Calculate service adoption score
        service_scores = []
        for col in available_services:
            if col == 'InternetService':
                score = (df_analysis[col] != 'No').astype(int)
            else:
                score = (df_analysis[col] == 'Yes').astype(int)
            service_scores.append(score)
        
        df_analysis['service_adoption_score'] = sum(service_scores)
        
        # Create behavioral segments
        max_score = len(available_services)
        if max_score > 0:
            bins = [0, max_score*0.25, max_score*0.5, max_score*0.75, max_score]
            labels = ['Basic User', 'Light User', 'Moderate User', 'Heavy User']
            df_analysis['behavioral_segment'] = pd.cut(
                df_analysis['service_adoption_score'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Behavioral segment distribution
                behavior_counts = df_analysis['behavioral_segment'].value_counts()
                behavior_fig = px.pie(
                    values=behavior_counts.values,
                    names=behavior_counts.index,
                    title="Customer Behavioral Segments",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(behavior_fig, use_container_width=True)
            
            with col2:
                # Behavioral pattern vs Churn
                if 'Churn' in df.columns:
                    churn_by_behavior = df_analysis.groupby('behavioral_segment')['Churn'].apply(
                        lambda x: (x == 'Yes').mean() * 100
                    ).reset_index()
                    churn_by_behavior.columns = ['Behavioral Segment', 'Churn Rate (%)']
                    
                    behavior_churn_fig = px.bar(
                        churn_by_behavior,
                        x='Behavioral Segment',
                        y='Churn Rate (%)',
                        title="Churn Rate by Usage Pattern",
                        color='Churn Rate (%)',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(behavior_churn_fig, use_container_width=True)
            
            # Service adoption heatmap
            st.subheader("🔥 Service Adoption Heatmap")
            
            # Create service adoption matrix by segment
            adoption_matrix = []
            for segment in labels:
                segment_data = df_analysis[df_analysis['behavioral_segment'] == segment]
                if len(segment_data) > 0:
                    adoption_rates = []
                    for service in available_services:
                        if service == 'InternetService':
                            rate = (segment_data[service] != 'No').mean() * 100
                        else:
                            rate = (segment_data[service] == 'Yes').mean() * 100
                        adoption_rates.append(rate)
                    adoption_matrix.append(adoption_rates)
            
            if adoption_matrix:
                heatmap_fig = px.imshow(
                    adoption_matrix,
                    labels=dict(x="Services", y="Customer Segments", color="Adoption Rate (%)"),
                    x=available_services,
                    y=labels,
                    color_continuous_scale='RdYlBu_r',
                    title="Service Adoption Rate by Customer Segment"
                )
                heatmap_fig.update_layout(height=400)
                st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.warning("Service usage columns not found for behavioral segmentation.")

def ml_based_clustering(df):
    """Perform ML-based customer clustering using K-Means."""
    st.subheader("🤖 ML-Based Customer Clustering")
    
    # Select numerical features for clustering
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    available_features = [col for col in numerical_features if col in df.columns]
    
    if len(available_features) >= 2:
        # Prepare data for clustering
        df_clustering = df[available_features].copy()
        
        # Handle missing values
        df_clustering = df_clustering.fillna(df_clustering.mean())
        
        # Convert TotalCharges to numeric if needed
        if 'TotalCharges' in df_clustering.columns:
            df_clustering['TotalCharges'] = pd.to_numeric(df_clustering['TotalCharges'], errors='coerce')
            df_clustering['TotalCharges'] = df_clustering['TotalCharges'].fillna(df_clustering['TotalCharges'].mean())
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_clustering)
        
        # Elbow method for optimal k
        st.subheader("📈 Optimal Number of Clusters")
        
        k_range = range(2, 8)
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        elbow_fig = px.line(
            x=list(k_range),
            y=inertias,
            title="Elbow Method for Optimal K",
            labels={'x': 'Number of Clusters (k)', 'y': 'Within-cluster Sum of Squares'}
        )
        elbow_fig.add_scatter(x=list(k_range), y=inertias, mode='markers', marker=dict(size=8))
        st.plotly_chart(elbow_fig, use_container_width=True)
        
        # Cluster selection
        optimal_k = st.slider("Select Number of Clusters", min_value=2, max_value=7, value=3)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to original data
        df_clustered = df.copy()
        df_clustered['cluster'] = [f'Cluster {i+1}' for i in cluster_labels]
        
        # Visualize clusters
        if len(available_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D cluster visualization
                cluster_fig = px.scatter(
                    df_clustered,
                    x=available_features[0],
                    y=available_features[1],
                    color='cluster',
                    title=f"Customer Clusters ({available_features[0]} vs {available_features[1]})",
                    hover_data=available_features
                )
                st.plotly_chart(cluster_fig, use_container_width=True)
            
            with col2:
                # Cluster size distribution
                cluster_counts = df_clustered['cluster'].value_counts()
                cluster_dist_fig = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title="Cluster Size Distribution"
                )
                st.plotly_chart(cluster_dist_fig, use_container_width=True)
        
        # PCA visualization for higher dimensions
        if len(available_features) > 2:
            st.subheader("🔍 PCA Visualization")
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            pca_df = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])
            pca_df['cluster'] = df_clustered['cluster']
            
            pca_fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='cluster',
                title=f"Customer Clusters in PCA Space (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})"
            )
            st.plotly_chart(pca_fig, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("📊 Cluster Characteristics")
        
        cluster_stats = []
        for cluster in df_clustered['cluster'].unique():
            cluster_data = df_clustered[df_clustered['cluster'] == cluster]
            stats = {'Cluster': cluster, 'Size': len(cluster_data)}
            
            for feature in available_features:
                stats[f'Avg {feature}'] = cluster_data[feature].mean()
            
            if 'Churn' in df_clustered.columns:
                stats['Churn Rate (%)'] = (cluster_data['Churn'] == 'Yes').mean() * 100
            
            cluster_stats.append(stats)
        
        cluster_df = pd.DataFrame(cluster_stats)
        # Round numeric columns
        numeric_cols = [col for col in cluster_df.columns if 'Avg' in col or 'Rate' in col]
        for col in numeric_cols:
            cluster_df[col] = cluster_df[col].round(2)
        
        st.dataframe(cluster_df, use_container_width=True)
        
        # Download clustered data
        csv = df_clustered.to_csv(index=False)
        st.download_button(
            label="📥 Download Clustered Customer Data",
            data=csv,
            file_name="clustered_customer_data.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("At least 2 numerical features (tenure, MonthlyCharges, TotalCharges) are required for ML-based clustering.")

def main():
    """Main application function."""
    # Sidebar navigation
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🤖 Model Training", "🔮 Predictions", "📊 Analytics"]
    )
    
    # Add app info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About This App
    This application predicts customer churn using machine learning models trained on telecom customer data.
    
    **Features:**
    - Multiple ML models (Decision Tree, Logistic Regression, SVM)
    - Interactive data visualizations
    - Real-time predictions with confidence scores
    - Model performance comparison
    """)
    
    # Main content based on page selection
    if page == "🏠 Home":
        st.title("🏠 Customer Churn Prediction Dashboard")
        st.markdown("""
        Welcome to the Customer Churn Prediction application! This tool helps predict whether customers 
        are likely to churn based on their service usage and account information.
        
        ### 🚀 Getting Started
        1. **Train Models**: Go to the Model Training page to train ML models on the dataset
        2. **Make Predictions**: Use the Predictions page to predict churn for individual customers
        3. **Explore Data**: Check out the Analytics page for data insights and visualizations
        
        ### 📊 Models Available
        - **Decision Tree Classifier**: Interpretable tree-based model
        - **Logistic Regression**: Linear probabilistic model
        - **SVM Classifier**: Support Vector Machine for complex patterns
        - **Ensemble Model**: Combines all models for better accuracy
        
        ### 🎯 Key Features
        - Real-time predictions with confidence scores
        - Interactive data visualizations
        - Model performance comparison
        - Feature importance analysis
        """)
        
        # Quick status check
        col1, col2 = st.columns(2)
        with col1:
            data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
            st.info(f"**Data Status**: {data_status}")
        
        with col2:
            model_status = "✅ Available" if check_models_exist() else "❌ Not Available"
            st.info(f"**Models Status**: {model_status}")
        
        # Quick actions
        st.subheader("🔧 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Load Data", type="secondary"):
                df = load_data()
                if df is not None:
                    st.success("Data loaded successfully!")
                    st.rerun()
        
        with col2:
            if st.button("Check Models", type="secondary"):
                if check_models_exist():
                    st.success("Trained models found!")
                else:
                    st.warning("No trained models found.")
        
        with col3:
            if st.button("View Analytics", type="secondary"):
                st.switch_page("Analytics")
    
    elif page == "🤖 Model Training":
        train_models_page()
    
    elif page == "🔮 Predictions":
        prediction_page()
    
    elif page == "📊 Analytics":
        analytics_page()

if __name__ == "__main__":
    main()
