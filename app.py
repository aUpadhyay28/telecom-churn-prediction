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

def train_models_page():
    """Model training page."""
    st.title("🤖 Model Training")
    
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
    
    # Training section
    st.subheader("Model Training")
    
    models_exist = check_models_exist()
    if models_exist:
        st.info("Trained models found. You can retrain or proceed to predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train All Models", type="primary"):
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
                    st.success("Models trained and saved successfully!")
                    
                    # Display performance metrics
                    st.subheader("Model Performance")
                    performance_df = pd.DataFrame(model_trainer.model_performance).T
                    st.dataframe(performance_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
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
        st.subheader("Raw Data Explorer")
        
        # Data filtering
        col1, col2 = st.columns(2)
        with col1:
            churn_filter = st.selectbox("Filter by Churn", ['All', 'Yes', 'No'])
        with col2:
            contract_filter = st.selectbox("Filter by Contract", ['All'] + df['Contract'].unique().tolist() if 'Contract' in df.columns else ['All'])
        
        # Apply filters
        filtered_df = df.copy()
        if churn_filter != 'All':
            filtered_df = filtered_df[filtered_df['Churn'] == churn_filter]
        if contract_filter != 'All':
            filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]
        
        st.write(f"Showing {len(filtered_df)} out of {len(df)} customers")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_customer_data.csv",
            mime="text/csv"
        )

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
