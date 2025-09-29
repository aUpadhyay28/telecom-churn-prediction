"""
Visualization module for customer churn prediction dashboard.
Creates interactive plots and charts using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import ChurnDataProcessor

class ChurnVisualizations:
    def __init__(self):
        """Initialize the visualization module."""
        self.color_palette = {
            'churn': '#e74c3c',
            'no_churn': '#27ae60',
            'primary': '#3498db',
            'secondary': '#95a5a6',
            'accent': '#f39c12'
        }
    
    def plot_churn_distribution(self, df):
        """
        Create a pie chart showing churn distribution.
        
        Args:
            df (pd.DataFrame): Dataset with churn information
            
        Returns:
            plotly.graph_objects.Figure: Pie chart
        """
        churn_counts = df['Churn'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['No Churn', 'Churn'],
            values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
            hole=0.3,
            marker_colors=[self.color_palette['no_churn'], self.color_palette['churn']]
        )])
        
        fig.update_layout(
            title="Customer Churn Distribution",
            annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df):
        """
        Create a bar chart showing feature importance for different models.
        
        Args:
            importance_df (pd.DataFrame): Feature importance data
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        if importance_df.empty:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Feature Importance", height=400)
            return fig
        
        # Get top 10 features per model
        top_features = importance_df.groupby('Model').apply(
            lambda x: x.nlargest(10, 'Importance')
        ).reset_index(drop=True)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            color='Model',
            orientation='h',
            title="Top 10 Most Important Features by Model",
            height=600
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Feature Importance",
            yaxis_title="Features"
        )
        
        return fig
    
    def plot_numerical_features_distribution(self, df):
        """
        Create distribution plots for numerical features.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            plotly.graph_objects.Figure: Subplot figure
        """
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=numerical_cols,
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(numerical_cols, 1):
            if col in df.columns:
                # Histogram for No Churn
                no_churn_data = df[df['Churn'] == 'No'][col]
                churn_data = df[df['Churn'] == 'Yes'][col]
                
                fig.add_trace(
                    go.Histogram(
                        x=no_churn_data,
                        name=f'No Churn - {col}',
                        marker_color=self.color_palette['no_churn'],
                        opacity=0.7,
                        showlegend=(i == 1)
                    ),
                    row=1, col=i
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=churn_data,
                        name=f'Churn - {col}',
                        marker_color=self.color_palette['churn'],
                        opacity=0.7,
                        showlegend=(i == 1)
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title="Distribution of Numerical Features by Churn Status",
            height=400,
            barmode='overlay'
        )
        
        return fig
    
    def plot_categorical_features(self, df):
        """
        Create bar charts for key categorical features.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            plotly.graph_objects.Figure: Subplot figure
        """
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'gender']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=categorical_cols,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                row = (i // 2) + 1
                col_idx = (i % 2) + 1
                
                # Calculate churn rate by category
                churn_rate = df.groupby(col)['Churn'].apply(
                    lambda x: (x == 'Yes').sum() / len(x) * 100
                ).reset_index()
                churn_rate.columns = [col, 'ChurnRate']
                
                fig.add_trace(
                    go.Bar(
                        x=churn_rate[col],
                        y=churn_rate['ChurnRate'],
                        marker_color=self.color_palette['primary'],
                        showlegend=False
                    ),
                    row=row, col=col_idx
                )
        
        fig.update_layout(
            title="Churn Rate by Categorical Features",
            height=600
        )
        
        # Update y-axis titles
        for i in range(1, 5):
            fig.update_yaxes(title_text="Churn Rate (%)", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        
        return fig
    
    def plot_correlation_heatmap(self, df):
        """
        Create a correlation heatmap for numerical features.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient numerical data for correlation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Feature Correlation Heatmap", height=400)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=500,
            width=600
        )
        
        return fig
    
    def plot_model_performance_comparison(self, model_performance):
        """
        Create a radar chart comparing model performance metrics.
        
        Args:
            model_performance (dict): Performance metrics for each model
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        if not model_performance:
            fig = go.Figure()
            fig.add_annotation(
                text="No model performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Model Performance Comparison", height=400)
            return fig
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for model_name, performance in model_performance.items():
            if model_name != 'ensemble':  # Skip ensemble for clarity
                values = [performance.get(metric, 0) for metric in metrics]
                values.append(values[0])  # Close the radar chart
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model_name.replace('_', ' ').title()
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison",
            height=500
        )
        
        return fig
    
    def plot_churn_by_tenure_and_charges(self, df):
        """
        Create a scatter plot showing relationship between tenure, charges, and churn.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot
        """
        if not all(col in df.columns for col in ['tenure', 'MonthlyCharges', 'Churn']):
            fig = go.Figure()
            fig.add_annotation(
                text="Required columns not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Tenure vs Monthly Charges", height=400)
            return fig
        
        fig = px.scatter(
            df,
            x='tenure',
            y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'No': self.color_palette['no_churn'], 
                              'Yes': self.color_palette['churn']},
            title="Customer Tenure vs Monthly Charges (by Churn Status)",
            labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'},
            opacity=0.6
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_prediction_confidence(self, predictions):
        """
        Create a bar chart showing prediction confidence for different models.
        
        Args:
            predictions (dict): Prediction results from different models
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        if not predictions or all('error' in pred for pred in predictions.values()):
            fig = go.Figure()
            fig.add_annotation(
                text="No valid predictions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Prediction Confidence", height=300)
            return fig
        
        models = []
        confidences = []
        predictions_labels = []
        colors = []
        
        for model_name, result in predictions.items():
            if 'error' not in result and 'confidence' in result:
                models.append(model_name.replace('_', ' ').title())
                confidences.append(result['confidence'])
                predictions_labels.append(result['prediction'])
                colors.append(self.color_palette['churn'] if result['prediction'] == 'Yes' 
                            else self.color_palette['no_churn'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=confidences,
                marker_color=colors,
                text=[f"{p}<br>{c:.2%}" for p, c in zip(predictions_labels, confidences)],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Model Prediction Confidence",
            xaxis_title="Models",
            yaxis_title="Confidence",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_dashboard_summary(self, df):
        """
        Create summary statistics for the dashboard.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Summary statistics
        """
        try:
            total_customers = len(df)
            churned_customers = len(df[df['Churn'] == 'Yes']) if 'Churn' in df.columns else 0
            churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
            
            avg_tenure = df['tenure'].mean() if 'tenure' in df.columns else 0
            avg_monthly_charges = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 0
            avg_total_charges = df['TotalCharges'].mean() if 'TotalCharges' in df.columns else 0
            
            summary = {
                'total_customers': total_customers,
                'churned_customers': churned_customers,
                'churn_rate': churn_rate,
                'avg_tenure': avg_tenure,
                'avg_monthly_charges': avg_monthly_charges,
                'avg_total_charges': avg_total_charges
            }
            
            return summary
            
        except Exception as e:
            print(f"Error creating dashboard summary: {str(e)}")
            return {
                'total_customers': 0,
                'churned_customers': 0,
                'churn_rate': 0,
                'avg_tenure': 0,
                'avg_monthly_charges': 0,
                'avg_total_charges': 0
            }

def main():
    """
    Test the visualization module.
    """
    # Load sample data
    data_processor = ChurnDataProcessor()
    df = data_processor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if df is not None:
        # Clean data for visualization
        df_clean = data_processor.clean_data(df)
        
        # Initialize visualizations
        viz = ChurnVisualizations()
        
        # Create sample plots
        print("Creating sample visualizations...")
        
        # Churn distribution
        fig1 = viz.plot_churn_distribution(df_clean)
        
        # Numerical features distribution
        fig2 = viz.plot_numerical_features_distribution(df_clean)
        
        # Categorical features
        fig3 = viz.plot_categorical_features(df_clean)
        
        # Summary statistics
        summary = viz.create_dashboard_summary(df_clean)
        print("Dashboard Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("Visualizations created successfully!")
    else:
        print("Failed to load data for visualization testing.")

if __name__ == "__main__":
    main()
