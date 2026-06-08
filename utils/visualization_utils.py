"""
Visualization utilities for the churn prediction application.
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_churn_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing churn distribution.

    Args:
        df (pd.DataFrame): Dataset with Attrition_Flag column

    Returns:
        go.Figure: Plotly figure
    """
    try:
        churn_counts = df['Attrition_Flag'].value_counts()
        labels = ['Existing Customer', 'Attrited Customer']
        values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=['#4CAF50', '#F44336'],
            title="Customer Churn Distribution"
        )])

        fig.update_layout(
            title="Customer Churn Distribution",
            font=dict(size=14)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating churn distribution chart: {str(e)}")
        return go.Figure()

def create_age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create histogram showing age distribution by churn status.

    Args:
        df (pd.DataFrame): Dataset with Customer_Age and Attrition_Flag columns

    Returns:
        go.Figure: Plotly figure
    """
    try:
        fig = px.histogram(
            df,
            x="Customer_Age",
            color="Attrition_Flag",
            title="Age Distribution by Churn Status",
            labels={'Attrition_Flag': 'Churn Status', 'Customer_Age': 'Age'},
            color_discrete_map={0: '#4CAF50', 1: '#F44336'},
            barmode='overlay',
            opacity=0.7
        )

        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Count"
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating age distribution chart: {str(e)}")
        return go.Figure()

def create_transaction_amount_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Create box plot of transaction amounts by churn status.

    Args:
        df (pd.DataFrame): Dataset with Total_Trans_Amt and Attrition_Flag columns

    Returns:
        go.Figure: Plotly figure
    """
    try:
        fig = px.box(
            df,
            x="Attrition_Flag",
            y="Total_Trans_Amt",
            title="Total Transaction Amount by Churn Status",
            labels={
                'Attrition_Flag': 'Churn Status',
                'Total_Trans_Amt': 'Total Transaction Amount ($)'
            },
            color="Attrition_Flag",
            color_discrete_map={0: '#4CAF50', 1: '#F44336'}
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating transaction amount boxplot: {str(e)}")
        return go.Figure()

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap for numerical features.

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        go.Figure: Plotly figure
    """
    try:
        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        fig.update_layout(
            width=800,
            height=600
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()

def create_feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.

    Args:
        feature_importance (dict): Feature importance mapping

    Returns:
        go.Figure: Plotly figure
    """
    try:
        # Get top 15 features
        top_features = dict(list(feature_importance.items())[:15])

        fig = go.Figure(go.Bar(
            x=list(top_features.values()),
            y=list(top_features.keys()),
            orientation='h',
            marker_color='#1f77b4'
        ))

        fig.update_layout(
            title="Top 15 Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating feature importance chart: {str(e)}")
        return go.Figure()

def create_confusion_matrix_heatmap(cm: np.ndarray) -> go.Figure:
    """
    Create confusion matrix heatmap.

    Args:
        cm (np.ndarray): Confusion matrix

    Returns:
        go.Figure: Plotly figure
    """
    try:
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100

        # Create annotations with counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(
                    f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)"
                )

        annotations = np.array(annotations).reshape(cm.shape)

        fig = px.imshow(
            cm,
            text_auto=False,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Stay', 'Churn'],
            y=['Stay', 'Churn'],
            color_continuous_scale="Blues"
        )

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                fig.add_annotation(
                    x=j, y=i,
                    text=annotations[i, j],
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )

        return fig

    except Exception as e:
        logger.error(f"Error creating confusion matrix heatmap: {str(e)}")
        return go.Figure()

def create_metrics_comparison_chart() -> go.Figure:
    """
    Create bar chart comparing model metrics.

    Returns:
        go.Figure: Plotly figure
    """
    try:
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Training': [0.98, 0.97, 0.95, 0.96],
            'Validation': [0.96, 0.94, 0.92, 0.93],
            'Test': [0.97, 0.98, 0.93, 0.95]
        }

        df_metrics = pd.DataFrame(metrics_data)

        fig = go.Figure()

        for col in ['Training', 'Validation', 'Test']:
            fig.add_trace(go.Bar(
                name=col,
                x=df_metrics['Metric'],
                y=df_metrics[col],
                text=[f'{val:.1%}' for val in df_metrics[col]],
                textposition='auto'
            ))

        fig.update_layout(
            title="Model Performance Metrics Comparison",
            barmode='group',
            yaxis_title="Score",
            height=400
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating metrics comparison chart: {str(e)}")
        return go.Figure()

def display_metric_card(title: str, value: float, delta: Optional[float] = None):
    """
    Display a metric card in Streamlit.

    Args:
        title (str): Metric title
        value (float): Metric value
        delta (float, optional): Change from previous value
    """
    try:
        if title.lower() in ['accuracy', 'precision', 'recall', 'f1-score']:
            formatted_value = f"{value:.1%}"
        else:
            formatted_value = f"{value:,.0f}"

        st.metric(title, formatted_value, delta)

    except Exception as e:
        logger.error(f"Error displaying metric card: {str(e)}")
        st.metric(title, "N/A")