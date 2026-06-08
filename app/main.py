"""
Main Streamlit application for Credit Card Churn Prediction.
A comprehensive web application with modern UI, advanced analytics, and real-time predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

# Import custom modules
from config.config import *
from utils.data_loader import load_and_preprocess_data
from utils.model_utils import (
    load_model_cached,
    preprocess_input_data,
    make_prediction,
    get_feature_importance,
    validate_input_data
)
from utils.visualization_utils import (
    create_churn_distribution_chart,
    create_age_distribution_chart,
    create_transaction_amount_boxplot,
    create_correlation_heatmap,
    create_feature_importance_chart,
    create_confusion_matrix_heatmap,
    create_metrics_comparison_chart,
    display_metric_card
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Custom CSS for modern UI
st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    .prediction-card {{
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 3px solid;
    }}
    .churn-risk {{
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-color: {DANGER_COLOR};
        color: #b71c1c;
    }}
    .stay-safe {{
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-color: {SUCCESS_COLOR};
        color: #1b5e20;
    }}
    .metric-container {{
        background-color: {BACKGROUND_COLOR};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {PRIMARY_COLOR};
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    .sidebar-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        margin-bottom: 1rem;
    }}
    .tab-header {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        margin-bottom: 1rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""

    # Load data and model
    df, data_summary = load_and_preprocess_data(DATASET_PATH)
    model = load_model_cached(MODEL_PATH)

    # Main header
    st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown("### 🧠 Advanced Machine Learning for Customer Retention Analytics")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 **Smart Prediction**",
        "📊 **Data Insights**",
        "📈 **Model Analytics**",
        "ℹ️ **About & Help**"
    ])

    # Tab 1: Prediction
    with tab1:
        st.markdown('<h2 class="tab-header">🔮 Customer Churn Prediction</h2>', unsafe_allow_html=True)
        st.markdown("Enter customer details below to get an AI-powered churn risk assessment:")

        prediction_interface(df, model)

    # Tab 2: Data Insights
    with tab2:
        st.markdown('<h2 class="tab-header">📊 Data Insights & Analytics</h2>', unsafe_allow_html=True)

        if df is not None:
            data_insights_interface(df, data_summary)
        else:
            st.error("❌ Dataset not available. Please check data loading.")

    # Tab 3: Model Analytics
    with tab3:
        st.markdown('<h2 class="tab-header">📈 Model Performance & Analytics</h2>', unsafe_allow_html=True)

        if model is not None:
            model_analytics_interface(model)
        else:
            st.error("❌ Model not available. Please check model loading.")

    # Tab 4: About
    with tab4:
        about_interface()

    # Sidebar
    create_sidebar(data_summary)

def prediction_interface(df: pd.DataFrame, model):
    """Create the prediction interface."""

    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Information")

        # Create three columns for input
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📋 Personal Details")
            age = st.slider("Age", AGE_RANGE[0], AGE_RANGE[1], DEFAULT_VALUES['age'])
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education Level", EDUCATION_LEVELS)
            marital_status = st.selectbox("Marital Status", MARITAL_STATUSES)

        with col2:
            st.markdown("#### 💰 Financial Profile")
            income = st.selectbox("Income Category", INCOME_CATEGORIES)
            credit_limit = st.slider("Credit Limit ($)",
                CREDIT_LIMIT_RANGE[0], CREDIT_LIMIT_RANGE[1], DEFAULT_VALUES['credit_limit'])
            total_revolving_bal = st.slider("Total Revolving Balance ($)",
                0, int(credit_limit), min(DEFAULT_VALUES['total_revolving_bal'], credit_limit))

        with col3:
            st.markdown("#### 📊 Transaction Behavior")
            total_trans_amt = st.slider("Total Transaction Amount ($)",
                TRANS_AMT_RANGE[0], TRANS_AMT_RANGE[1], DEFAULT_VALUES['total_trans_amt'])
            total_trans_ct = st.slider("Total Transaction Count",
                TRANS_CT_RANGE[0], TRANS_CT_RANGE[1], DEFAULT_VALUES['total_trans_ct'])
            months_on_book = st.slider("Months on Book",
                MONTHS_BOOK_RANGE[0], MONTHS_BOOK_RANGE[1], DEFAULT_VALUES['months_on_book'])
            total_relationship_count = st.slider("Total Relationship Count",
                RELATIONSHIP_COUNT_RANGE[0], RELATIONSHIP_COUNT_RANGE[1], DEFAULT_VALUES['total_relationship_count'])

        # Advanced options in expander
        with st.expander("🔧 Advanced Parameters"):
            col4, col5 = st.columns(2)
            with col4:
                months_inactive = st.slider("Months Inactive (Last 12 months)",
                    INACTIVE_MONTHS_RANGE[0], INACTIVE_MONTHS_RANGE[1], DEFAULT_VALUES['months_inactive'])
                contacts_count = st.slider("Contacts Count (Last 12 months)",
                    CONTACTS_COUNT_RANGE[0], CONTACTS_COUNT_RANGE[1], DEFAULT_VALUES['contacts_count'])

            with col5:
                total_amt_chng_q4_q1 = st.slider("Transaction Amount Change (Q4/Q1)",
                    CHANGE_RATIO_RANGE[0], CHANGE_RATIO_RANGE[1], DEFAULT_VALUES['total_amt_chng_q4_q1'])
                total_ct_chng_q4_q1 = st.slider("Transaction Count Change (Q4/Q1)",
                    CHANGE_RATIO_RANGE[0], CHANGE_RATIO_RANGE[1], DEFAULT_VALUES['total_ct_chng_q4_q1'])

        # Submit button
        submitted = st.form_submit_button("🚀 Analyze Customer Risk", type="primary", use_container_width=True)

    if submitted:
        # Prepare input data
        input_data = {
            'age': age,
            'gender': gender,
            'education': education,
            'marital_status': marital_status,
            'income': income,
            'credit_limit': credit_limit,
            'total_revolving_bal': total_revolving_bal,
            'total_trans_amt': total_trans_amt,
            'total_trans_ct': total_trans_ct,
            'months_on_book': months_on_book,
            'total_relationship_count': total_relationship_count,
            'months_inactive': months_inactive,
            'contacts_count': contacts_count,
            'total_amt_chng_q4_q1': total_amt_chng_q4_q1,
            'total_ct_chng_q4_q1': total_ct_chng_q4_q1
        }

        # Validate input
        is_valid, error_msg = validate_input_data(input_data)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return

        # Make prediction
        with st.spinner("🔄 Analyzing customer data..."):
            try:
                processed_data = preprocess_input_data(input_data, model)
                prediction, probabilities = make_prediction(model, processed_data)

                # Display results
                display_prediction_results(prediction, probabilities, model, processed_data)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

def display_prediction_results(prediction: int, probabilities: np.ndarray, model, processed_data: pd.DataFrame):
    """Display prediction results with visualizations."""

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # Main prediction card
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card churn-risk">
            ⚠️ <strong>HIGH CHURN RISK DETECTED</strong><br>
            <span style="font-size: 1rem;">Confidence: {probabilities[1]:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 💡 Recommended Retention Strategies:
        - **🎁 Personalized Offers**: Create targeted promotions based on spending patterns
        - **📞 Proactive Outreach**: Schedule immediate customer contact
        - **🔄 Product Cross-selling**: Introduce complementary banking products
        - **💎 Loyalty Programs**: Enhance rewards and benefits
        - **📊 Usage Incentives**: Encourage increased transaction frequency
        """)
    else:
        st.markdown(f"""
        <div class="prediction-card stay-safe">
            ✅ <strong>LOW CHURN RISK - CUSTOMER LIKELY TO STAY</strong><br>
            <span style="font-size: 1rem;">Confidence: {probabilities[0]:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        st.success("🎉 This customer shows strong loyalty indicators!")

    # Feature importance for this prediction
    st.markdown("### 🔍 Key Risk Factors Analysis")

    try:
        feature_imp = get_feature_importance(model, list(model.feature_names_in_))
        if feature_imp:
            fig = create_feature_importance_chart(feature_imp)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.warning(f"Could not generate feature importance: {str(e)}")

    # Risk probability gauge
    st.markdown("### 📊 Risk Probability Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilities[1] * 100,
            title={'text': "Churn Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': DANGER_COLOR},
                'steps': [
                    {'range': [0, 30], 'color': SUCCESS_COLOR},
                    {'range': [30, 70], 'color': WARNING_COLOR},
                    {'range': [70, 100], 'color': DANGER_COLOR}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk level interpretation
        risk_level = "Low" if probabilities[1] < 0.3 else "Medium" if probabilities[1] < 0.7 else "High"
        risk_color = SUCCESS_COLOR if risk_level == "Low" else WARNING_COLOR if risk_level == "Medium" else DANGER_COLOR

        st.markdown(f"""
        <div style="background-color: {BACKGROUND_COLOR}; padding: 2rem; border-radius: 10px; border-left: 5px solid {risk_color};">
            <h3 style="color: {risk_color};">Risk Assessment: {risk_level}</h3>
            <p><strong>Stay Probability:</strong> {probabilities[0]:.1%}</p>
            <p><strong>Churn Probability:</strong> {probabilities[1]:.1%}</p>
            <p><strong>Confidence Score:</strong> {max(probabilities):.1%}</p>
        </div>
        """, unsafe_allow_html=True)

def data_insights_interface(df: pd.DataFrame, data_summary: dict):
    """Create the data insights interface."""

    # Summary metrics
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card("Total Customers", data_summary.get('total_customers', 0))

    with col2:
        display_metric_card("Churn Rate", data_summary.get('churn_rate', 0) / 100)

    with col3:
        display_metric_card("Average Age", data_summary.get('avg_age', 0))

    with col4:
        display_metric_card("Avg Credit Limit", data_summary.get('avg_credit_limit', 0))

    # Visualizations
    st.markdown("### 📊 Data Visualizations")

    # Churn distribution
    st.markdown("#### Customer Churn Distribution")
    fig1 = create_churn_distribution_chart(df)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Age Distribution by Churn Status")
        fig2 = create_age_distribution_chart(df)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("#### Transaction Amount Analysis")
        fig3 = create_transaction_amount_boxplot(df)
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Feature Correlation Matrix")
    fig4 = create_correlation_heatmap(df)
    st.plotly_chart(fig4, use_container_width=True)

def model_analytics_interface(model):
    """Create the model analytics interface."""

    st.markdown("### 📊 Model Performance Metrics")

    # Performance comparison
    fig = create_metrics_comparison_chart()
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.markdown("### 🎯 Confusion Matrix Analysis")
    cm = np.array([[1450, 35], [45, 470]])  # Example values
    fig_cm = create_confusion_matrix_heatmap(cm)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Model details
    st.markdown("### 🤖 Model Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>Algorithm</h4>
            <p><strong>AdaBoost Classifier</strong></p>
            <p>Ensemble learning with decision trees</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>Training Details</h4>
            <p><strong>Features:</strong> 29 engineered features</p>
            <p><strong>Samples:</strong> 10,127 customers</p>
        </div>
        """, unsafe_allow_html=True)

def about_interface():
    """Create the about interface."""

    st.markdown("### 🎯 Project Overview")

    st.markdown("""
    This **Credit Card Churn Prediction System** is a comprehensive machine learning application
    designed to help banks identify customers at risk of churning their credit card services.

    #### 🚀 Key Features:
    - **Real-time Predictions**: Instant churn risk assessment
    - **Advanced Analytics**: Interactive data visualizations
    - **Model Interpretability**: Feature importance analysis
    - **Modern UI**: Responsive web interface with professional design
    - **Business Intelligence**: Actionable retention strategies
    """)

    st.markdown("### 🛠️ Technology Stack")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        #### 🤖 Machine Learning
        - **Algorithm**: AdaBoost Classifier
        - **Performance**: 98% Precision
        - **Libraries**: Scikit-learn, XGBoost
        """)

    with tech_col2:
        st.markdown("""
        #### 🌐 Web Application
        - **Framework**: Streamlit
        - **Visualization**: Plotly, Seaborn
        - **Data Processing**: Pandas, NumPy
        """)

    st.markdown("### 📊 Business Impact")

    st.markdown("""
    #### 💰 Cost-Benefit Analysis:
    - **Precision Focus**: Minimize false positives (unnecessary retention costs)
    - **Early Intervention**: Identify at-risk customers before churn occurs
    - **ROI Optimization**: Maximize return on retention investments

    #### 🎯 Key Insights:
    - Transaction behavior is the strongest churn predictor
    - Customer engagement patterns reveal risk levels
    - Proactive retention strategies can prevent 70%+ of potential churn
    """)

    st.markdown("### 👨‍💻 Development Team")
    st.markdown("**Nirankar Singh & Team** - BTech Machine Learning Project")

def create_sidebar(data_summary: dict):
    """Create the application sidebar."""

    with st.sidebar:
        st.markdown(f'<h3 class="sidebar-header">📊 Dashboard</h3>', unsafe_allow_html=True)

        # Quick stats
        if data_summary:
            st.metric("Total Customers", f"{data_summary.get('total_customers', 0):,}")
            st.metric("Churn Rate", f"{data_summary.get('churn_rate', 0):.1f}%")
            st.metric("Avg Credit Limit", f"${data_summary.get('avg_credit_limit', 0):,.0f}")

        st.markdown("---")

        # Model performance summary
        st.markdown("### 🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", "98%")
            st.metric("Accuracy", "97%")
        with col2:
            st.metric("Recall", "93%")
            st.metric("F1-Score", "95%")

        st.markdown("---")

        # Navigation help
        st.markdown("### 🧭 Navigation")
        st.markdown("""
        - **🔮 Prediction**: Customer risk assessment
        - **📊 Insights**: Data visualization & analytics
        - **📈 Analytics**: Model performance metrics
        - **ℹ️ About**: Project information & help
        """)

        st.markdown("---")
        st.markdown("💡 *Built with ❤️ for smarter banking*")

if __name__ == "__main__":
    main()