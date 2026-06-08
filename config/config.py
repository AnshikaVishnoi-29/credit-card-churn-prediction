"""
Configuration settings for the Credit Card Churn Prediction application.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"
STATIC_DIR = PROJECT_ROOT / "static"

# File paths
DATASET_PATH = PROJECT_ROOT / "web_user" / "BankChurners.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# Model configuration
MODEL_FEATURES = [
    'Customer_Age', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
    'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio', 'Gender_M', 'Education_Level_Doctorate',
    'Education_Level_Graduate', 'Education_Level_High School',
    'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
    'Education_Level_Unknown', 'Marital_Status_Married',
    'Marital_Status_Single', 'Marital_Status_Unknown',
    'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
    'Income_Category_$80K - $120K', 'Income_Category_Less than $40K',
    'Income_Category_Unknown', 'Card_Category_Gold', 'Card_Category_Platinum',
    'Card_Category_Silver'
]

# Application settings
APP_TITLE = "💳 Credit Card Churn Prediction System"
APP_ICON = "💳"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# UI Colors and Styling
PRIMARY_COLOR = "#1f77b4"
SUCCESS_COLOR = "#4CAF50"
WARNING_COLOR = "#FF9800"
DANGER_COLOR = "#F44336"
BACKGROUND_COLOR = "#f8f9fa"

# Model performance metrics (for display)
MODEL_METRICS = {
    'accuracy': 0.97,
    'precision': 0.98,
    'recall': 0.93,
    'f1_score': 0.95
}

# Default input values
DEFAULT_VALUES = {
    'age': 45,
    'credit_limit': 10000,
    'total_revolving_bal': 1000,
    'total_trans_amt': 5000,
    'total_trans_ct': 70,
    'months_on_book': 36,
    'total_relationship_count': 3,
    'months_inactive': 2,
    'contacts_count': 2,
    'total_amt_chng_q4_q1': 0.7,
    'total_ct_chng_q4_q1': 0.7
}

# Categorical options
INCOME_CATEGORIES = [
    'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'
]

EDUCATION_LEVELS = [
    'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'
]

MARITAL_STATUSES = ['Single', 'Married', 'Divorced']

CARD_CATEGORIES = ['Blue', 'Silver', 'Gold', 'Platinum']

# Slider ranges
AGE_RANGE = (18, 100)
CREDIT_LIMIT_RANGE = (1000, 50000)
TRANS_AMT_RANGE = (500, 20000)
TRANS_CT_RANGE = (10, 150)
MONTHS_BOOK_RANGE = (1, 60)
RELATIONSHIP_COUNT_RANGE = (1, 6)
INACTIVE_MONTHS_RANGE = (0, 6)
CONTACTS_COUNT_RANGE = (0, 6)
CHANGE_RATIO_RANGE = (0.0, 4.0)

# Chart colors
CHART_COLORS = {
    'existing': '#4CAF50',
    'attrited': '#F44336',
    'primary': '#1f77b4',
    'secondary': '#FF9800'
}