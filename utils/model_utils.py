"""
Model loading and prediction utilities for the churn prediction application.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Path):
    """
    Load the trained machine learning model with error handling.

    Args:
        model_path (Path): Path to the model file

    Returns:
        Trained model or None if failed
    """
    try:
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            logger.error(f"Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input_data(input_data: Dict[str, Any], model) -> pd.DataFrame:
    """
    Preprocess user input data to match model requirements.

    Args:
        input_data (dict): Raw user input data
        model: Trained model to get feature names from

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    try:
        # Create base dataframe
        df = pd.DataFrame([input_data])

        # Rename columns to match training data format
        column_mapping = {
            'age': 'Customer_Age',
            'gender': 'Gender',
            'education': 'Education_Level',
            'marital_status': 'Marital_Status',
            'income': 'Income_Category',
            'credit_limit': 'Credit_Limit',
            'total_revolving_bal': 'Total_Revolving_Bal',
            'total_trans_amt': 'Total_Trans_Amt',
            'total_trans_ct': 'Total_Trans_Ct',
            'months_on_book': 'Months_on_book',
            'total_relationship_count': 'Total_Relationship_Count',
            'months_inactive': 'Months_Inactive_12_mon',
            'contacts_count': 'Contacts_Count_12_mon',
            'total_amt_chng_q4_q1': 'Total_Amt_Chng_Q4_Q1',
            'total_ct_chng_q4_q1': 'Total_Ct_Chng_Q4_Q1'
        }

        df = df.rename(columns=column_mapping)

        # Add calculated features
        df['Avg_Open_To_Buy'] = df['Credit_Limit'] - df['Total_Revolving_Bal']
        df['Avg_Utilization_Ratio'] = df['Total_Revolving_Bal'] / df['Credit_Limit']

        # Handle potential division by zero
        df['Avg_Utilization_Ratio'] = df['Avg_Utilization_Ratio'].fillna(0)

        # Set default values for missing categorical features
        df['Dependent_count'] = 2  # Default
        df['Card_Category'] = 'Blue'  # Default

        # One-hot encode categorical variables (same as training)
        categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Get expected features from the model
        expected_features = list(model.feature_names_in_)

        # Create a dataframe with all expected features, defaulting to 0
        final_df = pd.DataFrame(0.0, index=[0], columns=expected_features)

        # Fill in the values we have
        for col in df_encoded.columns:
            if col in final_df.columns:
                value = df_encoded[col].iloc[0]
                # Convert to float if it's not already
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0  # Default for non-numeric strings
                final_df[col] = float(value)

        logger.info("Input data preprocessing completed successfully")
        return final_df

    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise

def make_prediction(model, input_data: pd.DataFrame) -> Tuple[int, np.ndarray]:
    """
    Make churn prediction using the trained model.

    Args:
        model: Trained machine learning model
        input_data (pd.DataFrame): Preprocessed input data

    Returns:
        tuple: (prediction, prediction_probabilities)
    """
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        logger.info(f"Prediction made successfully: {prediction[0]}")
        return prediction[0], prediction_proba[0]

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def get_feature_importance(model, feature_names: list) -> Dict[str, float]:
    """
    Extract feature importance from the model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names

    Returns:
        dict: Feature importance mapping
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance_values))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}

    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return {}

def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate user input data.

    Args:
        input_data (dict): User input data

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check credit limit
        if input_data.get('credit_limit', 0) <= 0:
            return False, "Credit limit must be greater than 0"

        # Check revolving balance
        if input_data.get('total_revolving_bal', 0) > input_data.get('credit_limit', 0):
            return False, "Total revolving balance cannot exceed credit limit"

        # Check age range
        if not 18 <= input_data.get('age', 0) <= 100:
            return False, "Age must be between 18 and 100"

        # Check transaction amounts
        if input_data.get('total_trans_amt', 0) < 0:
            return False, "Total transaction amount cannot be negative"

        if input_data.get('total_trans_ct', 0) < 0:
            return False, "Total transaction count cannot be negative"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"

@st.cache_resource
def load_model_cached(model_path: Path):
    """
    Cached function to load the model.

    Args:
        model_path (Path): Path to the model file

    Returns:
        Trained model
    """
    return load_model(model_path)