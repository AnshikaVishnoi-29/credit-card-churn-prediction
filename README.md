# Credit Card Churn Prediction Using Machine Learning

## Project Overview

This project leverages Machine Learning techniques to predict customer churn in the banking sector. By analyzing customer demographics, account information, and transaction behavior, the model identifies customers who are likely to discontinue their credit card services.

The objective is to help financial institutions proactively identify at-risk customers and implement effective retention strategies to reduce churn and improve customer lifetime value.

---

## Business Problem

Customer churn is a major challenge for financial institutions. Retaining existing customers is significantly more cost-effective than acquiring new ones. Predicting customer churn enables banks to take preventive actions, improve customer satisfaction, and increase profitability.

---

## Project Objectives

* Analyze customer demographics and transaction behavior
* Identify factors contributing to customer churn
* Build and compare multiple machine learning models
* Predict customers at risk of leaving
* Support business decision-making through predictive analytics

---

## Dataset Description

The dataset contains customer demographic, account, and transaction-related information, including:

* Customer Age
* Gender
* Education Level
* Marital Status
* Income Category
* Credit Limit
* Months on Book
* Total Relationship Count
* Total Transaction Amount
* Total Transaction Count
* Revolving Balance
* Average Utilization Ratio

---

## Technology Stack

### Programming Language

* Python

### Data Analysis & Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* LightGBM
* Random Forest Classifier
* Gradient Boosting Classifier

### Model Deployment

* Streamlit

### Model Serialization

* Joblib

---

## Machine Learning Models

Multiple machine learning algorithms were trained and evaluated to identify the most effective solution for customer churn prediction.

### Models Evaluated

* Random Forest Classifier
* Gradient Boosting Classifier
* LightGBM Classifier

### Why Ensemble Models?

Ensemble learning algorithms were selected because they:

* Improve prediction accuracy
* Reduce overfitting
* Handle complex customer behavior patterns
* Capture non-linear relationships effectively
* Deliver robust performance on structured business data

---

## Model Evaluation

The machine learning models were evaluated using industry-standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

The final model was selected after comparing the performance of all trained models on the test dataset. The selected model demonstrated strong predictive capability in identifying customers at risk of churn while maintaining a balance between precision and recall.

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Development
6. Model Training
7. Model Evaluation
8. Customer Churn Prediction
9. Interactive Application Deployment

---

## Key Features

* Customer churn prediction using Machine Learning
* Exploratory Data Analysis (EDA)
* Data preprocessing and transformation
* Feature engineering
* Multiple model comparison
* Customer risk identification
* Interactive prediction interface
* Business-focused insights and recommendations

---

## Project Structure

```text
credit-card-churn-prediction/
│
├── app/
├── config/
├── models/
├── utils/
├── web_user/
│
├── Churn_Project_Notebook.ipynb
├── run.py
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-churn-prediction.git
```

### Navigate to the Project Directory

```bash
cd credit-card-churn-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python run.py
```

---

## Application Preview

Add screenshots of the application, prediction results, and visualizations here.

```markdown
![Application Preview](app-preview.png)
```

---

## Business Impact

This project helps organizations:

* Identify customers likely to churn
* Improve customer retention strategies
* Reduce customer attrition
* Increase customer lifetime value
* Support data-driven business decisions

---

## Skills Demonstrated

* Machine Learning
* Customer Churn Prediction
* Predictive Analytics
* Random Forest
* Gradient Boosting
* LightGBM
* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Evaluation
* Python Programming
* Streamlit Development
* Business Intelligence
* Data-Driven Decision Making

---

## Future Enhancements

* Hyperparameter Optimization
* XGBoost Integration
* Real-Time Prediction Pipeline
* Cloud Deployment
* Interactive Business Dashboard Integration

---

## Author

### Anshika Vishnoi

Aspiring Data Analyst | Machine Learning Enthusiast | Power BI Developer

Passionate about transforming data into actionable insights through analytics, machine learning, and business intelligence solutions.
