### Project README: Enterprise Customer Churn Analytics Platform

#### Overview

This repository hosts a production-grade machine learning platform designed to identify customer attrition signatures. The system integrates a serialized Support Vector Machine (SVM) pipeline that processes demographic and behavioral input vectors to calculate real-time churn risk. The platform provides both a real-time diagnostic interface and a batch processing studio for enterprise-scale risk audits.

#### Business Impact

Retention management is critical for operational profitability. By identifying high-risk account profiles, this platform allows organizations to deploy targeted retention incentives and customer success touchpoints before subscription termination, directly mitigating revenue leakage.

#### Machine Learning Pipeline Architecture

The predictive engine employs a disciplined machine learning workflow consisting of data cleaning, categorical encoding, feature scaling, and hyperparameter optimization.

#### Model Benchmarking & Performance

The Support Vector Classifier was selected as the champion model following a 5-fold cross-validation grid search, demonstrating superior geometric decision boundary optimization over ensemble methods for this specific feature set.

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Support Vector Classifier (SVC) | 0.9450 | 0.9421 | 0.9510 | 0.9465 |
| Logistic Regression | 0.9400 | 0.9380 | 0.9440 | 0.9410 |
| K-Nearest Neighbors (KNN) | 0.9100 | 0.9050 | 0.9120 | 0.9085 |
| Random Forest | 0.9000 | 0.8990 | 0.9020 | 0.9005 |
| Decision Tree | 0.8900 | 0.8850 | 0.8920 | 0.8885 |

#### Analytical Visualizations

The Batch Exploratory module automatically generates two primary visual aids to facilitate data-driven decision making.

1. Behavioral Density Maps: These Kernel Density Estimate plots illustrate how feature variables like Monthly Charges and Tenure correlate with attrition probability.
2. Model Confusion Matrix: This heatmap visualizes systematic prediction errors, providing clear visibility into False Negatives and False Positives to ensure diagnostic accuracy.

#### How to Add Custom Visualizations

To integrate additional visualizations into your dashboard, locate the Batch Exploratory section in app.py. You can define new chart objects using Plotly or Matplotlib and map them to specific data frames.

Code Insertion Pattern:

# 1. Initialize figure container

fig = plt.figure(figsize=(10, 5))

# 2. Render plot using Seaborn or Matplotlib

sns.boxplot(data=raw_df, x='Churn', y='MonthlyCharges')

# 3. Render into Streamlit container

st.pyplot(fig)

#### Operational Workflow

1. Environment Setup: Install project dependencies specified in requirements.txt.
2. Asset Deployment: Ensure model.pkl and scaler_churn.pkl are located in the application root.
3. Execution: Run the platform via the command "streamlit run app.py".
4. Usage: Input account parameters in the diagnostic portal or upload historic database CSVs for batch risk auditing and targeted list generation.

#### Technical Specifications

The application architecture is optimized for cloud deployment and utilizes joblib for binary asset serialization, ensuring sub-millisecond inference speeds during real-time account risk assessments. All categorical data is pre-encoded to ensure compliance with linear algebra computational requirements.
