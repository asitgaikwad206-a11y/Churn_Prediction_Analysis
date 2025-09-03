
# Customer Churn Prediction

Overview:
This project predicts customer churn using machine learning and visual analytics.  
The dataset is from a telecom company, where the goal is to identify customers likely to leave ("churn") and provide actionable insights for retention strategies.

Objectives:
Customer churn is a critical challenge for subscription-based businesses.  
The objective of this project is to:  
- Identify key drivers of churn.  
- Build a predictive model using Logistic Regression.  
- Visualize churn patterns using Tableau dashboards.  
- Provide business insights to reduce churn and improve retention.

Workflow:
 1. Data Cleaning (handling missing values, encoding categorical variables).  
2. Exploratory Data Analysis (EDA).  
3. Feature Engineering.  
4. Logistic Regression Model.  
5. Model Evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC).  
6. Business Insights from Tableau Dashboard.

Evaluation:
- Accuracy: ~49%  
- Precision (Churn): 0.48  
- Recall (Churn): 0.39  
- ROC-AUC: 0.48  

⚠️ Insight: Logistic Regression alone did not perform well.  
This indicates churn is complex and may require advanced models (Random Forest, XGBoost, or SMOTE for class imbalance).

Visualization:
The interactive Tableau dashboard includes:  
- Churn vs Monthly Charges  
- Churn vs Tenure  
- Churn vs Contract Type  
- Payment Method vs Churn  
- Actual vs Predicted Churn (from ML Model)  

[Tableau Public Dashboard Link](https://public.tableau.com/views/churnanalysisdashboard_17569135607190/ChurnAnalysisDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

Insights:
- Month-to-Month customers have the highest churn rate.  
- Customers with higher monthly charges are more likely to churn.  
- Tenure is inversely related to churn (new customers churn more).  
- Electronic check users have higher churn compared to other payment methods.

Tools used:
-  Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn,Machine Learning-Logistic Regression)  
- Tableau for Visualization  
- Google Colab

Future enhancements:
- Test with Random Forest, XGBoost, and Ensemble models.  
- Use SMOTE for handling class imbalance.  
- Deploy the model as a Flask/Django web app.  

confusion matrix:
<img width="444" height="393" alt="image" src="https://github.com/user-attachments/assets/3e7ee2fb-c34d-4899-8bd8-9e6cf1259815" />

ROC Curve-Logistic Regression
<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/7986bbd2-2377-490b-b3d3-e9fa5e78d612" />

Press Recall Curve-Logistic Rgeression
<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/5dd53c49-fa4e-468e-93d5-49022509ee4a" />

Top features with the most impact for churns
<img width="767" height="547" alt="image" src="https://github.com/user-attachments/assets/7aaf5652-e3e6-4a85-afdc-b1ef5dfad7d8" />


