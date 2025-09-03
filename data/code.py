#👥 Customer Churn Prediction 🏃‍♂️ EDA 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc

#loading the data
df=pd.read_csv("customer_churn_data.csv")

#data preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
print(df.dtypes)
df_encoded=df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col != 'Churn':
        df_encoded[col]=LabelEncoder().fit_transform(df_encoded[col])
#target variable
df_encoded['Churn']=df_encoded['Churn'].map({'No':0 , 'Yes':1})

#features and target
x=df_encoded.drop('Churn',axis=1)
y=df_encoded['Churn']

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#standardscaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Logistic Regression
model=LogisticRegression(max_iter=1000,random_state=42)
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)

#evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:\n",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(x_test)[:,1]))


#visualization
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap="Blues")
plt.xlabel("Predicted ")
plt.ylabel("Actual ")
plt.legend()
plt.title("Confusion Matrix")
plt.show

#ROC(LOGISTIC REGRESSION)
y_prob = log_reg.predict_proba(X_test)[:,1]  # Probabilities for positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

#PRECISION RECALL CURVE
prec, rec, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(rec, prec, color="green", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
import pandas as pd
import numpy as np

# Get feature names from the original DataFrame
feature_names = x.columns

# Get coefficients from trained logistic regression model
coefficients = model.coef_[0]

# Create dataframe for interpretation
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Impact': np.exp(coefficients)  # odds ratio
})

# Sort by absolute impact
feature_importance['AbsCoeff'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='AbsCoeff', ascending=False)

print(feature_importance[['Feature', 'Coefficient', 'Impact']].head(15))

#Positive Coefficient → Increases likelihood of churn (customers with higher values of this feature are more likely to churn).
#Negative Coefficient → Decreases likelihood of churn (protective factor).
#Impact (Odds Ratio)
#1 → increases odds of churn.
#<1 → decreases odds of churn.

import matplotlib.pyplot as plt

top_features = feature_importance.head(15).sort_values(by='Coefficient')

plt.figure(figsize=(8,6))
plt.barh(top_features['Feature'], top_features['Coefficient'], color='teal')
plt.xlabel("Impact on Churn(Coefficient)")
plt.title("Features Impacting Churn")
plt.show()

