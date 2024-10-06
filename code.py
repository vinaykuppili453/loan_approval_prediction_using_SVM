#!/usr/bin/env python
# coding: utf-8

# In[42]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


# In[55]:


#load and inspect data
df = pd.read_excel(r"E:\clp project\LoanDataModified.xlsx") 


# In[56]:


df.head(10)


# In[58]:


df.tail(10)


# In[59]:


df.info()


# In[60]:


df.isnull().sum()


# In[61]:


#handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['LoanAmount_in_lakhs'].fillna(df['LoanAmount_in_lakhs'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[62]:


df.isnull().sum()


# In[63]:


#Engineering new features
df['loanAmount_log'] = np.log(df['LoanAmount_in_lakhs'])
df['TotalIncome'] = df['ApplicantIncome(lakhs)'] + df['CoapplicantIncome(lakhs)']
df['TotalIncome_log'] = np.log(df['TotalIncome'])


# In[64]:


#converting categorical features into strings
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df[categorical_features] = df[categorical_features].astype(str)


# In[65]:


#Define features and target.
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
X



# In[66]:


y = df['Loan_Status']
y


# In[107]:


#Split data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[108]:


#Balance the classes using SMOTE.
numerical_features = ['ApplicantIncome(lakhs)', 'CoapplicantIncome(lakhs)', 'LoanAmount_in_lakhs', 'Loan_Amount_Term', 'Credit_History', 'loanAmount_log', 'TotalIncome', 'TotalIncome_log']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


# In[109]:


#Balance the classes using SMOTE.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))
])


# In[110]:


#Set up and perform grid search to find the best hyperparameters.
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}


# In[111]:


#Train the model and make predictions.
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


# In[112]:


grid_search.fit(x_train, y_train)


# In[113]:


best_model = grid_search.best_estimator_


# In[114]:


y_pred = best_model.predict(x_test)


# In[115]:


y_probs = best_model.predict_proba(x_test)[:, 1]


# In[116]:


#Evaluate model performance using various metrics.
accuracy = accuracy_score(y_test, y_pred)


# In[117]:


conf_matrix = confusion_matrix(y_test, y_pred)


# In[118]:


print("Best model parameters:", grid_search.best_params_)


# In[119]:


print("Accuracy on test data:", accuracy)


# In[120]:


print("Confusion Matrix:\n", conf_matrix)


# In[121]:


print("Accuracy:", accuracy)


# In[122]:


print("Precision:", precision_score(y_test, y_pred, pos_label='Y'))


# In[123]:


print("Recall:", recall_score(y_test, y_pred, pos_label='Y'))


# In[124]:


print("F1 Score:", f1_score(y_test, y_pred, pos_label='Y'))


# In[125]:


print(classification_report(y_test, y_pred))


# In[126]:


#Plot the ROC curve to visualize model performance.
fpr, tpr, thresholds = roc_curve(y_test.map({'N': 0, 'Y': 1}), y_probs)  # Convert labels to 0 and 1 for ROC
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




