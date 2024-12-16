# ## Importing required libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

## Importing Data

insurance_data = pd.read_csv('insurance.csv')

## Feature Engineering

encoder = OneHotEncoder()
categorical_features = ['sex','smoker','region']
encoded_features = encoder.fit_transform(insurance_data[categorical_features]).toarray()
categorical_fearure_names = encoder.get_feature_names_out(categorical_features)
categorical_encoded_features = pd.DataFrame(encoded_features,columns=categorical_fearure_names)
numerical_features = insurance_data[['age','bmi','children']]
final_data = pd.concat([numerical_features,categorical_encoded_features,insurance_data['charges']],axis=1)

## Model Building & Evaluation

# Data Splitting in in-dependant & dependant
X = final_data.drop(['charges'],axis=1)
y = final_data['charges']

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=12)

# Model Building
model = LinearRegression()

# Model Training
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
# print("R2 Score : {}\nMean Squared Error: {}\nmean_absolute_error : {}\nroot_mean_squared_error: {}".format(r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),root_mean_squared_error(y_test,y_pred)))

## Building Streamlit front end
st.title('Insurance Charges Prediction')

# Printing Model Performance Metrics
st.subheader("Model Performance Metrics")
st.write("R2 Score: {}".format(r2_score(y_test,y_pred)))
st.write("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))

## Taking Input from the user
st.subheader('Predict Insurance Charges')
age = st.number_input("Age",min_value=0,max_value=100,step=1)
sex = st.selectbox("Sex",["Male","Female"])
bmi = st.number_input("BMI",min_value=1.0, value=25.0,max_value=100.0,step=0.1)
children = st.number_input("Number of Children",min_value=0,value=0,max_value=10,step=1)
smoker = st.selectbox("Smoker",['Yes','No'])
region = st.selectbox("Region",['southwest', 'southeast', 'northwest', 'northeast'])

## Process user Input

# Convert user input to dataframe
user_data = pd.DataFrame({'age':[age],'sex':[sex],'bmi':[bmi],'children':[children],'smoker':[smoker],'region':[region]})

# Convert input to lower case
user_data['sex'] = user_data['sex'].str.lower()
user_data['smoker'] = user_data['smoker'].str.lower()

## Align user input with training data

missing_columns = set(categorical_fearure_names) - set(encoder.get_feature_names_out(categorical_features))
user_encoded = encoder.transform(user_data[categorical_features]).toarray()
user_encoded_df = pd.DataFrame(user_encoded,columns=categorical_fearure_names)

## Creating Final Data

for col in missing_columns:
    user_encoded_df[col] = 0

user_final = pd.concat([user_data,user_encoded_df],axis=1)
user_final.drop(['sex','smoker','region'],axis=1,inplace=True)

if st.button("Predict"):
    prediction = model.predict(user_final)
    st.write("Predicted Insurance Charges : ",prediction[0])