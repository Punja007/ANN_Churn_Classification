import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

model = tf.keras.models.load_model('model.h5')
with open("onehot_encoder_geo.pkl", 'rb') as file:
    onehot_encoder_geo = pkl.load(file)

with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pkl.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pkl.load(file)


## Streamlit App
st.title("Customer Churn Prediction")

##inputs
Credit_score = st.number_input("Credit Score")
Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender", label_encoder_gender.classes_)
Age = st.slider("Age", 18, 100)
Tenure = st.slider("Tenure", 0, 10)
Balance = st.number_input("Balance")
NoOfProducts = st.slider("No Of Products", 0, 5)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
Estimated_Salary = st.number_input("Estimated Salary")



# taking the inputs
input_data={
    'CreditScore' : [Credit_score],
    'Gender' : [label_encoder_gender.transform([Gender])[0]],
    'Age' : [Age],
    'Tenure' : [Tenure],
    'Balance' : [Balance],
    'NumOfProducts' : [NoOfProducts],
    'HasCrCard' : [HasCrCard],
    'IsActiveMember' : [IsActiveMember],
    'EstimatedSalary' : [Estimated_Salary]
}

geo_encoded = onehot_encoder_geo.transform([[Geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

input_data = pd.DataFrame(input_data)
data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scaling the data
Scaled_data = scaler.transform(data)


#model prediction 
pred = model.predict(Scaled_data)
pred_proba = pred[0][0]

st.write(pred_proba)
if pred_proba>0.5:
    st.write("The Customer is likely to Churn")

else:
    st.write("The Customer not is likely to Churn")