from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import pickle


#----Prediction-----
def predict_risk(model, df):
    X_sc = scale_data(df)
    prediction = model.predict(X_sc)
    
    return prediction

#-----Cleaning-----
def scale_data(df):
    X_sc = scaler.transform(df)
    return X_sc

st.image('insurance_AdobeStock_269968279.webp')
st.title('Health Insurance Indicator')
st.subheader('Check to see what your premiums might be & recommendations to reduce them')
st.info('*Disclaimer* This is based of an analysis of *data* and should NOT be taken as financial advice. No input responses are saved.')

#------Sidebar Input-----
st.sidebar.title('Fill out your details')
age = st.sidebar.number_input('Age', 18, 100)
sex = st.sidebar.radio('Sex', ['female', 'male'])
if sex =='male':
    sex = 1
else:
    sex = 0
weight = st.sidebar.number_input('Weight (lbs)', 100, 1000)
height = st.sidebar.number_input('Height (Inches)', 55, 1000)
bmi = (weight*703)/(height**2)
children = st.sidebar.number_input('Children', 0, 100)
smoker = st.sidebar.radio('Smoker?', ['yes', 'no'])
if smoker =='yes':
    smoker = 0
else:
    smoker = 1
#region = st.sidebar.radio('Region?', ['northeast', 'southeast', 'southwest', 'northwest'])



#----Model Loading-----
model = pickle.load(open('rforest2.pickle', 'rb'))
scaler = pickle.load(open('standard_scalar.pickle', 'rb'))

#----Map features-------
features = {'Age' : age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker
}

#-----Convert to dataframe----
features_df  = pd.DataFrame([features])

if st.sidebar.button('Submit'):
    
    prediction = predict_risk(model, features_df)
    prediction = np.round(prediction, 0)
    st.subheader(f"Premium: ${int(prediction)}")