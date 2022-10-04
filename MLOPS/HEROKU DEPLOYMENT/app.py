import streamlit as st
from pickle import load
import numpy as np

st.title("Diamond Price Prediction")

regressor_rf = load(open('model/rf_model.pkl', 'rb'))
scaler = load(open('model/scaler.pkl', 'rb'))

clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}

carat = st.number_input('Carat range 0.2-5.01')
depth = st.number_input('Depth range 43-79')
table = st.number_input('Table range 43-95')
x = st.number_input('x range 0-10.7')
y = st.number_input('y range 0-58.9')
z = st.number_input('z range 0-31.8')
cut = st.selectbox(
     'How would be the cut of Diamond?',
     ('Fair', 'Good', 'Very Good', 'Ideal', 'Premium'))

color = st.selectbox(
     'What should be the color of Diamond?',
     ('J', 'I', 'H', 'G', 'F', 'E', 'D'))

clarity = st.selectbox(
     'How would you like to be contacted?',
     ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))

num_scal=scaler.transform([[1.16 ,61.5 ,55.0 ,6.75 ,6.81 ,4.17]])

cat_encod=np.array([clarity_encoder[clarity],color_encoder[color],cut_encoder[cut]])

if st.button("Predict")==True:
    prediction=regressor_rf.predict(np.concatenate((cat_encod, num_scal.flatten()), axis=None).reshape(1,-1)).item()
    st.write(prediction)
    st.balloons()
