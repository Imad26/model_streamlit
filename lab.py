import pickle
import streamlit as st
import pandas as pd

# Display app logo and title
st.image("https://seeklogo.com/images/E/ecole-hassania-des-travaux-publics-ehtp-logo-3D5770F217-seeklogo.com.png")
st.title("MDSE Machine Learning Course")
st.subheader("Iris Flower Prediction App")
st.markdown("This app predicts the type of iris flower based on input features.")

# User selection
st.multiselect('How would you like to use the prediction model', ['Jupiter', 'Mars', 'Neptune'])
st.markdown("<br>", unsafe_allow_html=True)
st.multiselect('How would you like to use the prediction model', ['Input the parameters directly', 'Load the data'])

# Sidebar for user input parameters
st.sidebar.title("")
st.sidebar.image("https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg")
st.sidebar.header("User Input Parameters")

# Input sliders
st.sidebar.markdown('Sepal length')
sep_l = st.sidebar.slider('Pick a number', 0, 10)

st.sidebar.markdown('Sepal width')
sep_w = st.sidebar.slider('Pick a number', 0, 6)

st.sidebar.markdown('Petal length')
pet_l = st.sidebar.slider('Pick a number', 0, 9)

st.sidebar.markdown('Petal width')
pet_w = st.sidebar.slider('Pick a number', 0, 7)

# Display user inputs
st.subheader("User Input Parameters")
data = pd.DataFrame([[sep_l, sep_w, pet_l, pet_w]], columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
st.write(data)

# Load the model
model = pickle.load(open('modeliris6.pkl', 'rb'))

# Make prediction
type = model.predict(data)
st.subheader("Prediction")
st.write(type)
