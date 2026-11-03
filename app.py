# Streamlit interface for entering passenger details and displaying predictions.
import streamlit as st
import pickle
import pandas as pd


df = pd.read_csv('data/train.csv')


def load_model():
    with open('spaceship-titanic-model.pkl', 'rb') as f:
        return pickle.load(f)

st.header("Spaceship Titanic Transport Prediction App")
st.write("Enter passenger details and luxury amenity spending to predict whether they were transported to an alternate dimension.")

with st.sidebar:
    st.image("parami.jpg")
    st.subheader("Student Information")
    st.text("Student Name: Htet Khant Linn")
    st.text("Student ID: PIUS20230064")
    st.text("Course: CS 300 - Introduction to Machine Learning")
    st.text("Spaceship Titanic Transport Prediction App")
    st.text("Model: Tuned Gradient Boosting")
    st.text("Test Accuracy: 80.24%")

col1, col2 = st.columns(2)

with col1:
    pass

with col2:
    pass

