# iris_streamlit_app.py

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit app title and description
st.title("Iris Species Prediction App")
st.write("This app predicts the species of Iris based on input sepal and petal dimensions.")

# Create sliders for user input
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()))

# Predict button
if st.button("Predict Species"):
    # Predict the species
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_species = target_names[prediction[0]]

    # Display the result
    st.write(f"The predicted species is: **{predicted_species}**")
