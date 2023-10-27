import streamlit as st
from RegressionModel import RegressionModel

st.header("Visualizations")

model = RegressionModel()

model.draw_visualizations()
model.predict_test_data()
