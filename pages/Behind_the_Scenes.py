import streamlit as st
from RegressionModel import RegressionModel

st.write("Behind the Scenes Look")

model = RegressionModel()

model.draw_visualizations()
model.predict_test_data()
