import streamlit as st
from ClassificationModel import ClassificationModel

st.header("Visualizations")

loading = st.progress(0, text="Loading up the dataset")
model = ClassificationModel()

loading.progress(20, text="Drawing correlation matrix")
model.draw_correlation_matrix()

loading.progress(40, text="Drawing distribution graphs")
model.draw_distribution_graphs(loading)

loading.progress(70, text="Drawing percentage-based graphs")
model.draw_percentage_graphs()

loading.progress(90, text="Drawing confusion matrix and accuracy score")
model.predict_test_data()

loading.empty()
