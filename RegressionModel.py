import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


class RegressionModel:
    def __init__(self):
        # Load data in
        self.data = pd.read_csv("data/diabetes_binary_health_indicators.csv")

        # PRE-PROCESSING
        # Convert float types to int
        self.data = self.data.astype({
            "Diabetes_binary": "int32",
            "HighBP": "int32",
            "HighChol": "int32",
            "CholCheck": "int32",
            "BMI": "int32",
            "Smoker": "int32",
            "Stroke": "int32",
            "HeartDiseaseorAttack": "int32",
            "PhysActivity": "int32",
            "Fruits": "int32",
            "Veggies": "int32",
            "HvyAlcoholConsump": "int32",
            "AnyHealthcare": "int32",
            "NoDocbcCost": "int32",
            "GenHlth": "int32",
            "MentHlth": "int32",
            "PhysHlth": "int32",
            "DiffWalk": "int32",
            "Sex": "int32",
            "Age": "int32",
            "Education": "int32",
            "Income": "int32",
        })

        # Rename columns for better readability in charts
        self.data = self.data.rename(columns={
            "Diabetes_binary": "Diabetes",
            "HighBP": "High Blood Pressure",
            "HighChol": "High Cholesterol",
            "CholCheck": "Recent Cholesterol Check",
            "BMI": "BMI",
            "Smoker": "History of Smoking",
            "Stroke": "History of Stroke",
            "HeartDiseaseorAttack": "History of Heart Disease",
            "PhysActivity": "Physical Activity",
            "Fruits": "Eating Fruits",
            "Veggies": "Eating Veggies",
            "HvyAlcoholConsump": "Alcohol Consumption",
            "AnyHealthcare": "Health Insurance",
            "NoDocbcCost": "Inability to Afford Medical Care",
            "GenHlth": "Poor General Health",
            "MentHlth": "Poor Mental Health",
            "PhysHlth": "Poor Physical Health",
            "DiffWalk": "Difficulty Walking",
            "Sex": "Male",
            "Age": "Age",
            "Education": "Education",
            "Income": "Income",
        })

        # Separate input (X) values and output (Y) values
        x = self.data.iloc[:, 1:].values
        y = self.data.iloc[:, 0].values

        # Split into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        # Scale the data
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def predict_data(self, given_attributes):
        # Scale given attributes
        scaled_attributes = self.scaler.transform([given_attributes])

        # Build Model
        classifier = LogisticRegression()
        classifier.fit(self.x_train, self.y_train)

        # Predict
        prediction = classifier.predict(scaled_attributes)

        if prediction == 0:
            st.info("It is likely that this patient does not have pre-diabetes or diabetes")
        else:
            st.info("It is likely that this patient has either pre-diabetes or diabetes")

    def predict_test_data(self):
        classifier = LogisticRegression()
        classifier.fit(self.x_train, self.y_train)

        y_pred = classifier.predict(self.x_test)

        # DRAW CONFUSION MATRIX
        with st.expander("Confusion Matrix on Test Data"):
            # CONFUSION MATRIX -- BASE MATRIX
            plt.subplots(figsize=(11, 9))                           # Create underyling matplot
            confusion = confusion_matrix(self.y_test, y_pred)       # Create confusion matrix
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            plot = sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap)    # Plot with Seaborn's heatmap

            # CONFUSION MATRIX -- LABELS
            plot.set_title("Diabetes Diagnosis: Actual Diagnosis vs. Predicted Diagnosis")  # Title the chart

            plot.set_xlabel("Predicted Diagnosis")                  # Label the x-axis
            plot.xaxis.set_ticklabels(["No diabetes", "Diabetes"])  # Label each x-axis option

            plot.set_ylabel("Actual Diagnosis")                     # Label the y-axis
            plot.yaxis.set_ticklabels(["No diabetes", "Diabetes"])  # Label each y-axis option

            # CONFUSION MATRIX -- WRITE TO UI
            st.pyplot(plot.get_figure())                            # Print matrix to frontend
            st.write(accuracy_score(self.y_test, y_pred))           # Print the accuracy score value

    def draw_visualizations(self):
        # DRAW CORRELATION MATRIX
        with st.expander("Correlation Matrix"):
            # CORRELATION MATRIX -- BASE MATRIX
            plt.subplots(figsize=(11, 9))                                     # Create underlying matplot
            corr = self.data.corr()                                           # Create correlation matrix

            # CORRELATION MATRIX -- STYLING
            mask = np.triu(np.ones_like(corr, dtype=bool))              # Create mask for top right triangle
            cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Create custom color palette

            # CORRELATION MATRIX -- DRAW FINAL MATRIX
            correlation_mat = sns.heatmap(corr, mask=mask, cmap=cmap,          # Draw the seaborn heatmap
                                          center=0, linewidths=.5)

            # CORRELATION MATRIX -- WRITE TO UI
            st.pyplot(correlation_mat.get_figure())

        # DRAW DISTRIBUTION CHARTS
        with st.expander("Distributions"):
            plt.subplots(figsize=(11, 9))
            diabetes_freq = sns.histplot(self.data, x="Diabetes", bins=2)
            st.pyplot(diabetes_freq.get_figure())

    def print_data(self):
        st.table(self.data.head())
        st.write(self.data.columns())
