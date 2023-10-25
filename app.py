import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# ---
# LOADING IN DATA
# ---

data = pd.read_csv("data/diabetes_full_health_indicators.csv")


# ---
# PRE-PROCESSING
# ---

# Convert float types to int
data = data.astype({
    "Diabetes_012": "int32",
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

# st.write(data.dtypes)

# Check for null values
# st.write("Check for null values")
# st.write(data.isnull().sum())

# Scale
ss = StandardScaler()
data['BMI'] = ss.fit_transform(data[['BMI']])
data['GenHlth'] = ss.fit_transform(data[['GenHlth']])
data['MentHlth'] = ss.fit_transform(data[['MentHlth']])
data['PhysHlth'] = ss.fit_transform(data[['MentHlth']])
data['Age'] = ss.fit_transform(data[['Age']])
data['Education'] = ss.fit_transform(data[['Education']])
data['Income'] = ss.fit_transform(data[['Income']])


# Separate input (X) values and output (Y) values
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# ---
# USER INTERFACE
# ---

# Checkboxes
# st.write("Current Health Status")
# high_bp = st.checkbox("Has high blood pressure")
# high_chol = st.checkbox("Has high cholesterol")
# chol_check = st.checkbox("Had their cholesterol checked in the last 5 years")
# stroke = st.checkbox("Has had a stroke")
# heart_disease = st.checkbox("Has had coronary heart disease or myocardial infarction")
# bmi = st.number_input("BMI", 0, 100)
#
#
# st.write("Habits and Activities")
# phys_activity = st.checkbox("Has been physically active in the last 30 days -- not job-related")
# fruits = st.checkbox("Consumes fruits at least once a day")
# veggies = st.checkbox("Consumes vegetables at least once a day")
# smoker = st.checkbox("Smoked more than 100 cigarettes (5 packs) in lifetime")
# alcohol = st.checkbox("Is a heavy drinker (14 drinks/week for men, 7 drinks/week for women")
# healthcare = st.checkbox("Has any kind of health coverage or insurance")
# doc_cost = st.checkbox("Any time in the last 12 months where a doctor was needed but avoided it for financial reasons")
#
# # Number Inputs
#
# st.write("General and Mental Health")
# gen_health = st.slider("Personal ranking of general health, where 1 = excellent, 5 = poor", 1, 5)
# mental_health = st.slider("On how many days out of the last 30 days was their mental health was poor?", 0, 30)
# phys_health = st.slider("On how many days out of the last 30 days was their physical health poor?", 1, 30)
# diff_walk = st.checkbox("Has serious difficulty walking/climbing stairs")
#
# # Demographics
# st.write("Demographics")
# sex = st.selectbox("Gender", ["Female", "Male", "Other"])  # Make 0 female, 1 male
# age = st.number_input("13-level age category (see below)", 1, 13)
# edu = st.number_input("EDUCA education level (see below)", 1, 6)
# income = st.number_input("Income scale (see below)", 1, 8)


# ---
# GRAPHS
# ---
corr = data.corr()
st.write(corr)


fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax)
st.write(fig)



# ---
# LOGISTIC REGRESSION
# ---

classifier = LogisticRegression()
# classifier = KNeighborsClassifier()
# classifier = DecisionTreeClassifier()
# classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)


# classifier.predict([[
#     high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity, fruits, veggies, alcohol,
#     healthcare, doc_cost, gen_health, mental_health, phys_health, diff_walk, sex, age, edu, income
# ]])

y_pred = classifier.predict(x_test)

# CONFUSION MATRIX

confusion = confusion_matrix(y_test, y_pred)
st.write(confusion)
st.write(accuracy_score(y_test, y_pred))
