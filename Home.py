import streamlit as st
import pandas as pd
from RegressionModel import RegressionModel


# ---
# CONFIGURE STREAMLIT
# ---

st.set_page_config(layout="wide")


# ---
# SURVEY FOR PREDICTOR
# ---

survey_col1, survey_col2, survey_col3, survey_col4 = st.columns(4)

# Health Status
with survey_col1:
    st.subheader("Current Health Status")
    high_bp = st.checkbox("Has high blood pressure")
    high_chol = st.checkbox("Has high cholesterol")
    chol_check = st.checkbox("Had their cholesterol checked in the last 5 years")
    stroke = st.checkbox("Has had a stroke")
    heart_disease = st.checkbox("Has had coronary heart disease or myocardial infarction")
    bmi = st.number_input("BMI", 0, 100, 30)

# Habits and Activities
with survey_col2:
    st.subheader("Habits and Activities")
    phys_activity = st.checkbox("Has been physically active in the last 30 days — not job-related")
    fruits = st.checkbox("Consumes fruits at least once a day")
    veggies = st.checkbox("Consumes vegetables at least once a day")
    smoker = st.checkbox("Smoked more than 100 cigarettes in lifetime")
    alcohol = st.checkbox("Is a heavy drinker (14 drinks/week for men, 7 drinks/week for women")
    healthcare = st.checkbox("Has any kind of health coverage or insurance")
    doc_cost = st.checkbox("There was a time in the last 12 months where a doctor was "
                           "needed but avoided for financial reasons")

# General and Mental Health
with survey_col3:
    st.subheader("General and Mental Health")
    gen_health = st.slider("Personal ranking of general health, where 1 = excellent, 5 = poor", 1, 5)
    mental_health = st.slider("How many days out of the last 30 days where their mental health was poor", 0, 30)
    phys_health = st.slider("How many days out of the last 30 days where their physical health was poor", 0, 30)
    diff_walk = st.checkbox("Has serious difficulty walking/climbing stairs")

# Demographics
with survey_col4:
    st.subheader("Demographics")
    sex = st.checkbox("Male")
    age = st.number_input("13-level age category (see below)", 1, 13)
    edu = st.number_input("EDUCA education level (see below)", 1, 6)
    income = st.number_input("Income scale (see below)", 1, 8)

# Explain Demographic Scales
with st.expander("Explain Demographic Scales"):
    explain_col1, explain_col2, explain_col3 = st.columns(3)
    with explain_col1:
        st.subheader("13-Level Age Category")

        ages = {
            "Level": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "Age Range": ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44",
                          "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69",
                          "Age 70 to 74", "Age 75 to 79", "Age 80 or older"]
        }

        ages_df = pd.DataFrame(ages)
        st.dataframe(ages_df.set_index(ages_df.columns[0]))

    with explain_col2:
        st.subheader("EDUCA Education Level")

        education = {
            "Level": [1, 2, 3, 4, 5, 6],
            "Education": ["Never attended school (or only kindergarten", "Grades 1-8", "Grades 9-11",
                          "Grade 12 or GED",
                          "College 1 year to 3 years (college or technical school)",
                          "College 4 years or more (graduate)"]
        }

        education_df = pd.DataFrame(education)
        st.dataframe(education_df.set_index(education_df.columns[0]))

    with explain_col3:
        st.subheader("Income Level")

        income_chart = {
            "Level": [1, 2, 3, 4, 5, 6, 7, 8],
            "Income": ["$0 to less than $10k", "$10k to less than $15k", "$15k to less than $20k",
                       "$20k to less than $25k", "$25k to less than $35k", "$35k to less than $50k",
                       "$35k to less than $75k", "$75k or more"]
        }

        income_df = pd.DataFrame(income_chart)
        st.dataframe(income_df.set_index(income_df.columns[0]))


# ---
# CALL REGRESSION MODEL
# ---

attributes = [high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity, fruits,
              veggies, alcohol, healthcare, doc_cost, gen_health, mental_health, phys_health, diff_walk, sex,
              age, edu, income]

model = RegressionModel()
st.button(label="Submit", on_click=model.predict_data, args=[attributes], type="primary", use_container_width=True)

# ---
# INFO POP-UP
# ---

st.info("Open the sidebar in the top left corner to see visualizations of the dataset")
