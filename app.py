import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys

# Streamlit app
st.title("Math Score Prediction App")

# Form input fields
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parent_education = st.selectbox(
        "Parental Level of Education",
        ["associate's degree", "bachelor's degree", "some college", "high school", "master's degree"]
    )
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_prep_score = st.selectbox("Test Preparation Course", ["completed", "uncompleted"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

    submit = st.form_submit_button("Predict Math Score")

if submit:
    try:
        custom_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parent_education=parent_education,
            lunch=lunch,
            test_prep_score=test_prep_score,
            reading_score=reading_score,
            writing_score=writing_score
        )

        data_df = custom_data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_df)

        st.success(f"The predicted Math score is: {prediction[0]}")

        # Chart to show reading, writing, and predicted math scores
        scores = {
            'Scores': ['Reading', 'Writing', 'Predicted Math'],
            'Values': [reading_score, writing_score, prediction[0]]
        }
        scores_df = pd.DataFrame(scores)

        fig, ax = plt.subplots()
        ax.bar(scores_df['Scores'], scores_df['Values'], color=['#4caf50', '#2196f3', '#ff5722'])
        plt.ylabel('Score')
        plt.title('Score Comparison')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")