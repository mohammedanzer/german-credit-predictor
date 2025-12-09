import numpy as np
import pandas as pd
import streamlit as st
import pickle

@st.cache_resource
def load_model():
    with open('logreg_pipeline.pkl','rb') as f:
        pipeline_dict = pickle.load(f)
    return pipeline_dict['Pipeline'], pipeline_dict['Label_Encoder_y']

pipeline, le_y= load_model()

st.title('German Credit Risk Prediction')
st.markdown("Enter customer details to predict credit risk")

job_options = {
    "unskilled and non-resident": 0,
    "unskilled and resident": 1,
    "skilled": 2,
    "highly skilled": 3
    }

#Create input columns
col1, col2 = st.columns(2)
with col1:
    age_input=st.number_input("Borrower Age", min_value=18, max_value=100, value=30)
    sex_input=st.selectbox("Gender", options=['Male','Female'])

    job_display = st.selectbox("Job Level", options=list(job_options.keys()), index=2)
    job_input = job_options[job_display]

    housing_input = st.selectbox("Housing Status", options=['free','own', 'rent'])
    saving_input = st.selectbox("Saving accounts", options=['Unkown','little', 'moderate', 'quite rich', 'rich'])

with col2:
    with col2:
        checking_input = st.selectbox("Checking account", options=['little', 'moderate', 'rich', 'no data'])
        creditamt_input = st.number_input("Loan Amount", min_value=250, max_value=18000, value=300)
        duration_input = st.slider("Loan Duration (months)", min_value=1, max_value=72, value=12, step=1)

    purpose_input = st.selectbox("Purpose", 
                                 options=['car (new)', 'car (used)', 'furniture/equipment',
                                          'radio/TV', 'domestic appliances', 'repairs', 'education',
                                          'vacation/others', 'retraining', 'business'])


input_data = {
    'Age': [age_input],
    'Sex': [sex_input],
    'Job': [job_input],
    'Housing': [housing_input],
    'Saving accounts': [saving_input],
    'Checking account': [checking_input],
    'Credit amount': [creditamt_input],
    'Duration': [duration_input],
    'Purpose': [purpose_input]
    }

input_df = pd.DataFrame(input_data)

if st.button("Predict Risk", type="primary"):
    try:
        # Predict using the pipeline
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        # Decode prediction
        risk_label = le_y.inverse_transform([prediction])[0]

        st.success(f"**Predicted Risk: {risk_label.upper()}**")

        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            good_idx = le_y.transform(['good'])[0]
            st.metric("Good Credit", f"{probability[good_idx]:.1%}")
        with col_prob2:
            bad_idx = le_y.transform(['bad'])[0]
            st.metric("Bad Credit", f"{probability[bad_idx]:.1%}")



    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Check if input data matches the training data format")




