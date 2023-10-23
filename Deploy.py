import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

st.title('FIFA player prediction app')

# Create input boxes for each feature
movement_reactions = st.number_input('Movement Reactions', min_value=0)
mentality_composure = st.number_input('Mentality Composure', min_value=0)
potential = st.number_input('Potential', min_value=0)
power_shot_power = st.number_input('Power Shot Power', min_value=0)
mentality_vision = st.number_input('Mentality Vision', min_value=0)
attacking_short_passing = st.number_input('Attacking Short Passing', min_value=0)
skill_long_passing = st.number_input('Skill Long Passing', min_value=0)
age = st.number_input('Age', min_value=0)
skill_ball_control = st.number_input('Skill Ball Control', min_value=0)

# Create a submit button
submit_button = st.button('Predict')

if submit_button:
    # Create a dictionary with the user input
    user_input = {
        'movement_reactions': movement_reactions,
        'mentality_composure': mentality_composure,
        'potential': potential,
        'power_shot_power': power_shot_power,
        'mentality_vision': mentality_vision,
        'attacking_short_passing': attacking_short_passing,
        'skill_long_passing': skill_long_passing,
        'age': age,
        'skill_ball_control': skill_ball_control
    }

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([user_input])

    # Ensure column order matches the order of training data
    expected_columns = ['movement_reactions', 'mentality_composure', 'potential', 'power_shot_power', 
                        'mentality_vision', 'attacking_short_passing', 'skill_long_passing', 'age', 'skill_ball_control']
    input_df = input_df[expected_columns]

    # Convert DataFrame to DMatrix
    dmatrix_input = xgb.DMatrix(input_df)

    # Load the pre-trained model
    rf_model = joblib.load('ML_model.pkl')

    # Print the model type (for debugging)
    st.write(f"Loaded model type: {rf_model.__class__.__name__}")

    try:
        prediction = rf_model.predict(dmatrix_input)
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        prediction = [0]  # Placeholder

    # Display the prediction
    st.write(f'Predicted Overall Rating: {prediction[0]}')
