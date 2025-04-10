import streamlit as st
import pickle
import pandas as pd

# Load the trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and cities list
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# App title
st.title('🏏 IPL Win Predictor')

# Input fields
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target Score')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs Completed')
with col5:
    wickets_out = st.number_input('Wickets Fallen')

# Predict button
if st.button('Predict Probability'):
    # Derived features
    run_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets_out
    crr = score / overs if overs > 0 else 0
    rrr = (run_left * 6) / balls_left if balls_left > 0 else 0

    # Prepare input DataFrame for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'run_left': [run_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Get prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Show prediction
    st.header(f"🏆 {batting_team} - {round(win * 100)}% Chance to Win")
    st.header(f"🎯 {bowling_team} - {round(loss * 100)}% Chance to Win")
