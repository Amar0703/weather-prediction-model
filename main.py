import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Title and Description
st.title("🌧️ Rain Prediction App")
st.write("Enter the weather details below to predict if it will rain tomorrow in Australia.")

# 2. Load and Train Model (Cached so it doesn't reload every time)
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('weatherAUS.csv')
    cols_to_use = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'RainToday', 'RainTomorrow']
    df = df[cols_to_use].dropna()
    
    # Preprocessing
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    X = df[['MinTemp', 'MaxTemp', 'Humidity3pm', 'RainToday']]
    y = df['RainTomorrow']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Load data
try:
    model, accuracy = load_and_train_model()
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    st.error("Error loading data. Make sure weatherAUS.csv is in the repository.")

# 3. User Input (The Sidebar)
st.sidebar.header("User Input")
min_temp = st.sidebar.slider("Min Temperature (°C)", -10, 50, 20)
max_temp = st.sidebar.slider("Max Temperature (°C)", -10, 50, 30)
humidity = st.sidebar.slider("Humidity at 3pm (%)", 0, 100, 50)
rain_today = st.sidebar.selectbox("Did it rain today?", ["Yes", "No"])

# Convert input to numbers
rain_today_numeric = 1 if rain_today == "Yes" else 0

# 4. Prediction Button
if st.button("Predict Rain"):
    user_data = [[min_temp, max_temp, humidity, rain_today_numeric]]
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("It will likely RAIN tomorrow! ☔")
    else:
        st.success("It will likely be SUNNY tomorrow! ☀️")