import streamlit as st
import numpy as np
import pickle

# Load the trained model
model_filename = 'RandomForestClassifier.pkl'  # Ensure this matches your saved model filename
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Title of the web app
st.title("Seasonal Crop Recommendation System")
st.write("Welcome to our Crop Recommendation System! This app helps farmers choose the best crops based on their soil conditions and environmental factors.")

# Create a sidebar for input fields
with st.sidebar:
    st.header("Input Your Data")
    
    # Input fields for user data
    soil_type = st.selectbox("Select Soil Type",[
    "Alluvial Soil",
    "Black Soil",
    "Red Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Desert Soil",
    "Forest Soil",
    "Peaty and Marshy Soil"] ) #Verifies it's a tuple"])
    
    season = st.selectbox("Select Season", ["Kharif", "Rabi", "Zaid"])
    land_size = st.number_input("Size of Land (in acres)", min_value=0.0)
    
    # Input fields for N, P, K, temperature, humidity, pH, and rainfall
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=10)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("Soil pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Main content area
st.header("SEASONAL CROP RECOMMENDATION SYSTEM")
st.write("Please fill in the details in the sidebar to get a personalized crop recommendation.")

# Button to make prediction
if st.button("Recommend Crop"):
    # Prepare the input data for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"The recommended crop is: {prediction[0]}")

# Additional information section
#st.header("How It Works")
#st.write("Our system uses a machine learning model trained on a dataset of crop recommendations based on various environmental factors. The model predicts the best crop for your conditions by analyzing the inputs you provide.")

# Contact or feedback section
#st.header("Get in Touch")
#st.write("If you have any questions or need further assistance, please feel free to contact us at [cropprediction].")
