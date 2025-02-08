import streamlit as st
import pickle
import numpy as np

# Load the saved models and scalers
with open("crop.pkl", "rb") as crop_model_file:
    crop_model = pickle.load(crop_model_file)

with open("crop_scaler.pkl", "rb") as crop_scaler_file:
    crop_scaler = pickle.load(crop_scaler_file)

with open("fertilizer.pkl", "rb") as fertilizer_model_file:
    fertilizer_model = pickle.load(fertilizer_model_file)

with open("fertilizer_scaler.pkl", "rb") as fertilizer_scaler_file:
    fertilizer_scaler = pickle.load(fertilizer_scaler_file)

# Crop Recommendation Function
def crop_recommend(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = crop_scaler.transform(features)
    prediction = crop_model.predict(transformed_features)
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }
    return crop_dict.get(prediction[0], "Unknown Crop")

# Fertilizer Recommendation Function
def recommend_fertilizer(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    features = np.array([[Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    transformed_features = fertilizer_scaler.transform(features)
    prediction = fertilizer_model.predict(transformed_features)
    fert_dict = {
        1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28',
        5: '17-17-17', 6: '20-20', 7: '10-26-26'
    }
    return fert_dict.get(prediction[0], "Unknown Fertilizer")

# Streamlit Interface
def main():
    st.title("🌱 Crop and Fertilizer Recommendation System")
    
    # Sidebar for navigation
    menu = ["🌾 Crop Recommendation", "🌿 Fertilizer Recommendation"]
    choice = st.sidebar.selectbox("🔍 Select Recommendation Type", menu)
    
    if choice == "🌾 Crop Recommendation":
        st.subheader("🌾 Crop Recommendation")
        col1, col2 = st.columns(2)  # Create two columns
        
        with col1:
            N = st.number_input("💧 Nitrogen Level (N)", min_value=0, max_value=200, value=90)
            P = st.number_input("🪴 Phosphorus Level (P)", min_value=0, max_value=200, value=42)
            K = st.number_input("🌿 Potassium Level (K)", min_value=0, max_value=200, value=43)
            ph = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=6.1)

        with col2:
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
            humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, value=82)
            rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0, max_value=300, value=202)
        
        if st.button("🌱 Recommend Crop"):
            recommended_crop = crop_recommend(N, P, K, temperature, humidity, ph, rainfall)
            st.success(f"✅ {recommended_crop} is the best crop to be cultivated.")
    
    elif choice == "🌿 Fertilizer Recommendation":
        st.subheader("🌿 Fertilizer Recommendation")
        col1, col2 = st.columns(2)  # Create two columns
        
        # Soil and Crop Type Mappings
        soil_type_dict = {"Sandy": 1, "Loamy": 2, "Clayey": 3}
        crop_type_dict = {"Rice": 1, "Maize": 2, "Wheat": 3}

        with col1:
            Temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0)
            Humidity = st.number_input("💧 Humidity (fraction, e.g., 0.5)", min_value=0.0, max_value=1.0, value=0.5)
            Moisture = st.number_input("🌫️ Moisture (fraction, e.g., 0.6)", min_value=0.0, max_value=1.0, value=0.6)
            Phosphorous = st.number_input("🪴 Phosphorus Level (P)", min_value=0, max_value=200, value=6)

        with col2:
            Soil_Type = st.selectbox("🪨 Soil Type", list(soil_type_dict.keys()))
            Crop_Type = st.selectbox("🌾 Crop Type", list(crop_type_dict.keys()))
            Nitrogen = st.number_input("💧 Nitrogen Level (N)", min_value=0, max_value=200, value=10)
            Potassium = st.number_input("🌿 Potassium Level (K)", min_value=0, max_value=200, value=15)

        # Convert names to encoded values
        Soil_Type_encoded = soil_type_dict[Soil_Type]
        Crop_Type_encoded = crop_type_dict[Crop_Type]
        
        if st.button("🌿 Recommend Fertilizer"):
            recommended_fertilizer = recommend_fertilizer(
                Temperature, Humidity, Moisture, Soil_Type_encoded, Crop_Type_encoded, Nitrogen, Potassium, Phosphorous
            )
            st.success(f"✅ {recommended_fertilizer} is the best fertilizer for the given conditions.")

if __name__ == "__main__":
    main()
