import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import io

# Set page config
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="wide"
)

# Title and description
st.title("🌱 Smart Irrigation System with Reinforcement Learning")
st.markdown("""
This system uses reinforcement learning to optimize irrigation schedules based on sensor data, 
weather conditions, and crop types. Upload your irrigation machine data or use our demo data.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Irrigation Recommendations", "About"])

# Load or upload data
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data, True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, False
    else:
        # Load sample data
        try:
            data = pd.read_csv('irrigation_machine.csv')
            return data, True
        except:
            # Create sample data if file not found
            st.info("Using demo data as sample dataset was not found.")
            data = pd.DataFrame()
            data['Unnamed: 0'] = range(2000)
            for i in range(20):
                data[f'sensor_{i}'] = np.random.randint(0, 15, 2000)
            for i in range(3):
                data[f'parcel_{i}'] = np.random.randint(0, 2, 2000)
            return data, False

# Preprocess data
def preprocess_data(data):
    sensor_cols = [col for col in data.columns if 'sensor_' in col]
    scaler = StandardScaler()
    sensor_data = data[sensor_cols].values
    sensor_data_scaled = scaler.fit_transform(sensor_data)
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    states = kmeans.fit_predict(sensor_data_scaled)
    
    data['state'] = states
    
    # Map parcels to crop types
    parcel_cols = [col for col in data.columns if 'parcel_' in col]
    parcel_data = data[parcel_cols].values
    
    crop_types = []
    for i in range(len(data)):
        parcel_vec = parcel_data[i]
        if parcel_vec[0] == 1 and parcel_vec[1] == 0 and parcel_vec[2] == 0:
            crop_types.append('Wheat')
        elif parcel_vec[0] == 0 and parcel_vec[1] == 1 and parcel_vec[2] == 0:
            crop_types.append('Corn')
        elif parcel_vec[0] == 0 and parcel_vec[1] == 0 and parcel_vec[2] == 1:
            crop_types.append('Tomato')
        else:
            crop_types.append('Cotton')
    
    data['crop_type'] = crop_types
    
    return data, scaler, kmeans

# Load trained model (you would need to train and save this first)
def load_model():
    try:
        with open('model/trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

# Home page
if page == "Home":
    st.header("Welcome to the Smart Irrigation System")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your irrigation data CSV", type=["csv"])
    
    if uploaded_file is not None:
        data, success = load_data(uploaded_file)
        if success:
            st.success("Data loaded successfully!")
            st.session_state.data = data
    else:
        data, is_real_data = load_data()
        st.session_state.data = data
        if not is_real_data:
            st.warning("Using demo data. Upload your own CSV for better results.")
    
    if 'data' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Shape:**", st.session_state.data.shape)
            st.write("**Columns:**", list(st.session_state.data.columns))
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame(st.session_state.data.isnull().sum(), columns=['Missing Values'])
            st.dataframe(missing_df)

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if 'data' not in st.session_state:
        st.warning("Please load data from the Home page first.")
        st.stop()
    
    data = st.session_state.data
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        processed_data, scaler, kmeans = preprocess_data(data)
        st.session_state.processed_data = processed_data
        st.session_state.scaler = scaler
        st.session_state.kmeans = kmeans
    
    st.subheader("Processed Data with Crop Types")
    st.dataframe(st.session_state.processed_data[['sensor_0', 'sensor_1', 'parcel_0', 'parcel_1', 'parcel_2', 'crop_type', 'state']].head())
    
    st.subheader("Crop Distribution")
    crop_counts = st.session_state.processed_data['crop_type'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(crop_counts.index, crop_counts.values)
    ax.set_title('Crop Type Distribution')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Sensor Statistics")
    sensor_cols = [col for col in data.columns if 'sensor_' in col]
    sensor_stats = data[sensor_cols].describe()
    st.dataframe(sensor_stats)

# Irrigation Recommendations page
elif page == "Irrigation Recommendations":
    st.header("Irrigation Recommendations")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please process data from the Data Analysis page first.")
        st.stop()
    
    # User input for recommendations
    st.subheader("Get Irrigation Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weather = st.selectbox("Weather Condition", 
                              ["Sunny", "Cloudy", "Rainy", "Extreme Heat"])
    
    with col2:
        crop = st.selectbox("Crop Type", 
                           ["Wheat", "Corn", "Tomato", "Cotton"])
    
    with col3:
        moisture = st.slider("Soil Moisture Level", 0.0, 1.0, 0.4, 0.1)
    
    # Simple recommendation logic (replace with your trained model)
    if st.button("Get Recommendation"):
        # This is a simplified version - you would use your trained RL model here
        weather_impact = {
            "Sunny": 0.2,
            "Cloudy": 0.1,
            "Rainy": -0.1,
            "Extreme Heat": 0.3
        }
        
        crop_needs = {
            "Wheat": 0.4,
            "Corn": 0.5,
            "Tomato": 0.6,
            "Cotton": 0.45
        }
        
        # Simple logic for demonstration
        optimal_moisture = crop_needs[crop]
        moisture_diff = optimal_moisture - moisture
        weather_factor = weather_impact[weather]
        
        recommended_water = max(0, min(0.3, moisture_diff + weather_factor))
        
        st.success(f"Recommended irrigation: **{recommended_water:.2f} units** of water")
        
        # Display explanation
        st.info(f"""
        **Explanation:**
        - Optimal moisture for {crop}: {optimal_moisture}
        - Current moisture: {moisture}
        - Weather impact ({weather}): {weather_factor}
        - Moisture deficit: {moisture_diff:.2f}
        """)
        
        # Visual representation
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(['Current', 'Optimal', 'Recommended'], 
                [moisture, optimal_moisture, recommended_water], 
                color=['lightblue', 'lightgreen', 'orange'])
        ax.set_xlim(0, 1)
        ax.set_title('Moisture Levels and Recommendation')
        st.pyplot(fig)

# About page
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## Smart Irrigation System with Reinforcement Learning
    
    This project uses reinforcement learning to optimize irrigation schedules based on:
    - Sensor data from irrigation machines
    - Weather conditions
    - Crop types and their water needs
    - Soil moisture levels
    
    ### How it works:
    1. **Data Processing**: Sensor data is clustered into meaningful states
    2. **Reinforcement Learning**: A Q-learning agent learns optimal irrigation policies
    3. **Recommendations**: The system provides irrigation recommendations based on current conditions
    
    ### Technologies used:
    - Python
    - Streamlit for the web interface
    - Scikit-learn for data processing
    - Reinforcement Learning (Q-learning)
    - Pandas for data manipulation
    
    ### Dataset:
    The system uses the Irrigation Machine Dataset from Kaggle, which contains:
    - 20 sensor readings
    - 3 parcel indicators (for crop type identification)
    - 2000 data points
    
    ### Future enhancements:
    - Integration with real-time weather APIs
    - IoT sensor connectivity
    - Advanced deep reinforcement learning
    - Mobile app integration
    """)
    
    st.subheader("Project Repository")
    st.markdown("[GitHub Repository](https://github.com/your-username/smart-irrigation-system)")
