import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle  # Use built-in pickle instead of pickle5
import io
import random

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

# Reinforcement Learning Implementation
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        # Current Q-value
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            # Maximum Q-value for next state
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Update Q-value using Bellman equation
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
        
        # Decay exploration rate
        if not done:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate
            }, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data['exploration_rate']

# Irrigation Environment for RL Training
class IrrigationEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Define state parameters from data
        self.sensor_cols = [col for col in data.columns if 'sensor_' in col]
        self.soil_moisture = data['sensor_0'].values / 15.0  # Normalize to 0-1
        self.crop_types = data['crop_type'].values
        
        # Crop-specific optimal moisture levels
        self.crop_optimal_moisture = {
            'Wheat': 0.4,
            'Corn': 0.5,
            'Tomato': 0.6,
            'Cotton': 0.5
        }
    
    def reset(self):
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        # Create state from current data
        moisture_level = int(self.soil_moisture[self.current_step] * 10)  # 0-10
        crop = self.crop_types[self.current_step]
        crop_idx = ['Wheat', 'Corn', 'Tomato', 'Cotton'].index(crop)
        
        # Combine into state index
        state = moisture_level * 10 + crop_idx
        return min(state, 99)  # Ensure within bounds
    
    def step(self, action):
        # Action: 0=no irrigation, 1=light, 2=medium, 3=heavy
        current_moisture = self.soil_moisture[self.current_step]
        current_crop = self.crop_types[self.current_step]
        optimal_moisture = self.crop_optimal_moisture[current_crop]
        
        # Apply irrigation action
        irrigation_amount = action * 0.1  # Convert action to water amount
        new_moisture = min(1.0, current_moisture + irrigation_amount)
        
        # Calculate reward
        reward = self._calculate_reward(new_moisture, optimal_moisture, irrigation_amount)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self, moisture, optimal_moisture, water_used):
        # Reward for maintaining optimal moisture
        moisture_diff = abs(moisture - optimal_moisture)
        if moisture_diff < 0.05:
            moisture_reward = 10
        elif moisture_diff < 0.1:
            moisture_reward = 5
        elif moisture_diff < 0.2:
            moisture_reward = 0
        else:
            moisture_reward = -5
        
        # Penalty for water usage (encourage conservation)
        water_penalty = -2 * water_used
        
        # Bonus for using less water when moisture is adequate
        conservation_bonus = 3 if (water_used == 0 and moisture_diff < 0.1) else 0
        
        return moisture_reward + water_penalty + conservation_bonus

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

# Train RL model
def train_rl_model(data, episodes=1000):
    env = IrrigationEnvironment(data)
    state_size = 100  # moisture(0-10) * crops(4) = 40 states, using 100 for buffer
    action_size = 4   # 0: no irrigation, 1: light, 2: medium, 3: heavy
    
    agent = QLearningAgent(state_size, action_size)
    
    rewards_history = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Exploration: {agent.exploration_rate:.3f}")
    
    return agent, rewards_history

# Load trained model
def load_model():
    try:
        # Try to load pre-trained model
        agent = QLearningAgent(100, 4)
        agent.load_model('trained_irrigation_model.pkl')
        return agent
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
    
    # Train RL model
    st.subheader("Reinforcement Learning Training")
    if st.button("Train RL Model"):
        with st.spinner("Training reinforcement learning model... This may take a few minutes."):
            agent, rewards_history = train_rl_model(processed_data, episodes=500)
            st.session_state.rl_agent = agent
            st.session_state.training_rewards = rewards_history
            
            # Save the trained model
            agent.save_model('trained_irrigation_model.pkl')
            st.success("RL model trained successfully!")
            
            # Plot training progress
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(rewards_history)
            ax.set_title('RL Training Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.grid(True)
            st.pyplot(fig)
    
    # Show trained model info if available
    if 'rl_agent' in st.session_state:
        st.info("✅ RL model is trained and ready for recommendations!")
        st.write(f"Final exploration rate: {st.session_state.rl_agent.exploration_rate:.3f}")
    
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
    
    # Load or use trained RL model
    if 'rl_agent' not in st.session_state:
        st.session_state.rl_agent = load_model()
        if st.session_state.rl_agent is None:
            st.warning("No trained RL model found. Please train the model on the Data Analysis page first.")
            st.stop()
    
    # User input for recommendations
    st.subheader("Get Irrigation Recommendations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        weather = st.selectbox("Weather Condition", 
                              ["Sunny", "Cloudy", "Rainy", "Extreme Heat"])
    
    with col2:
        crop = st.selectbox("Crop Type", 
                           ["Wheat", "Corn", "Tomato", "Cotton"])
    
    with col3:
        growth_stage = st.selectbox("Growth Stage", 
                                  ["Germination", "Vegetative", "Flowering", "Maturation"])
    
    with col4:
        moisture = st.slider("Soil Moisture Level", 0.0, 1.0, 0.4, 0.1)
    
    # Get recommendation using RL model
    if st.button("Get Recommendation"):
        # Convert user inputs to state representation
        moisture_level = int(moisture * 10)
        crop_idx = ["Wheat", "Corn", "Tomato", "Cotton"].index(crop)
        growth_idx = ["Germination", "Vegetative", "Flowering", "Maturation"].index(growth_stage)
        weather_idx = ["Sunny", "Cloudy", "Rainy", "Extreme Heat"].index(weather)
        
        # Create state index (simplified for demo)
        state = moisture_level * 10 + crop_idx
        
        # Use RL agent to choose action
        action = st.session_state.rl_agent.choose_action(state)
        
        # Action mapping
        action_names = {
            0: "No irrigation",
            1: "Light irrigation (0.1 units)",
            2: "Medium irrigation (0.2 units)", 
            3: "Heavy irrigation (0.3 units)"
        }
        
        water_amounts = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}
        recommended_water = water_amounts[action]
        
        # Display RL-based recommendation
        st.success(f"**RL Recommendation:** {action_names[action]}")
        
        # Calculate optimal moisture for explanation
        growth_stages = ["Germination", "Vegetative", "Flowering", "Maturation"]
        growth_idx_calc = growth_stages.index(growth_stage)
        
        crop_optimal_moisture = {
            "Wheat": [0.3, 0.4, 0.5, 0.4],
            "Corn": [0.4, 0.5, 0.6, 0.5],
            "Tomato": [0.5, 0.6, 0.7, 0.5],
            "Cotton": [0.4, 0.5, 0.6, 0.4]
        }
        
        weather_impact = {
            "Sunny": 0.15,
            "Cloudy": 0.08,
            "Rainy": -0.05,
            "Extreme Heat": 0.25
        }
        
        optimal_moisture = crop_optimal_moisture[crop][growth_idx_calc]
        moisture_deficit = optimal_moisture - moisture
        weather_factor = weather_impact[weather]
        
        # Display explanation
        st.info(f"""
        **Decision Explanation (RL Model):**
        - Current state: Moisture={moisture:.2f}, Crop={crop}, Growth={growth_stage}, Weather={weather}
        - RL Agent selected action: {action} ({action_names[action]})
        - Q-value for chosen action: {st.session_state.rl_agent.q_table[state, action]:.2f}
        - Exploration rate: {st.session_state.rl_agent.exploration_rate:.3f}
        """)
        
        # Visual representation
        fig, ax = plt.subplots(figsize=(10, 2))
        bars = ax.barh(['Current Moisture', 'Optimal Moisture', 'Recommended Water'], 
                      [moisture, optimal_moisture, recommended_water], 
                      color=['lightblue', 'lightgreen', 'orange'])
        ax.set_xlim(0, 1)
        ax.set_title('RL-Based Irrigation Recommendation')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center')
        
        st.pyplot(fig)
        
        # Additional insights
        st.subheader("RL Model Insights")
        
        # Show Q-values for all actions in current state
        st.write("**Q-values for all possible actions in current state:**")
        q_values = st.session_state.rl_agent.q_table[state]
        action_df = pd.DataFrame({
            'Action': ['No irrigation', 'Light', 'Medium', 'Heavy'],
            'Q-value': q_values
        })
        st.dataframe(action_df)
        
        if moisture < optimal_moisture - 0.1:
            st.warning("⚠️ **Warning:** Soil is too dry! RL model recommends irrigation.")
        elif moisture > optimal_moisture + 0.1:
            st.warning("⚠️ **Warning:** Soil is too wet! RL model recommends no irrigation.")
        else:
            st.success("✅ Soil moisture is within optimal range.")
        
        if weather == "Extreme Heat":
            st.info("🌡️ **Note:** Extreme heat conditions increase water evaporation. RL model accounts for this.")
        elif weather == "Rainy":
            st.info("🌧️ **Note:** Rainy conditions may reduce irrigation needs. RL model considers weather impact.")

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
    """)
    
    st.subheader("Reinforcement Learning Approach")
    st.markdown("""
    The system uses Q-learning with:
    - **State space**: Moisture levels + crop types + weather conditions + growth stages
    - **Action space**: 4 irrigation levels (0.0, 0.1, 0.2, 0.3 units)
    - **Reward function**: Balances crop health and water conservation
    - **Exploration**: ε-greedy strategy with decay
    - **Q-table**: 100 states × 4 actions learning matrix
    
    ### RL Training Process:
    1. **State Representation**: Convert sensor data and conditions to discrete states
    2. **Action Selection**: Choose irrigation action using ε-greedy policy
    3. **Reward Calculation**: Evaluate action based on moisture levels and water usage
    4. **Q-value Update**: Update policy using Bellman equation
    5. **Exploration Decay**: Gradually reduce random actions as learning progresses
    """)
