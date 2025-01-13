import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="SOM Clustering App", layout="wide", page_icon="🤖")

# Apply custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton > button {
            background-color: #6A5ACD;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #483D8B;
        }
        .css-1aumxhk {
            padding-top: 2rem;
        }
            /* Responsive header styling */
        @media (max-width: 768px) {
            .responsive-title {
                font-size: 24px !important;
            }
        }
        @media (min-width: 769px) {
            .responsive-title {
                font-size: 36px !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown("<h1 class='responsive-title' style='text-align: center; color: #6A5ACD;'>🌟 Self-Organizing Map (SOM) Clustering 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6A5ACD;'>Danar Qusyairi</p>", unsafe_allow_html=True)

# Explanation of SOM in Bahasa Indonesia
with st.expander("ℹ️ Apa itu Self-Organizing Map (SOM)?"):
    st.markdown("""
    **Self-Organizing Map (SOM)** adalah salah satu metode dalam jaringan syaraf tiruan yang digunakan untuk visualisasi dan pengelompokan data secara tidak terawasi (*unsupervised learning*).
    
    🧩 **Cara Kerja SOM:**
    - SOM mengorganisasi data ke dalam grid 2D, sehingga data dengan karakteristik yang mirip akan dikelompokkan lebih dekat.
    - Membantu dalam menemukan pola dan struktur tersembunyi dalam data kompleks.
    
    🔍 **Manfaat SOM:**
    - Visualisasi data yang kompleks.
    - Segmentasi pelanggan.
    - Reduksi dimensi data.
    
    **Langkah-langkah Proses:**
    1. Data dinormalisasi untuk menyamakan skala.
    2. SOM dilatih untuk mengenali pola dari data.
    3. Data dipetakan ke grid SOM, menghasilkan klaster.
    
    Selamat mencoba dan eksplorasi data Anda! 🚀
    """)

# Upload dataset
uploaded_file = "UAS CLUSTERING.csv"

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Data preprocessing
    st.subheader("⚙️ Data Preprocessing")
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Encode categorical data
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    
    st.write("### Processed Data")
    st.dataframe(pd.DataFrame(data_scaled, columns=df.columns).head(), use_container_width=True)
    
    # SOM parameters with sidebar controls
    st.sidebar.header("🔧 SOM Parameters")
    x_dim = st.sidebar.slider("X dimension of SOM grid", min_value=2, max_value=20, value=5)
    y_dim = st.sidebar.slider("Y dimension of SOM grid", min_value=2, max_value=20, value=5)
    iterations = st.sidebar.number_input("Number of iterations", min_value=100, max_value=10000, value=1000)
    sigma = st.sidebar.number_input("Sigma (spread of neighborhood function)", min_value=0.1, max_value=5.0, value=1.0)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.5)
    
    if st.button("🚀 Run SOM"):
        with st.spinner("Training SOM, please wait..."):
            # Initialize and train SOM
            som = MiniSom(x=x_dim, y=y_dim, input_len=data_scaled.shape[1], sigma=sigma, learning_rate=learning_rate)
            som.random_weights_init(data_scaled)
            som.train_random(data_scaled, iterations)
            
            # Plotting the SOM distance map with Plotly for interactivity
            u_matrix = som.distance_map().T
            fig = px.imshow(u_matrix, color_continuous_scale='Viridis', title="Interactive SOM Distance Map (U-Matrix)")
            fig.update_layout(width=600, height=600)
            st.plotly_chart(fig)
            
            # Map each data point to its BMU (Best Matching Unit)
            cluster_labels = [som.winner(x) for x in data_scaled]
            
            # Add cluster labels to the original data
            df['Cluster'] = cluster_labels
            st.subheader("📊 Data with Cluster Labels")
            st.dataframe(df.head(), use_container_width=True)
            
            # Download clustered data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Clustered Data", csv, "clustered_data.csv", "text/csv")
else:
    st.info("📂 Please upload a CSV file to proceed.")
