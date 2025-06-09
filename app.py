import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.stats import shapiro, skew, kurtosis
import io

st.set_page_config(
    page_title="Car Listings Analysis Dashboard", page_icon="🚗", layout="wide"
)

st.title("🚗 Car Listings Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of car listings data, including data quality assessment,
univariate and bivariate analysis, correlation analysis, and clustering insights.
""")

@st.cache_data
def load_data():
    raw_data = pd.read_csv("./data/raw_data.csv")
    raw_data_cast = pd.read_csv("./data/raw_data_cast.csv")
    cleaned_data = pd.read_csv("./data/cleaned_data.csv")
    processed_data = pd.read_csv("./data/processed_data.csv")
    return raw_data, raw_data_cast, cleaned_data, processed_data

raw_data, raw_data_cast, cleaned_data, processed_data = load_data()

st.sidebar.header("Filters")

makes = ["All"] + sorted(raw_data_cast["Make"].unique().tolist())
selected_make = st.sidebar.selectbox("Select Make", makes)

min_price = float(raw_data_cast["price"].min())
max_price = float(raw_data_cast["price"].max())
price_range = st.sidebar.slider(
    "Price Range (€)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
)

filtered_df = raw_data_cast.copy()
if selected_make != "All":
    filtered_df = filtered_df[filtered_df["Make"] == selected_make]
filtered_df = filtered_df[
    (filtered_df["price"] >= price_range[0]) & (filtered_df["price"] <= price_range[1])
]

tab1, tab2, tab3, tab4, tab5= st.tabs([
    "Data Quality",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Categorical Analysis",
    "Correlation Analysis",
])

with tab1:
    st.header("Data Quality Analysis")
    
    st.subheader("Missing Values Analysis")
    
    missing_values = (raw_data.isnull().sum() / len(raw_data)) * 100
    missing_values = missing_values.sort_values(ascending=True)
    
    fig_missing = px.bar(
        x=missing_values.values,
        y=missing_values.index,
        title="Proportion of Missing Values by Column",
        labels={"x": "Percentage of Missing Values", "y": "Columns"}
    )
    st.plotly_chart(fig_missing, use_container_width=True)
    
    st.subheader("Missing Value Matrix")
    fig, ax = plt.subplots(figsize=(15, 6))
    msno.matrix(raw_data, ax=ax, fontsize=6)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Missing Value Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    msno.heatmap(raw_data, ax=ax, fontsize=6)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Univariate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = px.histogram(
            filtered_df,
            x="price",
            nbins=50,
            title="Price Distribution",
            labels={"price": "Price (€)"},
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        fig_mileage = px.histogram(
            filtered_df,
            x="Mileage",
            nbins=50,
            title="Mileage Distribution",
            labels={"Mileage": "Mileage (km)"},
        )
        st.plotly_chart(fig_mileage, use_container_width=True)
    
    with col2:
        fig_power = px.histogram(
            filtered_df,
            x="Power",
            nbins=50,
            title="Power Distribution",
            labels={"Power": "Power (kW)"},
        )
        st.plotly_chart(fig_power, use_container_width=True)
        
        fig_capacity = px.histogram(
            filtered_df,
            x="Cubic Capacity",
            nbins=50,
            title="Cubic Capacity Distribution",
            labels={"Cubic Capacity": "Cubic Capacity (cc)"},
        )
        st.plotly_chart(fig_capacity, use_container_width=True)
    
    st.subheader("Distribution Statistics")
    numeric_cols = ["price", "Mileage", "Power", "Cubic Capacity"]
    stats_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Skewness': [skew(filtered_df[col]) for col in numeric_cols],
        'Kurtosis': [kurtosis(filtered_df[col]) for col in numeric_cols]
    })
    st.dataframe(stats_df)

with tab3:
    st.header("Bivariate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price_mileage = px.scatter(
            filtered_df,
            x="Mileage",
            y="price",
            title="Price vs Mileage",
            labels={"price": "Price (€)", "Mileage": "Mileage (km)"},
        )
        st.plotly_chart(fig_price_mileage, use_container_width=True)
        
        fig_price_power = px.scatter(
            filtered_df,
            x="Power",
            y="price",
            title="Price vs Power",
            labels={"price": "Price (€)", "Power": "Power (kW)"},
        )
        st.plotly_chart(fig_price_power, use_container_width=True)
    
    with col2:
        fig_price_capacity = px.scatter(
            filtered_df,
            x="Cubic Capacity",
            y="price",
            title="Price vs Cubic Capacity",
            labels={"price": "Price (€)", "Cubic Capacity": "Cubic Capacity (cc)"},
        )
        st.plotly_chart(fig_price_capacity, use_container_width=True)

with tab4:
    st.header("Categorical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_make = px.box(
            filtered_df,
            x="Make",
            y="price",
            title="Price Distribution by Make",
            labels={"price": "Price (€)", "Make": "Car Make"},
        )
        st.plotly_chart(fig_make, use_container_width=True)
        
        fig_fuel = px.box(
            filtered_df,
            x="Fuel",
            y="price",
            title="Price Distribution by Fuel Type",
            labels={"price": "Price (€)", "Fuel": "Fuel Type"},
        )
        st.plotly_chart(fig_fuel, use_container_width=True)
    
    with col2:
        fig_transmission = px.box(
            filtered_df,
            x="Transmission",
            y="price",
            title="Price Distribution by Transmission",
            labels={"price": "Price (€)", "Transmission": "Transmission Type"},
        )
        st.plotly_chart(fig_transmission, use_container_width=True)
        
        fig_condition = px.box(
            filtered_df,
            x="Vehicle condition",
            y="price",
            title="Price Distribution by Vehicle Condition",
            labels={"price": "Price (€)", "Vehicle condition": "Vehicle Condition"},
        )
        st.plotly_chart(fig_condition, use_container_width=True)

with tab5:
    st.header("Correlation Analysis")
    
    numeric_cols = ["price", "Mileage", "Power", "Cubic Capacity"]
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )
    
    fig_corr.update_layout(title="Correlation Matrix", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Correlation Values")
    st.dataframe(corr_matrix.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))


st.sidebar.header("Summary Statistics")
st.sidebar.write(f"Total Cars: {len(filtered_df)}")
st.sidebar.write(f"Average Price: €{filtered_df['price'].mean():,.2f}")
st.sidebar.write(f"Average Mileage: {filtered_df['Mileage'].mean():,.0f} km")
st.sidebar.write(f"Average Power: {filtered_df['Power'].mean():,.0f} kW")
