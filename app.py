import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff

# Set page config
st.set_page_config(
    page_title="Car Listings Analysis Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Car Listings Analysis Dashboard")
st.markdown("""
This dashboard provides an interactive analysis of car listings data, including price distributions,
relationships between features, and categorical variable analysis.
""")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('./data/raw_data_cast.csv')
    return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Make filter
makes = ['All'] + sorted(df['Make'].unique().tolist())
selected_make = st.sidebar.selectbox("Select Make", makes)

# Price range filter
min_price = float(df['price'].min())
max_price = float(df['price'].max())
price_range = st.sidebar.slider(
    "Price Range (€)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Filter data based on selections
filtered_df = df.copy()
if selected_make != 'All':
    filtered_df = filtered_df[filtered_df['Make'] == selected_make]
filtered_df = filtered_df[
    (filtered_df['price'] >= price_range[0]) &
    (filtered_df['price'] <= price_range[1])
]

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Univariate Analysis",
    "Bivariate Analysis",
    "Categorical Analysis",
    "Correlation Analysis"
])

# Tab 1: Univariate Analysis
with tab1:
    st.header("Univariate Analysis")
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Distribution
        fig_price = px.histogram(
            filtered_df,
            x='price',
            nbins=50,
            title='Price Distribution',
            labels={'price': 'Price (€)'}
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Mileage Distribution
        fig_mileage = px.histogram(
            filtered_df,
            x='Mileage',
            nbins=50,
            title='Mileage Distribution',
            labels={'Mileage': 'Mileage (km)'}
        )
        st.plotly_chart(fig_mileage, use_container_width=True)
    
    with col2:
        # Power Distribution
        fig_power = px.histogram(
            filtered_df,
            x='Power',
            nbins=50,
            title='Power Distribution',
            labels={'Power': 'Power (kW)'}
        )
        st.plotly_chart(fig_power, use_container_width=True)
        
        # Cubic Capacity Distribution
        fig_capacity = px.histogram(
            filtered_df,
            x='Cubic Capacity',
            nbins=50,
            title='Cubic Capacity Distribution',
            labels={'Cubic Capacity': 'Cubic Capacity (cc)'}
        )
        st.plotly_chart(fig_capacity, use_container_width=True)

# Tab 2: Bivariate Analysis
with tab2:
    st.header("Bivariate Analysis")
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Mileage
        fig_price_mileage = px.scatter(
            filtered_df,
            x='Mileage',
            y='price',
            title='Price vs Mileage',
            labels={'price': 'Price (€)', 'Mileage': 'Mileage (km)'}
        )
        st.plotly_chart(fig_price_mileage, use_container_width=True)
        
        # Price vs Power
        fig_price_power = px.scatter(
            filtered_df,
            x='Power',
            y='price',
            title='Price vs Power',
            labels={'price': 'Price (€)', 'Power': 'Power (kW)'}
        )
        st.plotly_chart(fig_price_power, use_container_width=True)
    
    with col2:
        # Price vs Cubic Capacity
        fig_price_capacity = px.scatter(
            filtered_df,
            x='Cubic Capacity',
            y='price',
            title='Price vs Cubic Capacity',
            labels={'price': 'Price (€)', 'Cubic Capacity': 'Cubic Capacity (cc)'}
        )
        st.plotly_chart(fig_price_capacity, use_container_width=True)

# Tab 3: Categorical Analysis
with tab3:
    st.header("Categorical Analysis")
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Price by Make
        fig_make = px.box(
            filtered_df,
            x='Make',
            y='price',
            title='Price Distribution by Make',
            labels={'price': 'Price (€)', 'Make': 'Car Make'}
        )
        st.plotly_chart(fig_make, use_container_width=True)
        
        # Price by Fuel Type
        fig_fuel = px.box(
            filtered_df,
            x='Fuel',
            y='price',
            title='Price Distribution by Fuel Type',
            labels={'price': 'Price (€)', 'Fuel': 'Fuel Type'}
        )
        st.plotly_chart(fig_fuel, use_container_width=True)
    
    with col2:
        # Price by Transmission
        fig_transmission = px.box(
            filtered_df,
            x='Transmission',
            y='price',
            title='Price Distribution by Transmission',
            labels={'price': 'Price (€)', 'Transmission': 'Transmission Type'}
        )
        st.plotly_chart(fig_transmission, use_container_width=True)
        
        # Price by Drive Type
        fig_drive = px.box(
            filtered_df,
            x='Drive type',
            y='price',
            title='Price Distribution by Drive Type',
            labels={'price': 'Price (€)', 'Drive type': 'Drive Type'}
        )
        st.plotly_chart(fig_drive, use_container_width=True)

# Tab 4: Correlation Analysis
with tab4:
    st.header("Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = ['price', 'Mileage', 'Power', 'Cubic Capacity']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    # Create correlation heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig_corr.update_layout(
        title='Correlation Matrix',
        height=600
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Display correlation values
    st.subheader("Correlation Values")
    st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))

# Add summary statistics
st.sidebar.header("Summary Statistics")
st.sidebar.write(f"Total Cars: {len(filtered_df)}")
st.sidebar.write(f"Average Price: €{filtered_df['price'].mean():,.2f}")
st.sidebar.write(f"Average Mileage: {filtered_df['Mileage'].mean():,.0f} km")
st.sidebar.write(f"Average Power: {filtered_df['Power'].mean():,.0f} kW") 