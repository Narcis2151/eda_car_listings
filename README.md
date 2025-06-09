# Car Listings Analysis Dashboard

This is an interactive dashboard built with Streamlit and Plotly for analyzing car listings data. The dashboard provides various visualizations and insights about car prices, features, and their relationships.

## Features

- Interactive filters for car make and price range
- Univariate analysis of key features (price, mileage, power, cubic capacity)
- Bivariate analysis showing relationships between features
- Categorical analysis of car attributes
- Correlation analysis between numeric features
- Real-time summary statistics

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the processed data file at `./data/processed_data.csv`
2. Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

