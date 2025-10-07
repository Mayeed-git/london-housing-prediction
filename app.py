import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="London Housing Analysis", layout="wide")

# Title
st.title("London Housing Price Analysis")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('london_houses.csv', encoding='latin-1')
    # Rename the price column to avoid encoding issues
    price_col = [col for col in data.columns if 'Price' in col][0]
    data.rename(columns={price_col: 'Price (GBP)'}, inplace=True)
    return data

# Prepare training data
@st.cache_data
def prepare_data(data):
    # Encode categorical variables
    encoded_data = pd.get_dummies(data[['Neighborhood', 'Property Type']], dtype=int)

    # Drop columns
    train_data = data.drop(['Address', 'Neighborhood', 'Property Type', 'Heating Type',
                            'Balcony', 'Interior Style'], axis=1)

    # Join encoded data
    train_data = train_data.join(encoded_data)

    # Drop non-numerical columns
    non_numerical_cols = train_data.select_dtypes(exclude=np.number).columns
    train_data = train_data.drop(non_numerical_cols, axis=1)

    return train_data

# Train model
@st.cache_resource
def train_model(train_data):
    x = train_data.drop('Price (GBP)', axis=1)
    y = train_data['Price (GBP)']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)

    return model, score, x_train, x_test, y_train, y_test

# Load and prepare data
data = load_data()
train_data = prepare_data(data)
model, score, x_train, x_test, y_train, y_test = train_model(train_data)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page:",
                        ["Data Overview", "Exploratory Analysis", "Model Performance", "Price Prediction"])

# Data Overview Page
if page == "Data Overview":
    st.header("Data Overview")

    st.subheader("Raw Data")
    st.dataframe(data)

    st.subheader("Data Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape:**", data.shape)
        st.write("**Columns:**", data.shape[1])
        st.write("**Rows:**", data.shape[0])

    with col2:
        st.write("**Data Types:**")
        st.write(data.dtypes.value_counts())

    st.subheader("Statistical Summary")
    st.dataframe(data.describe())

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Analysis")

    st.subheader("Processed Training Data")
    st.dataframe(train_data)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(20, 14))
    sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='YlGnBu', ax=ax, fmt='.2f')
    st.pyplot(fig)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")

    st.metric("Model R2 Score", f"{score:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Set")
        st.write(f"Shape: {x_train.shape}")
        st.dataframe(x_train.head())

    with col2:
        st.subheader("Test Set")
        st.write(f"Shape: {x_test.shape}")
        st.dataframe(x_test.head())

    # Predictions
    y_pred = model.predict(x_test)

    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price (GBP)')
    ax.set_ylabel('Predicted Price (GBP)')
    ax.set_title('Actual vs Predicted Prices')
    st.pyplot(fig)

# Price Prediction Page
elif page == "Price Prediction":
    st.header("Predict House Price")

    st.write("Enter property details to predict the price:")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        square_meters = st.number_input("Square Meters", min_value=20, max_value=500, value=100)

    with col2:
        building_age = st.number_input("Building Age", min_value=0, max_value=200, value=20)
        floors = st.number_input("Floors", min_value=1, max_value=10, value=2)

    with col3:
        neighborhood = st.selectbox("Neighborhood",
                                   ['Camden', 'Chelsea', 'Greenwich', 'Islington', 'Kensington',
                                    'Marylebone', 'Notting Hill', 'Shoreditch', 'Soho', 'Westminster'])
        property_type = st.selectbox("Property Type",
                                    ['Apartment', 'Detached House', 'Semi-Detached'])

    if st.button("Predict Price"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Square Meters': [square_meters],
            'Building Age': [building_age],
            'Floors': [floors]
        })

        # Add neighborhood columns
        for n in ['Camden', 'Chelsea', 'Greenwich', 'Islington', 'Kensington',
                  'Marylebone', 'Notting Hill', 'Shoreditch', 'Soho', 'Westminster']:
            input_data[f'Neighborhood_{n}'] = 1 if n == neighborhood else 0

        # Add property type columns
        for p in ['Apartment', 'Detached House', 'Semi-Detached']:
            input_data[f'Property Type_{p}'] = 1 if p == property_type else 0

        # Predict
        prediction = model.predict(input_data)[0]

        st.success(f"### Predicted Price: GBP {prediction:,.2f}")
