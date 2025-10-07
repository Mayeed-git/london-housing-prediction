# London Housing Price Prediction

A machine learning project that predicts London housing prices using Linear Regression. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## Overview

This project analyzes London housing data and builds a predictive model to estimate property prices based on various features such as location, size, age, and amenities.

### Model Performance
- **R² Score**: 0.9478 (94.78% accuracy)
- **Training Score**: 0.9471
- **Test Score**: 0.9478

## Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Machine Learning**: Linear Regression model with proper train/test split
- **Interactive Web App**: User-friendly Streamlit interface for predictions
- **Correlation Analysis**: Heatmap showing feature relationships

## Dataset

The dataset contains 1,000 London properties with 17 features:

- **Location**: Neighborhood (Camden, Chelsea, Greenwich, etc.)
- **Property Details**: Bedrooms, Bathrooms, Square Meters, Floors
- **Building Info**: Age, Status (Old/Renovated), Materials
- **Amenities**: Garden, Garage, Balcony, Heating Type
- **Style**: Interior Style, View
- **Price**: Property price in GBP

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mayeed-git/london-housing-prediction.git
cd london-housing-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Run the Jupyter Notebook

```bash
jupyter notebook London_Housing.ipynb
```

## App Pages

### 1. Data Overview
- View raw dataset
- Dataset statistics and information
- Data type distribution

### 2. Exploratory Analysis
- Processed training data
- Correlation heatmap
- Feature relationships

### 3. Model Performance
- R² Score metrics
- Training and test set visualization
- Actual vs Predicted prices scatter plot

### 4. Price Prediction
Interactive form to predict house prices by entering:
- Number of bedrooms and bathrooms
- Square meters
- Building age
- Number of floors
- Neighborhood
- Property type

## Project Structure

```
london-housing-prediction/
├── app.py                    # Streamlit web application
├── London_Housing.ipynb      # Jupyter notebook with analysis
├── london_houses.csv         # Dataset
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning model
- **matplotlib & seaborn**: Data visualization
- **Streamlit**: Web application framework

## Model Details

### Preprocessing
1. Encode categorical variables (Neighborhood, Property Type)
2. Drop non-essential columns (Address, Heating Type, etc.)
3. Remove non-numerical features
4. Split data (80% train, 20% test)

### Algorithm
- **Linear Regression** with scikit-learn
- No overfitting detected (similar train/test scores)
- 18 features used for prediction

## Results

The model achieves excellent performance with an R² score of 0.9478, indicating that it can explain 94.78% of the variance in London housing prices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit and scikit-learn**
