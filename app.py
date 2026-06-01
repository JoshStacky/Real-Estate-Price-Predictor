import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
warnings.filterwarnings('ignore')

@st.cache_resource
def train_model():
    # Load from same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Melbourne_housing_FULL.csv')
    df = pd.read_csv(csv_path)
    suburbs = ['Richmond', 'Reservoir', 'Bentleigh East']
    df = df[df['Suburb'].isin(suburbs)].copy()
    df = df[['Rooms', 'Type', 'Price', 'Distance', 'Bathroom', 'Car',
             'Landsize', 'BuildingArea', 'YearBuilt', 'Date', 'Suburb']].dropna()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['SaleYear'] = df['Date'].dt.year
    df['SaleMonth'] = df['Date'].dt.month
    df['PropertyAge'] = df['SaleYear'] - df['YearBuilt']
    df = pd.get_dummies(df, columns=['Type', 'Suburb'], drop_first=True)
    feature_cols = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize',
                    'BuildingArea', 'PropertyAge', 'SaleYear', 'SaleMonth']
    for col in ['Type_t', 'Type_u', 'Suburb_Reservoir', 'Suburb_Richmond']:
        if col in df.columns:
            feature_cols.append(col)
    X = df[feature_cols]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, feature_cols

st.title('🏠 Melbourne Housing Price Predictor')
st.write('Trained on real 2016–2018 sales data · Suburbs: Richmond, Reservoir, Bentleigh East')

with st.spinner('Training model on dataset...'):
    model, feature_cols = train_model()

st.success('Model ready!')

col1, col2 = st.columns(2)
with col1:
    suburb   = st.selectbox('Suburb', ['Richmond', 'Reservoir', 'Bentleigh East'])
    ptype    = st.selectbox('Type', ['h (house)', 't (townhouse)', 'u (unit)'])
    rooms    = st.slider('Rooms', 1, 10, 3)
    bathroom = st.slider('Bathrooms', 1, 5, 2)
with col2:
    car      = st.slider('Car spaces', 0, 4, 1)
    landsize = st.number_input('Land size (m²)', 0, 5000, 500)
    bldgarea = st.number_input('Building area (m²)', 0, 1000, 150)
    yearbuilt= st.number_input('Year built', 1900, 2020, 1990)

if st.button('Predict Price', type='primary'):
    prop_age = 2017 - yearbuilt
    row = {
        'Rooms': rooms, 'Distance': 10.0, 'Bathroom': bathroom, 'Car': car,
        'Landsize': landsize, 'BuildingArea': bldgarea,
        'PropertyAge': prop_age, 'SaleYear': 2017, 'SaleMonth': 6,
        'Type_t': 1 if 't' in ptype else 0,
        'Type_u': 1 if 'u' in ptype else 0,
        'Suburb_Reservoir': 1 if suburb == 'Reservoir' else 0,
        'Suburb_Richmond':  1 if suburb == 'Richmond'  else 0,
    }
    X_pred = pd.DataFrame([row])
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[feature_cols]
    price = model.predict(X_pred)[0]
    st.markdown(f"## Estimated Sale Price: **${price:,.0f}**")
    st.caption('Based on Random Forest trained on Melbourne Housing dataset (2016–2018).')