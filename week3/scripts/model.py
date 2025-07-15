# model.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler  # 添加这一行来导入 StandardScaler
import pandas as pd

def train_models(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train, y_train)
    return linear_model, rf_model, gbr_model

def predict_future(model, data):
    future_months = pd.DataFrame({
        'Month': list(range(1, 13)),
        'Year': [2025] * 12,
        'City Name': [data['City Name'].mean()] * 12,
        'Type': [data['Type'].mean()] * 12,
        'Package': [data['Package'].mean()] * 12,
        'Variety': [data['Variety'].mean()] * 12,
        'Sub Variety': [data['Sub Variety'].mean()] * 12,
        'Grade': [data['Grade'].mean()] * 12,
        'Origin': [data['Origin'].mean()] * 12,
        'Origin District': [data['Origin District'].mean()] * 12,
        'Item Size': [data['Item Size'].mean()] * 12,
        'Color': [data['Color'].mean()] * 12,
        'Environment': [data['Environment'].mean()] * 12,
        'Unit of Sale': [data['Unit of Sale'].mean()] * 12,
        'Quality': [data['Quality'].mean()] * 12,
        'Condition': [data['Condition'].mean()] * 12,
        'Appearance': [data['Appearance'].mean()] * 12,
        'Storage': [data['Storage'].mean()] * 12,
        'Crop': [data['Crop'].mean()] * 12,
        'Repack': [data['Repack'].mean()] * 12,
        'Trans Mode': [data['Trans Mode'].mean()] * 12
    })
    future_months_scaled = StandardScaler().fit_transform(future_months)
    future_pred = model.predict(future_months_scaled)
    return future_pred