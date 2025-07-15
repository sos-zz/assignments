# feature_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def process_features(data):
    categorical_features = ['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Grade', 'Origin', 'Origin District', 'Item Size', 'Color', 'Environment', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack', 'Trans Mode']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].astype(str))
    data = data.ffill().bfill()
    X = data[['Month', 'Year'] + categorical_features]
    y = data['平均价格']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y