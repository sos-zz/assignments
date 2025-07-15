# data_analysis.py
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['平均价格'] = (data['Low Price'] + data['High Price']) / 2
    return data