import sys
import os

from week3.scripts import feature_processing, data_analysis

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取scripts目录的路径
scripts_dir = os.path.join(current_dir, '')

# 将scripts目录添加到sys.path中
sys.path.append(scripts_dir)

# 导入其他模块
from configuration import *
from data_analysis import *
from evaluate import *
from feature_processing import *
from model import *
from utility import *

# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_models, predict_future
from evaluate import evaluate_model
from utility import visualize_results
from data_analysis import load_data
from feature_processing import process_features

def main():
    file_path = 'E:/新建文件夹/大三下/小学期-任萌/assignments/week2/US-pumpkins.csv'
    data = load_data(file_path)
    X_scaled, y = process_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    linear_model, rf_model, gbr_model = train_models(X_train, y_train)
    evaluate_model(y_test, linear_model.predict(X_test), '线性回归')
    evaluate_model(y_test, rf_model.predict(X_test), '随机森林')
    evaluate_model(y_test, gbr_model.predict(X_test), '梯度提升树')
    visualize_results(y_test, linear_model.predict(X_test), rf_model.predict(X_test), gbr_model.predict(X_test))
    future_pred = predict_future(rf_model, data)
    visualize_results(future_pred)

if __name__ == "__main__":
    main()