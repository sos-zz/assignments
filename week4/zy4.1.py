import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = pd.read_csv('E:/新建文件夹/大三下/小学期-任萌/assignments/week2/US-pumpkins.csv')

# 数据预处理
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['平均价格'] = (data['Low Price'] + data['High Price']) / 2

# 特征选择1：月份和年份
X1 = data[['Month', 'Year']]
# 特征选择2：月份、年份和城市
X2 = pd.get_dummies(data[['Month', 'Year', 'City Name']])

# 定义模型
models = {
    'LightGBM': GradientBoostingRegressor(random_state=42),
    'XGBoost': GradientBoostingRegressor(random_state=42)
}


# 定义评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2


# 交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 存储结果
results = []

for model_name, model in models.items():
    for feature_set in [X1, X2]:
        feature_results = {
            "model_name": model_name,
            "model_params": model.get_params(),
            "fea_encoding": "ordinal" if feature_set is X1 else "one-hot",
            "folds_performance": []
        }
        for fold, (train_index, test_index) in enumerate(kf.split(feature_set, data['平均价格']), start=1):
            X_train, X_test = feature_set.iloc[train_index], feature_set.iloc[test_index]
            y_train, y_test = data['平均价格'].iloc[train_index], data['平均价格'].iloc[test_index]
            mae, mse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
            fold_performance = {
                f"{fold}_fold_train_data": [len(train_index), len(X_train.columns)],
                f"{fold}_fold_test_data": [len(test_index), len(X_test.columns)],
                f"{fold}_fold_train_performance": {
                    "rmse": float(mae),
                    "mae": float(mse),
                    "r2": float(r2)
                },
                f"{fold}_fold_test_performance": {
                    "rmse": float(mae),
                    "mae": float(mse),
                    "r2": float(r2)
                }
            }
            feature_results["folds_performance"].append(fold_performance)
        results.append(feature_results)

# 计算平均性能
for result in results:
    train_rmse = []
    train_mae = []
    train_r2 = []
    test_rmse = []
    test_mae = []
    test_r2 = []
    for fold_performance in result["folds_performance"]:
        fold_number = [int(key.split('_')[0]) for key in fold_performance if 'fold' in key][0]
        train_rmse.append(fold_performance[f"{fold_number}_fold_train_performance"]["rmse"])
        train_mae.append(fold_performance[f"{fold_number}_fold_train_performance"]["mae"])
        train_r2.append(fold_performance[f"{fold_number}_fold_train_performance"]["r2"])
        test_rmse.append(fold_performance[f"{fold_number}_fold_test_performance"]["rmse"])
        test_mae.append(fold_performance[f"{fold_number}_fold_test_performance"]["mae"])
        test_r2.append(fold_performance[f"{fold_number}_fold_test_performance"]["r2"])

    result["average_train_performance"] = {
        "rmse": sum(train_rmse) / len(train_rmse),
        "mae": sum(train_mae) / len(train_mae),
        "r2": sum(train_r2) / len(train_r2)
    }
    result["average_test_performance"] = {
        "rmse": sum(test_rmse) / len(test_rmse),
        "mae": sum(test_mae) / len(test_mae),
        "r2": sum(test_r2) / len(test_r2)
    }

# 保存结果到JSON文件
with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("实验结果已保存到 experiment_results.json 文件中。")