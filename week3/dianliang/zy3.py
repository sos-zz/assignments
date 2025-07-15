import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

# 加载数据集
df = pd.read_csv('energy.csv')

# 将 timestamp 列转换为 datetime 类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 设置 timestamp 为索引
df.set_index('timestamp', inplace=True)

# 提取用电量列
load_data = df['load']

# 划分训练集和测试集
train_size = int(len(load_data) * 0.8)
train, test = load_data[:train_size], load_data[train_size:]

# 保存测试集的索引
test_index = test.index

# 使用 auto_arima 自动选择 ARIMA 模型的参数
stepwise_fit = auto_arima(load_data, start_p=0, start_q=0,
                          max_p=6, max_q=6, m=1,
                          seasonal=False, trace=True,
                          error_action='ignore',
                          suppress_warnings=True)
print('ARIMA 最优参数:', stepwise_fit.order)

# 构建 ARIMA 模型并进行预测
arima_model = ARIMA(train, order=stepwise_fit.order)
arima_result = arima_model.fit()
arima_forecast = arima_result.get_forecast(steps=len(test))
arima_pred = arima_forecast.predicted_mean

# 计算 ARIMA 模型的评估指标
arima_mse = mean_squared_error(test, arima_pred)
arima_mae = mean_absolute_error(test, arima_pred)
print(f'ARIMA 均方误差: {arima_mse}')
print(f'ARIMA 平均绝对误差: {arima_mae}')

# 准备 SVR 模型的数据
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 将数据转换为适合 SVR 的格式
look_back = 10
train = np.array(train).reshape(-1, 1)
test = np.array(test).reshape(-1, 1)
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 数据标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
trainX = scaler_X.fit_transform(trainX)
trainY = scaler_Y.fit_transform(trainY.reshape(-1, 1))
testX = scaler_X.transform(testX)

# 构建 SVR 模型并进行预测
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(trainX, trainY.ravel())
svr_pred = svr_model.predict(testX)
svr_pred = scaler_Y.inverse_transform(svr_pred.reshape(-1, 1))

# 计算 SVR 模型的评估指标
svr_mse = mean_squared_error(testY, svr_pred)
svr_mae = mean_absolute_error(testY, svr_pred)
print(f'SVR 均方误差: {svr_mse}')
print(f'SVR 平均绝对误差: {svr_mae}')

# 可视化结果
# 精确计算 new_index 的长度以匹配 testY 的长度
new_index = test_index[-len(testY):]
plt.figure(figsize=(15, 8))
plt.plot(new_index, testY, label='实际值')
plt.plot(new_index, arima_pred[-len(testY):], label='ARIMA 预测值', color='red')
plt.plot(new_index, svr_pred, label='SVR 预测值', color='green')
plt.title('用电量预测对比')
plt.xlabel('时间')
plt.ylabel('用电量负载')
plt.legend()
plt.show()