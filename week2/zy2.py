import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载数据
data = pd.read_csv('US-pumpkins.csv')

# 数据预处理
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['平均价格'] = (data['Low Price'] + data['High Price']) / 2

# 使用 Matplotlib 绘制价格随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['平均价格'], marker='o', linestyle='-')
plt.title('南瓜平均价格随时间变化趋势')
plt.xlabel('日期')
plt.ylabel('平均价格')
plt.grid(True)
plt.show()

# 使用 Seaborn 绘制不同品种南瓜的平均价格
plt.figure(figsize=(12, 6))
sns.boxplot(x='Variety', y='平均价格', data=data)
plt.title('不同品种南瓜的平均价格分布')
plt.xticks(rotation=45)
plt.xlabel('品种')
plt.ylabel('平均价格')
plt.show()

# 使用 Seaborn 绘制不同城市南瓜价格的分布
plt.figure(figsize=(12, 6))
sns.violinplot(x='City Name', y='平均价格', data=data)
plt.title('不同城市南瓜价格分布')
plt.xticks(rotation=45)
plt.xlabel('城市')
plt.ylabel('平均价格')
plt.show()

# 使用 Seaborn 绘制不同年份南瓜价格的变化
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='平均价格', hue='Variety', data=data)
plt.title('不同年份及品种南瓜平均价格变化')
plt.xlabel('年份')
plt.ylabel('平均价格')
plt.legend(title='品种')
plt.show()

# 选择特征（这里用月份、年份作为简单示例）
# 选择特征
X = data[['Month', 'Year']]
y = data['平均价格']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算误差
mae = mean_absolute_error(y_test, y_pred)
print(f"预测平均绝对误差（MAE）: {mae:.2f}")

# 可视化：真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('南瓜价格预测：真实值 vs 预测值')
plt.grid(True)
plt.show()

# 可视化：未来预测（2025年）
future_months = pd.DataFrame({
    'Month': list(range(1, 13)),
    'Year': [2025] * 12
})
future_pred = model.predict(future_months)

plt.figure(figsize=(12, 6))
plt.plot(future_months['Month'], future_pred, marker='o', linestyle='-', color='green')
plt.title('2025年未来12个月南瓜价格预测')
plt.xlabel('月份')
plt.ylabel('预测平均价格')
plt.grid(True)
plt.show()