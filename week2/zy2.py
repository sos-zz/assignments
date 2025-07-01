import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('US-pumpkins.csv')

# 数据预处理
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Average Price'] = (data['Low Price'] + data['High Price']) / 2

# 使用 Matplotlib 绘制价格随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Average Price'], marker='o', linestyle='-')
plt.title('Average Pumpkin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()

# 使用 Seaborn 绘制不同品种南瓜的平均价格
plt.figure(figsize=(12, 6))
sns.boxplot(x='Variety', y='Average Price', data=data)
plt.title('Average Price by Pumpkin Variety')
plt.xticks(rotation=45)
plt.show()

# 使用 Seaborn 绘制不同城市南瓜价格的分布
plt.figure(figsize=(12, 6))
sns.violinplot(x='City Name', y='Average Price', data=data)
plt.title('Average Price Distribution by City')
plt.xticks(rotation=45)
plt.show()

# 使用 Seaborn 绘制不同年份南瓜价格的变化
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Average Price', hue='Variety', data=data)
plt.title('Average Price by Year and Variety')
plt.show()