import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载数据
data = pd.read_csv('E:/新建文件夹/大三下/小学期-任萌/assignments/week2/US-pumpkins.csv')

# 数据预处理
# 指定日期格式以避免警告
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')  # 假设日期格式为月/日/年
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['平均价格'] = (data['Low Price'] + data['High Price']) / 2

# 处理类别特征
categorical_features = ['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Grade', 'Origin', 'Origin District', 'Item Size', 'Color', 'Environment', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack', 'Trans Mode']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature].astype(str))

# 填充缺失值
data = data.ffill().bfill()  # 修改填充缺失值的方式

# 选择特征和目标变量
X = data[['Month', 'Year'] + categorical_features]
y = data['平均价格']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练线性回归模型作为基线模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 训练随机森林模型
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 训练梯度提升树模型
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

evaluate_model(y_test, y_pred_linear, '线性回归')
evaluate_model(y_test, y_pred_rf, '随机森林')
evaluate_model(y_test, y_pred_gbr, '梯度提升树')

# 可视化：真实值 vs 预测值（线性回归）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='blue', label='线性回归')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('南瓜价格预测：真实值 vs 预测值（线性回归）')
plt.legend()
plt.grid(True)
plt.show()

# 可视化：真实值 vs 预测值（随机森林）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green', label='随机森林')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('南瓜价格预测：真实值 vs 预测值（随机森林）')
plt.legend()
plt.grid(True)
plt.show()

# 可视化：未来预测（2025年）
future_months = pd.DataFrame({
    'Month': list(range(1, 13)),
    'Year': [2025] * 12,
    'City Name': [data['City Name'].mean()] * 12,  # 假设未来城市为平均值
    'Type': [data['Type'].mean()] * 12,           # 添加缺失的特征
    'Package': [data['Package'].mean()] * 12,
    'Variety': [data['Variety'].mean()] * 12,
    'Sub Variety': [data['Sub Variety'].mean()] * 12,  # 添加缺失的特征
    'Grade': [data['Grade'].mean()] * 12,         # 添加缺失的特征
    'Origin': [data['Origin'].mean()] * 12,
    'Origin District': [data['Origin District'].mean()] * 12,
    'Item Size': [data['Item Size'].mean()] * 12,
    'Color': [data['Color'].mean()] * 12,
    'Environment': [data['Environment'].mean()] * 12,  # 添加缺失的特征
    'Unit of Sale': [data['Unit of Sale'].mean()] * 12,
    'Quality': [data['Quality'].mean()] * 12,
    'Condition': [data['Condition'].mean()] * 12,
    'Appearance': [data['Appearance'].mean()] * 12,
    'Storage': [data['Storage'].mean()] * 12,
    'Crop': [data['Crop'].mean()] * 12,
    'Repack': [data['Repack'].mean()] * 12,
    'Trans Mode': [data['Trans Mode'].mean()] * 12
})
future_months_scaled = scaler.transform(future_months)
future_pred = rf_model.predict(future_months_scaled)  # 修改为 rf_model

plt.figure(figsize=(12, 6))
plt.plot(future_months['Month'], future_pred, marker='o', linestyle='-', color='green')
plt.title('2025年未来12个月南瓜价格预测')
plt.xlabel('月份')
plt.ylabel('预测平均价格')
plt.grid(True)
plt.show()
'''
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# 随机选择随机森林中的一棵树
estimators = rf_model.estimators_  # 获取随机森林中的所有树
tree_idx = np.random.randint(0, len(estimators))  # 随机选择一棵树
tree = estimators[tree_idx]

# 绘制树结构
plt.figure(figsize=(20, 10))
plot_tree(tree,
          feature_names=['Month', 'Year'] + categorical_features,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title(f'随机森林第 {tree_idx} 号树')
plt.show()

# 打印树的文本结构
tree_rules = export_text(tree, feature_names=['Month', 'Year'] + categorical_features)
print(tree_rules)
import random

# 随机选择一条从树根到叶子节点的路径
def random_path(tree):
    path = []
    node = 0  # 从根节点开始
    while tree.tree_.children_left[node] != tree.tree_.children_right[node]:  # 当前节点不是叶子节点
        path.append(node)
        if random.random() < 0.5:
            node = tree.tree_.children_left[node]
        else:
            node = tree.tree_.children_right[node]
    path.append(node)  # 添加叶子节点
    return path

# 获取随机路径
path = random_path(tree)
print(f"随机选择的路径：{path}")

# 解释路径
for node in path:
    if tree.tree_.children_left[node] == tree.tree_.children_right[node]:  # 叶子节点
        print(f"叶子节点 {node}: 预测值 = {tree.tree_.value[node][0][0]:.2f}")
    else:
        feature = tree.tree_.feature[node]
        threshold = tree.tree_.threshold[node]
        feature_name = ['Month', 'Year'] + categorical_features
        print(f"节点 {node}: 如果 {feature_name[feature]} <= {threshold:.2f} 则左分支，否则右分支")'''
