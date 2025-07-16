import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载数据
data = pd.read_csv('E:/新建文件夹/大三下/小学期-任萌/assignments/week2/US-pumpkins.csv')

# 数据预处理
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
data = data.fillna(method='ffill').fillna(method='bfill')

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

# 训练 LightGBM 模型
lgbm_model = lgb.LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

# === 停止树疯长：LightGBM ===
best_lgbm_model = lgb.LGBMRegressor(
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

eval_set = [(X_test, y_test)]
callbacks = [lgb.early_stopping(50)]

best_lgbm_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='l2',
    callbacks=callbacks
)

# 使用最佳迭代轮次
y_pred_best_lgbm = best_lgbm_model.predict(X_test, num_iteration=best_lgbm_model.best_iteration_)

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

evaluate_model(y_test, y_pred_linear, '线性回归')
evaluate_model(y_test, y_pred_lgbm, 'LightGBM')
evaluate_model(y_test, y_pred_best_lgbm, '剪枝后的 LightGBM')

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

# 可视化：真实值 vs 预测值（LightGBM）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.7, color='orange', label='LightGBM')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('南瓜价格预测：真实值 vs 预测值（LightGBM）')
plt.legend()
plt.grid(True)
plt.show()

# 可视化：真实值 vs 预测值（调优后的 LightGBM）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_lgbm, alpha=0.7, color='purple', label='调优后的 LightGBM')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('南瓜价格预测：真实值 vs 预测值（调优后的 LightGBM）')
plt.legend()
plt.grid(True)
plt.show()

# 将预测结果合并回原始数据集
results = data.loc[y_test.index].copy()
results['test_y'] = y_test
results['test_y_predict_linear'] = y_pred_linear
results['test_y_predict_lgbm'] = y_pred_lgbm
results['test_y_predict_best_lgbm'] = y_pred_best_lgbm

# 计算delta
results['delta_linear'] = results['test_y'] - results['test_y_predict_linear']
results['delta_lgbm'] = results['test_y'] - results['test_y_predict_lgbm']
results['delta_best_lgbm'] = results['test_y'] - results['test_y_predict_best_lgbm']

# 添加 weekday 列
results['weekday'] = results['Date'].dt.dayofweek

# 选择您想要展示的列
columns_of_interest = [
    'City Name', 'Package', 'Variety', 'Origin', 'Item Size', '平均价格', 'Year', 'Month',
    'weekday', 'test_y', 'test_y_predict_best_lgbm', 'delta_best_lgbm'
]

# 创建最终的表格
final_results = results[columns_of_interest]

# 展示表格
print(final_results.head())

# 如果需要，保存到CSV文件
final_results.to_csv('final_results.csv', index=False)