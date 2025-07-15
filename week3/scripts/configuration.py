# configuration.py
import matplotlib.pyplot as plt

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题