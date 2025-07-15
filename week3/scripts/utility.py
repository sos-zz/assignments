
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(y_test, *y_preds):
    for i, y_pred in enumerate(y_preds):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, label=f'Model {i+1}')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实价格')
        plt.ylabel('预测价格')
        plt.title(f'南瓜价格预测：真实值 vs 预测值（Model {i+1}）')
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_future_pred(future_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, 13)), future_pred, marker='o', linestyle='-', color='green')
    plt.title('2025年未来12个月南瓜价格预测')
    plt.xlabel('月份')
    plt.ylabel('预测平均价格')
    plt.grid(True)
    plt.show()