import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
data = pd.read_csv('nigerian-songs.csv')

# 数据预处理
# 将release_date转换为年份
data['release_date'] = pd.to_datetime(data['release_date'], format='%Y')
data['release_year'] = data['release_date'].dt.year

# 填充缺失值
data['artist_top_genre'].fillna('Unknown', inplace=True)

# 选择聚类特征
features = ['danceability', 'energy', 'acousticness', 'loudness', 'tempo', 'speechiness']
X = data[features]

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法则确定最佳聚类数
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# 根据肘部图选择聚类数
n_clusters = 5  # 假设最佳聚类数为5

# 进行KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# 分析每个聚类的中心点特征
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print("Cluster Centers:\n", cluster_centers_df)
#肘部法则的结果，用于确定最佳的聚类数。从图中可以看出，当聚类数从2增加到4时，惯性（Inertia）显著下降，之后下降幅度逐渐减小。这表明4个聚类可能是一个合理的选择，但也需要结合其他方法和领域知识进一步验证。
# 可视化聚类结果
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['cluster'], palette='viridis', legend='full')
plt.title('PCA of Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
#聚类结果可视化（PCA图）
#展示了使用PCA降维后的数据点分布，不同颜色代表不同的聚类。从图中可以看出，聚类之间有一定的分离，但也存在一些重叠，表明聚类结果在一定程度上能够区分不同的音乐特征组合，但也有一些特征相似的歌曲被分到了不同的聚类中。

# 分析聚类与流派的关系
genre_cluster = data.groupby(['artist_top_genre', 'cluster']).size().unstack(fill_value=0)
print("Genre-Cluster Distribution:\n", genre_cluster)

# 分析聚类与流行度的关系
popularity_cluster = data.groupby('cluster')['popularity'].mean().sort_values()
print("Average Popularity by Cluster:\n", popularity_cluster)

# 分析聚类与发行年份的关系
year_cluster = data.groupby(['release_year', 'cluster']).size().unstack(fill_value=0)
print("Year-Cluster Distribution:\n", year_cluster)
"""聚类与发行年份的关系（热图）
年份与聚类的热图，显示了不同年份的歌曲如何分布在不同的聚类中。从图中可以看出：
2014年和2015年的歌曲主要集中在聚类1和2中。
2016年和2017年的歌曲分布较为均匀，但聚类2和3中的歌曲数量较多。
2018年和2019年的歌曲在所有聚类中都有分布，显示出音乐风格的多样化。"""
# 可视化聚类与发行年份的关系
plt.figure(figsize=(12, 8))
sns.heatmap(year_cluster.T, cmap='viridis', annot=True, fmt='d')
plt.title('Year-Cluster Heatmap')
plt.xlabel('Release Year')
plt.ylabel('Cluster')
plt.show()