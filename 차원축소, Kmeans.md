- 군집 분류 결과를 feature 추가하여 사용, 혹은 군집별로 따로 분석하여 정확도 높이기
  - PCA :
    - 선형 차원 축소 기법
    - 데이터 분산 보존에 강함 (중요 특성유지하며 차원 축소)
    - 계산 효율 good 비교적 빠르게 수행
  - t-SNE :
    - 비선형 차원 축소 기법
    - 군집구조 보존에 강함 (저차원 데이터간 거리를 확률분포로 변환 -> 고차원에서 거리와 비교해 유사성 평가)
    - 계산 비용 높아 대규모 데이터에 적용어려움

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(train_enc)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(train_enc)

# K-Means clustering on PCA
kmeans_pca = KMeans(n_clusters=4, random_state=42)
kmeans_pca.fit(X_pca)

# K-Means clustering on t-SNE
kmeans_tsne = KMeans(n_clusters=4, random_state=42)
kmeans_tsne.fit(X_tsne)

# Visualize PCA + K-Means
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pca.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title('PCA + K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Visualize t-SNE + K-Means
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_tsne.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans_tsne.cluster_centers_[:, 0], kmeans_tsne.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title('t-SNE + K-Means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.tight_layout()
plt.show()
```
