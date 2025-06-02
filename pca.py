import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------
# Step 1: Load the dataset
# ---------------------------
df = pd.read_csv('datascience_salaries.csv')  # 

# ---------------------------
# Step 2: Select numeric columns and normalize
# ---------------------------
numeric_df = df.select_dtypes(include=['float64', 'int64'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_df)

# ---------------------------
# Step 3: Apply PCA
# ---------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# ---------------------------
# Step 4: Plot PCA
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, color='blue')
plt.title('PCA - 2D Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# ---------------------------
# Step 5: Apply t-SNE
# ---------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
tsne_result = tsne.fit_transform(scaled_data)
tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])

# ---------------------------
# Step 6: Plot t-SNE
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(tsne_df['Dim1'], tsne_df['Dim2'], alpha=0.7, color='green')
plt.title('t-SNE - 2D Projection')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

